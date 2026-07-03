import argparse
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np
import piexif
from tqdm import tqdm

MODEL_PATH = "yolov12m-face.pt"
BLUR_PASSES = 2  # ぼかし回数
BLUR_PAD_RATIO = 0.15  # 検出 box ちょうどだと輪郭が残るので少し広げてぼかす
BLUR_KSIZE_MIN = 51  # Gaussian kernel の最小サイズ
BATCH = 8  # 処理バッチ数
# 4000〜6000px の写真では YOLO デフォルトの 640 に縮小すると遠くの顔が
# 数 px になり検出漏れするため、デフォルトを大きめにしている
DEFAULT_IMGSZ = 1280
DEFAULT_CONF = 0.25  # YOLO デフォルトと同じ
# 再エンコードによる劣化をできるだけ抑える (cv2 デフォルトは 95)
DEFAULT_JPEG_QUALITY = 97
FACE_RATIO_HIGH_THRESHOLD = 0.008
FACE_RATIO_LOW_THRESHOLD = 0.004
FACE_TOP_BAND = 0.4  # 顔の中心がこの比率より上にあれば行灯の顔の疑い
IMG_EXTS = ("jpg", "jpeg", "png")

DETECTIONS_FILE = "detections.json"
DECISIONS_FILE = "decisions.json"
WINDOW_NAME = "Face Detection"


def list_images(in_dir):
    # カメラ由来の .JPG など大文字拡張子も拾う
    return sorted(
        p for ext in IMG_EXTS for p in in_dir.rglob(f"*.{ext}", case_sensitive=False)
    )


def load_json(path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def save_json(path, data):
    """テンポラリファイル経由のアトミック書き込み（中断でファイルを壊さないため）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def blur_face(img, box):
    """box の顔をぼかす (img を直接書き換える)"""
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    # 検出 box ちょうどだとあご・額の輪郭が残るので少し広げる
    pad_x = int((x2 - x1) * BLUR_PAD_RATIO)
    pad_y = int((y2 - y1) * BLUR_PAD_RATIO)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    # 固定カーネルだと大きい顔でぼかしが足りないので顔サイズに比例させる (奇数化)
    k = max(BLUR_KSIZE_MIN, (max(x2 - x1, y2 - y1) // 2) | 1)
    blurred = roi
    for _ in range(BLUR_PASSES):
        blurred = cv2.GaussianBlur(blurred, (k, k), 0)
    # 矩形だと不自然なので、境界をぼかした楕円マスクで合成する
    mask = np.zeros(roi.shape[:2], dtype=np.float32)
    center = ((x2 - x1) // 2, (y2 - y1) // 2)
    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    feather = (max(min(axes) // 4, 3) * 2) | 1
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)[..., None]
    img[y1:y2, x1:x2] = (blurred * mask + roi * (1.0 - mask)).astype(img.dtype)


def is_suspect(size, box):
    width, height = size
    x1, y1, x2, y2 = box
    face_ratio = (x2 - x1) * (y2 - y1) / (height * width)
    # 画像のサイズに対して顔のサイズが大きい場合は行灯の顔の疑い
    if face_ratio >= FACE_RATIO_HIGH_THRESHOLD:
        return True
    # 画像の上半分に少し大きめの顔がある場合は行灯の顔の疑い
    if y1 < height / 2 and face_ratio >= FACE_RATIO_LOW_THRESHOLD:
        return True
    # 画像の上部にある顔は、小さくても行灯の顔の疑い
    # (実際の人の顔は画角の下側に集まりやすく、上部の小さい顔を無条件に
    #  ぼかすと行灯の顔を巻き込むことがあったため)
    if (y1 + y2) / 2 < height * FACE_TOP_BAND:
        return True
    return False


def pick_device():
    """Apple Silicon GPU (MPS) が使えれば使う"""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return None


def detect(in_dir, work_dir, device="auto", imgsz=DEFAULT_IMGSZ, conf=DEFAULT_CONF):
    """フェーズ1: 全画像の顔検出を行い detections.json に保存する（無人・重い）"""
    # review/render ではモデルを読み込まなくて済むよう、ここで import する
    from supervision import Detections
    from ultralytics import YOLO

    if device == "auto":
        device = pick_device()
    print(f"Using device: {device or 'cpu'}")

    detections_path = work_dir / DETECTIONS_FILE
    meta = {"model": MODEL_PATH, "imgsz": imgsz, "conf": conf}
    detections = load_json(detections_path, {"meta": meta, "files": {}})
    if detections["meta"] != meta:
        # 設定の違う検出結果が混ざると結果に一貫性がなくなるので中断する
        print(
            f"Settings mismatch: {detections_path} was created with "
            f"{detections['meta']}, but current settings are {meta}. "
            "Use a fresh work dir (or delete it) to re-detect."
        )
        exit(1)
    files = detections["files"]

    paths = [
        p for p in list_images(in_dir) if p.relative_to(in_dir).as_posix() not in files
    ]
    skipped = len(list_images(in_dir)) - len(paths)
    if skipped:
        print(f"Skipping {skipped} already-detected files.")
    if not paths:
        print("Nothing to detect.")
        return

    model = YOLO(MODEL_PATH)
    for i in tqdm(range(0, len(paths), BATCH), desc="Detect"):
        batch_paths = paths[i : i + BATCH]
        imgs = [cv2.imread(str(p)) for p in batch_paths]
        # 読めないファイルが混ざっていると predict ごと落ちるので除いておく
        for p, img in zip(batch_paths, imgs):
            if img is None:
                print(f"Warning: cannot read {p}. Skipping.")
        batch = [(p, img) for p, img in zip(batch_paths, imgs) if img is not None]
        if not batch:
            continue
        batch_paths, imgs = zip(*batch)
        results = model.predict(
            list(imgs), device=device, imgsz=imgsz, conf=conf, verbose=False
        )

        # MPS の初回推論が誤った結果 (顔が全て消える) を返すことが稀にあったため、
        # 最初のバッチだけ CPU と突き合わせ、食い違ったら CPU に切り替える
        if i == 0 and device not in (None, "cpu"):
            cpu_results = model.predict(
                imgs, device="cpu", imgsz=imgsz, conf=conf, verbose=False
            )
            if [len(r.boxes) for r in results] != [len(r.boxes) for r in cpu_results]:
                print(f"Warning: `{device}` disagrees with `cpu`. Falling back to cpu.")
                device = "cpu"
                results = cpu_results
        for img, res, path in zip(imgs, results, batch_paths):
            # suspect 判定はここでは行わない。review/render 時に計算することで、
            # 閾値を変えても検出をやり直さずに済む
            faces = [
                {"box": [int(v) for v in fbox]}
                for fbox in Detections.from_ultralytics(res).xyxy
            ]
            files[path.relative_to(in_dir).as_posix()] = {
                "size": [img.shape[1], img.shape[0]],
                "faces": faces,
            }
        # バッチごとに保存し、中断しても再開できるようにする
        save_json(detections_path, detections)


def ask_andon(img, box, progress):
    """疑い顔を枠付きで表示して行灯の顔かどうか確認する

    y/n で回答する。q キーまたはウィンドウを閉じると中断 (None を返す)。
    """
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    img = img.copy()
    # 4000px 級の画像でも見えるよう、枠線・文字は画像サイズに比例させる
    thickness = max(2, w // 800)
    font_scale = w / 2000
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    cv2.putText(
        img,
        f"Is Andon? (y/n, q=quit) {progress}",
        (x1, max(int(40 * font_scale), y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        thickness,
    )
    # 画像そのままだと画面からはみ出すので、収まるサイズのウィンドウにする
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    scale = min(1.0, 1400 / w, 900 / h)
    cv2.resizeWindow(WINDOW_NAME, int(w * scale), int(h * scale))
    cv2.imshow(WINDOW_NAME, img)
    while True:
        key = cv2.waitKey(100)
        if key == ord("y"):
            return True
        if key == ord("n"):
            return False
        if key == ord("q"):
            return None
        # ウィンドウが閉じられたら中断扱いにする (放置すると抜けられなくなる)
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return None


def pending_suspects(detections, decisions):
    """未回答の行灯疑い顔を (相対パス, box) のリストで返す"""
    pending = []
    for rel_path, entry in detections["files"].items():
        decided = {tuple(d["box"]) for d in decisions.get(rel_path, [])}
        for face in entry["faces"]:
            if (
                is_suspect(entry["size"], face["box"])
                and tuple(face["box"]) not in decided
            ):
                pending.append((rel_path, face["box"]))
    return pending


def review(in_dir, work_dir):
    """フェーズ2: 行灯疑いの顔を人間が y/n で判断し decisions.json に保存する（対話）"""
    detections = load_json(work_dir / DETECTIONS_FILE, None)
    if detections is None:
        print(f"No {DETECTIONS_FILE} found in `{work_dir}`. Run detect first.")
        exit(1)
    decisions_path = work_dir / DECISIONS_FILE
    decisions = load_json(decisions_path, {})

    pending = pending_suspects(detections, decisions)
    if not pending:
        print("No suspect faces to review.")
        return

    print(f"{len(pending)} suspect faces to review.")
    for idx, (rel_path, box) in enumerate(pending):
        img = cv2.imread(str(in_dir / rel_path))
        if img is None:
            print(f"Warning: cannot read {in_dir / rel_path}. Skipping.")
            continue
        andon = ask_andon(img, box, f"[{idx + 1}/{len(pending)}]")
        if andon is None:
            print(
                "Review interrupted. Progress is saved; run review again to continue."
            )
            break
        if andon:
            print(f"'Andon' face marked in {rel_path}")
        decisions.setdefault(rel_path, []).append({"box": box, "andon": andon})
        # 1 回答ごとに保存し、中断しても回答をやり直さなくて済むようにする
        save_json(decisions_path, decisions)
    cv2.destroyAllWindows()


def copy_exif(src_path, dst_path):
    """元画像の EXIF を出力画像にコピーする (JPEG のみ)

    cv2.imwrite は EXIF を全て落とすため、書き出し後にコピーし直す。
    """
    if dst_path.suffix.lower() not in (".jpg", ".jpeg"):
        return
    try:
        exif_dict = piexif.load(str(src_path))
        # EXIF 内サムネイルにはぼかし前の画像が残るため必ず除去する
        exif_dict["thumbnail"] = None
        # cv2.imread は Orientation を解釈してピクセルを回転させるので、
        # タグを 1 にリセットしないとビューアで二重回転する
        exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
        piexif.insert(piexif.dump(exif_dict), str(dst_path))
    except Exception as e:
        print(f"Warning: failed to copy EXIF for {dst_path}: {e}")


def render(in_dir, out_dir, work_dir, force=False, jpeg_quality=DEFAULT_JPEG_QUALITY):
    """フェーズ3: 判断結果を使って顔をぼかし、out_dir に書き出す（無人・軽い）"""
    detections = load_json(work_dir / DETECTIONS_FILE, None)
    if detections is None:
        print(f"No {DETECTIONS_FILE} found in `{work_dir}`. Run detect first.")
        exit(1)
    decisions = load_json(work_dir / DECISIONS_FILE, {})

    # 未回答の疑い顔が残ったまま書き出すと誤ったぼかし方をするので中断する
    pending = pending_suspects(detections, decisions)
    if pending:
        print(f"{len(pending)} suspect faces are not reviewed yet. Run review first.")
        exit(1)

    skipped = 0
    for rel_path, entry in tqdm(detections["files"].items(), desc="Render"):
        save_path = out_dir / rel_path
        if save_path.exists() and not force:
            skipped += 1
            continue

        img = cv2.imread(str(in_dir / rel_path))
        if img is None:
            print(f"Warning: cannot read {in_dir / rel_path}. Skipping.")
            continue
        andon_boxes = {
            tuple(d["box"]) for d in decisions.get(rel_path, []) if d["andon"]
        }
        for face in entry["faces"]:
            box = face["box"]
            if tuple(box) in andon_boxes:
                continue
            blur_face(img, box)

        # 保存先のパスをインプット側のディレクトリ構造を保って作成
        save_path.parent.mkdir(parents=True, exist_ok=True)
        params = []
        if save_path.suffix.lower() in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        cv2.imwrite(str(save_path), img, params)
        copy_exif(in_dir / rel_path, save_path)

    if skipped:
        print(f"Skipped {skipped} already-rendered files (use --force to redo).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent(
            """
            Face Blurrer

            This script detects faces in images and applies a Gaussian blur to them.
            Processing is split into three resumable phases:

                detect: detect faces in all images -> <work_dir>/detections.json
                review: ask y/n for suspected andon faces -> <work_dir>/decisions.json
                render: blur non-andon faces and write results -> <out_dir>

            Running without a command executes all three phases in order.

            Default input directory: 'in'
            Default output directory: 'out'
            Default work directory: 'work'
        """
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["detect", "review", "render", "all"],
        default="all",
    )
    parser.add_argument("-i", "--in_dir", default="in")
    parser.add_argument("-o", "--out_dir", default="out")
    parser.add_argument("-w", "--work_dir", default="work")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-render files that already exist in out_dir",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="inference device: auto (default; mps if available), cpu, mps, ...",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"inference image size (default: {DEFAULT_IMGSZ}); "
        "larger finds smaller faces but is slower",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"detection confidence threshold (default: {DEFAULT_CONF}); "
        "lower finds more faces but with more false positives",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=f"jpeg output quality (default: {DEFAULT_JPEG_QUALITY})",
    )
    args = parser.parse_args()

    in_dir, out_dir, work_dir = (
        Path(args.in_dir),
        Path(args.out_dir),
        Path(args.work_dir),
    )
    if args.command in ("detect", "all"):
        detect(in_dir, work_dir, device=args.device, imgsz=args.imgsz, conf=args.conf)
    if args.command in ("review", "all"):
        review(in_dir, work_dir)
    if args.command in ("render", "all"):
        render(
            in_dir,
            out_dir,
            work_dir,
            force=args.force,
            jpeg_quality=args.jpeg_quality,
        )
