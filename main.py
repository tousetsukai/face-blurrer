import argparse
import json
import textwrap
from pathlib import Path

import cv2
import piexif
from tqdm import tqdm

MODEL_PATH = "yolov11m-face.pt"
BLUR_KSIZE = (101, 101)  # Gaussian kernel
BLUR_PASSES = 2  # ぼかし回数
BATCH = 8  # 処理バッチ数
FACE_RATIO_HIGH_THRESHOLD = 0.008
FACE_RATIO_LOW_THRESHOLD = 0.004
IMG_EXTS = ("jpg", "jpeg", "png")

DETECTIONS_FILE = "detections.json"
DECISIONS_FILE = "decisions.json"
WINDOW_NAME = "Face Detection"


def list_images(in_dir):
    return sorted(p for ext in IMG_EXTS for p in in_dir.rglob(f"*.{ext}"))


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


def blur(roi):
    """Apply Gaussian blur to the region of interest."""
    for _ in range(BLUR_PASSES):
        roi = cv2.GaussianBlur(roi, BLUR_KSIZE, 0)
    return roi


def is_suspect(img_shape, box):
    x1, y1, x2, y2 = box
    height, width = img_shape[:2]
    face_ratio = (x2 - x1) * (y2 - y1) / (height * width)
    # 画像のサイズに対して顔のサイズが大きい場合は行灯の顔の疑い
    if face_ratio >= FACE_RATIO_HIGH_THRESHOLD:
        return True
    # 画像の上半分に少し大きめの顔がある場合は行灯の顔の疑い
    if y1 < height / 2 and face_ratio >= FACE_RATIO_LOW_THRESHOLD:
        return True
    return False


def pick_device():
    """Apple Silicon GPU (MPS) が使えれば使う"""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return None


def detect(in_dir, work_dir, device="auto"):
    """フェーズ1: 全画像の顔検出を行い detections.json に保存する（無人・重い）"""
    # review/render ではモデルを読み込まなくて済むよう、ここで import する
    from supervision import Detections
    from ultralytics import YOLO

    if device == "auto":
        device = pick_device()
    print(f"Using device: {device or 'cpu'}")

    detections_path = work_dir / DETECTIONS_FILE
    detections = load_json(
        detections_path, {"meta": {"model": MODEL_PATH}, "files": {}}
    )
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
        results = model.predict(imgs, device=device, verbose=False)

        # MPS の初回推論が誤った結果 (顔が全て消える) を返すことが稀にあったため、
        # 最初のバッチだけ CPU と突き合わせ、食い違ったら CPU に切り替える
        if i == 0 and device not in (None, "cpu"):
            cpu_results = model.predict(imgs, device="cpu", verbose=False)
            if [len(r.boxes) for r in results] != [len(r.boxes) for r in cpu_results]:
                print(f"Warning: `{device}` disagrees with `cpu`. Falling back to cpu.")
                device = "cpu"
                results = cpu_results
        for img, res, path in zip(imgs, results, batch_paths):
            faces = []
            for fbox in Detections.from_ultralytics(res).xyxy:
                box = [int(v) for v in fbox]
                faces.append({"box": box, "suspect": is_suspect(img.shape, box)})
            files[path.relative_to(in_dir).as_posix()] = {"faces": faces}
        # バッチごとに保存し、中断しても再開できるようにする
        save_json(detections_path, detections)


def ask_andon(img, box):
    """疑い顔を枠付きで表示して行灯の顔かどうか確認する"""
    x1, y1, x2, y2 = box
    img = img.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        "Is Andon? (y/n)",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    cv2.imshow(WINDOW_NAME, img)
    while True:
        key = cv2.waitKey(0)
        if key == ord("y"):
            return True
        elif key == ord("n"):
            return False


def pending_suspects(detections, decisions):
    """未回答の行灯疑い顔を (相対パス, box) のリストで返す"""
    pending = []
    for rel_path, entry in detections["files"].items():
        decided = {tuple(d["box"]) for d in decisions.get(rel_path, [])}
        for face in entry["faces"]:
            if face["suspect"] and tuple(face["box"]) not in decided:
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
    for rel_path, box in pending:
        img = cv2.imread(str(in_dir / rel_path))
        andon = ask_andon(img, box)
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


def render(in_dir, out_dir, work_dir, force=False):
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
        andon_boxes = {
            tuple(d["box"]) for d in decisions.get(rel_path, []) if d["andon"]
        }
        for face in entry["faces"]:
            box = face["box"]
            if tuple(box) in andon_boxes:
                continue
            x1, y1, x2, y2 = box
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            img[y1:y2, x1:x2] = blur(roi)

        # 保存先のパスをインプット側のディレクトリ構造を保って作成
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)
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
    args = parser.parse_args()

    in_dir, out_dir, work_dir = (
        Path(args.in_dir),
        Path(args.out_dir),
        Path(args.work_dir),
    )
    if args.command in ("detect", "all"):
        detect(in_dir, work_dir, device=args.device)
    if args.command in ("review", "all"):
        review(in_dir, work_dir)
    if args.command in ("render", "all"):
        render(in_dir, out_dir, work_dir, force=args.force)
