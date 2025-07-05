import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from supervision import Detections

BLUR_KSIZE = (101, 101)  # Gaussian kernel
BLUR_PASSES = 2  # ぼかし回数
BATCH = 8  # 処理バッチ数
FACE_RATIO_HIGH_THRESHOLD = 0.008
FACE_RATIO_LOW_THRESHOLD = 0.004

# https://github.com/akanametov/yolo-face
face_model = YOLO("yolov11m-face.pt")


def ensure_empty_out_dir(out_dir="out"):
    """Ensure the output directory is empty."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"Output directory `{out_dir}` is not empty. Please clear it.")
        exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)


def blur(roi):
    """Apply Gaussian blur to the region of interest."""
    for _ in range(BLUR_PASSES):
        roi = cv2.GaussianBlur(roi, BLUR_KSIZE, 0)
    return roi


def is_andon_face(img, fbox):
    suspect = False
    x1, y1, x2, y2 = map(int, fbox)

    # 画像のサイズに対して顔のサイズが大きい場合は行灯の顔の疑い
    face_area = (x2 - x1) * (y2 - y1)
    img_area = img.shape[0] * img.shape[1]
    face_ratio = face_area / img_area
    if face_ratio >= FACE_RATIO_HIGH_THRESHOLD:
        suspect = True
    # 画像の上半分に少し大きめの顔がある場合は行灯の顔の疑い
    height = img.shape[0]
    if fbox[1] < height / 2 and face_ratio >= FACE_RATIO_LOW_THRESHOLD:
        suspect = True

    if suspect:
        # 怪しいものは画像を枠付きで表示して行灯の顔かどうか確認する
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
        cv2.imshow("Face Detection", img)
        while True:
            # ユーザーの入力を待つ
            key = cv2.waitKey(0)
            if key == ord("y"):
                cv2.destroyAllWindows()
                return True
            elif key == ord("n"):
                cv2.destroyAllWindows()
                return False


def process_dir(in_dir="in", out_dir="out"):
    ensure_empty_out_dir(out_dir)
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    paths = sorted(
        list(p for ext in ("jpg", "jpeg", "png") for p in in_dir.rglob(f"*.{ext}"))
    )

    for i in tqdm(range(0, len(paths), BATCH), desc="Batches"):
        batch_paths = paths[i : i + BATCH]
        imgs = [cv2.imread(str(p)) for p in batch_paths]

        # 1) 推論
        faces_res = face_model.predict(imgs, verbose=False)

        # 2) 各画像ごとにぼかし処理
        for img, fp, path in zip(imgs, faces_res, batch_paths):
            faces_xyxy = Detections.from_ultralytics(fp).xyxy

            if faces_xyxy is None or len(faces_xyxy) == 0:
                continue

            for fbox in faces_xyxy:
                x1, y1, x2, y2 = map(int, fbox)
                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                if is_andon_face(img, fbox):
                    print(f"'Andon' face detected in {path}")
                    continue
                img[y1:y2, x1:x2] = blur(roi)

            # 保存先のパスをインプット側のディレクトリ構造を保って作成
            rel_path = path.relative_to(in_dir)
            save_path = out_dir / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(save_path), img)


if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent(
            """
            Face Blurrer

            This script detects faces in images and applies a Gaussian blur to them.
            It processes images in batches for efficiency.

            Usage:
                python main.py -i <input_directory> -o <output_directory>

            Default input directory: 'in'
            Default output directory: 'out'
        """
        ),
    )
    parser.add_argument("-i", "--in_dir", default="in")
    parser.add_argument("-o", "--out_dir", default="out")
    args = parser.parse_args()
    process_dir(args.in_dir, args.out_dir)
