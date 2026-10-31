
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "runs/detect/surfacescope_imgsz512/weights/best.pt"
IMAGE_DIR = ROOT / "data/processed/neu_yolo/images/validation"
LABEL_DIR = ROOT / "data/processed/neu_yolo/labels/validation"
OUT_DIR = ROOT / "reports/difficult_class_comparison"

CLASSES = {
    0: "crazing",
    4: "rolled-in_scale",
}

def draw_boxes(image, label_path, predictions=None, class_filter=None):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Ground-truth boxes: green
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue

            cls, xc, yc, bw, bh = map(float, parts)
            cls = int(cls)

            if class_filter is not None and cls != class_filter:
                continue

            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            x2 = (xc + bw / 2) * width
            y2 = (yc + bh / 2) * height

            draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
            draw.text((x1, max(0, y1 - 14)), f"GT {CLASSES.get(cls, cls)}", fill="lime")

    # Predicted boxes: red
    if predictions is not None:
        for box in predictions.boxes:
            cls = int(box.cls.item())

            if class_filter is not None and cls != class_filter:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf.item())

            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text(
                (x1, min(height - 15, y1 + 2)),
                f"P {CLASSES.get(cls, cls)} {conf:.2f}",
                fill="red",
            )

    return image


def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL))

    for class_id, class_name in CLASSES.items():
        list_path = ROOT / "reports" / f"validation_{class_name}_images.txt"
        image_paths = [
            Path(x.strip())
            for x in list_path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]

        # Inspect the first 6 images for each class.
        for image_path in image_paths[:6]:
            label_path = LABEL_DIR / f"{image_path.stem}.txt"
            original = Image.open(image_path).convert("RGB")

            result = model.predict(
                source=str(image_path),
                imgsz=512,
                conf=0.10,
                device=0,
                verbose=False,
            )[0]

            comparison = draw_boxes(
                original,
                label_path,
                result,
                class_filter=class_id,
            )

            output_path = OUT_DIR / f"{class_name}_{image_path.stem}.jpg"
            comparison.save(output_path, quality=95)
            print(output_path)

    print(f"\nComparison images saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
