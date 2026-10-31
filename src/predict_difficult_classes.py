
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "runs" / "detect" / "surfacescope_imgsz512"
    / "weights" / "best.pt"
)

IMAGE_DIR = ROOT / "data" / "processed" / "neu_yolo" / "images" / "validation"
REPORT_DIR = ROOT / "reports"

TARGET_CLASSES = {
    0: "crazing",
    4: "rolled-in_scale",
}

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    for class_id, class_name in TARGET_CLASSES.items():
        list_path = REPORT_DIR / f"validation_{class_name}_images.txt"

        if not list_path.exists():
            raise FileNotFoundError(f"Image list not found: {list_path}")

        image_paths = [
            Path(line.strip())
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        output_name = f"diagnostic_{class_name.replace('-', '_')}"

        print(f"\nRunning predictions for {class_name}")
        print(f"Images: {len(image_paths)}")

        model.predict(
            source=[str(path) for path in image_paths],
            imgsz=512,
            conf=0.10,
            device=0,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(ROOT / "runs" / "detect"),
            name=output_name,
            exist_ok=True,
            verbose=False,
        )

        print(f"Predictions saved under runs/detect/{output_name}")

if __name__ == "__main__":
    main()
