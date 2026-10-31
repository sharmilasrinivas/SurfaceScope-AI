from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

IMAGE_PATH = PROJECT_ROOT / "sample.jpg"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "inference"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=512,
        conf=0.25,
        device=0,
        save=True,
        project=str(OUTPUT_DIR),
        name="final_model",
        exist_ok=True,
    )

    for result in results:
        print("\nDetection results:")
        if len(result.boxes) == 0:
            print("No defects detected.")
        else:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                label = result.names[class_id]
                coordinates = box.xyxy[0].tolist()

                print(
                    f"Defect: {label} | "
                    f"Confidence: {confidence:.3f} | "
                    f"Box: {coordinates}"
                )

    print(f"\nAnnotated image location: {OUTPUT_DIR / 'final_model'}")


if __name__ == "__main__":
    main()