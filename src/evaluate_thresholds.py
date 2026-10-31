
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "runs" / "detect" / "surfacescope_imgsz512"
    / "weights" / "best.pt"
)
DATA_YAML = ROOT / "data" / "processed" / "neu_yolo" / "data.yaml"

def main():
    model = YOLO(str(MODEL_PATH))

    for confidence in (0.10, 0.15, 0.20, 0.25):
        results = model.val(
            data=str(DATA_YAML),
            imgsz=512,
            batch=4,
            device=0,
            conf=confidence,
            plots=False,
            verbose=False,
        )

        print(
            f"conf={confidence:.2f} | "
            f"P={results.box.mp:.4f} | "
            f"R={results.box.mr:.4f} | "
            f"mAP50={results.box.map50:.4f} | "
            f"mAP50-95={results.box.map:.4f}"
        )

if __name__ == "__main__":
    main()
