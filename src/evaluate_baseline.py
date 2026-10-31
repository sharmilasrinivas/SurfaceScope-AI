
from pathlib import Path

import mlflow
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "runs" / "detect" / "surfacescope_baseline"
    / "weights" / "best.pt"
)
DATA_YAML = ROOT / "data" / "processed" / "neu_yolo" / "data.yaml"
MLFLOW_DB = ROOT / "mlflow.db"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment("SurfaceScope-AI")

    model = YOLO(str(MODEL_PATH))
    results = model.val(
        data=str(DATA_YAML),
        imgsz=320,
        batch=8,
        device=0,
        plots=True,
        project=str(ROOT / "runs" / "detect"),
        name="val_baseline320",
        exist_ok=True,
    )

    metrics = {
        "val_precision": float(results.box.mp),
        "val_recall": float(results.box.mr),
        "val_map50": float(results.box.map50),
        "val_map50_95": float(results.box.map),
    }

    with mlflow.start_run(run_name="eval-yolov8n-baseline320"):
        mlflow.set_tag("run_type", "validation")
        mlflow.set_tag("model", "YOLOv8n")
        mlflow.log_params({
            "image_size": 320,
            "batch_size": 8,
            "dataset": "NEU-DET-validation",
        })
        mlflow.log_metrics(metrics)

    print("\nBaseline validation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
