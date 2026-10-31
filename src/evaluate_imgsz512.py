
from pathlib import Path

import mlflow
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "surfacescope_imgsz512"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    ROOT
    / "data"
    / "processed"
    / "neu_yolo"
    / "data.yaml"
)

MLFLOW_DB = ROOT / "mlflow.db"
IMAGE_SIZE = 512


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )
    mlflow.set_experiment("SurfaceScope-AI")

    model = YOLO(str(MODEL_PATH))

    # Evaluate on the same validation split as the baseline.
    results = model.val(
        data=str(DATA_YAML),
        imgsz=IMAGE_SIZE,
        batch=4,
        device=0,
        plots=True,
        project=str(ROOT / "runs" / "detect"),
        name="val_imgsz512",
        exist_ok=True,
    )

    metrics = {
        "val_precision": float(results.box.mp),
        "val_recall": float(results.box.mr),
        "val_map50": float(results.box.map50),
        "val_map50_95": float(results.box.map),
    }

    with mlflow.start_run(run_name="eval-yolov8n-imgsz512"):
        mlflow.set_tag("run_type", "validation")
        mlflow.set_tag("model", "YOLOv8n")
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("batch_size", 4)
        mlflow.log_param("dataset", "NEU-DET-validation")

        mlflow.log_metrics(metrics)

        plot_dir = ROOT / "runs" / "detect" / "val_imgsz512"
        for filename in (
            "confusion_matrix.png",
            "results.png",
            "PR_curve.png",
            "F1_curve.png",
            "P_curve.png",
            "R_curve.png",
        ):
            artifact = plot_dir / filename
            if artifact.exists():
                mlflow.log_artifact(
                    str(artifact),
                    artifact_path="validation_plots",
                )

    print("\nValidation metrics logged to MLflow:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
