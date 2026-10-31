
from pathlib import Path
import mlflow

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")

# Confirmed 50-epoch run ID from your MLflow output
RUN_ID = "854f5dd329a042049c0a26c1195356af"

metrics = {
    "val_precision": 0.725,
    "val_recall": 0.681,
    "val_map50": 0.733,
    "val_map50_95": 0.384,
}

with mlflow.start_run(run_id=RUN_ID):
    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    mlflow.set_tag("validation_metrics_source", "Ultralytics validation output")
    mlflow.set_tag("experiment_description", "YOLOv8n 512px 50-epoch run")

print("Validation metrics logged to the existing 50-epoch run.")
