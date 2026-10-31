
from pathlib import Path
import csv
import yaml
import mlflow
import re
import math

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs" / "detect" / "surfacescope_baseline"
MODEL_PATH = RUN_DIR / "weights" / "best.pt"
ARGS_PATH = RUN_DIR / "args.yaml"
RESULTS_CSV = RUN_DIR / "results.csv"
DATA_YAML = ROOT / "data" / "processed" / "neu_yolo" / "data.yaml"

# Use SQLite as the MLflow tracking backend
MLFLOW_DB = ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")

mlflow.set_experiment("SurfaceScope-AI")

# Load saved training parameters
with ARGS_PATH.open("r", encoding="utf-8") as file:
    train_args = yaml.safe_load(file)

# Load the final epoch metrics
with RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))

if not rows:
    raise ValueError(f"No metrics found in {RESULTS_CSV}")

final_metrics = rows[-1]

def find_metric(*candidates):
    for candidate in candidates:
        if candidate in final_metrics:
            return float(final_metrics[candidate])
    return None

with mlflow.start_run(run_name="baseline-yolov8n"):
    # Record parameters
    for key in (
        "model", "epochs", "imgsz", "batch", "optimizer",
        "seed", "device", "workers"
    ):
        if key in train_args and train_args[key] is not None:
            mlflow.log_param(key, str(train_args[key]))

    # Record final-epoch training/validation metrics where available
    for key, value in final_metrics.items():
        try:
            metric_value = float(value)

            if math.isfinite(metric_value):
                # Replace characters MLflow does not allow
                safe_key = re.sub(r"[^a-zA-Z0-9_\-./ ]", "_", key)
                mlflow.log_metric(safe_key, metric_value)
        except (TypeError, ValueError):
            continue

    # Record the confirmed best-model validation metrics
    # Values are from your latest `yolo detect val` output.
    mlflow.log_metrics({
        "val_precision": 0.648,
        "val_recall": 0.671,
        "val_map50": 0.717,
        "val_map50_95": 0.380,
    })

    # Save relevant configuration and evaluation artifacts
    for path in (ARGS_PATH, RESULTS_CSV, DATA_YAML):
        if path.exists():
            mlflow.log_artifact(str(path))

    if MODEL_PATH.exists():
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_weights")
    else:
        raise FileNotFoundError(f"Best model not found: {MODEL_PATH}")

    print("Baseline successfully logged to MLflow.")
    print(f"Model: {MODEL_PATH}")
