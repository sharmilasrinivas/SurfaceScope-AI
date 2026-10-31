
from pathlib import Path

import mlflow
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "reports" / "confidence_thresholds.csv"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    # Use the same SQLite database as the existing MLflow experiments.
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("SurfaceScope-AI")

    df = pd.read_csv(CSV_PATH)

    required_columns = {
        "confidence", "precision", "recall", "map50", "map50_95"
    }
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV must contain these columns: {sorted(required_columns)}"
        )

    with mlflow.start_run(run_name="confidence-threshold-comparison"):
        mlflow.set_tag("experiment_type", "confidence_threshold_evaluation")
        mlflow.log_param("model", "YOLOv8n")
        mlflow.log_param("image_size", 320)
        mlflow.log_param("dataset", "NEU-DET validation")
        mlflow.log_param("threshold_count", len(df))

        for _, row in df.iterrows():
            confidence = float(row["confidence"])
            prefix = f"conf_{confidence:.2f}"

            mlflow.log_metric(f"{prefix}_precision", float(row["precision"]))
            mlflow.log_metric(f"{prefix}_recall", float(row["recall"]))
            mlflow.log_metric(f"{prefix}_map50", float(row["map50"]))
            mlflow.log_metric(f"{prefix}_map50_95", float(row["map50_95"]))

        mlflow.log_artifact(str(CSV_PATH), artifact_path="evaluation")

    print("Confidence-threshold comparison logged successfully!")
    print("MLflow database:", MLFLOW_DB)
    print("CSV logged:", CSV_PATH)


if __name__ == "__main__":
    main()
