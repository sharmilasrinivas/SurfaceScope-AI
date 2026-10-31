
from pathlib import Path

import torch
import mlflow
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "neu_yolo"
    / "data.yaml"
)

MODEL_NAME = "yolov8n.pt"
EPOCHS = 50
IMAGE_SIZE = 512
BATCH_SIZE = 4
SEED = 42

RUN_NAME = "yolov8n-imgsz512"
OUTPUT_NAME = "surfacescope_imgsz512"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is unavailable. Check your PyTorch installation."
        )

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset configuration:", DATA_YAML)

    # Use the same MLflow database and experiment as the baseline.
    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )
    mlflow.set_experiment("SurfaceScope-AI")

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.set_tag("experiment_group", "image_resolution_comparison")
        mlflow.set_tag("model", "YOLOv8n")
        mlflow.set_tag("dataset", "NEU-DET")

        mlflow.log_params({
            "model_name": MODEL_NAME,
            "epochs": EPOCHS,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "device": "cuda:0",
            "optimizer_deterministic": True,
        })

        model = YOLO(MODEL_NAME)

        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            seed=SEED,
            deterministic=True,
            workers=0,
            device=0,
            project=str(PROJECT_ROOT / "runs" / "detect"),
            name=OUTPUT_NAME,
            exist_ok=False,
            plots=True,
            save=True,
        )

        save_dir = Path(results.save_dir)
        best_weights = save_dir / "weights" / "best.pt"
        results_csv = save_dir / "results.csv"
        args_yaml = save_dir / "args.yaml"

        print("\nTraining completed.")
        print("Results directory:", save_dir)
        print("Best model:", best_weights)

        # Log the training artifacts to this MLflow run.
        if best_weights.exists():
            mlflow.log_artifact(
                str(best_weights),
                artifact_path="model_weights",
            )

        for artifact in (results_csv, args_yaml):
            if artifact.exists():
                mlflow.log_artifact(str(artifact))

        if not best_weights.exists():
            raise FileNotFoundError(
                f"Training finished but best.pt was not found: "
                f"{best_weights}"
            )


if __name__ == "__main__":
    main()
