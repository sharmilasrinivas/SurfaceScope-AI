
from pathlib import Path

import torch
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
IMAGE_SIZE = 320
BATCH_SIZE = 8
SEED = 42


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Dataset configuration:", DATA_YAML)

    # Load pretrained weights for transfer learning.
    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        seed=SEED,
        deterministic=True,
        workers=0,  # Reliable default for Windows
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="surfacescope_baseline_ep50",
        exist_ok=False,
        plots=True,
        save=True,
    )

    print("\nTraining completed.")
    print("Results directory:", results.save_dir)
    print(
        "Best model:",
        Path(results.save_dir) / "weights" / "best.pt",
    )


if __name__ == "__main__":
    main()
