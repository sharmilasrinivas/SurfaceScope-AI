from pathlib import Path
import gc
import mlflow
import torch
from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "neu_yolo" / "data.yaml"
PROJECT = ROOT / "runs" / "detect"
DB = ROOT / "mlflow.db"

# Start with a 3-epoch feasibility run.
# Change to 100 only after the short run completes without errors.
EPOCHS = 30
IMGSZ = 320
BATCH = 1
RUN_NAME = "surfacescope_rtdetr_smoke"

def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check your PyTorch installation.")

    mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
    mlflow.set_experiment("SurfaceScope-AI")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Starting RT-DETR feasibility run")
    print(f"epochs={EPOCHS}, imgsz={IMGSZ}, batch={BATCH}")

    model = RTDETR("rtdetr-l.pt")

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.set_tags({
            "project": "SurfaceScope-AI",
            "architecture": "RT-DETR",
            "stage": "feasibility",
        })
        mlflow.log_params({
            "model": "rtdetr-l.pt",
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "seed": 42,
            "device": "cuda:0",
            "workers": 0,
            "amp": False,
            "patience": 20,
        })

        model.train(
            data=str(DATA),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=0,
            workers=0,
            seed=42,
            deterministic=False,
            amp=False,
            patience=20,
            cache=False,
            project=str(PROJECT),
            name=RUN_NAME,
            exist_ok=False,
            plots=True,
            save=True,
            val=True,
        )

        run_dir = PROJECT / RUN_NAME
        best = run_dir / "weights" / "best.pt"
        results_csv = run_dir / "results.csv"

        if best.exists():
            mlflow.log_artifact(str(best), artifact_path="model_weights")
        if results_csv.exists():
            mlflow.log_artifact(str(results_csv), artifact_path="training")

        print("RT-DETR feasibility run completed.")
        print("Run directory:", run_dir)
        print("Best checkpoint:", best)

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
