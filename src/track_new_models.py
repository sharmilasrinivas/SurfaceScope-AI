from pathlib import Path
import mlflow

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
mlflow.set_experiment("SurfaceScope-AI")

print("Tracking URI:", mlflow.get_tracking_uri())
print("Experiment:", mlflow.get_experiment_by_name("SurfaceScope-AI").name)
print("Existing experiment database will be reused.")
