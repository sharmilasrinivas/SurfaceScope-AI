# SurfaceScope AI

YOLOv8-based steel surface defect detection using the NEU-DET dataset, with model evaluation and MLflow experiment tracking.

## Project overview

SurfaceScope AI detects and localizes six categories of steel surface defects in images:

- Crazing
- Inclusion
- Patches
- Pitted surface
- Rolled-in scale
- Scratches

The project trains an object-detection model, evaluates validation performance, compares experiments, and uses MLflow to track results and register the selected model.

> **Research/demo note:** Results reported here are validation results on the prepared NEU-DET dataset. They do not guarantee performance on unseen factory production lines.

## Selected model

- **Model:** YOLOv8n
- **Selected checkpoint:** `runs/detect/surfacescope_imgsz512-2/weights/best.pt`
- **Input size:** 512 × 512
- **Prediction confidence threshold:** 0.25
- **Prediction device:** CUDA GPU (`device=0`) in the current local setup
- **MLflow registered model:** `SurfaceScope-YOLOv8n`
- **Registered version:** 1 (`READY`)

## Dataset and preprocessing

The project uses the NEU-DET steel surface defect dataset prepared in YOLO format.

The processed dataset is expected at:

```text
data/processed/neu_yolo/
├── data.yaml
├── images/
│   ├── train/
│   └── validation/
└── labels/
    ├── train/
    └── validation/
```

The recorded split contains 1,439 training images and 360 validation images. The six defect categories are represented in the dataset configuration. Check `data/processed/neu_yolo/data.yaml` for the class names and dataset paths used by the local training setup.

## Validation results

The selected 512-pixel model achieved the following recorded validation metrics:

| Metric | Result |
|---|---:|
| Precision | 0.7270 (72.70%) |
| Recall | 0.6828 (68.28%) |
| mAP@50 | 0.7340 (73.40%) |
| mAP@50–95 | 0.3849 (38.49%) |

These figures summarize validation performance. mAP@50–95 is stricter than mAP@50 because it averages performance over a range of intersection-over-union thresholds.

## Model comparison and analysis

Existing analysis outputs are stored in `reports/`, including:

- `model_comparison.csv` — comparison of evaluated model configurations
- `confidence_thresholds.csv` — confidence-threshold analysis
- `difficult_class_error_analysis.csv` and the confidence-specific variants — difficult-class error analysis
- `validation_crazing_images.txt` and `validation_rolled-in_scale_images.txt` — lists of selected validation images for analysis

Use the CSV files as the source of truth for detailed comparisons and class-level analysis.

## Requirements

The project was developed in a Windows environment with Python 3.10 and a local NVIDIA GPU. The repository includes `requirements.txt`. A compatible NVIDIA driver and CUDA-enabled PyTorch installation are needed to use the GPU; CPU inference may be possible by changing the prediction device in the script.

## Setup (Windows PowerShell)

Open PowerShell in the project root.

If the virtual environment already exists, activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

If dependencies are already installed and the environment works, do not reinstall them unnecessarily.

## Run a prediction

1. Place the image to inspect at the project root and name it `sample.jpg`, or update `IMAGE_PATH` in `src/predict.py`.
2. Confirm that the selected checkpoint exists at `runs/detect/surfacescope_imgsz512-2/weights/best.pt`.
3. Run:

```powershell
python .\src\predict.py
```

The script loads the selected checkpoint, predicts at 512 × 512 with a confidence threshold of 0.25, prints detected class names, confidence scores and bounding-box coordinates, and saves annotated output under:

```text
runs/inference/final_model/
```

To open the output folder:

```powershell
explorer ".\runs\inference\final_model"
```

If the script reports that the image is missing, put an image named `sample.jpg` in the project root or change `IMAGE_PATH` in `src/predict.py`. If the checkpoint is missing, check the selected model path above.

## MLflow tracking and model registry

The project has an MLflow tracking database at `mlflow.db`. The registered model is `SurfaceScope-YOLOv8n`, version 1, with status `READY`.

Start the MLflow UI from the project root:

```powershell
mlflow ui --backend-store-uri "sqlite:///C:/Users/sharm/OneDrive/Desktop/SurfaceScopeAI/mlflow.db"
```

Then open the local address printed by MLflow in your browser (commonly `http://127.0.0.1:5000`).

The tracking database and artifact paths are local to the development machine. If the project is moved to another computer, update the paths and verify that the required artifacts are available. A registered model version with status `READY` confirms registry creation; it does not, by itself, mean that a production inference service has been deployed.

## Main project paths

```text
SurfaceScopeAI/
├── data/
│   └── processed/neu_yolo/
├── reports/
│   ├── model_comparison.csv
│   ├── confidence_thresholds.csv
│   └── difficult_class_error_analysis.csv
├── runs/
│   ├── detect/surfacescope_imgsz512-2/weights/best.pt
│   └── inference/final_model/
├── src/
│   ├── predict.py
│   ├── train.py
│   ├── train_imgsz512.py
│   └── evaluate_imgsz512.py
├── mlflow.db
├── requirements.txt
└── README.md
```

This is a high-level overview; the local project may contain additional scripts, experiments and artifacts.

## Limitations

- The reported metrics are based on the available validation split, not independent industrial deployment testing.
- Some defect categories are harder to detect than others; consult the existing class-wise error-analysis reports.
- Confidence scores reflect the model's predictions and are not a guarantee that a detected defect is real.
- Performance may change with different lighting, cameras, steel textures, image resolutions and production conditions.
- The current prediction script is a local demo, not a deployed production API or web application.

## Future improvements

Potential next steps, if the project is extended, include testing on independently collected industrial images, reviewing false positives and missed defects, calibrating confidence thresholds on a separate validation set, and packaging inference behind an application interface.

## Project status

The selected model has been trained and evaluated, the model comparison and error-analysis reports exist, MLflow tracking and registration have been verified, and the local prediction script has successfully generated annotated output.
