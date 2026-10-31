# SurfaceScope AI

### Intelligent Steel Surface Defect Detection


[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://surfacescope-ai.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/sharmilasrinivas/SurfaceScope-AI)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-green)](https://github.com/ultralytics/ultralytics)


**SurfaceScope AI** is a computer-vision application that detects and
localizes steel surface defects using a YOLOv8n object-detection model
trained on the NEU-DET dataset. It combines model evaluation, MLflow
experiment tracking and model registration, GPU-accelerated local
inference, and an interactive Streamlit web application.

-   **Live application:** https://surfacescope-ai.streamlit.app/
-   **GitHub repository:**
    https://github.com/sharmilasrinivas/SurfaceScope-AI

> **Research/demo note:** The reported metrics are validation results on
> the prepared NEU-DET validation split. They do not guarantee
> performance on unseen factory production lines.

## Features

-   Upload steel surface images through an interactive Streamlit
    interface.
-   Detect and localize six steel defect categories.
-   Display annotated images with bounding boxes and confidence scores.
-   Show total detections, defect-category distribution, and detection
    details.
-   Adjust the confidence threshold.
-   Run local inference using a CUDA-enabled NVIDIA GPU when available,
    with CPU fallback.
-   Review model-comparison, confidence-threshold, and difficult-class
    analysis reports.
-   Track experiments and register the selected model with MLflow.
-   Access a public demonstration through Streamlit Community Cloud.

## Supported defect categories

1.  Crazing
2.  Inclusion
3.  Patches
4.  Pitted surface
5.  Rolled-in scale
6.  Scratches

## Model and dataset

  
## Model and Dataset Configuration

| Item | Configuration |
|---|---|
| Model | YOLOv8n |
| Task | Object Detection |
| Dataset | NEU-DET Steel Surface Defect Dataset |
| Input Size | 512 × 512 |
| Selected Checkpoint | `models/best.pt` |
| Default Application Confidence Threshold | 0.25 |
| Local Development Python | 3.10.11 |
| Local GPU | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) |
| Cloud Deployment | Streamlit Community Cloud (CPU Inference) |


The prepared YOLO-format dataset used in the recorded experiment
contains **1,439 training images** and **360 validation images**,
covering six defect categories. The dataset itself is not included in
this repository. Obtain it separately and configure the dataset paths
before reproducing training.


## Validation Performance

The selected 512-pixel YOLOv8n model achieved the following validation results:

### Overall Validation Results

| Evaluation Metric | Score | Percentage |
|---|---:|---:|
| Precision | 0.7270 | 72.70% |
| Recall | 0.6828 | 68.28% |
| mAP@50 | 0.7340 | 73.40% |
| mAP@50–95 | 0.3849 | 38.49% |

### Class-wise Validation Results

| Defect Category | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Crazing | 0.672 | 0.342 | 0.530 | 0.227 |
| Inclusion | 0.758 | 0.805 | 0.828 | 0.463 |
| Patches | 0.861 | 0.891 | 0.926 | 0.576 |
| Pitted Surface | 0.906 | 0.690 | 0.798 | 0.415 |
| Rolled-in Scale | 0.578 | 0.492 | 0.516 | 0.227 |
| Scratches | 0.584 | 0.876 | 0.807 | 0.400 |

### Performance Analysis

- **Best-performing category:** Patches (mAP@50: 0.926).
- **Highest precision:** Pitted Surface (0.906).
- **Highest recall:** Patches (0.891).
- **Challenging categories:** Crazing and Rolled-in Scale.

The results indicate that model performance varies across defect categories. 
Refer to the CSV reports in `reports/` for additional analysis.

> **Note:** These metrics represent validation performance on the prepared NEU-DET validation dataset. They are not independent industrial test results.


## Application workflow

1.  Upload a steel surface image.
2.  The application loads the selected YOLOv8n checkpoint.
3.  The image is passed to the detector at 512-pixel inference size.
4.  The model returns predicted classes, confidence scores, and bounding
    boxes.
5.  The application displays the annotated image and detection summary.
6.  Download the annotated result.

The deployed application has been tested with multiple input images. In
one documented cloud test, it detected three objects across two
categories. Results vary with the input image and confidence threshold.

**Live demo:** https://surfacescope-ai.streamlit.app/

## Technology stack

  
## Technology Stack

| Area | Technologies |
|---|---|
| Programming Language | Python |
| Object Detection | Ultralytics YOLOv8n |
| Deep Learning | PyTorch, Torchvision |
| Computer Vision | OpenCV, Pillow |
| Data Analysis | Pandas |
| Interactive Interface | Streamlit |
| Experiment Tracking | MLflow |
| Dataset | NEU-DET |
| Local GPU Acceleration | NVIDIA CUDA, RTX 3050 |
| Version Control | Git and GitHub |
| Public Demo Hosting | Streamlit Community Cloud |


## Repository structure

``` text
SurfaceScope-AI/
├── app.py
├── packages.txt
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── best.pt
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_annotation_analysis.ipynb
│   ├── 03_dataset_validation.ipynb
│   ├── 04_annotation_quality.ipynb
│   └── 05_annotation_audit.ipynb
├── reports/
│   ├── confidence_thresholds.csv
│   ├── difficult_class_error_analysis.csv
│   ├── difficult_class_error_analysis_conf015.csv
│   ├── difficult_class_error_analysis_conf020.csv
│   ├── difficult_class_error_analysis_conf025.csv
│   ├── model_comparison.csv
│   ├── validation_crazing_images.txt
│   └── validation_rolled-in_scale_images.txt
└── src/
    ├── predict.py
    ├── train.py
    ├── train_imgsz512.py
    ├── train_imgsz512_ep50.py
    ├── evaluate_baseline.py
    ├── evaluate_imgsz512.py
    ├── evaluate_difficult_classes.py
    ├── evaluate_thresholds.py
    ├── compare_models.py
    ├── prepare_yolo_dataset.py
    └── additional experiment and analysis scripts
```

The repository excludes the dataset, local virtual environment, MLflow
database, training output directories, and experimental model
checkpoints. The selected deployment checkpoint is included at
`models/best.pt`.

## Run locally

### 1. Clone the repository

``` bash
git clone https://github.com/sharmilasrinivas/SurfaceScope-AI.git
cd SurfaceScope-AI
```

### 2. Create and activate a virtual environment

Windows PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU inference, install a PyTorch and Torchvision build compatible
with the machine's NVIDIA driver and CUDA environment. The appropriate
CUDA-enabled installation can differ between machines.

### 4. Launch the application

``` powershell
streamlit run app.py
```

Open the local address printed in the terminal, usually
`http://localhost:8501`. Upload a JPG, JPEG, or PNG steel surface image,
adjust the confidence threshold, and select **Detect Defects**.

### GPU and CPU behavior

The application selects CUDA when PyTorch reports that a CUDA device is
available. Otherwise, it uses CPU inference.

Check the local GPU environment:

``` powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The project was tested locally with an NVIDIA GeForce RTX 3050 Laptop
GPU with 4 GB of VRAM. Actual speed and memory usage depend on the
device, software versions, image, and workload.

## Run prediction from Python

The standalone prediction script is `src/predict.py`. Run it from the
repository root:

``` powershell
python .\src\predict.py
```

The script uses the selected checkpoint at `models/best.pt` and the
configured sample-image path. If it expects `sample.jpg`, place an
appropriate test image in the project root or update the image path in
the script.

The script prints detected classes, confidence scores, and bounding-box
coordinates, and saves annotated output under its configured inference
output directory.

## MLflow experiment tracking

MLflow was used during local experimentation to record evaluation
metrics and artifacts and to register the selected model.


## MLflow Model Registry

The selected YOLOv8n model was registered and verified using MLflow Model Registry.

| Registry Field | Recorded Value |
|---|---|
| Registered Model | `SurfaceScope-YOLOv8n` |
| Model Version | `1` |
| Registration Status | `READY` |


The local tracking database was `mlflow.db`, with experiment artifacts
stored under `mlruns/`. These local files are excluded from GitHub.

To open a locally available MLflow database from the project root:

``` powershell
mlflow ui --backend-store-uri "sqlite:///mlflow.db"
```

Then open the address printed in the terminal, commonly
`http://127.0.0.1:5000`.

A registered model version with status `READY` confirms registry
creation. It does not, by itself, mean that MLflow model serving or a
separate production inference endpoint has been deployed.

## Streamlit Cloud deployment

The public demonstration is hosted at:

**https://surfacescope-ai.streamlit.app/**

Deployment files:

-   `app.py` --- Streamlit application entry point.
-   `requirements.txt` --- Python dependencies.
-   `packages.txt` --- Linux system packages required by the cloud
    environment.
-   `models/best.pt` --- selected trained checkpoint.

The cloud application was configured with Python 3.12. The
`packages.txt` file contains the Linux packages used to resolve the
OpenCV `libGL.so.1` startup dependency:

``` text
libgl1
libglib2.0-0
```

**Hardware distinction:** Streamlit Community Cloud runs the public demo
using its hosted environment and CPU inference. It does not use the
NVIDIA RTX 3050 in the developer's laptop. Local execution can use the
laptop GPU when CUDA is available. Hosted GPU inference would require a
separately provisioned GPU-enabled service.

## Limitations

-   Metrics are based on the prepared NEU-DET validation split, not an
    independent industrial production test set.
-   Detection quality differs across classes; crazing and rolled-in
    scale were more challenging in the recorded class-wise evaluation.
-   Confidence scores are model outputs, not calibrated probabilities or
    guarantees that a defect is real.
-   Lighting, camera characteristics, surface texture, image resolution,
    and production conditions may affect performance.
-   The public Streamlit demo is a demonstration application, not a
    validated industrial inspection or safety-critical system.
-   The public cloud deployment uses CPU inference; local GPU
    performance should not be assumed for the hosted application.
-   Dataset files and local MLflow tracking data are not included in the
    repository.

## Future improvements

-   Evaluate the model on independently collected industrial steel
    images.
-   Improve detection of challenging defect categories using additional
    representative data.
-   Review false positives and missed detections with domain experts.
-   Calibrate confidence thresholds using a separate evaluation set.
-   Add batch image processing and downloadable detection summaries.
-   Add automated tests, application monitoring, and structured
    inference logging.
-   Use a dedicated GPU-enabled cloud service if hosted GPU inference is
    required.

## Project status

**Implemented and publicly demonstrated.**

-   YOLOv8n steel defect detector trained and evaluated on the prepared
    NEU-DET dataset.
-   Model comparison and difficult-class analysis reports generated.
-   MLflow experiment tracking and model registration verified.
-   Local CUDA inference verified on an NVIDIA RTX 3050 Laptop GPU.
-   Interactive Streamlit application implemented and tested with
    multiple images.
-   Application deployed and tested on Streamlit Community Cloud.

------------------------------------------------------------------------

**SurfaceScope AI - Intelligent Steel Surface Defect Detection**

-   GitHub: https://github.com/sharmilasrinivas/SurfaceScope-AI
-   Live demo: https://surfacescope-ai.streamlit.app/
