
from pathlib import Path
from collections import Counter
from io import BytesIO

import streamlit as st
import torch
import pandas as pd
from PIL import Image
from ultralytics import YOLO


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT / "models" / "best.pt"
)

IMG_SIZE = 512
DEFAULT_CONFIDENCE = 0.25

DEVICE = 0 if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SurfaceScope AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background-color: #1c2533;
    border: 1px solid #344054;
    padding: 18px;
    border-radius: 12px;
}

.stButton > button {
    width: 100%;
    border-radius: 9px;
    height: 3rem;
    font-weight: bold;
}

h1, h2, h3 {
    color: #f0f4f8;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL ONCE
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🔬 SurfaceScope AI")

st.markdown(
    "### Intelligent Steel Surface Defect Detection"
)

st.caption(
    "YOLOv8n | NEU-DET Dataset | MLflow Model Tracking"
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=DEFAULT_CONFIDENCE,
    step=0.05,
)

st.sidebar.info(
    f"Input resolution: {IMG_SIZE} × {IMG_SIZE}"
)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    device_label = f"GPU: {gpu_name}"
else:
    device_label = "CPU inference"

st.sidebar.success(device_label)

st.sidebar.markdown("---")

st.sidebar.markdown("**Supported defect classes**")

for class_name in CLASS_NAMES:
    st.sidebar.write(f"• {class_name.replace('_', ' ').title()}")


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

try:
    model = load_model()
    st.success("YOLOv8 model loaded successfully!")
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.subheader("📤 Upload Steel Surface Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info(
        "Upload a steel surface image to begin defect detection."
    )

    st.stop()


# Read uploaded image
image = Image.open(uploaded_file).convert("RGB")

st.subheader("Input Image")

st.image(
    image,
    caption="Uploaded steel surface image",
    width="stretch",
)


# --------------------------------------------------
# DETECTION
# --------------------------------------------------

if st.button("🚀 Detect Defects", type="primary"):

    with st.spinner("Analyzing steel surface using YOLOv8..."):

        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)
                torch.cuda.synchronize()

            results = model.predict(
                source=image,
                imgsz=IMG_SIZE,
                conf=confidence,
                device=DEVICE,
                verbose=False,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            result = results[0]

            # Get annotated image (Ultralytics returns BGR)
            annotated_bgr = result.plot()
            annotated_rgb = annotated_bgr[:, :, ::-1]

            annotated_image = Image.fromarray(annotated_rgb)

            # Detection information
            detections = []

            for box in result.boxes:
                class_id = int(box.cls.item())
                score = float(box.conf.item())

                class_name = result.names[class_id]

                coordinates = box.xyxy[0].tolist()

                detections.append({
                    "Defect": class_name.replace("_", " ").title(),
                    "Confidence": round(score * 100, 2),
                    "X1": round(coordinates[0], 2),
                    "Y1": round(coordinates[1], 2),
                    "X2": round(coordinates[2], 2),
                    "Y2": round(coordinates[3], 2),
                })

            # Speed metrics from Ultralytics
            speed = result.speed

            preprocess_ms = speed["preprocess"]
            inference_ms = speed["inference"]
            postprocess_ms = speed["postprocess"]

            total_ms = (
                preprocess_ms
                + inference_ms
                + postprocess_ms
            )

            st.session_state["annotated_image"] = annotated_image
            st.session_state["detections"] = detections
            st.session_state["speed"] = {
                "Preprocessing": preprocess_ms,
                "Inference": inference_ms,
                "Postprocessing": postprocess_ms,
                "Total": total_ms,
            }

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "annotated_image" in st.session_state:

    detections = st.session_state["detections"]
    speed = st.session_state["speed"]

    st.divider()

    st.header("📊 Detection Results")

    total_defects = len(detections)
    unique_classes = len(set(
        item["Defect"] for item in detections
    ))

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Defects",
        total_defects
    )

    col2.metric(
        "Defect Categories",
        unique_classes
    )

    col3.metric(
        "Inference Time",
        f"{speed['Inference']:.2f} ms"
    )

    st.divider()

    # Annotated image
    st.subheader("🎯 Annotated Detection Image")

    st.image(
        st.session_state["annotated_image"],
        caption="YOLOv8 defect detection output",
        width="stretch",
    )

    # Download result
    image_buffer = BytesIO()

    st.session_state["annotated_image"].save(
        image_buffer,
        format="PNG"
    )

    st.download_button(
        label="⬇️ Download Annotated Image",
        data=image_buffer.getvalue(),
        file_name="surfacescope_detection.png",
        mime="image/png",
    )

    st.divider()

    # Defect distribution
    st.subheader("📌 Defect Distribution")

    if detections:

        counts = Counter(
            item["Defect"] for item in detections
        )

        distribution_df = pd.DataFrame(
            list(counts.items()),
            columns=["Defect", "Count"]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(
                distribution_df.set_index("Defect")
            )

        with col2:
            st.dataframe(
                pd.DataFrame(detections),
                width="stretch",
                hide_index=True,
            )

    else:
        st.warning(
            "No defects detected at the selected confidence threshold."
        )

    # Inference performance
    st.divider()

    st.subheader("⚡ Inference Performance")

    performance_df = pd.DataFrame(
        [
            {"Stage": stage, "Time (ms)": round(value, 3)}
            for stage, value in speed.items()
        ]
    )

    st.dataframe(
        performance_df,
        width="stretch",
        hide_index=True,
    )

    if torch.cuda.is_available():
        memory_mb = (
            torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        )

        st.caption(
            f"Peak PyTorch GPU memory allocated: {memory_mb:.2f} MB"
        )

    st.success("Defect detection completed successfully!")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SurfaceScope AI | YOLOv8n | Steel Surface Defect Detection"
)
