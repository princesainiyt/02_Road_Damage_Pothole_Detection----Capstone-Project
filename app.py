
from pathlib import Path
from io import BytesIO

import streamlit as st
from PIL import Image
from ultralytics import YOLO

PROJECT_DIR = Path.home() / "RoadDamageDetection"
MODEL_PATH = (
    PROJECT_DIR
    / "results"
    / "yolov8n_baseline"
    / "weights"
    / "best.pt"
)

st.set_page_config(
    page_title="Road Damage and Pothole Detection",
    page_icon="🚧",
    layout="wide"
)

st.title("Road Damage and Pothole Detection")
st.write("Upload a road image to detect potholes using YOLOv8.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))

try:
    model = load_model()
    st.success("YOLOv8 model loaded successfully.")
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

confidence = st.slider(
    "Detection confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Detecting potholes..."):
        result = model.predict(
            source=image,
            imgsz=640,
            conf=confidence,
            device=0,
            verbose=False
        )[0]

    annotated_image = result.plot()[:, :, ::-1]

    original_col, result_col = st.columns(2)

    with original_col:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with result_col:
        st.subheader("Detection Result")
        st.image(annotated_image, use_container_width=True)

    boxes = result.boxes
    count = len(boxes) if boxes is not None else 0

    st.metric("Potholes detected", count)

    if count:
        detections = []

        for index, box in enumerate(boxes, start=1):
            detections.append({
                "Detection": index,
                "Class": result.names[int(box.cls.item())],
                "Confidence": round(float(box.conf.item()), 4),
                "X1": round(float(box.xyxy[0][0].item()), 1),
                "Y1": round(float(box.xyxy[0][1].item()), 1),
                "X2": round(float(box.xyxy[0][2].item()), 1),
                "Y2": round(float(box.xyxy[0][3].item()), 1)
            })

        st.subheader("Detection Details")
        st.dataframe(detections, use_container_width=True)
    else:
        st.info(
            "No potholes detected above this confidence threshold. "
            "This does not guarantee that the road is pothole-free."
        )

    buffer = BytesIO()
    Image.fromarray(annotated_image).save(buffer, format="PNG")

    st.download_button(
        "Download Annotated Image",
        data=buffer.getvalue(),
        file_name="pothole_detection_result.png",
        mime="image/png"
    )

st.caption(
    "Prototype for road inspection assistance. "
    "All detections require human verification."
)
