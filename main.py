import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from utils import set_background

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

set_background("./imgs/background.png")

MODEL_PATH = "./models/best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

def model_prediction(img):
    results = model.predict(
        img,
        verbose=False
    )
    result = results[0]
    annotated_img = result.plot()
    return annotated_img, result

st.title("🧠 Brain Tumor Detection 🧠")
st.write(
    "A simple brain tumor detection application "
    "using YOLO-based computer vision."
)
st.divider()
st.header("About the Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Architecture",
        "YOLO26"
    )
with col2:
    st.metric(
        "Training Images",
        "3000+"
    )
with col3:
    st.metric(
        "Training Epochs",
        "50"
    )
st.write(
    "The model was trained using more than 3,000 brain images "
    "to detect tumor regions."
)

st.header("Training Examples")
col1, col2 = st.columns(2)
with col1:
    st.image(
        "./imgs/train_batch9242.jpg",
        caption="Training Images",
        use_container_width=True
    )
with col2:
    st.image(
        "./imgs/val_batch2_pred.jpg",
        caption="Validation Predictions",
        use_container_width=True
    )
st.divider()

st.header("Detect Brain Tumor 🔍")
st.write(
    "Upload a brain image and click the button below "
    "to run the detection model."
)
uploaded_file = st.file_uploader(
    "Upload Brain Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Uploaded Image")
    st.image(
        image,
        width=500
    )
    if st.button(
        "Detect Tumor 🔍",
        type="primary"
    ):

        with st.spinner("Detecting tumor..."):
            image_array = np.array(image)
            prediction, result = model_prediction(
                image_array
            )

        st.subheader("Detection Results")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Image")
            st.image(
                image,
                use_container_width=True
            )
        with col2:
            st.write("Detection Result")
            st.image(
                prediction,
                use_container_width=True
            )
            
        detection_count = len(result.boxes)
        if detection_count > 0:
            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )
            highest_confidence = float(
                np.max(confidences)
            )
            st.success(
                f"{detection_count} tumor region(s) detected."
            )
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Tumors Detected",
                    detection_count
                )
            with col2:
                st.metric(
                    "Highest Confidence",
                    f"{highest_confidence:.1%}"
                )
        else:
            st.info(
                "No tumor region was detected."
            )

st.divider()
st.caption(
    "⚠️ This application is intended for research and "
    "educational purposes only and should not be used "
    "as a substitute for professional medical diagnosis."
)
