import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from utils import set_background

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# BACKGROUND
# ============================================================

set_background("./imgs/background.png")

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: transparent;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Hide Streamlit default elements */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* ---------- HERO ---------- */

    .hero {
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;

        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(12px);

        border: 1px solid rgba(255, 255, 255, 0.6);

        box-shadow:
            0 10px 40px rgba(0, 0, 0, 0.08);

        text-align: center;
    }

    .hero-icon {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;

        background: linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #64748b;
        margin-bottom: 1rem;
    }

    .badge {
        display: inline-block;
        padding: 0.45rem 1rem;

        border-radius: 999px;

        background: #eff6ff;
        color: #2563eb;

        font-size: 0.85rem;
        font-weight: 600;
    }


    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 1.7rem;
        font-weight: 750;
        color: #0f172a;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .section-description {
        color: #64748b;
        margin-bottom: 1.5rem;
    }


    /* ---------- INFO CARDS ---------- */

    .info-card {
        background: rgba(255, 255, 255, 0.94);

        border-radius: 18px;

        padding: 1.5rem;

        border: 1px solid #e2e8f0;

        box-shadow:
            0 8px 25px rgba(15, 23, 42, 0.06);

        height: 100%;
    }

    .info-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .info-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }

    .info-text {
        color: #64748b;
        line-height: 1.6;
        font-size: 0.95rem;
    }


    /* ---------- UPLOAD CARD ---------- */

    .upload-card {
        background: rgba(255, 255, 255, 0.96);

        border-radius: 24px;

        padding: 2rem;

        border: 1px solid #e2e8f0;

        box-shadow:
            0 12px 35px rgba(15, 23, 42, 0.08);

        margin-top: 1rem;
    }


    /* ---------- RESULT CARD ---------- */

    .result-card {
        background: rgba(255, 255, 255, 0.96);

        border-radius: 24px;

        padding: 2rem;

        border: 1px solid #e2e8f0;

        box-shadow:
            0 12px 35px rgba(15, 23, 42, 0.08);

        margin-top: 2rem;
    }


    /* ---------- METRICS ---------- */

    .metric-card {
        background: #f8fafc;

        border-radius: 16px;

        padding: 1.2rem;

        text-align: center;

        border: 1px solid #e2e8f0;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.5rem;
        font-weight: 750;
    }


    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;

        border-radius: 12px;

        padding: 0.75rem 1rem;

        font-weight: 700;

        border: none;

        background: linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );

        color: white;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 8px 20px rgba(37, 99, 235, 0.25);
    }


    /* ---------- IMAGE ---------- */

    .image-caption {
        text-align: center;

        color: #64748b;

        font-size: 0.9rem;

        margin-top: 0.5rem;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;

        margin-top: 4rem;

        padding-top: 2rem;

        border-top: 1px solid rgba(148, 163, 184, 0.3);

        color: #64748b;

        font-size: 0.85rem;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "./models/best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()


# ============================================================
# MODEL PREDICTION
# ============================================================

def model_prediction(img):

    results = model.predict(
        img,
        verbose=False
    )

    result = results[0]

    annotated_img = result.plot()

    return annotated_img, result


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-icon">🧠</div>

    <div class="hero-title">
        Brain Tumor Detection
    </div>

    <div class="hero-subtitle">
        AI-powered brain tumor detection using Computer Vision
    </div>

    <span class="badge">
        YOLO-based Object Detection
    </span>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">About the Model</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'An overview of the training process and model architecture.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="info-card">

        <div class="info-icon">🤖</div>

        <div class="info-title">
            YOLO Architecture
        </div>

        <div class="info-text">
            The model uses a YOLO-based architecture
            for detecting brain tumor regions from
            medical images.
        </div>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="info-card">

        <div class="info-icon">🖼️</div>

        <div class="info-title">
            Training Dataset
        </div>

        <div class="info-text">
            The model was trained using more than
            3,000 brain images to learn tumor
            detection patterns.
        </div>

    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="info-card">

        <div class="info-icon">⚡</div>

        <div class="info-title">
            Fast Detection
        </div>

        <div class="info-text">
            YOLO enables efficient object detection
            suitable for interactive AI applications.
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TRAINING VISUALIZATION
# ============================================================

st.markdown(
    '<div class="section-title">Training Examples</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        '<div class="info-title">Training Images</div>',
        unsafe_allow_html=True
    )

    st.image(
        "./imgs/train_batch9242.jpg",
        use_container_width=True
    )

    st.markdown(
        '<div class="image-caption">'
        'Examples of images used during model training.'
        '</div>',
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        '<div class="info-title">Validation Predictions</div>',
        unsafe_allow_html=True
    )

    st.image(
        "./imgs/val_batch2_pred.jpg",
        use_container_width=True
    )

    st.markdown(
        '<div class="image-caption">'
        'Example predictions generated during validation.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# DETECTION SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Try the Detection Model</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Upload a brain image and let the AI analyze it for possible tumor regions.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="upload-card">',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a brain image",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, JPEG"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# IMAGE PREVIEW
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown(
        '<div class="section-title">Image Preview</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            image,
            caption="Uploaded Brain Image",
            use_container_width=True
        )

    with col2:

        st.markdown("""
        <div class="info-card">

            <div class="info-title">
                Ready for Detection
            </div>

            <div class="info-text">
                Your image has been successfully uploaded.
                Click the button below to run the YOLO
                detection model.
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        detect_button = st.button(
            "🔍 Detect Tumor"
        )

        if detect_button:

            with st.spinner("Analyzing brain image..."):

                image_array = np.array(image)

                prediction, result = model_prediction(
                    image_array
                )

            # ====================================================
            # RESULTS
            # ====================================================

            st.markdown("""
            <div class="result-card">

                <div class="section-title">
                    Detection Results
                </div>

            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    '<div class="info-title">'
                    'Original Image'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.markdown(
                    '<div class="info-title">'
                    'Detection Result'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    prediction,
                    use_container_width=True
                )

            # ====================================================
            # DETECTION SUMMARY
            # ====================================================

            boxes = result.boxes

            detection_count = len(boxes)

            if detection_count > 0:

                confidences = boxes.conf.cpu().numpy()

                max_confidence = float(
                    np.max(confidences)
                )

                avg_confidence = float(
                    np.mean(confidences)
                )

                st.success(
                    f"⚠️ {detection_count} tumor region(s) detected."
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.markdown(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                Tumors Detected
                            </div>

                            <div class="metric-value">
                                {detection_count}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:

                    st.markdown(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                Highest Confidence
                            </div>

                            <div class="metric-value">
                                {max_confidence:.1%}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col3:

                    st.markdown(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                Average Confidence
                            </div>

                            <div class="metric-value">
                                {avg_confidence:.1%}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.info(
                    "✅ No tumor region was detected in this image."
                )
