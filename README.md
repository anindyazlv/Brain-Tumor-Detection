# Brain Tumor Detection

A simple web-based Brain Tumor Detection application built with Streamlit and YOLO26. The application allows users to upload a brain image and automatically detect tumor regions using a trained YOLO-based object detection model.

> ⚠️ Disclaimer: This application is intended for research and educational purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical diagnosis. Always consult a qualified medical professional for medical evaluation, diagnosis, and treatment decisions.

---
## Live Demo

Try the deployed application here:

[Open Brain Tumor Detection](https://yolo26-brain-tumor-detection.streamlit.app/)

---


## Features

- Brain tumor region detection using YOLO
- Upload brain images in JPG, JPEG, or PNG format
- Visualize detected tumor regions with bounding boxes
- Display the number of detected tumor regions
- Display the highest detection confidence
- Interactive web interface using Streamlit
- Cached model loading for faster application performance

---

## Repository Structure

```text
brain-tumor-detection/
│
├── imgs/
│   ├── background.png
│   ├── test_images1.jpg
│   ├── test_images2.jpg
│   ├── train_batch9242.jpg
│   └── val_batch2_pred.jpg
│
├── models/
│   └── best.pt
│
├── main.py
├── utils.py
├── packages.txt
├── requirements.txt
├── runtime.txt
└── README.md
```

### File and Folder Description

| File / Folder | Description |
|---|---|
| `imgs/` | Contains images used by the Streamlit interface, including the background, test images, and training examples. |
| `models/` | Contains the trained YOLO model. |
| `models/best.pt` | Trained YOLO model used for brain tumor detection. |
| `main.py` | Main Streamlit application containing the user interface and detection pipeline. |
| `utils.py` | Utility functions used by the application, including the background configuration. |
| `packages.txt` | System-level packages required by the deployment environment. |
| `requirements.txt` | Python dependencies required to run the application. |
| `runtime.txt` | Specifies the Python runtime version for deployment. |
| `README.md` | Documentation for the project. |

---
## Dataset

The model was trained using a brain tumor object detection dataset obtained from Roboflow.
**Dataset source:**  
[Brain Tumor Dataset — Roboflow Universe](https://universe.roboflow.com/gliomatumor/tangcuong)

The dataset contains brain images annotated with bounding boxes for different tumor categories and non-tumor images.

### Dataset Classes
The model was trained to detect four classes:

| Class | Description |
|---|---|
| `glioma_tumor` | Brain images containing glioma tumor regions |
| `meningioma_tumor` | Brain images containing meningioma tumor regions |
| `no_tumor` | Brain images without a detected tumor |
| `pituitary_tumor` | Brain images containing pituitary tumor regions |

The dataset therefore supports both tumor type detection and no-tumor identification.

The dataset was divided into three subsets for model development and evaluation:

| Dataset Split | Percentage | 
|---:|---|
| Training | 70% | 
| Validation | 20% | 
| Testing | 10% | 

## Model

The application uses a trained YOLO26 model for detecting brain tumor regions.
The model information displayed in the application includes:

| Parameter | Value |
|---|---:|
| Architecture | YOLO26n |
| Params | 2,375,616 |
| Training Images | 3,000+ |
| Training Epochs | 50 |

The trained model is stored at:

```text
models/best.pt
```

During application startup, the model is loaded once using Streamlit's `@st.cache_resource` decorator. This prevents the model from being unnecessarily reloaded during Streamlit interactions.

---
## Model Performance

**Overall Performance**

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|---:|---:|
| All | 453 | 460 | 0.820 | 0.820 | 0.866 | 0.577 |

**Class-wise Performance**

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|---:|---:|
| `glioma_tumor` | 140 | 146 | 0.723 | 0.603 | 0.680 | 0.338 |
| `meningioma_tumor` | 136 | 137 | 0.904 | 0.891 | 0.944 | 0.641 |
| `no_tumor` | 102 | 102 | 0.889 | 0.961 | 0.980 | 0.796 |
| `pituitary_tumor` | 75 | 75 | 0.765 | 0.827 | 0.860 | 0.533 |

## How It Works

The detection pipeline consists of the following steps:

```text
Upload Brain Image
        ↓
Convert Image to RGB
        ↓
Convert Image to NumPy Array
        ↓
YOLO Model Prediction
        ↓
Detect Tumor Regions
        ↓
Generate Annotated Image
        ↓
Display Detection Results
        ↓
Show Detection Count
and Highest Confidence
```
---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anindyazlv/brain-tumor-detection.git
cd brain-tumor-detection
```


### 2. Create a Virtual Environment

It is recommended to use a virtual environment.

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit application using:

```bash
streamlit run main.py
```

After starting the application, Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open the URL in your web browser to access the application.

---

## Deployment

This application can be deployed to a Streamlit-compatible hosting environment.

Before deployment, make sure the repository contains all required files:

```text
imgs/
models/
main.py
utils.py
packages.txt
requirements.txt
runtime.txt
```

In particular, the trained model must be available at:

```text
./models/best.pt
```

The application references the model using:

```python
MODEL_PATH = "./models/best.pt"
```

Therefore, changing the location of the model requires updating `MODEL_PATH` in `main.py`.

---


## Using the Application

### Step 1 — Upload an Image

Click Upload Brain Image and select a brain image in one of the supported formats:

- `.jpg`
- `.jpeg`
- `.png`

### Step 2 — Run Detection

After uploading the image, click:

**Detect Tumor 🔍**

The trained YOLO model will process the image.

### Step 3 — View Results

The application displays the original image alongside the detection result.

If tumor regions are detected, the application shows:

- Tumors Detected — Number of detected tumor regions
- Highest Confidence — Highest confidence score among the detected regions

For example:

```text
Tumors Detected       2
Highest Confidence    94.5%
```

If no tumor region is detected, the application displays:

```text
No tumor region was detected.
```

---
