
# 🧠 Brain Tumor Classification — Streamlit (light enhanced)
import os
import json
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image,ImageOps
from keras.models import load_model


st.markdown(
    """
    <style>
    .stApp {
        background-color: #D8BFD8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Brain Tumor Classifier")


st.header("Upload a brain MRI image to classify tumor type")

# -----------------------------
# 🔧 CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

# Use a raw string or forward slashes to avoid Windows backslash issues
MODEL_PATH = r"E:/Guvi/Raja_Project6/Brain Tumor MRI Image Classification/models_outputs/MobileNetV2_best.pkl"



IMG_SIZE = (224, 224)
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

st.title("🧠 Brain Tumor Classification")
st.write("Upload a **brain MRI image** below to classify the tumor type.")

# -----------------------------
# 🧩 LOAD TRAINED MODEL
# -----------------------------


@st.cache_resource(show_spinner=True)
def load_saved_model(path: str):
    with open(path, "rb") as f:
        m = pickle.load(f)
    return m


# Nice status in sidebar
with st.sidebar:
    st.header("Model")
    st.caption("Make sure the path is correct and file exists.")
    st.code(MODEL_PATH, language="text")

try:
    assert os.path.exists(MODEL_PATH), f"Model not found at:\n{MODEL_PATH}"
    model = load_saved_model(MODEL_PATH)
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    import traceback
    st.sidebar.error(f"❌ Model load failed:\n{e}\n\nTraceback:\n{traceback.format_exc()}")
    st.stop()

# -----------------------------
# 🔧 HELPERS
# -----------------------------
def preprocess_image(pil_img: Image.Image, target_hw=(224, 224)) -> np.ndarray:
    #PIL -> float32 [1,H,W,C] normalized to [0,1].
    arr = np.array(pil_img.convert("RGB"))
    arr = tf.image.resize(arr, target_hw)  # Tensor -> keeps dtype float32 later
    arr = tf.cast(arr, tf.float32) / 255.0
    arr = tf.expand_dims(arr, 0)  # [1,H,W,C]
    return arr.numpy()

def safe_softmax(vec: np.ndarray) -> np.ndarray:
    x = vec - np.max(vec, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

# -----------------------------
# 📤 IMAGE UPLOAD SECTION
# -----------------------------
uploaded_file = st.file_uploader("Upload an MRI Image (jpg/png/jpeg)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Safely open image
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not read the image file: {e}")
        st.stop()

    st.image(image, caption="🩺 Uploaded MRI Image", use_container_width=True)


    # -----------------------------
    # 🔮 PREDICT
    # -----------------------------
    # Preprocess
    img_batch = preprocess_image(image, IMG_SIZE)

    # Predict
    try:
        preds = model.predict(img_batch)
        preds = np.array(preds)
        # Handle logits -> convert to probabilities
        if preds.ndim == 2 and (preds.min() < 0 or preds.max() > 1):
            preds = safe_softmax(preds)
    except Exception as e:
        import traceback
        st.error(f"Prediction failed: {e}\n\nTraceback:\n{traceback.format_exc()}")
        st.stop()

    # Post-process
    if preds.ndim == 1:
        probs = preds
    else:
        probs = preds[0]

    pred_class = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_class] if len(CLASS_NAMES) == len(probs) else f"class_{pred_class}"
    confidence = float(probs[pred_class])

    # -----------------------------
    # 🎯 DISPLAY PREDICTION RESULTS
    # -----------------------------
    st.subheader("🧾 Prediction Results")
    st.success(f"**Predicted Tumor Type:** {pred_label}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    # Per-class bar chart
    st.markdown("### 🔍 Model Confidence per Class")
    conf_data = { (CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}") : float(probs[i]) for i in range(len(probs)) }
    st.bar_chart(conf_data)

    # Download result as JSON (handy)
    result = {
        "predicted_label": pred_label,
        "confidence": confidence,
        "per_class": conf_data,
    }
    st.download_button("⬇️ Download result (JSON)", data=json.dumps(result, indent=2),
                       file_name="prediction_result.json", mime="application/json")

else:
    st.info("👆 Please upload a brain MRI image to begin prediction.")

# -----------------------------
# 📚 ABOUT SECTION
# -----------------------------
with st.expander("ℹ️ About this App"):
    st.write("""
This application uses a **deep learning model** trained on brain MRI images
to classify the tumor type into one of four categories:
- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor
""")