# Brain-Tumor-MRI-Image-Classification🧠 Brain Tumor MRI Image Classification
A Streamlit web application that classifies brain MRI images into different tumor types using a deep learning model trained on medical imaging data.
The goal of this project is to assist in early tumor detection through AI-powered image analysis.

# 💡 Project Overview
This project demonstrates how deep learning can be applied in healthcare diagnostics.
By uploading a brain MRI image, the model predicts which tumor category the image belongs to.

# 🔍 Tumor Categories:
# 🧬 Glioma
# 🧠 Meningioma
# 🩺 Pituitary
# ✅ No Tumor

# ⚙️ Tech Stack
Category	Tools Used
Programming Language	Python 🐍
Frameworks	TensorFlow, Streamlit
Libraries	NumPy, Pillow (PIL), Pickle
Model Type	MobileNetV2 (Pretrained CNN)
Interface	Streamlit Web App

# 🚀 How It Works
Upload a brain MRI image (JPG/PNG).
The image is preprocessed (resized to 224×224 and normalized).
The trained MobileNetV2 deep learning model predicts the tumor type.
The app displays:
Predicted tumor category
Confidence score
Bar chart of class probabilities

# 🧠 Model Information
Model File: MobileNetV2_best.pkl
Input Size: 224×224 pixels
Framework: TensorFlow

