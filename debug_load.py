
import pickle
import os
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH_PKL = r"E:/Guvi/Raja_Project6/Brain Tumor MRI Image Classification/models_outputs/MobileNetV2_best.pkl"

def preprocess_image(pil_img: Image.Image, target_hw=(224, 224)) -> np.ndarray:
    #PIL -> float32 [1,H,W,C] normalized to [0,1].
    arr = np.array(pil_img.convert("RGB"))
    arr = tf.image.resize(arr, target_hw)  # Tensor -> keeps dtype float32 later
    arr = tf.cast(arr, tf.float32) / 255.0
    arr = tf.expand_dims(arr, 0)  # [1,H,W,C]
    return arr.numpy()

print(f"Checking PKL path: {MODEL_PATH_PKL}")
if os.path.exists(MODEL_PATH_PKL):
    print("PKL file exists.")
    try:
        # 1. Load Model
        print("Loading model via pickle...")
        with open(MODEL_PATH_PKL, "rb") as f:
            model = pickle.load(f)
        print("Successfully loaded PKL model.")

        # 2. Mock Image
        print("Creating mock image...")
        dummy_img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        
        # 3. Preprocess
        print("Preprocessing image...")
        img_batch = preprocess_image(dummy_img, (224, 224))
        print(f"Processed batch shape: {img_batch.shape}, dtype: {img_batch.dtype}")

        # 4. Predict
        print("Predicting...")
        preds = model.predict(img_batch)
        print("Prediction successful.")
        print(f"Predictions: {preds}")

    except Exception as e:
        print(f"FAILED during simulation: {e}")
        import traceback
        traceback.print_exc()
else:
    print("PKL file does not exist.")
