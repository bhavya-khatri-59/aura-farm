# disease_detector.py
# This module is responsible for loading the local TensorFlow model
# and running predictions on an image.

import os
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import json
from typing import Tuple

# --- Configuration ---
# Set the path to your model and class names file
MODEL_DIR = "Model"
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

# --- Load Model and Class Names ---
try:
    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load the class names from the JSON file
    with open(CLASS_NAMES_PATH, 'r') as f:
        CLASS_NAMES = json.load(f)

except (IOError, FileNotFoundError) as e:
    print(f"Error loading model or class names: {e}")
    print("Please ensure 'plant_disease_model.h5' and 'class_names.json' are in the 'Model' directory.")
    model = None
    CLASS_NAMES = []


def get_disease_prediction(image_bytes: bytes) -> Tuple[str, float]:
    """
    Takes image bytes, preprocesses the image, and predicts the disease.

    Args:
        image_bytes: The image of the plant leaf in bytes.

    Returns:
        A tuple containing the predicted disease name and the confidence score.
    """
    if model is None or not CLASS_NAMES:
        return "Error: Model not loaded", 0.0

    try:
        # Open the image from bytes
        img = Image.open(io.BytesIO(image_bytes))

        # Preprocess the image to match the model's input requirements
        # (e.g., resize, convert to array, normalize)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Create a batch

        # Make prediction
        predictions = model.predict(img_array)
        
        # Get the class with the highest probability
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        disease_name = CLASS_NAMES[predicted_class_index]
        return disease_name, confidence

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return "Error during prediction", 0.0

