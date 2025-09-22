# disease_detector.py
# This is the unified, local version of the disease detector.
# It loads the model from the ./Model directory and runs inference directly.

import os
import json
import numpy as np
from typing import Tuple, List
from PIL import Image
import tensorflow as tf
import tempfile

# --- Configuration ---
# The path to the directory where the trained model and class names are stored.
MODEL_DIR = "./Model"
DEFAULT_IMG_SIZE = (224, 224)

# --- Model Loading ---
# Load the model and class names once when the server starts up.
# This is much more efficient than loading it on every request.
model = None
class_names = []

try:
    model_path = os.path.join(MODEL_DIR, 'model.h5')
    class_names_path = os.path.join(MODEL_DIR, 'class_names.json')

    if not (os.path.exists(model_path) and os.path.exists(class_names_path)):
        print("WARNING: Model file or class_names.json not found in ./Model directory.")
        print("The application will run, but predictions will fail.")
    else:
        model = tf.keras.models.load_model(model_path)
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        print("Successfully loaded model and class names.")

except Exception as e:
    print(f"An error occurred during model loading: {e}")
    # The application can still start, but predictions will return an error.
    model = None


def _run_prediction_logic(image_path: str) -> List[dict]:
    """
    This function contains the core prediction logic, adapted from Rudy's script.
    It takes an image file path and returns a list of prediction dictionaries.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(DEFAULT_IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # Create a batch of 1

    preds = model.predict(arr)[0]
    top_indices = preds.argsort()[-3:][::-1] # Get top 3 predictions

    results = []
    for i in top_indices:
        class_name = class_names[i]
        prob = float(preds[i])
        results.append({"class": class_name, "prob": prob})
    return results


async def get_disease_prediction(image_bytes: bytes) -> Tuple[str, float]:
    """
    The main prediction function called by the FastAPI server.
    It saves the uploaded image bytes to a temporary file and runs prediction.

    Args:
        image_bytes: The raw bytes of the image file from the upload.

    Returns:
        A tuple with the top predicted disease name and its confidence score.
    """
    if model is None or not class_names:
        print("Error: Model is not loaded. Cannot perform prediction.")
        return "Model not loaded", 0.0

    # Create a temporary file to save the image bytes
    # This is necessary because the prediction logic expects a file path.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
        temp_image.write(image_bytes)
        temp_image_path = temp_image.name

    try:
        # Run the prediction on the saved temporary file
        predictions = _run_prediction_logic(temp_image_path)

        if not predictions:
            return "No prediction found", 0.0

        # Extract the top prediction
        top_prediction = predictions[0]
        disease_name = top_prediction.get("class", "Unknown Disease")
        confidence = top_prediction.get("prob", 0.0)

        print(f"Local prediction successful: {disease_name} ({confidence:.2%})")
        return disease_name, confidence

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return "Prediction Error", 0.0
    finally:
        # IMPORTANT: Clean up the temporary file after prediction
        os.remove(temp_image_path)

