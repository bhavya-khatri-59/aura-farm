import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Tuple, List

# Define global variables for the model and class names
# They will be loaded by the startup function
model = None
class_names = []

def load_model_on_startup():
    """
    Loads the trained model and class names from the /Model directory.
    This function is called once when the FastAPI application starts.
    """
    global model, class_names
    
    model_dir = './Model'
    model_path = os.path.join(model_dir, 'model.h5')
    class_names_path = os.path.join(model_dir, 'class_names.json')

    if not os.path.exists(model_path) or not os.path.exists(class_names_path):
        print(f"ERROR: Model files not found in {model_dir}. Please ensure model.h5 and class_names.json are present.")
        # In a real app, you might raise an exception here
        return

    try:
        model = tf.keras.models.load_model(model_path)
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        print("Successfully loaded model and class names.")
    except Exception as e:
        print(f"An error occurred during model loading: {e}")
        # Handle the error appropriately
        model = None
        class_names = []


async def get_disease_prediction(image_bytes: bytes) -> Tuple[str, float]:
    """
    Takes image bytes, preprocesses the image, and returns the top prediction.
    """
    if model is None or not class_names:
        print("Model is not loaded. Returning default error.")
        return "Error: Model not loaded", 0.0

    try:
        from io import BytesIO
        img_size = (224, 224)
        
        # Open image from bytes
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = img.resize(img_size)
        
        # Preprocess the image
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)  # Add batch dimension

        # Make prediction
        preds = model.predict(arr)[0]
        top_index = np.argmax(preds)
        
        disease_name = class_names[top_index]
        confidence = float(preds[top_index])
        
        return disease_name, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction", 0.0

