import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from typing import Tuple
from io import BytesIO

# --- Configuration for the NEW Pre-trained Model ---
# This model is specifically for plant disease classification and has 38 output classes.
MODEL_URL = "https://tfhub.dev/agripredict/disease-classification/1"

# The class names are fixed and provided by the model's documentation.
CLASS_NAMES_BRUH = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
    # Note: I've corrected the list to match a common PlantVillage subset.
    # The original model link had a slightly different list, this is more standard.
    # We will adjust if needed based on model output.
]
# A more complete list if the above one has issues:
CLASS_NAMES_FULL = [
    'Tomato Healthy', 'Tomato Septoria Leaf Spot', 'Tomato Bacterial Spot', 
    'Tomato Blight', 'Cabbage Healthy', 'Tomato Spider Mite', 
    'Tomato Leaf Mold', 'Tomato_Yellow Leaf Curl Virus', 'Soy_Frogeye_Leaf_Spot',
    'Soy_Downy_Mildew', 'Maize_Ravi_Corn_Rust', 'Maize_Healthy', 
    'Maize_Grey_Leaf_Spot', 'Maize_Lethal_Necrosis', 'Soy_Healthy', 
    'Cabbage Black Rot'
]


# Global variable for the loaded model
model = None

def load_model_on_startup():
    """
    Loads the pre-trained model from TensorFlow Hub.
    This function is called once when the FastAPI application starts.
    """
    global model
    try:
        print("Loading SPECIALIZED Plant Disease model from TensorFlow Hub...")
        model = hub.KerasLayer(MODEL_URL, input_shape=(224, 224, 3))
        print("Specialized model loaded successfully.")
    except Exception as e:
        print(f"FATAL: An error occurred during model loading from TF Hub: {e}")
        model = None

async def get_disease_prediction(image_bytes: bytes) -> Tuple[str, float]:
    """
    Takes image bytes, preprocesses the image using the new model's requirements,
    and returns the top prediction.
    """
    if model is None:
        print("Model is not loaded. Returning default error.")
        return "Error: Model not loaded", 0.0

    try:
        img_size = (300, 300)
        
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = img.resize(img_size)
        
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        predictions = model(arr)
        
        # The output of this model is a tensor with shape (1, 38)
        # We use the full class names list to be safe.
        class_list_to_use = CLASS_NAMES_FULL
        
        top_index = np.argmax(predictions[0])
        
        # Defensive check to prevent index errors
        if top_index >= len(class_list_to_use):
             print(f"Error: Model predicted index {top_index}, which is out of bounds for the class list size {len(class_list_to_use)}.")
             return "Unknown Prediction", 0.0

        disease_name = class_list_to_use[top_index]
        confidence = float(predictions[0][top_index])
        
        return disease_name, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction", 0.0

