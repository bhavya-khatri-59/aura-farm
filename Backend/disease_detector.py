import os
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Tuple, List
from io import BytesIO

# --- Using the correct, specialized disease classification model ---
MODEL_PATH = './Model' 

# --- The correct class names for this specific model ---
CLASS_NAMES_CORRECT = [
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
    Loads the pre-trained model from the local /Model directory.
    This function is called once when the FastAPI application starts.
    """
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"FATAL: Model directory not found at {MODEL_PATH}")
            return
            
        print("Loading SPECIALIZED Plant Disease model from local directory...")
        model = tf.saved_model.load(MODEL_PATH)
        print("Specialized model loaded successfully.")
    except Exception as e:
        print(f"FATAL: An error occurred during model loading: {e}")
        model = None

async def get_disease_prediction(image_bytes: bytes) -> Tuple[str, float]:
    """
    Takes image bytes, preprocesses the image, and returns the top prediction.
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
        
        tensor = tf.convert_to_tensor(arr, dtype=tf.float32)

        # The loaded model is a callable object that returns a dictionary
        # The key to the output tensor is often 'outputs' or similar
        serving_fn = model.signatures['serving_default']
        predictions = serving_fn(tensor)
        
        # Find the key for the output tensor in the predictions dictionary
        output_key = list(predictions.keys())[0]
        raw_predictions = predictions[output_key][0].numpy() # Use .numpy() to get the array

        class_list_to_use = CLASS_NAMES_CORRECT
        top_index = np.argmax(raw_predictions)
        
        if top_index >= len(class_list_to_use):
             print(f"Error: Model predicted index {top_index}, which is out of bounds.")
             return "Model/Class Mismatch", 0.0

        disease_name = class_list_to_use[top_index]
        confidence = float(raw_predictions[top_index])
        
        return disease_name, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction", 0.0

