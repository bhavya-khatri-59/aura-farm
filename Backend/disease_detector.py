import os
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Tuple, List
from io import BytesIO
import asyncio

# --- FIX #1: ROBUST FILE PATHING ---
# Get the absolute path to the directory where this script resides
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct an absolute path to the Model directory, which is much more reliable on a server
MODEL_PATH = os.path.join(_BASE_DIR, 'Model')

# The correct class names for the specialized model
CLASS_NAMES_CORRECT = [
    'Tomato Healthy', 'Tomato Septoria Leaf Spot', 'Tomato Bacterial Spot', 
    'Tomato Blight', 'Cabbage Healthy', 'Tomato Spider Mite', 
    'Tomato Leaf Mold', 'Tomato_Yellow Leaf Curl Virus', 'Soy_Frogeye_Leaf_Spot',
    'Soy_Downy_Mildew', 'Maize_Ravi_Corn_Rust', 'Maize_Healthy', 
    'Maize_Grey_Leaf_Spot', 'Maize_Lethal_Necrosis', 'Soy_Healthy', 
    'Cabbage Black Rot'
]

# --- LAZY LOADING LOGIC ---
model = None
# A lock to prevent multiple requests from trying to load the model simultaneously
model_load_lock = asyncio.Lock()

def _load_model_sync():
    """Synchronous function to perform the actual, slow model loading."""
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"FATAL: Model directory not found at the absolute path: {MODEL_PATH}")
            print(f"(Current working directory is: {os.getcwd()})")
            return None
            
        print(f"Loading SPECIALIZED Plant Disease model from: {MODEL_PATH}")
        loaded_model = tf.saved_model.load(MODEL_PATH)
        print("Specialized model loaded successfully.")
        return loaded_model
    except Exception as e:
        print(f"FATAL: An error occurred during model loading: {e}")
        return None

async def get_disease_prediction(image_bytes: bytes) -> Tuple[str, float]:
    """
    Takes image bytes, ensures the model is loaded (lazily), and returns the top prediction.
    """
    global model
    
    if model is None:
        async with model_load_lock:
            # Check again inside the lock in case another request loaded it while this one was waiting.
            if model is None:
                print("Model is not loaded yet. Attempting to load...")
                loop = asyncio.get_event_loop()
                # Run the slow, blocking I/O operation in a separate thread
                model = await loop.run_in_executor(None, _load_model_sync)

    # --- FIX #2: BETTER ERROR HANDLING ---
    # If loading failed, raise a proper exception instead of returning a generic string.
    # This provides a clearer error in the server logs.
    if model is None:
        raise RuntimeError("Model could not be loaded. Please check the server logs for the FATAL error message.")

    try:
        img_size = (300, 300)
        
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = img.resize(img_size)
        
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        tensor = tf.convert_to_tensor(arr, dtype=tf.float32)

        serving_fn = model.signatures['serving_default']
        predictions = serving_fn(tensor)
        
        output_key = list(predictions.keys())[0]
        raw_predictions = predictions[output_key][0].numpy()

        class_list_to_use = CLASS_NAMES_CORRECT
        top_index = np.argmax(raw_predictions)
        
        if top_index >= len(class_list_to_use):
             print(f"Error: Model predicted index {top_index}, which is out of bounds for the class list size of {len(class_list_to_use)}.")
             return "Model/Class Mismatch", 0.0

        disease_name = class_list_to_use[top_index]
        confidence = float(raw_predictions[top_index])
        
        return disease_name, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Re-raise the exception to be caught by the main error handler
        raise e

