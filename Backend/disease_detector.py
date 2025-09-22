# disease_detector_mock.py
# This is a MOCK file for testing the backend and frontend.
# It simulates the behavior of the real disease detector by returning a
# random disease and confidence score.

import random
from typing import Tuple
import time

# A list of possible diseases the mock detector can "predict".
MOCK_DISEASES = [
    'Tomato___Late_blight',
    'Tomato___healthy',
    'Potato___Early_blight',
    'Pepper__bell___Bacterial_spot',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Pepper__bell___healthy'
]


def get_disease_prediction(image_bytes: bytes) -> Tuple[str, float]:
    """
    Simulates a disease prediction. This function has the same signature as the
    real one but does not use the image_bytes.

    Args:
        image_bytes: The image of the plant leaf in bytes (not used in mock).

    Returns:
        A tuple containing a randomly selected disease name and a random confidence score.
    """
    # Simulate a delay, just like a real model would have
    time.sleep(1.5)

    # Choose a random disease from our list
    predicted_disease = random.choice(MOCK_DISEASES)

    # Generate a random confidence score between 85% and 99%
    confidence = random.uniform(0.85, 0.99)

    # Print a message to the console so you know the mock is being used
    print("---  MOCK DETECTOR ACTIVE ---")
    print(f"Mock Prediction: {predicted_disease} with {confidence:.2%} confidence")
    print("----------------------------")

    return predicted_disease, confidence
