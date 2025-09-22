# main.py
# This is the primary file for your FastAPI application.
# It defines the API endpoints and orchestrates the calls to the other modules.

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import Dict, Optional, List

# Import the handler functions from other modules
from disease_detector import get_disease_prediction
from llm_handler import get_conversational_response
from weather_service import get_weather_data
# NEW: Import the farmer network service
from farmer_network_service import get_nearby_farmer_data

# Initialize the FastAPI app
app = FastAPI(
    title="Plant Disease Diagnosis API",
    description="API to diagnose plant diseases, get remedies, and provide conversational support.",
    version="1.0.0"
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Remedies Data ---
try:
    with open("remedies.json", "r") as f:
        remedies_data = json.load(f)
except FileNotFoundError:
    print("ERROR: remedies.json not found. Please create it.")
    remedies_data = {}


# --- API Endpoints ---

@app.get("/", summary="Root endpoint to check server status")
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "ok", "message": "Welcome to the Plant Disease Diagnosis API!"}


@app.post("/diagnose", summary="Diagnose a plant disease from an image and prompt")
async def diagnose_plant(
    image: UploadFile = File(..., description="Image of the plant leaf."),
    prompt: str = Form(..., description="User's text query or question."),
    lat: float = Form(..., description="Latitude of the user's location."),
    lon: float = Form(..., description="Longitude of the user's location."),
    language: str = Form("English", description="Target language for the response."),
    history: Optional[str] = Form("[]", description="A JSON string representing the conversation history.")
) -> Dict[str, str]:
    """
    This endpoint now supports conversational history and adds context from nearby farmers.
    """
    try:
        try:
            conversation_history = json.loads(history)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid history format. Must be a valid JSON array.")

        final_response = ""
        # Only run full diagnosis on the first turn of the conversation.
        if not conversation_history:
            # 1. Read image content
            image_bytes = await image.read()
            # 2. Get disease prediction from the local ML model
            disease_name, confidence = get_disease_prediction(image_bytes)
            # 3. Get remedies for the predicted disease
            remedy_info = remedies_data.get(disease_name, {
                "description": "No specific remedy information found for this condition.",
                "organic_treatment": [],
                "chemical_treatment": []
            })
            diagnosis_context = {
                "name": disease_name,
                "confidence": f"{confidence:.2%}",
                "remedies": remedy_info
            }
            # 4. Fetch weather data for context
            weather_context = await get_weather_data(lat, lon)
            if not weather_context:
                weather_context = {"error": "Could not fetch weather data."}

            # 5. NEW: Get simulated data from nearby farmers
            nearby_farmers_context = get_nearby_farmer_data(lat, lon, disease_name)
            
            # 6. Get the INITIAL response from the LLM using all context
            final_response = await get_conversational_response(
                user_prompt=prompt,
                language=language,
                history=conversation_history,
                diagnosis=diagnosis_context,
                weather=weather_context,
                nearby_farmers=nearby_farmers_context # Pass new context
            )
        else:
            # This is a follow-up question. No need to re-diagnose.
            final_response = await get_conversational_response(
                user_prompt=prompt,
                language=language,
                history=conversation_history
            )

        return {"response": final_response}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

