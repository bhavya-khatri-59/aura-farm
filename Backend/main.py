from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import json
from typing import List, Optional

# Import your handler functions
from disease_detector import load_model_on_startup, get_disease_prediction
from llm_handler import get_conversational_response
from weather_service import get_weather
from farmer_network_service import get_nearby_farmer_data

# --- FastAPI App Initialization ---
app = FastAPI(title="AuraFarm AI Backend")

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    print("Server starting up...")
    load_model_on_startup()
    print("Startup complete. Model is ready.")

# --- Helper Function to Load Remedies ---
def load_remedies():
    try:
        with open("remedies.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("ERROR: remedies.json not found. Please create it.")
        return {}

remedies_db = load_remedies()

# --- API Endpoint ---
@app.post("/diagnose")
async def diagnose_plant(
    prompt: str = Form(...),
    # Image and location are now optional
    image: Optional[UploadFile] = File(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    history: str = Form("[]") # Receive history as a JSON string
):
    """
    This is the main endpoint for the AuraFarm app.
    It can function in two modes:
    1. Diagnosis Mode: If an image, lat, and lon are provided, it performs a full diagnosis.
    2. Conversational Mode: If no image is provided, it acts as a general chatbot.
    """
    try:
        # 1. Parse the conversation history from the JSON string
        try:
            conversation_history = json.loads(history)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid history format. Must be a valid JSON string.")

        # 2. Determine the mode based on whether an image is provided
        if image:
            # --- Diagnosis Mode ---
            if lat is None or lon is None:
                raise HTTPException(status_code=400, detail="Latitude and longitude are required when uploading an image.")

            # 2a. Read image bytes
            image_bytes = await image.read()
            
            # 2b. Get ML Model Prediction
            disease_name, confidence = await get_disease_prediction(image_bytes)

            # 2c. Get contextual data
            weather_data = get_weather(lat, lon)
            remedy_data = remedies_db.get(disease_name, {})
            farmer_data = get_nearby_farmer_data(lat, lon)
            
            diagnosis_result = {
                "name": disease_name,
                "confidence": f"{confidence * 100:.2f}%",
                "remedies": remedy_data
            }

            # 2d. Call the LLM with all the context
            final_response = await get_conversational_response(
                user_prompt=prompt,
                history=conversation_history,
                diagnosis=diagnosis_result,
                weather=weather_data,
                nearby_farmers=farmer_data
            )
        else:
            # --- Conversational Mode ---
            # No diagnosis needed. Call the LLM with only the prompt and history.
            final_response = await get_conversational_response(
                user_prompt=prompt,
                history=conversation_history
            )

        # 3. Return the final response
        return JSONResponse(content={"response": final_response})

    except Exception as e:
        print(f"An error occurred in the /diagnose endpoint: {e}")
        # In case of an unexpected error, return a generic error message
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

