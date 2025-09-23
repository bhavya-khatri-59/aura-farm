from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import json
from typing import List, Optional

# Import your handler functions
# NOTE: We no longer import or call load_model_on_startup here
from disease_detector import get_disease_prediction 
from llm_handler import get_conversational_response
from weather_service import get_weather_data as get_weather
from farmer_network_service import get_nearby_farmer_data

# --- FastAPI App Initialization ---
app = FastAPI(title="AuraFarm AI Backend")

# --- REMOVED STARTUP EVENT ---
# The model will now be loaded on the first request inside disease_detector.py

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
    image: Optional[UploadFile] = File(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    history: str = Form("[]")
):
    try:
        conversation_history = json.loads(history)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid history format. Must be a valid JSON string.")

    if image:
        # --- Diagnosis Mode ---
        if lat is None or lon is None:
            raise HTTPException(status_code=400, detail="Latitude and longitude are required when uploading an image.")

        image_bytes = await image.read()
        # The model will be loaded here on the first call
        disease_name, confidence = await get_disease_prediction(image_bytes)
        
        weather_data = await get_weather(lat, lon)
        remedy_data = remedies_db.get(disease_name, {})
        farmer_data = get_nearby_farmer_data(lat, lon, disease_name)
        
        diagnosis_result = {
            "name": disease_name,
            "confidence": f"{confidence * 100:.2f}%",
            "remedies": remedy_data
        }

        final_response = await get_conversational_response(
            user_prompt=prompt,
            history=conversation_history,
            diagnosis=diagnosis_result,
            weather=weather_data,
            nearby_farmers=farmer_data
        )
    else:
        # --- Conversational Mode ---
        final_response = await get_conversational_response(
            user_prompt=prompt,
            history=conversation_history
        )

    return JSONResponse(content={"response": final_response})

