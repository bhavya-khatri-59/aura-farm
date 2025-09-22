# llm_handler.py
# This module is responsible for all interactions with the Gemini LLM.

import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Dict, List, Optional
import math
import json

# --- Load environment variables from .env file ---
load_dotenv()

# --- Configuration ---
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set or found in .env file.")
    genai.configure(api_key=api_key)
    # Using gemini-1.5-flash as it is a fast and capable model suitable for this task
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    model = None

MAX_CONTEXT_TOKENS = 12000

def _estimate_tokens(text: str) -> int:
    # A simple heuristic for token estimation
    return math.ceil(len(text.split()) * 1.5)

def _truncate_history_by_tokens(history: List[Dict]) -> List[Dict]:
    if not history:
        return []
    
    current_token_count = 0
    truncated_conversation = []

    # Iterate backwards from the most recent message
    for message in reversed(history):
        # Ensure message content is a string
        message_text = message.get('parts', [''])[0]
        if not isinstance(message_text, str):
            message_text = str(message_text)
        
        message_tokens = _estimate_tokens(message_text)
        
        if current_token_count + message_tokens > MAX_CONTEXT_TOKENS:
            # Stop if adding the next message would exceed the token limit
            break
        
        truncated_conversation.insert(0, message)
        current_token_count += message_tokens
    
    return truncated_conversation

async def get_conversational_response(
    user_prompt: str,
    language: str,
    history: List[Dict],
    diagnosis: Optional[Dict] = None,
    weather: Optional[Dict] = None,
    nearby_farmers: Optional[List[Dict]] = None
) -> str:
    """
    Constructs a prompt and gets a response from Gemini, now with nearby farmer context.
    """
    if not model:
        return "Error: The AI model is not configured. Please check the API key."

    current_prompt = user_prompt
    
    # On the first turn, construct the detailed system prompt with all context
    if not history:
        system_instruction = f"""
        You are 'AuraFarm', a friendly and expert agricultural assistant chatbot for farmers.
        Your goal is to provide helpful, clear, and encouraging advice.
        - Analyze all provided context: the user's diagnosis, weather, remedies, and reports from nearby farmers.
        - If you see reports of the same disease nearby, mention that this could be a local outbreak and advise extra caution.
        - Address the farmer's specific question directly and conversationally.
        - Provide additional tips or best practices related to the diagnosis and treatment.
        - IMPORTANT: Use simple, non-technical language suitable for farmers.
        - Keep responses concise, ideally under 300 words.
        - Always respond in a friendly and supportive tone.
        - IMPORTANT: Respond entirely and only in the {language} language.
        """

        farmer_reports_str = "No data available."
        if nearby_farmers:
            farmer_reports_str = json.dumps(nearby_farmers, indent=2)

        prompt_context = f"""
        Here is the information I have gathered for the farmer:

        1.  **Farmer's Question:** "{user_prompt}"

        2.  **AI Diagnosis Result for their Plant:**
            -   Detected Condition: {diagnosis.get('name', 'N/A')}
            -   Confidence Score: {diagnosis.get('confidence', 'N/A')}

        3.  **Suggested Treatments:**
            -   Description: {diagnosis.get('remedies', {}).get('description', 'N/A')}
            -   Organic Options: {', '.join(diagnosis.get('remedies', {}).get('organic_treatment', []))}
            -   Chemical Options: {', '.join(diagnosis.get('remedies', {}).get('chemical_treatment', []))}

        4.  **Current Weather at the Farm:**
            -   Conditions: {weather.get('description', 'N/A')}
            -   Temperature: {weather.get('temp_c', 'N/A')}°C
            -   Humidity: {weather.get('humidity', 'N/A')}%

        5.  **Recent Disease Reports from Nearby Farms:**
            ```json
            {farmer_reports_str}
            ```

        Based on ALL of this information, please provide a comprehensive, helpful, and conversational response to the farmer.
        """
        
        # --- FIX APPLIED HERE ---
        # The 'system' role is not supported in the chat history.
        # Instead, we combine the system instruction and the first user prompt
        # into a single, comprehensive first message from the user.
        current_prompt = f"{system_instruction}\n\n{prompt_context}"

    try:
        # The history passed to start_chat will be empty on the first turn.
        # On subsequent turns, it will contain the valid user/model conversation.
        safe_history = _truncate_history_by_tokens(history)
        chat = model.start_chat(history=safe_history)
        response = await chat.send_message_async(current_prompt)
        return response.text
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"Sorry, I encountered an error while trying to generate a response. Please try again. (Error: {e})"

