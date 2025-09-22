from dotenv import load_dotenv
import os
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
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    model = None

MAX_CONTEXT_TOKENS = 12000

def _estimate_tokens(text: str) -> int:
    """A simple heuristic for token estimation."""
    return math.ceil(len(text.split()) * 1.5)

def _truncate_history_by_tokens(history: List[Dict]) -> List[Dict]:
    """Ensures the conversation history does not exceed the token limit."""
    if not history:
        return []
    
    current_token_count = 0
    truncated_conversation = []

    # Iterate backwards from the most recent message
    for message in reversed(history):
        # Ensure message content is a string
        message_text = ""
        parts = message.get('parts', [])
        if parts:
            first_part = parts[0]
            if isinstance(first_part, str):
                message_text = first_part
        
        message_tokens = _estimate_tokens(message_text)
        
        if current_token_count + message_tokens > MAX_CONTEXT_TOKENS:
            # Stop if adding the next message would exceed the token limit
            break
        
        truncated_conversation.insert(0, message)
        current_token_count += message_tokens
    
    return truncated_conversation

async def get_conversational_response(
    user_prompt: str,
    history: List[Dict],
    diagnosis: Optional[Dict] = None,
    weather: Optional[Dict] = None,
    nearby_farmers: Optional[List[Dict]] = None
) -> str:
    """
    Constructs a prompt and gets a response from Gemini.
    Uses a detailed prompt for diagnoses and a general prompt for conversational queries.
    """
    if not model:
        return "Error: The AI model is not configured. Please check the API key."

    # Combine the existing history with the new user prompt
    api_history = history + [{'role': 'user', 'parts': [user_prompt]}]

    # On the first turn of a conversation, prepend the appropriate system instructions.
    if not history:
        if diagnosis:
            # --- Diagnosis Mode System Prompt ---
            system_instruction = f"""
            You are 'AuraFarm', a friendly and expert agricultural assistant chatbot for farmers.
            Your goal is to provide helpful, clear, and encouraging advice.
            - Analyze all provided context: the user's diagnosis, weather, remedies, and reports from nearby farmers.
            - If you see reports of the same disease nearby, mention that this could be a local outbreak and advise extra caution.
            - Address the farmer's specific question directly and conversationally.
            - IMPORTANT: Respond entirely and only in the same language as the "Farmer's Question".
            """
            farmer_reports_str = json.dumps(nearby_farmers, indent=2) if nearby_farmers else "No data available."
            
            prompt_context = f"""
            Here is the information I have gathered for the farmer:

            1.  **Farmer's Question:** "{user_prompt}"
            2.  **AI Diagnosis Result for their Plant:** {json.dumps(diagnosis, indent=2)}
            3.  **Current Weather at the Farm:** {json.dumps(weather, indent=2)}
            4.  **Recent Disease Reports from Nearby Farms:** {farmer_reports_str}

            Based on ALL of this information, please provide a comprehensive, helpful, and conversational response.
            """
            # Prepend the instructions and context to the first user message
            api_history[0]['parts'][0] = system_instruction + "\n\n" + prompt_context
        else:
            # --- Conversational Mode System Prompt ---
            system_instruction = f"""
            You are 'AuraFarm', a friendly and expert agricultural assistant chatbot for farmers.
            Your goal is to provide helpful, clear, and encouraging advice.
            - Answer the user's questions based on your general agricultural knowledge.
            - Do not invent or hallucinate specific details like diagnoses, specific treatments, or weather conditions if they are not provided in the context.
            - If you don't know the answer, it is better to say you don't have enough information to be sure.
            - IMPORTANT: Respond entirely and only in the same language as the user's question.
            """
            # Prepend the instructions to the user's first general question
            api_history[0]['parts'][0] = system_instruction + "\n\n" + user_prompt

    try:
        # Truncate the history to ensure it fits within the context window
        safe_history = _truncate_history_by_tokens(api_history)
        
        # The history for the chat session should contain all but the very last message
        chat_session_history = safe_history[:-1]
        # The last message is the one we are sending to get a response
        last_message = safe_history[-1]['parts']

        chat = model.start_chat(history=chat_session_history)
        response = await chat.send_message_async(last_message)
        
        return response.text
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"Sorry, I encountered an error while trying to generate a response. Please try again. (Error: {e})"

