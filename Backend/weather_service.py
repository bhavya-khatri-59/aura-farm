# weather_service.py
# This module fetches weather data from an external API.

import os
import httpx
from typing import Dict, Optional

# --- Configuration ---
# Use an environment variable for the Weather API key.
# I'm using WeatherAPI.com here as an example. You can use OpenWeatherMap or others.
# Set this in your terminal: export WEATHER_API_KEY='Your_API_Key'
API_KEY = os.environ.get("WEATHER_API_KEY")
BASE_URL = "http://api.weatherapi.com/v1/current.json"

async def get_weather_data(lat: float, lon: float) -> Optional[Dict]:
    """
    Fetches current weather data for a given latitude and longitude.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        A dictionary with weather info or None if an error occurs.
    """
    if not API_KEY:
        print("Warning: WEATHER_API_KEY environment variable not set. Skipping weather fetch.")
        return None

    query_param = f"{lat},{lon}"
    params = {"key": API_KEY, "q": query_param}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(BASE_URL, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            data = response.json()
            
            # Extract only the necessary information
            current = data.get("current", {})
            return {
                "temp_c": current.get("temp_c"),
                "humidity": current.get("humidity"),
                "description": current.get("condition", {}).get("text"),
                "wind_kph": current.get("wind_kph"),
            }
    except httpx.RequestError as e:
        print(f"An error occurred while requesting weather data: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during weather fetch: {e}")
        return None
