# farmer_network_service.py
# This module simulates fetching data from a network of nearby farmers.
# For a hackathon, this data is hardcoded for speed and simplicity.

from typing import Dict, List
import random
from datetime import date, timedelta

# A pool of simulated disease reports to draw from.
SIMULATED_DISEASE_REPORTS = [
    {"farmer_id": "F728", "distance_km": 2.5, "diagnosis": "Tomato___Late_blight"},
    {"farmer_id": "F319", "distance_km": 4.1, "diagnosis": "Tomato___Septoria_leaf_spot"},
    {"farmer_id": "F882", "distance_km": 1.5, "diagnosis": "Tomato___Late_blight"},
    {"farmer_id": "F501", "distance_km": 5.0, "diagnosis": "Pepper__bell___Bacterial_spot"},
    {"farmer_id": "F112", "distance_km": 3.8, "diagnosis": "Potato___Early_blight"},
    {"farmer_id": "F431", "distance_km": 6.2, "diagnosis": "Tomato___Tomato_Yellow_Leaf_Curl_Virus"},
    {"farmer_id": "F609", "distance_km": 2.1, "diagnosis": "Potato___Late_blight"},
    {"farmer_id": "F218", "distance_km": 0.8, "diagnosis": "Tomato___Late_blight"},
]

def get_nearby_farmer_data(lat: float, lon: float, current_diagnosis: str) -> List[Dict]:
    """
    Simulates fetching recent diagnoses from nearby farmers.

    In a real app, this would query a database based on lat/lon.
    For the hackathon, we return a randomized subset of our hardcoded data
    and make it seem more relevant by ensuring the user's diagnosis is present.

    Args:
        lat: User's latitude (ignored in this simulation).
        lon: User's longitude (ignored in this simulation).
        current_diagnosis: The diagnosis for the current user's plant.

    Returns:
        A list of dictionaries, each representing a nearby farmer's report.
    """
    today = date.today()
    
    # Make a copy of the reports to avoid modifying the original list
    reports = [report.copy() for report in SIMULATED_DISEASE_REPORTS]

    # Dynamically make the simulation more relevant
    # Ensure at least two other "nearby" farmers have the same issue
    relevant_reports_needed = 2
    for report in reports:
        if report["diagnosis"] == current_diagnosis:
            relevant_reports_needed -= 1

    # Add new dynamic reports if needed to simulate an outbreak
    for _ in range(max(0, relevant_reports_needed)):
        reports.append({
            "farmer_id": f"F{random.randint(900, 999)}",
            "distance_km": round(random.uniform(1.0, 5.0), 1),
            "diagnosis": current_diagnosis
        })

    # Select a random subset of 3-5 reports to show the user
    num_reports_to_show = random.randint(3, 5)
    selected_reports = random.sample(reports, min(len(reports), num_reports_to_show))

    # Add a recent report date to each selected report
    for report in selected_reports:
        report["reported_on"] = (today - timedelta(days=random.randint(0, 7))).isoformat()

    # Sort by distance for realism
    selected_reports.sort(key=lambda x: x["distance_km"])

    return selected_reports
