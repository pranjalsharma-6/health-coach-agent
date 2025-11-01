# tools.py - Core Functions for Data and Persistence (POLISHED & STABILIZED)

import os
import random
import json
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta # <-- ADDED timedelta
import pandas as pd
from bson import json_util 
import logging 

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)

# Load environment variables (secrets and URI)
load_dotenv()

# --- 1. Database Connection & Helper ---

_db_client: Optional[MongoClient] = None

def get_mongo_client() -> MongoClient:
    """Initializes and returns the MongoDB client, connecting once, with error handling."""
    global _db_client
    if _db_client is None:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            logger.error("MONGODB_URI not found in environment variables. Cannot connect to database.")
            raise ValueError("MONGODB_URI not found in environment variables.")
        
        try:
            _db_client = MongoClient(uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000) 
            _db_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB!")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB. Check URI, network access, and credentials. Error: {e}")
            _db_client = None
            raise ConnectionError("MongoDB connection failed.")

    return _db_client

def get_collection(collection_name: str):
    """Returns the specified MongoDB collection for the agent's memory."""
    try:
        client = get_mongo_client()
        db = client["HealthCoachDB"] 
        return db[collection_name]
    except Exception as e:
        logger.error(f"Failed to get collection '{collection_name}'. Error: {e}")
        raise

# --- 2. Data Persistence Tools (MongoDB Interaction) ---

def save_user_plan(user_id: str, plan_data: Dict[str, Any]) -> str:
    """Saves the latest generated health plan to MongoDB, with error handling."""
    try:
        plans_collection = get_collection("plans")
        
        plan_data['user_id'] = user_id
        plan_data['created_at'] = datetime.now()
        plan_data['is_active'] = True
        
        plans_collection.update_many(
            {"user_id": user_id, "is_active": True},
            {"$set": {"is_active": False}}
        )

        result = plans_collection.insert_one(plan_data)
        logger.info(f"New plan saved for user {user_id} with ID: {result.inserted_id}")
        return f"Plan saved successfully with ID: {result.inserted_id}"
    except Exception as e:
        logger.error(f"Failed to save plan for user {user_id}. Error: {e}")
        return f"ERROR: Plan save failed. {e}"


def load_active_plan(user_id: str) -> Optional[Dict[str, Any]]:
    """Loads the current active health plan for the user, with error handling."""
    try:
        plans_collection = get_collection("plans")
        plan_document = plans_collection.find_one({"user_id": user_id, "is_active": True})
        
        if plan_document:
            clean_plan = json.loads(json_util.dumps(plan_document))
            logger.info(f"Active plan loaded for user {user_id}.")
            return clean_plan
        
        logger.info(f"No active plan found for user {user_id}.")
        return None
    except Exception as e:
        logger.error(f"Failed to load active plan for user {user_id}. Error: {e}")
        return None


# --- 3. Data Mocking Tool (Mock API/Wearable Data) ---

def get_daily_logs(user_id: str, date: str) -> Dict[str, Any]:
    """Mocks retrieving a user's daily health logs."""
    random.seed(hash(user_id + date) % 1000)
    starting_weight = 85.0 
    current_weight = round(starting_weight + random.uniform(-1.0, 1.0), 1)

    logs = {
        "user_id": user_id,
        "date": date,
        "weight_kg": current_weight,
        "calories_consumed": random.randint(2000, 2600),
        "activity_calories_burned": random.randint(400, 800),
        "steps": random.randint(6000, 14000),
        "meals_summary": "Breakfast: Eggs & Avocado (400 kcal). Lunch: Chicken Rice (700 kcal). Dinner: Steak & Veggies (800 kcal). Snacks: 2 protein bars (500 kcal total)."
    }
    logger.debug(f"Mock logs generated for {user_id} on {date}.")
    return logs

# --- NEW: Synthetic History Tool for Visualization ---

def generate_weight_history(user_id: str, initial_weight: float, weeks: int = 12) -> List[Dict[str, Any]]:
    """
    Generates synthetic weekly weight history for visualization purposes.
    """
    history = []
    # Target loss of 0.5kg per week for a moderate goal
    target_weekly_loss = 0.5 
    
    for i in range(weeks):
        ideal_weight = initial_weight - (target_weekly_loss * (i + 1))
        
        # Simulate actual weight fluctuating slightly around the ideal trend
        fluctuation = random.uniform(-0.4, 0.4) * (1 + i * 0.1) 
        actual_weight = ideal_weight + fluctuation
        
        history.append({
            "week": i + 1,
            "date": (datetime.now() - timedelta(weeks=weeks-i)).strftime("%Y-%m-%d"),
            "actual_weight_kg": round(actual_weight, 2),
            "target_trend_kg": round(ideal_weight, 2),
        })
        
    logger.debug(f"Generated {weeks} weeks of synthetic history for user {user_id}.")
    return history


# --- 4. Reliable Calculation Tool ---

MALE_ADJUST = 5
FEMALE_ADJUST = -161
ACTIVITY_FACTORS = {
    "sedentary": 1.2, "lightly active": 1.375, "moderately active": 1.55, "very active": 1.725,
}

def calculate_metrics(weight_kg: float, height_cm: float, age_years: int, gender: str, activity_level: str = "moderately active") -> Dict[str, float]:
    """Calculates BMR and TDEE with input validation and error handling."""
    
    try:
        if not all(isinstance(x, (int, float)) and x > 0 for x in [weight_kg, height_cm, age_years]):
             raise ValueError("Weight, height, and age must be positive numbers.")

        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years)
        
        if gender.lower() == 'male':
            bmr += MALE_ADJUST
        elif gender.lower() == 'female':
            bmr += FEMALE_ADJUST
        
        factor = ACTIVITY_FACTORS.get(activity_level.lower(), 1.55)
        tdee = bmr * factor
        
        target_deficit = 500 
        target_maintain = round(tdee, 0)
        target_lose = round(tdee - target_deficit, 0)

        logger.debug(f"Metrics calculated: TDEE={target_maintain} kcal.")
        
        return {
            "bmr_kcal": target_maintain - target_deficit,
            "tdee_kcal": target_maintain,
            "target_weight_loss_kcal": target_lose,
            "activity_factor_used": factor
        }
        
    except ValueError as ve:
        logger.error(f"Validation error in calculate_metrics: {ve}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during metric calculation: {e}")
        raise
