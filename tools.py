# tools.py - Core Functions for Data and Persistence (POLISHED & CLOUD-READY)

import os
import random
import json
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import pandas as pd
from bson import json_util 
import logging 

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)

# Load environment variables (for local development)
load_dotenv()

# --- 1. Database Connection & Helper ---

_db_client: Optional[MongoClient] = None

def get_mongo_client() -> MongoClient:
    """
    Initializes and returns the MongoDB client, connecting once, with error handling.
    Supports both Streamlit Cloud (secrets) and local development (.env file).
    """
    global _db_client
    if _db_client is None:
        uri = None
        
        # Try to get URI from Streamlit secrets first (for Streamlit Cloud deployment)
        try:
            import streamlit as st
            uri = st.secrets["mongodb"]["uri"]
            logger.info("✅ Using MongoDB URI from Streamlit secrets (Cloud deployment)")
        except (ImportError, KeyError, FileNotFoundError):
            # Fall back to environment variables (for local development)
            uri = os.getenv("MONGODB_URI")
            if uri:
                logger.info("✅ Using MongoDB URI from .env file (Local development)")
            else:
                logger.warning("⚠️ MONGODB_URI not found in Streamlit secrets or .env file")
        
        if not uri:
            error_msg = (
                "MONGODB_URI not found in environment. Please configure it:\n"
                "- For Streamlit Cloud: Add to app secrets (Settings > Secrets)\n"
                "- For local dev: Add to .env file as MONGODB_URI=your_connection_string"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Attempt connection with 10 second timeout
            _db_client = MongoClient(
                uri, 
                server_api=ServerApi('1'), 
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000
            )
            
            # Ping to verify connection
            _db_client.admin.command('ping')
            logger.info("🎉 Successfully connected to MongoDB!")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB. Error: {e}")
            logger.error("Check: 1) URI is correct, 2) Network access allowed, 3) Database user credentials")
            _db_client = None
            raise ConnectionError(f"MongoDB connection failed: {str(e)}")

    return _db_client


def get_collection(collection_name: str):
    """
    Returns the specified MongoDB collection for the agent's memory.
    
    Args:
        collection_name: Name of the collection (e.g., "plans", "logs")
    
    Returns:
        MongoDB collection object
    """
    try:
        client = get_mongo_client()
        db = client["HealthCoachDB"] 
        logger.debug(f"Accessing collection: {collection_name}")
        return db[collection_name]
    except Exception as e:
        logger.error(f"Failed to get collection '{collection_name}'. Error: {e}")
        raise


# --- 2. Data Persistence Tools (MongoDB Interaction) ---

def save_user_plan(user_id: str, plan_data: Dict[str, Any]) -> str:
    """
    Saves the latest generated health plan to MongoDB, with error handling.
    Automatically marks previous plans as inactive.
    
    Args:
        user_id: Unique identifier for the user
        plan_data: Dictionary containing the health plan details
    
    Returns:
        Success message with plan ID or error message
    """
    try:
        plans_collection = get_collection("plans")
        
        # Add metadata
        plan_data['user_id'] = user_id
        plan_data['created_at'] = datetime.now()
        plan_data['is_active'] = True
        
        # Deactivate all previous plans for this user
        plans_collection.update_many(
            {"user_id": user_id, "is_active": True},
            {"$set": {"is_active": False}}
        )

        # Insert the new plan
        result = plans_collection.insert_one(plan_data)
        logger.info(f"✅ New plan saved for user {user_id} with ID: {result.inserted_id}")
        return f"Plan saved successfully with ID: {result.inserted_id}"
        
    except Exception as e:
        logger.error(f"❌ Failed to save plan for user {user_id}. Error: {e}")
        return f"ERROR: Plan save failed. {str(e)}"


def load_active_plan(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Loads the current active health plan for the user, with error handling.
    
    Args:
        user_id: Unique identifier for the user
    
    Returns:
        Dictionary containing plan data, or None if no active plan exists
    """
    try:
        plans_collection = get_collection("plans")
        plan_document = plans_collection.find_one(
            {"user_id": user_id, "is_active": True},
            sort=[("created_at", -1)]  # Get most recent if multiple active plans exist
        )
        
        if plan_document:
            # Convert MongoDB ObjectId to string for JSON serialization
            clean_plan = json.loads(json_util.dumps(plan_document))
            logger.info(f"✅ Active plan loaded for user {user_id}")
            return clean_plan
        
        logger.info(f"ℹ️ No active plan found for user {user_id}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Failed to load active plan for user {user_id}. Error: {e}")
        return None


def get_plan_history(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieves plan history for a user (optional feature).
    
    Args:
        user_id: Unique identifier for the user
        limit: Maximum number of plans to retrieve
    
    Returns:
        List of plan documents sorted by creation date (newest first)
    """
    try:
        plans_collection = get_collection("plans")
        plans = list(plans_collection.find(
            {"user_id": user_id}
        ).sort("created_at", -1).limit(limit))
        
        clean_plans = [json.loads(json_util.dumps(plan)) for plan in plans]
        logger.info(f"Retrieved {len(clean_plans)} plans for user {user_id}")
        return clean_plans
        
    except Exception as e:
        logger.error(f"Failed to retrieve plan history for user {user_id}. Error: {e}")
        return []


# --- 3. Data Mocking Tool (Mock API/Wearable Data) ---

def get_daily_logs(user_id: str, date: str) -> Dict[str, Any]:
    """
    Mocks retrieving a user's daily health logs from a wearable device or app.
    In production, this would connect to actual APIs (Fitbit, Apple Health, etc.)
    
    Args:
        user_id: Unique identifier for the user
        date: Date string in format "YYYY-MM-DD"
    
    Returns:
        Dictionary with daily health metrics
    """
    # Use deterministic randomness based on user_id and date for consistency
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
        "sleep_hours": round(random.uniform(6.0, 8.5), 1),
        "water_intake_ml": random.randint(1500, 3000),
        "meals_summary": (
            "Breakfast: Eggs & Avocado (400 kcal). "
            "Lunch: Chicken Rice (700 kcal). "
            "Dinner: Steak & Veggies (800 kcal). "
            "Snacks: 2 protein bars (500 kcal total)."
        )
    }
    
    logger.debug(f"Mock logs generated for {user_id} on {date}")
    return logs


# --- 4. Synthetic History Tool for Visualization ---

def generate_weight_history(user_id: str, initial_weight: float, weeks: int = 12) -> List[Dict[str, Any]]:
    """
    Generates synthetic weekly weight history for visualization purposes.
    In production, this would query actual historical data from the database.
    
    Args:
        user_id: Unique identifier for the user
        initial_weight: Starting weight in kg
        weeks: Number of weeks of history to generate
    
    Returns:
        List of dictionaries with weekly weight data
    """
    history = []
    # Target loss of 0.5kg per week for a moderate, sustainable goal
    target_weekly_loss = 0.5 
    
    # Use user_id for consistent randomness
    random.seed(hash(user_id) % 1000)
    
    for i in range(weeks):
        # Calculate ideal weight following target trend
        ideal_weight = initial_weight - (target_weekly_loss * (i + 1))
        
        # Simulate actual weight fluctuating around the ideal trend
        # Fluctuation increases slightly over time to simulate real-world variance
        fluctuation = random.uniform(-0.4, 0.4) * (1 + i * 0.1) 
        actual_weight = max(ideal_weight + fluctuation, ideal_weight - 2)  # Don't go too far below target
        
        # Calculate date going backwards from now
        week_date = datetime.now() - timedelta(weeks=weeks-i)
        
        history.append({
            "week": i + 1,
            "date": week_date.strftime("%b %d\n%Y"),  # Format: "Nov 16\n2025"
            "actual_weight_kg": round(actual_weight, 2),
            "target_trend_kg": round(ideal_weight, 2),
        })
    
    logger.debug(f"Generated {weeks} weeks of synthetic history for user {user_id}")
    return history


# --- 5. Reliable Calculation Tool ---

# Constants for BMR calculation (Mifflin-St Jeor Equation)
MALE_ADJUST = 5
FEMALE_ADJUST = -161

ACTIVITY_FACTORS = {
    "sedentary": 1.2,           # Little to no exercise
    "lightly active": 1.375,    # Light exercise 1-3 days/week
    "moderately active": 1.55,  # Moderate exercise 3-5 days/week
    "very active": 1.725,       # Hard exercise 6-7 days/week
    "extremely active": 1.9     # Very hard exercise, physical job
}

def calculate_metrics(
    weight_kg: float, 
    height_cm: float, 
    age_years: int, 
    gender: str, 
    activity_level: str = "moderately active"
) -> Dict[str, float]:
    """
    Calculates BMR (Basal Metabolic Rate) and TDEE (Total Daily Energy Expenditure)
    using the Mifflin-St Jeor equation with input validation.
    
    Args:
        weight_kg: Current weight in kilograms
        height_cm: Height in centimeters
        age_years: Age in years
        gender: "male" or "female"
        activity_level: Activity level descriptor
    
    Returns:
        Dictionary with calculated metrics
    
    Raises:
        ValueError: If inputs are invalid
    """
    try:
        # Input validation
        if not all(isinstance(x, (int, float)) and x > 0 for x in [weight_kg, height_cm, age_years]):
            raise ValueError("Weight, height, and age must be positive numbers.")
        
        if weight_kg > 300 or weight_kg < 30:
            raise ValueError("Weight must be between 30 and 300 kg.")
        
        if height_cm > 250 or height_cm < 100:
            raise ValueError("Height must be between 100 and 250 cm.")
        
        if age_years > 120 or age_years < 10:
            raise ValueError("Age must be between 10 and 120 years.")

        # Calculate BMR using Mifflin-St Jeor Equation
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years)
        
        # Adjust for gender
        if gender.lower() == 'male':
            bmr += MALE_ADJUST
        elif gender.lower() == 'female':
            bmr += FEMALE_ADJUST
        else:
            logger.warning(f"Unknown gender '{gender}', defaulting to male adjustment")
            bmr += MALE_ADJUST
        
        # Get activity factor
        factor = ACTIVITY_FACTORS.get(activity_level.lower(), 1.55)
        if activity_level.lower() not in ACTIVITY_FACTORS:
            logger.warning(f"Unknown activity level '{activity_level}', using 'moderately active' (1.55)")
        
        # Calculate TDEE
        tdee = bmr * factor
        
        # Calculate targets
        target_deficit = 500  # 500 kcal deficit for ~0.5kg/week loss
        target_maintain = round(tdee, 0)
        target_lose = round(tdee - target_deficit, 0)
        target_gain = round(tdee + 300, 0)  # Slight surplus for muscle gain

        logger.debug(f"Metrics calculated: BMR={round(bmr, 0)}, TDEE={target_maintain} kcal")
        
        return {
            "bmr_kcal": round(bmr, 0),
            "tdee_kcal": target_maintain,
            "target_maintain_kcal": target_maintain,
            "target_weight_loss_kcal": target_lose,
            "target_weight_gain_kcal": target_gain,
            "activity_factor_used": factor,
            "daily_deficit_kcal": target_deficit
        }
        
    except ValueError as ve:
        logger.error(f"Validation error in calculate_metrics: {ve}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during metric calculation: {e}")
        raise


# --- 6. Health Status Check (Optional Helper) ---

def check_database_health() -> Dict[str, Any]:
    """
    Checks the health of the database connection and returns status info.
    Useful for debugging and monitoring.
    
    Returns:
        Dictionary with connection status and statistics
    """
    try:
        client = get_mongo_client()
        
        # Get server info
        server_info = client.server_info()
        
        # Get database stats
        db = client["HealthCoachDB"]
        stats = db.command("dbStats")
        
        return {
            "status": "healthy",
            "mongodb_version": server_info.get("version", "unknown"),
            "database_size_mb": round(stats.get("dataSize", 0) / (1024 * 1024), 2),
            "collections_count": stats.get("collections", 0),
            "connection_type": "Streamlit Secrets" if _is_using_streamlit_secrets() else "Environment Variable"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def _is_using_streamlit_secrets() -> bool:
    """Helper to check if using Streamlit secrets."""
    try:
        import streamlit as st
        st.secrets["mongodb"]["uri"]
        return True
    except:
        return False


# --- Module Initialization ---
logger.info("🔧 tools.py module loaded successfully")