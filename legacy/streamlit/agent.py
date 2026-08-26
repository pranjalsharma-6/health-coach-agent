# agent.py - Core Agent Logic and LLM Tool-Use Orchestration (POLISHED & CLOUD-READY)

import os
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# --- CORE LANGCHAIN IMPORTS ---
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

# Import your custom modules
from tools import get_daily_logs, calculate_metrics, save_user_plan, load_active_plan
from models import HealthPlan  # The structured output model

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )

# Load environment variables (for local development)
load_dotenv()


# --- 1. Initialize the LLM with Flexible API Key Loading ---

def get_api_key(key_name: str) -> Optional[str]:
    """
    Get API key from Streamlit secrets first, then fall back to environment variables.
    
    Args:
        key_name: Name of the API key (e.g., "GROQ_API_KEY", "OPENAI_API_KEY")
    
    Returns:
        API key string or None if not found
    """
    try:
        # Try Streamlit secrets (for Cloud deployment)
        import streamlit as st
        api_key = st.secrets.get(key_name)
        if api_key:
            logger.info(f"✅ {key_name} loaded from Streamlit secrets")
            return api_key
    except (ImportError, KeyError, FileNotFoundError):
        pass
    
    # Fall back to environment variables (for local development)
    api_key = os.getenv(key_name)
    if api_key:
        logger.info(f"✅ {key_name} loaded from environment variables")
        return api_key
    
    logger.warning(f"⚠️ {key_name} not found in Streamlit secrets or environment")
    return None


def initialize_llm() -> ChatGroq:
    """
    Initialize the LLM client. Tries GROQ first, falls back to OpenAI.
    
    Returns:
        Initialized LLM client
    
    Raises:
        ValueError: If no valid API key is found
    """
    # Try GROQ first (recommended - faster and free)
    groq_key = get_api_key("GROQ_API_KEY")
    if groq_key:
        logger.info("🚀 Using GROQ LLM")
        return ChatGroq(
            model="llama-3.3-70b-versatile",  # Fast and capable model
            temperature=0,
            api_key=groq_key,
            max_tokens=4096
        )
    
    # Fall back to OpenAI
    openai_key = get_api_key("OPENAI_API_KEY")
    if openai_key:
        logger.info("🚀 Using OpenAI LLM")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_key
        )
    
    # No API key found
    error_msg = (
        "No LLM API key found. Please configure either:\n"
        "1. GROQ_API_KEY (recommended - free at console.groq.com)\n"
        "2. OPENAI_API_KEY\n"
        "Add to .env file for local dev or Streamlit secrets for deployment"
    )
    logger.error(f"❌ {error_msg}")
    raise ValueError(error_msg)


# Initialize LLM
try:
    LLM = initialize_llm()
except ValueError as e:
    logger.critical(f"Failed to initialize LLM: {e}")
    LLM = None  # Will cause errors later, but allows import


# --- 2. Define User State (for the Agent's Context) ---
TEST_USER_ID: str = "kiit0001"
USER_PROFILE: Dict[str, Any] = {
    "user_id": TEST_USER_ID,
    "gender": "male",
    "age_years": 30,
    "height_cm": 175,
    "activity_level": "moderately active",
    "target_weight_kg": 75.0,
    "initial_weight_kg": 85.0,
    "goal": "Aggressively lose 10 kg over the next 12 weeks while building muscle mass. Must hit protein targets."
}


# --- 3. Define the Tools Available to the LLM ---
AGENT_TOOLS: List[Any] = [
    calculate_metrics,
    get_daily_logs,
    save_user_plan,
    load_active_plan,
]


# --- 4. The Core Planning Prompt ---
SYSTEM_PROMPT = """
You are an expert health and nutrition coach AI. Your role is to create personalized, 
science-based health plans that help users achieve their fitness goals safely and effectively.

**Your Expertise:**
- Nutrition science and macro/micronutrient planning
- Exercise programming for different goals (fat loss, muscle gain, general health)
- Behavioral psychology for sustainable habit formation
- Safe, evidence-based recommendations

**Your Approach:**
1. Analyze user profile (age, gender, activity level, goals)
2. Calculate appropriate caloric targets and macronutrient ratios
3. Design progressive workout plans
4. Create meal suggestions that are practical and nutritious
5. Consider sustainability and long-term adherence

**Safety First:**
- Never recommend extreme caloric deficits (>1000 kcal/day)
- Ensure adequate protein intake (1.6-2.2g/kg for muscle building)
- Promote balanced nutrition with variety
- Recommend medical consultation for special populations

**Output Quality:**
- Specific, actionable recommendations
- Clear rationale for each decision
- Realistic, achievable plans
- Personalized to user's context and preferences

Use the available tools to:
- get_daily_logs: Retrieve user's recent health data
- calculate_metrics: Compute BMR, TDEE, and caloric targets
- load_active_plan: Check existing plans
- save_user_plan: Store the generated plan
"""

PLANNING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", """
Analyze the provided user profile and generate a comprehensive 7-day health and nutrition plan.

**User Profile:**
- User ID: {user_id}
- Goal: {goal_description}
- Current Date: {current_date}

**Instructions:**
1. First, use calculate_metrics to determine caloric needs
2. Then use get_daily_logs to understand recent patterns
3. Create a detailed 7-day plan with:
   - Daily activities (type, duration, description)
   - 4 meals per day (breakfast, lunch, dinner, snack)
   - Caloric targets aligned with goals
   - Progressive difficulty/intensity
4. Save the plan using save_user_plan

Provide clear reasoning for your recommendations.
"""),
])


# --- 5. Initial Agent Setup and Planning ---

def run_initial_planning(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Sets up and runs the initial chain for generating the first plan.
    
    Args:
        user_id: Optional user ID (defaults to TEST_USER_ID)
    
    Returns:
        Dictionary with plan data and status
    """
    if LLM is None:
        error_msg = "LLM not initialized. Check API key configuration."
        logger.error(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}
    
    user_id = user_id or TEST_USER_ID
    logger.info(f"--- Starting Initial Planning for User: {user_id} ---")
    
    try:
        # 1. Bind tools to the LLM
        agent_chain = LLM.bind_tools(
            AGENT_TOOLS,
            tool_choice={"type": "function", "function": {"name": "save_user_plan"}}
        ).with_structured_output(
            schema=HealthPlan,
        )
        
        # 2. Prepare the context for the prompt
        context: Dict[str, str] = {
            "user_id": user_id,
            "goal_description": USER_PROFILE["goal"],
            "current_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        # 3. Run the chain (CRITICAL API CALL)
        logger.info("🔄 Invoking LLM for plan generation...")
        result = agent_chain.invoke(PLANNING_PROMPT.format(**context))
        
        # 4. Log Success
        logger.info("✅ Plan generation and saving successfully executed via LLM tool call")
        
        print("\n--- Plan Generation Complete ---")
        print(f"✅ Plan created for user: {user_id}")
        
        return {
            "status": "success",
            "message": "Plan generated successfully",
            "plan": result
        }

    except Exception as e:
        # Handle API or chain execution failure
        logger.error(f"❌ Plan generation failed: {e}")
        print("\n--- Plan Generation Failed ---")
        print(f"❌ Error: {str(e)}")
        print("💡 Check: 1) API key is valid, 2) API credits available, 3) Network connection")
        
        return {
            "status": "error",
            "message": str(e),
            "plan": None
        }


def check_agent_health() -> Dict[str, Any]:
    """
    Check if the agent is properly configured and can run.
    
    Returns:
        Dictionary with health status
    """
    health = {
        "llm_initialized": LLM is not None,
        "groq_key_available": get_api_key("GROQ_API_KEY") is not None,
        "openai_key_available": get_api_key("OPENAI_API_KEY") is not None,
        "tools_loaded": len(AGENT_TOOLS) > 0,
    }
    
    health["overall_status"] = "healthy" if health["llm_initialized"] else "unhealthy"
    
    return health


# --- Module Test ---
if __name__ == "__main__":
    # Check health first
    print("\n=== Agent Health Check ===")
    health = check_agent_health()
    for key, value in health.items():
        status = "✅" if value else "❌"
        print(f"{status} {key}: {value}")
    
    if health["overall_status"] == "healthy":
        print("\n=== Running Initial Planning ===")
        result = run_initial_planning()
        print(f"\n=== Result ===")
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
    else:
        print("\n❌ Agent is not healthy. Fix configuration issues above.")
    
    logger.info("Agent execution finished.")