# agent.py - Core Agent Logic and LLM Tool-Use Orchestration (POLISHED & STABILIZED)

import os
from dotenv import load_dotenv
from typing import Dict, Any, List
from datetime import datetime
import logging # <-- NEW: Logging module

# --- CORE LANGCHAIN IMPORTS ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

# Import your custom modules
from tools import get_daily_logs, calculate_metrics, save_user_plan, load_active_plan
from models import HealthPlan # The structured output model

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv()

# --- 1. Initialize the LLM ---
LLM: ChatOpenAI = ChatOpenAI( # <-- NEW: Type Hint
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# --- 2. Define User State (for the Agent's Context) ---
TEST_USER_ID: str = "kiit0001" # <-- NEW: Type Hint
USER_PROFILE: Dict[str, Any] = { # <-- NEW: Type Hint
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
AGENT_TOOLS: List[Any] = [ # <-- NEW: Type Hint
    calculate_metrics,
    get_daily_logs,
    save_user_plan,
    load_active_plan,
]

# --- 4. The Core Planning Prompt (No Change) ---
SYSTEM_PROMPT = """
... (Prompt content unchanged) ...
"""

PLANNING_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", "Analyze the provided user profile, history, and metrics. Generate a comprehensive 7-day health and nutrition plan for the goal: {goal_description}. The current date is {current_date}."),
    ]
)

# --- 5. Initial Agent Setup and Testing ---

def run_initial_planning() -> None: # <-- NEW: Type Hint
    """Sets up and runs the initial chain for generating the first plan, with error handling."""
    logger.info("--- Starting Initial Planning Chain ---")
    
    try:
        # 1. Bind tools to the LLM. 
        agent_chain = LLM.bind_tools(
            AGENT_TOOLS,
            tool_choice={"type": "function", "function": {"name": "save_user_plan"}} 
        ).with_structured_output(
            schema=HealthPlan, 
        )
        
        # 2. Prepare the context for the prompt
        context: Dict[str, str] = { # <-- NEW: Type Hint
            "user_id": TEST_USER_ID,
            "goal_description": USER_PROFILE["goal"],
            "current_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        # 3. Run the chain (CRITICAL API CALL)
        result = agent_chain.invoke(PLANNING_PROMPT.format(**context))
        
        # 4. Log Success
        logger.info("Plan generation and saving successfully executed via LLM tool call.")
        
        print("\n--- Plan Generation Attempt Complete ---")
        print(f"Agent's final call result (confirmation/plan data): {result}")

    except Exception as e:
        # Handle API or chain execution failure
        logger.error(f"FATAL ERROR: run_initial_planning failed. Check API key/credits or network. Error: {e}")
        print("\n--- Plan Generation Attempt Failed ---")
        print(f"FATAL ERROR: Execution halted. Check logs for details.")


if __name__ == "__main__":
    run_initial_planning()
    logger.info("Agent execution finished.")