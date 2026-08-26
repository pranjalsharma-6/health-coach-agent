# run_agent.py - LangGraph Implementation for Adaptive Planning Agent (POLISHED & STABILIZED)

from typing import TypedDict, Optional, Dict, List, Any
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging # <-- NEW: Logging module

# CORE LANGCHAIN IMPORTS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Import components from other files (ensure all are updated with logging/handling!)
from tools import get_daily_logs, calculate_metrics, save_user_plan, load_active_plan
from models import HealthPlan
from agent import LLM, AGENT_TOOLS, PLANNING_PROMPT, TEST_USER_ID, USER_PROFILE 

# --- Logging Configuration ---
# NOTE: This configuration needs to be executed once, often done in tools.py, but we'll ensure it's here too.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 
# To ensure logging works if not configured elsewhere:
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()

# --- 1. Define the Graph State (The Flowing Data) ---
class AgentState(TypedDict):
    """Represents the state of the agent's current task. (Includes Type Hints)"""
    user_id: str
    current_plan: Optional[Dict[str, Any]]
    progress_report: Optional[str]
    logs_data: Optional[Dict[str, Any]]
    replan_needed: bool
    plan_data: Optional[HealthPlan] 
    llm_context: str

# --- 2. Define the Graph Nodes (The Agent's Actions) ---

def fetch_data_node(state: AgentState) -> AgentState:
    """Node 1: Fetches the current plan and today's logs, with error handling."""
    logger.info("--- [Node: Fetch Data] ---")
    user_id = state['user_id']
    
    try:
        active_plan = load_active_plan(user_id)
        state['current_plan'] = active_plan

        today_date = datetime.now().strftime("%Y-%m-%d")
        logs = get_daily_logs(user_id, today_date)
        state['logs_data'] = logs
        
        state['llm_context'] = (
            f"USER PROFILE: {USER_PROFILE}\n"
            f"CURRENT ACTIVE PLAN: {active_plan if active_plan else 'None'}\n"
            f"TODAY'S LOGS: {logs}"
        )
        logger.info(f"Data fetched successfully for user {user_id}.")
        
    except ConnectionError:
        logger.error("FATAL: Failed to connect to MongoDB during fetch. Halting agent.")
        state['replan_needed'] = False 
        state['progress_report'] = "FATAL ERROR: Database connection failed."
        
    except Exception as e:
        logger.error(f"FATAL: Unexpected error in fetch_data_node: {e}")
        state['replan_needed'] = False
        state['progress_report'] = "FATAL ERROR: Data fetching failed."
        
    return state

def evaluate_progress_node(state: AgentState) -> AgentState:
    """Node 2: Evaluates compliance and progress to determine if replanning is needed."""
    logger.info("--- [Node: Evaluate Progress] ---")
    
    current_plan = state['current_plan']
    logs = state['logs_data']

    # AGENTIC REASONING LOGIC (Simplified Prototype)
    replan: bool = False
    report: str = ""
    
    if current_plan is None:
        report = "No active plan found. Initial plan required."
        replan = True
    else:
        try:
            # Get reliable calorie target for evaluation
            metrics = calculate_metrics(
                weight_kg=logs['weight_kg'], 
                height_cm=USER_PROFILE['height_cm'], 
                age_years=USER_PROFILE['age_years'], 
                gender=USER_PROFILE['gender']
            )
            target_calories = metrics['target_weight_loss_kcal']
            
            if logs['calories_consumed'] > target_calories * 1.2:
                report = f"Compliance Alert: Calories consumed ({logs['calories_consumed']}) were 20%+ over the target ({target_calories}). Plan adjustment is needed."
                replan = True
            elif logs['steps'] < 5000:
                report = f"Activity Alert: Steps ({logs['steps']}) were too low. Focus needs to shift to simple movement goals."
                replan = True
            else:
                report = "Progress is adequate. Maintaining current plan."
                
        except Exception as e:
            logger.warning(f"Error during metric calculation/evaluation: {e}. Forcing replan.")
            report = "FATAL: Metric calculation failed. Forcing replan to establish new base."
            replan = True # If metric calculation fails, better to replan than proceed with bad data

    state['progress_report'] = report
    state['replan_needed'] = replan
    state['llm_context'] += f"\nEVALUATION: {report}"
    
    return state

def planning_agent_node(state: AgentState) -> AgentState:
    """Node 3: The core LLM action to generate a new plan and save it, with robust error handling."""
    logger.info("--- [Node: Planning Agent] ---")
    logger.info(f"Reasoning for new plan: {state['progress_report']}")
    
    try:
        # 1. Bind the LLM to the tools and the required output structure
        agent_chain = LLM.bind_tools(
            AGENT_TOOLS,
            tool_choice={"type": "function", "function": {"name": "save_user_plan"}} 
        ).with_structured_output(
            schema=HealthPlan, 
        )
        
        # 2. Prepare the base context
        context = {
            "user_id": state['user_id'],
            "goal_description": USER_PROFILE["goal"],
            "current_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        # 3. Augment the prompt safely
        rendered_messages = PLANNING_PROMPT.format_messages(**context)
        base_system_content = rendered_messages[0].content
        user_message_content = rendered_messages[1].content
        
        augmented_system_content = (
            base_system_content + 
            f"\n\nCONTEXT AND HISTORY FOR ADAPTATION:\n{state['llm_context']}"
        )

        augmented_messages = [
            SystemMessage(content=augmented_system_content),
            HumanMessage(content=user_message_content)
        ]

        # 4. Run the chain with the augmented messages list (LLM API Call)
        result = agent_chain.invoke(augmented_messages)
        
        logger.info("Plan generated successfully by LLM.")
        
        # === CRITICAL FIX: Ensure Save to MongoDB ===
        if hasattr(result, 'dict'):
            plan_dict = result.dict()
        elif hasattr(result, 'model_dump'):
            plan_dict = result.model_dump()
        else:
            plan_dict = dict(result)

        # Save to MongoDB (uses the save_user_plan tool)
        save_result = save_user_plan(state['user_id'], plan_dict)
        logger.info(f"✅ Database Save Result: {save_result}")
        # === END FIX ===

        state['replan_needed'] = False
        state['plan_data'] = result 
        
    except Exception as e:
        # Handle all LLM/API errors (e.g., Auth, Rate Limit, Malformed Output)
        logger.error(f"FATAL ERROR: LLM planning or saving failed. Error: {e}")
        state['replan_needed'] = False
        state['progress_report'] = f"FATAL ERROR: Planning failed due to: {type(e).__name__}."
        state['plan_data'] = None
    
    return state


# --- 3. Define the LangGraph Structure (The State Machine) ---

def run_agent_loop(user_id: str):
    """Initializes and runs the LangGraph agent loop."""
    logger.info("\n\n--- Initializing LangGraph Agent Loop ---")
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("evaluate_progress", evaluate_progress_node)
    workflow.add_node("planning_agent", planning_agent_node)
    
    workflow.set_entry_point("fetch_data")
    
    workflow.add_edge("fetch_data", "evaluate_progress")
    
    # Conditional Edges: The Agentic Logic
    workflow.add_conditional_edges(
        "evaluate_progress",
        lambda state: "replan" if state['replan_needed'] else "end",
        {
            "replan": "planning_agent",
            "end": END
        }
    )
    
    workflow.add_edge("planning_agent", END)

    app = workflow.compile()
    
    initial_state = {
        "user_id": user_id,
        "replan_needed": False,
        "current_plan": None,
        "progress_report": None,
        "logs_data": None,
        "plan_data": None,
        "llm_context": ""
    }
    
    final_state = app.invoke(initial_state)
    
    logger.info("--- Agent Loop Finished ---")
    logger.info(f"Final Outcome: {final_state['progress_report']}")
    if final_state['plan_data']:
        logger.info("ACTION TAKEN: New plan was generated and saved to MongoDB.")
    else:
        logger.info("ACTION TAKEN: Plan maintained. No new plan generated.")
    
    return final_state

# --- Run the Agent (Main Entry Point) ---
if __name__ == "__main__":
    run_agent_loop(TEST_USER_ID)