# app.py - Streamlit Interface for Health Coach Agent (POLISHED & VISUALIZED)

import streamlit as st
import pandas as pd
from run_agent import run_agent_loop, TEST_USER_ID
# UPDATE: Import the new history tool
from tools import load_active_plan, generate_weight_history 
from agent import USER_PROFILE # Needed for chart data
from typing import Dict, Any, Optional, List

# NEW: Plotly imports for visualization
import plotly.graph_objects as go 
import plotly.express as px 


# --- CACHED DATA LOADER FIX ---
@st.cache_data(show_spinner="Fetching latest plan from database...")
def get_latest_plan_from_db(user_id):
    """Retrieves the active plan from the database using a cached function."""
    return load_active_plan(user_id)
# --- END CACHED DATA LOADER FIX ---


st.set_page_config(layout="wide")


# --- NEW: Chart Function ---

def display_progress_chart(user_profile: Dict[str, Any]):
    """Generates and displays a Plotly chart of synthetic weight progress."""
    st.subheader("📈 Progress Tracker (Synthetic Data)")
    
    # Generate data using the new tool
    history_list: List[Dict[str, Any]] = generate_weight_history(
        user_profile['user_id'], 
        user_profile['initial_weight_kg'], 
        weeks=12
    )
    history_df = pd.DataFrame(history_list)
    
    # Create the Plotly figure
    fig = go.Figure()

    # 1. Actual Weight (Line)
    fig.add_trace(go.Scatter(
        x=history_df['date'], y=history_df['actual_weight_kg'], mode='lines+markers',
        name='Actual Weight', line=dict(color='red')
    ))

    # 2. Target Trend (Dashed Line)
    fig.add_trace(go.Scatter(
        x=history_df['date'], y=history_df['target_trend_kg'], mode='lines',
        name='Target Trend', line=dict(color='green', dash='dash')
    ))

    fig.update_layout(
        title='Weight Loss Progress Over 12 Weeks',
        xaxis_title='Date / Week',
        yaxis_title='Weight (kg)',
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

# --- END NEW CHART FUNCTION ---


# --- Helper Function to Display Plan (Slightly modified with type hint) ---
def display_plan(plan_data: Dict[str, Any]):
    # Note: plan_data is a clean dictionary fetched from MongoDB
    if not plan_data:
        st.info("No active plan found. Click 'Run Agent' to generate one.")
        return
    
    # 1. Plan Overview
    st.header(plan_data.get('plan_title', 'Health Plan'))
    st.markdown(f"**Duration:** {plan_data.get('duration_days')} days")
    
    reasoning = plan_data.get('agent_reasoning', 'No rationale provided.')
    st.markdown(f"**Agent Rationale:** *{reasoning}*")
    
    st.markdown("---")
    
    # 2. Daily Plan Tabs
    daily_plans: List[Dict[str, Any]] = plan_data.get('daily_plans', [])
    clean_daily_plans = [p.dict() if hasattr(p, 'dict') else p for p in daily_plans]

    day_tabs = st.tabs([f"Day {i+1}" for i in range(len(clean_daily_plans))])

    for i, day_plan in enumerate(clean_daily_plans):
        with day_tabs[i]:
            activity = day_plan.get('activity', {})
            
            # Activity Display
            st.subheader(f"🏋️ Activity: {activity.get('activity_type', 'N/A')}")
            st.markdown(f"**Duration:** {activity.get('duration_minutes', 0)} min")
            st.markdown(f"**Goal:** {activity.get('description', 'No specific goal.')}")
            
            st.markdown("---")
            st.subheader("🍽️ Meals")
            
            # Meals Table Display
            meals_df = pd.DataFrame([
                {
                    "Type": meal.get('meal_type'),
                    "Suggestion": meal.get('recipe_suggestion'),
                    "Calories (est.)": meal.get('estimated_kcal')
                }
                for meal in day_plan.get('meals', [])
            ])
            st.dataframe(meals_df, use_container_width=True, hide_index=True)


# --- Streamlit Application Main Logic ---

st.title("🍏 Autonomous Health & Nutrition Coach")
st.markdown("---")

# NEW: Call the chart function immediately after the title
display_progress_chart(USER_PROFILE)

# 1. Sidebar for Agent Control/Info
with st.sidebar:
    st.header("Agent Controls")
    
    if st.button("▶️ Run Adaptive Agent Loop", type="primary"):
        st.info("Running agent loop... Please wait. This may take 30-60 seconds.")
        
        final_state = run_agent_loop(TEST_USER_ID)
        
        # 1. Clear the cache related to the plan data, forcing a fresh DB pull
        get_latest_plan_from_db.clear() 

        st.success("Agent run complete!")
        st.markdown(f"**Outcome:** {final_state.get('progress_report')}")
        
        # 2. Trigger the full UI refresh to run the initial loading block
        st.rerun() 
        

    # Optional: Button to clear memory for a fresh start 
    if st.button("⚠️ Clear All Plans (Reset DB)"):
        st.warning("Database reset initiated. Refresh page to see empty state.") 
        if 'plan_data' in st.session_state:
            del st.session_state['plan_data']
        st.rerun()


# 2. Main Area for Plan Display

# Initialize session state for plan data only if it doesn't exist
if 'plan_data' not in st.session_state:
    st.session_state['plan_data'] = get_latest_plan_from_db(TEST_USER_ID) 

# Display the plan
if st.session_state.get('plan_data'):
    display_plan(st.session_state['plan_data'])
else:
    st.warning("No plan loaded. Run the agent to generate a plan.")
