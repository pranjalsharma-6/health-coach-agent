# app.py - Streamlit Interface for Health Coach Agent (POLISHED & VISUALIZED)

import streamlit as st

# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Health & Nutrition Coach",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
from run_agent import run_agent_loop, TEST_USER_ID
from tools import load_active_plan, generate_weight_history 
from agent import USER_PROFILE
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go 
import plotly.express as px 


# --- CACHED DATA LOADER FIX ---
@st.cache_data(show_spinner="Fetching latest plan from database...")
def get_latest_plan_from_db(user_id):
    """Retrieves the active plan from the database using a cached function."""
    return load_active_plan(user_id)
# --- END CACHED DATA LOADER FIX ---


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
        x=history_df['date'], 
        y=history_df['actual_weight_kg'], 
        mode='lines+markers',
        name='Actual Weight', 
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))

    # 2. Target Trend (Dashed Line)
    fig.add_trace(go.Scatter(
        x=history_df['date'], 
        y=history_df['target_trend_kg'], 
        mode='lines',
        name='Target Trend', 
        line=dict(color='green', dash='dash', width=2)
    ))

    fig.update_layout(
        title='Weight Loss Progress Over 12 Weeks',
        xaxis_title='Date / Week',
        yaxis_title='Weight (kg)',
        hovermode="x unified",
        template="plotly_dark",
        height=400,
        font=dict(size=12)
    )

    st.plotly_chart(fig, use_container_width=True)
# --- END NEW CHART FUNCTION ---


# --- Helper Function to Display Plan ---
def display_plan(plan_data: Dict[str, Any]):
    """
    Displays the health plan with overview and daily tabs.
    
    Args:
        plan_data: Dictionary containing the complete plan
    """
    if not plan_data:
        st.info("No active plan found. Click 'Run Adaptive Agent Loop' to generate one.")
        return
    
    # 1. Plan Overview
    st.header(plan_data.get('plan_title', 'Health Plan'))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Duration:** {plan_data.get('duration_days')} days")
        reasoning = plan_data.get('agent_reasoning', 'No rationale provided.')
        st.markdown(f"**Agent Rationale:** *{reasoning}*")
    
    with col2:
        # Show plan metadata
        if 'created_at' in plan_data:
            created_date = plan_data['created_at'].get('$date', 'Unknown')
            st.markdown(f"**Created:** {created_date[:10] if isinstance(created_date, str) else 'Recently'}")
        st.markdown(f"**Status:** {'🟢 Active' if plan_data.get('is_active', False) else '🔴 Inactive'}")
    
    st.markdown("---")
    
    # 2. Daily Plan Tabs
    daily_plans: List[Dict[str, Any]] = plan_data.get('daily_plans', [])
    
    if not daily_plans:
        st.warning("No daily plans found in this plan.")
        return
    
    # Convert Pydantic models to dicts if needed
    clean_daily_plans = [p.dict() if hasattr(p, 'dict') else p for p in daily_plans]

    # Create tabs for each day
    day_tabs = st.tabs([f"Day {i+1}" for i in range(len(clean_daily_plans))])

    for i, day_plan in enumerate(clean_daily_plans):
        with day_tabs[i]:
            # Activity Section
            activity = day_plan.get('activity', {})
            
            st.subheader(f"🏋️ Activity: {activity.get('activity_type', 'N/A')}")
            
            activity_col1, activity_col2 = st.columns([2, 1])
            
            with activity_col1:
                st.markdown(f"**Goal:** {activity.get('description', 'No specific goal.')}")
            
            with activity_col2:
                duration = activity.get('duration_minutes', 0)
                st.metric("Duration", f"{duration} min")
            
            st.markdown("---")
            
            # Meals Section
            st.subheader("🍽️ Meals")
            
            meals = day_plan.get('meals', [])
            
            if meals:
                # Create meals table
                meals_df = pd.DataFrame([
                    {
                        "Type": meal.get('meal_type', 'N/A'),
                        "Suggestion": meal.get('recipe_suggestion', 'N/A'),
                        "Calories (est.)": meal.get('estimated_kcal', 0)
                    }
                    for meal in meals
                ])
                
                st.dataframe(
                    meals_df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Type": st.column_config.TextColumn("Meal Type", width="small"),
                        "Suggestion": st.column_config.TextColumn("Recipe Suggestion", width="large"),
                        "Calories (est.)": st.column_config.NumberColumn("Calories", format="%d kcal")
                    }
                )
                
                # Total calories for the day
                total_calories = sum(meal.get('estimated_kcal', 0) for meal in meals)
                st.metric("Total Daily Calories", f"{total_calories} kcal")
            else:
                st.info("No meals planned for this day.")


# --- Streamlit Application Main Logic ---

st.title("🍏 Autonomous Health & Nutrition Coach")
st.markdown("*AI-powered personalized health and fitness planning*")
st.markdown("---")

# Display progress chart at the top
display_progress_chart(USER_PROFILE)

st.markdown("---")

# --- Sidebar for Agent Control/Info ---
with st.sidebar:
    st.header("⚙️ Agent Controls")
    
    # User Profile Summary
    with st.expander("👤 User Profile", expanded=False):
        st.write(f"**User ID:** {USER_PROFILE['user_id']}")
        st.write(f"**Goal:** {USER_PROFILE['goal']}")
        st.write(f"**Weight:** {USER_PROFILE['initial_weight_kg']} kg")
        st.write(f"**Height:** {USER_PROFILE['height_cm']} cm")
        st.write(f"**Age:** {USER_PROFILE['age_years']} years")
    
    st.markdown("---")
    
    # Run Agent Button
    if st.button("▶️ Run Adaptive Agent Loop", type="primary", use_container_width=True):
        with st.spinner("🤖 Agent is thinking... This may take 30-60 seconds."):
            try:
                final_state = run_agent_loop(TEST_USER_ID)
                
                # Clear the cache to force fresh DB pull
                get_latest_plan_from_db.clear() 
                
                st.success("✅ Agent run complete!")
                
                progress_report = final_state.get('progress_report', 'Plan generated successfully')
                st.markdown(f"**Outcome:** {progress_report}")
                
                # Trigger UI refresh
                st.rerun() 
                
            except Exception as e:
                st.error(f"❌ Error running agent: {str(e)}")
                st.exception(e)
    
    st.markdown("---")
    
    # Clear Plans Button
    if st.button("⚠️ Clear All Plans (Reset DB)", use_container_width=True):
        if st.session_state.get('confirm_reset'):
            # Actually reset
            if 'plan_data' in st.session_state:
                del st.session_state['plan_data']
            st.session_state['confirm_reset'] = False
            st.warning("Database reset initiated. Click 'Run Agent' to generate a new plan.")
            st.rerun()
        else:
            # Ask for confirmation
            st.session_state['confirm_reset'] = True
            st.warning("⚠️ Click again to confirm reset!")
    
    st.markdown("---")
    
    # Connection Status
    with st.expander("🔌 Connection Status", expanded=False):
        try:
            from tools import check_database_health
            health = check_database_health()
            
            if health.get('status') == 'healthy':
                st.success("✅ Database: Connected")
                st.write(f"**MongoDB Version:** {health.get('mongodb_version', 'Unknown')}")
                st.write(f"**Collections:** {health.get('collections_count', 0)}")
            else:
                st.error("❌ Database: Disconnected")
                st.write(f"**Error:** {health.get('error', 'Unknown')}")
        except Exception as e:
            st.error(f"❌ Cannot check database status: {str(e)}")


# --- Main Area for Plan Display ---

# Initialize session state for plan data only if it doesn't exist
if 'plan_data' not in st.session_state:
    try:
        with st.spinner("Loading plan from database..."):
            st.session_state['plan_data'] = get_latest_plan_from_db(TEST_USER_ID)
    except Exception as e:
        st.error(f"Error loading plan: {str(e)}")
        st.session_state['plan_data'] = None

# Display the plan
if st.session_state.get('plan_data'):
    display_plan(st.session_state['plan_data'])
else:
    st.info("ℹ️ No plan loaded. Run the agent to generate a plan.")
    
    # Show helpful getting started message
    with st.expander("🚀 Getting Started", expanded=True):
        st.markdown("""
        ### How to use this app:
        
        1. **Review your profile** in the sidebar
        2. **Click 'Run Adaptive Agent Loop'** to generate your personalized plan
        3. **Wait 30-60 seconds** for the AI to create your plan
        4. **View your daily plan** with meals and activities
        5. **Track your progress** in the chart above
        
        The AI agent will analyze your goals and create a customized health plan tailored to your needs!
        """)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Built with ❤️ using Streamlit, LangGraph, and GROQ | 
    Data stored securely in MongoDB Atlas
    </div>
    """,
    unsafe_allow_html=True
)