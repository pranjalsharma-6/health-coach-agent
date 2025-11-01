# models.py - Pydantic Schemas for Structured Output (CORRECTED)

from pydantic import BaseModel, Field
from typing import List

# --- 1. Sub-models defining the details of a plan ---

class MealItem(BaseModel):
    """Defines a single meal within a day's plan."""
    meal_type: str = Field(description="Breakfast, Lunch, Dinner, or Snack.")
    recipe_suggestion: str = Field(description="A brief, specific, and easy-to-follow meal idea (e.g., Greek yogurt with honey and walnuts).")
    estimated_kcal: int = Field(description="Estimated calories for this single meal.")

class ActivityItem(BaseModel):
    """Defines a physical activity for the day."""
    activity_type: str = Field(description="e.g., Cardio, Strength Training, Yoga.")
    duration_minutes: int = Field(description="Suggested duration in minutes.")
    description: str = Field(description="A brief description of the goal for this activity (e.g., Focus on light recovery and stretching).")

# --- 2. Corrected Nested Daily Plan Model ---
# This must be a separate class, not defined inline.
class DailyPlan(BaseModel):
    """Defines the meals and activity for a single day."""
    day: int = Field(description="Day number in the sequence (1, 2, 3, etc.)")
    meals: List[MealItem] = Field(description="List of all meals for the day.")
    activity: ActivityItem = Field(description="The primary physical activity for the day.")


# --- 3. The main model for the entire plan output ---

class HealthPlan(BaseModel):
    """
    The complete, structured output model for the Agent's generated health plan.
    The LLM MUST generate JSON that conforms exactly to this schema.
    """
    plan_title: str = Field(description="A short, motivational title for the plan (e.g., 'Week 1: Focus on Protein').")
    duration_days: int = Field(description="The length of the plan in days (e.g., 7).")
    
    # Reasoning from the Agent (crucial for transparency)
    agent_reasoning: str = Field(description="A 2-3 sentence summary of *why* this plan was generated (e.g., 'Plan adjusted to increase daily protein intake based on low compliance with strength goals').")

    # This now references the defined DailyPlan class
    daily_plans: List[DailyPlan]