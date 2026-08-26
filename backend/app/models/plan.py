"""Plan models — the agent's structured output and its stored form.

Design note on recipes:
    A full 7-day plan with complete recipes for every meal is ~8k output tokens,
    which is slow and a reliability risk for structured generation. So the planner
    emits meals with names, macros and a one-line description, and full recipe
    detail is expanded lazily per-meal through a separate endpoint. Users open
    maybe three recipes a week — generating 28 up front is waste.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.common import MongoModel, utcnow
from app.models.enums import AgentDecision, MealStatus, MealType


class Recipe(BaseModel):
    """Full cooking detail for a meal. Generated on demand."""

    ingredients: List[str] = Field(
        description="Ingredients with quantities, e.g. '150g paneer, cubed'."
    )
    steps: List[str] = Field(description="Numbered cooking steps, each one sentence.")
    prep_minutes: int = Field(description="Total hands-on time in minutes.")
    serves: int = Field(default=1, description="Number of portions this makes.")
    tips: Optional[str] = Field(
        default=None, description="One optional swap or make-ahead tip."
    )


class MealItem(BaseModel):
    """A single planned meal."""

    meal_id: str = Field(
        description="Stable identifier, e.g. 'd1-breakfast'. Lowercase, no spaces."
    )
    meal_type: MealType
    name: str = Field(description="Short dish name, e.g. 'Masala oats with peanuts'.")
    description: str = Field(
        description="One sentence on what it is and why it fits the user's goal."
    )
    calories_kcal: int = Field(description="Estimated calories for this meal.")
    protein_g: int = Field(description="Estimated protein in grams.")
    carbs_g: int = Field(description="Estimated carbohydrate in grams.")
    fat_g: int = Field(description="Estimated fat in grams.")

    # Populated lazily — not generated with the plan.
    recipe: Optional[Recipe] = None

    # Runtime state, set by the user rather than the planner.
    status: MealStatus = MealStatus.PLANNED
    logged_at: Optional[datetime] = None


class ActivityItem(BaseModel):
    """The day's movement prescription."""

    activity_type: str = Field(
        description="e.g. 'Strength training — upper body', 'Brisk walk', 'Rest'."
    )
    duration_minutes: int = Field(description="Suggested duration; 0 for a rest day.")
    intensity: str = Field(description="One of: low, moderate, high.")
    description: str = Field(description="One sentence on the goal of this session.")
    target_steps: int = Field(default=8000, description="Daily step target.")


class DailyPlan(BaseModel):
    """One day of the plan."""

    day: int = Field(description="Day number, starting at 1.")
    theme: Optional[str] = Field(
        default=None, description="Optional short theme, e.g. 'High protein, low prep'."
    )
    meals: List[MealItem]
    activity: ActivityItem


class NutritionTargets(BaseModel):
    """Daily targets, computed deterministically — never by the LLM."""

    calories_kcal: int
    protein_g: int
    carbs_g: int
    fat_g: int
    water_ml: int = 2500


class HealthPlan(BaseModel):
    """The structured output the LLM must produce.

    Kept flat and small enough that a 70B model produces it reliably.
    """

    plan_title: str = Field(
        description="Short motivational title, e.g. 'Week 1: Protein First'."
    )
    duration_days: int = Field(description="Number of days covered, usually 7.")
    agent_reasoning: str = Field(
        description=(
            "Two to three sentences explaining WHY this plan looks the way it does, "
            "referencing the user's goal, diet type and any recent adherence signals."
        )
    )
    daily_plans: List[DailyPlan]


class PlanInDB(MongoModel):
    """A stored plan version.

    Plans are immutable and versioned. Replanning writes a new document and
    deactivates the old one, so the full decision history stays browsable.
    """

    user_id: str
    version: int = 1
    is_active: bool = True

    plan_title: str
    duration_days: int
    agent_reasoning: str
    daily_plans: List[DailyPlan]
    targets: NutritionTargets

    # Provenance — why this version exists.
    trigger: AgentDecision = AgentDecision.CREATE_INITIAL
    trigger_detail: Optional[str] = None
    parent_plan_id: Optional[str] = None

    created_at: datetime = Field(default_factory=utcnow)


class PlanSummary(BaseModel):
    """Lightweight plan representation for history lists."""

    id: str
    version: int
    plan_title: str
    agent_reasoning: str
    trigger: AgentDecision
    trigger_detail: Optional[str]
    is_active: bool
    created_at: datetime
