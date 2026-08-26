"""Log models — the sensing half of the agent loop.

Everything the agent reacts to enters the system through here.
"""

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.common import MongoModel, utcnow
from app.models.enums import AgentDecision, MealStatus


class MealLogEntry(BaseModel):
    """What actually happened with one planned meal."""

    meal_id: str
    status: MealStatus
    actual_calories_kcal: Optional[int] = None
    actual_protein_g: Optional[int] = None
    substitute_name: Optional[str] = Field(
        default=None, description="What they ate instead, if substituted."
    )
    note: Optional[str] = None
    logged_at: datetime = Field(default_factory=utcnow)


class MealLogRequest(BaseModel):
    """Payload for logging a single meal from the UI."""

    meal_id: str
    status: MealStatus
    actual_calories_kcal: Optional[int] = Field(default=None, ge=0, le=5000)
    actual_protein_g: Optional[int] = Field(default=None, ge=0, le=500)
    substitute_name: Optional[str] = Field(default=None, max_length=200)
    note: Optional[str] = Field(default=None, max_length=500)


class DailyLogInDB(MongoModel):
    """One document per user per day."""

    user_id: str
    log_date: date
    plan_id: Optional[str] = None
    plan_day: Optional[int] = None

    meals: List[MealLogEntry] = Field(default_factory=list)

    # Optional manual or wearable-sourced metrics.
    weight_kg: Optional[float] = None
    steps: Optional[int] = None
    sleep_hours: Optional[float] = None
    water_ml: Optional[int] = None

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class DailyMetricsRequest(BaseModel):
    """Manual entry of the day's body/activity metrics."""

    weight_kg: Optional[float] = Field(default=None, ge=30, le=300)
    steps: Optional[int] = Field(default=None, ge=0, le=100000)
    sleep_hours: Optional[float] = Field(default=None, ge=0, le=24)
    water_ml: Optional[int] = Field(default=None, ge=0, le=10000)


class AdherenceSnapshot(BaseModel):
    """Deterministic evaluation of how the user is tracking.

    Computed in plain Python from logs — this is evidence, not model output.
    It is what the agent's decision is made from.
    """

    date: date
    meals_planned: int
    meals_eaten: int
    meals_skipped: int
    meals_pending: int

    calories_target: int
    calories_consumed: int
    calories_remaining: int

    protein_target_g: int
    protein_consumed_g: int
    protein_remaining_g: int

    steps: Optional[int] = None
    sleep_hours: Optional[float] = None

    # Multi-day signals — the difference between a bad day and a bad pattern.
    skip_streak_days: int = 0
    skips_last_7_days: int = 0
    adherence_rate_7d: float = Field(
        default=1.0, description="Fraction of planned meals eaten over the last week."
    )

    @property
    def is_on_track(self) -> bool:
        return (
            self.meals_skipped == 0
            and self.calories_remaining >= -200
            and self.adherence_rate_7d >= 0.8
        )


class AgentEventInDB(MongoModel):
    """An entry in the agent's decision timeline.

    Written on every agent run, whether or not it changed anything. This is what
    makes the agent's behaviour auditable rather than magic.
    """

    user_id: str
    decision: AgentDecision
    rationale: str
    trigger_summary: str
    snapshot: Optional[AdherenceSnapshot] = None
    resulting_plan_id: Optional[str] = None
    duration_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=utcnow)
