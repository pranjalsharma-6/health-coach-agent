"""Agent state. The data that flows through the LangGraph workflow."""

import operator
from datetime import date
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from app.models.enums import AgentDecision
from app.models.log import AdherenceSnapshot, DailyLogInDB
from app.models.plan import (
    HealthPlan,
    MealPlanDraft,
    NutritionTargets,
    PlanCritique,
    PlanInDB,
    TrainingPlanDraft,
)
from app.models.profile import ProfileInDB


def keep_first_error(current: Optional[str], incoming: Optional[str]) -> Optional[str]:
    """Reducer for the `error` channel: the first failure reported wins.

    `plan_meals` and `plan_training` run in the same superstep, so a condition
    that breaks both. No API key, or a retired model, makes both write here at
    once. A plain LastValue channel rejects that with LangGraph's
    `InvalidUpdateError`, which replaces the real diagnosis with a framework
    error about concurrent writes.

    First-wins rather than last-wins because the two messages describe the same
    root cause, and the nutritionist's is the one the user cares about.
    Nothing in the graph ever clears this channel, so there is no case where
    holding the earlier value suppresses a recovery.
    """
    return current or incoming


class AgentState(TypedDict, total=False):
    """State passed between nodes.

    Every node returns a **partial** update. Only the keys it actually changed,
    rather than the whole dict. That isn't a style preference: the nutritionist
    and the trainer run concurrently, and LangGraph's default channel rejects two
    writes to the same key within one superstep. Two nodes each returning the
    full state would collide on every key they share.

    `total=False` because keys appear progressively: `snapshot` doesn't exist
    until `evaluate` runs, `saved_plan` not until `persist`.
    """

    # --- Input ---
    user_id: str
    today: date
    force_replan: bool

    # --- Populated by sense ---
    profile: Optional[ProfileInDB]
    targets: Optional[NutritionTargets]
    active_plan: Optional[PlanInDB]
    today_log: Optional[DailyLogInDB]
    recent_logs: List[DailyLogInDB]

    # --- Populated by evaluate ---
    snapshot: Optional[AdherenceSnapshot]

    # --- Populated by decide ---
    decision: AgentDecision
    trigger_detail: str

    # --- Populated by the specialists, concurrently ---
    meal_draft: Optional[MealPlanDraft]
    training_draft: Optional[TrainingPlanDraft]

    # --- Populated by critique ---
    # Named `critique_result` rather than `critique`: LangGraph forbids a state
    # key sharing a name with a node, and the node reads better as `critique`.
    critique_result: Optional[PlanCritique]
    critique_feedback: str
    critique_rounds: int

    # --- Populated by assemble / validate ---
    generated_plan: Optional[HealthPlan]
    validation_errors: List[str]
    validation_warnings: List[str]
    retry_feedback: str
    attempt: int

    # --- Populated by persist ---
    saved_plan: Optional[PlanInDB]

    # --- Throughout ---
    # Concatenated rather than overwritten, so both concurrent specialists can
    # record their progress without one clobbering the other.
    steps: Annotated[List[Dict[str, Any]], operator.add]
    error: Annotated[Optional[str], keep_first_error]


def new_state(user_id: str, today: date, force_replan: bool = False) -> AgentState:
    """Build the initial state for a run."""
    return AgentState(
        user_id=user_id,
        today=today,
        force_replan=force_replan,
        profile=None,
        targets=None,
        active_plan=None,
        today_log=None,
        recent_logs=[],
        snapshot=None,
        decision=AgentDecision.NO_ACTION,
        trigger_detail="",
        meal_draft=None,
        training_draft=None,
        critique_result=None,
        critique_feedback="",
        critique_rounds=0,
        generated_plan=None,
        validation_errors=[],
        validation_warnings=[],
        retry_feedback="",
        attempt=0,
        saved_plan=None,
        steps=[],
        error=None,
    )


def step(node: str, status: str, message: str, **extra: Any) -> Dict[str, Any]:
    """Build one entry for the run trace.

    Nodes collect these locally and return them under `steps`, where the reducer
    concatenates. Returning steps rather than mutating shared state is what lets
    two nodes record progress in the same superstep.

    The trace is streamed to the browser, so the user watches the agent reason
    instead of staring at a spinner, and it's what makes a run auditable
    afterwards.
    """
    entry: Dict[str, Any] = {"node": node, "status": status, "message": message}
    entry.update(extra)
    return entry
