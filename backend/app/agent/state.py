"""Agent state — the data that flows through the LangGraph workflow."""

from datetime import date
from typing import Any, Dict, List, Optional, TypedDict

from app.models.enums import AgentDecision
from app.models.log import AdherenceSnapshot, DailyLogInDB
from app.models.plan import HealthPlan, NutritionTargets, PlanInDB
from app.models.profile import ProfileInDB


class AgentState(TypedDict, total=False):
    """State passed between nodes.

    `total=False` because nodes populate it progressively — `snapshot` doesn't
    exist until `evaluate` runs, `saved_plan` not until `persist`.
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

    # --- Populated by generate / validate ---
    generated_plan: Optional[HealthPlan]
    validation_errors: List[str]
    validation_warnings: List[str]
    retry_feedback: str
    attempt: int

    # --- Populated by persist ---
    saved_plan: Optional[PlanInDB]

    # --- Throughout ---
    steps: List[Dict[str, Any]]
    error: Optional[str]


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
        generated_plan=None,
        validation_errors=[],
        validation_warnings=[],
        retry_feedback="",
        attempt=0,
        saved_plan=None,
        steps=[],
        error=None,
    )


def record_step(
    state: AgentState, node: str, status: str, message: str, **extra: Any
) -> None:
    """Append a step to the run trace.

    This trace is what gets streamed to the browser, so the user watches the
    agent reason instead of staring at a spinner — and it's what makes the
    agent's behaviour auditable after the fact.
    """
    step: Dict[str, Any] = {
        "node": node,
        "status": status,
        "message": message,
    }
    step.update(extra)
    state.setdefault("steps", []).append(step)
