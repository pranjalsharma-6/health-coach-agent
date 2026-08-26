"""The LangGraph planning agent.

    sense ──► evaluate ──► decide ──┬─(no_action)──────────────► record ──► END
                                    │
                                    └─(plan needed)──► generate ──► validate
                                                          ▲            │
                                                          │            ├─(valid)──► persist ──► END
                                                          └─(retry)────┘
                                                                       └─(exhausted)► record ──► END

Two things make this a real state machine rather than decoration:

1. `decide` genuinely branches — four outcomes, chosen from computed evidence,
   producing structurally different actions.
2. `validate` can send control *backwards* to `generate`. That cycle is how a
   rejected plan gets regenerated with specific corrective feedback.
"""

import time
from datetime import date
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.agent.llm import LLMUnavailableError, get_llm, get_structured_llm
from app.agent.prompts import (
    RECIPE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_recipe_prompt,
    build_user_prompt,
)
from app.agent.state import AgentState, new_state, record_step
from app.agent.validators import build_retry_feedback, validate_plan
from app.core.logging import get_logger
from app.db.repositories import (
    AgentEventRepository,
    LogRepository,
    PlanRepository,
    ProfileRepository,
)
from app.models.enums import AgentDecision
from app.models.log import AgentEventInDB
from app.models.plan import HealthPlan, PlanInDB, Recipe
from app.models.profile import ProfileInDB
from app.services.adherence import (
    STRUCTURAL_ADHERENCE_THRESHOLD,
    build_snapshot,
    describe_snapshot,
)
from app.services.nutrition import calculate_targets

logger = get_logger(__name__)

MAX_GENERATION_ATTEMPTS = 3
PLAN_DURATION_DAYS = 7

# Skipping meals this many days running means the plan doesn't fit their life.
STRUCTURAL_SKIP_STREAK = 3

# Minimum logged meals before the 7-day adherence rate is trusted as a signal.
MIN_MEALS_FOR_ADHERENCE_RULE = 8

# Eating this far over target, with meals still to come, warrants a rebalance.
CALORIE_OVERAGE_TRIGGER = 1.15


# --------------------------------------------------------------------------- #
# Node: sense
# --------------------------------------------------------------------------- #
async def sense_node(state: AgentState) -> AgentState:
    """Gather everything the agent needs to reason about."""
    user_id = state["user_id"]
    record_step(state, "sense", "running", "Gathering your profile and recent logs…")

    profile = await ProfileRepository.get(user_id)
    if profile is None:
        state["error"] = "No profile found. Complete onboarding first."
        record_step(state, "sense", "error", state["error"])
        return state

    state["profile"] = profile
    state["targets"] = calculate_targets(profile)
    state["active_plan"] = await PlanRepository.get_active(user_id)
    state["today_log"] = await LogRepository.get_or_create(user_id, state["today"])
    state["recent_logs"] = await LogRepository.get_recent(user_id, days=7)

    record_step(
        state,
        "sense",
        "done",
        (
            f"Loaded profile ({profile.diet_type}, {profile.goal}) and "
            f"{len(state['recent_logs'])} days of logs."
        ),
    )
    return state


# --------------------------------------------------------------------------- #
# Node: evaluate
# --------------------------------------------------------------------------- #
async def evaluate_node(state: AgentState) -> AgentState:
    """Compute the adherence snapshot. Pure arithmetic — no LLM."""
    if state.get("error"):
        return state

    record_step(state, "evaluate", "running", "Measuring how you're tracking…")

    snapshot = build_snapshot(
        target_date=state["today"],
        targets=state["targets"],
        plan=state.get("active_plan"),
        today_log=state.get("today_log"),
        recent_logs=state.get("recent_logs", []),
    )
    state["snapshot"] = snapshot

    record_step(state, "evaluate", "done", describe_snapshot(snapshot))
    return state


# --------------------------------------------------------------------------- #
# Node: decide
# --------------------------------------------------------------------------- #
async def decide_node(state: AgentState) -> AgentState:
    """Choose the action, deterministically, from the snapshot.

    Deliberately not an LLM call. The decision to intervene in someone's diet
    should be reproducible and explainable, and the rules below are both. The
    LLM is brought in afterwards to *execute* the decision, not to make it.
    """
    if state.get("error"):
        return state

    record_step(state, "decide", "running", "Deciding whether to change your plan…")

    snapshot = state["snapshot"]
    plan: Optional[PlanInDB] = state.get("active_plan")
    targets = state["targets"]

    decision, detail = _choose_action(state, snapshot, plan, targets)

    state["decision"] = decision
    state["trigger_detail"] = detail

    record_step(
        state,
        "decide",
        "done",
        detail,
        decision=decision.value,
    )
    return state


def _choose_action(state, snapshot, plan, targets) -> tuple[AgentDecision, str]:
    """The decision rules, ordered most-severe first."""
    # 1. Nothing to work from.
    if plan is None:
        return (
            AgentDecision.CREATE_INITIAL,
            "No active plan yet — building your first week.",
        )

    # 2. Explicit user request.
    if state.get("force_replan"):
        return (
            AgentDecision.STRUCTURAL_REPLAN,
            "You asked for a fresh plan.",
        )

    # 3. A sustained pattern of skipping: the plan is the problem.
    if snapshot.skip_streak_days >= STRUCTURAL_SKIP_STREAK:
        return (
            AgentDecision.STRUCTURAL_REPLAN,
            (
                f"You've skipped meals {snapshot.skip_streak_days} days running. "
                "That's a sign the plan doesn't fit your routine, so I'm changing "
                "its shape rather than asking you to try harder."
            ),
        )

    # Only trust the adherence rate once there's enough evidence behind it.
    # One skip out of two logged meals reads as 50% adherence, which would
    # otherwise tear up a brand-new plan on the user's first bad morning.
    if (
        snapshot.meals_logged_7d >= MIN_MEALS_FOR_ADHERENCE_RULE
        and snapshot.adherence_rate_7d < STRUCTURAL_ADHERENCE_THRESHOLD
    ):
        return (
            AgentDecision.STRUCTURAL_REPLAN,
            (
                f"Your 7-day adherence is {snapshot.adherence_rate_7d:.0%} across "
                f"{snapshot.meals_logged_7d} logged meals. Rebuilding around meals "
                "that are easier to hit."
            ),
        )

    # 4. The plan has run its course.
    days_elapsed = (state["today"] - plan.created_at.date()).days
    if days_elapsed >= plan.duration_days:
        return (
            AgentDecision.STRUCTURAL_REPLAN,
            f"Your {plan.duration_days}-day plan is complete — here's the next block.",
        )

    # 5. Today went sideways, but there's still road left.
    if snapshot.meals_pending > 0:
        if snapshot.meals_skipped > 0:
            return (
                AgentDecision.REBALANCE_DAY,
                (
                    f"You skipped {snapshot.meals_skipped} meal(s) today, leaving "
                    f"{snapshot.calories_remaining} kcal and "
                    f"{snapshot.protein_remaining_g}g protein to make up. "
                    "Rebalancing the rest of the day."
                ),
            )

        if snapshot.calories_consumed > targets.calories_kcal * CALORIE_OVERAGE_TRIGGER:
            return (
                AgentDecision.REBALANCE_DAY,
                (
                    f"You're at {snapshot.calories_consumed} kcal against a "
                    f"{targets.calories_kcal} target with meals still to come. "
                    "Lightening what's left."
                ),
            )

    # 6. Nothing to do.
    return (
        AgentDecision.NO_ACTION,
        "You're on track — keeping your current plan as it is.",
    )


# --------------------------------------------------------------------------- #
# Node: generate
# --------------------------------------------------------------------------- #
async def generate_node(state: AgentState) -> AgentState:
    """Call the LLM to produce a plan."""
    if state.get("error"):
        return state

    state["attempt"] = state.get("attempt", 0) + 1
    attempt = state["attempt"]

    label = (
        "Writing your plan…"
        if attempt == 1
        else f"Revising the plan (attempt {attempt}) to fix validation problems…"
    )
    record_step(state, "generate", "running", label, attempt=attempt)

    profile: ProfileInDB = state["profile"]

    user_prompt = build_user_prompt(
        profile=profile,
        targets=state["targets"],
        decision=state["decision"],
        snapshot=state.get("snapshot"),
        current_plan=state.get("active_plan"),
        trigger_detail=state.get("trigger_detail", ""),
        duration_days=PLAN_DURATION_DAYS,
    )

    if state.get("retry_feedback"):
        user_prompt += "\n\n---\n\n" + state["retry_feedback"]

    try:
        structured = get_structured_llm(HealthPlan)
        started = time.perf_counter()
        plan: HealthPlan = await structured.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        state["generated_plan"] = plan
        record_step(
            state,
            "generate",
            "done",
            f'Drafted "{plan.plan_title}" ({len(plan.daily_plans)} days).',
            duration_ms=elapsed_ms,
        )

    except LLMUnavailableError as exc:
        state["error"] = str(exc)
        record_step(state, "generate", "error", str(exc))

    except Exception as exc:
        # A malformed structured-output response lands here. Let the retry edge
        # handle it rather than failing the whole run on the first attempt.
        logger.exception("Plan generation failed on attempt %s", attempt)
        state["generated_plan"] = None
        state["validation_errors"] = [f"Generation failed: {type(exc).__name__}"]
        state["retry_feedback"] = (
            "Your previous response could not be parsed into the required schema. "
            "Return valid structured output matching the schema exactly."
        )
        record_step(
            state,
            "generate",
            "error",
            f"Generation attempt {attempt} failed: {type(exc).__name__}",
        )

    return state


# --------------------------------------------------------------------------- #
# Node: validate
# --------------------------------------------------------------------------- #
async def validate_node(state: AgentState) -> AgentState:
    """Check the generated plan against safety and constraint rules."""
    if state.get("error"):
        return state

    plan = state.get("generated_plan")
    if plan is None:
        # Generation already failed; the retry edge will decide what happens.
        return state

    record_step(state, "validate", "running", "Checking the plan is safe and on-diet…")

    result = validate_plan(plan, state["profile"], state["targets"])
    state["validation_errors"] = result.errors
    state["validation_warnings"] = result.warnings

    if result.is_valid:
        message = "Plan passed all checks."
        if result.warnings:
            message += f" ({len(result.warnings)} minor note(s).)"
        record_step(state, "validate", "done", message)
    else:
        state["retry_feedback"] = build_retry_feedback(result)
        record_step(
            state,
            "validate",
            "failed",
            f"Rejected: {result.errors[0]}"
            + (
                f" (+{len(result.errors) - 1} more)"
                if len(result.errors) > 1
                else ""
            ),
            errors=result.errors,
        )

    return state


# --------------------------------------------------------------------------- #
# Node: persist
# --------------------------------------------------------------------------- #
async def persist_node(state: AgentState) -> AgentState:
    """Save the validated plan as a new version and record the decision."""
    if state.get("error"):
        return state

    record_step(state, "persist", "running", "Saving your plan…")

    generated: HealthPlan = state["generated_plan"]
    parent = state.get("active_plan")

    plan = PlanInDB(
        user_id=state["user_id"],
        plan_title=generated.plan_title,
        duration_days=len(generated.daily_plans),
        agent_reasoning=generated.agent_reasoning,
        daily_plans=generated.daily_plans,
        targets=state["targets"],
        trigger=state["decision"],
        trigger_detail=state.get("trigger_detail"),
        parent_plan_id=str(parent.id) if parent else None,
    )

    saved = await PlanRepository.save_new_version(plan)
    state["saved_plan"] = saved

    await _record_event(state, resulting_plan_id=str(saved.id))

    record_step(
        state,
        "persist",
        "done",
        f'Saved "{saved.plan_title}" as version {saved.version}.',
        plan_id=str(saved.id),
        version=saved.version,
    )
    return state


# --------------------------------------------------------------------------- #
# Node: record (terminal, for runs that produce no plan)
# --------------------------------------------------------------------------- #
async def record_node(state: AgentState) -> AgentState:
    """Write the decision to the timeline even when nothing changed.

    Recording no-ops matters: it's the difference between "the agent decided
    you're fine" and "the agent didn't run".
    """
    if state.get("error"):
        record_step(state, "record", "error", state["error"])
        return state

    if state["decision"] != AgentDecision.NO_ACTION and not state.get("saved_plan"):
        state["error"] = (
            "Could not produce a valid plan after "
            f"{state.get('attempt', 0)} attempts. Your existing plan is unchanged."
        )
        record_step(state, "record", "error", state["error"])

    await _record_event(state)
    return state


async def _record_event(state: AgentState, resulting_plan_id: Optional[str] = None):
    """Append to the agent's decision timeline."""
    try:
        await AgentEventRepository.record(
            AgentEventInDB(
                user_id=state["user_id"],
                decision=state["decision"],
                rationale=(
                    state["generated_plan"].agent_reasoning
                    if state.get("generated_plan")
                    else state.get("trigger_detail", "")
                ),
                trigger_summary=state.get("trigger_detail", ""),
                snapshot=state.get("snapshot"),
                resulting_plan_id=resulting_plan_id,
            )
        )
    except Exception:
        # The timeline is valuable but not worth failing a good plan over.
        logger.exception("Failed to record agent event")


# --------------------------------------------------------------------------- #
# Conditional edges
# --------------------------------------------------------------------------- #
def route_after_decide(state: AgentState) -> str:
    if state.get("error"):
        return "record"
    if state["decision"] == AgentDecision.NO_ACTION:
        return "record"
    return "generate"


def route_after_validate(state: AgentState) -> str:
    """Persist, retry, or give up.

    This is the edge that makes the graph cyclic — a rejected plan goes back to
    `generate` carrying specific feedback about what was wrong with it.
    """
    if state.get("error"):
        return "record"

    plan_ok = state.get("generated_plan") is not None and not state.get(
        "validation_errors"
    )
    if plan_ok:
        return "persist"

    if state.get("attempt", 0) < MAX_GENERATION_ATTEMPTS:
        return "generate"

    return "record"


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_graph():
    """Compile the workflow. Structure is static, so this is cached below."""
    workflow = StateGraph(AgentState)

    workflow.add_node("sense", sense_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("persist", persist_node)
    workflow.add_node("record", record_node)

    workflow.set_entry_point("sense")

    workflow.add_edge("sense", "evaluate")
    workflow.add_edge("evaluate", "decide")

    workflow.add_conditional_edges(
        "decide",
        route_after_decide,
        {"generate": "generate", "record": "record"},
    )

    workflow.add_edge("generate", "validate")

    workflow.add_conditional_edges(
        "validate",
        route_after_validate,
        {"persist": "persist", "generate": "generate", "record": "record"},
    )

    workflow.add_edge("persist", END)
    workflow.add_edge("record", END)

    return workflow.compile()


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
        logger.info("LangGraph workflow compiled")
    return _compiled


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
async def run_agent(
    user_id: str, today: Optional[date] = None, force_replan: bool = False
) -> AgentState:
    """Run the full agent loop once and return the final state."""
    started = time.perf_counter()
    state = new_state(user_id, today or date.today(), force_replan)

    final: AgentState = await get_graph().ainvoke(state)

    elapsed = int((time.perf_counter() - started) * 1000)
    logger.info(
        "Agent run for %s finished in %dms (decision=%s, plan_saved=%s)",
        user_id,
        elapsed,
        final.get("decision"),
        bool(final.get("saved_plan")),
    )
    return final


async def stream_agent(
    user_id: str, today: Optional[date] = None, force_replan: bool = False
):
    """Run the agent, yielding each step as it completes.

    Used by the SSE endpoint so the frontend can show the agent working rather
    than a 30-second spinner.
    """
    initial = new_state(user_id, today or date.today(), force_replan)
    emitted = 0
    final: AgentState = initial

    # `values` mode yields the full merged state after each node completes, so
    # we diff the accumulated step list against what we've already sent.
    async for chunk in get_graph().astream(initial, stream_mode="values"):
        final = chunk
        steps = chunk.get("steps", [])
        while emitted < len(steps):
            yield steps[emitted]
            emitted += 1

    decision = final.get("decision")
    saved = final.get("saved_plan")

    yield {
        "node": "complete",
        "status": "done",
        "message": "Agent run finished.",
        "decision": getattr(decision, "value", decision),
        "plan_id": str(saved.id) if saved else None,
        "error": final.get("error"),
    }


# --------------------------------------------------------------------------- #
# Recipe expansion (outside the graph — enrichment, not a planning decision)
# --------------------------------------------------------------------------- #
async def generate_recipe(
    meal_name: str,
    description: str,
    calories: int,
    protein_g: int,
    profile: ProfileInDB,
) -> Recipe:
    """Expand one planned meal into a full recipe, on demand."""
    structured = get_structured_llm(Recipe)
    return await structured.ainvoke(
        [
            SystemMessage(content=RECIPE_SYSTEM_PROMPT),
            HumanMessage(
                content=build_recipe_prompt(
                    meal_name, description, calories, protein_g, profile
                )
            ),
        ]
    )
