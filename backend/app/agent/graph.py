"""The LangGraph planning agent.

    sense ─► evaluate ─► decide ─┬─(no action)─────────────────────► record ─► END
                                 │
                                 └─(plan needed)─► start_generation
                                                     │         │
                                          ┌──────────┘         └──────────┐
                                          ▼                              ▼
                                    plan_meals                     plan_training
                                          └──────────┐         ┌──────────┘
                                                     ▼         ▼
                                                     assemble
                                                        │
                                                        ▼
                                                     critique
                                          ┌─────────────┤
                                  (revise)│             │(ok / spent)
                                          │             ▼
                                          │          validate
                                          │             │
                                          ├─────────────┤(retry)
                                          │             ├─(valid)────► persist ─► END
                                          │             └─(exhausted)► record ──► END
                                          └──► plan_meals

Three things make this a real state machine rather than decoration:

1. `decide` genuinely branches — four outcomes, chosen from computed evidence,
   producing structurally different actions.
2. Two specialists run **concurrently** in the same superstep. A nutritionist
   plans food and a trainer plans movement, each seeing only the constraints
   its own half needs, so neither prompt has to carry the other's. Splitting
   them keeps each output small, and small structured outputs are the reliable
   ones.
3. Control flows *backwards* twice — from `critique` and from `validate` — so a
   plan gets revised with specific feedback rather than regenerated blind.

The ordering of the last two matters: the critic is an LLM and only advises,
while `validate` is deterministic and has the final say. A model that approves
an unsafe plan must not be able to make it safe.
"""

import time
from datetime import date
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.agent.llm import LLMUnavailableError, get_llm, get_structured_llm
from app.agent.prompts import (
    CRITIC_SYSTEM_PROMPT,
    NUTRITIONIST_SYSTEM_PROMPT,
    RECIPE_SYSTEM_PROMPT,
    TRAINER_SYSTEM_PROMPT,
    build_critic_prompt,
    build_critique_feedback,
    build_nutritionist_prompt,
    build_recipe_correction,
    build_recipe_prompt,
    build_trainer_prompt,
)
from app.agent.state import AgentState, new_state, step
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
from app.models.plan import (
    ActivityItem,
    DailyPlan,
    HealthPlan,
    MealPlanDraft,
    PlanCritique,
    PlanInDB,
    Recipe,
    TrainingPlanDraft,
)
from app.models.profile import ProfileInDB
from app.services.adherence import (
    STRUCTURAL_ADHERENCE_THRESHOLD,
    build_snapshot,
    describe_snapshot,
)
from app.services.ingredients import RecipeAnalysis, analyse_recipe
from app.services.nutrition import calculate_targets

logger = get_logger(__name__)

MAX_GENERATION_ATTEMPTS = 3
PLAN_DURATION_DAYS = 7

# The critic gets one revision round. Allowed to keep asking, it would spend
# the user's time on diminishing preferences.
MAX_CRITIQUE_ROUNDS = 1

# Skipping meals this many days running means the plan doesn't fit their life.
STRUCTURAL_SKIP_STREAK = 3

# Minimum logged meals before the 7-day adherence rate is trusted as a signal.
MIN_MEALS_FOR_ADHERENCE_RULE = 8

# Eating this far over target, with meals still to come, warrants a rebalance.
CALORIE_OVERAGE_TRIGGER = 1.15


# --------------------------------------------------------------------------- #
# Node: sense
# --------------------------------------------------------------------------- #
async def sense_node(state: AgentState) -> Dict[str, Any]:
    """Gather everything the agent needs to reason about."""
    user_id = state["user_id"]

    profile = await ProfileRepository.get(user_id)
    if profile is None:
        message = "No profile found. Complete onboarding first."
        return {"error": message, "steps": [step("sense", "error", message)]}

    active_plan = await PlanRepository.get_active(user_id)
    today_log = await LogRepository.get_or_create(user_id, state["today"])
    recent_logs = await LogRepository.get_recent(user_id, days=7)

    return {
        "profile": profile,
        "targets": calculate_targets(profile),
        "active_plan": active_plan,
        "today_log": today_log,
        "recent_logs": recent_logs,
        "steps": [
            step(
                "sense",
                "done",
                f"Loaded profile ({profile.diet_type}, {profile.goal}) and "
                f"{len(recent_logs)} days of logs.",
            )
        ],
    }


# --------------------------------------------------------------------------- #
# Node: evaluate
# --------------------------------------------------------------------------- #
async def evaluate_node(state: AgentState) -> Dict[str, Any]:
    """Compute the adherence snapshot. Pure arithmetic — no LLM."""
    if state.get("error"):
        return {}

    snapshot = build_snapshot(
        target_date=state["today"],
        targets=state["targets"],
        plan=state.get("active_plan"),
        today_log=state.get("today_log"),
        recent_logs=state.get("recent_logs", []),
    )
    return {
        "snapshot": snapshot,
        "steps": [step("evaluate", "done", describe_snapshot(snapshot))],
    }


# --------------------------------------------------------------------------- #
# Node: decide
# --------------------------------------------------------------------------- #
async def decide_node(state: AgentState) -> Dict[str, Any]:
    """Choose the action, deterministically, from the snapshot.

    Deliberately not an LLM call. The decision to intervene in someone's diet
    should be reproducible and explainable, and the rules below are both. The
    specialists are brought in afterwards to *execute* the decision, not to
    make it.
    """
    if state.get("error"):
        return {}

    decision, detail = _choose_action(
        state, state["snapshot"], state.get("active_plan"), state["targets"]
    )
    return {
        "decision": decision,
        "trigger_detail": detail,
        "steps": [step("decide", "done", detail, decision=decision.value)],
    }


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


# --------------------------------------------------------------------------- #
# Nodes: the specialists
# --------------------------------------------------------------------------- #
async def start_generation_node(state: AgentState) -> Dict[str, Any]:
    """Fan-out point for the specialists.

    Exists so a conditional edge has one node to target while two plain edges
    spread the work — LangGraph then runs both specialists in the same
    superstep, drafting food and training concurrently rather than in sequence.
    """
    if state.get("error"):
        return {}
    return {"attempt": state.get("attempt", 0) + 1}


async def plan_meals_node(state: AgentState) -> Dict[str, Any]:
    """The nutritionist: the food half of the week."""
    if state.get("error"):
        return {}

    attempt = state.get("attempt", 1)
    label = (
        "Nutritionist is choosing your meals…"
        if attempt == 1
        else f"Nutritionist is revising the meals (attempt {attempt})…"
    )

    profile: ProfileInDB = state["profile"]
    prompt = build_nutritionist_prompt(
        profile=profile,
        targets=state["targets"],
        decision=state["decision"],
        snapshot=state.get("snapshot"),
        current_plan=state.get("active_plan"),
        trigger_detail=state.get("trigger_detail", ""),
        duration_days=PLAN_DURATION_DAYS,
    )
    for extra in (state.get("retry_feedback"), state.get("critique_feedback")):
        if extra:
            prompt += "\n\n---\n\n" + extra

    try:
        started = time.perf_counter()
        draft: MealPlanDraft = await get_structured_llm(MealPlanDraft).ainvoke(
            [
                SystemMessage(content=NUTRITIONIST_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        return {
            "meal_draft": draft,
            "steps": [
                step("plan_meals", "running", label, attempt=attempt),
                step(
                    "plan_meals",
                    "done",
                    f'Drafted "{draft.plan_title}" — {len(draft.days)} days of meals.',
                    duration_ms=int((time.perf_counter() - started) * 1000),
                ),
            ],
        }

    except LLMUnavailableError as exc:
        return {"error": str(exc), "steps": [step("plan_meals", "error", str(exc))]}

    except Exception as exc:
        logger.exception("Meal planning failed on attempt %s", attempt)
        return {
            "meal_draft": None,
            "retry_feedback": (
                "Your previous response could not be parsed into the required "
                "schema. Return valid structured output matching it exactly."
            ),
            "steps": [
                step(
                    "plan_meals",
                    "error",
                    f"Meal drafting failed: {type(exc).__name__}",
                )
            ],
        }


async def plan_training_node(state: AgentState) -> Dict[str, Any]:
    """The trainer: the movement half of the week.

    Runs without sight of the meal plan. The critic reconciles the two
    afterwards — serialising them to share context would double the latency to
    remove a class of conflict the critic already catches.
    """
    if state.get("error"):
        return {}

    # Training rarely causes a validation failure — the validator only inspects
    # food — so a retry reuses the existing draft rather than paying for it again.
    if state.get("training_draft") is not None and not state.get("critique_feedback"):
        return {}

    profile: ProfileInDB = state["profile"]
    prompt = build_trainer_prompt(
        profile=profile,
        decision=state["decision"],
        snapshot=state.get("snapshot"),
        trigger_detail=state.get("trigger_detail", ""),
        duration_days=PLAN_DURATION_DAYS,
    )
    if state.get("critique_feedback"):
        prompt += "\n\n---\n\n" + state["critique_feedback"]

    try:
        started = time.perf_counter()
        draft: TrainingPlanDraft = await get_structured_llm(
            TrainingPlanDraft
        ).ainvoke(
            [
                SystemMessage(content=TRAINER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        rest_days = sum(1 for d in draft.days if d.activity.duration_minutes == 0)
        return {
            "training_draft": draft,
            "steps": [
                step("plan_training", "running", "Trainer is building your week…"),
                step(
                    "plan_training",
                    "done",
                    f"Planned {len(draft.days)} days, {rest_days} of them rest.",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                ),
            ],
        }

    except LLMUnavailableError as exc:
        return {"error": str(exc), "steps": [step("plan_training", "error", str(exc))]}

    except Exception as exc:
        logger.exception("Training planning failed")
        return {
            "training_draft": None,
            "steps": [
                step(
                    "plan_training",
                    "error",
                    f"Training drafting failed: {type(exc).__name__}",
                )
            ],
        }


async def assemble_node(state: AgentState) -> Dict[str, Any]:
    """Zip the two halves into one plan."""
    if state.get("error"):
        return {}

    meals: Optional[MealPlanDraft] = state.get("meal_draft")
    training: Optional[TrainingPlanDraft] = state.get("training_draft")

    if meals is None:
        return {
            "generated_plan": None,
            "validation_errors": ["The nutritionist produced no meals."],
        }

    by_day = {d.day: d.activity for d in (training.days if training else [])}
    fallback = ActivityItem(
        activity_type="Rest",
        duration_minutes=0,
        intensity="low",
        description="Recovery day — the trainer didn't cover this one.",
        target_steps=6000,
    )

    daily_plans = [
        DailyPlan(
            day=day.day,
            theme=day.theme,
            meals=day.meals,
            # A missing day becomes rest rather than an exception: half a plan
            # the user can follow beats none at all.
            activity=by_day.get(day.day, fallback),
        )
        for day in meals.days
    ]

    reasoning = meals.reasoning
    if training is not None and training.reasoning:
        reasoning = f"{meals.reasoning} {training.reasoning}"

    plan = HealthPlan(
        plan_title=meals.plan_title,
        duration_days=len(daily_plans),
        agent_reasoning=reasoning,
        daily_plans=daily_plans,
    )

    uncovered = [d.day for d in meals.days if d.day not in by_day]
    message = f"Combined {len(daily_plans)} days of meals and training."
    if uncovered:
        message += f" Days {uncovered} had no session — set to rest."

    return {
        "generated_plan": plan,
        "validation_errors": [],
        "steps": [step("assemble", "done", message)],
    }


async def critique_node(state: AgentState) -> Dict[str, Any]:
    """Review the assembled week for problems only visible once combined.

    Advisory. `validate` runs afterwards and has the final say — a model that
    approves an unsafe plan must not be able to make it safe.
    """
    if state.get("error") or state.get("generated_plan") is None:
        return {"critique_feedback": ""}

    # One revision round. A critic allowed to keep asking would spend the user's
    # time on diminishing preferences.
    if state.get("critique_rounds", 0) >= MAX_CRITIQUE_ROUNDS:
        return {"critique_feedback": ""}

    try:
        critique: PlanCritique = await get_structured_llm(PlanCritique).ainvoke(
            [
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(
                    content=build_critic_prompt(
                        state["profile"],
                        state["generated_plan"],
                        state.get("snapshot"),
                    )
                ),
            ]
        )
    except Exception as exc:
        # The critic is a nicety. Losing it must not cost the user their plan.
        logger.warning("Critique failed, continuing without it: %s", exc)
        return {
            "critique_feedback": "",
            "steps": [
                step(
                    "critique",
                    "done",
                    "Review unavailable — continuing to the safety checks.",
                )
            ],
        }

    rounds = state.get("critique_rounds", 0) + 1

    if critique.approved or not critique.issues:
        return {
            "critique_result": critique,
            "critique_rounds": rounds,
            "critique_feedback": "",
            "steps": [
                step("critique", "running", "Reviewer is reading the week…"),
                step("critique", "done", critique.summary),
            ],
        }

    return {
        "critique_result": critique,
        "critique_rounds": rounds,
        "critique_feedback": build_critique_feedback(critique.issues),
        "steps": [
            step("critique", "running", "Reviewer is reading the week…"),
            step(
                "critique",
                "failed",
                f"{critique.summary} ({len(critique.issues)} issue(s) to fix)",
                errors=critique.issues,
            ),
        ],
    }


# --------------------------------------------------------------------------- #
# Node: validate
# --------------------------------------------------------------------------- #
async def validate_node(state: AgentState) -> Dict[str, Any]:
    """Check the assembled plan against safety and constraint rules."""
    if state.get("error"):
        return {}

    plan = state.get("generated_plan")
    if plan is None:
        # Drafting already failed; the retry edge decides what happens next.
        return {}

    result = validate_plan(plan, state["profile"], state["targets"])

    if result.is_valid:
        message = "Plan passed all checks."
        if result.warnings:
            message += f" ({len(result.warnings)} minor note(s).)"
        return {
            "validation_errors": [],
            "validation_warnings": result.warnings,
            "steps": [
                step("validate", "running", "Checking the plan is safe and on-diet…"),
                step("validate", "done", message),
            ],
        }

    extra = f" (+{len(result.errors) - 1} more)" if len(result.errors) > 1 else ""
    return {
        "validation_errors": result.errors,
        "validation_warnings": result.warnings,
        "retry_feedback": build_retry_feedback(result),
        "steps": [
            step("validate", "running", "Checking the plan is safe and on-diet…"),
            step(
                "validate",
                "failed",
                f"Rejected: {result.errors[0]}{extra}",
                errors=result.errors,
            ),
        ],
    }


# --------------------------------------------------------------------------- #
# Node: persist
# --------------------------------------------------------------------------- #
async def persist_node(state: AgentState) -> Dict[str, Any]:
    """Save the validated plan as a new version and record the decision."""
    if state.get("error"):
        return {}

    generated: HealthPlan = state["generated_plan"]
    parent = state.get("active_plan")

    saved = await PlanRepository.save_new_version(
        PlanInDB(
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
    )

    await _record_event(state, resulting_plan_id=str(saved.id))

    return {
        "saved_plan": saved,
        "steps": [
            step("persist", "running", "Saving your plan…"),
            step(
                "persist",
                "done",
                f'Saved "{saved.plan_title}" as version {saved.version}.',
                plan_id=str(saved.id),
                version=saved.version,
            ),
        ],
    }


# --------------------------------------------------------------------------- #
# Node: record (terminal, for runs that produce no plan)
# --------------------------------------------------------------------------- #
async def record_node(state: AgentState) -> Dict[str, Any]:
    """Write the decision to the timeline even when nothing changed.

    Recording no-ops matters: it's the difference between "the agent decided
    you're fine" and "the agent didn't run".
    """
    if state.get("error"):
        await _record_event(state)
        return {"steps": [step("record", "error", state["error"])]}

    if state["decision"] != AgentDecision.NO_ACTION and not state.get("saved_plan"):
        message = (
            "Could not produce a valid plan after "
            f"{state.get('attempt', 0)} attempts. Your existing plan is unchanged."
        )
        await _record_event(state)
        return {"error": message, "steps": [step("record", "error", message)]}

    await _record_event(state)
    return {}


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


def route_after_critique(state: AgentState) -> str:
    """Send the plan back for revision, or on to the safety checks.

    Only the nutritionist is re-run unless the critic's findings concern the
    training too — regenerating both halves for a note about meal timing wastes
    a call and risks churning a training week that was fine.
    """
    if state.get("error"):
        return "validate"
    if state.get("critique_feedback"):
        return "revise"
    return "validate"


def route_after_validate(state: AgentState) -> str:
    """Persist, retry, or give up.

    The retry edge re-enters at `start_generation`, which bumps the attempt
    counter and fans out again. `plan_training` short-circuits on a retry, so in
    practice only the meals are redrawn — the validator inspects food, so food is
    what a rejection is about.
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
    workflow.add_node("start_generation", start_generation_node)
    workflow.add_node("plan_meals", plan_meals_node)
    workflow.add_node("plan_training", plan_training_node)
    workflow.add_node("assemble", assemble_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("persist", persist_node)
    workflow.add_node("record", record_node)

    workflow.set_entry_point("sense")

    workflow.add_edge("sense", "evaluate")
    workflow.add_edge("evaluate", "decide")

    workflow.add_conditional_edges(
        "decide",
        route_after_decide,
        {"generate": "start_generation", "record": "record"},
    )

    # Two edges out of one node is the fan-out: LangGraph runs both specialists
    # in the same superstep, then waits for both before `assemble`.
    workflow.add_edge("start_generation", "plan_meals")
    workflow.add_edge("start_generation", "plan_training")
    workflow.add_edge("plan_meals", "assemble")
    workflow.add_edge("plan_training", "assemble")

    workflow.add_edge("assemble", "critique")

    workflow.add_conditional_edges(
        "critique",
        route_after_critique,
        {"revise": "plan_meals", "validate": "validate"},
    )

    workflow.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "persist": "persist",
            "generate": "start_generation",
            "record": "record",
        },
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
# How far a recipe's summed weights may miss the meal's claimed macros.
# Generous, because the ingredient table is approximate and portion sizes are
# rounded to something a cook can actually measure.
RECIPE_MACRO_TOLERANCE = 0.30
MAX_RECIPE_ATTEMPTS = 2


async def generate_recipe(
    meal_name: str,
    description: str,
    calories: int,
    protein_g: int,
    profile: ProfileInDB,
) -> tuple[Recipe, RecipeAnalysis]:
    """Expand one planned meal into a full recipe, verified against its macros.

    The plan asserts a meal is 560 kcal and 48g protein. This sums what the
    recipe actually contains and checks the two agree — turning a claim into
    something computed. One correction attempt, then the recipe is returned
    regardless: a slightly-off recipe the user can cook beats an error page,
    and the analysis travels with it so the UI can be honest about the gap.
    """
    structured = get_structured_llm(Recipe)
    messages = [
        SystemMessage(content=RECIPE_SYSTEM_PROMPT),
        HumanMessage(
            content=build_recipe_prompt(
                meal_name, description, calories, protein_g, profile
            )
        ),
    ]

    recipe: Recipe = await structured.ainvoke(messages)
    analysis = analyse_recipe(recipe.ingredients)

    for attempt in range(2, MAX_RECIPE_ATTEMPTS + 1):
        if not analysis.is_reliable:
            # Too little of the dish is in our table to judge it fairly.
            logger.info(
                "Recipe for '%s' covers only %.0f%% known ingredients — "
                "skipping the macro check.",
                meal_name,
                analysis.coverage * 100,
            )
            break

        if _macros_agree(analysis, calories, protein_g):
            break

        logger.info(
            "Recipe for '%s' computes to %.0f kcal / %.0fg protein against a "
            "claim of %d / %d — requesting a correction (attempt %d).",
            meal_name,
            analysis.kcal,
            analysis.protein_g,
            calories,
            protein_g,
            attempt,
        )
        messages.append(
            HumanMessage(
                content=build_recipe_correction(
                    meal_name, calories, protein_g, analysis.kcal, analysis.protein_g
                )
            )
        )
        recipe = await structured.ainvoke(messages)
        analysis = analyse_recipe(recipe.ingredients)

    return recipe, analysis


def _macros_agree(
    analysis: RecipeAnalysis, calories: int, protein_g: int
) -> bool:
    """Do the summed weights land near what the meal claims?"""
    if calories <= 0:
        return True

    kcal_drift = abs(analysis.kcal - calories) / calories
    if kcal_drift > RECIPE_MACRO_TOLERANCE:
        return False

    if protein_g > 0:
        protein_drift = abs(analysis.protein_g - protein_g) / protein_g
        if protein_drift > RECIPE_MACRO_TOLERANCE:
            return False

    return True
