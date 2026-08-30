"""Scenario definitions for the agent evaluation suite.

A test asserts that code does what it does. An eval measures whether the agent
*decides well* across the situations it will actually meet, and, importantly,
records the cases it currently gets wrong instead of quietly omitting them.

Everything here is deterministic: the decision rules are plain Python, so the
suite needs no LLM, no database and no network, and produces the same numbers on
every run.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, List, Optional

from app.models.enums import AgentDecision, DietType, MealStatus, MealType
from app.models.plan import HealthPlan, NutritionTargets, PlanInDB
from app.models.profile import ProfileInDB
from app.services.adherence import build_snapshot
from tests.factories import (
    make_health_plan,
    make_log,
    make_plan_in_db,
    make_profile,
    make_targets,
)

TODAY = date(2026, 3, 15)
SLOTS = ["d1-breakfast", "d1-lunch", "d1-dinner", "d1-snack"]


# --------------------------------------------------------------------------- #
# Decision scenarios
# --------------------------------------------------------------------------- #
@dataclass
class DecisionScenario:
    """One situation the agent must classify correctly."""

    name: str
    situation: str
    expected: AgentDecision
    build: Callable[[], tuple]
    #: Why this case matters. Printed in the report so the suite is readable
    #: as documentation of the agent's intended behaviour.
    rationale: str = ""


def _case(plan, logs, today_log=None, targets=None, force=False):
    """Assemble the (state, snapshot, plan, targets) tuple `_choose_action` takes."""
    targets = targets or make_targets()
    snapshot = build_snapshot(
        target_date=TODAY,
        targets=targets,
        plan=plan,
        today_log=today_log,
        recent_logs=logs,
    )
    state = {"today": TODAY, "force_replan": force}
    return state, snapshot, plan, targets


def _no_plan():
    return _case(plan=None, logs=[])


def _on_track():
    targets = make_targets()
    plan = make_plan_in_db(targets, reference_date=TODAY)
    log = make_log(
        TODAY,
        [("d1-breakfast", MealStatus.EATEN), ("d1-lunch", MealStatus.EATEN)],
    )
    return _case(plan, [log], log, targets)


def _skip_with_meals_left():
    targets = make_targets()
    plan = make_plan_in_db(targets, reference_date=TODAY)
    log = make_log(
        TODAY,
        [("d1-breakfast", MealStatus.EATEN), ("d1-lunch", MealStatus.SKIPPED)],
    )
    return _case(plan, [log], log, targets)


def _skip_with_day_over():
    targets = make_targets()
    plan = make_plan_in_db(targets, reference_date=TODAY)
    log = make_log(
        TODAY,
        [
            ("d1-breakfast", MealStatus.EATEN),
            ("d1-lunch", MealStatus.EATEN),
            ("d1-dinner", MealStatus.EATEN),
            ("d1-snack", MealStatus.SKIPPED),
        ],
    )
    return _case(plan, [log], log, targets)


def _three_day_streak():
    targets = make_targets()
    plan = make_plan_in_db(targets, reference_date=TODAY, created_days_ago=3)
    logs = [
        make_log(TODAY, [("d4-lunch", MealStatus.SKIPPED)]),
        make_log(TODAY - timedelta(days=1), [("d3-lunch", MealStatus.SKIPPED)]),
        make_log(TODAY - timedelta(days=2), [("d2-lunch", MealStatus.SKIPPED)]),
    ]
    return _case(plan, logs, logs[0], targets)


def _low_adherence_large_sample():
    targets = make_targets()
    plan = make_plan_in_db(targets, reference_date=TODAY, created_days_ago=2)
    logs = [
        make_log(
            TODAY - timedelta(days=1),
            [
                ("d1-breakfast", MealStatus.SKIPPED),
                ("d1-lunch", MealStatus.SKIPPED),
                ("d1-dinner", MealStatus.SKIPPED),
                ("d1-snack", MealStatus.EATEN),
            ],
        ),
        make_log(
            TODAY - timedelta(days=3),
            [
                ("d2-breakfast", MealStatus.SKIPPED),
                ("d2-lunch", MealStatus.SKIPPED),
                ("d2-dinner", MealStatus.SKIPPED),
                ("d2-snack", MealStatus.EATEN),
            ],
        ),
        make_log(
            TODAY - timedelta(days=5),
            [
                ("d3-breakfast", MealStatus.SKIPPED),
                ("d3-lunch", MealStatus.EATEN),
            ],
        ),
    ]
    today_log = make_log(TODAY, [])
    return _case(plan, logs + [today_log], today_log, targets)


def _low_adherence_tiny_sample():
    """One skip out of two logged meals reads as 50%, but proves nothing."""
    targets = make_targets()
    plan = make_plan_in_db(targets, reference_date=TODAY)
    log = make_log(
        TODAY,
        [("d1-breakfast", MealStatus.EATEN), ("d1-lunch", MealStatus.SKIPPED)],
    )
    return _case(plan, [log], log, targets)


def _expired_plan():
    targets = make_targets()
    plan = make_plan_in_db(
        targets, reference_date=TODAY, created_days_ago=8, duration_days=7
    )
    log = make_log(TODAY, [])
    return _case(plan, [log], log, targets)


def _calorie_overage_with_meals_left():
    targets = make_targets(calories=2000, protein=170)
    plan = make_plan_in_db(targets, reference_date=TODAY)
    log = make_log(
        TODAY,
        [("d1-breakfast", MealStatus.EATEN), ("d1-lunch", MealStatus.EATEN)],
    )
    state, snapshot, plan, targets = _case(plan, [log], log, targets)
    snapshot.calories_consumed = 2600
    snapshot.calories_remaining = -600
    return state, snapshot, plan, targets


def _calorie_overage_day_over():
    targets = make_targets(calories=2000, protein=170)
    plan = make_plan_in_db(targets, reference_date=TODAY)
    log = make_log(
        TODAY,
        [
            ("d1-breakfast", MealStatus.EATEN),
            ("d1-lunch", MealStatus.EATEN),
            ("d1-dinner", MealStatus.EATEN),
            ("d1-snack", MealStatus.EATEN),
        ],
    )
    state, snapshot, plan, targets = _case(plan, [log], log, targets)
    snapshot.calories_consumed = 2600
    snapshot.calories_remaining = -600
    return state, snapshot, plan, targets


def _forced():
    targets = make_targets()
    plan = make_plan_in_db(targets, reference_date=TODAY)
    log = make_log(TODAY, [])
    return _case(plan, [log], log, targets, force=True)


def _streak_and_skip_together():
    targets = make_targets()
    plan = make_plan_in_db(targets, reference_date=TODAY, created_days_ago=3)
    logs = [
        make_log(TODAY, [("d4-lunch", MealStatus.SKIPPED)]),
        make_log(TODAY - timedelta(days=1), [("d3-lunch", MealStatus.SKIPPED)]),
        make_log(TODAY - timedelta(days=2), [("d2-lunch", MealStatus.SKIPPED)]),
    ]
    return _case(plan, logs, logs[0], targets)


def _no_logs_at_all():
    """A user who has never opened the app should not trigger a rebuild."""
    targets = make_targets()
    plan = make_plan_in_db(targets, reference_date=TODAY)
    return _case(plan, [], None, targets)


DECISION_SCENARIOS: List[DecisionScenario] = [
    DecisionScenario(
        name="cold_start",
        situation="Just onboarded, no plan exists",
        expected=AgentDecision.CREATE_INITIAL,
        build=_no_plan,
        rationale="Nothing to adapt yet. Build the first week.",
    ),
    DecisionScenario(
        name="on_track",
        situation="Two meals eaten, none skipped",
        expected=AgentDecision.NO_ACTION,
        build=_on_track,
        rationale="Intervening when nothing is wrong erodes trust in the agent.",
    ),
    DecisionScenario(
        name="never_logged",
        situation="Active plan, user has logged nothing",
        expected=AgentDecision.NO_ACTION,
        build=_no_logs_at_all,
        rationale="Silence is missing data, not non-adherence.",
    ),
    DecisionScenario(
        name="single_skip_recoverable",
        situation="Lunch skipped, dinner and snack still ahead",
        expected=AgentDecision.REBALANCE_DAY,
        build=_skip_with_meals_left,
        rationale="The day can still be salvaged by moving the budget forward.",
    ),
    DecisionScenario(
        name="single_skip_day_over",
        situation="Snack skipped, nothing left to eat",
        expected=AgentDecision.NO_ACTION,
        build=_skip_with_day_over,
        rationale="There is no remaining meal to rebalance into.",
    ),
    DecisionScenario(
        name="three_day_skip_streak",
        situation="Meals skipped three days running",
        expected=AgentDecision.STRUCTURAL_REPLAN,
        build=_three_day_streak,
        rationale="A pattern means the plan doesn't fit the user's life.",
    ),
    DecisionScenario(
        name="low_adherence_sufficient_evidence",
        situation="~30% adherence across 10 logged meals",
        expected=AgentDecision.STRUCTURAL_REPLAN,
        build=_low_adherence_large_sample,
        rationale="Enough evidence to conclude the plan itself is wrong.",
    ),
    DecisionScenario(
        name="low_adherence_insufficient_evidence",
        situation="1 skip out of 2 logged meals. 50%, but n=2",
        expected=AgentDecision.REBALANCE_DAY,
        build=_low_adherence_tiny_sample,
        rationale=(
            "Without a minimum sample the agent tears up a new plan on the "
            "user's first missed breakfast."
        ),
    ),
    DecisionScenario(
        name="plan_expired",
        situation="Day 8 of a 7-day plan",
        expected=AgentDecision.STRUCTURAL_REPLAN,
        build=_expired_plan,
        rationale="The block is finished; issue the next one.",
    ),
    DecisionScenario(
        name="calorie_overage_recoverable",
        situation="2,600 of 2,000 kcal with meals still to come",
        expected=AgentDecision.REBALANCE_DAY,
        build=_calorie_overage_with_meals_left,
        rationale="Lighten what's left rather than prescribing hunger.",
    ),
    DecisionScenario(
        name="calorie_overage_day_over",
        situation="2,600 of 2,000 kcal, every meal eaten",
        expected=AgentDecision.NO_ACTION,
        build=_calorie_overage_day_over,
        rationale="Nothing left to adjust; tomorrow's plan is unaffected.",
    ),
    DecisionScenario(
        name="user_forced_rebuild",
        situation="User pressed 'Rebuild from scratch'",
        expected=AgentDecision.STRUCTURAL_REPLAN,
        build=_forced,
        rationale="An explicit request outranks the agent's own judgement.",
    ),
    DecisionScenario(
        name="severity_ordering",
        situation="A skip today AND a three-day streak",
        expected=AgentDecision.STRUCTURAL_REPLAN,
        build=_streak_and_skip_together,
        rationale=(
            "Rules are ordered most-severe-first; the streak must win over the "
            "day rebalance."
        ),
    ),
]


# --------------------------------------------------------------------------- #
# Validator cases
# --------------------------------------------------------------------------- #
@dataclass
class ValidatorCase:
    """A plan the validator should accept or reject."""

    name: str
    should_reject: bool
    diet: DietType = DietType.VEGETARIAN
    allergies: List[str] = field(default_factory=list)
    mutate: Optional[Callable[[HealthPlan, NutritionTargets], None]] = None
    #: Documented misses. The suite reports these separately rather than
    #: pretending the detector is perfect.
    known_gap: bool = False
    note: str = ""


def _set_name(day: int, meal: int, name: str):
    def apply(plan: HealthPlan, _targets: NutritionTargets) -> None:
        plan.daily_plans[day].meals[meal].name = name

    return apply


def _set_description(day: int, meal: int, text: str):
    def apply(plan: HealthPlan, _targets: NutritionTargets) -> None:
        plan.daily_plans[day].meals[meal].description = text

    return apply


def _gut_protein(plan: HealthPlan, _targets: NutritionTargets) -> None:
    for meal in plan.daily_plans[0].meals:
        meal.protein_g = 5
        meal.carbs_g = round((meal.calories_kcal - 20 - meal.fat_g * 9) / 4)


def _halve_calories(plan: HealthPlan, _targets: NutritionTargets) -> None:
    for meal in plan.daily_plans[0].meals:
        meal.calories_kcal = round(meal.calories_kcal / 2)
        meal.protein_g = round(meal.protein_g / 2)
        meal.carbs_g = round(meal.carbs_g / 2)
        meal.fat_g = round(meal.fat_g / 2)


def _inflate_calories(plan: HealthPlan, _targets: NutritionTargets) -> None:
    for meal in plan.daily_plans[0].meals:
        meal.calories_kcal = round(meal.calories_kcal * 1.6)
        meal.carbs_g = round(meal.carbs_g * 1.6)


def _impossible_macros(plan: HealthPlan, _targets: NutritionTargets) -> None:
    meal = plan.daily_plans[0].meals[0]
    meal.calories_kcal = 200
    meal.protein_g = 50
    meal.carbs_g = 40
    meal.fat_g = 15


def _drop_a_meal(plan: HealthPlan, _targets: NutritionTargets) -> None:
    plan.daily_plans[0].meals.pop()


def _duplicate_ids(plan: HealthPlan, _targets: NutritionTargets) -> None:
    plan.daily_plans[1].meals[0].meal_id = plan.daily_plans[0].meals[0].meal_id



def _impossible_protein_density(plan: HealthPlan, _targets: NutritionTargets) -> None:
    """Macros that reconcile perfectly but describe an impossible food.

    200 kcal of pure protein is exactly 50g by the 4 kcal/g rule, so the
    reconciliation check is satisfied, but 0.25 g/kcal is beyond anything a
    vegetarian diet can reach.
    """
    meal = plan.daily_plans[0].meals[0]
    meal.calories_kcal = 200
    meal.protein_g = 50
    meal.carbs_g = 0
    meal.fat_g = 0


def _lean_meat_density(plan: HealthPlan, _targets: NutritionTargets) -> None:
    """Chicken-breast density. High, but entirely real."""
    meal = plan.daily_plans[0].meals[1]
    meal.name = "Grilled chicken breast with salad"
    meal.calories_kcal = 330
    meal.protein_g = 62
    meal.carbs_g = 8
    meal.fat_g = 7


VALIDATOR_CASES: List[ValidatorCase] = [
    ValidatorCase("clean_plan", should_reject=False),
    ValidatorCase(
        "meat_in_vegetarian_plan",
        should_reject=True,
        mutate=_set_name(2, 1, "Grilled chicken salad"),
    ),
    ValidatorCase(
        "egg_in_vegetarian_plan",
        should_reject=True,
        mutate=_set_name(0, 0, "Egg bhurji with toast"),
    ),
    ValidatorCase(
        "egg_in_eggetarian_plan_is_fine",
        should_reject=False,
        diet=DietType.EGGETARIAN,
        mutate=_set_name(0, 0, "Egg bhurji with toast"),
    ),
    ValidatorCase(
        "dairy_in_vegan_plan",
        should_reject=True,
        diet=DietType.VEGAN,
        mutate=_set_name(1, 0, "Paneer bhurji"),
    ),
    ValidatorCase(
        "root_vegetable_in_jain_plan",
        should_reject=True,
        diet=DietType.JAIN,
        mutate=_set_description(0, 1, "Served with onion and potato sabzi."),
    ),
    ValidatorCase(
        "pork_in_halal_plan",
        should_reject=True,
        diet=DietType.HALAL,
        mutate=_set_name(0, 2, "Bacon and eggs"),
    ),
    ValidatorCase(
        "meat_in_non_vegetarian_plan_is_fine",
        should_reject=False,
        diet=DietType.NON_VEGETARIAN,
        mutate=_set_name(0, 1, "Grilled chicken with quinoa"),
    ),
    ValidatorCase(
        "allergen_in_meal_name",
        should_reject=True,
        allergies=["peanut"],
        mutate=_set_name(0, 0, "Peanut butter toast"),
    ),
    ValidatorCase(
        "allergen_hidden_in_description",
        should_reject=True,
        allergies=["cashew"],
        mutate=_set_description(3, 2, "Finish with a sprinkle of cashew."),
    ),
    ValidatorCase(
        "substring_false_positive_hamper",
        should_reject=False,
        diet=DietType.HALAL,
        mutate=_set_description(0, 0, "Pack it in a tiffin hamper."),
        note="'ham' must not fire on 'hamper'.",
    ),
    ValidatorCase("protein_below_floor", should_reject=True, mutate=_gut_protein),
    ValidatorCase("calories_far_under", should_reject=True, mutate=_halve_calories),
    ValidatorCase("calories_far_over", should_reject=True, mutate=_inflate_calories),
    ValidatorCase(
        "macros_do_not_reconcile", should_reject=True, mutate=_impossible_macros
    ),
    ValidatorCase(
        "impossible_protein_density",
        should_reject=True,
        mutate=_impossible_protein_density,
        note=(
            "Macros reconcile exactly, so only the ingredient-derived density "
            "ceiling catches this one."
        ),
    ),
    ValidatorCase(
        "lean_meat_density_is_possible",
        should_reject=False,
        diet=DietType.NON_VEGETARIAN,
        mutate=_lean_meat_density,
        note=(
            "Guards the ceiling against false positives: 0.19 g/kcal is chicken "
            "breast, not a hallucination."
        ),
    ),
    ValidatorCase("wrong_meal_count", should_reject=True, mutate=_drop_a_meal),
    ValidatorCase("duplicate_meal_ids", should_reject=True, mutate=_duplicate_ids),
    # --- Documented gaps -------------------------------------------------- #
    ValidatorCase(
        "compound_word_dairy_for_vegan",
        should_reject=True,
        diet=DietType.VEGAN,
        mutate=_set_name(0, 0, "Banana milkshake"),
        known_gap=True,
        note=(
            "'milk' inside 'milkshake' is missed: the scan matches whole words "
            "plus a plural suffix, so compounds slip through. Tightening it "
            "risks the 'hamper' false positive, so the prompt constraint is the "
            "primary defence here."
        ),
    ),
    ValidatorCase(
        "brand_name_hides_meat",
        should_reject=True,
        mutate=_set_name(0, 1, "Classic McSpicy wrap"),
        known_gap=True,
        note=(
            "A keyword scan cannot know a brand name implies chicken. Catching "
            "this needs an ingredient-level lookup, not a word list."
        ),
    ),
]


def build_validator_plan(case: ValidatorCase) -> tuple[HealthPlan, ProfileInDB, NutritionTargets]:
    """Materialise a case into (plan, profile, targets)."""
    targets = make_targets()
    plan = make_health_plan(targets)
    profile = make_profile(diet_type=case.diet, allergies=case.allergies)

    if case.mutate is not None:
        case.mutate(plan, targets)

    return plan, profile, targets
