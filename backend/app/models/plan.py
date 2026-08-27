"""Plan models — the agent's structured output and its stored form.

Design note on recipes:
    A full 7-day plan with complete recipes for every meal is ~8k output tokens,
    which is slow and a reliability risk for structured generation. So the planner
    emits meals with names, macros and a one-line description, and full recipe
    detail is expanded lazily per-meal through a separate endpoint. Users open
    maybe three recipes a week — generating 28 up front is waste.
"""

import re
from datetime import datetime
from typing import Annotated, Any, List, Optional

from pydantic import BaseModel, BeforeValidator, Field

from app.models.common import MongoModel, utcnow
from app.models.enums import AgentDecision, MealStatus, MealType

# A legacy free-text ingredient line: "150g paneer, crumbled".
#
# The mass unit is REQUIRED for the number to count as grams. "1 capsicum" is a
# count, not one gram, and reading it as a gram would put a badly wrong figure
# into a feature whose entire point is not making numbers up. Without a unit the
# weight stays unknown and the analyser skips the line.
_LEGACY_INGREDIENT = re.compile(
    r"^\s*(?P<qty>[\d.]+)\s*(?P<unit>kg|g|ml|l)\b\s*(?P<rest>.+)$", re.IGNORECASE
)

# A leading amount with a non-mass unit: "1 tsp cumin seeds", "2 cups rice".
_COUNTED_INGREDIENT = re.compile(
    r"^\s*[\d./]+\s*"
    r"(?:tsp|teaspoons?|tbsp|tablespoons?|cups?|pinch(?:es)?|"
    r"cloves?|slices?|pieces?|nos?\.?|numbers?)?\s*"
    r"(?P<rest>.+)$",
    re.IGNORECASE,
)

_UNIT_TO_GRAMS = {"kg": 1000.0, "l": 1000.0, "g": 1.0, "ml": 1.0}


def _split_item(rest: str) -> tuple[str, Optional[str]]:
    item, _, preparation = rest.partition(",")
    return item.strip(), (preparation.strip() or None)


def _coerce_ingredient(value: Any) -> Any:
    """Accept a plain string where a structured ingredient is expected.

    Recipes generated before quantities were structured are stored as strings
    like "150g paneer, crumbled". Parsing them on read means old plans keep
    rendering instead of failing validation, and their macros become computable
    wherever the line actually carried a weight.
    """
    if not isinstance(value, str):
        return value

    weighed = _LEGACY_INGREDIENT.match(value)
    if weighed:
        item, preparation = _split_item(weighed.group("rest"))
        try:
            grams = float(weighed.group("qty")) * _UNIT_TO_GRAMS[
                weighed.group("unit").lower()
            ]
        except (ValueError, KeyError):
            grams = None
        return {"item": item, "quantity_g": grams, "preparation": preparation}

    counted = _COUNTED_INGREDIENT.match(value)
    rest = counted.group("rest") if counted else value
    item, preparation = _split_item(rest)

    return {"item": item or value.strip(), "quantity_g": None, "preparation": preparation}


class RecipeIngredient(BaseModel):
    """One ingredient with an explicit weight.

    Structured rather than free text so a meal's macros can be *computed* from
    what's actually in it, instead of only bounded by what's physically
    possible. `quantity_g` may be None for things that resist weighing —
    "a pinch of hing", "2 green chillies" — which the analyser then skips.
    """

    item: str = Field(
        description="The food itself, with no quantity or preparation, e.g. 'paneer'."
    )
    quantity_g: Optional[float] = Field(
        default=None,
        description=(
            "Weight in grams (use millilitres for liquids, 1ml = 1g). Null only "
            "for seasonings too small or awkward to weigh."
        ),
    )
    preparation: Optional[str] = Field(
        default=None, description="How it's prepared, e.g. 'finely chopped'."
    )

    def render(self) -> str:
        """Human-readable line, for the UI and for diet keyword scanning."""
        amount = f"{self.quantity_g:g}g " if self.quantity_g else ""
        suffix = f", {self.preparation}" if self.preparation else ""
        return f"{amount}{self.item}{suffix}"


class Recipe(BaseModel):
    """Full cooking detail for a meal. Generated on demand."""

    ingredients: List[
        Annotated[RecipeIngredient, BeforeValidator(_coerce_ingredient)]
    ] = Field(description="Every ingredient with its weight in grams.")
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


class ExercisePrescription(BaseModel):
    """One movement, with enough detail to actually perform it.

    "Upper body training, 45 min" is a category, not a prescription. A beginner
    cannot act on it and nothing downstream can check it. Sets, reps and rest
    make it followable; naming an exercise from the table makes it verifiable.
    """

    name: str = Field(
        description=(
            "Exact name from the provided exercise list. Do not invent "
            "exercises or rename them."
        )
    )
    sets: int = Field(description="Number of working sets.", ge=1, le=10)
    reps: str = Field(
        description=(
            "Reps or duration as text, since these differ in kind: '8-12', "
            "'30 seconds', '10 each side', '20 minutes'."
        )
    )
    rest_seconds: int = Field(
        default=90, description="Rest between sets. 0 for continuous work.", ge=0, le=300
    )
    cue: Optional[str] = Field(
        default=None,
        description=(
            "One short form cue. Left empty is fine — the table's own cue is "
            "used, which is more reliable than a generated one."
        ),
    )


class ActivityItem(BaseModel):
    """The day's movement prescription."""

    activity_type: str = Field(
        description="e.g. 'Strength training — full body', 'Easy cardio', 'Rest'."
    )
    duration_minutes: int = Field(description="Suggested duration; 0 for a rest day.")
    intensity: str = Field(description="One of: low, moderate, high.")
    description: str = Field(description="One sentence on the goal of this session.")
    target_steps: int = Field(default=8000, description="Daily step target.")

    # Empty on a rest day, and on nothing else.
    exercises: List[ExercisePrescription] = Field(default_factory=list)


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


# --------------------------------------------------------------------------- #
# Specialist outputs
#
# The plan is produced by three narrower agents rather than one. Splitting the
# schemas is what makes that possible: each specialist returns only its own
# slice, so its prompt carries only the constraints relevant to that slice and
# its output stays small enough to generate reliably.
# --------------------------------------------------------------------------- #


class MealDraftItem(BaseModel):
    """One meal, as the nutritionist is asked to produce it.

    Deliberately narrower than `MealItem`. That model also carries `recipe`
    (generated lazily, per meal, on demand), and `status` / `logged_at`, which
    are runtime state set when the user logs a meal. None of the three are the
    planner's to invent, and every field in a structured-output schema is
    something the model has to reason about and the provider has to constrain.

    Dropping them, and `meal_id` — a deterministic string that code can build
    more reliably than a model can — takes the schema from 1142 tokens to a
    fraction of that. Small structured outputs are the reliable ones; the meal
    schema was the one place that principle was not being applied.
    """

    meal_type: MealType
    name: str = Field(description="Short dish name, e.g. 'Masala oats with peanuts'.")
    description: str = Field(
        description="One sentence on what it is and why it fits the user's goal."
    )
    calories_kcal: int = Field(description="Estimated calories for this meal.")
    protein_g: int = Field(description="Estimated protein in grams.")
    carbs_g: int = Field(description="Estimated carbohydrate in grams.")
    fat_g: int = Field(description="Estimated fat in grams.")

    @classmethod
    def from_meal_item(cls, meal: "MealItem") -> "MealDraftItem":
        """Narrow a stored meal back to its draft form.

        The inverse of `to_meal_item`, for anything that needs to hand an
        existing plan back to the nutritionist's schema.
        """
        return cls(
            meal_type=meal.meal_type,
            name=meal.name,
            description=meal.description,
            calories_kcal=meal.calories_kcal,
            protein_g=meal.protein_g,
            carbs_g=meal.carbs_g,
            fat_g=meal.fat_g,
        )

    def to_meal_item(self, day: int) -> "MealItem":
        """Widen to the stored model, assigning the id here rather than asking
        the model for one."""
        return MealItem(
            meal_id=f"d{day}-{MealType(self.meal_type).value}",
            meal_type=self.meal_type,
            name=self.name,
            description=self.description,
            calories_kcal=self.calories_kcal,
            protein_g=self.protein_g,
            carbs_g=self.carbs_g,
            fat_g=self.fat_g,
        )


class DayMeals(BaseModel):
    """One day's meals, without the training."""

    day: int = Field(description="Day number, starting at 1.")
    theme: Optional[str] = Field(
        default=None, description="Optional short theme, e.g. 'High protein, low prep'."
    )
    meals: List[MealDraftItem]

    def to_meal_items(self) -> List["MealItem"]:
        """The stored form of this day's meals, with ids assigned."""
        return [meal.to_meal_item(self.day) for meal in self.meals]


class MealPlanDraft(BaseModel):
    """The nutritionist's output."""

    plan_title: str = Field(
        description="Short motivational title, e.g. 'Week 1: Protein First'."
    )
    reasoning: str = Field(
        description=(
            "Two to three sentences on why these meals, referencing the user's "
            "diet type, goal and any recent adherence signals."
        )
    )
    days: List[DayMeals]


class ExerciseDraft(BaseModel):
    """One movement, as the trainer is asked to produce it.

    No `cue`: the table's form cue is filled in during assembly and is more
    reliable than a generated one, so asking for it produced 35 fields of
    `"cue": null` per week — pure output the model had to emit correctly for
    no benefit. In json_mode nothing constrains the output, and a long nested
    array is where a model drifts and stops closing its braces.
    """

    name: str = Field(
        description=(
            "Exact name from the provided exercise list. Do not invent "
            "exercises or rename them."
        )
    )
    sets: int = Field(description="Number of working sets.", ge=1, le=10)
    reps: str = Field(
        description=(
            "Reps or duration as text, since these differ in kind: '8-12', "
            "'30 seconds', '10 each side', '20 minutes'."
        )
    )
    rest_seconds: int = Field(
        default=90, description="Rest between sets. 0 for continuous work.", ge=0, le=300
    )

    def to_prescription(self) -> "ExercisePrescription":
        return ExercisePrescription(
            name=self.name,
            sets=self.sets,
            reps=self.reps,
            rest_seconds=self.rest_seconds,
        )


class ActivityDraft(BaseModel):
    """A day's session, as the trainer is asked to produce it.

    Narrower than `ActivityItem` for the same reason: `target_steps` has a
    sensible default and does not need generating seven times.
    """

    activity_type: str = Field(
        description="e.g. 'Strength training — full body', 'Easy cardio', 'Rest'."
    )
    duration_minutes: int = Field(description="Suggested duration; 0 for a rest day.")
    intensity: str = Field(description="One of: low, moderate, high.")
    description: str = Field(description="One sentence on the goal of this session.")
    exercises: List[ExerciseDraft] = Field(
        default_factory=list,
        description="Empty on a rest day, and on nothing else.",
    )

    def to_activity_item(self) -> "ActivityItem":
        return ActivityItem(
            activity_type=self.activity_type,
            duration_minutes=self.duration_minutes,
            intensity=self.intensity,
            description=self.description,
            exercises=[e.to_prescription() for e in self.exercises],
        )


class DayTraining(BaseModel):
    """One day's activity, without the food."""

    day: int = Field(description="Day number, starting at 1.")
    activity: ActivityDraft


class TrainingPlanDraft(BaseModel):
    """The trainer's output."""

    reasoning: str = Field(
        description=(
            "Two to three sentences on why this training week is shaped this "
            "way — the split, the progression, and where recovery sits."
        )
    )
    days: List[DayTraining]


class PlanCritique(BaseModel):
    """The critic's review of the assembled plan.

    Advisory only, and deliberately downstream of nothing: the deterministic
    validator still runs afterwards and has the final say. A model that approves
    an unsafe plan must not be able to make it safe.
    """

    approved: bool = Field(
        description="True if the plan is coherent and needs no revision."
    )
    issues: List[str] = Field(
        default_factory=list,
        description=(
            "Specific, actionable problems — a missing rest day, leg training "
            "the morning after a long run, elaborate cooking on the user's "
            "busiest days. Empty when approved."
        ),
    )
    summary: str = Field(
        description="One sentence the user could read, explaining the verdict."
    )


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
