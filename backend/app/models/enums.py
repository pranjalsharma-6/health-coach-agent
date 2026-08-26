"""Domain enums.

These are deliberately explicit rather than free-text. Every value here is a hard
constraint the planner must respect and the validator can check against — which is
only possible if the vocabulary is closed.
"""

from enum import Enum


class DietType(str, Enum):
    """Dietary pattern.

    Modelled with more granularity than most Western apps offer, because
    "vegetarian" is not one thing and a plan the user won't eat has an
    adherence rate of zero.
    """

    VEGETARIAN = "vegetarian"           # no meat, fish, or egg; dairy ok
    EGGETARIAN = "eggetarian"           # vegetarian + eggs
    VEGAN = "vegan"                     # no animal products at all
    JAIN = "jain"                       # vegetarian + no root vegetables
    NON_VEGETARIAN = "non_vegetarian"   # everything
    HALAL = "halal"                     # non-veg, halal-compliant only

    @property
    def forbidden_keywords(self) -> list[str]:
        """Ingredient keywords that must never appear in a plan for this diet.

        Used by the output validator as a cheap, deterministic safety net over
        the LLM's generation. Not exhaustive — it catches the obvious failures.
        """
        meat = [
            "chicken", "mutton", "beef", "pork", "lamb", "bacon", "ham",
            "fish", "prawn", "shrimp", "salmon", "tuna", "crab", "meat",
        ]
        egg = ["egg", "omelette", "omelet", "mayonnaise"]
        dairy = ["milk", "paneer", "cheese", "yogurt", "curd", "butter", "ghee", "cream"]
        roots = ["potato", "onion", "garlic", "carrot", "radish", "beetroot", "ginger"]

        if self is DietType.VEGETARIAN:
            return meat + egg
        if self is DietType.EGGETARIAN:
            return meat
        if self is DietType.VEGAN:
            return meat + egg + dairy
        if self is DietType.JAIN:
            return meat + egg + roots
        if self is DietType.HALAL:
            return ["pork", "bacon", "ham", "lard", "alcohol", "wine", "beer"]
        return []

    @property
    def label(self) -> str:
        return {
            DietType.VEGETARIAN: "Vegetarian",
            DietType.EGGETARIAN: "Eggetarian",
            DietType.VEGAN: "Vegan",
            DietType.JAIN: "Jain",
            DietType.NON_VEGETARIAN: "Non-vegetarian",
            DietType.HALAL: "Halal",
        }[self]


class Goal(str, Enum):
    FAT_LOSS = "fat_loss"
    MUSCLE_GAIN = "muscle_gain"
    MAINTENANCE = "maintenance"
    ENDURANCE = "endurance"
    GENERAL_HEALTH = "general_health"


class Cuisine(str, Enum):
    NORTH_INDIAN = "north_indian"
    SOUTH_INDIAN = "south_indian"
    CONTINENTAL = "continental"
    EAST_ASIAN = "east_asian"
    MEDITERRANEAN = "mediterranean"
    MIXED = "mixed"


class ActivityLevel(str, Enum):
    SEDENTARY = "sedentary"
    LIGHTLY_ACTIVE = "lightly_active"
    MODERATELY_ACTIVE = "moderately_active"
    VERY_ACTIVE = "very_active"
    EXTREMELY_ACTIVE = "extremely_active"

    @property
    def multiplier(self) -> float:
        """Mifflin-St Jeor activity multiplier applied to BMR."""
        return {
            ActivityLevel.SEDENTARY: 1.2,
            ActivityLevel.LIGHTLY_ACTIVE: 1.375,
            ActivityLevel.MODERATELY_ACTIVE: 1.55,
            ActivityLevel.VERY_ACTIVE: 1.725,
            ActivityLevel.EXTREMELY_ACTIVE: 1.9,
        }[self]


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class CookingSkill(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class BudgetTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MealType(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class MealStatus(str, Enum):
    """Lifecycle of a single planned meal — the signal the agent senses on."""

    PLANNED = "planned"
    EATEN = "eaten"
    SKIPPED = "skipped"
    SUBSTITUTED = "substituted"


class AgentDecision(str, Enum):
    """The four actions the agent can take after evaluating progress.

    These are the conditional edges of the LangGraph state machine.
    """

    NO_ACTION = "no_action"                 # on track, leave the plan alone
    REBALANCE_DAY = "rebalance_day"         # redistribute today's remaining budget
    STRUCTURAL_REPLAN = "structural_replan" # regenerate the remaining days
    CREATE_INITIAL = "create_initial"       # no plan exists yet
