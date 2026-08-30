"""A curated ingredient reference table.

Why this exists rather than a USDA API call: USDA FoodData Central has almost no
coverage of the food Kaya actually plans. There is no entry for rajma chawal,
poha, or a chapati as anyone in India would make one. A nutrition database that
doesn't know the cuisine is worse than none, because it produces confident wrong
numbers.

So this is a small, deliberately hand-checked table of the ingredients that show
up in these plans, at approximate per-100g values drawn from the Indian Food
Composition Tables and USDA standard reference entries.

**Scope, stated honestly.** These are reference values for two jobs:

1. Establishing a *physical ceiling* on protein density, so the validator can
   reject claims that are impossible for a given diet.
2. Grounding the planner's prompt in real foods with real numbers, so it picks
   from things that exist instead of inventing.

They are NOT a clinical database, and they are not precise enough to compute a
user's intake from. Values vary with variety, preparation and fat content;
treat them as the right order of magnitude, not the truth.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from app.models.enums import DietType


@dataclass(frozen=True)
class Ingredient:
    """Approximate composition per 100 g, as prepared for eating."""

    key: str
    name: str
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    #: Search terms that identify this ingredient in a meal name/description.
    aliases: tuple[str, ...] = ()
    #: Excluded from the protein-density ceiling. A supplement rather than a
    #: meal component, and including it would inflate the bound past anything
    #: a plate of food can reach.
    is_supplement: bool = False

    @property
    def protein_density(self) -> float:
        """Grams of protein per kilocalorie. The scale-free comparison."""
        return self.protein_g / self.kcal if self.kcal else 0.0


# --------------------------------------------------------------------------- #
# The table
#
# `forbidden_keywords` on DietType decides diet compatibility, so entries here
# carry no diet flags of their own. One source of truth for what a diet
# excludes, not two that can drift apart.
# --------------------------------------------------------------------------- #

_TABLE: List[Ingredient] = [
    # --- Pulses and legumes (cooked) ---
    Ingredient("toor_dal", "Toor dal", 121, 7.0, 20.0, 0.4, ("toor", "arhar", "tur dal")),
    Ingredient("moong_dal", "Moong dal", 105, 7.0, 19.0, 0.4, ("moong", "mung")),
    Ingredient("masoor_dal", "Masoor dal", 116, 9.0, 20.0, 0.4, ("masoor", "red lentil", "lentil")),
    Ingredient("chana_dal", "Chana dal", 164, 8.9, 27.0, 2.6, ("chana dal", "bengal gram")),
    Ingredient("rajma", "Rajma", 127, 8.7, 22.8, 0.5, ("rajma", "kidney bean")),
    Ingredient("chole", "Chole", 164, 8.9, 27.4, 2.6, ("chole", "chickpea", "chana", "garbanzo")),
    Ingredient("lobia", "Lobia", 116, 7.7, 20.8, 0.5, ("lobia", "black-eyed pea")),
    Ingredient("sprouts", "Moong sprouts", 30, 3.0, 5.9, 0.2, ("sprout",)),

    # --- Soy ---
    Ingredient("soya_chunks", "Soya chunks (dry)", 345, 52.0, 33.0, 0.5, ("soya chunk", "soy chunk", "nutrela", "meal maker")),
    Ingredient("tofu", "Tofu (firm)", 144, 17.3, 2.8, 8.7, ("tofu",)),
    Ingredient("tempeh", "Tempeh", 192, 20.3, 7.6, 10.8, ("tempeh",)),
    Ingredient("soy_milk", "Soy milk", 54, 3.3, 6.3, 1.8, ("soy milk", "soya milk")),

    # --- Dairy ---
    Ingredient("paneer", "Paneer (full fat)", 296, 18.3, 3.6, 22.8, ("paneer",)),
    Ingredient("paneer_low_fat", "Paneer (low fat)", 206, 24.0, 4.0, 11.0, ("low-fat paneer", "low fat paneer")),
    Ingredient("curd", "Curd", 60, 3.1, 4.7, 3.3, ("curd", "dahi", "yogurt", "yoghurt", "raita")),
    Ingredient("greek_yogurt", "Greek yogurt (low fat)", 59, 10.2, 3.6, 0.4, ("greek yogurt", "greek yoghurt", "hung curd")),
    Ingredient("milk", "Milk (toned)", 58, 3.2, 4.7, 3.0, ("milk",)),
    Ingredient("cheese", "Cheese (cheddar)", 403, 25.0, 1.3, 33.0, ("cheese",)),
    Ingredient("ghee", "Ghee", 900, 0.0, 0.0, 100.0, ("ghee",)),
    Ingredient("butter", "Butter", 717, 0.9, 0.1, 81.0, ("butter",)),
    Ingredient("whey", "Whey protein powder", 400, 80.0, 8.0, 5.0, ("whey", "protein powder", "protein shake"), is_supplement=True),

    # --- Egg ---
    Ingredient("egg", "Egg (whole)", 155, 12.6, 1.1, 10.6, ("egg", "omelette", "omelet", "bhurji")),
    Ingredient("egg_white", "Egg white", 52, 10.9, 0.7, 0.2, ("egg white",)),

    # --- Meat, fish, seafood ---
    Ingredient("chicken_breast", "Chicken breast", 165, 31.0, 0.0, 3.6, ("chicken breast", "chicken")),
    Ingredient("chicken_thigh", "Chicken thigh", 209, 26.0, 0.0, 10.9, ("chicken thigh",)),
    Ingredient("mutton", "Mutton", 143, 27.1, 0.0, 3.0, ("mutton", "goat", "lamb")),
    Ingredient("fish", "Fish (rohu)", 97, 16.6, 0.0, 1.4, ("rohu", "fish", "pomfret")),
    Ingredient("salmon", "Salmon", 208, 20.0, 0.0, 13.0, ("salmon",)),
    Ingredient("prawns", "Prawns", 99, 24.0, 0.2, 0.3, ("prawn", "shrimp")),
    Ingredient("tuna", "Tuna (in water)", 116, 25.5, 0.0, 0.8, ("tuna",)),

    # --- Grains and staples ---
    Ingredient("rice", "Rice (white, cooked)", 130, 2.7, 28.0, 0.3, ("rice", "chawal", "pulao", "biryani")),
    Ingredient("brown_rice", "Brown rice (cooked)", 112, 2.6, 24.0, 0.9, ("brown rice",)),
    Ingredient("roti", "Chapati", 250, 9.0, 46.0, 4.0, ("roti", "chapati", "phulka", "paratha")),
    Ingredient("atta", "Wheat flour", 341, 12.1, 69.0, 1.7, ("atta", "wheat flour")),
    Ingredient("oats", "Oats (dry)", 389, 16.9, 66.0, 6.9, ("oats", "oatmeal")),
    Ingredient("poha", "Poha (dry)", 346, 6.6, 77.0, 1.2, ("poha", "flattened rice")),
    Ingredient("quinoa", "Quinoa (cooked)", 120, 4.4, 21.0, 1.9, ("quinoa",)),
    Ingredient("bread", "Whole wheat bread", 247, 13.0, 41.0, 3.4, ("bread", "toast")),
    Ingredient("idli", "Idli", 132, 3.4, 27.0, 0.4, ("idli",)),
    Ingredient("dosa", "Dosa", 168, 3.9, 29.0, 3.7, ("dosa", "uttapam")),
    Ingredient("upma", "Upma", 130, 3.0, 20.0, 4.0, ("upma", "rava", "semolina", "suji")),
    Ingredient("pasta", "Pasta (cooked)", 131, 5.0, 25.0, 1.1, ("pasta", "noodle", "macaroni")),

    # --- Vegetables ---
    Ingredient("spinach", "Spinach", 23, 2.9, 3.6, 0.4, ("spinach", "palak")),
    Ingredient("potato", "Potato (boiled)", 87, 2.0, 20.0, 0.1, ("potato", "aloo")),
    Ingredient("cauliflower", "Cauliflower", 25, 1.9, 5.0, 0.3, ("cauliflower", "gobi")),
    Ingredient("okra", "Okra", 33, 1.9, 7.0, 0.2, ("okra", "bhindi")),
    Ingredient("eggplant", "Eggplant", 25, 1.0, 6.0, 0.2, ("eggplant", "brinjal", "baingan")),
    Ingredient("tomato", "Tomato", 18, 0.9, 3.9, 0.2, ("tomato",)),
    Ingredient("onion", "Onion", 40, 1.1, 9.3, 0.1, ("onion",)),
    Ingredient("carrot", "Carrot", 41, 0.9, 9.6, 0.2, ("carrot",)),
    Ingredient("peas", "Green peas", 81, 5.4, 14.5, 0.4, ("peas", "matar")),
    Ingredient("capsicum", "Capsicum", 31, 1.0, 6.0, 0.3, ("capsicum", "bell pepper")),
    Ingredient("cucumber", "Cucumber", 15, 0.7, 3.6, 0.1, ("cucumber", "kheera")),
    Ingredient("mushroom", "Mushroom", 22, 3.1, 3.3, 0.3, ("mushroom",)),
    Ingredient("bottle_gourd", "Bottle gourd", 14, 0.6, 3.4, 0.0, ("lauki", "bottle gourd", "doodhi")),
    Ingredient("cabbage", "Cabbage", 25, 1.3, 5.8, 0.1, ("cabbage", "patta gobi")),

    # --- Nuts and seeds ---
    Ingredient("almonds", "Almonds", 579, 21.2, 21.6, 49.9, ("almond", "badam")),
    Ingredient("peanuts", "Peanuts", 567, 25.8, 16.1, 49.2, ("peanut", "groundnut", "moongphali")),
    Ingredient("walnuts", "Walnuts", 654, 15.2, 13.7, 65.2, ("walnut", "akhrot")),
    Ingredient("cashew", "Cashew", 553, 18.2, 30.2, 43.9, ("cashew", "kaju")),
    Ingredient("chia", "Chia seeds", 486, 16.5, 42.0, 30.7, ("chia",)),
    Ingredient("flaxseed", "Flaxseed", 534, 18.3, 28.9, 42.2, ("flaxseed", "alsi")),
    Ingredient("peanut_butter", "Peanut butter", 588, 25.0, 20.0, 50.0, ("peanut butter",)),
    Ingredient("sesame", "Sesame seeds", 573, 17.7, 23.4, 49.7, ("sesame", "til")),

    # --- Fruit ---
    Ingredient("banana", "Banana", 89, 1.1, 22.8, 0.3, ("banana", "kela")),
    Ingredient("apple", "Apple", 52, 0.3, 13.8, 0.2, ("apple",)),
    Ingredient("orange", "Orange", 47, 0.9, 11.8, 0.1, ("orange", "mosambi")),
    Ingredient("papaya", "Papaya", 43, 0.5, 10.8, 0.3, ("papaya",)),
    Ingredient("mango", "Mango", 60, 0.8, 15.0, 0.4, ("mango", "aam")),

    # --- Fats ---
    Ingredient("oil", "Vegetable oil", 884, 0.0, 0.0, 100.0, ("oil",)),
    Ingredient("olive_oil", "Olive oil", 884, 0.0, 0.0, 100.0, ("olive oil",)),
    Ingredient("coconut_oil", "Coconut oil", 862, 0.0, 0.0, 100.0, ("coconut oil",)),
]

BY_KEY: Dict[str, Ingredient] = {item.key: item for item in _TABLE}


# --------------------------------------------------------------------------- #
# Diet compatibility
# --------------------------------------------------------------------------- #
def _mentions_forbidden(item: Ingredient, diet: DietType) -> bool:
    """Does this ingredient trip the diet's forbidden-keyword list?

    Reuses `DietType.forbidden_keywords` rather than tagging each row, so the
    two can't drift apart.
    """
    haystack = f"{item.key} {item.name} {' '.join(item.aliases)}".lower()
    return any(word in haystack for word in diet.forbidden_keywords)


def allowed_for(diet: DietType) -> List[Ingredient]:
    """Every table entry a given diet permits."""
    return [item for item in _TABLE if not _mentions_forbidden(item, diet)]


# --------------------------------------------------------------------------- #
# Protein-density ceiling
# --------------------------------------------------------------------------- #

# Headroom over the theoretical maximum, covering preparation differences and
# the approximate nature of the table. Deliberately generous: this bound exists
# to catch the impossible, and a false rejection costs a regeneration attempt.
CEILING_TOLERANCE = 1.15


def max_protein_density(diet: DietType) -> float:
    """The highest grams-of-protein-per-kcal this diet can physically reach.

    Computed from the table rather than hardcoded, so adding an ingredient
    updates the bound automatically. Supplements are excluded. Whey powder is
    80% protein and would push the ceiling past anything a plate of food can
    reach, defeating the check.
    """
    candidates = [
        item.protein_density
        for item in allowed_for(diet)
        if not item.is_supplement and item.kcal > 0
    ]
    return max(candidates) if candidates else 0.0


def protein_ceiling(diet: DietType) -> float:
    """The density above which a claim is treated as impossible."""
    return max_protein_density(diet) * CEILING_TOLERANCE


def is_protein_claim_possible(
    calories_kcal: int, protein_g: int, diet: DietType
) -> bool:
    """Could any real meal of this diet deliver this protein at these calories?

    This answers *possible*, not *likely*. A vegetarian meal genuinely can hit
    ~0.15 g/kcal if it is largely soya chunks, so an optimistic-but-achievable
    claim passes. Catching merely implausible composition would need
    ingredient-level quantities, and guessing at those produces false
    rejections, which cost a regeneration attempt on a plan that was fine.
    """
    if calories_kcal <= 0:
        return False
    return (protein_g / calories_kcal) <= protein_ceiling(diet)


# --------------------------------------------------------------------------- #
# Lookup and prompt grounding
# --------------------------------------------------------------------------- #
def match(name: str) -> Optional[Ingredient]:
    """Resolve a single ingredient name against the table.

    Longest alias wins: "low fat paneer" must beat plain "paneer", and
    "chicken breast" must beat "chicken", or the macros come out wrong.
    """
    lowered = name.lower().strip()
    if not lowered:
        return None

    best: Optional[Ingredient] = None
    best_len = 0

    for item in _TABLE:
        for term in (item.name.lower(), *item.aliases):
            if term in lowered and len(term) > best_len:
                best, best_len = item, len(term)

    return best


@dataclass(frozen=True)
class RecipeAnalysis:
    """Macros computed from an ingredient list, with an honesty score."""

    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    #: Fraction of the recipe's total weighed mass we could identify.
    coverage: float
    matched: tuple[str, ...]
    unmatched: tuple[str, ...]

    @property
    def is_reliable(self) -> bool:
        """Whether the totals are worth comparing against a claim.

        Below this, too much of the dish is unaccounted for and the computed
        figure would understate it. Reporting a mismatch then would blame the
        model for our table's gaps.
        """
        return self.coverage >= MIN_COVERAGE_FOR_COMPARISON


# Below this share of identified mass, the computed macros are treated as
# indicative only. The gap is our table's, not the recipe's.
MIN_COVERAGE_FOR_COMPARISON = 0.7


def analyse_recipe(ingredients) -> RecipeAnalysis:
    """Sum macros over a recipe's weighed ingredients.

    Ingredients without a weight (a pinch of hing, two green chillies) are
    skipped rather than guessed at. They contribute little, and inventing a
    mass would put made-up numbers into a feature whose whole point is not
    making numbers up.
    """
    kcal = protein = carbs = fat = 0.0
    weighed_mass = matched_mass = 0.0
    matched: List[str] = []
    unmatched: List[str] = []

    for entry in ingredients:
        grams = getattr(entry, "quantity_g", None)
        name = getattr(entry, "item", "") or ""

        if not grams or grams <= 0:
            continue

        weighed_mass += grams
        found = match(name)

        if found is None:
            unmatched.append(name)
            continue

        matched.append(name)
        matched_mass += grams
        scale = grams / 100.0
        kcal += found.kcal * scale
        protein += found.protein_g * scale
        carbs += found.carbs_g * scale
        fat += found.fat_g * scale

    coverage = (matched_mass / weighed_mass) if weighed_mass else 0.0

    return RecipeAnalysis(
        kcal=round(kcal, 1),
        protein_g=round(protein, 1),
        carbs_g=round(carbs, 1),
        fat_g=round(fat, 1),
        coverage=round(coverage, 3),
        matched=tuple(matched),
        unmatched=tuple(unmatched),
    )


def find_in_text(text: str) -> Set[str]:
    """Return the keys of ingredients mentioned in a piece of text."""
    lowered = text.lower()
    found = set()
    for item in _TABLE:
        terms = (item.name.lower(), *item.aliases)
        if any(term in lowered for term in terms):
            found.add(item.key)
    return found


def top_protein_sources(diet: DietType, limit: int = 10) -> List[Ingredient]:
    """The most protein-dense real foods available to this diet.

    Fed into the planning prompt so the model chooses from things that exist,
    with numbers that are approximately right, rather than inventing both.
    """
    candidates = [
        item
        for item in allowed_for(diet)
        if not item.is_supplement and item.protein_g >= 5
    ]
    candidates.sort(key=lambda i: i.protein_density, reverse=True)
    return candidates[:limit]


def protein_reference_block(diet: DietType, limit: int = 10) -> str:
    """Render the protein sources as a compact prompt table."""
    rows = top_protein_sources(diet, limit)
    if not rows:
        return ""

    lines = [
        "### Protein sources available on this diet (per 100g, approximate)",
        "",
        "| Food | kcal | Protein |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {item.name} | {round(item.kcal)} | {item.protein_g:g}g |" for item in rows
    )
    lines.append("")
    lines.append(
        "Hitting the protein target means building meals around these. Use real "
        "portions. The numbers above are per 100g."
    )
    return "\n".join(lines)
