"""Prompt construction for the planning agent.

The design principle here: give the model *constraints and evidence*, never
arithmetic. Calorie and macro targets arrive pre-computed from
`services.nutrition`; the model's job is choosing food that hits them and
explaining itself, not deciding what the numbers should be.
"""

from typing import Optional

from app.models.enums import AgentDecision, Cuisine, DietType, Goal, MealType
from app.models.log import AdherenceSnapshot
from app.models.plan import NutritionTargets, PlanInDB
from app.models.profile import ProfileInDB
from app.services.ingredients import protein_reference_block

# --------------------------------------------------------------------------- #
# Static guidance
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are Kaya, an expert nutrition and fitness coach.

You design practical, culturally appropriate meal and activity plans that people \
actually follow. You are not a medical professional and you never diagnose.

## Non-negotiable rules

1. NEVER contradict the calorie and macro targets you are given. They were computed \
from validated equations with safety limits applied. Treat them as fixed.
2. NEVER suggest a food the user's diet type forbids. This is the single most \
important constraint — a plan the user cannot eat is worthless.
3. NEVER include an allergen the user has listed. Not as a main ingredient, not as \
a garnish, not as an optional extra.
4. Meals must be REALISTIC for the user's cooking skill, prep time and budget.
5. Every meal's macros must be plausible for the portion described. Do not claim \
40g of protein from a bowl of rice.
6. If the user has medical notes, respect them and suggest professional consultation \
where warranted — but still produce the plan.

## What makes a good plan

- **Variety without complexity.** Repeat 2-3 staple breakfasts across the week; \
people don't cook seven different breakfasts.
- **Front-load protein.** Hitting the protein target is the hardest part of most \
plans, especially vegetarian ones. Distribute it across every meal rather than \
loading it into dinner.
- **Real, named dishes.** "Rajma chawal with a side of cucumber raita", not \
"protein source with complex carbohydrate".
- **Honest effort.** If the user has 15 minutes, plan 15-minute food.
- **Progressive activity.** Build intensity across the week and include at least \
one genuine rest day.

## Tone

Warm, direct, specific. You are a coach, not a chatbot. Never lecture about \
willpower. When someone slips, adjust the plan — don't moralise about it.
"""


# --------------------------------------------------------------------------- #
# Constraint blocks
# --------------------------------------------------------------------------- #

_DIET_RULES: dict[DietType, str] = {
    DietType.VEGETARIAN: (
        "STRICTLY VEGETARIAN. No meat, poultry, fish, seafood, or eggs — including "
        "hidden egg in mayonnaise, cakes, and some breads. Dairy IS allowed: paneer, "
        "curd, milk, ghee, and cheese are your primary protein anchors, alongside "
        "dal, rajma, chana, soya chunks, and tofu."
    ),
    DietType.EGGETARIAN: (
        "EGGETARIAN. Eggs and dairy ARE allowed. No meat, poultry, fish, or seafood. "
        "Eggs are the most efficient protein source available here — use them "
        "generously at breakfast."
    ),
    DietType.VEGAN: (
        "STRICTLY VEGAN. No animal products of any kind: no meat, fish, eggs, dairy, "
        "honey, ghee, paneer, curd, or butter. Protein must come from tofu, tempeh, "
        "soya chunks, seitan, legumes, lentils, nuts, seeds, and fortified plant "
        "milks. Hitting the protein target here needs deliberate planning — check "
        "every meal carries its share."
    ),
    DietType.JAIN: (
        "STRICTLY JAIN VEGETARIAN. No meat, fish, or eggs. Additionally NO ROOT "
        "VEGETABLES: no onion, garlic, potato, carrot, radish, beetroot, ginger, or "
        "turmeric root. Dairy IS allowed. Season with asafoetida (hing), cumin, "
        "coriander, and green chilli instead of onion and garlic. Protein comes from "
        "dairy, paneer, and permitted legumes."
    ),
    DietType.NON_VEGETARIAN: (
        "NON-VEGETARIAN. All foods are permitted. Favour lean protein — chicken "
        "breast, fish, eggs, lean mutton — and still include vegetarian meals for "
        "variety, cost, and fibre."
    ),
    DietType.HALAL: (
        "HALAL. No pork or pork derivatives (bacon, ham, lard, gelatin), no alcohol "
        "in cooking. All meat must be described as halal. Chicken, lamb, mutton, "
        "fish, eggs, and dairy are all permitted."
    ),
}

_CUISINE_GUIDANCE: dict[Cuisine, str] = {
    Cuisine.NORTH_INDIAN: (
        "North Indian home cooking — roti, sabzi, dal, rajma, chole, paneer dishes, "
        "curd, parathas. Use everyday Indian kitchen ingredients."
    ),
    Cuisine.SOUTH_INDIAN: (
        "South Indian home cooking — idli, dosa, sambar, rasam, upma, poha, curd "
        "rice, coconut chutney, avial. Use everyday South Indian ingredients."
    ),
    Cuisine.CONTINENTAL: (
        "Continental / Western — salads, grain bowls, grilled proteins, pasta, "
        "sandwiches, roasted vegetables."
    ),
    Cuisine.EAST_ASIAN: (
        "East Asian — stir-fries, steamed dishes, noodle and rice bowls, tofu, "
        "miso, kimchi, light broths."
    ),
    Cuisine.MEDITERRANEAN: (
        "Mediterranean — olive oil, legumes, whole grains, grilled fish and "
        "vegetables, yoghurt, hummus, salads."
    ),
    Cuisine.MIXED: (
        "Mixed cuisine — vary across Indian and international dishes through the "
        "week. Keep ingredients accessible."
    ),
}

_GOAL_FRAMING: dict[Goal, str] = {
    Goal.FAT_LOSS: (
        "Priority: preserve lean mass while in a deficit. Protein at every meal, "
        "high-volume low-calorie vegetables to manage hunger, and strength training "
        "to protect muscle."
    ),
    Goal.MUSCLE_GAIN: (
        "Priority: consistent calorie surplus with enough protein to build. "
        "Progressive resistance training, and calorie-dense foods so the user isn't "
        "uncomfortably full."
    ),
    Goal.MAINTENANCE: (
        "Priority: sustainability and nutrient quality. Steady energy, balanced "
        "macros, enjoyable food."
    ),
    Goal.ENDURANCE: (
        "Priority: fuelling training. Carbohydrate timed around sessions, adequate "
        "protein for recovery, and hydration."
    ),
    Goal.GENERAL_HEALTH: (
        "Priority: fibre, micronutrients, and variety. Whole foods, plenty of "
        "vegetables, minimal ultra-processed items."
    ),
}


def build_constraints_block(profile: ProfileInDB) -> str:
    """Render the user's hard constraints as an unmissable prompt section."""
    diet = DietType(profile.diet_type)
    cuisine = Cuisine(profile.cuisine_preference)
    goal = Goal(profile.goal)

    lines = [
        "## HARD CONSTRAINTS — violating any of these makes the plan unusable",
        "",
        f"**Diet type — {diet.label}:** {_DIET_RULES[diet]}",
        "",
        f"**Cuisine:** {_CUISINE_GUIDANCE[cuisine]}",
        "",
        f"**Goal framing:** {_GOAL_FRAMING[goal]}",
        "",
    ]

    if profile.allergies:
        lines.append(
            f"**ALLERGIES — must not appear anywhere:** {', '.join(profile.allergies)}"
        )
    if profile.disliked_foods:
        lines.append(f"**Dislikes — avoid:** {', '.join(profile.disliked_foods)}")

    lines.extend(
        [
            f"**Meals per day:** exactly {profile.meals_per_day}",
            f"**Cooking skill:** {profile.cooking_skill} — keep techniques within reach",
            f"**Max prep time:** {profile.max_prep_minutes} minutes per meal",
            f"**Budget:** {profile.budget_tier}",
            f"**Eats out ~{profile.eat_out_per_week}x/week** — leave room for that",
        ]
    )

    if profile.medical_notes:
        lines.append(f"**Medical notes:** {profile.medical_notes}")

    # Ground the model in real foods with real numbers. Protein is the hardest
    # target to hit — especially on vegetarian and vegan diets — and without a
    # reference the model tends to assert a figure rather than build a meal that
    # reaches it.
    reference = protein_reference_block(diet)
    if reference:
        lines.extend(["", reference])

    return "\n".join(lines)


def build_targets_block(targets: NutritionTargets, meals_per_day: int) -> str:
    """Render the pre-computed targets, with per-meal arithmetic done for the model."""
    per_meal_kcal = round(targets.calories_kcal / meals_per_day)
    per_meal_protein = round(targets.protein_g / meals_per_day)

    return f"""## DAILY TARGETS — computed, not negotiable

- Calories: **{targets.calories_kcal} kcal/day**
- Protein: **{targets.protein_g}g/day**
- Carbohydrate: **{targets.carbs_g}g/day**
- Fat: **{targets.fat_g}g/day**
- Water: {targets.water_ml} ml/day

Across {meals_per_day} meals that averages roughly **{per_meal_kcal} kcal** and \
**{per_meal_protein}g protein** per meal. Vary meal sizes sensibly (a snack is \
smaller than dinner), but each day's totals must land within 5% of the targets above."""


def _meal_slots(meals_per_day: int) -> str:
    """Name the meal slots so meal_id values come back predictable."""
    order = [MealType.BREAKFAST, MealType.LUNCH, MealType.DINNER, MealType.SNACK]
    if meals_per_day <= 3:
        slots = order[:meals_per_day]
    else:
        slots = order[:3] + [MealType.SNACK] * (meals_per_day - 3)

    return ", ".join(s.value for s in slots)


# --------------------------------------------------------------------------- #
# Mode-specific instructions
# --------------------------------------------------------------------------- #


def build_user_prompt(
    profile: ProfileInDB,
    targets: NutritionTargets,
    decision: AgentDecision,
    snapshot: Optional[AdherenceSnapshot] = None,
    current_plan: Optional[PlanInDB] = None,
    trigger_detail: str = "",
    duration_days: int = 7,
) -> str:
    """Assemble the full user-turn prompt for the requested planning mode."""
    sections = [
        build_constraints_block(profile),
        "",
        build_targets_block(targets, profile.meals_per_day),
        "",
        f"## USER",
        f"- {profile.age_years}y {profile.gender}, {profile.height_cm}cm, "
        f"{profile.current_weight_kg}kg (BMI {profile.bmi})",
        f"- Goal: {profile.goal}"
        + (
            f", targeting {profile.target_weight_kg}kg over "
            f"{profile.target_timeline_weeks} weeks"
            if profile.target_weight_kg
            else ""
        ),
        f"- Activity level: {profile.activity_level}",
        "",
    ]

    if snapshot is not None:
        sections.extend([_build_evidence_block(snapshot), ""])

    sections.append(_build_task_block(
        decision, profile, current_plan, trigger_detail, duration_days
    ))

    return "\n".join(sections)


def _build_evidence_block(snapshot: AdherenceSnapshot) -> str:
    """Render observed adherence — the reason a replan is happening."""
    lines = [
        "## OBSERVED BEHAVIOUR — this is why you are being asked to act",
        "",
        f"- Today: {snapshot.meals_eaten} eaten, {snapshot.meals_skipped} skipped, "
        f"{snapshot.meals_pending} still pending out of {snapshot.meals_planned} planned",
        f"- Calories: {snapshot.calories_consumed} of {snapshot.calories_target} kcal "
        f"({snapshot.calories_remaining} remaining)",
        f"- Protein: {snapshot.protein_consumed_g} of {snapshot.protein_target_g}g "
        f"({snapshot.protein_remaining_g}g remaining)",
        f"- 7-day adherence: {snapshot.adherence_rate_7d:.0%}",
        f"- Skips in the last 7 days: {snapshot.skips_last_7_days}",
    ]
    if snapshot.skip_streak_days > 1:
        lines.append(
            f"- **{snapshot.skip_streak_days} consecutive days with skipped meals** — "
            "this is a pattern, not a one-off. The plan is probably the problem."
        )
    if snapshot.steps is not None:
        lines.append(f"- Steps today: {snapshot.steps}")
    if snapshot.sleep_hours is not None:
        lines.append(f"- Sleep: {snapshot.sleep_hours}h")

    return "\n".join(lines)


def _build_task_block(
    decision: AgentDecision,
    profile: ProfileInDB,
    current_plan: Optional[PlanInDB],
    trigger_detail: str,
    duration_days: int,
) -> str:
    """The mode-specific instruction — what kind of plan to produce and why."""
    slots = _meal_slots(profile.meals_per_day)
    common = f"""
## OUTPUT REQUIREMENTS

- Produce exactly **{duration_days} days**, numbered 1 to {duration_days}.
- Each day has exactly **{profile.meals_per_day} meals** in this order: {slots}.
- `meal_id` must follow the pattern `d{{day}}-{{meal_type}}`, e.g. `d1-breakfast`. \
For a repeated slot, suffix an index: `d1-snack-2`.
- Do NOT include recipe steps — only the dish name, a one-sentence description, \
and macros. Full recipes are generated separately, on demand.
- `agent_reasoning` must be 2-3 sentences explaining this specific plan for this \
specific person. Reference their diet type and, where relevant, what you observed \
in their behaviour. No generic filler.
"""

    if decision == AgentDecision.CREATE_INITIAL:
        task = """## TASK — Create the first plan

This user has just completed onboarding. Build their opening week.

Make it achievable rather than optimal. The first week's job is to establish the
habit, not to be the perfect diet. Favour simple, repeatable meals and start
activity at a level they will definitely complete."""

    elif decision == AgentDecision.REBALANCE_DAY:
        task = f"""## TASK — Rebalance around a missed meal

Trigger: {trigger_detail}

The user has missed a meal or overshot their intake today. Regenerate the plan so
the **remaining** meals absorb the difference.

- If they SKIPPED and are under target, redistribute the missing calories and
  protein into the meals still ahead of them. Do not try to claw back the entire
  deficit in one sitting — cap any single meal at roughly 40% of the daily target.
- If they OVERSHOT, lighten the remaining meals, but never below a sensible
  minimum. Do not prescribe going hungry to punish a slip.
- Keep meals they have already eaten exactly as they were. Regenerate the full
  {duration_days}-day structure, but only the current day's remaining meals should
  differ meaningfully from before.
- In `agent_reasoning`, state plainly what you changed and why, addressed to the
  user. Be matter-of-fact, not disappointed."""

    else:  # STRUCTURAL_REPLAN
        task = f"""## TASK — Structural replan

Trigger: {trigger_detail}

This isn't one bad day. The user has repeatedly failed to follow the plan, which
means **the plan does not fit their life**. Do not reissue a similar plan with
more encouragement.

Change the structure:
- If breakfasts are consistently skipped, stop planning breakfast. Move to fewer,
  larger meals, or make breakfast something requiring zero preparation.
- If prep-heavy meals are being skipped, drop to assembly-only food.
- If evening meals fail, they may be eating out — plan for restaurant-realistic
  choices instead of pretending they'll cook.
- Reduce activity volume if sessions are being missed. A workout they complete
  beats one they skip.

In `agent_reasoning`, name the pattern you saw and the structural change you made.
Frame it as the plan adapting to them, never as them failing the plan."""

    if current_plan is not None:
        task += f"""

### Their current plan (for continuity)
Title: "{current_plan.plan_title}"
Original reasoning: {current_plan.agent_reasoning}
Keep what was working; change what wasn't."""

    return task + "\n" + common


# --------------------------------------------------------------------------- #
# Recipe expansion
# --------------------------------------------------------------------------- #

RECIPE_SYSTEM_PROMPT = """You are a practical home-cooking instructor.

Write recipes for real kitchens: common ingredients, ordinary equipment, and
steps a nervous beginner can follow. Never assume specialist equipment.

## Quantities

Every substantial ingredient MUST carry `quantity_g`, its weight in grams
(use millilitres for liquids — treat 1 ml as 1 g). This is not a formatting
preference: the weights are summed against a nutrition table to check the
recipe actually delivers the macros it claims, so a missing weight means the
check silently skips that ingredient.

- `item` is the food alone — "paneer", not "150g paneer, crumbled".
- Put preparation in `preparation` — "crumbled", "finely chopped".
- Leave `quantity_g` null ONLY for seasonings genuinely too small to weigh:
  a pinch of asafoetida, a couple of curry leaves. Never for a vegetable, a
  grain, a protein source, or any fat you cook in.
- Weigh things as they are eaten. If the recipe uses 60g of dry rice, say so
  in the steps, but list the ingredient at its cooked weight."""


def build_recipe_prompt(
    meal_name: str,
    description: str,
    calories: int,
    protein_g: int,
    profile: ProfileInDB,
) -> str:
    """Prompt for lazily expanding one meal into a full recipe."""
    diet = DietType(profile.diet_type)

    allergen_line = (
        f"\n- MUST NOT CONTAIN (allergies): {', '.join(profile.allergies)}"
        if profile.allergies
        else ""
    )

    return f"""Write a single-serving recipe for: **{meal_name}**

{description}

It must land near **{calories} kcal** and **{protein_g}g protein** for one portion.

Constraints:
- Diet: {diet.label}. {_DIET_RULES[diet]}{allergen_line}
- Maximum hands-on time: {profile.max_prep_minutes} minutes
- Cook's skill level: {profile.cooking_skill}
- Budget: {profile.budget_tier}

Give every ingredient a weight in grams, numbered steps of one sentence each,
honest prep time, and one useful swap or make-ahead tip.

The weights must actually add up to roughly the calories and protein above —
they are checked against a nutrition table, not taken on trust."""


def build_recipe_correction(
    meal_name: str,
    claimed_kcal: int,
    claimed_protein_g: int,
    computed_kcal: float,
    computed_protein_g: float,
) -> str:
    """Ask for a corrected recipe when the weights don't add up.

    Names both figures rather than saying "that was wrong": the model needs to
    know which direction and by how much to fix the portions.
    """
    return f"""Your previous recipe for **{meal_name}** was REJECTED.

Summing your ingredient weights against a nutrition table gives
**{computed_kcal:.0f} kcal** and **{computed_protein_g:.0f}g protein**, but the
meal is supposed to provide **{claimed_kcal} kcal** and **{claimed_protein_g}g
protein**.

Rewrite the recipe with portion sizes that genuinely reach those figures. If the
protein is short, increase the protein-dense ingredient rather than scaling
everything up — that would overshoot the calories. Keep every quantity in grams."""
