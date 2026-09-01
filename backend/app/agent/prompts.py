"""Prompt construction for the planning agent.

The design principle here: give the model *constraints and evidence*, never
arithmetic. Calorie and macro targets arrive pre-computed from
`services.nutrition`; the model's job is choosing food that hits them and
explaining itself, not deciding what the numbers should be.
"""

from typing import Dict, List, Optional, Sequence

from app.models.enums import AgentDecision, Cuisine, DietType, Goal, MealType
from app.models.log import AdherenceSnapshot
from app.models.plan import HealthPlan, MealItem, NutritionTargets, PlanInDB
from app.models.profile import ProfileInDB
from app.services.ingredients import protein_reference_block

# --------------------------------------------------------------------------- #
# Static guidance
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Constraint blocks
# --------------------------------------------------------------------------- #

_DIET_RULES: dict[DietType, str] = {
    DietType.VEGETARIAN: (
        "STRICTLY VEGETARIAN. No meat, poultry, fish, seafood, or eggs. Including "
        "hidden egg in mayonnaise, cakes, and some breads. Dairy IS allowed: paneer, "
        "curd, milk, ghee, and cheese are your primary protein anchors, alongside "
        "dal, rajma, chana, soya chunks, and tofu."
    ),
    DietType.EGGETARIAN: (
        "EGGETARIAN. Eggs and dairy ARE allowed. No meat, poultry, fish, or seafood. "
        "Eggs are the most efficient protein source available here. Use them "
        "generously at breakfast."
    ),
    DietType.VEGAN: (
        "STRICTLY VEGAN. No animal products of any kind: no meat, fish, eggs, dairy, "
        "honey, ghee, paneer, curd, or butter. Protein must come from tofu, tempeh, "
        "soya chunks, seitan, legumes, lentils, nuts, seeds, and fortified plant "
        "milks. Hitting the protein target here needs deliberate planning. Check "
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
        "NON-VEGETARIAN. All foods are permitted. Favour lean protein. Chicken "
        "breast, fish, eggs, lean mutton, and still include vegetarian meals for "
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
        "North Indian home cooking. Roti, sabzi, dal, rajma, chole, paneer dishes, "
        "curd, parathas. Use everyday Indian kitchen ingredients."
    ),
    Cuisine.SOUTH_INDIAN: (
        "South Indian home cooking. Idli, dosa, sambar, rasam, upma, poha, curd "
        "rice, coconut chutney, avial. Use everyday South Indian ingredients."
    ),
    Cuisine.CONTINENTAL: (
        "Continental / Western. Salads, grain bowls, grilled proteins, pasta, "
        "sandwiches, roasted vegetables."
    ),
    Cuisine.EAST_ASIAN: (
        "East Asian. Stir-fries, steamed dishes, noodle and rice bowls, tofu, "
        "miso, kimchi, light broths."
    ),
    Cuisine.MEDITERRANEAN: (
        "Mediterranean. Olive oil, legumes, whole grains, grilled fish and "
        "vegetables, yoghurt, hummus, salads."
    ),
    Cuisine.MIXED: (
        "Mixed cuisine. Vary across Indian and international dishes through the "
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


def _describe_cuisines(preferences) -> str:
    """Render one or more cuisine choices as a single instruction.

    Selecting several means "draw from all of these across the week", not
    "fuse them into one dish". A model handed a bare list will cheerfully
    invent a miso rajma, so the multi-cuisine phrasing says so explicitly.

    Values arrive as plain strings from Mongo (`use_enum_values=True`) and as
    enum members from freshly validated input, so coerce before indexing.
    """
    cuisines = [Cuisine(c) for c in (preferences or [])] or [Cuisine.MIXED]

    if len(cuisines) == 1:
        return _CUISINE_GUIDANCE[cuisines[0]]

    names = ", ".join(c.value.replace("_", " ") for c in cuisines)
    details = " ".join(_CUISINE_GUIDANCE[c] for c in cuisines)
    return (
        f"The user eats across several cuisines ({names}). Draw from all of "
        f"them over the week. Whole dishes from one tradition at a time, "
        f"never blended into a single dish. {details}"
    )


def build_constraints_block(profile: ProfileInDB) -> str:
    """Render the user's hard constraints as an unmissable prompt section."""
    diet = DietType(profile.diet_type)
    goal = Goal(profile.goal)

    lines = [
        "## HARD CONSTRAINTS. Violating any of these makes the plan unusable",
        "",
        f"**Diet type. {diet.label}:** {_DIET_RULES[diet]}",
        "",
        f"**Cuisine:** {_describe_cuisines(profile.cuisine_preferences)}",
        "",
        f"**Goal framing:** {_GOAL_FRAMING[goal]}",
        "",
    ]

    if profile.allergies:
        lines.append(
            f"**ALLERGIES. Must not appear anywhere:** {', '.join(profile.allergies)}"
        )
    if profile.disliked_foods:
        lines.append(f"**Dislikes. Avoid:** {', '.join(profile.disliked_foods)}")

    lines.extend(
        [
            f"**Meals per day:** exactly {profile.meals_per_day}",
            f"**Cooking skill:** {profile.cooking_skill}. Keep techniques within reach",
            f"**Max prep time:** {profile.max_prep_minutes} minutes per meal",
            f"**Budget:** {profile.budget_tier}",
            f"**Eats out ~{profile.eat_out_per_week}x/week**. Leave room for that",
        ]
    )

    if profile.medical_notes:
        lines.append(f"**Medical notes:** {profile.medical_notes}")

    # Ground the model in real foods with real numbers. Protein is the hardest
    # target to hit. Especially on vegetarian and vegan diets, and without a
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

    return f"""## DAILY TARGETS. Computed, not negotiable

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


def _build_evidence_block(snapshot: AdherenceSnapshot) -> str:
    """Render observed adherence. The reason a replan is happening."""
    lines = [
        "## OBSERVED BEHAVIOUR. This is why you are being asked to act",
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
            f"- **{snapshot.skip_streak_days} consecutive days with skipped meals**. "
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
    """The mode-specific instruction. What kind of plan to produce and why."""
    slots = _meal_slots(profile.meals_per_day)
    common = f"""
## OUTPUT REQUIREMENTS

- Produce exactly **{duration_days} days**, numbered 1 to {duration_days}.
- Each day has exactly **{profile.meals_per_day} meals** in this order: {slots}.
- `meal_id` must follow the pattern `d{{day}}-{{meal_type}}`, e.g. `d1-breakfast`. \
For a repeated slot, suffix an index: `d1-snack-2`.
- Do NOT include recipe steps. Only the dish name, a one-sentence description, \
and macros. Full recipes are generated separately, on demand.
- `agent_reasoning` must be 2-3 sentences explaining this specific plan for this \
specific person. Reference their diet type and, where relevant, what you observed \
in their behaviour. No generic filler.
"""

    if decision == AgentDecision.CREATE_INITIAL:
        task = """## TASK. Create the first plan

This user has just completed onboarding. Build their opening week.

Make it achievable rather than optimal. The first week's job is to establish the
habit, not to be the perfect diet. Favour simple, repeatable meals and start
activity at a level they will definitely complete."""

    elif decision == AgentDecision.REBALANCE_DAY:
        task = f"""## TASK. Rebalance around a missed meal

Trigger: {trigger_detail}

The user has missed a meal or overshot their intake today. Regenerate the plan so
the **remaining** meals absorb the difference.

- If they SKIPPED and are under target, redistribute the missing calories and
  protein into the meals still ahead of them. Do not try to claw back the entire
  deficit in one sitting. Cap any single meal at roughly 40% of the daily target.
- If they OVERSHOT, lighten the remaining meals, but never below a sensible
  minimum. Do not prescribe going hungry to punish a slip.
- Keep meals they have already eaten exactly as they were. Regenerate the full
  {duration_days}-day structure, but only the current day's remaining meals should
  differ meaningfully from before.
- In `agent_reasoning`, state plainly what you changed and why, addressed to the
  user. Be matter-of-fact, not disappointed."""

    else:  # STRUCTURAL_REPLAN
        task = f"""## TASK. Structural replan

Trigger: {trigger_detail}

This isn't one bad day. The user has repeatedly failed to follow the plan, which
means **the plan does not fit their life**. Do not reissue a similar plan with
more encouragement.

Change the structure:
- If breakfasts are consistently skipped, stop planning breakfast. Move to fewer,
  larger meals, or make breakfast something requiring zero preparation.
- If prep-heavy meals are being skipped, drop to assembly-only food.
- If evening meals fail, they may be eating out. Plan for restaurant-realistic
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
(use millilitres for liquids. Treat 1 ml as 1 g). This is not a formatting
preference: the weights are summed against a nutrition table to check the
recipe actually delivers the macros it claims, so a missing weight means the
check silently skips that ingredient.

- `item` is the food alone. "paneer", not "150g paneer, crumbled".
- Put preparation in `preparation`. "crumbled", "finely chopped".
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

The weights must actually add up to roughly the calories and protein above.
They are checked against a nutrition table, not taken on trust."""


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
everything up, that would overshoot the calories. Keep every quantity in grams."""


# --------------------------------------------------------------------------- #
# Specialist prompts
#
# One generalist prompt had to carry diet rules, macro targets, training
# programming and tone all at once. Splitting it lets each specialist see only
# what it needs, which keeps each prompt short and each output small, and small
# structured outputs are the reliable ones.
# --------------------------------------------------------------------------- #

NUTRITIONIST_SYSTEM_PROMPT = """You are Kaya's nutritionist. You plan food, and \
only food. Another specialist handles training, so never mention exercise.

## Non-negotiable rules

1. NEVER contradict the calorie and macro targets you are given. They were \
computed from validated equations with safety limits applied. Treat them as fixed.
2. NEVER suggest a food the user's diet type forbids. A plan the user cannot eat \
is worthless.
3. NEVER include an allergen the user has listed, not as a main ingredient, not \
as a garnish, not as an optional extra.
4. Meals must be REALISTIC for the user's cooking skill, prep time and budget.
5. Every meal's macros must be plausible for the portion described. Do not claim \
40g of protein from a bowl of rice.

## What makes a good week of food

- **Variety without complexity.** Repeat 2-3 staple breakfasts; nobody cooks \
seven different ones.
- **Front-load protein.** It's the hardest target to hit, especially on \
vegetarian and vegan diets. Spread it across every meal rather than loading \
dinner.
- **Real, named dishes.** "Rajma chawal with cucumber raita", not "protein \
source with complex carbohydrate".
- **Honest effort.** If the user has 15 minutes, plan 15-minute food.

Warm, direct, specific. Never lecture about willpower."""


TRAINER_SYSTEM_PROMPT = """You are Kaya's strength and conditioning coach. You \
plan training, and only training. Another specialist handles food, so never \
prescribe meals, calories or macros.

## Programming rules

1. **Match the goal.** Fat loss keeps resistance training to protect lean mass. \
Muscle gain needs progressive overload. Endurance needs volume with recovery.
2. **At least one genuine rest day** in seven. Two if the user is sedentary or \
their adherence is poor.
3. **Never stack conflicting sessions.** Heavy lower-body work should not sit the \
day after a long run, and two high-intensity days should not be consecutive.
4. **Start where the user actually is.** A sedentary beginner gets walking and \
bodyweight work, not a five-day split. A session they complete beats one they \
skip.
5. **Progress within the week**. Build intensity or volume across the block, \
then taper into the rest day.
6. **Respect the signals.** Low step counts or poor sleep mean less volume, not \
more discipline.

Set a realistic daily step target alongside each session. For a rest day, say \
so plainly and set duration to 0."""


CRITIC_SYSTEM_PROMPT = """You are Kaya's reviewer. Two specialists have each \
planned their half of a week. One the food, one the training, without seeing \
the other's work. Your job is to catch what neither could.

You are looking for problems that only appear when the halves are put together, \
or that span the whole week:

- Training and food pulling in opposite directions. A heavy session on the \
lowest-calorie day, or leg work the morning after a long run.
- No genuine rest day, or two hard days back to back.
- Elaborate cooking stacked on days the user is also training hardest.
- The week ignoring something visible in the user's recent behaviour.
- Meals or sessions that simply repeat with no progression across the block.

## How to judge

Approve unless something is genuinely wrong. A plan that is merely not what you \
would have written is fine. Churn costs the user a slower response and gains \
them nothing.

Do NOT re-check arithmetic. Calorie totals, macro reconciliation, diet \
compliance and allergens are verified separately by code that does not make \
mistakes. Flagging them here wastes a revision round.

When you reject, every issue must be specific and actionable: name the day and \
what to change. "Day 4 pairs the week's heaviest squat session with its lowest \
calorie day. Move the session to day 5" is useful. "Could be better balanced" \
is not."""


def build_today_block(
    planned_meals: Sequence[MealItem],
    statuses: Dict[str, str],
) -> str:
    """Meal by meal, what has already happened today.

    The rebalance instruction says to keep what was eaten and redistribute into
    what is still ahead. Without this block that instruction cannot be followed:
    the model was told only that one meal was eaten and two are pending, never
    which ones. It was rewriting the day blind and being asked to preserve
    something it could not see.
    """
    if not planned_meals:
        return ""

    lines = ["## TODAY SO FAR", ""]
    for meal in planned_meals:
        meal_type = getattr(meal.meal_type, "value", meal.meal_type)
        status = statuses.get(meal.meal_id, "planned")
        mark = {
            "eaten": "ALREADY EATEN. Keep this meal exactly as it is",
            "substituted": "ATE SOMETHING ELSE. Keep this slot as it is",
            "skipped": "SKIPPED. Its calories and protein need absorbing elsewhere",
        }.get(status, "STILL TO COME. This one you may change")
        lines.append(
            f"- **{meal_type}**: {meal.name} "
            f"({meal.calories_kcal} kcal, {meal.protein_g}g protein). {mark}."
        )

    lines.extend(
        [
            "",
            "Reproduce every ALREADY EATEN meal with the same name and the same "
            "numbers. They have happened and cannot be replanned. Change only "
            "the meals marked STILL TO COME.",
        ]
    )
    return "\n".join(lines)


def build_nutritionist_prompt(
    profile: ProfileInDB,
    targets: NutritionTargets,
    decision: AgentDecision,
    snapshot: Optional[AdherenceSnapshot] = None,
    current_plan: Optional[PlanInDB] = None,
    trigger_detail: str = "",
    duration_days: int = 7,
    today_block: str = "",
) -> str:
    """The food half of the plan."""
    sections = [
        build_constraints_block(profile),
        "",
        build_targets_block(targets, profile.meals_per_day),
        "",
        _build_user_block(profile),
        "",
    ]

    if snapshot is not None:
        sections.extend([_build_evidence_block(snapshot), ""])

    if today_block:
        sections.extend([today_block, ""])

    sections.append(
        _build_task_block(
            decision, profile, current_plan, trigger_detail, duration_days
        )
    )
    return "\n".join(sections)


def training_level_for(profile: ProfileInDB) -> "Level":
    """Infer a training level from what onboarding actually asks.

    There is no fitness-experience question yet, so activity level is the
    proxy. It errs low deliberately: prescribing a barbell deadlift to someone
    who has never lifted is a worse failure than prescribing goblet squats to
    someone who has.
    """
    from app.services.exercises import Level

    level = str(profile.activity_level)
    if level in {"very_active", "extremely_active"}:
        return Level.INTERMEDIATE
    return Level.BEGINNER


def build_exercise_block(profile: ProfileInDB) -> str:
    """The exercises the trainer may choose from, grouped by pattern.

    The counterpart to grounding the nutritionist in the ingredient table. A
    model asked for "an upper body session" invents plausible-sounding names;
    a model handed a list picks from movements that exist, at a difficulty the
    user can perform, with a form cue already written.
    """
    from app.models.enums import TrainingStyle
    from app.services.exercises import Pattern, allowed_for

    level = training_level_for(profile)
    styles = [TrainingStyle(s) for s in (profile.training_styles or [])]
    available = allowed_for(level, styles)

    chosen = ", ".join(s.label for s in styles) or "bodyweight"

    lines = [
        "## EXERCISE LIST. Choose only from these, by exact name",
        "",
        f"The user trains this way: **{chosen}**. Selected for a {level.value} "
        "trainee. Anything not on this list is unavailable, however well it "
        "would fit.",
        "",
    ]

    for pattern in Pattern:
        matching = [e for e in available if e.pattern is pattern]
        if not matching:
            continue
        names = ", ".join(
            f"{e.name} ({e.default_sets}x{e.default_reps})" for e in matching
        )
        lines.append(f"**{pattern.value}:** {names}")

    return "\n".join(lines)


def build_trainer_prompt(
    profile: ProfileInDB,
    decision: AgentDecision,
    snapshot: Optional[AdherenceSnapshot] = None,
    trigger_detail: str = "",
    duration_days: int = 7,
) -> str:
    """The training half of the plan.

    Deliberately not given the meal plan: the specialists run in parallel, and
    the critic reconciles them afterwards. Serialising them to share context
    would double the latency to remove a class of conflict the critic already
    catches.
    """
    goal = Goal(profile.goal)

    sections = [
        "## TRAINING BRIEF",
        "",
        _build_user_block(profile),
        f"- Activity level: {profile.activity_level}",
        f"- Goal framing: {_GOAL_FRAMING[goal]}",
        "",
    ]

    if profile.medical_notes:
        sections.extend(
            [f"**Medical notes. Programme around these:** {profile.medical_notes}", ""]
        )

    if snapshot is not None:
        sections.extend([_build_evidence_block(snapshot), ""])

    if trigger_detail:
        sections.extend([f"**Why you're being asked:** {trigger_detail}", ""])

    if decision == AgentDecision.STRUCTURAL_REPLAN:
        sections.append(
            "This is a restructure, not a refresh. If sessions were being missed, "
            "cut the volume rather than repeating it with more encouragement.\n"
        )

    sections.extend([build_exercise_block(profile), ""])

    sections.append(
        f"""## OUTPUT REQUIREMENTS

- Exactly **{duration_days} days**, numbered 1 to {duration_days}.
- One activity per day. Rest days are activities too. Name them plainly, set
  `duration_minutes` to 0 and leave `exercises` empty.
- **Every non-rest day must list its exercises**, each with sets, reps and rest.
  "Upper body training" on its own is not a plan anyone can follow.
- Use exact names from the exercise list. Do not invent or rename movements.
- 3 or 4 exercises for a strength session; 1 to 2 for a cardio or mobility
  day. Keep it tight. A long list is where output goes wrong.
- A strength session of three or more movements must train more than one
  pattern. Do not build a session entirely out of pushes.
- Keep one or two easy mobility days in the week even when the user picked a
  single style, and say so in the day's description. A week of nothing but
  hard sessions is how people get hurt and stop. Naming the exception is the
  point: silently ignoring what they asked for is worse than explaining it.
- `reasoning` is two to three sentences on the shape of the week: the split, the
  progression, and where recovery sits."""
    )
    return "\n".join(sections)


def build_critic_prompt(
    profile: ProfileInDB,
    plan: HealthPlan,
    snapshot: Optional[AdherenceSnapshot] = None,
) -> str:
    """Ask the reviewer to judge the assembled week."""
    lines = [
        "## THE USER",
        "",
        _build_user_block(profile),
        f"- Diet: {DietType(profile.diet_type).label}",
        f"- Cooking: {profile.cooking_skill}, up to {profile.max_prep_minutes} min a meal",
        f"- Eats out ~{profile.eat_out_per_week}x a week",
        "",
    ]

    if profile.medical_notes:
        lines.extend([f"- Medical notes: {profile.medical_notes}", ""])

    if snapshot is not None:
        lines.extend([_build_evidence_block(snapshot), ""])

    lines.extend([f"## THE PLAN. {plan.plan_title}", ""])

    for day in plan.daily_plans:
        total_kcal = sum(m.calories_kcal for m in day.meals)
        meals = "; ".join(f"{m.meal_type.value}: {m.name}" for m in day.meals)
        lines.append(
            f"**Day {day.day}** ({total_kcal} kcal). "
            f"{day.activity.activity_type}, {day.activity.duration_minutes} min "
            f"({day.activity.intensity}). {meals}"
        )

    lines.extend(
        [
            "",
            "Review the week as a whole. Approve it unless something is genuinely "
            "wrong; if you reject, name the day and the change in every issue.",
        ]
    )
    return "\n".join(lines)


def build_critique_feedback(issues: List[str]) -> str:
    """Turn the critic's findings into a revision instruction."""
    lines = [
        "A reviewer read the assembled week and asked for changes.",
        "",
    ]
    lines.extend(f"{i}. {issue}" for i, issue in enumerate(issues, start=1))
    lines.extend(
        [
            "",
            "Regenerate your half of the plan addressing every point that applies "
            "to it. Ignore any that concern the other specialist's work.",
        ]
    )
    return "\n".join(lines)


def _build_user_block(profile: ProfileInDB) -> str:
    """The one-line physical description both specialists need."""
    target = (
        f", targeting {profile.target_weight_kg}kg over "
        f"{profile.target_timeline_weeks} weeks"
        if profile.target_weight_kg
        else ""
    )
    return (
        f"- {profile.age_years}y {profile.gender}, {profile.height_cm}cm, "
        f"{profile.current_weight_kg}kg (BMI {profile.bmi})\n"
        f"- Goal: {profile.goal}{target}"
    )


def build_non_negotiables(profile: ProfileInDB) -> str:
    """The constraints that outrank any feedback, restated at the end.

    Appended after critique or validation feedback rather than relying on the
    copy at the top of the prompt. A reviewer asking for variety is a
    suggestion; the diet type is not, and a revision that satisfies the
    reviewer by breaking the diet is worse than no revision at all.
    """
    diet = DietType(profile.diet_type)

    lines = [
        "## THESE OVERRIDE EVERYTHING ABOVE",
        "",
        f"1. **Diet. {diet.label}.** {_DIET_RULES[diet]} No feedback, however "
        "reasonable, justifies breaking this.",
        f"2. **Exactly {profile.meals_per_day} meals on every single day.** "
        "Count them before you answer.",
        "3. **Only the days you were asked for**, numbered exactly as "
        "instructed. Not fewer, not more.",
    ]

    if profile.allergies:
        lines.append(
            f"4. **Allergies. Must not appear anywhere:** "
            f"{', '.join(profile.allergies)}"
        )

    return "\n".join(lines)
