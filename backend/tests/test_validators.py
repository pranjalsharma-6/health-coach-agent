"""Tests for plan validation.

These are the tests that prove the system doesn't blindly trust the LLM.
"""

from app.agent.validators import validate_plan
from app.models.enums import DietType, MealType
from tests.factories import make_health_plan, make_meal, make_profile, make_targets


class TestHappyPath:
    def test_well_formed_plan_passes(self):
        targets = make_targets()
        plan = make_health_plan(targets)
        result = validate_plan(plan, make_profile(), targets)

        assert result.is_valid, result.summary()


class TestDietCompliance:
    def test_meat_in_a_vegetarian_plan_is_rejected(self):
        targets = make_targets()
        plan = make_health_plan(targets)
        plan.daily_plans[2].meals[1].name = "Grilled chicken salad"

        result = validate_plan(plan, make_profile(diet_type=DietType.VEGETARIAN), targets)

        assert not result.is_valid
        assert any("chicken" in e for e in result.errors)

    def test_egg_is_rejected_for_vegetarian_but_allowed_for_eggetarian(self):
        targets = make_targets()

        veg_plan = make_health_plan(targets)
        veg_plan.daily_plans[0].meals[0].name = "Egg bhurji with toast"
        veg_result = validate_plan(
            veg_plan, make_profile(diet_type=DietType.VEGETARIAN), targets
        )
        assert not veg_result.is_valid

        egg_plan = make_health_plan(targets)
        egg_plan.daily_plans[0].meals[0].name = "Egg bhurji with toast"
        egg_result = validate_plan(
            egg_plan, make_profile(diet_type=DietType.EGGETARIAN), targets
        )
        assert egg_result.is_valid, egg_result.summary()

    def test_dairy_is_rejected_for_vegan(self):
        targets = make_targets()
        plan = make_health_plan(targets)
        plan.daily_plans[1].meals[0].name = "Paneer bhurji"

        result = validate_plan(plan, make_profile(diet_type=DietType.VEGAN), targets)

        assert not result.is_valid
        assert any("paneer" in e for e in result.errors)

    def test_root_vegetables_are_rejected_for_jain(self):
        targets = make_targets()
        plan = make_health_plan(targets)
        plan.daily_plans[0].meals[1].description = "Served with onion and potato sabzi."

        result = validate_plan(plan, make_profile(diet_type=DietType.JAIN), targets)

        assert not result.is_valid

    def test_word_boundary_prevents_false_positives(self):
        """'ham' must not fire on 'hamper' — a naive substring check would."""
        targets = make_targets()
        plan = make_health_plan(targets)
        plan.daily_plans[0].meals[0].description = "Pack it in a tiffin hamper."

        result = validate_plan(plan, make_profile(diet_type=DietType.HALAL), targets)

        assert result.is_valid, result.summary()

    def test_non_vegetarian_plan_accepts_meat(self):
        targets = make_targets()
        plan = make_health_plan(targets)
        plan.daily_plans[0].meals[1].name = "Grilled chicken with quinoa"

        result = validate_plan(
            plan, make_profile(diet_type=DietType.NON_VEGETARIAN), targets
        )
        assert result.is_valid, result.summary()


class TestAllergens:
    def test_declared_allergen_is_rejected(self):
        targets = make_targets()
        plan = make_health_plan(targets)
        plan.daily_plans[0].meals[0].name = "Peanut butter toast"

        result = validate_plan(plan, make_profile(allergies=["peanut"]), targets)

        assert not result.is_valid
        assert any("peanut" in e for e in result.errors)

    def test_allergen_hidden_in_description_is_caught(self):
        targets = make_targets()
        plan = make_health_plan(targets)
        plan.daily_plans[3].meals[2].description = "Finish with a sprinkle of cashew."

        result = validate_plan(plan, make_profile(allergies=["cashew"]), targets)

        assert not result.is_valid


class TestNutritionAccuracy:
    def test_day_far_under_calorie_target_is_rejected(self):
        targets = make_targets(calories=2000)
        plan = make_health_plan(targets)
        for meal in plan.daily_plans[0].meals:
            meal.calories_kcal = 100
            meal.protein_g = 8
            meal.carbs_g = 10
            meal.fat_g = 3

        result = validate_plan(plan, make_profile(), targets)

        assert not result.is_valid
        assert any("kcal" in e for e in result.errors)

    def test_day_under_protein_floor_is_rejected(self):
        targets = make_targets(calories=2000, protein=170)
        plan = make_health_plan(targets)
        # Keep calories right, gut the protein.
        for meal in plan.daily_plans[0].meals:
            meal.protein_g = 5
            meal.carbs_g = round((meal.calories_kcal - 20 - meal.fat_g * 9) / 4)

        result = validate_plan(plan, make_profile(), targets)

        assert not result.is_valid
        assert any("protein" in e for e in result.errors)

    def test_overshooting_protein_is_allowed(self):
        """More protein than target is fine — only shortfalls are failures."""
        targets = make_targets(calories=2000, protein=140)
        plan = make_health_plan(targets)

        result = validate_plan(plan, make_profile(), targets)
        assert result.is_valid, result.summary()

    def test_impossible_macros_are_rejected(self):
        """50g protein from a 200 kcal meal is not physically possible."""
        targets = make_targets()
        plan = make_health_plan(targets)
        plan.daily_plans[0].meals[0].calories_kcal = 200
        plan.daily_plans[0].meals[0].protein_g = 50
        plan.daily_plans[0].meals[0].carbs_g = 40
        plan.daily_plans[0].meals[0].fat_g = 15

        result = validate_plan(plan, make_profile(), targets)

        assert not result.is_valid
        assert any("inconsistent" in e for e in result.errors)


class TestStructure:
    def test_wrong_meal_count_is_rejected(self):
        targets = make_targets()
        plan = make_health_plan(targets, meals_per_day=4)
        plan.daily_plans[0].meals.pop()

        result = validate_plan(plan, make_profile(meals_per_day=4), targets)

        assert not result.is_valid
        assert any("meals" in e for e in result.errors)

    def test_duplicate_meal_ids_are_rejected(self):
        targets = make_targets()
        plan = make_health_plan(targets)
        plan.daily_plans[1].meals[0].meal_id = plan.daily_plans[0].meals[0].meal_id

        result = validate_plan(plan, make_profile(), targets)

        assert not result.is_valid
        assert any("Duplicate" in e for e in result.errors)

    def test_empty_plan_is_rejected(self):
        targets = make_targets()
        plan = make_health_plan(targets)
        plan.daily_plans = []

        result = validate_plan(plan, make_profile(), targets)
        assert not result.is_valid

    def test_oversized_single_meal_warns_without_blocking(self):
        targets = make_targets(calories=2000)
        plan = make_health_plan(targets)
        # Move most of the day into one meal, keeping the day's total correct.
        day = plan.daily_plans[0]
        day.meals[0] = make_meal("d1-breakfast", MealType.BREAKFAST, calories=1400, protein=100)
        remaining = 2000 - 1400
        for meal in day.meals[1:]:
            meal.calories_kcal = remaining // 3
            meal.protein_g = 25
            meal.fat_g = round(meal.calories_kcal * 0.25 / 9)
            meal.carbs_g = round(
                (meal.calories_kcal - 100 - meal.fat_g * 9) / 4
            )

        result = validate_plan(plan, make_profile(), targets)

        assert any("one sitting" in w for w in result.warnings)
