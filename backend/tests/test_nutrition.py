"""Tests for the deterministic nutrition engine.

These matter more than typical unit tests: this module produces the numbers a
user's health depends on, and it's the layer that keeps the LLM from being
trusted with them.
"""

import pytest

from app.models.enums import ActivityLevel, DietType, Gender, Goal
from app.models.profile import ProfileBase
from app.services import nutrition


def make_profile(**overrides) -> ProfileBase:
    defaults = dict(
        gender=Gender.MALE,
        age_years=30,
        height_cm=175,
        current_weight_kg=85.0,
        target_weight_kg=75.0,
        goal=Goal.FAT_LOSS,
        activity_level=ActivityLevel.MODERATELY_ACTIVE,
        target_timeline_weeks=12,
        diet_type=DietType.VEGETARIAN,
    )
    defaults.update(overrides)
    return ProfileBase(**defaults)


class TestBMR:
    def test_male_matches_mifflin_st_jeor(self):
        # 10(80) + 6.25(180) - 5(30) + 5 = 800 + 1125 - 150 + 5 = 1780
        assert nutrition.calculate_bmr(80, 180, 30, Gender.MALE) == pytest.approx(1780)

    def test_female_matches_mifflin_st_jeor(self):
        # 10(65) + 6.25(165) - 5(28) - 161 = 650 + 1031.25 - 140 - 161 = 1380.25
        assert nutrition.calculate_bmr(65, 165, 28, Gender.FEMALE) == pytest.approx(
            1380.25
        )

    def test_other_gender_sits_between_male_and_female(self):
        male = nutrition.calculate_bmr(70, 170, 30, Gender.MALE)
        female = nutrition.calculate_bmr(70, 170, 30, Gender.FEMALE)
        other = nutrition.calculate_bmr(70, 170, 30, Gender.OTHER)
        assert female < other < male


class TestSafetyFloors:
    def test_calorie_floor_blocks_starvation_targets(self):
        """A tiny, very sedentary user on an aggressive timeline must not be
        prescribed a dangerous target."""
        profile = make_profile(
            gender=Gender.FEMALE,
            age_years=60,
            height_cm=150,
            current_weight_kg=50.0,
            target_weight_kg=45.0,
            activity_level=ActivityLevel.SEDENTARY,
            target_timeline_weeks=4,
        )
        energy = nutrition.calculate_energy_profile(profile)

        assert energy.target_kcal >= nutrition.ABSOLUTE_MIN_KCAL_FEMALE
        assert energy.safety_floor_applied is True

    def test_deficit_is_clamped(self):
        """An absurd timeline produces a slow plan, not an unsafe one."""
        profile = make_profile(
            current_weight_kg=120.0, target_weight_kg=70.0, target_timeline_weeks=4
        )
        energy = nutrition.calculate_energy_profile(profile)

        assert abs(energy.deficit_or_surplus_kcal) <= nutrition.MAX_DAILY_DEFICIT_KCAL

    def test_surplus_is_clamped(self):
        profile = make_profile(goal=Goal.MUSCLE_GAIN)
        energy = nutrition.calculate_energy_profile(profile)

        assert 0 < energy.deficit_or_surplus_kcal <= nutrition.MAX_DAILY_SURPLUS_KCAL


class TestMacroTargets:
    def test_protein_scales_with_bodyweight_and_goal(self):
        profile = make_profile(goal=Goal.FAT_LOSS, current_weight_kg=80.0)
        targets = nutrition.calculate_targets(profile)

        expected = round(80.0 * nutrition.PROTEIN_G_PER_KG[Goal.FAT_LOSS])
        assert targets.protein_g == expected

    def test_fat_never_drops_below_essential_minimum(self):
        profile = make_profile(
            gender=Gender.FEMALE,
            height_cm=155,
            current_weight_kg=95.0,
            target_weight_kg=60.0,
            target_timeline_weeks=8,
        )
        targets = nutrition.calculate_targets(profile)

        assert targets.fat_g >= round(95.0 * nutrition.MIN_FAT_G_PER_KG)

    def test_macros_reconcile_with_calorie_target(self):
        """Macro grams must add back up to roughly the calorie target."""
        profile = make_profile()
        targets = nutrition.calculate_targets(profile)

        derived = (
            targets.protein_g * nutrition.KCAL_PER_G_PROTEIN
            + targets.carbs_g * nutrition.KCAL_PER_G_CARB
            + targets.fat_g * nutrition.KCAL_PER_G_FAT
        )
        # Rounding to whole grams costs a few kcal.
        assert derived == pytest.approx(targets.calories_kcal, abs=15)

    def test_carbs_never_go_negative(self):
        profile = make_profile(
            gender=Gender.FEMALE,
            height_cm=150,
            current_weight_kg=100.0,
            target_weight_kg=55.0,
            target_timeline_weeks=4,
            goal=Goal.FAT_LOSS,
        )
        targets = nutrition.calculate_targets(profile)
        assert targets.carbs_g >= 0

    def test_macro_split_sums_to_about_100(self):
        targets = nutrition.calculate_targets(make_profile())
        split = nutrition.macro_split_percent(targets)
        assert sum(split.values()) == pytest.approx(100, abs=2)


class TestProjections:
    def test_fat_loss_projects_negative_weekly_change(self):
        assert nutrition.estimated_weekly_change_kg(make_profile()) < 0

    def test_muscle_gain_projects_positive_weekly_change(self):
        profile = make_profile(goal=Goal.MUSCLE_GAIN)
        assert nutrition.estimated_weekly_change_kg(profile) > 0

    def test_maintenance_projects_no_change(self):
        profile = make_profile(goal=Goal.MAINTENANCE, target_weight_kg=None)
        assert nutrition.estimated_weekly_change_kg(profile) == pytest.approx(0, abs=0.1)


class TestDietConstraints:
    def test_vegetarian_forbids_meat_and_egg(self):
        forbidden = DietType.VEGETARIAN.forbidden_keywords
        assert "chicken" in forbidden
        assert "egg" in forbidden

    def test_eggetarian_allows_egg_but_not_meat(self):
        forbidden = DietType.EGGETARIAN.forbidden_keywords
        assert "egg" not in forbidden
        assert "chicken" in forbidden

    def test_vegan_forbids_dairy(self):
        assert "paneer" in DietType.VEGAN.forbidden_keywords

    def test_jain_forbids_root_vegetables(self):
        forbidden = DietType.JAIN.forbidden_keywords
        assert "onion" in forbidden
        assert "potato" in forbidden

    def test_non_vegetarian_has_no_restrictions(self):
        assert DietType.NON_VEGETARIAN.forbidden_keywords == []
