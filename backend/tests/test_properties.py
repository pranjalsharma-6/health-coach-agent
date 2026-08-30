"""Property-based tests over the nutrition engine and adherence evaluator.

The example-based tests in `test_nutrition.py` check cases someone thought of.
These check invariants across the *whole* valid input space. Every combination
of gender, age, height, weight, goal, activity level and timeline the API will
accept, which is the only way to make a real safety claim about the numbers.

Hypothesis shrinks any counterexample to a minimal failing profile, so a break
here arrives as a specific, reproducible input rather than a vague suspicion.
"""

from datetime import date, timedelta

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from app.models.enums import (
    ActivityLevel,
    DietType,
    Gender,
    Goal,
    MealStatus,
)
from app.models.profile import ProfileBase
from app.services import nutrition
from app.services.adherence import build_snapshot
from tests.factories import make_log, make_plan_in_db, make_targets

# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

# Bounds mirror the Pydantic constraints on ProfileBase. Generating outside
# them would only test Pydantic's validators, not our maths.
profiles = st.builds(
    ProfileBase,
    gender=st.sampled_from(list(Gender)),
    age_years=st.integers(min_value=13, max_value=100),
    height_cm=st.floats(min_value=100, max_value=250, allow_nan=False),
    current_weight_kg=st.floats(min_value=30, max_value=300, allow_nan=False),
    target_weight_kg=st.one_of(
        st.none(),
        st.floats(min_value=30, max_value=300, allow_nan=False),
    ),
    goal=st.sampled_from(list(Goal)),
    activity_level=st.sampled_from(list(ActivityLevel)),
    target_timeline_weeks=st.integers(min_value=4, max_value=52),
    diet_type=st.sampled_from(list(DietType)),
)

SETTINGS = settings(
    max_examples=400,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)


# --------------------------------------------------------------------------- #
# Safety invariants. These are the claims the README makes
# --------------------------------------------------------------------------- #
class TestSafetyInvariants:
    @given(profiles)
    @SETTINGS
    def test_calorie_target_never_below_the_floor(self, profile: ProfileBase):
        """No profile anywhere in the input space yields a starvation target."""
        energy = nutrition.calculate_energy_profile(profile)

        floor = (
            nutrition.ABSOLUTE_MIN_KCAL_FEMALE
            if Gender(profile.gender) == Gender.FEMALE
            else nutrition.ABSOLUTE_MIN_KCAL_MALE
        )
        assert energy.target_kcal >= floor

    @given(profiles)
    @SETTINGS
    def test_deficit_never_exceeds_the_safe_maximum(self, profile: ProfileBase):
        energy = nutrition.calculate_energy_profile(profile)

        # The floor may *raise* a target above TDEE-clamp territory; what must
        # never happen is a deficit deeper than the documented maximum.
        if energy.deficit_or_surplus_kcal < 0:
            assert (
                abs(energy.deficit_or_surplus_kcal)
                <= nutrition.MAX_DAILY_DEFICIT_KCAL
            )

    @given(profiles)
    @SETTINGS
    def test_surplus_beyond_the_clamp_only_comes_from_the_safety_floor(
        self, profile: ProfileBase
    ):
        """The clamp governs the *goal adjustment*, not the final delta.

        Hypothesis found the case that makes the distinction matter: a 30 kg,
        100 cm, sedentary 13-year-old has a TDEE of 938 kcal, so raising the
        target to the 1500 kcal floor leaves a 562 kcal surplus. Above the
        clamp, on a fat-loss goal. That is the floor doing its job, not the
        clamp failing. What must never happen is a surplus that large arising
        from the goal adjustment alone.
        """
        energy = nutrition.calculate_energy_profile(profile)

        if energy.deficit_or_surplus_kcal > nutrition.MAX_DAILY_SURPLUS_KCAL:
            assert energy.safety_floor_applied, (
                "a surplus above the clamp is only legitimate when the calorie "
                "floor raised the target"
            )

    @given(profiles)
    @SETTINGS
    def test_goal_adjustment_alone_respects_both_clamps(self, profile: ProfileBase):
        """The pre-floor adjustment is always inside the documented bounds."""
        energy = nutrition.calculate_energy_profile(profile)
        adjustment = nutrition._goal_adjustment(profile, energy.tdee_kcal)

        assert -nutrition.MAX_DAILY_DEFICIT_KCAL <= adjustment
        assert adjustment <= nutrition.MAX_DAILY_SURPLUS_KCAL

    @given(profiles)
    @SETTINGS
    def test_fat_never_below_the_essential_minimum(self, profile: ProfileBase):
        """Dietary fat has a hard physiological floor, whatever the target."""
        targets = nutrition.calculate_targets(profile)
        minimum = round(profile.current_weight_kg * nutrition.MIN_FAT_G_PER_KG)
        assert targets.fat_g >= minimum

    @given(profiles)
    @SETTINGS
    def test_protein_scales_with_weight_and_goal(self, profile: ProfileBase):
        targets = nutrition.calculate_targets(profile)
        expected = round(
            profile.current_weight_kg * nutrition.PROTEIN_G_PER_KG[Goal(profile.goal)]
        )
        assert targets.protein_g == expected

    @given(profiles)
    @SETTINGS
    def test_no_macro_is_ever_negative(self, profile: ProfileBase):
        targets = nutrition.calculate_targets(profile)
        assert targets.protein_g >= 0
        assert targets.carbs_g >= 0
        assert targets.fat_g >= 0
        assert targets.calories_kcal > 0


class TestArithmeticConsistency:
    @given(profiles)
    @SETTINGS
    def test_macros_reconcile_with_the_calorie_target(self, profile: ProfileBase):
        """Grams must add back up to the target. Within rounding.

        Carbohydrate absorbs the remainder, so this holds unless the protein and
        fat floors alone already exceed the target, which is a real (logged)
        edge case for a very heavy person on a very low target.
        """
        targets = nutrition.calculate_targets(profile)

        derived = (
            targets.protein_g * nutrition.KCAL_PER_G_PROTEIN
            + targets.carbs_g * nutrition.KCAL_PER_G_CARB
            + targets.fat_g * nutrition.KCAL_PER_G_FAT
        )

        floors_exceed_target = (
            targets.protein_g * nutrition.KCAL_PER_G_PROTEIN
            + targets.fat_g * nutrition.KCAL_PER_G_FAT
        ) >= targets.calories_kcal

        if floors_exceed_target:
            # Carbs are clamped at zero, so the total overshoots by design.
            assert targets.carbs_g == 0
            assert derived >= targets.calories_kcal
        else:
            assert abs(derived - targets.calories_kcal) <= 15

    @given(profiles)
    @SETTINGS
    def test_macro_split_always_sums_to_about_100(self, profile: ProfileBase):
        targets = nutrition.calculate_targets(profile)
        split = nutrition.macro_split_percent(targets)
        assert abs(sum(split.values()) - 100) <= 2

    @given(profiles)
    @SETTINGS
    def test_bmr_is_always_positive_and_below_tdee(self, profile: ProfileBase):
        energy = nutrition.calculate_energy_profile(profile)
        assert energy.bmr_kcal > 0
        # Every activity multiplier is > 1.
        assert energy.tdee_kcal >= energy.bmr_kcal

    @given(profiles)
    @SETTINGS
    def test_projection_sign_matches_the_energy_balance(self, profile: ProfileBase):
        energy = nutrition.calculate_energy_profile(profile)
        projection = nutrition.estimated_weekly_change_kg(profile)

        if energy.deficit_or_surplus_kcal < 0:
            assert projection <= 0
        elif energy.deficit_or_surplus_kcal > 0:
            assert projection >= 0


# --------------------------------------------------------------------------- #
# Adherence invariants
# --------------------------------------------------------------------------- #
TODAY = date(2026, 3, 15)

meal_statuses = st.lists(
    st.sampled_from(list(MealStatus)), min_size=0, max_size=4
)


class TestAdherenceInvariants:
    @given(statuses=meal_statuses)
    @SETTINGS
    def test_meal_counts_always_partition_the_plan(self, statuses):
        """eaten + skipped + pending must equal planned, always."""
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY)

        slots = ["d1-breakfast", "d1-lunch", "d1-dinner", "d1-snack"]
        log = make_log(TODAY, list(zip(slots, statuses)))

        snapshot = build_snapshot(
            target_date=TODAY,
            targets=targets,
            plan=plan,
            today_log=log,
            recent_logs=[log],
        )

        assert (
            snapshot.meals_eaten + snapshot.meals_skipped + snapshot.meals_pending
            == snapshot.meals_planned
        )
        assert snapshot.meals_eaten >= 0
        assert snapshot.meals_skipped >= 0
        assert snapshot.meals_pending >= 0

    @given(statuses=meal_statuses)
    @SETTINGS
    def test_adherence_rate_stays_within_bounds(self, statuses):
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY)
        slots = ["d1-breakfast", "d1-lunch", "d1-dinner", "d1-snack"]
        log = make_log(TODAY, list(zip(slots, statuses)))

        snapshot = build_snapshot(
            target_date=TODAY,
            targets=targets,
            plan=plan,
            today_log=log,
            recent_logs=[log],
        )

        assert 0.0 <= snapshot.adherence_rate_7d <= 1.0
        assert snapshot.meals_logged_7d >= 0
        assert snapshot.skip_streak_days >= 0

    @given(days=st.integers(min_value=0, max_value=14))
    @SETTINGS
    def test_skip_streak_never_exceeds_the_logged_window(self, days):
        """The streak can't claim more consecutive days than exist in the logs."""
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY, created_days_ago=days)

        logs = [
            make_log(TODAY - timedelta(days=n), [("d1-lunch", MealStatus.SKIPPED)])
            for n in range(days)
        ]

        snapshot = build_snapshot(
            target_date=TODAY,
            targets=targets,
            plan=plan,
            today_log=logs[0] if logs else None,
            recent_logs=logs,
        )
        assert snapshot.skip_streak_days <= days

    @given(
        consumed=st.integers(min_value=0, max_value=6000),
        target=st.integers(min_value=1200, max_value=4000),
    )
    @SETTINGS
    def test_remaining_is_always_target_minus_consumed(self, consumed, target):
        """Remaining may legitimately go negative. Eating over target is real."""
        assume(target > 0)
        targets = make_targets(calories=target, protein=int(target / 12))
        plan = make_plan_in_db(targets, reference_date=TODAY)
        log = make_log(TODAY, [])

        snapshot = build_snapshot(
            target_date=TODAY,
            targets=targets,
            plan=plan,
            today_log=log,
            recent_logs=[log],
        )
        assert (
            snapshot.calories_remaining
            == snapshot.calories_target - snapshot.calories_consumed
        )
