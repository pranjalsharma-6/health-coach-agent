"""The exercise table, and what makes a session usable.

The training side used to emit "Strength training. Upper body, 45 min" and
call it a plan. That is a category: a beginner cannot act on it, and no check
downstream can tell a good one from a bad one. The food side had a curated
table, computed macros and a validator; the training side had a sentence.
"""

import pytest

from app.agent.prompts import build_exercise_block, training_level_for
from app.agent.validators import validate_plan
from app.models.enums import ActivityLevel, DietType
from app.models.plan import ExercisePrescription
from app.services.exercises import (
    EXERCISES,
    Equipment,
    Level,
    Pattern,
    allowed_for,
    find,
    find_problems,
)
from tests.factories import (
    make_exercises,
    make_health_plan,
    make_profile,
    make_targets,
)


class TestTheTable:
    def test_every_exercise_carries_a_form_cue(self):
        """The cue is why a beginner can use this at all. An entry without one
        is the vague prescription in a different costume."""
        missing = [e.name for e in EXERCISES if not e.cue.strip()]
        assert not missing, f"no form cue for: {missing}"

    def test_every_pattern_has_a_bodyweight_option(self):
        """Someone training in a hostel room with no equipment must still get a
        session, not an apology."""
        for pattern in (Pattern.PUSH, Pattern.PULL, Pattern.SQUAT, Pattern.HINGE,
                        Pattern.CORE, Pattern.CARDIO):
            options = [
                e for e in EXERCISES
                if e.pattern is pattern and e.equipment is Equipment.NONE
            ]
            assert options, f"no equipment-free option for {pattern.value}"

    def test_every_pattern_has_a_beginner_option(self):
        for pattern in (Pattern.PUSH, Pattern.PULL, Pattern.SQUAT, Pattern.HINGE,
                        Pattern.CORE):
            options = [
                e for e in EXERCISES
                if e.pattern is pattern and e.level is Level.BEGINNER
            ]
            assert options, f"no beginner option for {pattern.value}"

    def test_names_are_unique(self):
        names = [e.name.lower() for e in EXERCISES]
        assert len(names) == len(set(names))

    def test_lookup_is_case_insensitive(self):
        assert find("push-ups") is find("Push-Ups")

    def test_levels_are_cumulative(self):
        """The basics do not stop working because someone got stronger."""
        beginner = set(e.name for e in allowed_for(Level.BEGINNER))
        advanced = set(e.name for e in allowed_for(Level.ADVANCED))
        assert beginner <= advanced


class TestSessionProblems:
    LEVEL = Level.BEGINNER

    def test_a_balanced_session_is_accepted(self):
        """Bodyweight only, because that is the default style."""
        assert find_problems(
            ["Push-ups", "Inverted row", "Bodyweight squat", "Plank"], self.LEVEL
        ) == []

    def test_an_invented_exercise_is_rejected(self):
        """The whole point of the table is that the model does not make these
        up. An unknown name has no cue and no known difficulty."""
        problems = find_problems(["Turbo shoulder blaster"], self.LEVEL)
        assert any("not in the exercise table" in p for p in problems)

    def test_a_style_the_user_did_not_choose_is_rejected(self):
        """A barbell squat for someone training in a bedroom."""
        problems = find_problems(["Barbell back squat"], self.LEVEL)
        assert any("full_gym" in p for p in problems)

    def test_the_same_movement_is_fine_for_someone_with_a_gym(self):
        """The old code excluded barbells outright, so a gym member was handed
        push-ups. Choosing full_gym is what unlocks them."""
        from app.models.enums import TrainingStyle

        assert find_problems(
            ["Barbell back squat"], Level.INTERMEDIATE, [TrainingStyle.FULL_GYM]
        ) == []

    def test_an_exercise_above_the_users_level_is_rejected(self):
        problems = find_problems(["Ab wheel rollout"], self.LEVEL)
        assert any("advanced" in p for p in problems)

    def test_a_session_of_one_pattern_is_rejected(self):
        """Three pushes is not a full-body session however it is labelled."""
        problems = find_problems(
            ["Push-ups", "Incline push-ups", "Dumbbell shoulder press"], self.LEVEL
        )
        assert any("pattern" in p for p in problems)

    def test_two_exercises_of_one_pattern_is_allowed(self):
        """A short accessory session is legitimate; the rule targets a session
        claiming to be complete, not any repetition at all."""
        assert find_problems(["Push-ups", "Incline push-ups"], self.LEVEL) == []

    def test_cardio_does_not_trip_the_pattern_rule(self):
        assert find_problems(["Brisk walk"], self.LEVEL) == []


class TestLevelInference:
    def test_a_sedentary_user_is_a_beginner(self):
        profile = make_profile(activity_level=ActivityLevel.SEDENTARY)
        assert training_level_for(profile) is Level.BEGINNER

    def test_a_very_active_user_gets_more(self):
        profile = make_profile(activity_level=ActivityLevel.VERY_ACTIVE)
        assert training_level_for(profile) is Level.INTERMEDIATE

    def test_nobody_is_assumed_advanced(self):
        """Erring low is deliberate: prescribing a barbell deadlift to someone
        who has never lifted is worse than prescribing goblet squats to
        someone who has."""
        for level in ActivityLevel:
            profile = make_profile(activity_level=level)
            assert training_level_for(profile) is not Level.ADVANCED


class TestPromptGrounding:
    def test_the_list_reaches_the_trainer(self):
        block = build_exercise_block(make_profile())
        assert "Push-ups" in block
        assert "push:" in block

    def test_it_withholds_what_the_user_cannot_do(self):
        block = build_exercise_block(
            make_profile(activity_level=ActivityLevel.SEDENTARY)
        )
        assert "Barbell deadlift" not in block, "barbell needs equipment"
        assert "Ab wheel rollout" not in block, "advanced for a sedentary user"


class TestPlanValidation:
    @staticmethod
    def _plan_and_profile():
        targets = make_targets()
        profile = make_profile(diet_type=DietType.VEGETARIAN)
        return make_health_plan(targets), profile, targets

    def test_a_well_formed_plan_passes(self):
        plan, profile, targets = self._plan_and_profile()
        assert validate_plan(plan, profile, targets).is_valid

    def test_a_session_with_no_exercises_is_rejected(self):
        """The shipped behaviour, and the reason for all of this."""
        plan, profile, targets = self._plan_and_profile()
        plan.daily_plans[0].activity.exercises = []

        result = validate_plan(plan, profile, targets)
        assert not result.is_valid
        assert any("no exercises" in e for e in result.errors)

    def test_an_invented_exercise_is_rejected(self):
        plan, profile, targets = self._plan_and_profile()
        plan.daily_plans[0].activity.exercises = [
            ExercisePrescription(name="Mega chest destroyer", sets=3, reps="10")
        ]

        result = validate_plan(plan, profile, targets)
        assert not result.is_valid

    def test_a_rest_day_needs_no_exercises(self):
        plan, profile, targets = self._plan_and_profile()
        plan.daily_plans[0].activity.activity_type = "Rest"
        plan.daily_plans[0].activity.duration_minutes = 0
        plan.daily_plans[0].activity.exercises = []

        assert validate_plan(plan, profile, targets).is_valid

    def test_a_rest_day_listing_exercises_only_warns(self):
        """Odd, not unsafe. A warning does not burn a generation attempt."""
        plan, profile, targets = self._plan_and_profile()
        plan.daily_plans[0].activity.activity_type = "Rest"
        plan.daily_plans[0].activity.duration_minutes = 0
        plan.daily_plans[0].activity.exercises = make_exercises()

        result = validate_plan(plan, profile, targets)
        assert result.is_valid
        assert result.warnings


class TestFormCueFallback:
    """The model may leave `cue` empty; the table's cue fills the gap.

    Without this the instruction "leave cue empty unless you have something to
    add" produces sessions with no cues at all, which is the vague
    prescription again, one level down.
    """

    @staticmethod
    async def _run_agent(monkeypatch):
        from datetime import date

        from app.agent import graph
        from app.models.enums import DietType
        from app.models.plan import ExerciseDraft, PlanCritique, TrainingPlanDraft
        from tests.factories import (  # noqa: F811
            make_critique,
            make_log,
            make_meal_draft,
            make_profile,
            make_targets,
            make_training_draft,
            scope_to_requested_days,
        )

        targets = make_targets()
        training = make_training_draft()
        for day in training.days:
            if day.activity.exercises:
                day.activity.exercises = [
                    ExerciseDraft(name="Push-ups", sets=3, reps="8-12"),
                    ExerciseDraft(name="Inverted row", sets=3, reps="10"),
                    ExerciseDraft(name="Bodyweight squat", sets=3, reps="12"),
                ]

        def factory(schema, **_budget):
            class Stub:
                async def ainvoke(self, messages):
                    if schema is TrainingPlanDraft:
                        return training
                    if schema is PlanCritique:
                        return make_critique()
                    return scope_to_requested_days(
                        make_meal_draft(targets), messages
                    )

            return Stub()

        class FakePlanRepo:
            @staticmethod
            async def get_active(_u):
                return None

            @staticmethod
            async def save_new_version(plan):
                plan.id, plan.version = "p1", 1
                return plan

        class FakeLogRepo:
            @staticmethod
            async def get_or_create(_u, d):
                return make_log(d, [])

            @staticmethod
            async def get_recent(_u, days=7):
                return []

        class FakeProfileRepo:
            @staticmethod
            async def get(_u):
                return make_profile(diet_type=DietType.VEGETARIAN)

        class FakeEventRepo:
            @staticmethod
            async def record(e):
                return e

        monkeypatch.setattr(graph, "PlanRepository", FakePlanRepo)
        monkeypatch.setattr(graph, "LogRepository", FakeLogRepo)
        monkeypatch.setattr(graph, "ProfileRepository", FakeProfileRepo)
        monkeypatch.setattr(graph, "AgentEventRepository", FakeEventRepo)
        monkeypatch.setattr(graph, "get_structured_llm", factory)

        return await graph.run_agent("u1", today=date(2026, 3, 15))

    async def test_a_missing_cue_is_filled_from_the_table(self, monkeypatch):
        final = await self._run_agent(monkeypatch)
        plan = final["saved_plan"]
        assert plan is not None

        session = next(
            d.activity for d in plan.daily_plans if d.activity.exercises
        )
        pushups = next(e for e in session.exercises if e.name == "Push-ups")
        assert pushups.cue
        assert "straight line" in pushups.cue

    async def test_every_exercise_in_a_shipped_plan_has_a_cue(self, monkeypatch):
        final = await self._run_agent(monkeypatch)
        plan = final["saved_plan"]

        for day in plan.daily_plans:
            for exercise in day.activity.exercises:
                assert exercise.cue, f"day {day.day}: {exercise.name} has no cue"

    async def test_the_model_is_not_asked_for_cues_at_all(self):
        """`ExerciseDraft` has no cue field.

        Asking for one produced 35 `"cue": null` fields per week. Output the
        model had to get right for no benefit, in the exact place its JSON was
        drifting.
        """
        from app.models.plan import ExerciseDraft

        assert "cue" not in ExerciseDraft.model_fields
