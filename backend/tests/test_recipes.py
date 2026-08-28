"""Tests for structured recipe quantities and macro verification."""

import pytest

from app.agent import graph
from app.models.enums import DietType
from app.models.plan import Recipe, RecipeIngredient
from app.services.ingredients import analyse_recipe
from tests.factories import make_profile


def make_recipe(ingredients, steps=("Cook it.",), prep=15, serves=1) -> Recipe:
    return Recipe(
        ingredients=ingredients, steps=list(steps), prep_minutes=prep, serves=serves
    )


class TestLegacyStringCoercion:
    """Recipes stored before quantities were structured must still load."""

    @pytest.mark.parametrize(
        "line,item,grams",
        [
            ("200g paneer, crumbled", "paneer", 200.0),
            ("150 ml milk", "milk", 150.0),
            ("1.5kg chicken breast", "chicken breast", 1500.0),
            ("60 g oats", "oats", 60.0),
        ],
    )
    def test_mass_units_are_parsed(self, line, item, grams):
        recipe = make_recipe([line])
        assert recipe.ingredients[0].item == item
        assert recipe.ingredients[0].quantity_g == grams

    @pytest.mark.parametrize(
        "line,item",
        [
            ("1 capsicum, finely chopped", "capsicum"),
            ("1 tsp cumin seeds", "cumin seeds"),
            ("2 tbsp coriander, chopped", "coriander"),
            ("3 cloves garlic, minced", "garlic"),
        ],
    )
    def test_counts_and_spoons_carry_no_weight(self, line, item):
        """"1 capsicum" is one vegetable, not one gram.

        Reading a bare count as grams would feed a wildly wrong number into a
        feature whose whole purpose is not making numbers up.
        """
        recipe = make_recipe([line])
        assert recipe.ingredients[0].item == item
        assert recipe.ingredients[0].quantity_g is None

    def test_preparation_is_split_out(self):
        recipe = make_recipe(["100g onion, finely sliced"])
        assert recipe.ingredients[0].preparation == "finely sliced"

    def test_unparseable_line_is_kept_verbatim(self):
        recipe = make_recipe(["a pinch of hing"])
        assert recipe.ingredients[0].item == "a pinch of hing"
        assert recipe.ingredients[0].quantity_g is None

    def test_structured_input_passes_through_untouched(self):
        recipe = make_recipe(
            [{"item": "paneer", "quantity_g": 150, "preparation": "cubed"}]
        )
        assert recipe.ingredients[0] == RecipeIngredient(
            item="paneer", quantity_g=150, preparation="cubed"
        )

    def test_render_round_trips_for_display(self):
        recipe = make_recipe(["200g paneer, crumbled"])
        assert recipe.ingredients[0].render() == "200g paneer, crumbled"


class TestRecipeAnalysis:
    def test_sums_macros_over_weighed_ingredients(self):
        recipe = make_recipe(
            [
                {"item": "paneer", "quantity_g": 100},
                {"item": "rice", "quantity_g": 100},
            ]
        )
        analysis = analyse_recipe(recipe.ingredients)

        # Paneer 296 + rice 130 per 100g.
        assert analysis.kcal == pytest.approx(426, abs=1)
        assert analysis.protein_g == pytest.approx(21.0, abs=0.5)
        assert analysis.coverage == 1.0
        assert analysis.is_reliable

    def test_unweighed_ingredients_are_skipped_not_guessed(self):
        recipe = make_recipe(
            [
                {"item": "paneer", "quantity_g": 100},
                {"item": "hing", "quantity_g": None},
            ]
        )
        analysis = analyse_recipe(recipe.ingredients)

        # The pinch contributes nothing and doesn't dent coverage either — it
        # was never weighed, so it isn't part of the denominator.
        assert analysis.kcal == pytest.approx(296, abs=1)
        assert analysis.coverage == 1.0

    def test_unknown_ingredients_reduce_coverage(self):
        recipe = make_recipe(
            [
                {"item": "paneer", "quantity_g": 100},
                {"item": "zorblax", "quantity_g": 100},
            ]
        )
        analysis = analyse_recipe(recipe.ingredients)

        assert analysis.coverage == pytest.approx(0.5)
        assert "zorblax" in analysis.unmatched
        assert not analysis.is_reliable

    def test_empty_recipe_is_not_reliable(self):
        analysis = analyse_recipe([])
        assert analysis.coverage == 0.0
        assert not analysis.is_reliable

    def test_longest_alias_wins(self):
        """'chicken breast' must not be scored as generic chicken."""
        lean = analyse_recipe(
            make_recipe([{"item": "chicken breast", "quantity_g": 100}]).ingredients
        )
        fatty = analyse_recipe(
            make_recipe([{"item": "chicken thigh", "quantity_g": 100}]).ingredients
        )
        assert lean.fat_g < fatty.fat_g


class TestMacroAgreement:
    def test_close_enough_passes(self):
        analysis = analyse_recipe(
            make_recipe([{"item": "paneer", "quantity_g": 100}]).ingredients
        )
        assert graph._macros_agree(analysis, calories=300, protein_g=18)

    def test_calorie_shortfall_fails(self):
        analysis = analyse_recipe(
            make_recipe([{"item": "paneer", "quantity_g": 100}]).ingredients
        )
        assert not graph._macros_agree(analysis, calories=600, protein_g=18)

    def test_protein_shortfall_fails(self):
        analysis = analyse_recipe(
            make_recipe([{"item": "rice", "quantity_g": 300}]).ingredients
        )
        # ~390 kcal is close to the claim, but 8g protein against 40g is not.
        assert not graph._macros_agree(analysis, calories=390, protein_g=40)

    def test_zero_calorie_claim_is_not_judged(self):
        analysis = analyse_recipe([])
        assert graph._macros_agree(analysis, calories=0, protein_g=0)


class TestGenerationVerification:
    """The retry loop that turns a claimed recipe into a checked one."""

    @staticmethod
    def _stub(monkeypatch, *recipes):
        queue = list(recipes)
        calls = {"n": 0}

        class Stub:
            async def ainvoke(self, _messages):
                calls["n"] += 1
                return queue.pop(0) if len(queue) > 1 else queue[0]

        monkeypatch.setattr(
            graph, "get_structured_llm", lambda _schema, **_budget: Stub()
        )
        return calls

    async def test_accurate_recipe_is_accepted_first_time(self, monkeypatch):
        good = make_recipe([{"item": "paneer", "quantity_g": 100}])
        calls = self._stub(monkeypatch, good)

        recipe, analysis = await graph.generate_recipe(
            "Paneer bhurji", "Scrambled paneer.", 300, 18, make_profile()
        )

        assert calls["n"] == 1
        assert analysis.is_reliable
        assert recipe is good

    async def test_short_recipe_triggers_one_correction(self, monkeypatch):
        thin = make_recipe([{"item": "rice", "quantity_g": 50}])       # ~65 kcal
        fixed = make_recipe([{"item": "paneer", "quantity_g": 100}])   # ~296 kcal
        calls = self._stub(monkeypatch, thin, fixed)

        recipe, analysis = await graph.generate_recipe(
            "Paneer bhurji", "Scrambled paneer.", 300, 18, make_profile()
        )

        assert calls["n"] == 2, "should have asked for a correction"
        assert recipe is fixed
        assert analysis.kcal == pytest.approx(296, abs=1)

    async def test_gives_up_after_the_attempt_budget(self, monkeypatch):
        """A wrong-but-cookable recipe beats an error page."""
        thin = make_recipe([{"item": "rice", "quantity_g": 50}])
        calls = self._stub(monkeypatch, thin)

        recipe, analysis = await graph.generate_recipe(
            "Paneer bhurji", "Scrambled paneer.", 600, 40, make_profile()
        )

        assert calls["n"] == graph.MAX_RECIPE_ATTEMPTS
        assert recipe is not None
        # The gap is reported rather than hidden.
        assert not graph._macros_agree(analysis, 600, 40)

    async def test_low_coverage_skips_the_check_rather_than_blaming_the_model(
        self, monkeypatch
    ):
        """When our table doesn't know the food, that's our gap, not the recipe's."""
        exotic = make_recipe(
            [
                {"item": "zorblax", "quantity_g": 200},
                {"item": "paneer", "quantity_g": 20},
            ]
        )
        calls = self._stub(monkeypatch, exotic)

        _recipe, analysis = await graph.generate_recipe(
            "Zorblax curry", "Exotic.", 600, 40, make_profile()
        )

        assert calls["n"] == 1, "must not retry over ingredients we can't identify"
        assert not analysis.is_reliable

    async def test_diet_appropriate_profile_is_passed_through(self, monkeypatch):
        good = make_recipe([{"item": "tofu", "quantity_g": 200}])
        self._stub(monkeypatch, good)

        _recipe, analysis = await graph.generate_recipe(
            "Tofu stir fry", "Vegan.", 290, 35, make_profile(diet_type=DietType.VEGAN)
        )
        assert analysis.protein_g == pytest.approx(34.6, abs=0.5)


class ProviderError(Exception):
    """Shaped like a Groq/OpenAI error: a status code and the provider's body."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class TestARecipeThatWillNotFitOnTheFirstTry:
    """A long recipe used to fail once and be reported as "BadRequestError".

    How much a recipe writes is not knowable before it is written. A one-pan
    tofu scramble is 475 tokens; "rajma with cauliflower rice and a cucumber
    salad" is three dishes and 861. One flat reservation is right for one and
    short for the other, and the short case truncated, came back as a 400, and
    was surfaced to the user as the name of an exception class.

    The plan path had already learned both halves of this — retry a truncated
    structured output, and say what went wrong. The recipe path had neither.
    """

    @staticmethod
    def _failing_then_working(monkeypatch, recipe, failures: int):
        """Fail with a truncated structured output `failures` times, then work."""
        seen: list[int] = []

        def factory(_schema, **budget):
            attempt = budget.get("attempt", 1)
            seen.append(attempt)

            class Stub:
                async def ainvoke(self, _messages):
                    if len(seen) <= failures:
                        raise ProviderError(
                            "tool_use_failed: Failed to call a function.", 400
                        )
                    return recipe

            return Stub()

        monkeypatch.setattr(graph, "get_structured_llm", factory)
        return seen

    async def test_a_truncated_recipe_is_retried(self, monkeypatch):
        good = make_recipe([{"item": "kidney beans", "quantity_g": 150}])
        seen = self._failing_then_working(monkeypatch, good, failures=1)

        recipe, _analysis = await graph.generate_recipe(
            "Rajma with cauliflower rice",
            "Hearty rajma over low-carb cauliflower rice.",
            310,
            28,
            make_profile(diet_type=DietType.VEGETARIAN),
        )

        assert recipe is good
        # A third call can follow: the macro-correction loop re-asks when the
        # summed weights miss the meal's claim. Only the first two are the
        # retry under test.
        assert seen[:2] == [1, 2], "the truncated draft was not retried"

    async def test_the_retry_asks_for_more_room(self, monkeypatch):
        """Otherwise it is the same request twice, and truncates twice."""
        good = make_recipe([{"item": "kidney beans", "quantity_g": 150}])
        seen = self._failing_then_working(monkeypatch, good, failures=1)

        await graph.generate_recipe(
            "Rajma with cauliflower rice", "Hearty rajma.", 310, 28, make_profile()
        )

        from app.agent.llm import budget_for
        from app.models.plan import Recipe as RecipeSchema

        first, second = (budget_for(RecipeSchema, attempt=n) for n in seen[:2])
        assert second > first

    async def test_it_gives_up_rather_than_looping(self, monkeypatch):
        good = make_recipe([{"item": "kidney beans", "quantity_g": 150}])
        seen = self._failing_then_working(monkeypatch, good, failures=99)

        with pytest.raises(ProviderError):
            await graph.generate_recipe(
                "Rajma with cauliflower rice", "Hearty rajma.", 310, 28, make_profile()
            )

        assert len(seen) == graph.MAX_RECIPE_DRAFT_ATTEMPTS

    async def test_a_config_error_is_not_retried(self, monkeypatch):
        """A rejected key fails identically every time. Retrying is theatre."""
        good = make_recipe([{"item": "paneer", "quantity_g": 100}])
        seen = self._failing_then_working(monkeypatch, good, failures=99)

        def factory(_schema, **budget):
            seen.append(budget.get("attempt", 1))

            class Stub:
                async def ainvoke(self, _messages):
                    raise ProviderError("invalid api key", 401)

            return Stub()

        monkeypatch.setattr(graph, "get_structured_llm", factory)
        seen.clear()

        with pytest.raises(ProviderError):
            await graph.generate_recipe(
                "Paneer bhurji", "Quick and high protein.", 400, 30, make_profile()
            )

        assert len(seen) == 1


class TestTheRecipeErrorSaysSomething:
    def test_the_exception_class_name_is_not_the_message(self):
        """What the user actually saw was "Could not generate the recipe:
        BadRequestError" — a class name, naming none of the dozen things a 400
        can mean and suggesting nothing to do about any of them."""
        from app.agent.llm import describe_llm_failure

        message = describe_llm_failure(
            ProviderError("tool_use_failed: Failed to call a function.", 400)
        ).message

        assert "BadRequestError" not in message
        assert len(message.split()) > 8, "a diagnosis is longer than two words"
