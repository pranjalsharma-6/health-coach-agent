"""Tests for the ingredient reference table and the protein-density ceiling."""

import pytest

from app.models.enums import DietType
from app.services import ingredients as ing


class TestTableIntegrity:
    def test_keys_are_unique(self):
        keys = [item.key for item in ing._TABLE]
        assert len(keys) == len(set(keys))

    @pytest.mark.parametrize("item", ing._TABLE, ids=lambda i: i.key)
    def test_macros_reconcile_with_stated_calories(self, item):
        """Every row must obey the 4/4/9 rule within a sane margin.

        A typo here would silently distort the ceiling every diet is judged by,
        so the table checks itself.
        """
        derived = item.protein_g * 4 + item.carbs_g * 4 + item.fat_g * 9

        if item.kcal == 0:
            assert derived == 0
            return

        absolute = abs(derived - item.kcal)
        relative = absolute / item.kcal

        # Atwater (4/4/9) counts total carbohydrate, but dietary fibre is only
        # partly metabolised, so whole foods always derive a little high. On a
        # 23 kcal food like spinach that gap is ~6 kcal. Trivial in absolute
        # terms but a 29% relative drift, so a purely relative bound would
        # reject correct data. Allow whichever margin is more forgiving.
        assert relative <= 0.25 or absolute <= 8, (
            f"{item.name}: macros imply {derived:.0f} kcal but the row says "
            f"{item.kcal:.0f} (off by {absolute:.0f} kcal, {relative:.0%})"
        )

    @pytest.mark.parametrize("item", ing._TABLE, ids=lambda i: i.key)
    def test_values_are_physically_sane(self, item):
        assert item.kcal >= 0
        assert item.protein_g >= 0
        assert item.carbs_g >= 0
        assert item.fat_g >= 0
        # Nothing edible is more than ~95% protein by mass.
        assert item.protein_g <= 95


class TestDietFiltering:
    def test_vegetarian_excludes_meat_and_egg(self):
        allowed = {i.key for i in ing.allowed_for(DietType.VEGETARIAN)}
        assert "chicken_breast" not in allowed
        assert "egg" not in allowed
        assert "paneer" in allowed

    def test_eggetarian_keeps_egg_but_not_meat(self):
        allowed = {i.key for i in ing.allowed_for(DietType.EGGETARIAN)}
        assert "egg" in allowed
        assert "chicken_breast" not in allowed

    def test_vegan_excludes_dairy(self):
        allowed = {i.key for i in ing.allowed_for(DietType.VEGAN)}
        assert "paneer" not in allowed
        assert "curd" not in allowed
        assert "tofu" in allowed

    def test_jain_excludes_root_vegetables(self):
        allowed = {i.key for i in ing.allowed_for(DietType.JAIN)}
        assert "potato" not in allowed
        assert "onion" not in allowed
        assert "carrot" not in allowed
        assert "paneer" in allowed

    def test_non_vegetarian_allows_everything(self):
        allowed = ing.allowed_for(DietType.NON_VEGETARIAN)
        assert len(allowed) == len(ing._TABLE)


class TestProteinCeiling:
    def test_every_diet_has_a_usable_ceiling(self):
        for diet in DietType:
            assert 0.05 < ing.protein_ceiling(diet) < 0.5

    def test_non_vegetarian_ceiling_beats_vegan(self):
        """Animal protein really is more calorie-efficient; the bound should say so."""
        assert ing.protein_ceiling(DietType.NON_VEGETARIAN) > ing.protein_ceiling(
            DietType.VEGAN
        )

    def test_supplements_do_not_inflate_the_ceiling(self):
        """Whey is 80% protein; including it would defeat the check."""
        whey = ing.BY_KEY["whey"]
        assert whey.is_supplement
        assert ing.max_protein_density(DietType.VEGETARIAN) < whey.protein_density

    def test_real_foods_are_never_flagged_impossible(self):
        """Every row must pass its own diet's ceiling. Else it's self-refuting."""
        for diet in DietType:
            for item in ing.allowed_for(diet):
                if item.is_supplement or item.kcal <= 0:
                    continue
                assert ing.is_protein_claim_possible(
                    round(item.kcal), round(item.protein_g), diet
                ), f"{item.name} exceeds the {diet.value} ceiling"

    def test_chicken_breast_portion_is_possible_for_non_vegetarians(self):
        assert ing.is_protein_claim_possible(330, 62, DietType.NON_VEGETARIAN)

    def test_absurd_claim_is_rejected(self):
        assert not ing.is_protein_claim_possible(200, 50, DietType.VEGETARIAN)

    def test_zero_calorie_meal_is_never_possible(self):
        assert not ing.is_protein_claim_possible(0, 10, DietType.VEGETARIAN)

    def test_ceiling_derives_from_the_table(self):
        """Adding a denser food should raise the bound, not require a constant edit."""
        veg_max = ing.max_protein_density(DietType.VEGETARIAN)
        densest = max(
            (
                i.protein_density
                for i in ing.allowed_for(DietType.VEGETARIAN)
                if not i.is_supplement
            )
        )
        assert veg_max == densest


class TestLookupAndPrompt:
    def test_finds_ingredients_by_alias(self):
        found = ing.find_in_text("Rajma chawal with a side of cucumber raita")
        assert "rajma" in found
        assert "rice" in found  # via the "chawal" alias
        assert "cucumber" in found

    def test_returns_nothing_for_unrecognised_text(self):
        assert ing.find_in_text("Zorblax casserole") == set()

    def test_prompt_block_lists_only_permitted_foods(self):
        block = ing.protein_reference_block(DietType.VEGAN)
        assert "Tofu" in block
        assert "Paneer" not in block
        assert "Chicken" not in block

    def test_prompt_block_is_ordered_by_density(self):
        rows = ing.top_protein_sources(DietType.NON_VEGETARIAN, 5)
        densities = [r.protein_density for r in rows]
        assert densities == sorted(densities, reverse=True)

    def test_prompt_block_excludes_supplements(self):
        block = ing.protein_reference_block(DietType.VEGETARIAN)
        assert "Whey" not in block
