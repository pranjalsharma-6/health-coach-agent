"""Plan retrieval and lazy recipe expansion."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.agent.graph import generate_recipe
from app.agent.llm import LLMUnavailableError, describe_llm_failure, is_configured
from app.api.deps import CurrentProfile, CurrentUser
from app.core.logging import get_logger
from app.db.repositories import PlanRepository
from app.models.plan import PlanInDB, PlanSummary, Recipe
from app.services.ingredients import RecipeAnalysis, analyse_recipe

router = APIRouter(prefix="/plans", tags=["plans"])
logger = get_logger(__name__)


class MacroCheck(BaseModel):
    """What the recipe's weights actually add up to, versus what it claims.

    Returned to the client rather than kept internal: the point of computing
    macros from ingredients is that the user can see the sum, so a gap is
    visible instead of quietly papered over.
    """

    computed_kcal: float
    computed_protein_g: float
    claimed_kcal: int
    claimed_protein_g: int
    #: Share of the recipe's weighed mass found in the ingredient table.
    coverage: float
    #: False when coverage is too low for the comparison to be fair.
    reliable: bool
    #: Ingredients carrying a weight that we could not identify.
    unmatched: List[str]


class RecipeResponse(BaseModel):
    meal_id: str
    meal_name: str
    recipe: Recipe
    cached: bool
    macro_check: Optional[MacroCheck] = None


@router.get("/active", response_model=Optional[PlanInDB])
async def get_active_plan(user: CurrentUser) -> Optional[PlanInDB]:
    """The user's current plan, or null if the agent hasn't produced one yet."""
    return await PlanRepository.get_active(str(user.id))


@router.get("/history", response_model=List[PlanSummary])
async def get_plan_history(user: CurrentUser, limit: int = 20) -> List[PlanSummary]:
    """Every plan version, newest first — the agent's decision history."""
    plans = await PlanRepository.list_history(str(user.id), limit=limit)
    return [
        PlanSummary(
            id=str(p.id),
            version=p.version,
            plan_title=p.plan_title,
            agent_reasoning=p.agent_reasoning,
            trigger=p.trigger,
            trigger_detail=p.trigger_detail,
            is_active=p.is_active,
            created_at=p.created_at,
        )
        for p in plans
    ]


@router.get("/{plan_id}", response_model=PlanInDB)
async def get_plan(plan_id: str, user: CurrentUser) -> PlanInDB:
    plan = await PlanRepository.get_by_id(plan_id)
    if plan is None or plan.user_id != str(user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Plan not found."
        )
    return plan


@router.post("/meals/{meal_id}/recipe", response_model=RecipeResponse)
async def expand_recipe(
    meal_id: str, user: CurrentUser, profile: CurrentProfile
) -> RecipeResponse:
    """Generate the full recipe for one planned meal, on demand.

    Recipes are expanded lazily rather than generated with the plan: a week's
    worth is ~8k output tokens, most of which nobody reads. Once expanded, the
    recipe is written back to the plan so the next request is free.
    """
    plan = await PlanRepository.get_active(str(user.id))
    if plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No active plan."
        )

    target_meal = None
    for day in plan.daily_plans:
        for meal in day.meals:
            if meal.meal_id == meal_id:
                target_meal = meal
                break
        if target_meal:
            break

    if target_meal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No meal '{meal_id}' in the active plan.",
        )

    if target_meal.recipe is not None:
        # Re-run the sum rather than storing it: the ingredient table can change,
        # and a stale figure is worse than a recomputed one.
        cached_analysis = analyse_recipe(target_meal.recipe.ingredients)
        return RecipeResponse(
            meal_id=meal_id,
            meal_name=target_meal.name,
            recipe=target_meal.recipe,
            cached=True,
            macro_check=_to_macro_check(cached_analysis, target_meal),
        )

    if not is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No LLM provider configured.",
        )

    try:
        recipe, analysis = await generate_recipe(
            meal_name=target_meal.name,
            description=target_meal.description,
            calories=target_meal.calories_kcal,
            protein_g=target_meal.protein_g,
            profile=profile,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        # `type(exc).__name__` is not a diagnosis. "BadRequestError" told the
        # user nothing they could act on and hid which of a dozen causes it
        # was; `describe_llm_failure` already knows how to read a provider
        # error and is what the agent's own timeline uses.
        logger.exception("Recipe generation failed for meal %s", meal_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not generate the recipe. {describe_llm_failure(exc).message}",
        ) from exc

    # Cache it back into the plan without creating a new version — enrichment
    # isn't a planning decision and shouldn't appear in the history.
    target_meal.recipe = recipe
    await PlanRepository.replace_active(plan)

    return RecipeResponse(
        meal_id=meal_id,
        meal_name=target_meal.name,
        recipe=recipe,
        cached=False,
        macro_check=_to_macro_check(analysis, target_meal),
    )


def _to_macro_check(analysis: RecipeAnalysis, meal) -> MacroCheck:
    return MacroCheck(
        computed_kcal=analysis.kcal,
        computed_protein_g=analysis.protein_g,
        claimed_kcal=meal.calories_kcal,
        claimed_protein_g=meal.protein_g,
        coverage=analysis.coverage,
        reliable=analysis.is_reliable,
        unmatched=list(analysis.unmatched),
    )
