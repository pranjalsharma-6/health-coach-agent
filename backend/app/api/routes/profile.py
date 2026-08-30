"""Profile and onboarding routes."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.api.deps import CurrentProfile, CurrentUser
from app.core.logging import get_logger
from app.db.repositories import ProfileRepository, UserRepository
from app.models.profile import ProfileCreate, ProfileInDB, ProfileUpdate
from app.models.plan import NutritionTargets
from app.services.nutrition import (
    calculate_energy_profile,
    calculate_targets,
    estimated_weekly_change_kg,
    macro_split_percent,
)

router = APIRouter(prefix="/profile", tags=["profile"])
logger = get_logger(__name__)


class TargetsResponse(BaseModel):
    """Everything the dashboard needs to render the user's numbers."""

    bmr_kcal: int
    tdee_kcal: int
    targets: NutritionTargets
    macro_split_percent: Dict[str, int]
    deficit_or_surplus_kcal: int
    estimated_weekly_change_kg: float
    safety_floor_applied: bool
    bmi: float


@router.post("", response_model=ProfileInDB, status_code=status.HTTP_201_CREATED)
async def create_profile(payload: ProfileCreate, user: CurrentUser) -> ProfileInDB:
    """Complete onboarding. Idempotent. Resubmitting replaces the profile."""
    profile = await ProfileRepository.upsert(
        ProfileInDB(user_id=str(user.id), **payload.model_dump())
    )
    await UserRepository.mark_onboarded(str(user.id))
    logger.info("Profile saved for user %s (diet=%s)", user.email, payload.diet_type)
    return profile


@router.get("", response_model=ProfileInDB)
async def read_profile(profile: CurrentProfile) -> ProfileInDB:
    return profile


@router.patch("", response_model=ProfileInDB)
async def update_profile(payload: ProfileUpdate, user: CurrentUser) -> ProfileInDB:
    """Partially update the profile. Only supplied fields change."""
    changes: Dict[str, Any] = payload.model_dump(exclude_unset=True, exclude_none=True)

    updated = await ProfileRepository.patch(str(user.id), changes)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Profile not found. Complete onboarding first.",
        )
    return updated


@router.get("/targets", response_model=TargetsResponse)
async def read_targets(profile: CurrentProfile) -> TargetsResponse:
    """Return computed nutrition targets.

    These come from `services.nutrition`. Pure arithmetic, no LLM involved.
    """
    energy = calculate_energy_profile(profile)
    targets = calculate_targets(profile)

    return TargetsResponse(
        bmr_kcal=energy.bmr_kcal,
        tdee_kcal=energy.tdee_kcal,
        targets=targets,
        macro_split_percent=macro_split_percent(targets),
        deficit_or_surplus_kcal=energy.deficit_or_surplus_kcal,
        estimated_weekly_change_kg=estimated_weekly_change_kg(profile),
        safety_floor_applied=energy.safety_floor_applied,
        bmi=profile.bmi,
    )
