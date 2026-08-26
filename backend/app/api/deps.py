"""Shared FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token
from app.db.repositories import ProfileRepository, UserRepository
from app.models.profile import ProfileInDB
from app.models.user import UserInDB

bearer_scheme = HTTPBearer(auto_error=False)

_CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ],
) -> UserInDB:
    """Resolve the authenticated user from the Bearer token."""
    if credentials is None:
        raise _CREDENTIALS_ERROR

    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise _CREDENTIALS_ERROR

    user_id = payload.get("sub")
    if not user_id:
        raise _CREDENTIALS_ERROR

    user = await UserRepository.get_by_id(user_id)
    if user is None or not user.is_active:
        raise _CREDENTIALS_ERROR

    return user


CurrentUser = Annotated[UserInDB, Depends(get_current_user)]


async def get_current_profile(user: CurrentUser) -> ProfileInDB:
    """Resolve the user's health profile, requiring that onboarding is complete.

    Most of the app is meaningless without a profile — there's nothing to plan
    against — so this fails with a specific 409 the frontend can route on rather
    than a generic error.
    """
    profile = await ProfileRepository.get(str(user.id))
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Profile not found. Complete onboarding first.",
        )
    return profile


CurrentProfile = Annotated[ProfileInDB, Depends(get_current_profile)]
