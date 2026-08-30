"""Authentication routes."""

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUser
from app.core.logging import get_logger
from app.core.security import create_access_token, hash_password, verify_password
from app.db.repositories import UserRepository
from app.models.user import Token, UserCreate, UserInDB, UserLogin, UserPublic

router = APIRouter(prefix="/auth", tags=["auth"])
logger = get_logger(__name__)


def _to_public(user: UserInDB) -> UserPublic:
    return UserPublic(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        onboarded=user.onboarded,
        created_at=user.created_at,
    )


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate) -> Token:
    """Create an account and return a token, so the user is signed in immediately."""
    email = payload.email.lower()

    if await UserRepository.get_by_email(email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    try:
        hashed = hash_password(payload.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    user = await UserRepository.create(
        UserInDB(
            email=email,
            hashed_password=hashed,
            full_name=payload.full_name.strip(),
        )
    )
    logger.info("Registered user %s", email)

    return Token(
        access_token=create_access_token(str(user.id)),
        user=_to_public(user),
    )


@router.post("/login", response_model=Token)
async def login(payload: UserLogin) -> Token:
    """Exchange email and password for an access token."""
    user = await UserRepository.get_by_email(payload.email.lower())

    # Same error for "no such user" and "wrong password". Don't leak which
    # emails have accounts.
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="This account is disabled."
        )

    return Token(
        access_token=create_access_token(str(user.id)),
        user=_to_public(user),
    )


@router.get("/me", response_model=UserPublic)
async def me(user: CurrentUser) -> UserPublic:
    """Return the signed-in user. Used by the frontend to restore a session."""
    return _to_public(user)
