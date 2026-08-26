"""User account models — authentication only. Health data lives in Profile."""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from app.models.common import MongoModel, PyObjectId, utcnow


class UserInDB(MongoModel):
    """The stored user document. Never returned from the API."""

    email: EmailStr
    hashed_password: str
    full_name: str
    is_active: bool = True
    onboarded: bool = False
    created_at: datetime = Field(default_factory=utcnow)


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)
    full_name: str = Field(min_length=1, max_length=100)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserPublic(BaseModel):
    """Safe user representation — no password hash."""

    id: PyObjectId
    email: EmailStr
    full_name: str
    onboarded: bool
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
