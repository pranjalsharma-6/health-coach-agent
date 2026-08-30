"""Data access layer.

Route handlers and the agent both talk to these, never to Motor directly. This
keeps Mongo-specific details (ObjectId coercion, index-shaped queries) in one
place, and means the storage engine could change without touching the API.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from bson.errors import InvalidId

from app.core.logging import get_logger
from app.db.mongo import get_database
from app.models.log import (
    AgentEventInDB,
    DailyLogInDB,
    MealLogEntry,
    SessionLogEntry,
)
from app.models.plan import PlanInDB
from app.models.profile import ProfileInDB
from app.models.user import UserInDB

logger = get_logger(__name__)


def _to_object_id(value: str) -> Optional[ObjectId]:
    """Parse a string id, returning None rather than raising on garbage input."""
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


def _iso_date(value: date) -> str:
    """Mongo has no date type. Store dates as ISO strings for stable sorting."""
    return value.isoformat()


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
class UserRepository:
    @staticmethod
    async def create(user: UserInDB) -> UserInDB:
        db = get_database()
        result = await db.users.insert_one(user.to_mongo())
        user.id = str(result.inserted_id)
        return user

    @staticmethod
    async def get_by_email(email: str) -> Optional[UserInDB]:
        db = get_database()
        doc = await db.users.find_one({"email": email.lower()})
        return UserInDB.from_mongo(doc)

    @staticmethod
    async def get_by_id(user_id: str) -> Optional[UserInDB]:
        oid = _to_object_id(user_id)
        if oid is None:
            return None
        db = get_database()
        doc = await db.users.find_one({"_id": oid})
        return UserInDB.from_mongo(doc)

    @staticmethod
    async def mark_onboarded(user_id: str) -> None:
        oid = _to_object_id(user_id)
        if oid is None:
            return
        db = get_database()
        await db.users.update_one({"_id": oid}, {"$set": {"onboarded": True}})


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
class ProfileRepository:
    @staticmethod
    async def upsert(profile: ProfileInDB) -> ProfileInDB:
        db = get_database()
        data = profile.to_mongo()
        data["updated_at"] = datetime.now(timezone.utc)
        await db.profiles.update_one(
            {"user_id": profile.user_id}, {"$set": data}, upsert=True
        )
        stored = await db.profiles.find_one({"user_id": profile.user_id})
        return ProfileInDB.from_mongo(stored)

    @staticmethod
    async def get(user_id: str) -> Optional[ProfileInDB]:
        db = get_database()
        doc = await db.profiles.find_one({"user_id": user_id})
        return ProfileInDB.from_mongo(doc)

    @staticmethod
    async def patch(user_id: str, changes: Dict[str, Any]) -> Optional[ProfileInDB]:
        if not changes:
            return await ProfileRepository.get(user_id)
        db = get_database()
        changes["updated_at"] = datetime.now(timezone.utc)
        await db.profiles.update_one({"user_id": user_id}, {"$set": changes})
        return await ProfileRepository.get(user_id)


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #
class PlanRepository:
    @staticmethod
    async def save_new_version(plan: PlanInDB) -> PlanInDB:
        """Insert a plan and deactivate any previous active plan atomically enough.

        Deactivate-then-insert (rather than the reverse) means a crash between the
        two leaves the user with no active plan. Recoverable by running the agent,
        rather than two active plans, which would be ambiguous.
        """
        db = get_database()

        latest = await db.plans.find_one(
            {"user_id": plan.user_id}, sort=[("version", -1)]
        )
        plan.version = (latest["version"] + 1) if latest else 1
        if latest and plan.parent_plan_id is None:
            plan.parent_plan_id = str(latest["_id"])

        await db.plans.update_many(
            {"user_id": plan.user_id, "is_active": True},
            {"$set": {"is_active": False}},
        )

        result = await db.plans.insert_one(plan.to_mongo())
        plan.id = str(result.inserted_id)
        logger.info(
            "Saved plan v%s for user %s (trigger=%s)",
            plan.version,
            plan.user_id,
            plan.trigger,
        )
        return plan

    @staticmethod
    async def get_active(user_id: str) -> Optional[PlanInDB]:
        db = get_database()
        doc = await db.plans.find_one(
            {"user_id": user_id, "is_active": True}, sort=[("created_at", -1)]
        )
        return PlanInDB.from_mongo(doc)

    @staticmethod
    async def get_by_id(plan_id: str) -> Optional[PlanInDB]:
        oid = _to_object_id(plan_id)
        if oid is None:
            return None
        db = get_database()
        doc = await db.plans.find_one({"_id": oid})
        return PlanInDB.from_mongo(doc)

    @staticmethod
    async def list_history(user_id: str, limit: int = 20) -> List[PlanInDB]:
        db = get_database()
        cursor = db.plans.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        return [PlanInDB.from_mongo(doc) async for doc in cursor]

    @staticmethod
    async def replace_active(plan: PlanInDB) -> Optional[PlanInDB]:
        """Overwrite the active plan in place, without creating a new version.

        Used for lazily-expanded recipe detail, which is enrichment rather than a
        planning decision and shouldn't pollute the version history.
        """
        oid = _to_object_id(plan.id or "")
        if oid is None:
            return None
        db = get_database()
        data = plan.to_mongo()
        await db.plans.update_one({"_id": oid}, {"$set": data})
        return await PlanRepository.get_by_id(plan.id)


# --------------------------------------------------------------------------- #
# Daily logs
# --------------------------------------------------------------------------- #
class LogRepository:
    @staticmethod
    async def get_or_create(user_id: str, log_date: date) -> DailyLogInDB:
        db = get_database()
        key = {"user_id": user_id, "log_date": _iso_date(log_date)}
        doc = await db.daily_logs.find_one(key)
        if doc:
            return DailyLogInDB.from_mongo(doc)

        fresh = DailyLogInDB(user_id=user_id, log_date=log_date)
        data = fresh.to_mongo()
        data["log_date"] = _iso_date(log_date)
        result = await db.daily_logs.insert_one(data)
        fresh.id = str(result.inserted_id)
        return fresh

    @staticmethod
    async def upsert_meal(
        user_id: str, log_date: date, entry: MealLogEntry
    ) -> DailyLogInDB:
        """Record a meal outcome, replacing any earlier entry for the same meal."""
        db = get_database()
        key = {"user_id": user_id, "log_date": _iso_date(log_date)}

        await LogRepository.get_or_create(user_id, log_date)

        # Drop a prior entry for this meal so re-logging corrects rather than duplicates.
        await db.daily_logs.update_one(
            key, {"$pull": {"meals": {"meal_id": entry.meal_id}}}
        )
        await db.daily_logs.update_one(
            key,
            {
                "$push": {"meals": entry.model_dump(mode="json")},
                "$set": {"updated_at": datetime.now(timezone.utc)},
            },
        )
        doc = await db.daily_logs.find_one(key)
        return DailyLogInDB.from_mongo(doc)

    @staticmethod
    async def upsert_session(
        user_id: str, log_date: date, entry: SessionLogEntry
    ) -> DailyLogInDB:
        """Record the day's training outcome, replacing any earlier entry.

        Keyed on the plan day rather than the date so that re-logging corrects
        instead of stacking, the same way meals do.
        """
        db = get_database()
        key = {"user_id": user_id, "log_date": _iso_date(log_date)}

        await LogRepository.get_or_create(user_id, log_date)

        await db.daily_logs.update_one(
            key, {"$pull": {"sessions": {"plan_day": entry.plan_day}}}
        )
        await db.daily_logs.update_one(
            key,
            {
                "$push": {"sessions": entry.model_dump(mode="json")},
                "$set": {"updated_at": datetime.now(timezone.utc)},
            },
        )
        doc = await db.daily_logs.find_one(key)
        return DailyLogInDB.from_mongo(doc)

    @staticmethod
    async def update_metrics(
        user_id: str, log_date: date, metrics: Dict[str, Any]
    ) -> DailyLogInDB:
        db = get_database()
        key = {"user_id": user_id, "log_date": _iso_date(log_date)}
        await LogRepository.get_or_create(user_id, log_date)

        clean = {k: v for k, v in metrics.items() if v is not None}
        clean["updated_at"] = datetime.now(timezone.utc)
        await db.daily_logs.update_one(key, {"$set": clean})

        doc = await db.daily_logs.find_one(key)
        return DailyLogInDB.from_mongo(doc)

    @staticmethod
    async def get_range(
        user_id: str, start: date, end: date
    ) -> List[DailyLogInDB]:
        db = get_database()
        cursor = db.daily_logs.find(
            {
                "user_id": user_id,
                "log_date": {"$gte": _iso_date(start), "$lte": _iso_date(end)},
            }
        ).sort("log_date", 1)
        return [DailyLogInDB.from_mongo(doc) async for doc in cursor]

    @staticmethod
    async def get_recent(user_id: str, days: int = 7) -> List[DailyLogInDB]:
        today = date.today()
        return await LogRepository.get_range(user_id, today - timedelta(days=days - 1), today)


# --------------------------------------------------------------------------- #
# Agent events
# --------------------------------------------------------------------------- #
class AgentEventRepository:
    @staticmethod
    async def record(event: AgentEventInDB) -> AgentEventInDB:
        db = get_database()
        result = await db.agent_events.insert_one(event.to_mongo())
        event.id = str(result.inserted_id)
        return event

    @staticmethod
    async def list_recent(user_id: str, limit: int = 30) -> List[AgentEventInDB]:
        db = get_database()
        cursor = (
            db.agent_events.find({"user_id": user_id})
            .sort("created_at", -1)
            .limit(limit)
        )
        return [AgentEventInDB.from_mongo(doc) async for doc in cursor]
