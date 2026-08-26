"""MongoDB connection lifecycle.

Two things here directly address the "database goes to sleep" problem from the
Streamlit version:

1. The client is created during application *startup*, not lazily on the first
   request. The TLS handshake and auth round-trip happen while the container is
   booting, so the first real user doesn't pay for them.
2. The connection pool is configured with a floor (`minPoolSize`), so idle
   connections stay open instead of being torn down and re-established.
"""

from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import PyMongoError
from pymongo.server_api import ServerApi

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


async def connect_to_mongo() -> None:
    """Open the connection pool and verify it. Called on app startup."""
    global _client, _database

    if not settings.mongodb_uri:
        logger.error("MONGODB_URI is not set — database features will fail.")
        return

    try:
        _client = AsyncIOMotorClient(
            settings.mongodb_uri,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=8000,
            connectTimeoutMS=8000,
            # Keep connections warm rather than reopening them per request.
            minPoolSize=1,
            maxPoolSize=20,
            maxIdleTimeMS=60_000,
            retryWrites=True,
        )
        await _client.admin.command("ping")
        _database = _client[settings.mongodb_db_name]
        logger.info("Connected to MongoDB (db=%s)", settings.mongodb_db_name)

        await _ensure_indexes()

    except PyMongoError as exc:
        logger.error("MongoDB connection failed: %s", exc)
        _client = None
        _database = None


async def close_mongo_connection() -> None:
    """Close the pool cleanly on shutdown."""
    global _client, _database
    if _client is not None:
        _client.close()
        _client = None
        _database = None
        logger.info("MongoDB connection closed")


def get_database() -> AsyncIOMotorDatabase:
    """Return the live database handle.

    Raises rather than returning None so callers fail loudly at the boundary
    instead of hitting an AttributeError three frames deep.
    """
    if _database is None:
        raise RuntimeError(
            "Database is not connected. Check MONGODB_URI and the startup logs."
        )
    return _database


def is_connected() -> bool:
    return _database is not None


async def _ensure_indexes() -> None:
    """Create the indexes the query patterns actually need. Idempotent."""
    db = _database
    if db is None:
        return

    try:
        await db.users.create_index("email", unique=True)

        await db.profiles.create_index("user_id", unique=True)

        # The hot path: "the active plan for this user".
        await db.plans.create_index([("user_id", 1), ("is_active", -1)])
        await db.plans.create_index([("user_id", 1), ("created_at", -1)])

        # One log document per user per day.
        await db.daily_logs.create_index([("user_id", 1), ("log_date", -1)], unique=True)

        await db.agent_events.create_index([("user_id", 1), ("created_at", -1)])

        logger.info("MongoDB indexes verified")
    except PyMongoError as exc:
        # Non-fatal: the app works without indexes, just slower.
        logger.warning("Could not create indexes: %s", exc)


async def health_check() -> dict:
    """Report database status for the /health endpoint."""
    if _client is None:
        return {"status": "disconnected", "error": "No client initialised"}
    try:
        info = await _client.server_info()
        return {
            "status": "healthy",
            "version": info.get("version", "unknown"),
            "database": settings.mongodb_db_name,
        }
    except PyMongoError as exc:
        return {"status": "unhealthy", "error": str(exc)}
