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
from pymongo.errors import InvalidURI, PyMongoError
from pymongo.server_api import ServerApi

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None

# Why the last connection attempt failed, so /health and any request that needs
# the database can say something better than "not connected".
_connection_error: Optional[str] = None


class DatabaseUnavailableError(RuntimeError):
    """The database is needed for this request and isn't connected.

    Distinct from a bare RuntimeError so the API layer can answer 503 with the
    reason instead of a 500 and a stack trace — a dependency being down is not
    a bug in the endpoint.
    """


# Characters that carry meaning in the userinfo part of a URI and so must be
# percent-encoded inside a username or password.
_MUST_ENCODE = {
    ":": "%3A",
    "/": "%2F",
    "?": "%3F",
    "#": "%23",
    "[": "%5B",
    "]": "%5D",
    "@": "%40",
}


def describe_connection_failure(exc: Exception) -> str:
    """Say which layer failed, rather than quoting pymongo at the user.

    The three cases look nothing alike from the outside and need completely
    different responses, but they arrive as one exception type with a wall of
    topology description attached.
    """
    detail = str(exc)

    if "getaddrinfo failed" in detail or "nodename nor servname" in detail:
        return (
            "The database hostname could not be resolved, which means the "
            "machine has no working DNS — not that the connection string is "
            "wrong. Check the internet connection first. On a campus or office "
            "network, try a phone hotspot: some block outbound MongoDB traffic."
        )

    if "timed out" in detail.lower() or "ServerSelectionTimeout" in type(exc).__name__:
        return (
            "The database hostname resolved but did not answer. Usually Atlas "
            "> Network Access does not list your current IP address, or a "
            "firewall is blocking port 27017."
        )

    if "auth" in detail.lower() or "not authorized" in detail.lower():
        return (
            "The database rejected the username or password. Check "
            "MONGODB_URI, and remember special characters must be "
            "percent-encoded."
        )

    return f"Could not reach MongoDB: {detail[:200]}"


def describe_credential_problem(uri: str) -> Optional[str]:
    """Explain an unescaped-credentials error in terms of what to change.

    pymongo says "must be escaped according to RFC 3986", which is accurate and
    unusable. This names the offending characters and their replacements.

    It never returns the password itself — only which characters appear in it —
    because the result is written to logs.
    """
    if "://" not in uri:
        return None

    _, _, rest = uri.partition("://")
    if "@" not in rest:
        return None

    # rsplit: the *last* @ separates credentials from the host, so a password
    # containing @ — the whole reason we are here — still splits correctly.
    userinfo, _, _ = rest.rpartition("@")
    username, _, password = userinfo.partition(":")

    problems = []
    for label, value in (("username", username), ("password", password)):
        found = sorted({c for c in value if c in _MUST_ENCODE})
        if found:
            fixes = ", ".join(f"{c} -> {_MUST_ENCODE[c]}" for c in found)
            problems.append(f"{label} contains {fixes}")

    if not problems:
        return None

    return (
        "MONGODB_URI has unescaped characters in the credentials: "
        + "; ".join(problems)
        + ". Replace them in the .env value. Leave the @ that separates the "
        "password from the hostname alone — only the one *inside* the "
        "password gets encoded."
    )


async def connect_to_mongo() -> None:
    """Open the connection pool and verify it. Called on app startup."""
    global _client, _database, _connection_error

    if not settings.mongodb_uri:
        _connection_error = "MONGODB_URI is not set in the environment."
        _fail_loudly(_connection_error)
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

        _connection_error = None
        await _ensure_indexes()

    except InvalidURI as exc:
        # The connection string itself is malformed — almost always a password
        # with a special character in it, pasted straight from Atlas.
        hint = describe_credential_problem(settings.mongodb_uri)
        _connection_error = hint or str(exc)
        _fail_loudly(_connection_error, detail=str(exc) if hint else None)
        _client = None
        _database = None

    except PyMongoError as exc:
        _connection_error = describe_connection_failure(exc)
        _fail_loudly(_connection_error, detail=str(exc)[:200])
        _client = None
        _database = None


def _fail_loudly(message: str, detail: Optional[str] = None) -> None:
    """Make a fatal startup problem impossible to scroll past.

    Uvicorn prints "Application startup complete" regardless, so a single ERROR
    line above it reads like a warning about something that recovered. It has
    not recovered: every request that touches the database will fail.
    """
    logger.error("=" * 72)
    logger.error("DATABASE UNAVAILABLE - the API started but cannot serve data")
    logger.error("")
    for line in message.split(". "):
        if line.strip():
            logger.error("  %s", line.strip().rstrip(".") + ".")
    if detail:
        logger.error("")
        logger.error("  (%s)", detail)
    logger.error("")
    logger.error("  Fix backend/.env, then save any .py file to reload.")
    logger.error("=" * 72)


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
        raise DatabaseUnavailableError(
            _connection_error
            or "Database is not connected. Check MONGODB_URI and the startup logs."
        )
    return _database


def connection_error() -> Optional[str]:
    """Why the last connection attempt failed, if it did."""
    return _connection_error


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
