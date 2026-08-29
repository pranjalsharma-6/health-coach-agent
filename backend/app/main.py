"""Kaya API — FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import agent, auth, logs, plans, profile
from app.core.config import settings
from app.agent.llm import configuration_problem
from app.agent.llm import is_configured as llm_is_configured
from app.core.logging import get_logger, setup_logging
from app.db import mongo

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Open connections at startup so the first request doesn't pay for them.

    This is the fix for the Streamlit version's cold-start problem: the TLS
    handshake and auth round-trip to Atlas happen here, during boot, rather than
    lazily inside the first user's request.
    """
    logger.info("Starting %s (env=%s)", settings.app_name, settings.environment)
    _warn_about_cors()
    _warn_about_llm_configuration()
    await mongo.connect_to_mongo()
    yield
    await mongo.close_mongo_connection()
    logger.info("Shutdown complete")


def _warn_about_cors() -> None:
    """Say what the browser will be allowed to do, at boot.

    A CORS problem is silent on both sides — the browser reports a missing
    header, the server logs a request it answered — so the running value is
    printed where it can be compared against the site that is failing. On a
    deployed API that still allows only localhost, nothing on the internet can
    call it, and that deserves the same banner a missing database gets.
    """
    origins = settings.cors_origins
    logger.info("CORS allows: %s", origins or "(nothing)")

    local_only = all("localhost" in o or "127.0.0.1" in o for o in origins)
    if settings.is_production and (not origins or local_only):
        logger.error("=" * 72)
        logger.error("CORS NOT CONFIGURED - the API started but no website may call it")
        logger.error("")
        logger.error("  Allowed origins: %s", origins or "(none)")
        logger.error("  Every browser request will fail with a CORS error.")
        logger.error("  Set CORS_ORIGINS to your site's URL, e.g.")
        logger.error("    CORS_ORIGINS=https://your-app.vercel.app")
        logger.error("=" * 72)


def _warn_about_llm_configuration() -> None:
    """Say at boot that the model settings cannot work, rather than at use.

    A model name that belongs to a different provider is a one-line mistake in
    `.env` that only shows up as a 404 in the middle of a plan run, several
    minutes and one filled-in questionnaire later. Startup is where it is
    cheapest to find, so it gets the same unmissable banner the database uses.
    """
    problem = configuration_problem()
    if not problem:
        return

    logger.error("=" * 72)
    logger.error("LLM MISCONFIGURED - the API started but cannot generate plans")
    logger.error("")
    for line in problem.split(". "):
        if line.strip():
            logger.error("  %s", line.strip().rstrip(".") + ".")
    logger.error("=" * 72)


app = FastAPI(
    title=settings.app_name,
    description=(
        "Autonomous nutrition coaching API. Generates preference-aware meal and "
        "activity plans, senses adherence, and replans when the user's week "
        "diverges from the plan."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

def add_cors(target: FastAPI) -> None:
    """Wire the browser allow list onto an app.

    A function rather than three lines inline so a test can exercise the real
    wiring against a chosen configuration. The alternative — reloading this
    module with different environment variables — swaps the shared `settings`
    object out from under every other module that imported it, which broke
    three unrelated tests before this existed.
    """
    target.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


add_cors(app)

@app.exception_handler(mongo.DatabaseUnavailableError)
async def _database_unavailable(_: Request, exc: mongo.DatabaseUnavailableError):
    """Answer 503 with the reason rather than 500 with a traceback.

    The database being down is not a fault in the endpoint that happened to
    touch it, and a caller can act on "the server cannot reach its database"
    where it cannot act on "Internal Server Error".
    """
    logger.error("Request failed — database unavailable: %s", exc)

    content = {"detail": "The server cannot reach its database."}
    if not settings.is_production:
        # The reason names configuration problems, which is exactly what you
        # want on your own machine and nothing a stranger needs. In production
        # it stays in the logs, where the loud startup banner already put it.
        content["reason"] = str(exc)

    return JSONResponse(status_code=503, content=content)


app.include_router(auth.router, prefix="/api/v1")
app.include_router(profile.router, prefix="/api/v1")
app.include_router(plans.router, prefix="/api/v1")
app.include_router(logs.router, prefix="/api/v1")
app.include_router(agent.router, prefix="/api/v1")


@app.get("/", tags=["meta"])
async def root() -> dict:
    return {
        "name": settings.app_name,
        "version": app.version,
        "docs": "/docs",
        "status": "ok",
    }


@app.get("/health", tags=["meta"])
async def health() -> dict:
    """Liveness and dependency check.

    Also serves as the target for an uptime pinger, which keeps the host from
    spinning the container down between visits.
    """
    db_status = await mongo.health_check()
    healthy = db_status.get("status") == "healthy"

    if not healthy:
        # health_check() reports the live probe; connection_error() remembers why
        # startup failed. On a bad URI there is no client to probe at all, so
        # without this the response says "disconnected" and nothing more.
        reason = mongo.connection_error()
        if reason and not settings.is_production:
            db_status = {**db_status, "reason": reason}

    return {
        "status": "healthy" if healthy else "degraded",
        "database": db_status,
        # `is_configured()` knows about Gemini and about LLM_PROVIDER; the old
        # two-key check reported "configured" for a key the resolved provider
        # would never use.
        "llm_configured": llm_is_configured() and configuration_problem() is None,
        "environment": settings.environment,
        # The allowed origins, reported back.
        #
        # A CORS misconfiguration is invisible from both ends: the browser only
        # says a header was missing, and the server logs a request it answered
        # normally. Nothing tells you what the server actually believes its
        # allowed origins are, so a wrong value looks exactly like no value —
        # and you cannot tell a trailing slash from a stale deploy.
        #
        # These are not a secret. CORS works by announcing them to any browser
        # that asks, and they are the public URL of your own site.
        "cors_origins": settings.cors_origins,
    }
