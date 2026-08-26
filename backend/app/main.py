"""Kaya API — FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth, profile
from app.core.config import settings
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
    await mongo.connect_to_mongo()
    yield
    await mongo.close_mongo_connection()
    logger.info("Shutdown complete")


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1")
app.include_router(profile.router, prefix="/api/v1")


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

    return {
        "status": "healthy" if healthy else "degraded",
        "database": db_status,
        "llm_configured": bool(settings.groq_api_key or settings.openai_api_key),
        "environment": settings.environment,
    }
