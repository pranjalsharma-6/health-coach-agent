"""Agent execution routes, including the streamed run."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.graph import run_agent, stream_agent
from app.agent.llm import is_configured
from app.api.deps import CurrentProfile, CurrentUser
from app.core.logging import get_logger
from app.db.repositories import AgentEventRepository
from app.models.enums import AgentDecision
from app.models.log import AdherenceSnapshot, AgentEventInDB
from app.models.plan import PlanInDB

router = APIRouter(prefix="/agent", tags=["agent"])
logger = get_logger(__name__)


class AgentRunResponse(BaseModel):
    decision: AgentDecision
    trigger_detail: str
    snapshot: Optional[AdherenceSnapshot]
    plan: Optional[PlanInDB]
    steps: List[Dict[str, Any]]
    attempts: int
    validation_warnings: List[str]
    error: Optional[str]


def _require_llm() -> None:
    if not is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No LLM provider configured. Set GROQ_API_KEY in backend/.env "
                "(free at console.groq.com)."
            ),
        )


@router.post("/run", response_model=AgentRunResponse)
async def run(
    user: CurrentUser,
    _: CurrentProfile,
    force_replan: bool = Query(
        default=False, description="Skip the decision rules and replan regardless."
    ),
) -> AgentRunResponse:
    """Run the agent loop once, blocking until it finishes.

    Prefer `/agent/stream` in the UI — this exists for scripting and testing.
    """
    _require_llm()

    final = await run_agent(str(user.id), force_replan=force_replan)

    decision = final.get("decision", AgentDecision.NO_ACTION)
    return AgentRunResponse(
        decision=decision,
        trigger_detail=final.get("trigger_detail", ""),
        snapshot=final.get("snapshot"),
        plan=final.get("saved_plan"),
        steps=final.get("steps", []),
        attempts=final.get("attempt", 0),
        validation_warnings=final.get("validation_warnings", []),
        error=final.get("error"),
    )


@router.get("/stream")
async def stream(
    user: CurrentUser,
    _: CurrentProfile,
    force_replan: bool = Query(default=False),
) -> StreamingResponse:
    """Run the agent, streaming each step as a Server-Sent Event.

    This is what replaces the old 60-second spinner: the user sees the agent
    sense, evaluate, decide, draft, validate and save, in order, as it happens.
    """
    _require_llm()
    user_id = str(user.id)

    async def event_source() -> AsyncIterator[str]:
        try:
            async for step in stream_agent(user_id, force_replan=force_replan):
                yield f"data: {json.dumps(step, default=str)}\n\n"
        except Exception as exc:
            logger.exception("Agent stream failed for user %s", user_id)
            payload = {
                "node": "error",
                "status": "error",
                "message": f"The agent run failed: {type(exc).__name__}",
            }
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Stop nginx-style proxies from buffering the stream into one chunk.
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/events", response_model=List[AgentEventInDB])
async def list_events(
    user: CurrentUser, limit: int = Query(default=30, ge=1, le=100)
) -> List[AgentEventInDB]:
    """The agent's decision timeline — every run, including the no-ops.

    Recording decisions that changed nothing is deliberate: it's the difference
    between "the agent checked and you're fine" and "the agent never ran".
    """
    return await AgentEventRepository.list_recent(str(user.id), limit=limit)
