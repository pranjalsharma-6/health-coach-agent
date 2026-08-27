"""Check the configured LLM and list the models the key can actually use.

Run from backend/ with the venv active:

    python -m app.tools.check_llm

Exists because providers retire models on a rolling schedule. When that happens
the agent fails with a 404 and the only useful next step is "find out what your
key can reach now" — which otherwise means digging through provider docs that
may already be out of date.
"""

import asyncio
import sys

import httpx

from app.core.config import settings

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"


def _print(line: str = "") -> None:
    print(line, flush=True)


async def _fetch_models(url: str, key: str) -> list[str]:
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(url, headers={"Authorization": f"Bearer {key}"})
        response.raise_for_status()
        return sorted(m["id"] for m in response.json().get("data", []))


def looks_like_a_chat_model(name: str) -> bool:
    """Filter out speech, embedding and moderation models.

    The list is long and mostly irrelevant — you want the ones that can hold a
    conversation and return structured output.
    """
    lowered = name.lower()
    skip = ("whisper", "tts", "embed", "guard", "distil", "playai")
    return not any(term in lowered for term in skip)


async def main() -> int:
    _print("Kaya — LLM configuration check")
    _print("=" * 60)

    if settings.groq_api_key:
        provider, url, key = "Groq", GROQ_MODELS_URL, settings.groq_api_key
    elif settings.openai_api_key:
        provider, url, key = "OpenAI", OPENAI_MODELS_URL, settings.openai_api_key
    else:
        _print("No API key found in backend/.env")
        _print("Set GROQ_API_KEY (free at https://console.groq.com).")
        return 1

    _print(f"Provider      : {provider}")
    _print(f"Key           : {key[:7]}…{key[-4:]} ({len(key)} chars)")
    _print(f"LLM_MODEL     : {settings.llm_model}")
    _print()

    try:
        models = await _fetch_models(url, key)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        _print(f"The provider rejected the request (HTTP {status}).")
        if status in (401, 403):
            _print()
            _print("The key is not valid. Check backend/.env:")
            _print("  - a Groq key starts with 'gsk_'")
            _print("  - no quotes around the value, no trailing spaces")
            _print("  - generate a fresh one at https://console.groq.com/keys")
        return 1
    except httpx.HTTPError as exc:
        _print(f"Could not reach {provider}: {exc}")
        _print("Check your internet connection or any proxy/firewall.")
        return 1

    chat_models = [m for m in models if looks_like_a_chat_model(m)]

    if settings.llm_model in models:
        _print(f"OK — '{settings.llm_model}' is available. Nothing to change.")
        _print()
        _print("If the agent still fails, the problem is not the model name.")
        return 0

    _print(f"PROBLEM — '{settings.llm_model}' is NOT available on this key.")
    _print("This is what produces the 404 / NotFoundError in the run timeline.")
    _print()
    _print("Chat models your key CAN use:")
    _print()
    for name in chat_models:
        _print(f"    {name}")
    _print()
    _print("Pick one and set it in backend/.env:")
    if chat_models:
        _print()
        _print(f"    LLM_MODEL={chat_models[0]}")
    _print()
    _print("Prefer a large instruction-tuned model — the agent asks for")
    _print("structured JSON, and small models are unreliable at it.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
