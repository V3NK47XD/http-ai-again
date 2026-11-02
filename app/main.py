"""FastAPI application exposing /ping and curl-friendly /prompt routes."""

import json
import logging
import urllib.parse
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import PlainTextResponse

from app.config import get_settings
from app.llm import UnifiedLLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")


def extract_prompt_from_request(raw_bytes: bytes, query_params: dict) -> str:
    """Extracts user prompt text from raw body bytes or query params.

    Supports:
    - Plain text body: `curl -d "Hello world" http://localhost:8000/prompt`
    - Raw binary body: `curl --data-binary @prompt.txt http://localhost:8000/prompt`
    - Form data: `curl -d "prompt=Hello world" http://localhost:8000/prompt`
    - JSON payload: `curl -d '{"prompt": "Hello"}' http://localhost:8000/prompt`
    - Query parameter fallback: `curl http://localhost:8000/prompt?text=Hello`
    """
    raw_text = raw_bytes.decode("utf-8", errors="replace").strip()

    if raw_text:
        # Check if payload is JSON
        if raw_text.startswith("{") and raw_text.endswith("}"):
            try:
                data = json.loads(raw_text)
                if isinstance(data, dict):
                    for key in ("prompt", "text", "message", "query", "input", "q"):
                        val = data.get(key)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
            except Exception:
                pass

        # Check if payload looks like form-encoded key=value (e.g. prompt=... or text=...)
        if "=" in raw_text and not (" " in raw_text.split("=", 1)[0]):
            try:
                parsed = urllib.parse.parse_qs(raw_text, keep_blank_values=True)
                for key in ("prompt", "text", "message", "query", "input", "q"):
                    if key in parsed and parsed[key] and parsed[key][0].strip():
                        return parsed[key][0].strip()
            except Exception:
                pass

        return raw_text

    # Fallback to query parameters
    for key in ("prompt", "text", "message", "query", "input", "q"):
        val = query_params.get(key)
        if val and val.strip():
            return val.strip()

    return ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager ensuring UnifiedLLM instance is available on app.state."""
    if not hasattr(app.state, "llm") or app.state.llm is None:
        app.state.llm = UnifiedLLM()
    yield


app = FastAPI(
    title="Pydantic-AI Groq & Gemini Service",
    description="Unified API with Groq primary and Gemini fallback via Pydantic-AI",
    lifespan=lifespan,
)


@app.api_route("/ping", methods=["GET", "POST"], response_class=PlainTextResponse)
async def ping() -> PlainTextResponse:
    """Ping health-check endpoint. Returns 'pong'."""
    return PlainTextResponse("pong\n", status_code=status.HTTP_200_OK)


@app.api_route("/prompt", methods=["POST", "GET"], response_class=PlainTextResponse)
async def prompt(request: Request) -> PlainTextResponse:
    """Prompt endpoint accepting text body without headers or auth.

    Executes Groq API first, with automatic Gemini fallback on failure.
    """
    body_bytes = await request.body()
    query_dict = dict(request.query_params)
    user_prompt = extract_prompt_from_request(body_bytes, query_dict)

    if not user_prompt:
        return PlainTextResponse(
            "Error: Prompt body cannot be empty.\n"
            "Usage: curl -d \"Your prompt here\" http://localhost:8000/prompt\n",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    llm: UnifiedLLM = getattr(request.app.state, "llm", None) or UnifiedLLM()

    try:
        response_text = await llm.generate(user_prompt)
        clean_response = response_text.rstrip("\n") + "\n"
        return PlainTextResponse(clean_response, status_code=status.HTTP_200_OK)
    except RuntimeError as e:
        logger.error("LLM execution failure: %s", e)
        return PlainTextResponse(
            f"Error: {e}\n",
            status_code=status.HTTP_502_BAD_GATEWAY,
        )
    except Exception as e:
        logger.exception("Unexpected error handling prompt: %s", e)
        return PlainTextResponse(
            f"Internal Error: {e}\n",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
