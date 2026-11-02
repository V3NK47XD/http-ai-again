"""Tests for FastAPI endpoints /ping, /prompt, and dedicated model routes."""

import pytest
from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel

from app.llm import UnifiedLLM, _should_fallback_to_next
from app.main import app
from tests.test_llm import FailingModel


@pytest.fixture
def client_with_mock_llm():
    """Provides a TestClient with mock UnifiedLLM injected into app.state."""
    mock_agent = Agent(TestModel(custom_output_text="Echo response from LLM"))
    named_mocks = {
        "qwen": Agent(TestModel(custom_output_text="Qwen response")),
        "gemini-lite": Agent(TestModel(custom_output_text="Gemini Lite response")),
        "gemini": Agent(TestModel(custom_output_text="Gemini response")),
        "gpt": Agent(TestModel(custom_output_text="GPT response")),
        "gemma": Agent(TestModel(custom_output_text="Gemma response")),
    }
    app.state.llm = UnifiedLLM(agent=mock_agent, named_agents=named_mocks)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_with_3tier_fallback_llm():
    """Provides a TestClient where primary Groq and Gemini Flash fail, and Gemini Flash Lite succeeds."""
    groq_failing = FailingModel("Groq 429 rate limit")
    gemini_failing = FailingModel("Gemini Flash 503 unavailable")
    gemini_lite_working = TestModel(custom_output_text="Response from Gemini 3.5 Flash Lite")

    fallback = FallbackModel(groq_failing, gemini_failing, gemini_lite_working, fallback_on=_should_fallback_to_next)
    app.state.llm = UnifiedLLM(agent=Agent(fallback))
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_with_all_failing_llm():
    """Provides a TestClient where all 3 LLMs in the chain fail."""
    m1 = FailingModel("Groq API error")
    m2 = FailingModel("Gemini Flash API error")
    m3 = FailingModel("Gemini Lite API error")
    fallback = FallbackModel(m1, m2, m3, fallback_on=_should_fallback_to_next)
    app.state.llm = UnifiedLLM(agent=Agent(fallback))
    with TestClient(app) as client:
        yield client


def test_ping_get(client_with_mock_llm):
    """GET /ping returns pong with 200 OK."""
    response = client_with_mock_llm.get("/ping")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "pong\n"


def test_ping_post(client_with_mock_llm):
    """POST /ping returns pong with 200 OK."""
    response = client_with_mock_llm.post("/ping")
    assert response.status_code == 200
    assert response.text == "pong\n"


def test_prompt_plain_curl(client_with_mock_llm):
    """curl -d "Hello world" http://localhost:8000/prompt (default x-www-form-urlencoded)."""
    response = client_with_mock_llm.post(
        "/prompt",
        content="What is the circumference of Earth?",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200
    assert response.text == "Echo response from LLM\n"


def test_prompt_binary_multi_line(client_with_mock_llm):
    """curl --data-binary (multiline plain text without explicit headers)."""
    body = "First line of prompt\nSecond line of prompt\nThird line"
    response = client_with_mock_llm.post("/prompt", content=body)
    assert response.status_code == 200
    assert response.text == "Echo response from LLM\n"


def test_prompt_json(client_with_mock_llm):
    """curl -H 'Content-Type: application/json' -d '{"prompt": "Hello"}'."""
    response = client_with_mock_llm.post(
        "/prompt",
        json={"prompt": "Hello via JSON"},
    )
    assert response.status_code == 200
    assert response.text == "Echo response from LLM\n"


def test_prompt_form_key(client_with_mock_llm):
    """curl -d 'prompt=Hello+from+form'."""
    response = client_with_mock_llm.post(
        "/prompt",
        content="prompt=Hello+from+form",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200
    assert response.text == "Echo response from LLM\n"


def test_prompt_query_param(client_with_mock_llm):
    """curl http://localhost:8000/prompt?text=Hello."""
    response = client_with_mock_llm.get("/prompt?text=Hello+from+query")
    assert response.status_code == 200
    assert response.text == "Echo response from LLM\n"


def test_prompt_empty_returns_400(client_with_mock_llm):
    """Empty body returns 400 Bad Request with usage tips."""
    response = client_with_mock_llm.post("/prompt", content="")
    assert response.status_code == 400
    assert "Error: Prompt body cannot be empty" in response.text
    assert "Usage: curl -d" in response.text
    assert "/prompt" in response.text


def test_prompt_3tier_fallback_execution(client_with_3tier_fallback_llm):
    """When Groq and Gemini Flash fail, /prompt returns Gemini Lite output with 200 OK."""
    response = client_with_3tier_fallback_llm.post("/prompt", content="Why is the sky blue?")
    assert response.status_code == 200
    assert response.text == "Response from Gemini 3.5 Flash Lite\n"


def test_prompt_all_providers_fail(client_with_all_failing_llm):
    """When all providers fail, /prompt returns 502 with aggregated error text."""
    response = client_with_all_failing_llm.post("/prompt", content="Hello")
    assert response.status_code == 502
    assert "All LLM providers failed" in response.text
    assert "Groq API error" in response.text
    assert "Gemini Flash API error" in response.text
    assert "Gemini Lite API error" in response.text


def test_dedicated_routes_success(client_with_mock_llm):
    """Dedicated routes return their respective model responses."""
    routes = {
        "/qwen": "Qwen response\n",
        "/gemini-lite": "Gemini Lite response\n",
        "/gemini": "Gemini response\n",
        "/gpt": "GPT response\n",
        "/gemma": "Gemma response\n",
    }
    for route, expected_text in routes.items():
        res = client_with_mock_llm.post(route, content="Explain relativity")
        assert res.status_code == 200, f"Route {route} failed with {res.status_code}"
        assert res.text == expected_text, f"Route {route} returned unexpected body: {res.text}"


def test_dedicated_route_empty_body(client_with_mock_llm):
    """Empty body on dedicated routes returns 400 with path in usage message."""
    res = client_with_mock_llm.post("/qwen", content="")
    assert res.status_code == 400
    assert "Usage: curl -d \"Your prompt here\" http://localhost:8000/qwen" in res.text
