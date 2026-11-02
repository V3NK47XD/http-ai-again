"""Tests for UnifiedLLM multi-tier fallback and dedicated model routes."""

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.llm import UnifiedLLM, _should_fallback_to_next


class FailingModel(TestModel):
    """Model that always raises an error to simulate API failure."""

    def __init__(self, error_message: str = "Provider failure: 429 Rate Limit"):
        super().__init__()
        self.error_message = error_message

    async def request(self, *args, **kwargs):
        raise RuntimeError(self.error_message)


@pytest.mark.asyncio
async def test_groq_primary_succeeds():
    """When Groq succeeds, Groq response is returned and fallbacks are not called."""
    groq_mock = TestModel(custom_output_text="Groq primary response")
    gemini_mock = TestModel(custom_output_text="Gemini fallback response")
    gemini_lite_mock = TestModel(custom_output_text="Gemini Lite fallback response")

    fallback_model = FallbackModel(groq_mock, gemini_mock, gemini_lite_mock, fallback_on=_should_fallback_to_next)
    agent = Agent(fallback_model)
    llm = UnifiedLLM(agent=agent)

    result = await llm.generate("Explain black holes")
    assert result == "Groq primary response"


@pytest.mark.asyncio
async def test_groq_fails_gemini_fallback_succeeds():
    """When Groq encounters an error, 1st fallback (Gemini Flash) handles the request."""
    groq_failing = FailingModel("Groq API 429 Rate limit exceeded")
    gemini_mock = TestModel(custom_output_text="Gemini 3.5 Flash response")
    gemini_lite_mock = TestModel(custom_output_text="Gemini Lite fallback response")

    fallback_model = FallbackModel(groq_failing, gemini_mock, gemini_lite_mock, fallback_on=_should_fallback_to_next)
    agent = Agent(fallback_model)
    llm = UnifiedLLM(agent=agent)

    result = await llm.generate("Explain quantum computing")
    assert result == "Gemini 3.5 Flash response"


@pytest.mark.asyncio
async def test_groq_and_gemini_fail_gemini_lite_fallback_succeeds():
    """When Groq and Gemini Flash both fail, 2nd fallback (Gemini 3.5 Flash Lite) handles request."""
    groq_failing = FailingModel("Groq API 500 Internal Error")
    gemini_failing = FailingModel("Gemini Flash 429 Quota Exceeded")
    gemini_lite_mock = TestModel(custom_output_text="Gemini 3.5 Flash Lite rescue response")

    fallback_model = FallbackModel(groq_failing, gemini_failing, gemini_lite_mock, fallback_on=_should_fallback_to_next)
    agent = Agent(fallback_model)
    llm = UnifiedLLM(agent=agent)

    result = await llm.generate("Explain general relativity")
    assert result == "Gemini 3.5 Flash Lite rescue response"


@pytest.mark.asyncio
async def test_all_three_models_fail_raises_runtime_error():
    """When all 3 tiers fail, a descriptive RuntimeError is raised with all failure reasons."""
    m1 = FailingModel("Groq down (503)")
    m2 = FailingModel("Gemini Flash quota exceeded (429)")
    m3 = FailingModel("Gemini Flash Lite timeout (504)")

    fallback_model = FallbackModel(m1, m2, m3, fallback_on=_should_fallback_to_next)
    agent = Agent(fallback_model)
    llm = UnifiedLLM(agent=agent)

    with pytest.raises(RuntimeError) as exc_info:
        await llm.generate("Test prompt")

    err_str = str(exc_info.value)
    assert "All LLM providers failed" in err_str
    assert "Groq down" in err_str
    assert "Gemini Flash quota exceeded" in err_str
    assert "Gemini Flash Lite timeout" in err_str


@pytest.mark.asyncio
async def test_named_agents_generation():
    """Named routes (qwen, gemini-lite, gemini, gpt, gemma) invoke respective models."""
    named = {
        "qwen": Agent(TestModel(custom_output_text="Qwen 3.8-27b response")),
        "gemini-lite": Agent(TestModel(custom_output_text="Gemini 3.5 Flash Lite response")),
        "gemini": Agent(TestModel(custom_output_text="Gemini 3.5 Flash response")),
        "gpt": Agent(TestModel(custom_output_text="GPT-OSS 120b response")),
        "gemma": Agent(TestModel(custom_output_text="Gemma 4-31b-it response")),
    }
    llm = UnifiedLLM(named_agents=named)

    assert await llm.generate_named("qwen", "hi") == "Qwen 3.8-27b response"
    assert await llm.generate_named("gemini-lite", "hi") == "Gemini 3.5 Flash Lite response"
    assert await llm.generate_named("gemini", "hi") == "Gemini 3.5 Flash response"
    assert await llm.generate_named("gpt", "hi") == "GPT-OSS 120b response"
    assert await llm.generate_named("gemma", "hi") == "Gemma 4-31b-it response"


def test_no_keys_configured_raises_helpful_error():
    """When neither GROQ_API_KEY nor GEMINI_API_KEY is configured, clear error is thrown."""
    settings = Settings(
        GROQ_API_KEY=None,
        GEMINI_API_KEY=None,
        GOOGLE_API_KEY=None,
    )
    llm = UnifiedLLM(settings=settings)
    with pytest.raises(RuntimeError) as exc_info:
        llm.get_agent()

    assert "No LLM API keys configured" in str(exc_info.value)


def test_dedicated_route_missing_key():
    """Dedicated routes raise clear errors when required provider key is missing."""
    settings = Settings(
        GROQ_API_KEY=None,
        GEMINI_API_KEY=None,
        GOOGLE_API_KEY=None,
    )
    llm = UnifiedLLM(settings=settings)
    with pytest.raises(RuntimeError) as exc_info_groq:
        llm.get_named_agent("qwen")
    assert "GROQ_API_KEY is not configured" in str(exc_info_groq.value)

    with pytest.raises(RuntimeError) as exc_info_gemini:
        llm.get_named_agent("gemini")
    assert "GEMINI_API_KEY" in str(exc_info_gemini.value)
