"""Tests for UnifiedLLM fallback and execution behavior."""

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.llm import UnifiedLLM, _should_fallback_to_gemini


class FailingModel(TestModel):
    """Model that always raises an error to simulate API failure."""

    def __init__(self, error_message: str = "Provider failure: 429 Rate Limit"):
        super().__init__()
        self.error_message = error_message

    async def request(self, *args, **kwargs):
        raise RuntimeError(self.error_message)


@pytest.mark.asyncio
async def test_groq_primary_succeeds():
    """When Groq succeeds, Groq response is returned and Gemini is not needed."""
    groq_mock = TestModel(custom_output_text="Groq primary response")
    gemini_mock = TestModel(custom_output_text="Gemini fallback response")

    fallback_model = FallbackModel(groq_mock, gemini_mock, fallback_on=_should_fallback_to_gemini)
    agent = Agent(fallback_model)
    llm = UnifiedLLM(agent=agent)

    result = await llm.generate("Explain black holes")
    assert result == "Groq primary response"


@pytest.mark.asyncio
async def test_groq_fails_gemini_fallback_succeeds():
    """When Groq encounters an error, FallbackModel catches it and uses Gemini."""
    groq_failing = FailingModel("Groq API 429 Rate limit exceeded")
    gemini_mock = TestModel(custom_output_text="Gemini fallback succeeded")

    fallback_model = FallbackModel(groq_failing, gemini_mock, fallback_on=_should_fallback_to_gemini)
    agent = Agent(fallback_model)
    llm = UnifiedLLM(agent=agent)

    result = await llm.generate("Explain quantum computing")
    assert result == "Gemini fallback succeeded"


@pytest.mark.asyncio
async def test_both_models_fail_raises_runtime_error():
    """When both primary and fallback fail, a descriptive RuntimeError is raised."""
    groq_failing = FailingModel("Groq down (503)")
    gemini_failing = FailingModel("Gemini quota exceeded (429)")

    fallback_model = FallbackModel(groq_failing, gemini_failing, fallback_on=_should_fallback_to_gemini)
    agent = Agent(fallback_model)
    llm = UnifiedLLM(agent=agent)

    with pytest.raises(RuntimeError) as exc_info:
        await llm.generate("Test prompt")

    err_str = str(exc_info.value)
    assert "All LLM providers failed" in err_str
    assert "Groq down" in err_str
    assert "Gemini quota exceeded" in err_str


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
