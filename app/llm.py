"""Unified LLM interaction layer using Pydantic-AI with Groq primary and Gemini fallback."""

import logging
from typing import Any
from pydantic_ai import Agent
from pydantic_ai.exceptions import FallbackExceptionGroup
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.groq import GroqProvider

from app.config import Settings, get_settings

logger = logging.getLogger("unified_llm")


def _should_fallback_to_gemini(exc: Exception) -> bool:
    """Predicate called when the primary model (Groq) fails.

    Logs a warning with the error details and returns True to trigger Gemini fallback.
    """
    logger.warning("Primary Groq API call failed (%s: %s). Falling back to Gemini API...", type(exc).__name__, exc)
    return True


class UnifiedLLM:
    """Unified LLM interface managing Groq primary and Gemini fallback via Pydantic-AI."""

    def __init__(
        self,
        agent: Agent | None = None,
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self._custom_agent = agent
        self._agent: Agent | None = agent

    def _build_agent(self) -> Agent:
        """Constructs the Pydantic-AI agent according to available API credentials."""
        settings = self._settings
        groq_key = settings.groq_api_key
        gemini_key = settings.effective_gemini_api_key

        if groq_key and gemini_key:
            logger.info(
                "Configuring FallbackModel: Groq ('%s') primary -> Gemini ('%s') fallback",
                settings.groq_model,
                settings.gemini_model,
            )
            groq_provider = GroqProvider(api_key=groq_key)
            groq_model = GroqModel(settings.groq_model, provider=groq_provider)

            gemini_provider = GoogleProvider(api_key=gemini_key)
            gemini_model = GoogleModel(settings.gemini_model, provider=gemini_provider)

            fallback_model = FallbackModel(
                groq_model,
                gemini_model,
                fallback_on=_should_fallback_to_gemini,
            )
            return Agent(fallback_model, system_prompt=settings.system_prompt)

        if groq_key:
            logger.info("Configuring single model: Groq ('%s')", settings.groq_model)
            groq_provider = GroqProvider(api_key=groq_key)
            groq_model = GroqModel(settings.groq_model, provider=groq_provider)
            return Agent(groq_model, system_prompt=settings.system_prompt)

        if gemini_key:
            logger.info("Configuring single model: Gemini ('%s')", settings.gemini_model)
            gemini_provider = GoogleProvider(api_key=gemini_key)
            gemini_model = GoogleModel(settings.gemini_model, provider=gemini_provider)
            return Agent(gemini_model, system_prompt=settings.system_prompt)

        raise RuntimeError(
            "No LLM API keys configured. Please set GROQ_API_KEY (for Groq primary) "
            "and/or GEMINI_API_KEY (for Gemini fallback) in your .env file or environment."
        )

    def get_agent(self) -> Agent:
        """Retrieves or builds the active Pydantic-AI agent."""
        if self._custom_agent is not None:
            return self._custom_agent
        if self._agent is None:
            self._agent = self._build_agent()
        return self._agent

    async def generate(self, prompt: str) -> str:
        """Passes the user prompt to the model (Groq first, fallback to Gemini).

        Returns the response text string.
        """
        agent = self.get_agent()
        try:
            result = await agent.run(prompt)
            # Log any fallback attempts recorded in response
            failed_attempts = getattr(result.response, "failed_attempts", None)
            if failed_attempts:
                for attempt in failed_attempts:
                    logger.warning(
                        "Fallback succeeded after model '%s' failed: %s",
                        attempt.model_name,
                        attempt.error,
                    )
            return str(result.output)
        except FallbackExceptionGroup as e:
            err_details = "; ".join(f"{type(x).__name__}: {x}" for x in getattr(e, "exceptions", []))
            logger.error("Both primary and fallback models failed: %s", err_details)
            raise RuntimeError(f"All LLM providers failed: {err_details}") from e
        except Exception as e:
            logger.error("LLM execution error: %s", e)
            raise
