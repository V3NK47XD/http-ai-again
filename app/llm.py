"""Unified LLM interaction layer using Pydantic-AI with Groq primary and Gemini fallbacks."""

import logging
from typing import Any
from pydantic_ai import Agent
from pydantic_ai.exceptions import FallbackExceptionGroup
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.groq import GroqProvider

from app.config import Settings, get_settings

logger = logging.getLogger("unified_llm")


def _should_fallback_to_next(exc: Exception) -> bool:
    """Predicate called when a model fails in the fallback chain.

    Logs a warning with error details and returns True to advance to the next fallback model.
    """
    logger.warning("Model call failed (%s: %s). Advancing to next fallback model...", type(exc).__name__, exc)
    return True


class UnifiedLLM:
    """Unified LLM interface managing multi-tier fallbacks and dedicated model routes."""

    def __init__(
        self,
        agent: Agent | None = None,
        named_agents: dict[str, Agent] | None = None,
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self._custom_agent = agent
        self._agent: Agent | None = agent
        self._named_agents: dict[str, Agent] = named_agents.copy() if named_agents else {}

    def _get_groq_provider(self) -> GroqProvider:
        key = self._settings.groq_api_key
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your .env file or environment."
            )
        return GroqProvider(api_key=key)

    def _get_gemini_provider(self) -> GoogleProvider:
        key = self._settings.effective_gemini_api_key
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not configured. "
                "Please set GEMINI_API_KEY in your .env file or environment."
            )
        return GoogleProvider(api_key=key)

    def _build_fallback_agent(self) -> Agent:
        """Constructs the multi-tier agent for /prompt according to available credentials.

        Tier 1: Groq API (e.g. llama-3.3-70b-versatile)
        Tier 2 (Fallback 1): Gemini API (e.g. gemini-3.5-flash)
        Tier 3 (Fallback 2): Gemini API (e.g. gemini-3.5-flash-lite)
        """
        settings = self._settings
        groq_key = settings.groq_api_key
        gemini_key = settings.effective_gemini_api_key

        if groq_key and gemini_key:
            logger.info(
                "Configuring 3-tier FallbackModel: Groq ('%s') -> Gemini ('%s') -> Gemini Lite ('%s')",
                settings.groq_model,
                settings.gemini_model,
                settings.gemini_lite_model,
            )
            groq_provider = GroqProvider(api_key=groq_key)
            groq_model = GroqModel(settings.groq_model, provider=groq_provider)

            gemini_provider = GoogleProvider(api_key=gemini_key)
            gemini_model = GoogleModel(settings.gemini_model, provider=gemini_provider)
            gemini_lite_model = GoogleModel(settings.gemini_lite_model, provider=gemini_provider)

            fallback_model = FallbackModel(
                groq_model,
                gemini_model,
                gemini_lite_model,
                fallback_on=_should_fallback_to_next,
            )
            return Agent(fallback_model, system_prompt=settings.system_prompt)

        if gemini_key:
            logger.info(
                "Configuring 2-tier Gemini fallback: Gemini ('%s') -> Gemini Lite ('%s')",
                settings.gemini_model,
                settings.gemini_lite_model,
            )
            gemini_provider = GoogleProvider(api_key=gemini_key)
            gemini_model = GoogleModel(settings.gemini_model, provider=gemini_provider)
            gemini_lite_model = GoogleModel(settings.gemini_lite_model, provider=gemini_provider)

            fallback_model = FallbackModel(
                gemini_model,
                gemini_lite_model,
                fallback_on=_should_fallback_to_next,
            )
            return Agent(fallback_model, system_prompt=settings.system_prompt)

        if groq_key:
            logger.info("Configuring single model: Groq ('%s')", settings.groq_model)
            groq_provider = GroqProvider(api_key=groq_key)
            groq_model = GroqModel(settings.groq_model, provider=groq_provider)
            return Agent(groq_model, system_prompt=settings.system_prompt)

        raise RuntimeError(
            "No LLM API keys configured. Please set GROQ_API_KEY (for Groq primary) "
            "and/or GEMINI_API_KEY (for Gemini fallbacks) in your .env file or environment."
        )

    def get_agent(self) -> Agent:
        """Retrieves or builds the active Pydantic-AI agent for /prompt."""
        if self._custom_agent is not None:
            return self._custom_agent
        if self._agent is None:
            self._agent = self._build_fallback_agent()
        return self._agent

    def get_named_agent(self, name: str) -> Agent:
        """Retrieves or builds a dedicated model agent by route identifier."""
        if name in self._named_agents:
            return self._named_agents[name]

        settings = self._settings
        agent: Agent

        if name == "qwen":
            # qwen/qwen3.8-27b (groq api)
            provider = self._get_groq_provider()
            agent = Agent(GroqModel(settings.qwen_model, provider=provider), system_prompt=settings.system_prompt)
        elif name == "gpt":
            # openai/gpt-oss-120b (groq api)
            provider = self._get_groq_provider()
            agent = Agent(GroqModel(settings.gpt_model, provider=provider), system_prompt=settings.system_prompt)
        elif name == "gemini-lite":
            # gemini-3.5-flash-lite (gemini api)
            provider = self._get_gemini_provider()
            agent = Agent(GoogleModel(settings.gemini_lite_model, provider=provider), system_prompt=settings.system_prompt)
        elif name == "gemini":
            # gemini-3.5-flash (gemini api)
            provider = self._get_gemini_provider()
            agent = Agent(GoogleModel(settings.gemini_model, provider=provider), system_prompt=settings.system_prompt)
        elif name == "gemma":
            # gemma-4-31b-it (gemini api)
            provider = self._get_gemini_provider()
            agent = Agent(GoogleModel(settings.gemma_model, provider=provider), system_prompt=settings.system_prompt)
        elif name == "prompt":
            agent = self.get_agent()
        else:
            raise ValueError(f"Unknown model route: {name}")

        self._named_agents[name] = agent
        return agent

    async def _execute_agent(self, agent: Agent, prompt: str) -> str:
        """Runs an agent and formats the output or error."""
        try:
            result = await agent.run(prompt)
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
            logger.error("All models in fallback chain failed: %s", err_details)
            raise RuntimeError(f"All LLM providers failed: {err_details}") from e
        except Exception as e:
            logger.error("LLM execution error: %s", e)
            raise

    async def generate(self, prompt: str) -> str:
        """Runs prompt through the 3-tier fallback chain (/prompt)."""
        agent = self.get_agent()
        return await self._execute_agent(agent, prompt)

    async def generate_named(self, name: str, prompt: str) -> str:
        """Runs prompt through a specific dedicated model agent."""
        agent = self.get_named_agent(name)
        return await self._execute_agent(agent, prompt)
