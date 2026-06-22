"""LLM provider abstraction.

The rest of the pipeline calls ``get_llm()`` and gets back a LangChain chat
model — it never imports a provider SDK directly. Swapping providers is a
one-line change in ``.env`` (``LLM_PROVIDER`` / ``LLM_MODEL``); the only code
that knows about Anthropic vs OpenAI lives here.

Default: Anthropic Claude (``claude-sonnet-4-6``) via ``langchain-anthropic``,
which wraps the official ``anthropic`` SDK under the hood.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import settings


def get_llm() -> BaseChatModel:
    """Construct the configured chat model. Raises a clear error if its key is missing."""
    cfg = settings.llm

    if cfg.provider == "anthropic":
        if not cfg.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
                "your key (get one at https://console.anthropic.com/settings/keys)."
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    if cfg.provider == "openai":
        # Optional alternative provider — demonstrates the switchable design.
        if not cfg.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set but LLM_PROVIDER=openai.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{cfg.provider}'. Use 'anthropic' or 'openai'."
    )
