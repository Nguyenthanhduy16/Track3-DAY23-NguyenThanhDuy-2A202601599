"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os


def get_llm(model: str | None = None, temperature: float = 0.0):
    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. GEMINI_API_KEY → ChatGoogleGenerativeAI
    2. OPENROUTER_API_KEY → ChatOpenAI routed through OpenRouter
    3. OPENAI_API_KEY → ChatOpenAI
    4. ANTHROPIC_API_KEY → ChatAnthropic

    Override model with the `model` parameter or LLM_MODEL env var.
    """
    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )

    if os.getenv("OPENROUTER_API_KEY"):
        # OpenRouter exposes an OpenAI-compatible API, so the plain OpenAI
        # client works unmodified — just point base_url at OpenRouter and
        # use an OpenRouter-style model id (e.g. "openai/gpt-4o-mini").
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=temperature,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://localhost"),
                "X-Title": os.getenv("OPENROUTER_SITE_NAME", "langgraph-agent-lab"),
            },
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        return ChatAnthropic(
            model=model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key found. Set GEMINI_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY, "
        "or ANTHROPIC_API_KEY in .env\n"
        "See .env.example for configuration."
    )
