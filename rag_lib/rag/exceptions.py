"""RAG-layer exceptions.

Kept in their own module so callers (router + service) can import them
without pulling in chromadb / httpx / pydantic-ai transitively.
"""

from __future__ import annotations


class LLMUnreachable(RuntimeError):
    """Raised when the active LLM provider can't be reached.

    The router maps this to a 503 with an actionable message that names
    the relevant env var for the active provider (``RADAR_OLLAMA_URL``
    for Ollama, ``RADAR_ANTHROPIC_API_KEY`` for Anthropic, etc.) so the
    operator can diagnose service-down / wrong-host / missing-key cases
    from the error alone.
    """


# Back-compat alias. The Phase 9 chat path raised ``OllamaUnreachable``
# directly; callers and tests still import that name. Keeping it as an
# alias means the rename is a single-file change instead of a fan-out.
OllamaUnreachable = LLMUnreachable


class LLMNotConfigured(RuntimeError):
    """Raised when the requested provider is selected but unusable.

    Two cases:
      * No API key in the environment for a third-party provider.
      * The ``llm-providers`` extras group isn't installed, so the
        provider's client library (``pydantic_ai``) can't be imported.

    The router maps this to a 503 with a hint pointing at the fix.
    """


class RetrievalEmpty(RuntimeError):
    """Raised when retrieval finds no chunks for the request scope.

    Reserved for callers that want to short-circuit before invoking the
    LLM. The default chat path falls through with an empty source list
    and lets the LLM say "no passages contained the answer" — keeping
    this exception available means a future caller (eval harness,
    streaming front-end) can opt into a hard-fail instead.
    """
