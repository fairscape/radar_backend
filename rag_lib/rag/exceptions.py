"""RAG-layer exceptions.

Kept in their own module so callers (router + service) can import them
without pulling in chromadb / httpx transitively.
"""

from __future__ import annotations


class OllamaUnreachable(RuntimeError):
    """Raised when the configured Ollama endpoint can't be reached.

    The router maps this to a 503 with an actionable message that names
    ``RADAR_OLLAMA_URL`` and the configured model so the operator can
    fix the failure mode (service down, wrong host, model not pulled).
    """


class RetrievalEmpty(RuntimeError):
    """Raised when retrieval finds no chunks for the request scope.

    Reserved for callers that want to short-circuit before invoking the
    LLM. The default chat path falls through with an empty source list
    and lets the LLM say "no passages contained the answer" — keeping
    this exception available means a future caller (eval harness,
    streaming front-end) can opt into a hard-fail instead.
    """
