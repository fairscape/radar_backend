"""Ollama client.

Thin wrapper over ``POST {url}/api/chat``. We use httpx (already a
runtime dep) instead of the ``ollama`` SDK so the demo does not need
the phase1b extras to run the chat path. Any transport-level failure
(connection error, timeout, non-2xx) is normalized to ``OllamaUnreachable``
so the router has a single exception to map to 503.

``build_prompt`` is the deterministic prompt-template helper Phase 9
ships; future iterations can swap in a richer template without changing
the call sites.
"""

from __future__ import annotations

from typing import Any, Iterable

import httpx

from .exceptions import OllamaUnreachable


SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using "
    "ONLY information that appears literally in the numbered passages "
    "below. Do not use outside knowledge.\n"
    "\n"
    "Rules:\n"
    "1. For every factual claim, immediately follow it with the [N] "
    "marker of the passage that LITERALLY contains the words supporting "
    "that claim. The passage text must back the exact claim — do not "
    "cite a passage on a related topic if it does not state the fact.\n"
    "2. When useful, quote the supporting sentence verbatim in "
    "quotation marks before the citation.\n"
    "3. If no passage states the answer, reply exactly: "
    "\"The provided passages do not state this.\" — do not guess and "
    "do not draw on training data.\n"
    "4. Never fabricate a citation. If a [N] marker doesn't actually "
    "support the claim next to it, you have made an error."
)


class OllamaClient:
    """Minimal POST /api/chat client.

    Construction does not touch the network. ``generate`` raises
    ``OllamaUnreachable`` on any failure to reach the server or get a
    valid response back.
    """

    def __init__(self, url: str, model: str, *, timeout: float = 60.0) -> None:
        if not url:
            raise ValueError("OllamaClient requires a non-empty url")
        if not model:
            raise ValueError("OllamaClient requires a non-empty model")
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, messages: list[dict]) -> str:
        """Send ``messages`` and return the assistant text.

        Raises ``OllamaUnreachable`` on connection error, timeout, or
        non-2xx response. The error string carries the underlying
        exception class so the operator-facing 503 detail can pinpoint
        whether it's a transport failure or a server-side error.
        """
        endpoint = f"{self.url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Ollama defaults num_ctx to 2048 regardless of the model's
            # native window. qwen2.5 natively supports 32k. 16k fits a
            # 10-chunk RAG prompt (~10k tokens) plus answer with room
            # to spare; on qwen2.5:7b Q4 it costs ~0.9 GB of KV cache
            # on top of ~5 GB model+overhead — comfortable on an 8 GB
            # GTX 1080. Bump to 32768 if you have more VRAM (tight on
            # 8 GB), or drop to 8192 if running on a smaller GPU.
            "options": {
                "num_ctx": 16384,
                # Low temperature keeps the model anchored to the
                # retrieved passages instead of wandering into
                # training-data priors. RAG quality is sensitive to
                # this — at default (0.7+) qwen2.5 will happily make
                # up plausible-sounding details. 0.2 is conservative
                # without being deterministic.
                "temperature": 0.2,
            },
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, json=payload)
        except httpx.HTTPError as exc:
            raise OllamaUnreachable(
                f"could not reach Ollama at {self.url}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise OllamaUnreachable(
                f"Ollama returned HTTP {response.status_code} "
                f"from {endpoint}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaUnreachable(
                f"Ollama returned non-JSON response from {endpoint}: {exc}"
            ) from exc

        message = data.get("message") if isinstance(data, dict) else None
        content = (message or {}).get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise OllamaUnreachable(
                f"Ollama response from {endpoint} missing message.content"
            )
        return content


def build_prompt(query: str, retrieved_chunks: Iterable[dict]) -> list[dict]:
    """Compose system + user messages from retrieved chunks.

    Each chunk gets a ``[N]`` marker (1-indexed) the LLM is instructed
    to cite. Long chunks are truncated to ~300 whitespace tokens to
    keep total context under ~4k for the default Ollama model.
    """
    passage_lines: list[str] = []
    for n, chunk in enumerate(retrieved_chunks, start=1):
        title = (chunk.get("title") or "").strip() or "untitled"
        text = (chunk.get("text") or "").strip()
        text = _truncate_words(text, max_words=500)
        passage_lines.append(f"[{n}] {title} — {text}")

    if passage_lines:
        passages_block = "\n".join(passage_lines)
    else:
        passages_block = "(no passages were retrieved for this query.)"

    user_content = (
        f"Passages:\n{passages_block}\n\n"
        f"Question: {query}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _truncate_words(text: str, *, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


# Re-exported so callers can ``from rag_lib.rag.llm import OllamaUnreachable``
# without bouncing through ``exceptions``.
__all__ = ["OllamaClient", "OllamaUnreachable", "build_prompt", "SYSTEM_PROMPT"]


def _ensure_imported() -> Any:
    # Touch the symbol so static analyzers don't drop the re-export.
    return OllamaUnreachable
