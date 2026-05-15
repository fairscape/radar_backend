"""LLM clients for the RAG chat path.

Three providers are wired through a single ``LLMClient`` protocol so
``services.chat.post_chat`` doesn't care which is active:

  * ``OllamaClient`` — local Ollama via httpx (no extras required).
  * ``AnthropicClient`` — Claude via pydantic-ai (optional ``llm-providers`` extras).
  * ``OpenAIClient`` — GPT via pydantic-ai (same extras).

Any transport-level or auth failure normalizes to ``LLMUnreachable`` so
the router has a single exception to map to 503. ``LLMNotConfigured``
is raised at construction time when the operator selects a provider
without a key or without the extras installed — the router maps it to
503 with a hint pointing at the right env var or install command.

``build_prompt`` is provider-agnostic (returns OpenAI-style messages);
each client adapts that shape to its underlying SDK.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

import httpx

from .exceptions import LLMNotConfigured, LLMUnreachable, OllamaUnreachable


SUPPORTED_PROVIDERS: tuple[str, ...] = ("ollama", "anthropic", "openai")


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


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface every provider implementation satisfies."""

    model: str

    def generate(self, messages: list[dict]) -> str:
        """Return the assistant text for the OpenAI-shaped ``messages`` list.

        Implementations raise ``LLMUnreachable`` on transport / auth /
        empty-response failures so the router can map them uniformly.
        """
        ...


class OllamaClient:
    """Minimal POST /api/chat client.

    Construction does not touch the network. ``generate`` raises
    ``LLMUnreachable`` (aliased as ``OllamaUnreachable`` for back-compat)
    on any failure to reach the server or get a valid response back.
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
            raise LLMUnreachable(
                f"could not reach Ollama at {self.url}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise LLMUnreachable(
                f"Ollama returned HTTP {response.status_code} "
                f"from {endpoint}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMUnreachable(
                f"Ollama returned non-JSON response from {endpoint}: {exc}"
            ) from exc

        message = data.get("message") if isinstance(data, dict) else None
        content = (message or {}).get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMUnreachable(
                f"Ollama response from {endpoint} missing message.content"
            )
        return content


def _split_system(messages: list[dict]) -> tuple[str, str]:
    """Peel the leading ``system`` message off ``messages``.

    pydantic-ai's ``Agent`` takes the system prompt at construction
    time and a single user prompt to ``run_sync``. Our ``build_prompt``
    output is always ``[system, user]`` so this is a simple split; we
    still defend against a missing system part by falling back to the
    module-level ``SYSTEM_PROMPT``.
    """
    system = SYSTEM_PROMPT
    user_parts: list[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            system = content
        else:
            user_parts.append(content)
    return system, "\n\n".join(user_parts).strip()


def _build_pydantic_ai_agent(
    provider_name: str,
    api_key: str,
    model_name: str,
    *,
    timeout: float,
    system_prompt: str,
):
    """Construct a pydantic-ai ``Agent`` for the given provider.

    Imports are local so the demo install (without the ``llm-providers``
    extras) can still import this module — the failure surfaces only
    when an operator actually selects Anthropic/OpenAI.
    """
    try:
        from pydantic_ai import Agent
        from pydantic_ai.settings import ModelSettings
    except ImportError as exc:
        raise LLMNotConfigured(
            f"pydantic-ai not installed — run "
            f"`pip install 'rag_lib[llm-providers]'` to enable {provider_name}"
        ) from exc

    if provider_name == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        model = AnthropicModel(
            model_name,
            provider=AnthropicProvider(api_key=api_key),
        )
    elif provider_name == "openai":
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider

        model = OpenAIModel(
            model_name,
            provider=OpenAIProvider(api_key=api_key),
        )
    else:
        raise ValueError(f"unsupported pydantic-ai provider {provider_name!r}")

    return Agent(
        model,
        system_prompt=system_prompt,
        model_settings=ModelSettings(temperature=0.2, timeout=timeout),
    )


def _extract_output(result: Any) -> str:
    """Read assistant text from a pydantic-ai ``run_sync`` result.

    The attribute name shifted from ``data`` to ``output`` around
    0.0.40; supporting both keeps us forward-compatible without pinning
    the SDK to a single point release.
    """
    for attr in ("output", "data"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    raise LLMUnreachable("LLM returned an empty response")


class AnthropicClient:
    """Claude via pydantic-ai. Network only on ``generate``."""

    def __init__(self, api_key: str, model: str, *, timeout: float = 120.0) -> None:
        if not api_key:
            raise LLMNotConfigured(
                "Anthropic selected but RADAR_ANTHROPIC_API_KEY is not set"
            )
        if not model:
            raise ValueError("AnthropicClient requires a non-empty model")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, messages: list[dict]) -> str:
        system, user = _split_system(messages)
        agent = _build_pydantic_ai_agent(
            "anthropic",
            self.api_key,
            self.model,
            timeout=self.timeout,
            system_prompt=system,
        )
        try:
            result = agent.run_sync(user)
        except LLMNotConfigured:
            raise
        except Exception as exc:
            raise LLMUnreachable(
                f"Anthropic request failed: {type(exc).__name__}: {exc}"
            ) from exc
        return _extract_output(result)


class OpenAIClient:
    """GPT via pydantic-ai. Network only on ``generate``."""

    def __init__(self, api_key: str, model: str, *, timeout: float = 120.0) -> None:
        if not api_key:
            raise LLMNotConfigured(
                "OpenAI selected but RADAR_OPENAI_API_KEY is not set"
            )
        if not model:
            raise ValueError("OpenAIClient requires a non-empty model")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, messages: list[dict]) -> str:
        system, user = _split_system(messages)
        agent = _build_pydantic_ai_agent(
            "openai",
            self.api_key,
            self.model,
            timeout=self.timeout,
            system_prompt=system,
        )
        try:
            result = agent.run_sync(user)
        except LLMNotConfigured:
            raise
        except Exception as exc:
            raise LLMUnreachable(
                f"OpenAI request failed: {type(exc).__name__}: {exc}"
            ) from exc
        return _extract_output(result)


def resolve_provider(settings: Any, provider: str | None = None) -> str:
    """Resolve the active provider name (request override → settings default)."""
    name = (provider or getattr(settings, "RADAR_LLM_PROVIDER", "ollama") or "ollama").lower()
    if name not in SUPPORTED_PROVIDERS:
        raise ValueError(f"unknown LLM provider {name!r}")
    return name


def provider_model(settings: Any, provider: str) -> str:
    """The model identifier the operator has configured for ``provider``."""
    if provider == "ollama":
        return settings.RADAR_OLLAMA_MODEL
    if provider == "anthropic":
        return settings.RADAR_ANTHROPIC_MODEL
    if provider == "openai":
        return settings.RADAR_OPENAI_MODEL
    raise ValueError(f"unknown LLM provider {provider!r}")


def provider_configured(settings: Any, provider: str) -> bool:
    """True when the operator has provided everything ``provider`` needs.

    Ollama is treated as always-configured (the URL has a default and a
    network failure surfaces later as ``LLMUnreachable``). Third-party
    providers require a non-empty API key in the environment.
    """
    if provider == "ollama":
        return bool(getattr(settings, "RADAR_OLLAMA_URL", ""))
    if provider == "anthropic":
        return bool(getattr(settings, "RADAR_ANTHROPIC_API_KEY", "") or "")
    if provider == "openai":
        return bool(getattr(settings, "RADAR_OPENAI_API_KEY", "") or "")
    return False


def get_llm_client(settings: Any, provider: str | None = None) -> LLMClient:
    """Construct the active provider's client.

    Resolves the provider (request override → settings default),
    validates that the corresponding key / extras are available, and
    returns a client ready to ``.generate(messages)``. Raises
    ``ValueError`` for an unknown provider and ``LLMNotConfigured``
    for a known-but-unusable one.
    """
    name = resolve_provider(settings, provider)
    if name == "ollama":
        return OllamaClient(
            settings.RADAR_OLLAMA_URL,
            settings.RADAR_OLLAMA_MODEL,
            timeout=settings.RADAR_OLLAMA_TIMEOUT,
        )
    timeout = float(getattr(settings, "RADAR_LLM_TIMEOUT", 120.0))
    if name == "anthropic":
        return AnthropicClient(
            settings.RADAR_ANTHROPIC_API_KEY or "",
            settings.RADAR_ANTHROPIC_MODEL,
            timeout=timeout,
        )
    if name == "openai":
        return OpenAIClient(
            settings.RADAR_OPENAI_API_KEY or "",
            settings.RADAR_OPENAI_MODEL,
            timeout=timeout,
        )
    raise ValueError(f"unknown LLM provider {name!r}")


def build_prompt(query: str, retrieved_chunks: Iterable[dict]) -> list[dict]:
    """Compose system + user messages from retrieved chunks.

    Each chunk gets a ``[N]`` marker (1-indexed) the LLM is instructed
    to cite. Long chunks are truncated to ~500 whitespace tokens to
    keep total context under the default Ollama window.
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


__all__ = [
    "AnthropicClient",
    "LLMClient",
    "LLMNotConfigured",
    "LLMUnreachable",
    "OllamaClient",
    "OllamaUnreachable",
    "OpenAIClient",
    "SUPPORTED_PROVIDERS",
    "SYSTEM_PROMPT",
    "build_prompt",
    "get_llm_client",
    "provider_configured",
    "provider_model",
    "resolve_provider",
]
