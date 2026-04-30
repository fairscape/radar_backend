"""RAG plumbing for chat: chunking, indexing, retrieval, LLM client.

The chat service composes these pieces:
  - ``indexer.index_paper`` writes chunks of a vault doc into the user's
    Chroma collection at upload time.
  - ``retriever.retrieve`` pulls top-k chunks at query time, optionally
    filtered by a list of profile slugs.
  - ``llm.OllamaClient`` talks to a local Ollama; when unreachable it
    raises ``OllamaUnreachable`` which the router maps to 503.

Chroma is a phase1b extra; imports are deferred inside helpers so this
package stays importable without it.
"""

from .exceptions import OllamaUnreachable, RetrievalEmpty

__all__ = ["OllamaUnreachable", "RetrievalEmpty"]
