#!/bin/bash
# Ollama for RADAR — embeddings only.
#
# UMLS concept extraction vectorises its concepts through Ollama, so the
# feature is dead without this running. Chat also routes here, but the
# chat model is deliberately not pulled, so chat stays unavailable.
#
# Loopback only: this is an unauthenticated inference endpoint and must
# never be reachable from the network. Caddy doesn't proxy to it either
# — only the backend talks to it.
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS=/bigtemp/nkw3mr/ollama/models
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_MAX_LOADED_MODELS=1
# CUDA, not Vulkan (2026-11-11, records/35): Ollama 0.32.7 picked Vulkan on
# every start, ~1 s an embedding; on CUDA ~0.1 s, the same vectors.
export OLLAMA_VULKAN=0
export GGML_VK_VISIBLE_DEVICES=

exec /bigtemp/nkw3mr/ollama/bin/ollama serve
