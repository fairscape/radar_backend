"""UMLS concept extraction and OpenAlex topic mapping.

All imports are lazy so the package can be imported without scispacy
installed — callers check ``settings.RADAR_UMLS_ENABLED`` before
touching the heavy modules.
"""

from __future__ import annotations


def extract_umls_concepts(*args, **kwargs):
    from .extractor import extract_umls_concepts as _fn
    return _fn(*args, **kwargs)


def map_concepts_to_topics(*args, **kwargs):
    from .topic_mapper import map_concepts_to_topics as _fn
    return _fn(*args, **kwargs)


def merge_umls_topics(*args, **kwargs):
    from .topic_mapper import merge_umls_topics as _fn
    return _fn(*args, **kwargs)
