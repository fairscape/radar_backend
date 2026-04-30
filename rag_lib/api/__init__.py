"""FastAPI service for RADAR.

Mounted at ``/api/`` in front of the SQLite layer in ``rag_lib.db`` and
the existing rag_lib selectors / gatherers / vault. Phase 4 ships the
skeleton (settings, logging, lifespan, /health, stub routers); later
phases fill in the bodies.
"""
