"""Shared test fixtures.

The suites here are offline by design: OpenAlex, Prosopia and the
embedders are all stubbed or injected. ``rag_lib.orcid_registry`` is the
one thing that reaches the network without being handed a client --
``list_works`` calls ``fetch_claimed`` itself, because the annotation is
advisory and the caller should not have to care. That is convenient in
production and wrong in a test suite: the ORCID iDs in these fixtures are
real, so the calls succeed, the assertions then depend on what those
records happen to contain today, and every run pays the latency.

So it is stubbed for every test by default. A test that wants the real
thing asks for ``live_orcid_registry``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_orcid_registry(request, monkeypatch):
    """Make ``fetch_claimed`` return None unless a test opts out.

    ``None`` is the "registry unreadable" answer, which is a state the
    production code already has to handle, so stubbing it does not put the
    tests on a path the deployment never takes.
    """
    if "live_orcid_registry" in request.fixturenames:
        return
    import rag_lib.orcid_registry as reg
    monkeypatch.setattr(reg, "fetch_claimed", lambda *a, **k: None)
    # services.orcid imported the name directly, so patching the module
    # attribute alone would leave that binding pointing at the real one.
    import rag_lib.api.services.orcid as orcid_service
    monkeypatch.setattr(orcid_service, "fetch_claimed", lambda *a, **k: None)


@pytest.fixture
def live_orcid_registry():
    """Opt back in to the real pub.orcid.org. Marks the test as networked."""
    return True
