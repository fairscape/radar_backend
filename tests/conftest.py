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
def _no_ambient_env_file(monkeypatch):
    """Keep the deployment's own ``.env`` out of the test suite.

    ``Settings.model_config`` names ``env_file=".env"``, resolved against
    the cwd -- and the cwd for a test run is the repository root, which is
    exactly where the running deployment keeps its ``.env``. So the suite
    silently inherits production configuration. Measured on 2026-09-28:
    that file's ``RADAR_REQUIRE_AUTH=true`` turned 76 tests red with 401s,
    for a suite that was 104/573 failing until the file was accounted for
    and 28/573 after. Nothing in the failure pointed at the cause.

    Tests configure themselves through monkeypatched environment
    variables, which still win; this only removes the file nobody asked
    for. Real environment variables are left alone, so `RADAR_...=x pytest`
    keeps working as an override.
    """
    from rag_lib.api import settings as settings_module

    def _drop_cached_settings() -> None:
        # A test may have replaced get_settings with a plain function, which
        # has no cache to clear -- test_umls_gate does exactly that, and an
        # unguarded call turned its teardown into an error.
        clear = getattr(settings_module.get_settings, "cache_clear", None)
        if clear is not None:
            clear()

    monkeypatch.setitem(settings_module.Settings.model_config, "env_file", None)
    _drop_cached_settings()
    yield
    _drop_cached_settings()


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
