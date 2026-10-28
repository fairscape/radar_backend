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
def _openalex_not_paused():
    """A 429 one test provokes pauses OpenAlex for the process
    (``rag_lib.openalex_status``); it must not refuse the next test's calls."""
    from rag_lib import openalex_status

    openalex_status.reset()
    yield
    openalex_status.reset()


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


@pytest.fixture(autouse=True)
def _no_chroma(request, monkeypatch):
    """Keep Chroma out of the API tests unless a test asks for it.

    An upload (``vault.upload``) and a chat open the user's Chroma
    collection to count or store chunks. Late in a full run -- after a few
    hundred tests have each opened their own ``PersistentClient`` -- the
    rust binding's ``start`` stopped returning: the process sat idle for
    hours instead of failing (twice, in different tests, 2026-10-21). No
    test here checks the chunks a real collection holds: ``test_api_vault``
    asserts 0 and ``test_api_chat`` stubs the collection itself. A test
    that wants the real thing names ``live_chroma`` in its fixtures.
    """
    if "live_chroma" in request.fixturenames:
        return
    import rag_lib.rag.indexer as rag_indexer
    monkeypatch.setattr(rag_indexer, "index_user_collection", lambda *a, **k: None)
    monkeypatch.setattr(rag_indexer, "index_user_chat_collection", lambda *a, **k: None)


@pytest.fixture
def live_chroma():
    """Opt out of ``_no_chroma``: the test opens a real collection."""
    return True


class _NoPapersGatherer:
    """OpenAlex found nothing: a scan that succeeds and stores no paper."""

    name = "no-papers"
    last_tier_used = None

    def __init__(self, **_):
        pass

    def fetch(self, profile, since, *, limit=None):
        return []

    def fetch_years(self, profile, year_from, year_to, *, per_year):
        return []

    def cost(self):
        return {"api_calls": 0, "wall_seconds": 0.0}


@pytest.fixture(autouse=True)
def _offline_scans(monkeypatch):
    """Scans in tests never reach OpenAlex: by default every one finds
    nothing. Saving an interest runs its first scan (FUNCTIONAL_SPEC D.2-5),
    so every test that saves one scans. A test that wants papers sets its
    own ``scheduler_jobs.OpenAlexGatherer``, which replaces this one."""
    from rag_lib.scheduler import jobs as scheduler_jobs

    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", _NoPapersGatherer)


@pytest.fixture(autouse=True)
def _saves_run_at_once(monkeypatch):
    """Saving an interest hands its first scan to the scheduler and answers
    at once (``jobs.save_draft``). In tests that job runs inside the
    request instead, as the scheduler would run it, so what a test reads
    next is the saved interest. Every other job is scheduled as usual."""
    from rag_lib.api.routers import profiles as profiles_router
    from rag_lib.scheduler import jobs as scheduler_jobs

    schedule = profiles_router._schedule_run

    def run(scheduler, db, run_id, tier, func, *, args, kwargs, id):
        if func is scheduler_jobs.save_draft:
            return func(*args, **kwargs)
        return schedule(scheduler, db, run_id, tier, func, args=args, kwargs=kwargs, id=id)

    monkeypatch.setattr(profiles_router, "_schedule_run", run)
