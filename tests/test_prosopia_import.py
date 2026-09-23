"""Prosopia import — the resolution ladder and the import endpoint.

No network on either side. The Prosopia side is a stand-in for the SDK's
``ResearcherProfile`` — real ``PaperRecord`` objects, a name, and a lazy
summaries mapping — because that object, not an HTTP response, is what
RADAR now consumes. The OpenAlex side is a stub whose canned answers are
keyed by the identifier each rung uses, so a test can say "this record
carries only a DOI" and assert that the DOI rung is the one that caught
it.

The rung a paper came in on is the thing worth testing. Every paper ends
up as a seed either way — the failure mode this guards against is a
profile that *looks* imported while actually having been reconstructed
by fuzzy title search, or worse, matched to the wrong paper entirely.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import httpx
import pytest
from researcher_profiles import PaperRecord

from rag_lib.api import settings as settings_module
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import users as users_repo
from rag_lib.openalex_client import OpenAlexClient
from rag_lib.paper import Paper
from rag_lib.prosopia_client import ProfileNotFound, ProsopiaError, read_summary
from rag_lib.api.services.prosopia import normalize_ref
from rag_lib.prosopia_seeds import (
    WorkCache,
    prefetch_works,
    resolve_seed,
    synthetic_id,
)
from tests.fake_openalex_client import canned_openalex_work


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubOpenAlex:
    """The OpenAlexClient surface the ladder uses, answered from dicts.

    Each rung reads its own map, so a fixture can make exactly one rung
    succeed. ``api_calls`` counts every lookup, which is how the
    prefetch tests prove the batch did the work.
    """

    def __init__(
        self,
        *,
        by_id: dict[str, dict] | None = None,
        by_doi: dict[str, dict] | None = None,
        by_pmcid: dict[str, dict] | None = None,
        title_hits: dict[str, list[dict]] | None = None,
        batched: bool = True,
    ):
        self.by_id = dict(by_id or {})
        self.by_doi = {k.lower(): v for k, v in (by_doi or {}).items()}
        self.by_pmcid = dict(by_pmcid or {})
        self.title_hits = dict(title_hits or {})
        self.batched = batched
        self.api_calls = 0
        self.calls: list[tuple[str, str]] = []

    def paginate_filter(self, filter_str, *, limit=None, per_page=200):
        self.api_calls += 1
        self.calls.append(("paginate_filter", filter_str))
        if not self.batched or not filter_str.startswith("doi:"):
            return []
        wanted = filter_str[len("doi:"):].split("|")
        return [self.by_doi[d.lower()] for d in wanted if d.lower() in self.by_doi]

    def get_work(self, openalex_id):
        self.api_calls += 1
        self.calls.append(("get_work", openalex_id))
        return self.by_id.get(openalex_id)

    def lookup_by_doi(self, doi):
        self.api_calls += 1
        self.calls.append(("lookup_by_doi", doi))
        return self.by_doi.get((doi or "").lower())

    def lookup_by_pmcid(self, pmcid):
        self.api_calls += 1
        self.calls.append(("lookup_by_pmcid", pmcid))
        return self.by_pmcid.get(pmcid)

    def search_by_title(self, title, year_hint=None, *, per_page=5):
        self.api_calls += 1
        self.calls.append(("search_by_title", title))
        return list(self.title_hits.get(title, []))[:per_page]

    # paper_from_work and its two helpers are pure — reuse the real ones
    # so the stub produces byte-identical Paper objects.
    def paper_from_work(self, work, **kwargs) -> Paper:
        return OpenAlexClient.paper_from_work(self, work, **kwargs)  # type: ignore[arg-type]

    def slim_work(self, work):
        return OpenAlexClient.slim_work(self, work)  # type: ignore[arg-type]

    @staticmethod
    def reconstruct_abstract(inv_index):
        return OpenAlexClient.reconstruct_abstract(inv_index)


class StubSummaries(Mapping):
    """``prof.summaries`` — a mapping that records what was asked for.

    The real one fetches one body per key on access, so "which keys were
    touched" is the assertion that proves the import does not pull all 82
    summaries to improve the handful of papers that need them.
    """

    def __init__(self, bodies: dict[str, str] | None = None):
        self._bodies = dict(bodies or {})
        self.reads: list[str] = []

    def __getitem__(self, key: str) -> str:
        self.reads.append(key)
        return self._bodies[key]

    def __iter__(self):
        return iter(self._bodies)

    def __len__(self) -> int:
        return len(self._bodies)


class StubProfile:
    """The ``ResearcherProfile`` surface the import reads: three members."""

    def __init__(self, name, papers, summaries=None):
        self.name = name
        self.papers = list(papers)
        self.summaries = summaries if summaries is not None else StubSummaries()


class StubProsopia:
    """Serves one canned profile, or raises."""

    def __init__(self, profile=None, *, error=None):
        self.profile = profile
        self.error = error
        self.fetches = 0

    def fetch_profile(self, slug):
        self.fetches += 1
        self.last_slug = slug
        if self.error is not None:
            raise self.error
        return self.profile

    # ORCID → slug, the way the real client scans the profile list.
    orcids = {"0000-0001-5643-4068": "sheffield-nathan"}

    def resolve_orcid(self, orcid):
        try:
            return self.orcids[orcid.upper()]
        except KeyError as exc:
            raise ProfileNotFound(f"no prosopia profile with ORCID '{orcid}'") from exc


# ---------------------------------------------------------------------------
# Fixture records — the shapes live Prosopia data actually produces
# ---------------------------------------------------------------------------
#
# Real ``PaperRecord`` objects, built through the same validation the SDK
# runs, so a field the SDK renames or coerces breaks these here rather
# than in production.


def _record(**fields) -> PaperRecord:
    return PaperRecord.model_validate(fields)


RECORD_FULL = _record(
    name="Atacformer: a foundation model for ATAC-seq",
    summary="Introduces Atacformer.",
    paper_id="leroy2025atacformer",
    doi="10.1101/2025.11.03.685753",
    openalex_id="W4415881950",
    datePublished="2025",
    isPartOf={"@type": "Periodical", "name": "bioRxiv"},
    full_text_link=(
        "https://www.biorxiv.org/content/biorxiv/early/2025/11/04/"
        "2025.11.03.685753.full.pdf"
    ),
    access="open",
)

RECORD_DOI_ONLY = _record(
    name="Protein Kinase A Inhibition Epigenetically Silences Ren1",
    summary="PKA inhibition silences renin.",
    paper_id="smith2023protein",
    doi="10.1101/2023.09.19.558267",
    datePublished="2023",
    full_text_link=(
        "https://www.biorxiv.org/content/biorxiv/early/2023/09/22/"
        "2023.09.19.558267.full.pdf"
    ),
)

RECORD_PMC_ONLY = _record(
    name="LOLA: enrichment analysis for genomic region sets",
    summary="Region set enrichment in R.",
    paper_id="sheffield2016lola",
    pmcid="PMC4743627",
    datePublished="2016",
    full_text_link="https://pmc.ncbi.nlm.nih.gov/articles/PMC4743627/",
)

RECORD_TITLE_ONLY = _record(
    name="Taming the reference genome jungle",
    summary="The refget sequence collection standard.",
    paper_id="campbell2025taming",
    datePublished="2025",
)

RECORD_NOTHING = _record(
    name="A paper that is in no index anywhere",
    summary="Only the Prosopia summary describes this work.",
    paper_id="ghost2020nowhere",
    datePublished="2020",
)


def _work(doi, oid, title, year=2024):
    return canned_openalex_work(
        doi=doi, openalex_id=f"https://openalex.org/{oid}", title=title,
        year=year, venue="A Journal", abstract_words=["an", "abstract"],
    )


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def test_read_summary_returns_empty_for_a_paper_without_one():
    """A profile that publishes no summary for a paper is normal, not broken."""
    profile = StubProfile("N", [], StubSummaries({"a": "body"}))
    assert read_summary(profile, "a") == "body"
    assert read_summary(profile, "b") == ""


def test_read_summary_survives_a_fetch_that_raises():
    """``summaries`` fetches on access, so one bad artifact is one 502."""

    class Exploding(StubSummaries):
        def __getitem__(self, key):
            raise RuntimeError("server error 500")

    assert read_summary(StubProfile("N", [], Exploding()), "a") == ""


# ---------------------------------------------------------------------------
# The ladder, rung by rung
# ---------------------------------------------------------------------------


def test_resolve_seed_work_id_rung():
    work = _work("10.1101/2025.11.03.685753", "W4415881950", "Atacformer", 2025)
    client = StubOpenAlex(by_id={"W4415881950": work})
    res = resolve_seed(RECORD_FULL, client, slug="sheffield-nathan")
    assert res.rung == "work_id"
    assert res.openalex_id == "https://openalex.org/W4415881950"
    assert res.paper["source"] == "prosopia"
    assert res.paper["abstract"] == "an abstract"
    # One rung, one call: the identifier must not be re-litigated lower down.
    assert client.api_calls == 1


def test_resolve_seed_doi_rung():
    work = _work("10.1101/2023.09.19.558267", "W4386953349", "Protein Kinase A", 2023)
    client = StubOpenAlex(by_doi={"10.1101/2023.09.19.558267": work})
    res = resolve_seed(RECORD_DOI_ONLY, client, slug="s")
    assert res.rung == "doi"
    assert res.openalex_id == "https://openalex.org/W4386953349"


def test_resolve_seed_pmcid_rung():
    work = _work("10.1093/bioinformatics/btv612", "W2184563578", "LOLA", 2016)
    client = StubOpenAlex(by_pmcid={"PMC4743627": work})
    res = resolve_seed(RECORD_PMC_ONLY, client, slug="s")
    assert res.rung == "pmcid"
    assert res.openalex_id == "https://openalex.org/W2184563578"


def test_resolve_seed_title_rung():
    work = _work("10.1093/bioinformatics/btag554", "W4414879798",
                 "Taming the reference genome jungle", 2026)
    client = StubOpenAlex(
        title_hits={"Taming the reference genome jungle": [work]},
    )
    res = resolve_seed(RECORD_TITLE_ONLY, client, slug="s")
    assert res.rung == "title"
    assert res.openalex_id == "https://openalex.org/W4414879798"


def test_resolve_seed_rejects_a_near_miss_title():
    """A plausible neighbour is worse than no match.

    OpenAlex ranks by relevance, so the top hit for a title it does not
    hold is some adjacent paper. Accepting it would put a paper the
    researcher never wrote into their seed corpus, silently.
    """
    wrong = _work("10.1/x", "W999", "Taming the pangenome graph jungle beast", 2025)
    client = StubOpenAlex(
        title_hits={"Taming the reference genome jungle": [wrong]},
    )
    res = resolve_seed(RECORD_TITLE_ONLY, client, slug="s")
    assert res.rung == "none"
    assert res.openalex_id == synthetic_id("s", "campbell2025taming")


def test_resolve_seed_falls_to_synthetic_id_and_embeds_the_summary():
    client = StubOpenAlex()
    res = resolve_seed(RECORD_NOTHING, client, slug="sheffield-nathan")
    assert res.rung == "none"
    assert res.resolved is False
    assert res.openalex_id == "prosopia:sheffield-nathan:ghost2020nowhere"
    assert res.paper["abstract"] == ""
    assert res.paper["body_text"] == "Only the Prosopia summary describes this work."
    assert res.paper["title"] == "A paper that is in no index anywhere"
    assert res.paper["year"] == 2020


def test_resolve_seed_fallback_text_overrides_the_summary():
    client = StubOpenAlex()
    res = resolve_seed(
        RECORD_NOTHING, client, slug="s", fallback_text="summary\n\nartifact body",
    )
    assert res.paper["body_text"] == "summary\n\nartifact body"


def test_resolve_seed_survives_a_lookup_that_raises():
    """A rung that explodes drops to the next one, not out of the import."""

    class Exploding(StubOpenAlex):
        def get_work(self, openalex_id):
            raise RuntimeError("OpenAlex 429: rate-limited.")

    work = _work("10.1101/2025.11.03.685753", "W4415881950", "Atacformer", 2025)
    client = Exploding(by_doi={"10.1101/2025.11.03.685753": work})
    res = resolve_seed(RECORD_FULL, client, slug="s")
    assert res.rung == "doi"


# ---------------------------------------------------------------------------
# Batched prefetch
# ---------------------------------------------------------------------------


def test_prefetch_batches_dois_and_keys_both_ways():
    records = [RECORD_FULL, RECORD_DOI_ONLY, RECORD_PMC_ONLY]
    works = {
        "10.1101/2025.11.03.685753":
            _work("10.1101/2025.11.03.685753", "W4415881950", "Atacformer"),
        "10.1101/2023.09.19.558267":
            _work("10.1101/2023.09.19.558267", "W4386953349", "Protein Kinase A"),
    }
    client = StubOpenAlex(by_doi=works)
    cache = prefetch_works(records, client)
    # Two DOIs (the PMC record carries none) in one request.
    assert client.api_calls == 1
    assert set(cache.by_doi) == set(works)
    assert set(cache.by_id) == {"W4415881950", "W4386953349"}


def test_prefetch_chunks_at_the_batch_size():
    records = [
        _record(paper_id=f"p{i}", doi=f"10.1234/{i}", name=f"Paper {i}")
        for i in range(120)
    ]
    client = StubOpenAlex()
    prefetch_works(records, client, batch_size=50)
    assert client.api_calls == 3


def test_prefetch_batch_failure_degrades_to_per_paper():
    class Failing(StubOpenAlex):
        def paginate_filter(self, filter_str, *, limit=None, per_page=200):
            raise RuntimeError("OpenAlex 429: rate-limited.")

    work = _work("10.1101/2025.11.03.685753", "W4415881950", "Atacformer", 2025)
    client = Failing(by_id={"W4415881950": work})
    cache = prefetch_works([RECORD_FULL], client)
    assert cache.by_id == {} and cache.by_doi == {}
    # The ladder still resolves, just at one request per paper.
    res = resolve_seed(RECORD_FULL, client, slug="s", cache=cache)
    assert res.rung == "work_id"


def test_cached_work_costs_no_request():
    work = _work("10.1101/2025.11.03.685753", "W4415881950", "Atacformer", 2025)
    cache = WorkCache(by_id={"W4415881950": work})
    client = StubOpenAlex()
    res = resolve_seed(RECORD_FULL, client, slug="s", cache=cache)
    assert res.rung == "work_id"
    assert client.api_calls == 0


def test_stale_prosopia_id_still_resolves_on_the_doi_rung():
    """A merged OpenAlex work leaves the published id dangling.

    Prosopia records the id it saw; OpenAlex merges works and the old id
    stops resolving. The DOI is the durable identifier, so the ladder
    must keep walking rather than treating the miss as unresolvable.
    """
    work = _work("10.1101/2025.11.03.685753", "W5555555555", "Atacformer", 2025)
    cache = WorkCache(by_doi={"10.1101/2025.11.03.685753": work})
    client = StubOpenAlex()
    res = resolve_seed(RECORD_FULL, client, slug="s", cache=cache)
    assert res.rung == "doi"
    assert res.openalex_id == "https://openalex.org/W5555555555"
    # One GET for the dangling id, then the cache answers the DOI.
    assert client.api_calls == 1


# ---------------------------------------------------------------------------
# The endpoint — async over gather_runs, like the wizard dry-run
# ---------------------------------------------------------------------------


def _stub_profile(summaries=None):
    return StubProfile(
        "Nathan C. Sheffield",
        [
            RECORD_FULL, RECORD_DOI_ONLY, RECORD_PMC_ONLY,
            RECORD_TITLE_ONLY, RECORD_NOTHING,
        ],
        StubSummaries(summaries or {}),
    )


def _stub_openalex():
    return StubOpenAlex(
        by_doi={
            "10.1101/2025.11.03.685753":
                _work("10.1101/2025.11.03.685753", "W4415881950", "Atacformer", 2025),
            "10.1101/2023.09.19.558267":
                _work("10.1101/2023.09.19.558267", "W4386953349",
                      "Protein Kinase A Inhibition Epigenetically Silences Ren1", 2023),
        },
        by_pmcid={
            "PMC4743627": _work(
                "10.1093/bioinformatics/btv612", "W2184563578",
                "LOLA: enrichment analysis for genomic region sets", 2016,
            ),
        },
        title_hits={
            "Taming the reference genome jungle": [
                _work("10.1093/bioinformatics/btag554", "W4414879798",
                      "Taming the reference genome jungle", 2026),
            ],
        },
    )


def test_normalize_ref_accepts_slugs_and_urls():
    assert normalize_ref("sheffield-nathan") == "sheffield-nathan"
    assert normalize_ref("https://orcid.org/0000-0001-5643-4068") == "0000-0001-5643-4068"
    assert normalize_ref(
        "https://prosopia.databio.org/sheffield-nathan"
    ) == "sheffield-nathan"
    assert normalize_ref(
        "https://prosopia.databio.org/api/v1/profiles/sheffield-nathan/"
    ) == "sheffield-nathan"
    assert normalize_ref("  ") == ""


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    monkeypatch.setenv("RADAR_DB_PATH", str(db))
    monkeypatch.setenv("RADAR_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("RADAR_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RADAR_LOG_JSON", "false")
    monkeypatch.setenv("RADAR_SCHEDULER_ENABLED", "false")
    monkeypatch.setenv("RADAR_DEFAULT_EMBEDDING_MODEL", "placeholder-v1")
    monkeypatch.setenv("RADAR_UMLS_ENABLED", "false")
    settings_module.get_settings.cache_clear()

    conn = connect(db)
    apply_migrations(conn)
    users_repo.upsert(conn, "demo@example.com")
    conn.close()

    yield db
    settings_module.get_settings.cache_clear()


def _app(prosopia, openalex):
    from rag_lib.api.app import create_app
    from rag_lib.api.routers.prosopia import (
        get_import_openalex_client,
        get_prosopia_client,
    )

    app = create_app()
    app.dependency_overrides[get_prosopia_client] = lambda: prosopia
    if openalex is not None:
        app.dependency_overrides[get_import_openalex_client] = lambda: openalex
    return app


def _request(app, method, path, **kwargs) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test",
            ) as client:
                return await client.request(method, path, **kwargs)

    return asyncio.run(_run())


def _start(app, **body) -> httpx.Response:
    return _request(app, "POST", "/api/import/prosopia", json=body)


def test_start_returns_the_draft_slug_and_a_run_id(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    resp = _start(app, ref="sheffield-nathan")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["draft_slug"] == "nathan-c-sheffield"
    assert isinstance(body["run_id"], int)
    assert set(body) == {"draft_slug", "run_id"}

    # The draft is real before the job runs — that is the point of doing
    # the Prosopia read inline.
    conn = connect(env)
    try:
        row = conn.execute(
            "SELECT is_draft FROM profiles WHERE slug = ?", ("nathan-c-sheffield",),
        ).fetchone()
        assert row is not None and row["is_draft"] == 1
    finally:
        conn.close()


def test_status_reports_the_rung_each_paper_came_in_on(env):
    prosopia = StubProsopia(
        _stub_profile({"ghost2020nowhere": "A longer summary."}),
    )
    app = _app(prosopia, _stub_openalex())
    run_id = _start(app, ref="sheffield-nathan").json()["run_id"]

    resp = _request(app, "GET", f"/api/import/prosopia/{run_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["run"]["id"] == run_id
    assert body["run"]["finished_at"]
    assert body["run"]["error"] is None
    assert body["run"]["tier_used"] == "prosopia_import"

    result = body["result"]
    assert result["slug"] == "sheffield-nathan"
    assert result["draft_slug"] == "nathan-c-sheffield"
    assert result["name"] == "Nathan C. Sheffield"
    assert result["drafted"] == 5
    assert result["resolved_by"] == {
        "work_id": 1, "doi": 1, "pmcid": 1, "title": 1, "none": 1,
    }
    assert result["unresolved"] == ["ghost2020nowhere"]


def test_import_attaches_and_embeds_every_seed(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    draft_slug = _start(app, ref="sheffield-nathan").json()["draft_slug"]

    conn = connect(env)
    try:
        row = conn.execute(
            "SELECT id, embedding_model FROM profiles WHERE slug = ?", (draft_slug,),
        ).fetchone()
        profile_id = int(row["id"])

        seeds = [
            r["openalex_id"] for r in conn.execute(
                "SELECT openalex_id FROM profile_seeds WHERE profile_id = ?",
                (profile_id,),
            )
        ]
        assert len(seeds) == 5
        assert "prosopia:sheffield-nathan:ghost2020nowhere" in seeds

        # Every seed carries a vector under the draft's model, otherwise
        # the coherence join in wizard step 2 comes back empty.
        n_vecs = conn.execute(
            """
            SELECT COUNT(*) AS n FROM profile_seeds ps
            JOIN paper_embeddings pe USING (openalex_id)
            WHERE ps.profile_id = ? AND pe.embedding_model = ?
            """,
            (profile_id, row["embedding_model"]),
        ).fetchone()["n"]
        assert n_vecs == 5

        # The synthetic row is embedded from its summary, not its title.
        ghost = conn.execute(
            "SELECT body_text, abstract, source FROM papers WHERE openalex_id = ?",
            ("prosopia:sheffield-nathan:ghost2020nowhere",),
        ).fetchone()
        assert ghost["source"] == "prosopia"
        assert ghost["abstract"] == ""
        assert "Only the Prosopia summary" in ghost["body_text"]
    finally:
        conn.close()


def test_import_reads_a_summary_only_for_the_paper_that_needs_one(env):
    profile = _stub_profile({
        "ghost2020nowhere": "Artifact prose.",
        "campbell2025taming": "Prose nobody asked for.",
    })
    app = _app(StubProsopia(profile), _stub_openalex())
    _start(app, ref="sheffield-nathan")

    # ``summaries`` is lazy, so a read is a request. The resolved papers
    # describe themselves through OpenAlex and never need one; only the
    # unresolved paper costs a round trip.
    assert profile.summaries.reads == ["ghost2020nowhere"]

    conn = connect(env)
    try:
        body = conn.execute(
            "SELECT body_text FROM papers WHERE openalex_id = ?",
            ("prosopia:sheffield-nathan:ghost2020nowhere",),
        ).fetchone()["body_text"]
        assert "Only the Prosopia summary" in body
        assert "Artifact prose." in body
    finally:
        conn.close()


def test_import_accepts_a_profile_url_and_a_name_override(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    resp = _start(
        app,
        ref="https://prosopia.databio.org/api/v1/profiles/sheffield-nathan",
        name="Chromatin reading list",
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["draft_slug"] == "chromatin-reading-list"


def test_import_honours_an_embedding_model_override(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    draft_slug = _start(
        app, ref="sheffield-nathan", embedding_model="placeholder-v1",
    ).json()["draft_slug"]
    conn = connect(env)
    try:
        model = conn.execute(
            "SELECT embedding_model FROM profiles WHERE slug = ?", (draft_slug,),
        ).fetchone()["embedding_model"]
        assert model == "placeholder-v1"
    finally:
        conn.close()


def test_an_orcid_ref_resolves_to_the_profile_slug(env):
    stub = StubProsopia(_stub_profile())
    app = _app(stub, _stub_openalex())
    resp = _start(app, ref="https://orcid.org/0000-0001-5643-4068")
    assert resp.status_code == 200, resp.text
    assert stub.last_slug == "sheffield-nathan"
    assert resp.json()["draft_slug"] == "nathan-c-sheffield"


def test_an_unknown_orcid_is_404_on_the_post(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    resp = _start(app, ref="0000-0002-0000-0000")
    assert resp.status_code == 404, resp.text
    assert "ORCID" in resp.json()["detail"]


def test_unknown_profile_is_404_on_the_post(env):
    app = _app(StubProsopia(error=ProfileNotFound("nope")), _stub_openalex())
    resp = _start(app, ref="nobody-here")
    assert resp.status_code == 404
    assert "nobody-here" in resp.json()["detail"]

    # No draft and no run row: nothing happened that needs cleaning up.
    conn = connect(env)
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM profiles").fetchone()["n"] == 0
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM gather_runs"
        ).fetchone()["n"] == 0
    finally:
        conn.close()


def test_upstream_failure_is_502_on_the_post(env):
    app = _app(StubProsopia(error=ProsopiaError("connection reset")), _stub_openalex())
    assert _start(app, ref="sheffield-nathan").status_code == 502


def test_empty_ref_is_400(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    assert _start(app, ref="  ").status_code == 400


def test_status_404s_for_an_unknown_run(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    assert _request(app, "GET", "/api/import/prosopia/9999").status_code == 404


def test_status_404s_for_another_users_run(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    run_id = _start(app, ref="sheffield-nathan").json()["run_id"]

    # Re-home the draft on a second user; the run must stop being
    # visible to the first. Run ids are bare integers in the URL, so
    # ownership has to be checked, not inferred.
    conn = connect(env)
    try:
        other = users_repo.upsert(conn, "someone-else@example.com")
        conn.execute(
            "UPDATE profiles SET user_id = ? WHERE slug = ?",
            (int(other["id"]), "nathan-c-sheffield"),
        )
        conn.commit()
    finally:
        conn.close()

    assert _request(app, "GET", f"/api/import/prosopia/{run_id}").status_code == 404


def test_import_of_an_empty_profile_still_creates_a_draft(env):
    profile = _stub_profile()
    profile.papers = []
    app = _app(StubProsopia(profile), _stub_openalex())
    resp = _start(app, ref="sheffield-nathan")
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    result = _request(
        app, "GET", f"/api/import/prosopia/{run_id}",
    ).json()["result"]
    assert result["drafted"] == 0
    assert result["unresolved"] == []
    assert result["resolved_by"] == {
        "work_id": 0, "doi": 0, "pmcid": 0, "title": 0, "none": 0,
    }


def test_no_scheduler_closes_the_run_with_an_error(env):
    """Without the inline test seam the job needs the scheduler.

    ``RADAR_SCHEDULER_ENABLED=false`` is the test default, so this is
    also the shape of the production failure when someone deploys the
    API with the scheduler switched off.
    """
    app = _app(StubProsopia(_stub_profile()), None)
    resp = _start(app, ref="sheffield-nathan")
    assert resp.status_code == 503

    conn = connect(env)
    try:
        row = conn.execute(
            "SELECT error, finished_at FROM gather_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["finished_at"] and "scheduler disabled" in row["error"]
    finally:
        conn.close()


def test_the_job_body_records_a_failure_on_the_run_row(env):
    """A job that blows up must leave a readable reason, not a hung poll."""
    from rag_lib.db.repos import gather_runs as gather_runs_repo
    from rag_lib.scheduler.jobs import import_prosopia_profile

    settings = settings_module.get_settings()
    conn = connect(env)
    try:
        from rag_lib.api.services import prosopia as prosopia_service

        plan = prosopia_service.prepare_import(
            conn, settings, user_id=1, ref="sheffield-nathan",
            prosopia_client=StubProsopia(_stub_profile()),
        )
        run_id = gather_runs_repo.start(
            conn, profile_id=plan.profile_id, user_id=1, tier_used="prosopia_import",
        )
    finally:
        conn.close()

    plan.embedding_model = "no-such-embedder"
    import_prosopia_profile(plan, settings=settings, run_id=run_id)

    conn = connect(env)
    try:
        row = gather_runs_repo.get(conn, run_id)
        assert row["finished_at"]
        assert "no-such-embedder" in row["error"]
        assert row["result_json"] is None
    finally:
        conn.close()
