"""The OpenAlex API key (user 11-05): sent on every request when set, and
never in an error's text -- the request URL carries it."""

from __future__ import annotations

import pytest
import requests

from rag_lib import openalex_status
from rag_lib.api import settings as settings_module
from rag_lib.openalex_client import OpenAlexClient
from rag_lib.openalex_status import OpenAlexUnavailable

KEY = "k3y-SECRET-123"


class _Resp:
    def __init__(self, code=200, body=None, text="", url=""):
        self.status_code = code
        self._body = body if body is not None else {"results": [], "meta": {"count": 0}}
        self.text = text
        self.headers = {}
        self.url = url

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error: x for url: {self.url}", response=self)


class _Session:
    def __init__(self, resp=None, exc=None):
        self.headers = {}
        self.calls = []
        self.resp, self.exc = resp, exc

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params or {}))
        if self.exc:
            raise self.exc
        return self.resp or _Resp()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("RADAR_OPENALEX_API_KEY", raising=False)
    settings_module.get_settings.cache_clear()
    openalex_status.reset()
    yield
    openalex_status.reset()
    settings_module.get_settings.cache_clear()


def _client(session, **kw):
    return OpenAlexClient("a@example.com", session=session, rate_limit_sleep=0, **kw)


def test_the_key_is_sent_on_every_request():
    s = _Session()
    c = _client(s, api_key=KEY)
    c._get("/works", {"search": "x"})
    c._get("/works/W1")
    assert [p.get("api_key") for p in s.calls] == [KEY, KEY]


def test_no_key_sends_none():
    s = _Session()
    _client(s)._get("/works")
    assert "api_key" not in s.calls[0]


def test_the_key_comes_from_the_settings(monkeypatch):
    monkeypatch.setenv("RADAR_OPENALEX_API_KEY", KEY)
    settings_module.get_settings.cache_clear()
    s = _Session()
    _client(s)._get("/works")
    assert s.calls[0]["api_key"] == KEY


def test_an_empty_key_is_no_key(monkeypatch):
    monkeypatch.setenv("RADAR_OPENALEX_API_KEY", "")
    settings_module.get_settings.cache_clear()
    s = _Session()
    _client(s)._get("/works")
    assert "api_key" not in s.calls[0]


def _all_text(exc: BaseException) -> str:
    """Everything a log line or traceback could print of ``exc``."""
    import traceback
    return "".join(traceback.format_exception(exc)) + str(getattr(exc, "detail", ""))


def test_a_connection_error_does_not_carry_the_key():
    url = f"https://api.openalex.org/works?search=x&api_key={KEY}"
    s = _Session(exc=requests.ConnectionError(f"Max retries exceeded with url: {url}"))
    with pytest.raises(OpenAlexUnavailable) as got:
        _client(s, api_key=KEY)._get("/works", {"search": "x"})
    assert KEY not in _all_text(got.value) and "<api_key>" in _all_text(got.value)


def test_an_http_error_does_not_carry_the_key():
    s = _Session(resp=_Resp(404, url=f"https://api.openalex.org/works/W1?api_key={KEY}"))
    with pytest.raises(requests.HTTPError) as got:
        _client(s, api_key=KEY)._get("/works/W1")
    assert got.value.response.status_code == 404
    assert KEY not in _all_text(got.value)


def test_a_429_body_does_not_carry_the_key():
    s = _Session(resp=_Resp(429, text=f"slow down, api_key={KEY}"))
    with pytest.raises(OpenAlexUnavailable) as got:
        _client(s, api_key=KEY)._get("/works", {"search": "x"})
    assert KEY not in _all_text(got.value)


def test_a_reply_that_is_not_json_does_not_carry_the_key():
    class _Bad(_Resp):
        def json(self):
            raise ValueError(f"bad page for api_key={KEY}")
    s = _Session(resp=_Bad())
    with pytest.raises(OpenAlexUnavailable) as got:
        _client(s, api_key=KEY)._get("/works")
    assert KEY not in _all_text(got.value)
