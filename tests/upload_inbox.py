"""An interest to upload into, for tests whose upload names none.

Every upload names an interest now (FUNCTIONAL_SPEC 1.1: the wizard's first
step, an interest's Seeds tab); a test that only needs a paper in the vault
uploads it into a draft named "Inbox", one per app and user.
"""

from __future__ import annotations

_BOXES: dict = {}


def inbox(request, app, headers=None) -> str:
    """``request(app, method, path, **kw)`` is the test module's own helper."""
    key = (id(app), (headers or {}).get("X-User-Email"))
    if key not in _BOXES:
        r = request(app, "POST", "/api/profiles/draft", json={"name": "Inbox"}, headers=headers or {})
        assert r.status_code == 200, r.text
        _BOXES[key] = r.json()["slug"]
    return _BOXES[key]
