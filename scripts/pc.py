"""A small reader for Microsoft Planetary Computer's public STAC catalog (no key: a short-lived token per
collection is handed out to anyone)."""
import json
import time
import urllib.request

API = "https://planetarycomputer.microsoft.com/api"
_tokens = {}


def _get(req):
    """The JSON answer to a request, tried again on a failed lookup or a dropped connection."""
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            if attempt == 5:
                raise
            time.sleep(3 + 5 * attempt)


def search(collection, box, limit=200, **more):
    """Items of a collection that touch a lon/lat box. `more` adds STAC search fields such as datetime or query."""
    return _get(urllib.request.Request(f"{API}/stac/v1/search", json.dumps(
        {"collections": [collection], "bbox": list(box), "limit": limit, **more}).encode(),
        {"Content-Type": "application/json"}))["features"]


def signed(collection, href):
    """An asset address with the collection's read token, as a GDAL /vsicurl/ path."""
    if collection not in _tokens:
        _tokens[collection] = _get(f"{API}/sas/v1/token/{collection}")["token"]
    return f"/vsicurl/{href}?{_tokens[collection]}"
