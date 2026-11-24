"""A small reader for Microsoft Planetary Computer's public STAC catalog (no key: a short-lived token per
collection is handed out to anyone, and asked for again before it runs out)."""
import calendar
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
    tok = _tokens.get(collection)
    if tok is None or tok[1] - time.time() < 300:  # a token can have as little as 15 minutes left when handed out
        d = _get(f"{API}/sas/v1/token/{collection}")
        _tokens[collection] = tok = (d["token"], calendar.timegm(time.strptime(d["msft:expiry"], "%Y-%m-%dT%H:%M:%SZ")))
    return f"/vsicurl/{href}?{tok[0]}"
