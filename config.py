"""Central configuration. Secrets come from the environment (or a local .env),
never from source — so nothing sensitive lands in git.

Copy .env.example -> .env and fill in TRIBE_API_TOKEN (and TRIBE_BASE_URL when
your Lightning session URL changes).
"""

from __future__ import annotations

import os


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency). Existing env vars take precedence."""
    if not os.path.isfile(path):
        return
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()

# The Lightning deployment URL is an endpoint, not a secret; keep a convenience
# default but allow overriding (the cloudspaces URL rotates between sessions).
BASE_URL = os.environ.get(
    "TRIBE_BASE_URL",
    "https://8000-01kvkq7h52p2hmk3mt5vrdqmfg.cloudspaces.litng.ai",
)

# The token IS a secret: no default. Missing -> require_token() raises.
TOKEN = os.environ.get("TRIBE_API_TOKEN")


def require_token() -> str:
    if not TOKEN:
        raise SystemExit(
            "TRIBE_API_TOKEN is not set.\n"
            "  -> copy .env.example to .env and add your token, or set the env var."
        )
    return TOKEN
