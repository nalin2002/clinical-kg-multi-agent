"""Load API credentials from environment variables or api_keys.json.

Preference order:
  1. Standard env vars (after loading ``.env`` from repo root and cwd)
  2. Legacy ``api_keys.json`` in repo root or cwd

Env vars:
  - ``OPENROUTER_API_KEY``
  - ``ANTHROPIC_API_KEY``
  - ``COOPERATIVE_KG_PROVIDER`` (optional default: ``openrouter``)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parents[2]

_ENV_OPENROUTER = "OPENROUTER_API_KEY"
_ENV_ANTHROPIC = "ANTHROPIC_API_KEY"
_ENV_PROVIDER = "COOPERATIVE_KG_PROVIDER"


def _load_dotenv_files() -> None:
    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv()


def _read_api_keys_json() -> dict:
    for path in (_REPO_ROOT / "api_keys.json", Path("api_keys.json")):
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except (json.JSONDecodeError, OSError):
                return {}
    return {}


def resolve_provider(cli_provider: str | None = None) -> str:
    """Return the active API provider name."""
    if cli_provider:
        return cli_provider
    _load_dotenv_files()
    env_provider = os.environ.get(_ENV_PROVIDER, "").strip()
    if env_provider:
        return env_provider
    return _read_api_keys_json().get("provider") or "openrouter"


def get_api_key(provider: str) -> str:
    """Return the API key for *provider* (``openrouter`` or ``anthropic``)."""
    _load_dotenv_files()
    if provider == "anthropic":
        key = os.environ.get(_ENV_ANTHROPIC, "").strip()
        legacy_field = "anthropic"
        env_name = _ENV_ANTHROPIC
    else:
        key = os.environ.get(_ENV_OPENROUTER, "").strip()
        legacy_field = "openrouter"
        env_name = _ENV_OPENROUTER

    if key:
        return key

    key = _read_api_keys_json().get(legacy_field) or ""
    if key:
        return str(key).strip()

    raise SystemExit(
        f"No API key for {provider!r}. Set {env_name} in .env "
        f"(see .env.example) or add {legacy_field!r} to api_keys.json."
    )
