# config.py
from dotenv import load_dotenv
import os
import re
from dataclasses import dataclass

# Load .env
loaded = load_dotenv(dotenv_path=".env")
if not loaded:
    raise RuntimeError(".env file not found or could not be loaded")

# Validation helper
_URL_RE = re.compile(r"^https?://[A-Za-z0-9\.\-_/]+$")

def _read_env(key: str, *, required: bool = False, pattern: str | None = None) -> str:
    val = os.getenv(key)
    if required and (val is None or val.strip() == ""):
        raise RuntimeError(f"Missing required env var: {key}")
    if pattern and val and not re.fullmatch(pattern, val):
        raise RuntimeError(f"Env var {key} failed pattern validation")
    return val

@dataclass(frozen=True)
class _Env:
    API_GATEWAY_URL: str

def _load_env() -> _Env:
    return _Env(
        API_GATEWAY_URL=_read_env("API_GATEWAY_URL", required=True),
    )

# Load on importcccccc
env = _load_env()
