"""Settings resolution for docrag.

Resolution order (highest wins):
  1. Environment variable
  2. app.settings.json at the repo root
  3. Built-in default

Loads ``.env`` at the repo root (via python-dotenv) if present so the same
config is available to every entry point.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start: str) -> str:
    """Walk up to the dir containing pyproject.toml (or .git). Under the
    src-layout the package lives at <repo>/src/rag/, so the old "parent of the
    package dir" heuristic would wrongly point at <repo>/src/ and lose the
    existing .env / app.settings.json / .index / corpora. Anchor on the project
    marker instead. (DOCRAG_INDEX_DIR / DOCRAG_DOCS_ROOT still override paths.)"""
    d = start
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "pyproject.toml")) or \
                os.path.isdir(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(os.path.dirname(start))  # <repo>/src/rag -> <repo>


# <repo>/src/rag/settings.py -> <repo>
_REPO_ROOT = _find_repo_root(_THIS_DIR)

_ENV_PATH = os.path.join(_REPO_ROOT, ".env")
_SETTINGS_PATH = os.path.join(_REPO_ROOT, "app.settings.json")


def _load_dotenv_once() -> None:
    """Best-effort .env load. Silent if python-dotenv is missing."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    if os.path.isfile(_ENV_PATH):
        load_dotenv(_ENV_PATH, override=False)


_load_dotenv_once()


_SETTINGS_CACHE: Optional[dict] = None


def _settings() -> dict:
    """Load app.settings.json (cached). Missing file = empty dict."""
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is not None:
        return _SETTINGS_CACHE
    if os.path.isfile(_SETTINGS_PATH):
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
                _SETTINGS_CACHE = json.load(f) or {}
        except (OSError, ValueError):
            _SETTINGS_CACHE = {}
    else:
        _SETTINGS_CACHE = {}
    return _SETTINGS_CACHE


def get(key: str, default: Any = None) -> Any:
    """Resolve a config value: env var > app.settings.json > default."""
    val = os.environ.get(key)
    if val is not None and val != "":
        return val
    settings = _settings()
    if key in settings and settings[key] not in (None, ""):
        return settings[key]
    return default


def corpus_config(corpus: str) -> dict:
    """Per-corpus config block from app.settings.json, e.g.
    ``{"corpora": {"zeihan": {"domain": "youtube", "chat": {...}}}}``.
    Returns {} when absent."""
    corpora = _settings().get("corpora") or {}
    cfg = corpora.get((corpus or "").strip().lower())
    return cfg if isinstance(cfg, dict) else {}


# ---------- Paths ----------

def repo_root() -> str:
    return _REPO_ROOT


def docs_root() -> str:
    """Root directory of corpora to index (``{root}/{corpus}/...``).

    Default: ``<repo>/corpora``. Override via DOCRAG_DOCS_ROOT or the
    ``docs_root`` setting.
    """
    val = os.environ.get("DOCRAG_DOCS_ROOT")
    if val:
        return os.path.expandvars(val)
    val = _settings().get("docs_root")
    if val:
        return os.path.expandvars(val)
    return os.path.join(_REPO_ROOT, "corpora")


def index_dir() -> str:
    """Local index directory (per-corpus SQLite DBs + feedback logs).

    Default: ``<repo>/.index``. Created on first access. Override via
    DOCRAG_INDEX_DIR or the ``index_dir`` setting.
    """
    val = os.environ.get("DOCRAG_INDEX_DIR")
    if not val:
        val = _settings().get("index_dir")
    if not val:
        val = os.path.join(_REPO_ROOT, ".index")
    val = os.path.expandvars(val)
    os.makedirs(val, exist_ok=True)
    return val


# ---------- Azure OpenAI ----------

def azure_endpoint() -> str:
    return get("AZURE_OPENAI_ENDPOINT", "") or ""


def azure_api_key() -> str:
    return get("AZURE_OPENAI_API_KEY", "") or ""


def azure_api_version() -> str:
    return get("AZURE_OPENAI_API_VERSION", "2024-10-21") or "2024-10-21"


def embedding_deployment() -> str:
    return get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large") \
        or "text-embedding-3-large"


def embedding_dims() -> int:
    raw = get("AZURE_OPENAI_EMBEDDING_DIMS", 3072)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 3072


def chat_deployment_fast() -> str:
    return get("AZURE_OPENAI_DEPLOYMENT_FAST", "") or chat_deployment()


def chat_deployment() -> str:
    return get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini") or "gpt-4o-mini"


def chat_deployment_synthesis() -> str:
    """Deployment for grounded answer synthesis.

    Synthesis must do real regulatory reasoning (apply thresholds / scope rules
    to the question's facts, including the converse of a scope limiter), so it
    defaults to the full chat deployment rather than the cheaper FAST one.
    Override with AZURE_OPENAI_DEPLOYMENT_SYNTHESIS.
    """
    return get("AZURE_OPENAI_DEPLOYMENT_SYNTHESIS", "") or chat_deployment()
