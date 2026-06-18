"""registry.py -- resolve a Domain by name or by corpus.

Domain modules are imported lazily so importing the registry never pulls both
domains' (disjoint) dependency stacks. Resolution order for a corpus:
  1. app.settings.json  ->  corpora.<corpus>.domain
  2. built-in known corpus names
  3. default: building_codes (the original use case)
"""

from __future__ import annotations

import importlib

from . import settings
from .domains.base import Domain

# Domain name -> import path of a module exposing a module-level ``DOMAIN``.
_DOMAIN_MODULES = {
    "building_codes": "rag.domains.building_codes",
    "youtube": "rag.domains.youtube",
}

# Known corpus -> domain, for corpora not listed in app.settings.json.
_KNOWN_CORPUS_DOMAINS = {
    "building-codes": "building_codes",
    "udo": "building_codes",
}

_DEFAULT_DOMAIN = "building_codes"

_CACHE: dict[str, Domain] = {}


def available_domains() -> list[str]:
    return sorted(_DOMAIN_MODULES)


def get_domain(name: str) -> Domain:
    name = (name or "").strip().lower()
    if name not in _DOMAIN_MODULES:
        raise ValueError("unknown domain %r (known: %s)"
                         % (name, ", ".join(available_domains())))
    if name not in _CACHE:
        mod = importlib.import_module(_DOMAIN_MODULES[name])
        _CACHE[name] = mod.DOMAIN
    return _CACHE[name]


def domain_name_for_corpus(corpus: str) -> str:
    corpus = (corpus or "").strip().lower()
    cfg_name = settings.corpus_config(corpus).get("domain")
    if cfg_name:
        return str(cfg_name).strip().lower()
    return _KNOWN_CORPUS_DOMAINS.get(corpus, _DEFAULT_DOMAIN)


def domain_for_corpus(corpus: str) -> Domain:
    return get_domain(domain_name_for_corpus(corpus))
