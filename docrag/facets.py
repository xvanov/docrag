"""facets.py -- jurisdiction (location) + edition (version) facets.

Two query-time selectors are derived from a chunk's ``path`` (+ its ``edition``
label for display):

  * LOCATION (global) -- which jurisdiction layers apply. "Durham, NC" stacks
    model I-Codes + North Carolina + Durham; "North Carolina" drops Durham;
    "Model" keeps only the I-Codes baseline.
  * VERSION (per document) -- which edition/year of each code to use. A
    "document" is one (jurisdiction, code-family) pair, e.g. model+ibc, which
    may exist in several years (IBC 2021 + IBC 2024). Default: latest available.

Path conventions (corpora/building-codes/):
    model/2024/<abbr>/...                      -> model, family=<abbr>, year 2024
    model/2021_International_Building_Code.pdf  -> model, family=ibc, year 2021
    model/<year>/<abbr>/...                     -> model, family=<abbr>, year
    north-carolina/<NCxxx slug>/...             -> north-carolina, family, 2024
    durham/...                                  -> durham, family=udo, unversioned
"""
from __future__ import annotations

import os
import re

# --- Jurisdiction tiers ------------------------------------------------------
# Every jurisdiction folder maps to a regulatory tier. Locations are defined by
# the tiers they include, so NEW jurisdiction folders (e.g. another federal or
# NC-state source) are picked up automatically without editing the location
# list. Unknown folders fall back to "other" and are treated as broadly
# applicable (only the strict "model" location excludes them).

_TIER_RULES = (
    (lambda j: j == "model", "model"),
    (lambda j: j == "federal", "federal"),
    (lambda j: j == "north-carolina" or j == "nc-state" or j == "ncdot"
     or j.startswith("nc-") or j.startswith("nc_"), "state"),
    (lambda j: j == "durham" or j.startswith("durham"), "local"),
)


def tier_of(jurisdiction: str) -> str:
    j = (jurisdiction or "").strip().lower()
    for pred, tier in _TIER_RULES:
        if pred(j):
            return tier
    return "other"


LOCATIONS = [
    {"key": "durham-nc", "label": "Durham, NC",
     "tiers": {"model", "federal", "state", "local", "other"},
     "answer_location": "Durham, North Carolina"},
    {"key": "north-carolina", "label": "North Carolina (statewide)",
     "tiers": {"model", "federal", "state", "other"},
     "answer_location": "North Carolina"},
    {"key": "model", "label": "Model I-Codes only",
     "tiers": {"model"},
     "answer_location": "the model I-Codes"},
]
DEFAULT_LOCATION = "durham-nc"
_LOC_BY_KEY = {loc["key"]: loc for loc in LOCATIONS}


def location(key: str | None) -> dict:
    return _LOC_BY_KEY.get((key or "").strip().lower()) or _LOC_BY_KEY[DEFAULT_LOCATION]


def location_allows(location_key: str | None, jurisdiction: str) -> bool:
    return tier_of(jurisdiction) in location(location_key)["tiers"]


def answer_location(location_key: str | None) -> str:
    return location(location_key)["answer_location"]


# --- Code-family display names ----------------------------------------------

_MODEL_FAMILY_NAMES = {
    "ibc": "IBC", "irc": "IRC", "ipc": "IPC", "imc": "IMC", "ifgc": "IFGC",
    "ifc": "IFC", "iebc": "IEBC", "iecc": "IECC", "iecc-ashrae": "IECC + ASHRAE 90.1",
    "ipmc": "IPMC", "iccpc": "ICCPC", "ipsdc": "IPSDC", "ispsc": "ISPSC",
    "iwuic": "IWUIC", "izc": "IZC", "igcc": "IGCC",
}
_NC_FAMILY_NAMES = {
    "ncbc": "NC Building Code", "ncrc": "NC Residential Code",
    "ncpc": "NC Plumbing Code", "ncmc": "NC Mechanical Code",
    "ncfgc": "NC Fuel Gas Code", "ncfc": "NC Fire Prevention Code",
    "ncebc": "NC Existing Building Code", "ncecc": "NC Energy Conservation Code",
    "ncapc": "NC Administrative Code",
}


def _year_from(text: str) -> str | None:
    m = re.search(r"(19|20)\d{2}", text or "")
    return m.group(0) if m else None


def facet_of(path: str, edition: str | None = None) -> dict:
    """Parse a chunk path into {jurisdiction, family, year, doc_key, doc_label}.

    ``doc_key`` = "<jurisdiction>:<family>" identifies a document across
    editions; ``year`` is the edition year (string) or None if unversioned.
    """
    p = (path or "").replace("\\", "/").lstrip("/")
    segs = p.split("/")
    juris = segs[0] if segs else ""

    family = ""
    year = None

    if juris == "model":
        if len(segs) >= 3 and re.fullmatch(r"(19|20)\d{2}", segs[1]):
            # model/<year>/<abbr>/...
            year = segs[1]
            family = segs[2].lower()
        else:
            # model/<file> -- parse year + family from the filename
            base = segs[1] if len(segs) > 1 else ""
            year = _year_from(base)
            low = base.lower()
            if "international_building_code" in low or re.search(r"\bibc\b", low):
                family = "ibc"
            else:
                family = re.sub(r"[^a-z0-9]+", "-", os.path.splitext(low)[0]).strip("-")
    elif juris == "north-carolina":
        slug = segs[1] if len(segs) > 1 else ""
        m = re.match(r"([a-zA-Z]+)", slug)
        family = (m.group(1).lower() if m else slug.lower())
        year = _year_from(slug) or "2024"
    elif juris == "durham":
        family = "udo"
        year = _year_from(edition or "") or _year_from(p)
    else:
        family = (segs[1].lower() if len(segs) > 1 else juris)
        year = _year_from(edition or "") or _year_from(p)

    doc_key = "%s:%s" % (juris, family)
    return {"jurisdiction": juris, "family": family, "year": year,
            "doc_key": doc_key, "doc_label": doc_label(juris, family, edition)}


def doc_label(jurisdiction: str, family: str, edition: str | None = None) -> str:
    """A stable, edition-independent label for a document (for the version UI)."""
    if jurisdiction == "model":
        return _MODEL_FAMILY_NAMES.get(family, family.upper()) + " (model)"
    if jurisdiction == "north-carolina":
        return _NC_FAMILY_NAMES.get(family, "NC " + family.upper())
    if jurisdiction == "durham":
        return "Durham UDO"
    # fall back to a year-stripped edition label
    if edition:
        return re.sub(r"\b(19|20)\d{2}\b", "", edition).strip(" -:") or edition
    return family.upper()


# --- Corpus version enumeration (for the UI) --------------------------------


def corpus_versions(conn) -> list[dict]:
    """Enumerate documents in the corpus that have a selectable edition.

    Returns one entry per (jurisdiction, family):
      {doc_key, doc_label, jurisdiction, family, years:[...desc...],
       default_year, versioned: bool}
    Only entries with >1 year are truly "versioned" (need a dropdown); the rest
    are reported too so the UI can show a fixed edition if desired.
    """
    rows = conn.execute(
        "SELECT DISTINCT path, edition FROM chunks"
    ).fetchall()
    docs: dict[str, dict] = {}
    for path, edition in rows:
        f = facet_of(path, edition)
        key = f["doc_key"]
        d = docs.setdefault(key, {
            "doc_key": key, "doc_label": f["doc_label"],
            "jurisdiction": f["jurisdiction"], "family": f["family"],
            "years": set()})
        if f["year"]:
            d["years"].add(f["year"])
    out = []
    for d in docs.values():
        years = sorted(d["years"], reverse=True)
        out.append({
            "doc_key": d["doc_key"], "doc_label": d["doc_label"],
            "jurisdiction": d["jurisdiction"], "family": d["family"],
            "years": years,
            "default_year": years[0] if years else None,
            "versioned": len(years) > 1,
        })
    out.sort(key=lambda x: (x["jurisdiction"], x["doc_label"]))
    return out


def resolve_versions(conn, overrides: dict | None = None) -> dict:
    """Effective {doc_key: year} for every *versioned* document: latest by
    default, with any valid user override applied. Single-edition docs are
    omitted (nothing to filter)."""
    overrides = overrides or {}
    eff: dict[str, str] = {}
    for d in corpus_versions(conn):
        if not d["versioned"]:
            continue
        chosen = str(overrides.get(d["doc_key"]) or "")
        eff[d["doc_key"]] = chosen if chosen in d["years"] else d["default_year"]
    return eff
