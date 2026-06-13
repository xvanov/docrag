#!/usr/bin/env bash
# Fetch FREE, legally-downloadable mainstream/engineered/green construction-methods
# sources into corpora/methods/. Govt (public domain), industry-association free
# downloads, and open building-science pubs only. No paywalled/borrow-only material.
#
#   bash scraper/fetch_methods_mainstream.sh
#
# Idempotent: skips files already on disk. Re-run safe.
# Verifies each download: HTTP 200, >50KB, and real PDF (leading %PDF). Anything that
# 403/404s or returns HTML is deleted and reported under FAILED.
#
# Gated/paywalled/borrow-only items are NOT here -- see the report / PENDING_SOURCES.md.
#
# NOTE: docs.nrel.gov is unreachable from this network (DNS); NREL/DOE Building America
# content is pulled from the equivalent www1.eere.energy.gov mirror instead.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

ok=0; fail=0; skip=0
FAILED=()

fetch() {  # fetch <relative_path> <url>
  local rel="$1" url="$2"
  local dest="$ROOT/$rel"
  if [[ -s "$dest" ]]; then echo "SKIP  $rel"; skip=$((skip+1)); return; fi
  mkdir -p "$(dirname "$dest")"
  if curl -sSL -f --connect-timeout 30 --max-time 600 -A "$UA" -o "$dest" "$url" 2>/dev/null; then
    local sz magic
    sz=$(wc -c < "$dest" 2>/dev/null || echo 0)
    magic=$(head -c 4 "$dest" 2>/dev/null | tr -d '\0')
    if [[ "$sz" -lt 51200 ]]; then
      echo "FAIL  $rel (too small ${sz}B)"; rm -f "$dest"; fail=$((fail+1)); FAILED+=("$rel  <-  $url")
    elif [[ "$magic" != "%PDF" ]]; then
      echo "FAIL  $rel (not a PDF, magic='${magic}')"; rm -f "$dest"; fail=$((fail+1)); FAILED+=("$rel  <-  $url")
    else
      echo "OK    $rel  ($((sz/1024)) KB)"; ok=$((ok+1))
    fi
  else
    echo "FAIL  $rel"; rm -f "$dest"; fail=$((fail+1)); FAILED+=("$rel  <-  $url")
  fi
}

ME="corpora/methods"

echo "===== METHODS: CONCRETE (NRMCA Concrete in Practice -- free assoc. PDFs) ====="
# NRMCA CIP one-pagers: mix/curing/weather/QA-QC/reinforcement/durability best practice.
fetch "$ME/concrete/NRMCA-CIP-11-curing-in-place-concrete.pdf"              "https://www.nrmca.org/wp-content/uploads/2021/01/11pr.pdf"
fetch "$ME/concrete/NRMCA-CIP-12-hot-weather-concreting.pdf"                "https://www.nrmca.org/wp-content/uploads/2021/01/12pr.pdf"
fetch "$ME/concrete/NRMCA-CIP-14-finishing-concrete-flatwork.pdf"           "https://www.nrmca.org/wp-content/uploads/2021/01/14pr.pdf"
fetch "$ME/concrete/NRMCA-CIP-15-chemical-admixtures.pdf"                   "https://www.nrmca.org/wp-content/uploads/2021/01/15pr.pdf"
fetch "$ME/concrete/NRMCA-CIP-24-synthetic-fibers.pdf"                      "https://www.nrmca.org/wp-content/uploads/2021/01/24pr.pdf"
fetch "$ME/concrete/NRMCA-CIP-25-corrosion-of-steel-in-concrete.pdf"        "https://www.nrmca.org/wp-content/uploads/2021/01/25pr.pdf"
fetch "$ME/concrete/NRMCA-CIP-27-cold-weather-concreting.pdf"               "https://www.nrmca.org/wp-content/uploads/2021/01/27pr.pdf"
fetch "$ME/concrete/NRMCA-CIP-29-vapor-retarders-under-slabs.pdf"           "https://www.nrmca.org/wp-content/uploads/2021/01/29pr.pdf"
fetch "$ME/concrete/NRMCA-CIP-30-supplementary-cementitious-materials.pdf"  "https://www.nrmca.org/wp-content/uploads/2021/01/30pr.pdf"
fetch "$ME/concrete/NRMCA-CIP-34-making-cylinders-in-the-field.pdf"         "https://www.nrmca.org/wp-content/uploads/2021/01/34pr.pdf"
fetch "$ME/concrete/NRMCA-CIP-35-compressive-strength-testing.pdf"          "https://www.nrmca.org/wp-content/uploads/2021/01/35pr.pdf"
fetch "$ME/concrete/NRMCA-CIP-41-acceptance-testing-of-concrete.pdf"        "https://www.nrmca.org/wp-content/uploads/2021/01/41pr.pdf"
fetch "$ME/concrete/NRMCA-CIP-44-durability-requirements.pdf"               "https://www.nrmca.org/wp-content/uploads/2021/01/44pr.pdf"

echo "===== METHODS: STEEL (AISI S240 already on disk) ====="
# AISC 341/358/360/303 are free *but* behind aisc.org bot protection (403/404 to curl) --
# see gated list in report. AISI S240-20 already acquired via fetch_sources.sh.

echo "===== METHODS: GENERAL CONTRACTING (OSHA construction -- US govt, public domain) ====="
fetch "$ME/general-contracting/OSHA-2202-construction-industry-digest.pdf"      "https://www.osha.gov/sites/default/files/publications/OSHA2202.pdf"
fetch "$ME/general-contracting/OSHA-2226-trenching-and-excavation-safety.pdf"   "https://www.osha.gov/sites/default/files/publications/OSHA2226.pdf"
fetch "$ME/general-contracting/OSHA-3146-fall-protection-in-construction.pdf"   "https://www.osha.gov/sites/default/files/publications/OSHA3146.pdf"
fetch "$ME/general-contracting/OSHA-3150-scaffold-use-in-construction.pdf"      "https://www.osha.gov/sites/default/files/publications/OSHA3150.pdf"
fetch "$ME/general-contracting/OSHA-3151-personal-protective-equipment.pdf"     "https://www.osha.gov/sites/default/files/publications/osha3151.pdf"
fetch "$ME/general-contracting/OSHA-3115-control-of-hazardous-energy-lockout.pdf" "https://www.osha.gov/sites/default/files/publications/osha3115.pdf"
fetch "$ME/general-contracting/OSHA-3124-stairways-and-ladders.pdf"             "https://www.osha.gov/sites/default/files/publications/OSHA3124.pdf"
fetch "$ME/general-contracting/OSHA-3666-fall-prevention-training-guide.pdf"    "https://www.osha.gov/sites/default/files/publications/OSHA3666.pdf"
fetch "$ME/general-contracting/OSHA-3681-crystalline-silica-construction-factsheet.pdf" "https://www.osha.gov/sites/default/files/publications/OSHA3681.pdf"
fetch "$ME/general-contracting/OSHA-3902-respirable-crystalline-silica-construction-guide.pdf" "https://www.osha.gov/sites/default/files/publications/OSHA3902.pdf"
fetch "$ME/general-contracting/OSHA-3998-recommended-practices-safety-health-programs.pdf" "https://www.osha.gov/sites/default/files/publications/OSHA3998.pdf"
# GSA construction QA/commissioning (US govt, via WBDG mirror)
fetch "$ME/general-contracting/GSA-building-commissioning-guide-2020.pdf"       "https://www.wbdg.org/FFC/GSA/gsa_commissioning_guide_2020.pdf"

echo "===== METHODS: GREEN / SUSTAINABLE BUILDING (DOE/EPA -- US govt, public domain) ====="
fetch "$ME/green-building/DOE-zero-energy-ready-home-v2-rev2-requirements.pdf"  "https://www.energy.gov/sites/default/files/2024-09/DOE%20ZERH%20V2%20%28Rev.%202%29%20National%20Program%20Requirements.pdf"
fetch "$ME/green-building/DOE-building-america-best-practices-climate-guide-7.1.pdf" "https://www1.eere.energy.gov/buildings/publications/pdfs/building_america/ba_climateguide_7_1.pdf"
fetch "$ME/green-building/EPA-watersense-labeled-homes-introductory-guide.pdf"  "https://www.epa.gov/system/files/documents/2022-09/ws-WaterSense-Labeled-Homes-Introductory-Guide.pdf"
fetch "$ME/green-building/EPA-indoor-airplus-construction-specifications-v1r5.pdf" "https://www.epa.gov/system/files/documents/2025-12/iap-v1r5-2025-december-508_final.pdf"
fetch "$ME/green-building/DOE-building-america-builders-handbook-mixed-humid-vol4.pdf" "https://www1.eere.energy.gov/buildings/publications/pdfs/building_america/38448.pdf"

echo "===== METHODS: BUILDING SCIENCE (envelope / moisture / airtightness) ====="
# Building Science Corporation -- free Building Science Digests / info sheets.
fetch "$ME/building-science/BSC-BSD-012-moisture-control-new-residential.pdf"   "https://buildingscience.com/sites/default/files/migrate/pdf/BSD-012_Moisture_Control_New_Bldgs.pdf"
fetch "$ME/building-science/BSC-info-511-basement-insulation-all-climates.pdf"  "https://buildingscience.com/sites/default/files/migrate/pdf/BSCInfo_511_Basement_Insulation.pdf"
# NIST envelope thermal/airtightness design guidelines (US govt, via WBDG mirror).
fetch "$ME/building-science/NIST-IR-4821-envelope-design-guidelines-thermal-airtightness.pdf" "https://www.wbdg.org/FFC/NIST/nist4821.pdf"
# NIST commercial building airtightness requirements & measurements (US govt).
fetch "$ME/building-science/NIST-commercial-building-airtightness-requirements.pdf" "https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=909521"
# DOE Building America measure guidelines (envelope / drainage plane / ventilation).
fetch "$ME/building-science/DOE-measure-guideline-taped-insulating-sheathing-drainage-planes.pdf" "https://www1.eere.energy.gov/buildings/publications/pdfs/building_america/measure_guideline_guidance.pdf"
fetch "$ME/building-science/DOE-measure-guideline-ventilation-cooling.pdf"      "https://www1.eere.energy.gov/buildings/publications/pdfs/building_america/measure_guide_vent_cooling.pdf"

echo ""
echo "========================================"
echo "OK=$ok  SKIP=$skip  FAIL=$fail"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "---- FAILED ----"
  for f in "${FAILED[@]}"; do echo "  $f"; done
fi
