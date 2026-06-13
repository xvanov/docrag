#!/usr/bin/env bash
# Fetch free, directly-downloadable building/methods sources into the corpora tree.
# Idempotent: skips files already on disk. Re-run safe. Reports OK/FAIL/SKIP per file.
#
#   bash scraper/fetch_sources.sh
#
# Gated sources (registration/login walls) are NOT here -- see scraper/GATED_SOURCES.md.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

ok=0; fail=0; skip=0
declare -a FAILED

fetch() {  # fetch <relative_path> <url>
  local rel="$1" url="$2"
  local dest="$ROOT/$rel"
  if [[ -s "$dest" ]]; then echo "SKIP  $rel"; skip=$((skip+1)); return; fi
  mkdir -p "$(dirname "$dest")"
  if curl -sSL -f --connect-timeout 30 --max-time 600 -A "$UA" -o "$dest" "$url" 2>/dev/null; then
    local sz; sz=$(wc -c < "$dest" 2>/dev/null || echo 0)
    if [[ "$sz" -lt 1024 ]]; then echo "FAIL  $rel (tiny ${sz}B)"; rm -f "$dest"; fail=$((fail+1)); FAILED+=("$rel  <-  $url"); else
      echo "OK    $rel  ($((sz/1024)) KB)"; ok=$((ok+1)); fi
  else
    echo "FAIL  $rel"; rm -f "$dest"; fail=$((fail+1)); FAILED+=("$rel  <-  $url")
  fi
}

CB="corpora/building-codes"
ME="corpora/methods"

echo "===== CODES: NC STATE ====="
fetch "$CB/nc-state/NCGS-Chapter-160D-land-development.pdf" "https://www.ncleg.gov/EnactedLegislation/Statutes/PDF/ByChapter/Chapter_160D.pdf"
fetch "$CB/nc-state/15A-NCAC-18E-onsite-wastewater.pdf" "https://ehs.dph.ncdhhs.gov/oswp/docs/15A-NCAC-18E.pdf"
fetch "$CB/nc-state/NC-erosion-sediment-control-design-manual-2013.pdf" "https://www.deq.nc.gov/energy-mineral-and-land-resources/land-quality/erosion-and-sediment-control-planning-and-design-manual/erosion-design-manual-rev-may-2013-compressed/download"
fetch "$CB/nc-state/15A-NCAC-02H-1000-post-construction-stormwater.pdf" "https://www.deq.nc.gov/energy-mineral-and-land-resources/stormwater/stormwater-rule-readoption/15a-ncac-02h-1000/download"

echo "===== CODES: NC WATERSHED / BUFFER RULES ====="
OAH="http://reports.oah.state.nc.us/ncac/title%2015a%20-%20environmental%20quality/chapter%2002%20-%20environmental%20management/subchapter%20b"
fetch "$CB/nc-state/watershed/02B-0267-jordan-lake-buffer.pdf"  "$OAH/15a%20ncac%2002b%20.0267.pdf"
fetch "$CB/nc-state/watershed/02B-0714-neuse-buffer.pdf"        "$OAH/15a%20ncac%2002b%20.0714.pdf"
fetch "$CB/nc-state/watershed/02B-0624-water-supply-watershed.pdf" "$OAH/15a%20ncac%2002b%20.0624.pdf"
fetch "$CB/nc-state/watershed/02B-0610-buffer-definitions.pdf"  "$OAH/15a%20ncac%2002b%20.0610.pdf"
fetch "$CB/nc-state/watershed/02B-0611-buffer-authorization.pdf" "$OAH/15a%20ncac%2002b%20.0611.pdf"
fetch "$CB/nc-state/watershed/02B-0612-buffer-forest-harvest.pdf" "$OAH/15a%20ncac%2002b%20.0612.pdf"

echo "===== CODES: NCDOT ====="
fetch "$CB/ncdot/policy-street-driveway-access.pdf" "https://connect.ncdot.gov/projects/Roadway/RoadwayDesignAdministrativeDocuments/Policy%20on%20Street%20and%20Driveway%20Access.pdf"
fetch "$CB/ncdot/subdivision-roads-min-construction-standards.pdf" "https://connect.ncdot.gov/resources/Asset-Management/StateMaintOpsDocs/January%202010%20Subdivision%20Manual%20-%20Revised%20July%202020.pdf"
fetch "$CB/ncdot/2024-standard-specifications-roads-structures.pdf" "https://connect.ncdot.gov/resources/Specifications/2024StandardSpecifications/2024%20Standard%20Specifications%20for%20Roads%20and%20Structures.pdf"

echo "===== CODES: FEDERAL OVERLAYS ====="
fetch "$CB/federal/2010-ADA-standards-accessible-design.pdf" "https://archive.ada.gov/regs2010/2010ADAStandards/Guidance_2010ADAStandards.pdf"
fetch "$CB/federal/fair-housing-act-design-manual-HUD.pdf" "https://www.huduser.gov/portal/publications/PDF/FAIRHOUSING/fairfull.pdf"
fetch "$CB/federal/FEMA-480-floodplain-management-study-guide.pdf" "https://www.fema.gov/sites/default/files/documents/fema-480_floodplain-management-study-guide_local-officials.pdf"

echo "===== CODES: DURHAM CITY / COUNTY ====="
fetch "$CB/durham-local/reference-guide-for-development.pdf" "https://www.durhamnc.gov/DocumentCenter/View/3331/Reference-Guide-for-Development-"
fetch "$CB/durham-local/stormwater-design-manual-addendum.pdf" "https://www.durhamnc.gov/DocumentCenter/View/3082"
fetch "$CB/durham-local/historic-properties-local-review-criteria-2020.pdf" "https://www.durhamnc.gov/DocumentCenter/View/9716/Historic-Properties-Local-Review-Criteria-2020-PDF"
fetch "$CB/durham-county/durham-county-stormwater-ordinance-2023.pdf" "https://www.dconc.gov/Engineering-and-Environmental1/Documents/230410-Durham-Co-Stormwater-Ord-FINAL.pdf"

echo "===== METHODS: WOOD / TIMBER ====="
fetch "$ME/wood/USDA-wood-handbook-FPL-GTR-190.pdf" "https://www.fpl.fs.usda.gov/documnts/fplgtr/fpl_gtr190.pdf"

echo "===== METHODS: STEEL ====="
fetch "$ME/steel/AISI-S240-20-cold-formed-steel-framing.pdf" "https://www.steelframing.org/assets/Library/Standards/AISI-S240-20.pdf"

echo "===== METHODS: SIPS ====="
fetch "$ME/sips/SIP-engineering-design-guide-2019.pdf" "https://epsbuildings.com/wp-content/uploads/2024/06/SIP-Engineering-Design-Guide-July2019.pdf"

echo "===== METHODS: HAZARD-RESISTANT RESIDENTIAL ====="
fetch "$ME/hazard/FEMA-232-earthquake-resistant-residential.pdf" "https://www.wbdg.org/FFC/DHS/fema232.pdf"

echo "===== METHODS: EARTHEN ====="
fetch "$ME/earthen/NM-earthen-building-materials-code-14.7.4.pdf" "https://www.rld.nm.gov/wp-content/uploads/2023/07/14.7.4-Earthen-Bldg-Integrated.pdf"

echo "===== METHODS: EXPERIMENTAL (ferrocement) ====="
fetch "$ME/experimental/USN-ferrocement-manual-vol1.pdf" "https://www.boatdesign.net/ferro/ferro-1.pdf"

echo ""
echo "========================================"
echo "OK=$ok  SKIP=$skip  FAIL=$fail"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "---- FAILED ----"
  for f in "${FAILED[@]}"; do echo "  $f"; done
fi
