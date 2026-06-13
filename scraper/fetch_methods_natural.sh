#!/usr/bin/env bash
# Fetch FREE / LEGAL natural & alternative construction-methods sources into corpora/methods/.
# Scope: earthbag/superadobe, CEB/earth-brick, rammed earth, adobe/mudbrick, cob,
#        strawbale/light-straw-clay/hempcrete, natural/traditional, earthen-standards surveys.
#
# Only legally-free material: govt (USAID/ERIC), academic open-access, NGO free resources
# (Cob Research Institute, Ecological Building Network/CASBA, CRATerre/Auroville mirrors),
# and openly-downloadable Internet Archive items (NOT borrow-only, NOT unauthorized uploads
# of copyrighted commercial books). Copyrighted books are listed in the task report, not fetched.
#
# Idempotent: skips files already on disk. Re-run safe. Reports OK/FAIL/SKIP per file.
#   bash scraper/fetch_methods_natural.sh
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
  if curl -sSL -f --connect-timeout 30 --max-time 900 -A "$UA" -o "$dest" "$url" 2>/dev/null; then
    local sz; sz=$(wc -c < "$dest" 2>/dev/null || echo 0)
    # must be >50KB and start with %PDF
    local magic; magic=$(head -c 4 "$dest" 2>/dev/null)
    if [[ "$sz" -lt 51200 ]]; then
      echo "FAIL  $rel (tiny ${sz}B)"; rm -f "$dest"; fail=$((fail+1)); FAILED+=("$rel  <-  $url")
    elif [[ "$magic" != "%PDF" ]]; then
      echo "FAIL  $rel (not PDF: '$magic')"; rm -f "$dest"; fail=$((fail+1)); FAILED+=("$rel  <-  $url")
    else
      echo "OK    $rel  ($((sz/1024)) KB)"; ok=$((ok+1))
    fi
  else
    echo "FAIL  $rel"; rm -f "$dest"; fail=$((fail+1)); FAILED+=("$rel  <-  $url")
  fi
}

M="corpora/methods"

echo "===== EARTHBAG / SUPERADOBE ====="
# Ralph Pelly - Plastic Limit Analysis of Earthbag Structures (academic study, free on earthbagbuilding.com)
fetch "$M/earthbag/pelly-plastic-limit-analysis-earthbag-structures.pdf" \
  "https://www.earthbagbuilding.com/pdf/pelly.pdf"

echo "===== COMPRESSED EARTH BLOCK (CEB) / EARTH BRICK ====="
# CRATerre-EAG (Rigassi) - Manual of Production (open mirror)
fetch "$M/earth-block-ceb/craterre-ceb-manual-of-production-rigassi.pdf" \
  "https://rexresearch1.com/HouseConstructionLibrary/CompressedEarthBlockManualProduction.pdf"
# CRATerre (Guillaud/Joffroy/Odul) - CEB Vol II: Manual of Design & Construction (Internet Archive, text PDF)
fetch "$M/earth-block-ceb/craterre-ceb-vol2-design-and-construction-manual.pdf" \
  "https://archive.org/download/CEB_manual_design_construction/6020_CEB_vol2.pdf"
# CEB Design Manual (Internet Archive, text PDF) - alternate full design manual
fetch "$M/earth-block-ceb/ceb-design-manual-guillaud-joffroy-odul.pdf" \
  "https://archive.org/download/ceb-design-manual/CEB_Design_manual_text.pdf"
# SKAT & CRATerre - Stabilised CEB Production Manual for the Great Lakes Region (Internet Archive)
fetch "$M/earth-block-ceb/skat-craterre-sceb-production-manual-great-lakes.pdf" \
  "https://archive.org/download/SCEB_Production/13-12-03_CEB-Production.pdf"
# Auroville Earth Institute (Satprem Maini) - Building the Future with Earth (GRIHA / NCGD 2012)
fetch "$M/earth-block-ceb/auroville-maini-building-the-future-with-earth-griha2012.pdf" \
  "https://www.grihaindia.org/events/ncgd/2012/pdf/satprem.pdf"
echo "===== RAMMED EARTH ====="
# Rammed Earth Structures: a code of practice (Keable; UK 1996 / Zimbabwe national std) - open mirror
fetch "$M/rammed-earth/rammed-earth-structures-code-of-practice-keable.pdf" \
  "https://www.rexresearch1.com/HouseConstructionLibrary/RammedEarthstructurescodepractice.pdf"
# Niroumand et al. - Rammed earth theory in earth architecture (academicjournals open access)
fetch "$M/rammed-earth/niroumand-rammed-earth-theory-in-earth-architecture.pdf" \
  "https://academicjournals.org/article/article1389266044_Niroumand%20et%20al.pdf"
# NZS 4297:1998 Engineering Design of Earth Buildings (third-party posted copy)
fetch "$M/rammed-earth/nzs-4297-1998-engineering-design-of-earth-buildings.pdf" \
  "https://ecohabitar.org/wp-content/uploads/2023/08/NZD4297-1998-Engineering_Design_of_Earth_Buildings.pdf"
# NZS 4298:1998 Materials & Workmanship for Earth Buildings (CRI-hosted supporting doc)
fetch "$M/rammed-earth/nzs-4298-1998-materials-and-workmanship-earth-buildings.pdf" \
  "https://cobcode.s3.amazonaws.com/supporting-docs/NZS4298-1998-Materials_and_Workmanship_For_Earth_Buildings.pdf"

echo "===== ADOBE / MUDBRICK ====="
# USAID/Texas A&M (Wolfskill/Dunlap/Callaway) - Handbook for Building Homes of Earth (ERIC fulltext)
fetch "$M/adobe-mudbrick/usaid-handbook-for-building-homes-of-earth-wolfskill.pdf" \
  "https://files.eric.ed.gov/fulltext/ED242877.pdf"

echo "===== COB ====="
# Cob Research Institute - IRC Appendix U approval & use overview
fetch "$M/cob/cri-irc-appendixU-cob-construction-approval-overview.pdf" \
  "https://cobcode.org/docs/IRCAppendixU-CobConstruction_Approval&UseOverview_7.15.20.pdf"
# CRI - proposed Appendix U public comment (RB299-19)
fetch "$M/cob/cri-rb299-19-proposed-appendixU-cob-public-comment.pdf" \
  "https://cobcode.s3.amazonaws.com/RB299-19_IRC_ProposedAppendixU_CobConstruction_PublicComment.pdf"
# CRI - cob structural calculations outline
fetch "$M/cob/cri-cob-structural-calculations-outline-2019.pdf" \
  "https://cobcode.s3.amazonaws.com/AppendixU_CobConstruction-StructuralCalculationsOutline-2019.02.06.pdf"
# CRI - ASTM E119 fire-resistance test report (cob wall, NTA QS032921-80)
fetch "$M/cob/cri-astm-e119-fire-resistance-test-report-cob-wall.pdf" \
  "https://cobcode.s3.amazonaws.com/supporting-docs/QS032921-80+Report-D(Signed).pdf"
# CRI - fire-resistance engineering evaluation (cob wall)
fetch "$M/cob/cri-fire-resistance-engineering-evaluation-cob-wall.pdf" \
  "https://cobcode.s3.amazonaws.com/supporting-docs/QS032921-80+Engineering+Evaluation_2022-1-10+(Signed)-unlocked.pdf"

echo "===== STRAW BALE / HEMPCRETE (CASBA / Ecological Building Network) ====="
EBN="https://ecobuildnetwork.org/wp-content/uploads/2024/12"
# IRC 2015 Appendix S - Strawbale Construction (code text)
fetch "$M/strawbale-hempcrete/irc-2015-appendixS-strawbale-construction.pdf" \
  "$EBN/AppendixS_SBConstruction_2015IRC.pdf"
# Bruce King - Load-Bearing Straw Bale Construction: worldwide testing summary
fetch "$M/strawbale-hempcrete/king-load-bearing-strawbale-worldwide-testing-summary.pdf" \
  "https://tallerconco.org/wp-content/uploads/2017/05/LoadBearingTestsWorldWide-BruceKing.pdf"
# Basis for Prescriptive Use of Plastered Strawbale Walls as Braced Wall Panels in the IRC
fetch "$M/strawbale-hempcrete/prescriptive-strawbale-braced-wall-panels-irc.pdf" \
  "$EBN/PrescriptiveStrawbaleBracedPanelsIRC.Feb_.28.2016-1.pdf"
# Seismic Design Factors & Allowable Shears for Strawbale Wall Assemblies
fetch "$M/strawbale-hempcrete/seismic-design-factors-allowable-shears-strawbale.pdf" \
  "$EBN/Seismic-Design-Factors-and-Allowable-Shears-for-Strawbale-Wall-Assemblies.September.26.2013.pdf"
# In-plane cyclic test of plastered straw bale walls (U. Washington 1999)
fetch "$M/strawbale-hempcrete/inplane-cyclic-test-plastered-strawbale-uwashington-1999.pdf" \
  "$EBN/inplanecyclicsbwalls_uwashington1999-1.pdf"
# In-plane cyclic test of plastered straw bale walls (U. Illinois, Ash/Aschheim 2003)
fetch "$M/strawbale-hempcrete/inplane-cyclic-test-plastered-strawbale-uillinois-2003.pdf" \
  "$EBN/InPlane_Cyclic_Test_of_Plastered_Straw_Bale_Wall_Assemblies_Ash_Aschheim_Mar_2003_Small.pdf"
# In-plane monotonic test of plastered straw bale wall (Cal Poly 2000)
fetch "$M/strawbale-hempcrete/inplane-monotonic-test-plastered-strawbale-calpoly-2000.pdf" \
  "$EBN/inplanemonotonicsbwall_calypoly2000-1.pdf"
# ASTM E84 surface burning characteristics of straw bales
fetch "$M/strawbale-hempcrete/astm-e84-surface-burning-characteristics-straw-bales.pdf" \
  "$EBN/astm_e84-1_surface-burning-characterists.pdf"
# ASTM E119 1-hr fire resistance, non-loadbearing wall w/ earth plaster (Intertek 2006)
fetch "$M/strawbale-hempcrete/astm-e119-1hr-fire-nonloadbearing-earth-plaster-strawbale.pdf" \
  "$EBN/Fire_Tests_ASTM_E119-05a_1-HR_Nonbearing_SB_Wall_Intertek_2006.pdf"
# Thermal performance of straw bale wall systems (Stone 2003)
fetch "$M/strawbale-hempcrete/thermal-performance-straw-bale-wall-systems-stone-2003.pdf" \
  "$EBN/Thermal_Performance_of_Straw_Bale_Wall_Systems_Stone_2003.pdf"
# Moisture properties of straw and plaster/straw assemblies (Straube 2003)
fetch "$M/strawbale-hempcrete/moisture-properties-straw-plaster-assemblies-straube-2003.pdf" \
  "$EBN/Moisture-Properties_of_plaster_and_Stucco_for_Strawbale_Buildings_Straube_2003.pdf"
# Properties of earth, lime, and lime-cement plasters (Lerner/Donahue 2003)
fetch "$M/strawbale-hempcrete/structural-testing-earth-lime-plasters-strawbale-lerner-2003.pdf" \
  "$EBN/Structural_Testing_of_Plasters_For_Straw_Bale_Construction_Lerner_Donahue_2003-1.pdf"
# Building with Hemp and Lime (hempcrete primer, Vote Hemp / Stanwix-Sparrow)
fetch "$M/strawbale-hempcrete/building-with-hemp-and-lime-primer.pdf" \
  "https://www.votehemp.com/wp-content/uploads/2018/09/building_with_hemp_and_lime.pdf"
# Hempcrete fact sheet (GreenBuildingAdvisor / Healthy Materials Lab)
fetch "$M/strawbale-hempcrete/hempcrete-fact-sheet.pdf" \
  "https://images.greenbuildingadvisor.com/app/uploads/2022/11/23182829/5406_1669246108_Hempcrete-Fact-Sheet-1.pdf"

echo ""
echo "========================================"
echo "OK=$ok  SKIP=$skip  FAIL=$fail"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "---- FAILED ----"
  for f in "${FAILED[@]}"; do echo "  $f"; done
fi
