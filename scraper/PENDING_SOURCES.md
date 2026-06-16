# Pending sources — not yet acquired

Identified as useful but not downloadable via plain `curl`. Two classes.

## 1. Bot-walled (need Playwright — we already have it in `icc_session.py`)

`fema.gov` sits behind Akamai bot manager → returns **403** to curl regardless of
User-Agent/headers. Fetch via a headless browser instead.

| Source | URL | Bucket |
|---|---|---|
| FEMA 480 — NFIP Floodplain Mgmt Requirements (study guide) | https://www.fema.gov/sites/default/files/documents/fema-480_floodplain-management-study-guide_local-officials.pdf | codes/federal |
| FEMA P-499 — Home Builder's Guide to Coastal Construction (37 fact sheets) | https://www.fema.gov/sites/default/files/2020-08/fema499_2010_edition.pdf | methods/hazard |
| FEMA NFIP Technical Bulletins (TB-0…TB-11) | https://www.fema.gov/emergency-managers/risk-management/building-science/national-flood-insurance-technical-bulletins | codes/federal |

Note: FEMA 232 already acquired via a wbdg.org mirror — same trick (find a
non-fema.gov mirror) may unblock the above without Playwright.

## 1b. JS/session-gated — resisted Playwright (manual browser download needed)

Attempted via `scraper/fetch_botwalled.py` (Playwright). These sit behind a
Cloudflare JS/session gate that defeats download-event, page-navigation, AND
APIRequestContext (`request.get`) — tougher than FEMA's Akamai (which yielded to
Chromium's download event). The static URLs return Cloudflare **404/403** to every
programmatic client. **Grab via a real logged-in browser (one click each):**

| Source | Direct URL (open in browser) | Bucket |
|---|---|---|
| ICC-ES ESR-4126 — Cal-Earth SuperAdobe (earthbag → code-allowed; HIGH value) | https://icc-es.org/wp-content/uploads/report-directory/ESR-4126.pdf | methods/earthbag |
| MDPI — Sustainable Earth Construction Materials: State-of-Art Review (CC-BY) | https://www.mdpi.com/2071-1050/16/2/670 (Download PDF) | methods/earthen-reviews |
| MDPI — Compressed Stabilized Earth Blocks: PRISMA systematic review (CC-BY) | https://www.mdpi.com/2075-5309/16/8/1633 | methods/earthen-reviews |
| MDPI — Carbon Conscious Construction: CSEB (CC-BY) | https://www.mdpi.com/2075-5309/15/23/4362 | methods/earthen-reviews |
| MDPI — Rammed Earth Architecture case study (CC-BY) | https://www.mdpi.com/2075-5309/14/12/4034 | methods/earthen-reviews |

## 2. Registration / paywall gated (need an account or purchase)

| Source | Where | Bucket | Access |
|---|---|---|---|
| **AISC 360/341/358/303-22 steel specs** | aisc.org / store.accuristech.com | methods | **PURCHASE (copyrighted/DRM — NOT free; earlier "free" claim was wrong)** |
| FPInnovations CLT Handbook (US ed.) | fpinnovations.ca | methods | free, form/registration |
| WoodWorks Mass Timber Design Manual | woodworks.org | methods | free, form-gated |
| APA Engineered Wood Construction Guide E30 | apawood.org | methods | free, registration |
| BIA Technical Notes (brick, ~100 bulletins) | gobrick.com | methods | free, registration |
| NCMA TEK notes (CMU, ~200 bulletins) | ncma.org | methods | free, registration |
| ICC-ES AC509 (3D-printed concrete) | shop.iccsafe.org | methods | purchase |
| ICC-ES AC462 (shipping containers) | via `ICCG52019` on codes.iccsafe.org | methods | ICC Premium scrape |
| ISO 22156 bamboo structures | iso.org | methods | purchase (~CHF 181) |
| ICC Commentaries (IBC/IRC/IFC/IECC 2024) | codes.iccsafe.org | codes | ICC Premium scrape |
| ICC 500 storm shelters / ICC 600 high-wind | codes.iccsafe.org | codes | ICC Premium scrape |

## 3. Cannot ingest (copyright — cite by reference only)

ASCE 7, ACI 318, TMS 402/602, AWC NDS, ASHRAE 90.1/62.1, NFPA 70/13/72.
Referenced by the I-Codes but licensed text; RAG cites section numbers, not full text.

## 4. Portal-only (HTML, no bulk PDF — needs periodic re-scrape)

Durham City Code (Municode), Durham County Code (Municode), UDO pending-amendments
tracker, draft New UDO modules, NC DEQ Stormwater SCM Design Manual (per-chapter PDFs).
