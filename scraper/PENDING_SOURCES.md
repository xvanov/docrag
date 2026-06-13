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

## 2. Registration / paywall gated (need an account or purchase)

| Source | Where | Bucket | Access |
|---|---|---|---|
| FPInnovations CLT Handbook (US ed.) | fpinnovations.ca | methods | free, form/registration |
| WoodWorks Mass Timber Design Manual | woodworks.org | methods | free, form-gated |
| APA Engineered Wood Construction Guide E30 | apawood.org | methods | free, registration |
| BIA Technical Notes (brick, ~100 bulletins) | gobrick.com | methods | free, registration |
| NCMA TEK notes (CMU, ~200 bulletins) | ncma.org | methods | free, registration |
| AISC 360/341 specs | aisc.org | methods | free, registration |
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
