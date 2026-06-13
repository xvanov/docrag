# docrag

A generalized document RAG: drop PDFs / DOCX / TXT / MD into a **corpus**,
index them, and chat with grounded, cited answers.

A standalone fork of the Microvellum knowledge-RAG pipeline with all
brand / code-tier / manifest machinery removed. The unit of partition is a
*corpus* -- a named set of documents at `{docs_root}/{corpus}/` indexed into
`{index_dir}/{corpus}.db`.

Retrieval is hybrid: vector similarity (Azure OpenAI `text-embedding-3-large`)
fused with SQLite FTS5 BM25 via Reciprocal Rank Fusion. A short-uppercase-code
word-boundary filter and a low-confidence gate suppress hallucination on
out-of-corpus queries; the LLM is instructed to answer only from retrieved
chunks and to cite every claim with `[N]`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then fill in your Azure OpenAI keys
```

Requires Python 3.10+ and an Azure OpenAI resource with one embedding
deployment and one chat deployment.

## Index a corpus

Put documents under `corpora/<name>/`, then:

```powershell
# Dry run: plan + cost estimate, no embedding
python -m docrag.index build --corpus udo --dry-run

# Real build (incremental: unchanged files are skipped by SHA256)
python -m docrag.index build --corpus udo --confirm

python -m docrag.index status --corpus udo
python -m docrag.index purge  --corpus udo --file relative/path.pdf
```

Embedding cost is roughly **$0.10 - $0.20 per ~300-page PDF**. Builds over
$1.00 require `--confirm`.

## The `building-codes` corpus (combined, multi-source)

`building-codes` is a single corpus that holds building regulations from
several jurisdictions at once, so one chat searches all of them and surfaces
the most relevant passages regardless of source:

```
corpora/building-codes/
  model/    2021 International Building Code (IBC)   -- the base model code
  durham/   Durham UDO                               -- local ordinance
  <state>/  (add state codes here, e.g. north-carolina/)
```

Retrieval (hybrid vector + BM25 RRF) ranks chunks across every document in the
corpus, so an answer can pull from the IBC and Durham in the same response.
Each citation `[N]` carries its own `source_file` + page, and the LLM is
instructed to attribute each claim to its source document and to flag when a
local code amends or is more specific than the model code.

**Add another jurisdiction** (e.g. North Carolina): drop the PDF under a new
subfolder and reindex -- unchanged files are skipped by SHA256, so only the new
document is embedded.

```powershell
# via CLI
copy "NC_Building_Code.pdf" corpora\building-codes\north-carolina\
python -m docrag.index build --corpus building-codes --confirm

# or via the web UI: Upload -> existing corpus "building-codes"
```

### Cross-jurisdiction "mode" (IBC + North Carolina + Durham)

`building-codes` now holds three layers of regulation at once:

```
corpora/building-codes/
  model/            2021 International Building Code (IBC)        -- model baseline
  north-carolina/   2024 NC State Building Codes (9 books, HTML)  -- state amendments
  durham/           Durham UDO                                    -- local ordinance
```

A single question is answered from **all three** via *jurisdiction-balanced
retrieval*: hits are round-robin interleaved across the three buckets so one
code can't crowd out the others, and the LLM is told to structure the answer as
model baseline -> NC amendment -> Durham local rule, attributing each layer and
saying which governs. This mode is **on by default for `building-codes`**
(`server/server.py: _BALANCED_CORPORA`); any other corpus uses plain retrieval.

```powershell
# Terminal Q&A across all three
python -m docrag.ask "minimum stair riser height for an exit stair?"
python -m docrag.ask --sources "fire separation distance"   # chunks only, no LLM
python -m docrag.ask --no-balance "..."                       # plain RRF

# Web UI: just pick the building-codes corpus -- balancing is automatic.
# The /api/chat body also accepts an explicit {"balance": true|false}.
```

The **North Carolina codes are HTML** scraped from ICC Digital Codes (see
`scraper/`). `.html` files are tag-stripped on extraction; each NC chunk is
labeled by jurisdiction + book (e.g. *"NC 2024 Building Code"*) and chapter, so
citations are readable. `_`-prefixed sidecars (`_toc.json`, `_manifest.json`,
`_coverage.json`) are skipped by the indexer.

### Scraping the NC codes (`scraper/`)

```powershell
.venv\Scripts\python.exe scraper\scrape.py --list   # show the 9 NC 2024 books
.venv\Scripts\python.exe scraper\scrape.py          # scrape all (resumable)
```

Needs `ICC_USER` / `ICC_PASS` in `.env` (a Digital Codes Premium account).
`icc_session.py` logs in with Playwright and reuses cached cookies; `scrape.py`
pulls each book's TOC and every chapter's full HTML via the site's own JSON API.
For authorized personal/local use of your own subscription -- don't redistribute
the corpus.

## Chat (web UI)

```powershell
python server/server.py --start      # opens at http://localhost:8099/
python server/server.py --resolve    # print the actual bound port
python server/server.py --stop
```

The UI has a corpus picker, a "Sources only" toggle (skips the LLM, returns
raw chunks), a Top-K slider, and an **Upload** button that drops a file into a
corpus and reindexes it in place -- so you can build a corpus entirely from the
browser. Citations `[N]` link to the source chunks; PDF sources open at the
cited page.

## Retrieval architecture (structure-aware, 2026 best-practice)

Dense, cross-referenced code/legal text needs more than flat chunking. The
pipeline:

- **Structure-aware chunking.** HTML code books are parsed into their
  `<section>` tree (number, title, breadcrumb, parent); PDFs use per-jurisdiction
  heading detection with a monotonicity check (falling back to page chunks when
  structure is weak). Each smallest section is an embedded **leaf**; each section
  is also stored as a **parent-document node** whose `full_text` = own body + all
  descendants.
- **Metadata enrichment.** Every leaf's embedding input is prefixed with
  `edition | breadcrumb | § number title | jurisdiction`, and that string is
  indexed in a separate BM25 column (down-weighted vs. body) so exact-section
  queries hit.
- **Hybrid + rerank.** Vector + BM25 → Reciprocal Rank Fusion → optional
  cross-encoder rerank (`bge-reranker-v2-m3`) → collapse leaves to their parent
  section (small-to-big), de-duping ancestors.
- **Citation graph.** Cross-references ("see §705.8", "NCGS 160D-1110") are parsed
  at index time into `section_refs`; at query time one-hop expansion pulls the
  resolved referenced sections in as labeled supporting context.
- **Adaptive jurisdiction balance.** Interleaves model/state/local only when no
  single jurisdiction dominates the top hits.

**Optional reranker** (recommended; biggest single accuracy lever):

```powershell
pip install FlagEmbedding   # pulls torch; model auto-downloads on first query
```

If it's not installed, retrieval degrades gracefully to RRF order. Toggle with
`DOCRAG_RERANK=0`; cross-reference expansion with `DOCRAG_EXPAND_REFS=0`.

## Evaluation

A section-grounded golden set lives at `evalset/<corpus>.jsonl` (question +
expected section numbers / phrases). Because chunks carry section numbers,
retrieval scoring is exact:

```powershell
python -m docrag.eval --corpus building-codes            # retrieval + answer
python -m docrag.eval --corpus building-codes --no-answer # retrieval only (no LLM)
```

Reports hit-rate@k, MRR, and the non-refused + cited answer rate.

## Supported file types

`.pdf .docx .doc .txt .md .markdown .rst .csv .log .json .xml .html .htm`
Images, archives, and other binaries are skipped automatically.

## Layout

```
docrag/
  settings.py    config (.env + app.settings.json), corpus/index paths
  extractors.py  PDF / DOCX / DOC / HTML extraction (HTML -> section tree)
  extract.py     walk a corpus folder, decide eligibility, extract
  chunk.py       structure-aware chunking (section leaves + parent nodes) + enrichment
  embed.py       Azure OpenAI embedding client (batched, backoff)
  rerank.py      optional cross-encoder reranker (no-op if unavailable)
  db.py          SQLite + sqlite-vec + FTS5 + section_nodes + section_refs (one DB per corpus)
  query.py       hybrid + RRF + rerank + parent-collapse + citation-graph expansion
  answer.py      grounded synthesis with [N] citations, budget guard + refusal gate
  index.py       build / status / purge CLI (+ citation-graph resolution)
  eval.py        section-grounded retrieval + answer evaluation
server/
  server.py      stdlib http.server web app
  web/           index.html + chat.css + chat.js (no build step)
evalset/         golden sets ({corpus}.jsonl)
```

## Self-tests

```powershell
python -m docrag.db --self-test    # exercises the schema end-to-end -> "ok"
```
