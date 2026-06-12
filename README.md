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

## Supported file types

`.pdf .docx .doc .txt .md .markdown .rst .csv .log .json .xml .html .htm`
Images, archives, and other binaries are skipped automatically.

## Layout

```
docrag/
  settings.py    config (.env + app.settings.json), corpus/index paths
  extractors.py  PDF / DOCX / DOC text extraction (vendored, PyPDF2 + python-docx)
  extract.py     walk a corpus folder, decide eligibility, extract
  chunk.py       page-wise (PDF) / sliding-window (text) chunking
  embed.py       Azure OpenAI embedding client (batched, backoff)
  db.py          SQLite + sqlite-vec + FTS5 schema (one DB per corpus)
  query.py       hybrid vector + BM25 retrieval (RRF)
  answer.py      grounded synthesis with [N] citations + refusal gate
  index.py       build / status / purge CLI
server/
  server.py      stdlib http.server web app
  web/           index.html + chat.css + chat.js (no build step)
```

## Self-tests

```powershell
python -m docrag.db --self-test    # exercises the schema end-to-end -> "ok"
```
