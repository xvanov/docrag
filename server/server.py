"""server.py -- localhost web chat for docrag.

A single-file stdlib http.server wrapping the docrag pipeline. No build step,
no frameworks. The UI lives in the sibling web/ folder.

    python server/server.py --start [--port PORT] [--host HOST]
    python server/server.py --resolve
    python server/server.py --stop

Routes:
    GET  /                  -> web/index.html
    GET  /chat.js, /chat.css
    GET  /api/corpora       -> {"corpora": [...]}
    GET  /api/health        -> {"ok": true, ...}
    GET  /source?corpus=&path=   -> serve a source doc inline (PDF preview)
    POST /api/chat          -> {corpus, query, history?, sources_only?, top_k?}
    POST /api/rate          -> append a JSONL feedback line
    POST /api/upload        -> multipart file upload + reindex

PID file at {index_dir}/docrag_server.pid stores ``{pid}\\n{port}\\n``.
"""

# Bootstrap: put the repo root on sys.path so `import docrag` resolves.
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.dirname(_HERE)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import datetime
import http.server
import json
import os
import re
import signal
import socket
import queue
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse

from rag import settings
from rag import facets
from rag.answer import answer as rag_answer_call
from rag.reason import answer as agentic_answer_call
from rag import settings as _settings
from rag.query import rag_query


def _agentic_enabled() -> bool:
    val = (_settings.get("DOCRAG_AGENTIC", "1") or "1").strip().lower()
    return val not in ("0", "false", "no", "off")
from rag.db import open_db, valid_corpus


DEFAULT_PORT = 8099
PORT_SCAN_RANGE = 10
WEB_DIR = os.path.join(_HERE, "web")
HISTORY_CAP = 6

# Corpora that default to jurisdiction-balanced retrieval (IBC / NC / Durham).
_BALANCED_CORPORA = {"building-codes"}

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}

SOURCE_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".markdown": "text/markdown; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".xml": "text/xml; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
}

_UPLOAD_MAX_BYTES = 100 * 1024 * 1024
_UPLOAD_ALLOWED_EXT = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".rst",
    ".csv", ".log", ".json", ".xml", ".html", ".htm",
}
_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._\-]")


def _now_iso_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _validate_corpus(corpus) -> str:
    if not isinstance(corpus, str):
        raise ValueError("invalid corpus: %r" % (corpus,))
    candidate = corpus.strip().lower()
    if not valid_corpus(candidate):
        raise ValueError("invalid corpus: %r" % (corpus,))
    return candidate


def _sanitize_filename(raw):
    base = os.path.basename(str(raw or ""))
    base = _FILENAME_SAFE_RE.sub("_", base).strip("._-")
    return base[:200] if base else None


def _list_corpora() -> list:
    idx = settings.index_dir()
    if not os.path.isdir(idx):
        return []
    out = [f[:-3] for f in os.listdir(idx) if f.lower().endswith(".db")]
    out.sort()
    return out


# Preferred unified corpus for the chat UI. The UI is not a corpus switcher --
# it locks onto one corpus and just lists the documents inside it. (The
# standalone "udo" corpus is fully contained in "building-codes".)
_PRIMARY_CORPORA = ("building-codes",)
# Subfolder -> human label for the source list.
_JURISDICTION_LABELS = {
    "model": "Model code (IBC)",
    "durham": "Durham (local)",
    "north-carolina": "North Carolina (state)",
}


def _primary_corpus() -> str:
    """The corpus the chat UI locks onto: first preferred one that exists,
    else the first corpus alphabetically, else "" (none indexed)."""
    corpora = _list_corpora()
    for pref in _PRIMARY_CORPORA:
        if pref in corpora:
            return pref
    return corpora[0] if corpora else ""


def _corpus_sources(corpus: str) -> list:
    """List the distinct source documents indexed in a corpus, with chunk
    counts and a jurisdiction label derived from the top-level subfolder."""
    conn = open_db(corpus)
    try:
        rows = conn.execute(
            "SELECT source_file, MIN(path) AS p, COUNT(*) AS n "
            "FROM chunks GROUP BY source_file ORDER BY p"
        ).fetchall()
    finally:
        conn.close()
    from rag.answer import _authority_tier
    _TIER_LABEL = {
        "STATE": "NC state law & code", "LOCAL": "Durham ordinance (UDO)",
        "LOCAL-GUIDANCE": "Durham agency guidance (durhamnc.gov)",
        "MODEL": "Model codes (IBC/IRC...)", "FEDERAL": "Federal (ADA/FEMA)",
        "OTHER": "Other",
    }
    out = []
    for source_file, path, n in rows:
        folder = (path or "").split("/")[0] if "/" in (path or "") else ""
        label = _JURISDICTION_LABELS.get(folder, folder.replace("-", " ").title())
        tier = _authority_tier({"path": path})
        out.append({
            "file": source_file,
            "path": path,
            "jurisdiction": label,
            "tier": tier,
            "tier_label": _TIER_LABEL.get(tier, tier),
            "chunks": int(n),
        })
    return out


# --- Feedback log ------------------------------------------------------------


def _feedback_path(corpus: str) -> str:
    corpus = _validate_corpus(corpus)
    base = os.path.join(settings.index_dir(), "feedback")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "%s.jsonl" % corpus)


def _write_feedback_line(payload: dict) -> str:
    corpus = _validate_corpus(payload.get("corpus", ""))
    path = _feedback_path(corpus)
    entry = {
        "ts": _now_iso_utc(), "corpus": corpus,
        "query": payload.get("query", ""), "chunk_id": payload.get("chunk_id"),
        "file": payload.get("file"), "path": payload.get("path"),
        "page": payload.get("page"), "kind": payload.get("kind"),
        "rank": payload.get("rank"), "rating": payload.get("rating", ""),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")
    return path


def _trace_path(corpus: str) -> str:
    base = os.path.join(settings.index_dir(), "traces")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "%s.jsonl" % _validate_corpus(corpus))


def _write_trace_record(corpus, query, envelope, meta):
    """Append one structured debug record (full chain: plan -> retrieval ->
    each verify round's LLM output + timings -> final answer) per request, so
    conversations can be reviewed/debugged later. Best-effort; never raises."""
    try:
        env = envelope or {}
        rec = {
            "ts": _now_iso_utc(),
            "thread_id": meta.get("thread_id"),
            "corpus": corpus,
            "query": query,
            "location": meta.get("location"),
            "balance": meta.get("balance"),
            "sources_only": bool(env.get("sources_only")),
            "elapsed_ms": env.get("elapsed_ms"),
            "refused": bool(env.get("refused")),
            "refusal_reason": env.get("refusal_reason"),
            "status": env.get("status"),
            "error": meta.get("error"),
            "tokens": env.get("tokens"),
            "n_citations": len(env.get("citations") or []),
            "n_chunks": len(env.get("chunks") or []),
            "answer": env.get("answer"),
            "plan": env.get("plan"),
            "trace": env.get("trace"),
        }
        with open(_trace_path(corpus), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("[trace] write failed: %s\n" % e)


# --- Source path resolution --------------------------------------------------


def _resolve_source(corpus: str, raw_path: str):
    """Resolve a chunk path to a file under ``{docs_root}/{corpus}/``.

    Returns abs path or None. Rejects traversal / absolute / drive specifiers.
    """
    norm = raw_path.replace("\\", "/").lstrip("/")
    if ".." in norm.split("/"):
        return None
    if os.path.isabs(norm) or (len(norm) >= 2 and norm[1] == ":"):
        return None
    root = os.path.abspath(os.path.join(settings.docs_root(), corpus))
    candidate = os.path.abspath(os.path.join(root, norm))
    try:
        if os.path.commonpath([root, candidate]) != root:
            return None
    except ValueError:
        return None
    return candidate if os.path.isfile(candidate) else None


# --- Upload + reindex --------------------------------------------------------


def _parse_multipart(content_type, body_bytes):
    """Parse multipart/form-data into {field: {value|filename+body}}."""
    m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type or "")
    if not m:
        raise ValueError("no boundary in Content-Type")
    boundary_str = (m.group(1) or m.group(2) or "").strip()
    if not boundary_str:
        raise ValueError("empty boundary")
    boundary = ("--" + boundary_str).encode("ascii")

    out = {}
    for raw in body_bytes.split(boundary):
        raw = raw.lstrip(b"\r\n")
        if not raw or raw.startswith(b"--"):
            continue
        hdr_end = raw.find(b"\r\n\r\n")
        if hdr_end < 0:
            continue
        header_block = raw[:hdr_end].decode("utf-8", errors="replace")
        body = raw[hdr_end + 4:]
        if body.endswith(b"\r\n"):
            body = body[:-2]
        cd = re.search(r"Content-Disposition:\s*form-data([^\r\n]*)",
                       header_block, re.IGNORECASE)
        if not cd:
            continue
        name_m = re.search(r'name="([^"]*)"', cd.group(1))
        if not name_m:
            continue
        name = name_m.group(1)
        fname_m = re.search(r'filename="([^"]*)"', cd.group(1))
        if fname_m is not None:
            out[name] = {"filename": fname_m.group(1), "body": body}
        else:
            out[name] = {"value": body.decode("utf-8", errors="replace")}
    return out


def _save_and_reindex(corpus, filename, body_bytes):
    docs_root = settings.docs_root()
    corpus_dir = os.path.join(docs_root, corpus)
    new_corpus = not os.path.isdir(corpus_dir)
    os.makedirs(corpus_dir, exist_ok=True)

    file_path = os.path.join(corpus_dir, filename)
    with open(file_path, "wb") as f:
        f.write(body_bytes)

    cmd = [sys.executable, "-m", "rag.index", "build",
           "--corpus", corpus, "--confirm"]
    sys.stderr.write("[chat] /api/upload running indexer: %s\n" % " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=15 * 60, cwd=_REPO_ROOT)
    stdout_tail = "\n".join((proc.stdout or "").strip().splitlines()[-30:])
    stderr_tail = "\n".join((proc.stderr or "").strip().splitlines()[-30:])
    return {
        "ok": proc.returncode == 0, "corpus": corpus, "filename": filename,
        "new_corpus": new_corpus, "size_bytes": len(body_bytes),
        "indexer_exit": proc.returncode,
        "indexer_summary": stdout_tail or "(no stdout)",
        "indexer_stderr_tail": stderr_tail,
    }


# --- PID file helpers --------------------------------------------------------


def _pid_file_path() -> str:
    return os.path.join(settings.index_dir(), "docrag_server.pid")


def _write_pid_file(pid: int, port: int) -> None:
    path = _pid_file_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("%d\n%d\n" % (pid, port))


def _read_pid_file():
    path = _pid_file_path()
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return int(lines[0]), (int(lines[1]) if len(lines) > 1 else None)
    except (OSError, ValueError, IndexError):
        return None, None


def _clear_pid_file() -> None:
    try:
        if os.path.isfile(_pid_file_path()):
            os.remove(_pid_file_path())
    except OSError:
        pass


# --- HTTP handler ------------------------------------------------------------


class DocRagHandler(http.server.BaseHTTPRequestHandler):
    # Per-connection socket timeout so a client that navigates away mid-stream
    # can't wedge a worker on a blocking write (was freezing the whole server).
    timeout = 120

    def log_message(self, fmt, *args):  # noqa: A002
        pass

    def handle(self):
        # Clients (mobile / tailnet) routinely drop a connection mid-response;
        # on Windows that surfaces as WinError 10053/10054. Swallow those so a
        # normal disconnect doesn't dump a traceback or look like a crash. The
        # threaded server isolates the thread either way.
        try:
            super().handle()
        except (ConnectionError, socket.timeout):
            pass

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status, message):
        self._send_json({"error": message, "code": status}, status=status)

    def _serve_static(self, rel_path):
        safe = os.path.normpath(rel_path).lstrip(os.sep).lstrip("/")
        if ".." in safe.replace("\\", "/").split("/"):
            self._send_error_json(403, "Forbidden")
            return
        full = os.path.join(WEB_DIR, safe)
        if not os.path.isfile(full):
            self._send_error_json(404, "Not found: %s" % safe)
            return
        ext = os.path.splitext(full)[1].lower()
        with open(full, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", MIME_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(content)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError("invalid JSON body: %s" % e)

    # --- GET ---

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._serve_static("index.html")
        elif path == "/chat.js":
            self._serve_static("chat.js")
        elif path == "/chat.css":
            self._serve_static("chat.css")
        elif path == "/api/corpora":
            self._handle_corpora()
        elif path == "/api/sources":
            self._handle_sources(urllib.parse.parse_qs(parsed.query))
        elif path == "/api/facets":
            self._handle_facets(urllib.parse.parse_qs(parsed.query))
        elif path == "/api/health":
            self._send_json({"ok": True, "ts": _now_iso_utc()})
        elif path == "/source":
            self._handle_source(urllib.parse.parse_qs(parsed.query))
        elif path.startswith("/api/"):
            self._send_error_json(404, "Unknown endpoint: %s" % path)
        else:
            self._serve_static(path.lstrip("/"))

    # --- POST ---

    def do_POST(self):  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/upload":
            self._handle_upload()
            return
        try:
            body = self._read_body()
        except ValueError as e:
            self._send_error_json(400, str(e))
            return
        if path == "/api/chat":
            self._handle_chat(body)
        elif path == "/api/rate":
            self._handle_rate(body)
        else:
            self._send_error_json(404, "Unknown endpoint: %s" % path)

    # --- handlers ---

    def _handle_corpora(self):
        try:
            self._send_json({"corpora": _list_corpora(),
                             "primary": _primary_corpus()})
        except Exception as e:  # noqa: BLE001
            self._send_error_json(500, str(e))

    def _handle_sources(self, qs):
        raw_corpus = (qs.get("corpus") or [""])[0] or _primary_corpus()
        if not raw_corpus:
            self._send_json({"corpus": "", "sources": []})
            return
        try:
            corpus = _validate_corpus(raw_corpus)
        except ValueError:
            self._send_error_json(400, "invalid corpus name")
            return
        try:
            sources = _corpus_sources(corpus)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("[sources] %s\n" % e)
            self._send_error_json(500, str(e))
            return
        self._send_json({"corpus": corpus, "sources": sources})

    def _handle_facets(self, qs):
        """Location options (static) + per-document version options (from the
        index) for the selectors. Versionless / empty corpora still return the
        location list so the global selector always works."""
        raw_corpus = (qs.get("corpus") or [""])[0] or _primary_corpus()
        out = {
            "locations": [{"key": loc["key"], "label": loc["label"]}
                          for loc in facets.LOCATIONS],
            "default_location": facets.DEFAULT_LOCATION,
            "versions": [],
            "corpus": "",
        }
        if not raw_corpus:
            self._send_json(out)
            return
        try:
            corpus = _validate_corpus(raw_corpus)
        except ValueError:
            self._send_error_json(400, "invalid corpus name")
            return
        out["corpus"] = corpus
        try:
            conn = open_db(corpus)
            try:
                out["versions"] = facets.corpus_versions(conn)
            finally:
                conn.close()
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("[facets] %s\n" % e)
        self._send_json(out)

    def _handle_chat(self, body):
        raw_corpus = body.get("corpus") or ""
        query = (body.get("query") or "").strip()
        history = body.get("history") or []
        sources_only = bool(body.get("sources_only", False))
        try:
            top_k = int(body.get("top_k", 12))
        except (TypeError, ValueError):
            top_k = 12
        top_k = max(1, min(top_k, 20))

        if not raw_corpus:
            self._send_error_json(400, "missing 'corpus'")
            return
        if not query:
            self._send_error_json(400, "missing 'query'")
            return
        try:
            corpus = _validate_corpus(raw_corpus)
        except ValueError:
            self._send_error_json(400, "invalid corpus name")
            return
        if isinstance(history, list) and len(history) > HISTORY_CAP:
            history = history[-HISTORY_CAP:]

        # Cross-jurisdiction "mode": balance retrieval across IBC / NC / Durham.
        # Auto-on for the multi-source building-codes corpus; client may override
        # with an explicit {"balance": true|false}.
        if "balance" in body:
            balance = bool(body.get("balance"))
        else:
            balance = corpus in _BALANCED_CORPORA
        if balance:
            top_k = max(top_k, 15)

        # Location (global) + version (per-document) selectors -> retrieval
        # filters. Location maps to allowed jurisdictions; versions default to
        # the latest edition of each document with any client override applied.
        location_key = (body.get("location") or facets.DEFAULT_LOCATION)
        version_overrides = body.get("versions") or {}
        filters = None
        if corpus in _BALANCED_CORPORA:
            try:
                conn = open_db(corpus)
                try:
                    eff_versions = facets.resolve_versions(conn, version_overrides)
                finally:
                    conn.close()
            except Exception as e:  # noqa: BLE001
                sys.stderr.write("[chat] version resolve failed: %s\n" % e)
                eff_versions = {}
            filters = {
                "location": location_key,
                "versions": eff_versions,
            }

        # Source include/omit (UI toggle): list of path prefixes/globs to mute.
        exclude_sources = body.get("exclude_sources") or []
        if isinstance(exclude_sources, list) and exclude_sources:
            filters = filters or {}
            filters["exclude_globs"] = [str(g) for g in exclude_sources]

        # Streaming path: for the slow agentic pipeline, stream live stage
        # updates (Server-Sent Events) so the UI shows what's happening instead
        # of a bare spinner. Only when the client opts in and the agentic path
        # is actually in play; everything else uses the plain JSON path below.
        want_stream = (bool(body.get("stream"))
                       and not sources_only
                       and balance and _agentic_enabled())
        if want_stream:
            self._handle_chat_stream(
                corpus=corpus, query=query, history=history, top_k=top_k,
                filters=filters, balance=balance, location_key=location_key,
                thread_id=body.get("thread_id"))
            return

        t0 = time.monotonic()
        try:
            if sources_only:
                retrieval = rag_query(corpus, query, top_k=top_k,
                                      filters=filters, balance=balance)
                envelope = {
                    "answer": None, "citations": [],
                    "chunks": retrieval.get("results") or [],
                    "refused": False, "refusal_reason": None,
                    "status": retrieval.get("status", "ok"),
                    "tokens": {"prompt": None, "completion": None},
                    "sources_only": True,
                }
            else:
                answer_fn = rag_answer_call
                if balance and _agentic_enabled():
                    answer_fn = agentic_answer_call  # hypothesize-verify pipeline
                envelope = answer_fn(corpus=corpus, query=query,
                                     history=history, top_k=top_k,
                                     filters=filters, balance=balance,
                                     location=location_key)
                envelope["sources_only"] = False
        except FileNotFoundError as e:
            sys.stderr.write("[chat] DB missing: %s\n" % e)
            self._send_error_json(404, "corpus index not found: %s" % corpus)
            return
        except EnvironmentError as e:
            sys.stderr.write("[chat] env error: %s\n" % e)
            self._send_error_json(503, str(e))
            return
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("[chat] error: %s\n%s\n" % (e, traceback.format_exc()))
            self._send_error_json(500, str(e))
            return

        envelope["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        envelope["corpus"] = corpus
        envelope["query"] = query
        envelope["balance"] = balance
        envelope["location"] = location_key
        _write_trace_record(corpus, query, envelope, {
            "thread_id": body.get("thread_id"), "location": location_key,
            "balance": balance})
        envelope.pop("trace", None)
        envelope.pop("plan", None)
        self._send_json(envelope)

    def _sse(self, event, data):
        """Write one Server-Sent Event. Returns False if the socket is gone."""
        try:
            payload = "event: %s\ndata: %s\n\n" % (
                event, json.dumps(data, ensure_ascii=True))
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _handle_chat_stream(self, corpus, query, history, top_k,
                            filters, balance, location_key, thread_id=None):
        """Run the agentic answer in a worker thread, streaming live stage
        events to the client as SSE, then a terminal 'done' (or 'error')."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
        self.end_headers()

        events = queue.Queue()
        result = {}

        def on_progress(ev):
            events.put(("progress", ev))

        def run():
            try:
                env = agentic_answer_call(
                    corpus=corpus, query=query, history=history, top_k=top_k,
                    filters=filters, balance=balance, location=location_key,
                    progress=on_progress)
                env["sources_only"] = False
                result["env"] = env
            except FileNotFoundError as e:
                result["error"] = ("not_found", "corpus index not found: %s" % corpus)
                sys.stderr.write("[chat-stream] DB missing: %s\n" % e)
            except EnvironmentError as e:
                result["error"] = ("env", str(e))
                sys.stderr.write("[chat-stream] env error: %s\n" % e)
            except Exception as e:  # noqa: BLE001
                result["error"] = ("error", str(e))
                sys.stderr.write("[chat-stream] error: %s\n%s\n"
                                 % (e, traceback.format_exc()))
            finally:
                events.put(("__end__", None))

        t0 = time.monotonic()
        worker = threading.Thread(target=run, daemon=True)
        worker.start()

        STREAM_MAX = 180  # hard ceiling (s) on one answer stream
        alive = True
        while True:
            if time.monotonic() - t0 > STREAM_MAX:
                self._sse("error", {"error": "answer timed out", "kind": "timeout"})
                return
            try:
                kind, payload = events.get(timeout=10)
            except queue.Empty:
                # heartbeat keeps intermediaries from closing an idle stream;
                # if the client is gone, stop pumping (worker is a daemon).
                if not alive:
                    return
                alive = self._sse("ping", {"t": int(time.monotonic() - t0)})
                continue
            if kind == "__end__":
                break
            if alive:
                alive = self._sse(kind, payload)
                if not alive:
                    return  # client disconnected -- free the handler

        worker.join(timeout=1.0)
        if "error" in result:
            kind, msg = result["error"]
            _write_trace_record(corpus, query, {}, {
                "thread_id": thread_id, "location": location_key,
                "balance": balance, "error": msg})
            self._sse("error", {"error": msg, "kind": kind})
            return
        env = result.get("env") or {}
        env["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        env["corpus"] = corpus
        env["query"] = query
        env["balance"] = balance
        env["location"] = location_key
        _write_trace_record(corpus, query, env, {
            "thread_id": thread_id, "location": location_key, "balance": balance})
        env.pop("trace", None)   # debug-only; keep the client payload lean
        env.pop("plan", None)
        self._sse("done", env)

    def _handle_source(self, qs):
        raw_corpus = (qs.get("corpus") or [""])[0]
        raw_path = (qs.get("path") or [""])[0]
        if not raw_corpus or not raw_path:
            self._send_error_json(400, "missing 'corpus' or 'path'")
            return
        try:
            corpus = _validate_corpus(raw_corpus)
        except ValueError:
            self._send_error_json(400, "invalid corpus name")
            return
        candidate = _resolve_source(corpus, raw_path)
        if not candidate:
            self._send_error_json(404, "source not found: %s" % raw_path)
            return
        ext = os.path.splitext(candidate)[1].lower()
        ctype = SOURCE_MIME_TYPES.get(ext, "application/octet-stream")
        try:
            with open(candidate, "rb") as f:
                content = f.read()
        except OSError as e:
            sys.stderr.write("[source] read failed: %s\n" % e)
            self._send_error_json(500, "read failed")
            return
        fname = os.path.basename(candidate)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Content-Disposition",
                         'inline; filename="%s"' % fname.replace('"', ""))
        self.send_header("Cache-Control", "private, max-age=300")
        self.end_headers()
        self.wfile.write(content)

    def _handle_rate(self, body):
        for key in ("corpus", "rating"):
            if not body.get(key):
                self._send_error_json(400, "missing '%s'" % key)
                return
        if body.get("rating") not in ("good", "bad", "clear"):
            self._send_error_json(400, "rating must be good|bad|clear")
            return
        try:
            path = _write_feedback_line(body)
        except ValueError:
            self._send_error_json(400, "invalid corpus name")
            return
        except OSError as e:
            self._send_error_json(500, "feedback write failed: %s" % e)
            return
        self._send_json({"ok": True, "path": path})

    def _handle_upload(self):
        ctype = self.headers.get("Content-Type", "") or ""
        if "multipart/form-data" not in ctype.lower():
            self._send_error_json(400, "expected multipart/form-data")
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            self._send_error_json(400, "empty body")
            return
        if length > _UPLOAD_MAX_BYTES:
            self._send_error_json(413, "file too large (max %d bytes)" % _UPLOAD_MAX_BYTES)
            return
        try:
            raw = self.rfile.read(length)
            parts = _parse_multipart(ctype, raw)
        except ValueError as e:
            self._send_error_json(400, "bad multipart: %s" % e)
            return
        except Exception as e:  # noqa: BLE001
            self._send_error_json(500, "read body failed: %s" % e)
            return

        corpus_raw = (parts.get("corpus") or {}).get("value") or ""
        file_part = parts.get("file")
        if not corpus_raw.strip():
            self._send_error_json(400, "missing 'corpus' field")
            return
        if not file_part or not file_part.get("filename"):
            self._send_error_json(400, "missing 'file' field")
            return
        try:
            corpus = _validate_corpus(corpus_raw.strip().lower())
        except ValueError:
            self._send_error_json(400, "invalid corpus (use lowercase a-z, 0-9, -, _)")
            return
        filename = _sanitize_filename(file_part.get("filename") or "")
        if not filename:
            self._send_error_json(400, "invalid filename")
            return
        ext = os.path.splitext(filename)[1].lower()
        if ext not in _UPLOAD_ALLOWED_EXT:
            self._send_error_json(400, "extension %r not allowed (allowed: %s)"
                                  % (ext, ", ".join(sorted(_UPLOAD_ALLOWED_EXT))))
            return
        body_bytes = file_part.get("body") or b""
        if not body_bytes:
            self._send_error_json(400, "empty file body")
            return
        try:
            result = _save_and_reindex(corpus, filename, body_bytes)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("[chat] /api/upload failed: %s\n%s\n"
                             % (e, traceback.format_exc()))
            self._send_error_json(500, "upload failed: %s" % e)
            return
        self._send_json(result)


# --- lifecycle ---------------------------------------------------------------


def _port_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def _find_port(preferred: int = DEFAULT_PORT) -> int:
    for offset in range(PORT_SCAN_RANGE + 1):
        if _port_available(preferred + offset):
            return preferred + offset
    raise RuntimeError("No available port in range %d-%d"
                       % (preferred, preferred + PORT_SCAN_RANGE))


def start_server(port=None, host="127.0.0.1"):
    if port is None:
        port = _find_port(DEFAULT_PORT)
    elif not _port_available(port):
        sys.stderr.write("[docrag] port %d busy; scanning\n" % port)
        port = _find_port(port)

    # Threaded so one slow agentic answer (or a stalled SSE client) never blocks
    # other requests -- critical now that answers stream for tens of seconds and
    # the app is shared over the tailnet.
    server = http.server.ThreadingHTTPServer((host, port), DocRagHandler)
    _write_pid_file(os.getpid(), port)

    def _on_sigint(_s, _f):
        try:
            server.shutdown()
        except Exception:
            pass

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):
        pass

    sys.stderr.write("[docrag] listening on http://%s:%d\n" % (host, port))
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.server_close()
        except Exception:
            pass
        _clear_pid_file()
        sys.stderr.write("[docrag] stopped\n")


def stop_server() -> int:
    pid, _ = _read_pid_file()
    if not pid:
        sys.stderr.write("[docrag] no PID file at %s\n" % _pid_file_path())
        return 1
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           capture_output=True, timeout=10)
        else:
            os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError) as e:
        sys.stderr.write("[docrag] kill pid=%d failed: %s\n" % (pid, e))
    _clear_pid_file()
    sys.stderr.write("[docrag] stopped pid=%d\n" % pid)
    return 0


def resolve_port() -> int:
    _pid, port = _read_pid_file()
    print(port if port else "")
    return 0


def _cli(argv):
    if len(argv) < 2:
        sys.stderr.write("[docrag] usage: server.py --start [--port P] "
                         "[--host H] | --resolve | --stop\n")
        return 2
    cmd = argv[1]
    if cmd == "--start":
        port = None
        host = "127.0.0.1"
        for i, arg in enumerate(argv):
            if arg == "--port" and i + 1 < len(argv):
                try:
                    port = int(argv[i + 1])
                except ValueError:
                    sys.stderr.write("[docrag] bad --port: %s\n" % argv[i + 1])
                    return 2
            elif arg == "--host" and i + 1 < len(argv):
                host = argv[i + 1]
        start_server(port, host=host)
        return 0
    if cmd == "--stop":
        return stop_server()
    if cmd == "--resolve":
        return resolve_port()
    sys.stderr.write("[docrag] unknown command: %s\n" % cmd)
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))
