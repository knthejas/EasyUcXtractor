"""Serve the local EasyUcXtractor RAG interface and its small JSON API."""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from _bootstrap import ROOT
sys.path.insert(0, str(ROOT / "src"))

from easyucxtractor.config import llm_settings
from easyucxtractor.library import DocumentLibrary, MAX_PDF_BYTES
from easyucxtractor.llm.ollama_client import answer_from_evidence, small_talk_response
from easyucxtractor.paths import KNOWLEDGE_DIR
from easyucxtractor.retrieval.bm25_retriever import load_chunks, retrieve, source_label
from run_rag import find_dataset

UI_FILE = ROOT / "ui" / "index.html"
MAX_REQUEST_BYTES = 64 * 1024
LIBRARY = None
LIBRARY_LOCK = threading.Lock()


def get_library() -> DocumentLibrary:
    global LIBRARY
    with LIBRARY_LOCK:
        if LIBRARY is None:
            LIBRARY = DocumentLibrary()
    return LIBRARY


def list_datasets() -> list[dict]:
    datasets = [dict(row, label=row["filename"], origin="library")
                for row in get_library().list_documents()]
    for path in sorted(KNOWLEDGE_DIR.glob("*")):
        kb_path = path / "knowledge_base.jsonl"
        if not kb_path.is_file():
            continue
        with kb_path.open(encoding="utf-8") as source:
            count = sum(1 for line in source if line.strip())
        datasets.append({
            "id": path.name,
            "label": path.name.rsplit("-", 1)[0].replace("_", " "),
            "chunk_count": count,
            "status": "ready",
            "stage": "complete",
            "progress": 100,
            "origin": "cli",
        })
    return datasets


def ollama_status() -> dict:
    endpoint, model, _api_key = llm_settings()
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = base + "/api/tags"
    try:
        opener = build_opener(ProxyHandler({}))
        request = Request(url, headers={"Accept": "application/json"})
        with opener.open(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        installed = [entry.get("name", "") for entry in payload.get("models", [])]
        model_available = any(name == model or name.startswith(model + ":") for name in installed)
        return {"available": True, "model": model, "model_available": model_available}
    except (URLError, OSError, TimeoutError, ValueError, TypeError):
        return {"available": False, "model": model, "model_available": False}


def ask_question(question: str, dataset_id: str | None) -> dict:
    small_talk = small_talk_response(question)
    if small_talk:
        return {"dataset_id": None, "answer": small_talk, "evidence": [], "mode": "conversation"}
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("Select a ready datasheet before asking a datasheet question.")
    if len(dataset_id) == 64 and all(character in "0123456789abcdef" for character in dataset_id):
        knowledge, parameters = get_library().knowledge_paths(dataset_id)
    else:
        available = {row["id"] for row in list_datasets() if row["origin"] == "cli"}
        if dataset_id not in available:
            raise ValueError("Unknown datasheet. Select a document from the sidebar.")
        dataset = find_dataset(dataset_id)
        knowledge, parameters = dataset / "knowledge_base.jsonl", dataset / "parameters.jsonl"
    chunks = load_chunks(knowledge, parameters)
    results = retrieve(question, chunks)
    answer = answer_from_evidence(question, results)
    evidence = [{
        "source": source_label(chunk),
        "content": chunk.get("content", "")[:1800],
        "content_type": chunk.get("content_type", "unknown"),
    } for _score, chunk in results]
    return {"dataset_id": dataset_id, "answer": answer, "evidence": evidence, "mode": "rag"}


class AppHandler(BaseHTTPRequestHandler):
    server_version = "EasyUcXtractor/1.0"

    def log_message(self, format_string, *args):
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            try:
                body = UI_FILE.read_bytes()
            except OSError:
                self.send_error(500, "UI file is unavailable")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/datasets":
            self.send_json(200, {"datasets": list_datasets()})
        elif self.path == "/api/status":
            self.send_json(200, ollama_status())
        elif self.path.startswith("/api/documents/"):
            try:
                identifier = self.path.removeprefix("/api/documents/")
                self.send_json(200, {"document": get_library().public(get_library().get(identifier))})
            except ValueError as error:
                self.send_json(404, {"error": str(error)})
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.send_json(403, {"error": "Requests must originate from this local app."})
            return
        if self.path == "/api/upload":
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > MAX_PDF_BYTES:
                    self.send_json(413, {"error": "Choose a PDF between 1 byte and 50 MB."})
                    return
                if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/pdf":
                    self.send_json(415, {"error": "Only PDF uploads are accepted."})
                    return
                self.connection.settimeout(60)
                content = self.rfile.read(size)
                if len(content) != size:
                    raise ValueError("Upload was incomplete. Please try again.")
                filename = unquote(self.headers.get("X-Filename", "upload.pdf"))
                document, reused = get_library().upload(content, filename)
                self.send_json(200 if reused else 202, {"document": document, "reused": reused})
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            except Exception:
                self.send_json(500, {"error": "Upload failed. Check local disk space and try again."})
            return
        if self.path.startswith("/api/documents/") and self.path.endswith("/retry"):
            try:
                identifier = self.path[len("/api/documents/"):-len("/retry")]
                self.send_json(202, {"document": get_library().retry(identifier)})
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            except Exception:
                self.send_json(500, {"error": "Retry could not start. Check the local server and disk space."})
            return
        if self.path != "/api/ask":
            self.send_error(404, "Not found")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > MAX_REQUEST_BYTES:
                self.send_json(413, {"error": "Request must be between 1 byte and 64 KB."})
                return
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Request must be a JSON object.")
            question = payload.get("question")
            dataset_id = payload.get("dataset_id") or None
            if not isinstance(question, str) or not question.strip():
                self.send_json(400, {"error": "Enter a question before asking."})
                return
            self.send_json(200, ask_question(question.strip(), dataset_id))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, {"error": "Request body must be valid JSON encoded as UTF-8."})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except SystemExit as error:
            self.send_json(400, {"error": "Select an available datasheet from the sidebar."})
        except Exception as error:
            print(f"Request failed: {type(error).__name__}")
            self.send_json(500, {"error": "The question could not be processed. Check the server terminal."})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local EasyUcXtractor web interface.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback only)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("For safety, the web app only supports loopback host addresses.")
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    try:
        library_lock = get_library().acquire_server_lock()
    except RuntimeError as error:
        server.server_close()
        parser.error(str(error))
    get_library().recover_interrupted()
    print(f"EasyUcXtractor is running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the local server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping EasyUcXtractor.")
    finally:
        server.server_close()
        get_library().executor.shutdown(wait=True)
        library_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
