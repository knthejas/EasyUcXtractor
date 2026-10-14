"""Serve the local EasyUcXtractor RAG interface and its small JSON API."""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from _bootstrap import ROOT
sys.path.insert(0, str(ROOT / "src"))

from easyucxtractor.config import llm_settings
from easyucxtractor.llm.ollama_client import answer_from_evidence, small_talk_response
from easyucxtractor.paths import KNOWLEDGE_DIR
from easyucxtractor.retrieval.bm25_retriever import load_chunks, retrieve, source_label
from run_rag import find_dataset

UI_FILE = ROOT / "ui" / "index.html"
MAX_REQUEST_BYTES = 64 * 1024


def list_datasets() -> list[dict]:
    datasets = []
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
    dataset = find_dataset(dataset_id)
    chunks = load_chunks(dataset / "knowledge_base.jsonl", dataset / "parameters.jsonl")
    results = retrieve(question, chunks)
    answer = answer_from_evidence(question, results)
    evidence = [{
        "source": source_label(chunk),
        "content": chunk.get("content", "")[:1800],
        "content_type": chunk.get("content_type", "unknown"),
    } for _score, chunk in results]
    return {"dataset_id": dataset.name, "answer": answer, "evidence": evidence, "mode": "rag"}


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
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        if self.path != "/api/ask":
            self.send_error(404, "Not found")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > MAX_REQUEST_BYTES:
                self.send_json(413, {"error": "Request must be between 1 byte and 64 KB."})
                return
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            question = payload.get("question")
            dataset_id = payload.get("dataset_id") or None
            if not isinstance(question, str) or not question.strip():
                self.send_json(400, {"error": "Enter a question before asking."})
                return
            self.send_json(200, ask_question(question.strip(), dataset_id))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, {"error": "Request body must be valid JSON encoded as UTF-8."})
        except SystemExit as error:
            self.send_json(400, {"error": str(error)})
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
    print(f"EasyUcXtractor is running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the local server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping EasyUcXtractor.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
