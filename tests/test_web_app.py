"""Checks for the local web API adapter around the existing RAG pipeline."""

import sys
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import serve_app
from easyucxtractor.library import DocumentLibrary
from easyucxtractor.llm import ollama_client
from test_library import pdf_bytes, make_outputs


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.library = DocumentLibrary(Path(self.temp.name) / "library")
        self.addCleanup(self.library.executor.shutdown, wait=True)
        self.library_patch = patch.object(serve_app, "LIBRARY", self.library)
        self.library_patch.start()
        self.addCleanup(self.library_patch.stop)
        self.legacy = Path(self.temp.name) / "legacy"
        self.legacy.mkdir()
        self.paths_patch = patch.object(serve_app, "KNOWLEDGE_DIR", self.legacy)
        self.paths_patch.start()
        self.addCleanup(self.paths_patch.stop)

    def document(self, text):
        self.library.processor = lambda identifier: make_outputs(self.library, identifier, text)
        row, _ = self.library.upload(pdf_bytes(text), "same.pdf")
        self.library.executor.submit(lambda: None).result(timeout=10)
        return row

    def test_dataset_listing_uses_generated_knowledge_bases(self):
        self.document("fixture temperature")
        datasets = serve_app.list_datasets()
        self.assertTrue(datasets)
        self.assertEqual(datasets[0]["status"], "ready")
        self.assertEqual(datasets[0]["origin"], "library")
        self.assertNotIn("outputs", datasets[0])

    def test_empty_library_still_supports_conversation(self):
        self.assertEqual(serve_app.list_datasets(), [])
        response = serve_app.ask_question("hello", None)
        self.assertEqual(response["mode"], "conversation")
        with self.assertRaises(ValueError):
            serve_app.ask_question("voltage", None)

    def test_question_api_reuses_retrieval_and_returns_labeled_evidence(self):
        self.document("Junction Temperature range is specified in this fixture.")
        dataset = serve_app.list_datasets()[0]
        with patch.object(serve_app, "answer_from_evidence", return_value="Example answer") as answer:
            result = serve_app.ask_question("What is the junction temperature range?", dataset["id"])
        self.assertEqual(result["dataset_id"], dataset["id"])
        self.assertEqual(result["answer"], "Example answer")
        self.assertTrue(result["evidence"])
        self.assertTrue(all(entry["source"].startswith("[") for entry in result["evidence"]))
        self.assertTrue(any("Junction Temperature" in entry["content"] for entry in result["evidence"]))
        answer.assert_called_once()

    def test_retrieval_isolation_and_restart_selection(self):
        first = self.document("alpha voltage is 12 V")
        second = self.document("beta voltage is 24 V")
        restarted = DocumentLibrary(self.library.root)
        self.addCleanup(restarted.executor.shutdown, wait=True)
        with patch.object(serve_app, "LIBRARY", restarted):
            for row, expected, excluded in ((first, "alpha", "beta"), (second, "beta", "alpha")):
                with patch.object(serve_app, "answer_from_evidence", return_value="fixture answer") as answer:
                    result = serve_app.ask_question("voltage", row["id"])
                supplied = " ".join(chunk["content"] for _, chunk in answer.call_args.args[1])
                self.assertIn(expected, supplied)
                self.assertNotIn(excluded, supplied)
                self.assertTrue(all(row["id"] in item["source"] for item in result["evidence"]))
        with self.assertRaises(ValueError):
            serve_app.ask_question("voltage", "../outside")
        with self.assertRaises(ValueError):
            serve_app.ask_question("voltage", None)

    def test_missing_model_returns_only_selected_evidence(self):
        first = self.document("alpha voltage is 12 V")
        self.document("beta voltage is 24 V")
        with patch.object(ollama_client, "llm_settings", return_value=(
                "http://localhost:1", "missing-model", None)):
            with patch.object(ollama_client, "urlopen", side_effect=URLError("unavailable")):
                result = serve_app.ask_question("voltage", first["id"])
        self.assertIn("LLM request failed", result["answer"])
        self.assertIn("alpha voltage", result["answer"])
        self.assertNotIn("beta voltage", result["answer"])

    def test_unready_selection_is_rejected_without_retrieval(self):
        self.library.processor = lambda identifier: None
        row, _ = self.library.upload(pdf_bytes(), "failed.pdf")
        self.library.executor.submit(lambda: None).result(timeout=10)
        with patch.object(serve_app, "retrieve") as retrieval:
            with self.assertRaises(ValueError):
                serve_app.ask_question("voltage", row["id"])
        retrieval.assert_not_called()

    def test_http_upload_status_reuse_and_invalid_body(self):
        self.library.processor = lambda identifier: make_outputs(self.library, identifier)
        server = ThreadingHTTPServer(("127.0.0.1", 0), serve_app.AppHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self.addCleanup(cleanup)
        base = f"http://127.0.0.1:{server.server_port}"
        def upload(content, name="valid.pdf", content_type="application/pdf"):
            request = Request(base + "/api/upload", data=content, headers={
                "Content-Type": content_type, "X-Filename": name,
            }, method="POST")
            with urlopen(request, timeout=5) as response:
                return json.load(response)
        first = upload(pdf_bytes())
        self.library.executor.submit(lambda: None).result(timeout=10)
        second = upload(pdf_bytes(), "renamed.pdf")
        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])
        self.assertEqual(second["document"]["status"], "ready")
        with urlopen(base + "/api/documents/" + first["document"]["id"], timeout=5) as response:
            status = json.load(response)
        self.assertNotIn(str(self.library.root), json.dumps(status))
        for content, name, content_type, code in (
            (b"%PDF-broken", "bad.pdf", "application/pdf", 400),
            (pdf_bytes(), "bad.txt", "application/pdf", 400),
            (pdf_bytes(), "valid.pdf", "text/plain", 415),
        ):
            with self.subTest(name=name, code=code), self.assertRaises(HTTPError) as failure:
                upload(content, name, content_type)
            self.assertEqual(failure.exception.code, code)
        request = Request(base + "/api/ask", data=b"[]", headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as failure:
            urlopen(request, timeout=5)
        self.assertEqual(failure.exception.code, 400)

    def test_ui_file_exists(self):
        self.assertTrue(serve_app.UI_FILE.is_file())

    def test_greeting_bypasses_dataset_lookup_and_returns_conversation_mode(self):
        with patch.object(serve_app, "find_dataset") as find_dataset:
            with patch.object(serve_app, "retrieve") as retrieve:
                result = serve_app.ask_question("hello", None)
        self.assertEqual(result["mode"], "conversation")
        self.assertIsNone(result["dataset_id"])
        self.assertEqual(result["evidence"], [])
        self.assertIn("Hello!", result["answer"])
        find_dataset.assert_not_called()
        retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
