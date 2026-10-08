"""Focused tests for retrieval-to-LLM integration; no model or network required."""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from easyucxtractor import config
from easyucxtractor.llm import ollama_client
from easyucxtractor.retrieval.bm25_retriever import source_label


class LLMIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.settings = patch.object(ollama_client, "llm_settings", return_value=("", "", None))
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.chunk = {
            "document_id": "test-document", "page": 14, "section": "Functional Range",
            "spec_number": "test-spec", "chunk_id": "test-chunk",
            "content": "Test parameter: min=2, max=7 V. Condition: test mode.",
            "source_text": "This unselected source field must not be sent.",
        }
        self.results = [(1.0, self.chunk)]
        self.question = "What is the test parameter range?"

    def response(self, body):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = body
        return response

    def test_default_configuration_targets_local_ollama_without_key(self):
        with patch.object(ollama_client, "urlopen") as transport:
            with patch.object(ollama_client, "llm_settings", wraps=config.llm_settings):
                # Smoke defaults are checked directly; actual endpoint calls use mocked transport below.
                self.assertEqual(config.llm_settings(), ("http://localhost:11434", "qwen2.5:3b", None))
        transport.assert_not_called()

    def test_missing_configuration_keeps_evidence_mode(self):
        answer = ollama_client.answer_from_evidence(self.question, self.results)
        self.assertEqual(answer, "Retrieved source evidence (no LLM configured):\n\n"
                         + source_label(self.chunk) + "\n" + self.chunk["content"])

    def test_success_sends_only_question_and_selected_evidence(self):
        with patch.object(ollama_client, "llm_settings", return_value=("http://localhost:11434", "qwen2.5:3b", None)):
            body = json.dumps({"choices": [{"message": {"content": "Grounded answer."}}]}).encode()
            with patch.object(ollama_client, "urlopen", return_value=self.response(body)) as transport:
                answer = ollama_client.answer_from_evidence(self.question, self.results)
        self.assertIn("Grounded answer.", answer)
        self.assertIn("Retrieved source labels", answer)
        request = transport.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/v1/chat/completions")
        self.assertIsNone(request.get_header("Authorization"))
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "qwen2.5:3b")
        self.assertEqual(payload["temperature"], 0)
        self.assertIn(self.question, payload["messages"][1]["content"])
        self.assertIn(source_label(self.chunk), payload["messages"][1]["content"])
        self.assertNotIn(self.chunk["source_text"], request.data.decode())
        self.assertIn("general knowledge", payload["messages"][0]["content"])
        self.assertIn("insufficient", payload["messages"][0]["content"])

    def test_legacy_configuration_is_preserved(self):
        with patch.dict(os.environ, {"RAG_LLM_URL": "http://legacy/v1", "RAG_LLM_MODEL": "legacy-model"}, clear=True):
            self.assertEqual(config.llm_settings(), ("http://legacy/v1", "legacy-model", None))

    def test_api_key_optional_and_remote_key_supported(self):
        body = b'{"choices": [{"message": {"content": "test answer"}}]}'
        with patch.object(ollama_client, "llm_settings", return_value=("http://localhost:11434", "qwen", None)):
            with patch.object(ollama_client, "urlopen", return_value=self.response(body)) as transport:
                answer = ollama_client.answer_from_evidence(self.question, self.results)
        self.assertIn("test answer", answer)
        self.assertIsNone(transport.call_args.args[0].get_header("Authorization"))
        with patch.object(ollama_client, "llm_settings", return_value=("https://api.example", "model", "secret")):
            with patch.object(ollama_client, "urlopen", return_value=self.response(body)) as transport:
                ollama_client.answer_from_evidence(self.question, self.results)
        self.assertEqual(transport.call_args.args[0].get_header("Authorization"), "Bearer secret")

    def test_transport_failures_fall_back(self):
        errors = [URLError("private detail"), TimeoutError("private detail"),
                  HTTPError("http://localhost", 401, "private detail", {}, None)]
        for error in errors:
            with self.subTest(error=type(error).__name__), patch.object(
                    ollama_client, "llm_settings", return_value=("http://localhost:11434", "qwen", None)):
                with patch.object(ollama_client, "urlopen", side_effect=error):
                    answer = ollama_client.answer_from_evidence(self.question, self.results)
            self.assertIn("LLM request failed", answer)
            self.assertIn(self.chunk["content"], answer)
            self.assertNotIn("private detail", answer)

    def test_bad_responses_fall_back(self):
        bodies = [b"not JSON", b"\xff", b"{}", b"null", b'{"choices": []}',
                  b'{"choices": [null]}', b'{"choices": [{"message": {"content": null}}]}',
                  b'{"choices": [{"message": {"content": " "}}]}']
        for body in bodies:
            with self.subTest(body=body), patch.object(
                    ollama_client, "llm_settings", return_value=("http://localhost:11434", "qwen", None)):
                with patch.object(ollama_client, "urlopen", return_value=self.response(body)):
                    answer = ollama_client.answer_from_evidence(self.question, self.results)
            self.assertIn("LLM request failed", answer)
            self.assertIn(self.chunk["content"], answer)

    def test_no_evidence_skips_model(self):
        with patch.object(ollama_client, "urlopen") as transport:
            answer = ollama_client.answer_from_evidence(self.question, [])
        self.assertEqual(answer, "I could not find supporting evidence in the indexed datasheet.")
        transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
