"""Checks for the local web API adapter around the existing RAG pipeline."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import serve_app


class WebAppTests(unittest.TestCase):
    def test_dataset_listing_uses_generated_knowledge_bases(self):
        datasets = serve_app.list_datasets()
        self.assertTrue(datasets)
        self.assertTrue(all(item["chunk_count"] > 0 for item in datasets))
        self.assertTrue(all((serve_app.KNOWLEDGE_DIR / item["id"] / "knowledge_base.jsonl").is_file()
                            for item in datasets))

    def test_question_api_reuses_retrieval_and_returns_labeled_evidence(self):
        dataset = serve_app.list_datasets()[0]
        with patch.object(serve_app, "answer_from_evidence", return_value="Example answer") as answer:
            result = serve_app.ask_question("What is the junction temperature range?", dataset["id"])
        self.assertEqual(result["dataset_id"], dataset["id"])
        self.assertEqual(result["answer"], "Example answer")
        self.assertTrue(result["evidence"])
        self.assertTrue(all(entry["source"].startswith("[") for entry in result["evidence"]))
        self.assertTrue(any("Junction Temperature" in entry["content"] for entry in result["evidence"]))
        answer.assert_called_once()

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
