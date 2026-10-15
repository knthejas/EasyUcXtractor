"""Registry lifecycle tests with a valid tiny PDF and isolated local storage."""

import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from easyucxtractor.library import DocumentLibrary, document_stage_lock


def pdf_bytes(text="test"):
    stream = f"BT /F1 12 Tf 40 700 Td ({text}) Tj ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"]
    content = b"%PDF-1.4\n"
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(content))
        content += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(content)
    content += b"xref\n0 6\n0000000000 65535 f \n"
    content += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    return content + f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()


def make_outputs(library, identifier, text="fixture-only evidence"):
    folder = library.root / identifier
    extracted, knowledge = folder / "extracted", folder / "knowledge_base"
    extracted.mkdir(exist_ok=True)
    knowledge.mkdir(exist_ok=True)
    for name in ("output.json", "output.md", "figures.jsonl"):
        (extracted / name).write_text("{}", encoding="utf-8")
    record = {"chunk_id": identifier, "document_id": identifier, "content": text,
              "page": 1, "section": "Fixture", "content_type": "text"}
    (knowledge / "knowledge_base.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (knowledge / "parameters.jsonl").write_text("", encoding="utf-8")
    (knowledge / "validation_report.json").write_text('{"status":"pass"}', encoding="utf-8")


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.library = DocumentLibrary(Path(self.temp.name))
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.library.executor.shutdown, wait=True)

    def drain(self):
        self.library.executor.submit(lambda: None).result(timeout=10)

    def test_invalid_uploads_are_rejected(self):
        for content, name in ((b"", "a.pdf"), (pdf_bytes(), "a.txt"),
                              (b"hello", "a.pdf"), (b"%PDF-corrupt", "a.pdf")):
            with self.subTest(name=name, content=content[:12]), self.assertRaises(ValueError):
                self.library.upload(content, name)
        self.assertEqual(self.library.list_documents(), [])

    def test_concurrent_duplicate_claim_and_ready_reuse(self):
        calls = []
        gate = threading.Event()
        def process(identifier):
            calls.append(identifier)
            gate.wait(5)
            make_outputs(self.library, identifier)
        self.library.processor = process
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(lambda _: self.library.upload(pdf_bytes(), "../../unsafe.pdf"), range(4)))
        self.assertEqual(sum(not reused for _, reused in rows), 1)
        gate.set()
        self.drain()
        row, reused = self.library.upload(pdf_bytes(), "renamed.pdf")
        self.assertTrue(reused)
        self.assertEqual(row["status"], "ready")
        self.assertEqual(row["filename"], "unsafe.pdf")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("outputs", row)
        knowledge, _ = self.library.knowledge_paths(row["id"])
        record = json.loads(knowledge.read_text(encoding="utf-8"))
        self.assertIn("unsafe.pdf", record["document_id"])
        self.assertTrue(record["chunk_id"].startswith(row["id"] + "-"))

    def test_failure_retry_and_same_filename_different_content(self):
        self.library.processor = lambda identifier: (_ for _ in ()).throw(RuntimeError("private path"))
        row, _ = self.library.upload(pdf_bytes(), "same.pdf")
        self.drain()
        self.assertEqual(self.library.get(row["id"])["status"], "failed")
        self.assertNotIn("private path", self.library.get(row["id"])["error"])
        self.library.processor = lambda identifier: make_outputs(self.library, identifier)
        self.library.retry(row["id"])
        self.drain()
        self.assertEqual(self.library.get(row["id"])["status"], "ready")
        other, _ = self.library.upload(pdf_bytes("other"), "same.pdf")
        self.drain()
        self.assertNotEqual(row["id"], other["id"])

    def test_persistence_and_interrupted_job_recovery(self):
        self.library.processor = lambda identifier: make_outputs(self.library, identifier)
        row, _ = self.library.upload(pdf_bytes(), "persist.pdf")
        self.drain()
        restarted = DocumentLibrary(self.library.root)
        self.addCleanup(restarted.executor.shutdown, wait=True)
        self.assertEqual(restarted.get(row["id"])["status"], "ready")
        self.library._update(row["id"], status="processing")
        restarted.recover_interrupted()
        self.assertEqual(restarted.get(row["id"])["stage"], "interrupted")

    def test_missing_or_failed_validation_never_ready(self):
        self.library.processor = lambda identifier: None
        row, _ = self.library.upload(pdf_bytes(), "missing.pdf")
        self.drain()
        self.assertEqual(self.library.get(row["id"])["status"], "failed")
        def fail_validation(identifier):
            make_outputs(self.library, identifier)
            report = self.library.root / identifier / "knowledge_base" / "validation_report.json"
            report.write_text('{"status":"fail"}', encoding="utf-8")
        self.library.processor = fail_validation
        self.library.retry(row["id"])
        self.drain()
        self.assertEqual(self.library.get(row["id"])["status"], "failed")

    def test_ready_reuse_detects_missing_outputs(self):
        self.library.processor = lambda identifier: make_outputs(self.library, identifier)
        row, _ = self.library.upload(pdf_bytes(), "saved.pdf")
        self.drain()
        (self.library.root / row["id"] / "knowledge_base" / "parameters.jsonl").unlink()
        reused_row, reused = self.library.upload(pdf_bytes(), "saved.pdf")
        self.assertTrue(reused)
        self.assertEqual(reused_row["status"], "failed")

    def test_only_one_server_can_own_library(self):
        owner = self.library.acquire_server_lock()
        try:
            with self.assertRaises(RuntimeError):
                self.library.acquire_server_lock()
        finally:
            owner.close()
        self.library.acquire_server_lock().close()

    def test_surviving_extraction_stage_prevents_overlapping_retry(self):
        with document_stage_lock(self.library.root):
            with self.assertRaises(RuntimeError):
                with document_stage_lock(self.library.root):
                    self.fail("Overlapping stage should not acquire the lock")

    def test_pipeline_calls_existing_stages_with_document_specific_paths(self):
        self.library.processor = lambda identifier: None
        row, _ = self.library.upload(pdf_bytes(), "stages.pdf")
        self.drain()
        with patch("easyucxtractor.library.subprocess.run") as run:
            self.library._process(row["id"])
        self.assertEqual(run.call_count, 5)
        folder = self.library.root / row["id"]
        stages = []
        for call in run.call_args_list:
            arguments = call.args[0]
            stages.append(arguments[arguments.index("--_stage") + 1])
            self.assertEqual(arguments[arguments.index("--_pdf") + 1], str(folder / "original.pdf"))
            self.assertEqual(arguments[arguments.index("--_extracted") + 1], str(folder / "extracted"))
            self.assertEqual(arguments[arguments.index("--_knowledge") + 1], str(folder / "knowledge_base"))
            self.assertTrue(call.kwargs["check"])
        self.assertEqual(stages, ["extract", "figures", "knowledge", "parameters", "validate"])

    def test_development_validation_failure_has_actionable_error(self):
        def process(identifier):
            make_outputs(self.library, identifier)
            self.library._update(identifier, stage="validate")
            report = self.library.root / identifier / "knowledge_base" / "validation_report.json"
            report.write_text(json.dumps({"status": "fail", "errors": [
                "known-fact check failed: junction temperature value not found"
            ]}), encoding="utf-8")
            raise ValueError("private filesystem path")
        self.library.processor = process
        row, _ = self.library.upload(pdf_bytes(), "valid-but-different.pdf")
        self.drain()
        self.assertIn("development validator", self.library.get(row["id"])["error"])
        self.assertNotIn("private filesystem path", self.library.get(row["id"])["error"])


if __name__ == "__main__":
    unittest.main()