"""Local content-addressed document registry and background extraction jobs."""

import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pypdfium2 as pdfium

from easyucxtractor.paths import ROOT

MAX_PDF_BYTES = 50 * 1024 * 1024
STAGES = ("extract", "figures", "knowledge", "parameters", "validate")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def document_stage_lock(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / "stage.lock").open("a+b")
    if handle.seek(0, 2) == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("A previous stage is still processing this PDF. Retry after it exits.") from None
    try:
        yield
    finally:
        handle.close()


class DocumentLibrary:
    def __init__(self, root: Path = ROOT / "data" / "library", processor=None):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "registry.sqlite3"
        self.processor = processor or self._process
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="datasheet")
        self.pdf_lock = threading.Lock()
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, size INTEGER NOT NULL,
                pages INTEGER NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL,
                progress INTEGER NOT NULL, error TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, outputs TEXT NOT NULL DEFAULT '{}'
            )""")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def recover_interrupted(self):
        with self._connect() as connection:
            connection.execute(
                "UPDATE documents SET status='failed', stage='interrupted', error=?, updated_at=? "
                "WHERE status='processing'",
                ("Processing was interrupted by a server restart. Retry this document.", timestamp()),
            )

    def acquire_server_lock(self):
        handle = (self.root / "server.lock").open("a+b")
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RuntimeError("Another UI server is using this library. Stop it before starting a second server.") from None
        return handle

    def get(self, identifier: str) -> dict:
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{64}", identifier):
            raise ValueError("Unknown library document. Select a document from the library.")
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("Unknown library document. Select a document from the library.")
        return dict(row)

    def public(self, row: dict) -> dict:
        return {key: row[key] for key in (
            "id", "filename", "size", "pages", "status", "stage", "progress", "error",
            "created_at", "updated_at",
        )}

    def list_documents(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [self.public(dict(row)) for row in rows]

    def upload(self, content: bytes, filename: str) -> tuple[dict, bool]:
        if not content or len(content) > MAX_PDF_BYTES:
            raise ValueError("Choose a PDF between 1 byte and 50 MB.")
        filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        filename = re.sub(r"[\x00-\x1f\x7f]", "", filename).strip()[:180]
        if not filename.lower().endswith(".pdf") or not content.startswith(b"%PDF-"):
            raise ValueError("Only PDF files are accepted. Choose a valid .pdf document.")
        try:
            with self.pdf_lock:
                pdf = pdfium.PdfDocument(content)
                try:
                    pages = len(pdf)
                    if pages < 1:
                        raise ValueError("Empty PDF")
                    page = pdf[0]
                    try:
                        page.get_size()
                    finally:
                        page.close()
                finally:
                    pdf.close()
        except Exception:
            raise ValueError("The PDF could not be opened. Use a valid, unencrypted PDF.") from None
        identifier = hashlib.sha256(content).hexdigest()
        folder = self.root / identifier
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT * FROM documents WHERE id=?", (identifier,)).fetchone()
            if existing:
                if existing["status"] == "ready" and not all(
                    (self.root / path).is_file() for path in json.loads(existing["outputs"]).values()
                ):
                    connection.execute("UPDATE documents SET status='failed',error=?,updated_at=? WHERE id=?",
                                       ("Saved outputs are missing. Retry this document.", timestamp(), identifier))
                    existing = connection.execute("SELECT * FROM documents WHERE id=?", (identifier,)).fetchone()
                return self.public(dict(existing)), True
            folder.mkdir(exist_ok=True)
            (folder / "original.pdf").write_bytes(content)
            now = timestamp()
            connection.execute(
                "INSERT INTO documents (id,filename,size,pages,status,stage,progress,created_at,updated_at) "
                "VALUES (?,?,?,?, 'processing','queued',0,?,?)",
                (identifier, filename, len(content), pages, now, now),
            )
        self.executor.submit(self._run, identifier)
        return self.public(self.get(identifier)), False

    def retry(self, identifier: str) -> dict:
        self.get(identifier)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            claimed = connection.execute(
                "UPDATE documents SET status='processing',stage='queued',progress=0,error=NULL,updated_at=? "
                "WHERE id=? AND status='failed'", (timestamp(), identifier),
            ).rowcount
        if claimed:
            self.executor.submit(self._run, identifier)
        return self.public(self.get(identifier))

    def _update(self, identifier, **fields):
        fields["updated_at"] = timestamp()
        with self._connect() as connection:
            connection.execute(
                "UPDATE documents SET " + ",".join(f"{key}=?" for key in fields) + " WHERE id=?",
                [*fields.values(), identifier],
            )

    def _run(self, identifier):
        try:
            self.processor(identifier)
            folder = self.root / identifier
            extracted, knowledge = folder / "extracted", folder / "knowledge_base"
            required = [extracted / "output.json", extracted / "output.md", extracted / "figures.jsonl",
                        knowledge / "knowledge_base.jsonl", knowledge / "parameters.jsonl",
                        knowledge / "validation_report.json"]
            if not all(path.is_file() for path in required):
                raise ValueError("Missing pipeline outputs")
            report = json.loads(required[-1].read_text(encoding="utf-8"))
            if report.get("status") not in {"pass", "pass_with_warnings"}:
                raise ValueError("Validation did not succeed")
            if not required[3].read_text(encoding="utf-8").strip():
                raise ValueError("Empty knowledge base")
            self.identify_records(identifier)
            self._update(identifier, status="ready", stage="complete", progress=100, error=None,
                         outputs=json.dumps({path.name: path.relative_to(self.root).as_posix()
                                             for path in [folder / "original.pdf", *required]}))
        except Exception as error:
            print(f"Library processing failed: {type(error).__name__}")
            message = (
                "Extraction or validation failed. Check Docling model availability, proxy access, "
                "and the PDF contents, then retry. See the local processing log for details."
            )
            if self.get(identifier)["stage"] == "validate":
                report_path = self.root / identifier / "knowledge_base" / "validation_report.json"
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    if any(str(item).startswith("known-fact check failed:") for item in report.get("errors", [])):
                        message = (
                            "Validation failed: the existing development validator requires a junction "
                            "temperature of -40 to 150 degrees Celsius. This PDF does not match that "
                            "check. Review its validation report; retrying unchanged data will not fix it."
                        )
                except (OSError, ValueError):
                    pass
            self._update(identifier, status="failed", stage="failed", error=message)

    def identify_records(self, identifier):
        row = self.get(identifier)
        folder = self.root / identifier / "knowledge_base"
        label = f"{row['filename']} [{identifier[:12]}]"
        for name in ("knowledge_base.jsonl", "parameters.jsonl"):
            path = folder / name
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for record in records:
                record["document_id"] = label
                for key in ("chunk_id", "source_chunk_id"):
                    if record.get(key) and not record[key].startswith(identifier + "-"):
                        record[key] = f"{identifier}-{record[key]}"
            temporary = path.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n"
                                         for record in records), encoding="utf-8")
            temporary.replace(path)

    def _process(self, identifier):
        folder = self.root / identifier
        extracted, knowledge = folder / "extracted", folder / "knowledge_base"
        with (folder / "processing.log").open("w", encoding="utf-8") as log:
            for index, stage in enumerate(STAGES):
                self._update(identifier, stage=stage, progress=index * 20)
                subprocess.run([
                    sys.executable, "-u", str(ROOT / "scripts" / "extract_datasheet.py"),
                    "--_stage", stage, "--_pdf", str(folder / "original.pdf"),
                    "--_extracted", str(extracted), "--_knowledge", str(knowledge),
                ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    def knowledge_paths(self, identifier: str) -> tuple[Path, Path]:
        row = self.get(identifier)
        if row["status"] != "ready":
            raise ValueError("This datasheet is not ready. Wait for processing or retry a failed document.")
        folder = self.root / identifier / "knowledge_base"
        knowledge, parameters = folder / "knowledge_base.jsonl", folder / "parameters.jsonl"
        if not knowledge.is_file() or not parameters.is_file():
            self._update(identifier, status="failed", error="Saved outputs are missing. Retry this document.")
            raise ValueError("Saved outputs are missing. Retry this document.")
        return knowledge, parameters