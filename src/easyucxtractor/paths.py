"""Repository-relative paths and per-datasheet output locations."""

from pathlib import Path
import hashlib
import re

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "data" / "pdfs"
EXTRACTED_DIR = ROOT / "data" / "extracted"
KNOWLEDGE_DIR = ROOT / "data" / "knowledge_base"


def datasheet_id(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_path.stem).strip("._-") or "datasheet"
    return f"{slug}-{digest.hexdigest()[:10]}"


def output_directories(pdf_path: Path) -> tuple[Path, Path, str]:
    identifier = datasheet_id(pdf_path)
    return EXTRACTED_DIR / identifier, KNOWLEDGE_DIR / identifier, identifier
