"""Convert Docling JSON and linked figures into retrieval-ready JSONL chunks."""

import json
from pathlib import Path

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

MAX_CHUNK_CHARS = 1200
TEXT_LABELS = {DocItemLabel.TEXT, DocItemLabel.LIST_ITEM, DocItemLabel.CAPTION, DocItemLabel.FOOTNOTE}
ROOT = Path(__file__).resolve().parents[3]


def _page_of(item) -> int | None:
    return item.prov[0].page_no if item.prov else None


def build_chunks(doc: DoclingDocument, document_id: str, component: str):
    chunks = []
    buffer_texts: list[str] = []
    buffer_page = None
    current_section = None

    def flush():
        nonlocal buffer_texts, buffer_page
        if not buffer_texts:
            return
        content = "\n".join(buffer_texts).strip()
        if content:
            chunks.append({
                "chunk_id": f"{document_id}-{len(chunks):05d}",
                "document_id": document_id,
                "component": component,
                "content_type": "text",
                "content": content,
                "section": current_section,
                "page": buffer_page,
                "source_text": content,
            })
        buffer_texts = []
        buffer_page = None

    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)
        if label == DocItemLabel.SECTION_HEADER:
            flush()
            current_section = item.text.strip()
        elif label == DocItemLabel.TABLE:
            flush()
            table_md = item.export_to_markdown(doc=doc)
            chunks.append({
                "chunk_id": f"{document_id}-{len(chunks):05d}",
                "document_id": document_id,
                "component": component,
                "content_type": "table",
                "content": table_md,
                "section": current_section,
                "page": _page_of(item),
                "source_text": table_md,
            })
        elif label in TEXT_LABELS:
            text = item.text.strip() if item.text else ""
            if not text:
                continue
            if buffer_page is None:
                buffer_page = _page_of(item)
            buffer_texts.append(text)
            if sum(map(len, buffer_texts)) >= MAX_CHUNK_CHARS:
                flush()
    flush()
    return chunks


def append_figure_chunks(chunks, document_id: str, component: str, figures_jsonl: Path) -> None:
    if not figures_jsonl.exists():
        return
    with figures_jsonl.open(encoding="utf-8") as source:
        for line in source:
            figure = json.loads(line)
            if figure.get("decorative_asset"):
                continue
            content = figure.get("content", "")
            chunks.append({
                "chunk_id": f"{document_id}-{len(chunks):05d}",
                "document_id": document_id,
                "component": component,
                "content_type": "figure",
                "content": content,
                "section": figure.get("section"),
                "page": figure.get("page"),
                "source_text": content,
                "image_path": figure.get("image_path"),
                "caption": figure.get("caption"),
                "extracted_text": figure.get("extracted_text", []),
                "bbox": figure.get("bbox"),
                "source_ref": figure.get("source_ref"),
                "description": figure.get("description"),
            })


def build_knowledge_base(input_json: Path, figures_jsonl: Path, output_jsonl: Path) -> int:
    doc = DoclingDocument.load_from_json(input_json)
    source_name = doc.origin.filename if doc.origin else input_json.stem
    document_id = Path(source_name).stem
    component = doc.name or document_id
    chunks = build_chunks(doc, document_id, component)
    append_figure_chunks(chunks, document_id, component, figures_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} knowledge chunks to {output_jsonl}")
    return len(chunks)
