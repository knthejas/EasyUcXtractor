"""Docling PDF extraction using the existing accurate table configuration."""

import json
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

from easyucxtractor.extraction.document_postprocess import clean_document


def extract_pdf(pdf_path: Path, output_dir: Path) -> tuple[Path, Path]:
    pdf_path = pdf_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True, force_backend_text=True)
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    options.table_structure_options.do_cell_matching = True
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    })

    print(f"Processing: {pdf_path}")
    result = converter.convert(str(pdf_path))
    clean_document(result.document)

    markdown_path = output_dir / "output.md"
    markdown_path.write_text(result.document.export_to_markdown(), encoding="utf-8")
    json_path = output_dir / "output.json"
    json_path.write_text(
        json.dumps(result.document.export_to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Created: {markdown_path}")
    print(f"Created: {json_path}")
    return json_path, markdown_path
