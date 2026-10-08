"""Extract one or more local PDFs into isolated generated-data directories."""

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import ROOT
sys.path.insert(0, str(ROOT / "src"))

from easyucxtractor.paths import PDF_DIR, output_directories


def resolve_pdf(argument: str) -> Path:
    path = Path(argument).expanduser()
    if not path.is_absolute():
        from_cwd = path.resolve()
        path = from_cwd if from_cwd.exists() else (ROOT / path).resolve()
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF does not exist or is not a PDF: {path}")
    return path


def run_stage(stage: str, pdf_path: Path) -> None:
    extracted_dir, knowledge_dir, _identifier = output_directories(pdf_path)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    if stage == "extract":
        from easyucxtractor.extraction.pdf_extractor import extract_pdf
        extract_pdf(pdf_path, extracted_dir)
    elif stage == "figures":
        from easyucxtractor.extraction.figure_extractor import extract_figures
        extract_figures(pdf_path, extracted_dir / "output.json", extracted_dir)
    elif stage == "knowledge":
        from easyucxtractor.extraction.knowledge_builder import build_knowledge_base
        build_knowledge_base(extracted_dir / "output.json", extracted_dir / "figures.jsonl",
                             knowledge_dir / "knowledge_base.jsonl")
    elif stage == "parameters":
        from easyucxtractor.retrieval.parameter_builder import build_parameters
        build_parameters(knowledge_dir / "knowledge_base.jsonl", knowledge_dir / "parameters.jsonl")
    elif stage == "validate":
        from easyucxtractor.validation import validate_artifacts
        report = validate_artifacts(
            extracted_dir / "output.json",
            knowledge_dir / "knowledge_base.jsonl",
            knowledge_dir / "parameters.jsonl",
            extracted_dir / "figures.jsonl",
            knowledge_dir / "validation_report.json",
        )
        if report["status"] == "fail":
            raise RuntimeError(f"Validation failed for {pdf_path}")


def process_pdf(pdf_path: Path, reuse_extraction: bool = False) -> None:
    extracted_dir, knowledge_dir, identifier = output_directories(pdf_path)
    stages = ("figures", "knowledge", "parameters", "validate") if reuse_extraction else (
        "extract", "figures", "knowledge", "parameters", "validate"
    )
    for stage in stages:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--_stage", stage,
             "--_pdf", str(pdf_path)],
            cwd=ROOT,
            check=True,
        )
    print(f"Datasheet ID: {identifier}")
    print(f"Extracted artifacts: {extracted_dir}")
    print(f"Knowledge artifacts: {knowledge_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract PDFs and build datasheet RAG records.")
    parser.add_argument("pdfs", nargs="*", help="PDF paths; omitted means discover data/pdfs/*.pdf")
    parser.add_argument("--reuse-extraction", action="store_true",
                        help="Reuse existing output.json for this PDF and rebuild downstream artifacts")
    parser.add_argument("--_stage", choices=("extract", "figures", "knowledge", "parameters", "validate"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--_pdf", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._stage:
        if not args._pdf:
            parser.error("Internal stage requires a PDF path")
        run_stage(args._stage, resolve_pdf(args._pdf))
        return 0
    if args.pdfs:
        pdf_paths = [resolve_pdf(arg) for arg in args.pdfs]
    else:
        pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        parser.error(f"No PDF files found. Add a PDF to {PDF_DIR} or pass a PDF path.")
    for pdf_path in pdf_paths:
        process_pdf(pdf_path, reuse_extraction=args.reuse_extraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
