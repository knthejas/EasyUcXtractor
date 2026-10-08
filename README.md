# EasyUcXtractor

**Datasheets In. Engineering Decisions Out.**

EasyUcXtractor is a local-first prototype for extracting text, tables, figure metadata, and supported numeric parameter rows from semiconductor datasheets, then retrieving relevant evidence for a question and optionally asking a local Ollama model to summarize it.

Datasheets are intentionally **not stored in this Git repository**. Vendor datasheets may be copyrighted or subject to distribution restrictions. Each team member must obtain files through an authorized source and keep them locally in `data/pdfs/`.

## Problem

Engineering information is spread across long, table-heavy datasheets. This project prepares extracted content for traceable search and a grounded question-answering demo. It provides source page/section/chunk labels so users can check retrieved claims against the original PDF. It is a prototype, not an engineering sign-off tool.

## Current Architecture

```text
PDF -> Docling extraction -> text/tables/figure text + structured parameter rows
    -> JSONL searchable store -> BM25-style lexical retrieval -> Ollama LLM -> answer with source labels
```

See [docs/architecture.md](docs/architecture.md) for module-by-module responsibilities and data boundaries.

## Repository Structure

```text
EasyUcXtractor/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── src/easyucxtractor/
│   ├── __init__.py
│   ├── config.py
│   ├── paths.py
│   ├── validation.py
│   ├── extraction/
│   │   ├── document_postprocess.py
│   │   ├── figure_extractor.py
│   │   ├── knowledge_builder.py
│   │   └── pdf_extractor.py
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   └── parameter_builder.py
│   └── llm/
│       ├── __init__.py
│       └── ollama_client.py
├── scripts/
│   ├── _bootstrap.py
│   ├── extract_datasheet.py
│   └── run_rag.py
├── tests/
│   ├── __init__.py
│   ├── test_paths.py
│   └── test_rag_demo.py
├── data/
│   ├── pdfs/.gitkeep
│   ├── extracted/.gitkeep
│   └── knowledge_base/.gitkeep
└── docs/architecture.md
```

Generated files under `data/extracted/` and `data/knowledge_base/` and all local PDFs are ignored by Git. `.gitkeep` files preserve the empty input/output directories in a clone.

## Installation

Requirements: Python 3.11, Git, and (for generated answers) [Ollama](https://ollama.com/) with a model available locally. CPU inference works but may be slower than GPU inference.

From a clone, in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`; use `python3` in place of `py -3.11` if needed. The scripts bootstrap the `src/` package path and resolve project data paths from their own location, so they can be launched from outside the repository root.

## Add a Datasheet

Place an authorized local PDF in `data/pdfs/`, for example `data/pdfs/example.pdf`. Do not commit the PDF. You can also pass a PDF path elsewhere on disk directly to the extraction CLI; outputs are still written into this repository's data directories.

## Extract Datasheets

Discover and process all PDFs under `data/pdfs/`:

```powershell
python scripts/extract_datasheet.py
```

Or process an explicit PDF:

```powershell
python scripts/extract_datasheet.py data/pdfs/example.pdf
```

The CLI runs Docling extraction, figure cropping, knowledge chunking, supported parameter parsing, and structural validation. Extraction uses native PDF text, accurate TableFormer table structure, and the existing header/ToC cleanup. It may take several minutes per PDF. Files are stored separately per datasheet using a normalized filename plus a short content hash:

```text
data/extracted/<datasheet-id>/output.json
data/extracted/<datasheet-id>/output.md
data/extracted/<datasheet-id>/figures.jsonl
data/extracted/<datasheet-id>/figures/*.png

data/knowledge_base/<datasheet-id>/knowledge_base.jsonl
data/knowledge_base/<datasheet-id>/parameters.jsonl
data/knowledge_base/<datasheet-id>/validation_report.json
```

The ID helps prevent different PDFs/revisions from overwriting each other's outputs. Re-running extraction for the same PDF content reuses that ID and regenerates its files.

If Docling model downloads are blocked but `output.json` already exists for this PDF, rebuild only the downstream artifacts with:

```powershell
python scripts/extract_datasheet.py --reuse-extraction
```

This reuses the existing Docling JSON, recrops figures, rebuilds the knowledge and parameter JSONL files, and validates them. It does not perform a fresh PDF-to-Docling conversion.

## Ask Questions with RAG

Ollama must be running and the configured model must be installed. Defaults are `http://localhost:11434` and `qwen2.5:3b`; no API key is required. For a fresh Ollama setup:

```powershell
ollama pull qwen2.5:3b
```

Configure the current PowerShell session and ask a question:

```powershell
$env:OLLAMA_URL = "http://localhost:11434"
$env:OLLAMA_MODEL = "qwen2.5:3b"
$env:NO_PROXY = "localhost,127.0.0.1,::1"
python scripts/run_rag.py "What is the junction temperature range?" --show-evidence
```

When exactly one processed datasheet is present, it is selected automatically. With multiple datasheets, select the printed datasheet ID (or a unique prefix):

```powershell
python scripts/run_rag.py --datasheet example-abc1234567 "What is the operating supply voltage?" --show-evidence
```

The CLI displays selected evidence on request and the answer. It uses the existing lexical BM25-style retriever. The model sees only the question and top retrieved evidence with source labels, not the entire PDF or full knowledge base. It is instructed to answer only from that evidence, preserve units and rating context, and report insufficient evidence rather than guess. These instructions cannot guarantee correct answers; inspect citations and source excerpts.

Legacy variables `RAG_LLM_URL`, `RAG_LLM_MODEL`, and `RAG_LLM_API_KEY` remain supported and take precedence over `OLLAMA_URL`/`OLLAMA_MODEL`. Ollama does not require a key. `.env.example` documents safe local defaults but is not automatically loaded. Do not create or commit a real `.env` file or secrets.

## Tests

Run the complete test suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```

Tests mock HTTP/model responses; they do not make network calls or certify answer quality. For a live smoke test, use `python scripts/run_rag.py "What is the junction temperature range?" --show-evidence` with Ollama running.

## Current Capabilities

- Extracts text and structured table grids from local PDFs using Docling.
- Exports Docling JSON and Markdown for inspection.
- Crops images with page/bounding-box provenance and uses linked captions/recognized text where available.
- Builds text, table, and eligible figure JSONL chunks plus separate structured parameter JSONL.
- Validates artifact structure, provenance links, figure files, and one reviewed known parameter value.
- Retrieves evidence with the existing BM25-style lexical ranker.
- Sends the question and retrieved evidence to local Ollama, then falls back to evidence if the model is unavailable or returns an invalid response.

## Current Limitations

- Lexical retrieval can miss paraphrases or rank a related but wrong measurement. A Qwen 2.5 3B smoke test confused a 20.7 MHz bridge-driver frequency with maximum PWM frequency and asserted Ethernet was unsupported when retrieved excerpts were silent.
- A CAN FD response conflated a 2 Mbps data-phase example with 5 MBit/s CAN FD tolerance/glitch-filter evidence. Distinguish protocol examples, feature support, and tolerance specifications.
- The model sometimes omits citations; the application appends ranked source labels, which preserve provenance but are not proof of relevance.
- Parameter extraction supports recognized table layouts only. Symbolic expressions remain unevaluated; figures without text are not visually interpreted.
- Structural validation is not extraction accuracy, retrieval precision/recall, or answer correctness. Verify engineering-critical claims against the original datasheet.
- A full PDF extraction needs Docling model files available locally or network access to fetch them; corporate proxy restrictions can block the first run. `--reuse-extraction` only helps when a suitable `output.json` already exists.

## Roadmap

Potential later work, not implemented: a manually reviewed multi-datasheet benchmark, improved retrieval evaluation, embeddings/vector search, richer figure understanding, requirement extraction, and deterministic feasibility analysis. These capabilities are not part of the current project.

## Team Workflow

1. Clone the repository and install `requirements.txt` in a local virtual environment.
2. Install/run Ollama and pull `qwen2.5:3b` if using generated answers.
3. Place authorized PDFs in `data/pdfs/`; keep PDFs, generated artifacts, virtual environments, and credentials out of Git.
4. Run `python scripts/extract_datasheet.py` and inspect the validation report and source pages.
5. Ask questions with `python scripts/run_rag.py "..." --show-evidence`; review both retrieved text and model response.
6. Add or update tests when changing extraction, retrieval, or the model request boundary.
7. Before sharing changes, run the test suite and check `git status --short` to ensure only source, tests, and documentation are staged.

Use branches and code review according to the team's Git workflow. This setup does not create a GitHub repository, configure a remote, commit, or push.
