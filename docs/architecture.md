# Current Architecture

The current implementation turns local datasheet PDFs into searchable text/table/parameter records and uses lexical retrieval to supply evidence to an LLM. It does not implement future roadmap components such as embeddings, a vector database, requirement extraction, or feasibility analysis.

```text
PDF
↓
Document ingestion
↓
┌──────────────┬──────────────┬─────────────────┐
↓              ↓              ↓
Text          Tables       Structured parameters
└──────────────┬──────────────┴─────────────────┘
↓
Searchable store
↓
Retrieval
↓
LLM
↓
Engineering answer
```

## Module Responsibilities

- `scripts/extract_datasheet.py` is the command-line orchestrator. It discovers local PDFs or accepts explicit paths, assigns a content-hash-based datasheet output ID, then runs extraction and artifact generation.
- `src/easyucxtractor/extraction/pdf_extractor.py` runs Docling with native PDF text and accurate TableFormer table extraction. It writes `output.json` and `output.md` under `data/extracted/<datasheet-id>/`.
- `src/easyucxtractor/extraction/document_postprocess.py` removes repeated running headers and table-of-contents dot leaders before export.
- `src/easyucxtractor/extraction/figure_extractor.py` crops pictures using Docling provenance and stores figure metadata and PNGs alongside the extracted document. Small repeated decorative assets remain recorded but are excluded from retrieval chunks.
- `src/easyucxtractor/extraction/knowledge_builder.py` flattens the Docling document graph into per-record text/table chunks and adds non-decorative figure metadata. It writes `knowledge_base.jsonl` under `data/knowledge_base/<datasheet-id>/`.
- `src/easyucxtractor/retrieval/parameter_builder.py` parses supported table rows into `parameters.jsonl`, preserving unsupported expressions as text. Text chunks, table chunks, and structured parameter rows are the indexed/searchable records.
- `src/easyucxtractor/retrieval/bm25_retriever.py` loads text/table/figure chunks plus parsed parameter rows and applies the existing BM25-style lexical ranking. It is not an embedding or vector search implementation.
- `src/easyucxtractor/llm/ollama_client.py` formats the question and retrieved evidence with source labels, sends them to an OpenAI-compatible chat-completions endpoint such as Ollama, and falls back to the retrieved evidence if the request fails. It asks for evidence-only answers and appends exact ranked labels if the model omits citations; labels still require relevance checking.
- `src/easyucxtractor/config.py` centralizes local Ollama defaults and support for the prior `RAG_LLM_*` environment variables.
- `src/easyucxtractor/validation.py` checks structural integrity and one reviewed known fact. A pass is not an accuracy score.
- `scripts/run_rag.py` selects a generated dataset and runs retrieval/answer generation. `--show-evidence` displays the retrieved excerpts as well as the model response.

## Data Boundaries

Datasheet PDFs are local inputs under `data/pdfs/` and are intentionally excluded from Git because vendor datasheets may be copyrighted or subject to distribution restrictions. Generated extraction and knowledge-base artifacts are also ignored by default. Each PDF has an output directory named from a normalized filename and a short content hash, reducing the risk that a different datasheet or revision overwrites unrelated results.

The LLM receives only the user's question and the selected evidence records (including their existing source labels). It does not receive the full PDF, `output.json`, the entire knowledge base, or unselected records. Local Ollama inference uses `http://localhost:11434` by default and does not need an API key.

## Limitations

- Retrieval is lexical and may rank a related but incorrect frequency/bitrate passage above the needed fact.
- The small Qwen model has demonstrated answer-grounding failures, including confusing PWM frequency with bridge-driver frequency and claiming a feature is absent when the retrieved excerpts are merely silent.
- The validator checks format, provenance links, files, and a known fact; it does not certify extraction accuracy.
- Some figures have no searchable caption/text, and structured parameter parsing supports only recognized table layouts.
- Model citations, values, conditions, and abstentions require review against the original PDF.
