"""Ask a question against one generated datasheet knowledge base."""

import argparse
import sys
from pathlib import Path

from _bootstrap import ROOT
sys.path.insert(0, str(ROOT / "src"))

from easyucxtractor.llm.ollama_client import answer_from_evidence
from easyucxtractor.paths import KNOWLEDGE_DIR
from easyucxtractor.retrieval.bm25_retriever import load_chunks, retrieve, source_label


def find_dataset(selector: str | None) -> Path:
    candidates = sorted(path for path in KNOWLEDGE_DIR.glob("*") if (path / "knowledge_base.jsonl").is_file())
    if selector:
        matches = [path for path in candidates if path.name == selector or path.name.startswith(selector)]
        if len(matches) != 1:
            raise SystemExit(f"Dataset selector {selector!r} matched {len(matches)} datasets. Available: "
                             + ", ".join(path.name for path in candidates))
        return matches[0]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"No knowledge bases found under {KNOWLEDGE_DIR}. Run extract_datasheet.py first.")
    raise SystemExit("Multiple datasheets found. Use --datasheet with one of: "
                     + ", ".join(path.name for path in candidates))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieve evidence and ask the configured Ollama model.")
    parser.add_argument("question", nargs="*", help="Question to ask; omitted prompts interactively")
    parser.add_argument("--datasheet", help="Generated dataset ID (or unique ID prefix) when multiple exist")
    parser.add_argument("--show-evidence", action="store_true", help="Print retrieved excerpts before the answer")
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip() or input("Ask a datasheet question: ").strip()
    dataset = find_dataset(args.datasheet)
    chunks = load_chunks(dataset / "knowledge_base.jsonl", dataset / "parameters.jsonl")
    results = retrieve(question, chunks)
    print(f"Question: {question}\nDatasheet: {dataset.name}\n")
    if args.show_evidence:
        print("Retrieved evidence:")
        for _, chunk in results:
            print(f"{source_label(chunk)}\n{chunk.get('content', '')[:1800]}\n")
    print(answer_from_evidence(question, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
