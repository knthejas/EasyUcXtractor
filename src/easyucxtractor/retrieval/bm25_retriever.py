"""Existing lexical BM25-style retrieval over text, table, and parameter chunks."""

import json
import math
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*", re.IGNORECASE)
QUERY_STOP_WORDS = {"a", "an", "and", "can", "does", "for", "ic", "is", "it", "of", "on", "the", "to", "what", "work", "with"}
QUERY_TERM_ALIASES = {"maximum": "max", "minimum": "min", "nominal": "typ", "typical": "typ"}


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def query_tokens(text: str) -> list[str]:
    return [QUERY_TERM_ALIASES.get(token, token) for token in tokens(text)
            if QUERY_TERM_ALIASES.get(token, token) not in QUERY_STOP_WORDS]


def load_chunks(knowledge_path: Path, parameters_path: Path | None = None):
    with knowledge_path.open(encoding="utf-8") as source:
        chunks = [json.loads(line) for line in source]
    if parameters_path and parameters_path.exists():
        with parameters_path.open(encoding="utf-8") as source:
            for line in source:
                parameter = json.loads(line)
                values = ", ".join(f"{key}={parameter[key]}" for key in ("min", "typ", "max")
                                    if parameter.get(key) is not None)
                content = (f"{parameter.get('parameter', '')} ({parameter.get('symbol', '')}): "
                           f"{values} {parameter.get('unit', '')}. "
                           f"Condition: {parameter.get('condition') or 'not specified'}. "
                           f"Specification: {parameter.get('spec_number') or 'not specified'}. "
                           f"Parameter category: {parameter.get('parameter_key') or ''}.")
                chunks.append({
                    "chunk_id": f"parameter-{parameter.get('source_chunk_id')}-{parameter.get('spec_number')}",
                    "document_id": parameter.get("document_id"), "component": parameter.get("component"),
                    "content_type": "parameter", "content": content, "section": parameter.get("section"),
                    "page": parameter.get("page"), "source_text": parameter.get("source_text"),
                    "spec_number": parameter.get("spec_number"), "parameter_key": parameter.get("parameter_key"),
                })
    return chunks


def retrieve(query: str, chunks: list[dict], limit: int = 5):
    query_terms = query_tokens(query)
    if not query_terms:
        return []
    documents = [tokens(chunk.get("content", "")) for chunk in chunks]
    document_frequency = Counter(term for doc_tokens in documents for term in set(doc_tokens))
    average_length = sum(map(len, documents)) / max(len(documents), 1)
    total_documents = len(documents)
    scored = []
    query_bigrams = list(zip(query_terms, query_terms[1:]))
    for chunk, doc_tokens in zip(chunks, documents):
        if not doc_tokens:
            continue
        frequencies = Counter(doc_tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * len(doc_tokens) / max(average_length, 1))
            score += inverse_frequency * frequency * 2.2 / denominator
        document_text = " ".join(doc_tokens)
        for first, second in query_bigrams:
            if f"{first} {second}" in document_text:
                score += 3.0
        if chunk.get("content_type") == "parameter":
            score *= 1.25
        if score:
            scored.append((score, chunk))
    return sorted(scored, key=lambda item: item[0], reverse=True)[:limit]


def source_label(chunk: dict) -> str:
    suffix = f" | {chunk['spec_number']}" if chunk.get("spec_number") else ""
    return f"[{chunk.get('document_id')} | p. {chunk.get('page')} | {chunk.get('section')}{suffix} | {chunk.get('chunk_id')}]"
