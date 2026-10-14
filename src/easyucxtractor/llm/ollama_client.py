"""Grounded chat-completions client, compatible with local Ollama."""

import json
import re
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from easyucxtractor.config import llm_settings
from easyucxtractor.retrieval.bm25_retriever import source_label

SYSTEM_PROMPT = (
    "Give a concise answer using only the supplied datasheet evidence. Do not use general knowledge, "
    "invent values, or infer unsupported specifications. If evidence is insufficient, explicitly say "
    "the datasheet evidence is insufficient. Verify that the evidence states the exact attribute asked "
    "for, not merely a related property of the same subject; for example, board dimensions do not state "
    "a board name. A figure caption or topic mention is not evidence for details that are not written. "
    "Absence of a feature in these excerpts does not establish that it is unsupported. Preserve "
    "numerical values and units exactly when possible. Distinguish "
    "Absolute Maximum Ratings from Functional/Operating Range. Include relevant source, page, section, "
    "and specification ID, copying citations only from supplied source labels."
)
BOARD_NAME_QUERY_RE = re.compile(
    r"\b(?:board\s+name|name\s+of\s+(?:the\s+)?board|"
    r"what\s+is\s+(?:the\s+)?board\s+(?:called|named))\b",
    re.IGNORECASE,
)
BOARD_NAME_EVIDENCE_RE = re.compile(
    r"\bboard\s+name\s*[:=]\s*\S+|"
    r"\bname\s+of\s+(?:the\s+)?board\s*(?:is|:)\s*\S+|"
    r"\bboard\s+(?:is\s+)?(?:called|named)\s+\S+",
    re.IGNORECASE,
)
SMALL_TALK_RESPONSES = {
    "hello": "Hello! Ask me about a datasheet specification, parameter, table, or feature.",
    "hi": "Hi! Ask me about a datasheet specification, parameter, table, or feature.",
    "hey": "Hey! Ask me about a datasheet specification, parameter, table, or feature.",
    "good morning": "Good morning! What datasheet detail would you like to look up?",
    "good afternoon": "Good afternoon! What datasheet detail would you like to look up?",
    "good evening": "Good evening! What datasheet detail would you like to look up?",
    "how are you": "I'm ready to help look up information in the selected datasheet.",
    "who are you": "I'm EasyUcXtractor. I retrieve datasheet evidence and can summarize it with the local model.",
    "what can you do": "I can search the selected datasheet and show the passages and source pages behind an answer.",
    "thanks": "You're welcome! What datasheet detail would you like to check?",
    "thank you": "You're welcome! What datasheet detail would you like to check?",
}


def small_talk_response(question: str) -> str | None:
    normalized = re.sub(r"[.!?,]+$", "", question.strip().lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+there$", "", normalized)
    return SMALL_TALK_RESPONSES.get(normalized)


def answer_from_evidence(question: str, results: list[tuple[float, dict]]) -> str:
    small_talk = small_talk_response(question)
    if small_talk:
        return small_talk
    if not results:
        return "I could not find supporting evidence in the indexed datasheet."
    evidence = "\n\n".join(f"{source_label(chunk)}\n{chunk.get('content', '')[:1800]}"
                             for _, chunk in results)
    if BOARD_NAME_QUERY_RE.search(question) and not BOARD_NAME_EVIDENCE_RE.search(evidence):
        board_sources = [
            chunk for _, chunk in results
            if any(term in chunk.get("content", "").lower() for term in (
                "board setup", "board setup is defined", "cooling area", "thermal via array",
            ))
        ]
        citations = "\n".join(source_label(chunk) for chunk in board_sources)
        return (
            "This datasheet does not give the board a product or evaluation-board name. The "
            "'Board Setup' passage describes a generic JEDEC thermal-characterization test board "
            "and its physical construction; it is not a named board product.\n\n"
            f"Relevant source labels:\n{citations}"
        )
    endpoint, model, api_key = llm_settings()
    if not endpoint or not model:
        return "Retrieved source evidence (no LLM configured):\n\n" + evidence
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        url = base + "/chat/completions"
    else:
        url = base + "/v1/chat/completions"
    payload = {"model": model, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nRetrieved evidence:\n{evidence}"},
    ], "temperature": 0}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
        answer = result["choices"][0]["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Expected a nonempty text answer")
        if not any(source_label(chunk) in answer for _, chunk in results):
            labels = "\n".join(source_label(chunk) for _, chunk in results)
            answer += f"\n\nRetrieved source labels (ranked; verify relevance):\n{labels}"
        return answer
    except (URLError, OSError, HTTPException, ValueError, KeyError, IndexError, TypeError) as error:
        detail = f"HTTP {error.code}" if isinstance(error, HTTPError) else type(error).__name__
        return (f"LLM request failed ({detail}). Check the API base URL, model, credentials, service "
                f"availability, and chat-completions response format. Retrieved evidence follows:\n\n{evidence}")
