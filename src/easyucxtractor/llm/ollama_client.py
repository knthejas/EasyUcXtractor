"""Grounded chat-completions client, compatible with local Ollama."""

import json
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from easyucxtractor.config import llm_settings
from easyucxtractor.retrieval.bm25_retriever import source_label

SYSTEM_PROMPT = (
    "Give a concise answer using only the supplied datasheet evidence. Do not use general knowledge, "
    "invent values, or infer unsupported specifications. If evidence is insufficient, explicitly say "
    "the datasheet evidence is insufficient; absence of a feature in these excerpts does not establish "
    "that it is unsupported. Preserve numerical values and units exactly when possible. Distinguish "
    "Absolute Maximum Ratings from Functional/Operating Range. Include relevant source, page, section, "
    "and specification ID, copying citations only from supplied source labels."
)


def answer_from_evidence(question: str, results: list[tuple[float, dict]]) -> str:
    if not results:
        return "I could not find supporting evidence in the indexed datasheet."
    evidence = "\n\n".join(f"{source_label(chunk)}\n{chunk.get('content', '')[:1800]}"
                             for _, chunk in results)
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
