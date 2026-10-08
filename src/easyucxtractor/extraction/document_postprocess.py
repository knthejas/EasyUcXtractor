"""Remove known running-header and table-of-contents noise before export."""

import re
from collections import Counter

from docling_core.types.doc.labels import DocItemLabel

DOT_LEADER_RE = re.compile(r"(?:\.\s){3,}\.?")
RUNNING_HEADER_MIN_COUNT = 4


def _clean_dot_leaders(text: str) -> str:
    return DOT_LEADER_RE.sub(" ", text).strip()


def strip_toc_dot_leaders(doc) -> None:
    for item in doc.texts:
        if item.text and DOT_LEADER_RE.search(item.text):
            item.text = _clean_dot_leaders(item.text)
    for table in doc.tables:
        for cell in table.data.table_cells:
            if cell.text and DOT_LEADER_RE.search(cell.text):
                cell.text = _clean_dot_leaders(cell.text)


def remove_running_headers(doc) -> int:
    headers = [item for item in doc.texts if item.label == DocItemLabel.SECTION_HEADER]
    counts = Counter(item.text.strip() for item in headers)
    seen = set()
    to_delete = []
    for item in headers:
        key = item.text.strip()
        if counts[key] >= RUNNING_HEADER_MIN_COUNT:
            if key in seen:
                to_delete.append(item)
            else:
                seen.add(key)
    if to_delete:
        doc.delete_items(node_items=to_delete)
    return len(to_delete)


def clean_document(doc) -> None:
    removed = remove_running_headers(doc)
    strip_toc_dot_leaders(doc)
    print(f"Postprocess: removed {removed} running-header duplicates, stripped ToC dot leaders.")
