"""Extract deterministic numeric/specification records from table chunks."""

import json
import re
from pathlib import Path

NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", header.lower()).strip()


def find_column(headers: list[str], *terms: str) -> int | None:
    for index, header in enumerate(map(normalize_header, headers)):
        if all(term in header for term in terms):
            return index
    return None


def parse_value(value: str):
    value = value.strip()
    if value in {"", "-", "–", "—"}:
        return None
    if NUMERIC_RE.fullmatch(value):
        number = float(value)
        return int(number) if number.is_integer() else number
    return value


def is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?", line))


def is_banner_row(cells: list[str]) -> bool:
    nonempty = [cell for cell in cells if cell]
    return len(nonempty) > 1 and len(set(nonempty)) == 1


def canonical_parameter(parameter: str, symbol: str) -> str | None:
    text = f"{parameter} {symbol}".lower()
    if "junction temperature" in text:
        return "junction_temperature"
    if "storage temperature" in text:
        return "storage_temperature"
    if "operating temperature" in text or ("functional range" in text and "temperature" in text):
        return "operating_temperature"
    if "supply voltage" in text or "bridge supply voltage" in text:
        return "supply_voltage"
    if "can supply voltage" in text:
        return "can_supply_voltage"
    if "output voltage" in text and "vcc1" in text:
        return "vcc1_output_voltage"
    if "overcurrent limitation" in text:
        return "vcc1_overcurrent_limit"
    if "spi" in text and "interface" in text:
        return "spi_interface"
    return None


def table_rows(markdown: str):
    lines = [line for line in markdown.splitlines() if line.strip().startswith("|")]
    for index, line in enumerate(lines):
        if is_separator(line) or index + 1 >= len(lines):
            continue
        headers = split_row(line)
        if not all(any(term in normalize_header(h) for h in headers)
                   for term in ("parameter", "symbol", "unit", "min", "max")):
            continue
        cols = {name: find_column(headers, *terms) for name, terms in {
            "parameter": ("parameter",), "symbol": ("symbol",), "min": ("min",),
            "typ": ("typ",), "max": ("max",), "unit": ("unit",),
            "condition": ("note",), "number": ("number",),
        }.items()}
        width = len(headers)
        for row_line in lines[index + 2:]:
            if is_separator(row_line):
                continue
            cells = split_row(row_line)
            if len(cells) != width or is_banner_row(cells):
                continue
            parameter, symbol = cells[cols["parameter"]], cells[cols["symbol"]]
            if not parameter or not symbol:
                continue
            yield {
                "parameter": parameter, "symbol": symbol,
                "min": parse_value(cells[cols["min"]]),
                "typ": parse_value(cells[cols["typ"]]) if cols["typ"] is not None else None,
                "max": parse_value(cells[cols["max"]]),
                "unit": cells[cols["unit"]],
                "condition": cells[cols["condition"]] if cols["condition"] is not None else None,
                "spec_number": cells[cols["number"]] if cols["number"] is not None else None,
            }
        return


def build_parameters(input_jsonl: Path, output_jsonl: Path) -> int:
    records = []
    with input_jsonl.open(encoding="utf-8") as source:
        for line in source:
            chunk = json.loads(line)
            if chunk.get("content_type") != "table":
                continue
            for row in table_rows(chunk.get("content", "")):
                row.update({
                    "parameter_key": canonical_parameter(row["parameter"], row["symbol"]),
                    "document_id": chunk["document_id"], "component": chunk["component"],
                    "section": chunk["section"], "page": chunk["page"],
                    "source_chunk_id": chunk["chunk_id"], "source_text": chunk["source_text"],
                })
                records.append(row)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} parameter records to {output_jsonl}")
    return len(records)
