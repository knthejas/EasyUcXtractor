"""Validate generated datasheet artifact structure and one reviewed fact."""

import json
from pathlib import Path

from easyucxtractor.paths import ROOT


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
    return rows


def validate_artifacts(output_json: Path, knowledge_jsonl: Path, parameters_jsonl: Path,
                       figures_jsonl: Path, report_path: Path) -> dict:
    errors, warnings = [], []
    document = json.loads(output_json.read_text(encoding="utf-8"))
    chunks, parameters, figures = (read_jsonl(p) for p in (knowledge_jsonl, parameters_jsonl, figures_jsonl))
    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        errors.append("knowledge base has duplicate chunk_id values")
    allowed_types = {"text", "table", "figure"}
    empty_figure_chunks = 0
    for index, chunk in enumerate(chunks, 1):
        if chunk.get("content_type") not in allowed_types:
            errors.append(f"knowledge chunk {index} has unsupported content_type")
        if not chunk.get("content", "").strip():
            if chunk.get("content_type") == "figure":
                empty_figure_chunks += 1
            else:
                warnings.append(f"non-figure knowledge chunk {index} has empty content")
        if not isinstance(chunk.get("page"), int) or chunk["page"] < 1:
            errors.append(f"knowledge chunk {index} has invalid page provenance")
        # Some valid content, especially cover pages, appears before the first
        # section heading. Document identity and page provenance are required;
        # section labels are optional when the source PDF has none.
        if not chunk.get("document_id"):
            errors.append(f"knowledge chunk {index} lacks document_id")
        if chunk.get("content_type") == "figure":
            image_path = ROOT / chunk.get("image_path", "")
            if not image_path.is_file():
                errors.append(f"figure chunk {index} points to missing image: {image_path}")
            if chunk.get("decorative_asset"):
                errors.append(f"decorative figure leaked into knowledge chunk {index}")
    chunk_id_set = set(chunk_ids)
    unresolved = 0
    for index, parameter in enumerate(parameters, 1):
        if parameter.get("source_chunk_id") not in chunk_id_set:
            errors.append(f"parameter record {index} references a missing source chunk")
        if not isinstance(parameter.get("page"), int) or parameter["page"] < 1:
            errors.append(f"parameter record {index} has invalid page provenance")
        unresolved += sum(isinstance(parameter.get(key), str) and parameter[key].strip()
                          and parameter[key] not in {"-", "–", "—"} for key in ("min", "typ", "max"))
    picture_refs = {picture.get("self_ref") for picture in document.get("pictures", [])}
    figure_refs = {figure.get("source_ref") for figure in figures}
    if picture_refs != figure_refs:
        errors.append(f"figure coverage mismatch: {len(picture_refs - figure_refs)} missing, {len(figure_refs - picture_refs)} extra")
    opening_text = " ".join(item.get("text", "") for item in document.get("texts", [])[:50])
    is_tle9563 = "TLE9563-3QX" in opening_text.upper()
    if is_tle9563 and not any(p.get("parameter") == "Junction Temperature" and p.get("min") == -40
               and p.get("max") == 150 and p.get("unit") == "°C" for p in parameters):
        errors.append("known-fact check failed: junction temperature -40 to 150 °C not found")
    counts = {
        "docling_texts": len(document.get("texts", [])),
        "docling_tables": len(document.get("tables", [])),
        "docling_pictures": len(document.get("pictures", [])),
        "knowledge_chunks": len(chunks),
        "knowledge_by_type": {kind: sum(c.get("content_type") == kind for c in chunks) for kind in sorted(allowed_types)},
        "parameter_records": len(parameters), "figure_records": len(figures),
        "figures_without_searchable_text": empty_figure_chunks,
        "unresolved_parameter_expressions": unresolved,
        "warnings": len(warnings), "errors": len(errors),
    }
    report = {"status": "fail" if errors else "pass_with_warnings" if warnings else "pass",
              "counts": counts, "errors": errors, "warnings": warnings}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validation: {report['status']} ({len(errors)} errors)")
    if errors:
        print("\n".join(errors))
    print(f"Report: {report_path}")
    return report
