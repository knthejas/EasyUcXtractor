"""Crop Docling picture provenance from a source PDF without model downloads."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from easyucxtractor.paths import ROOT

RENDER_SCALE = 2.0
DECORATIVE_REPEAT_MIN_PAGES = 5
DECORATIVE_MAX_PAGE_AREA = 0.015
DECORATIVE_POSITION_TOLERANCE = 0.02


def bbox_of(prov):
    bbox = prov["bbox"]
    return {"left": bbox["l"], "top": bbox["t"], "right": bbox["r"],
            "bottom": bbox["b"], "coord_origin": bbox["coord_origin"]}


def ref_index(ref: str) -> tuple[str, int]:
    collection, index = ref[2:].split("/")
    return collection, int(index)


def text_from_ref(document: dict, ref: dict) -> str:
    collection, index = ref_index(ref["$ref"])
    return document.get(collection, [])[index].get("text", "").strip()


def texts_from_refs(document: dict, refs: list[dict]) -> list[str]:
    texts, seen = [], set()
    for ref in refs:
        text = text_from_ref(document, ref)
        if text and text not in seen:
            texts.append(text)
            seen.add(text)
    return texts


def crop_figure(page, prov: dict):
    page_width, page_height = page.get_size()
    image = page.render(scale=RENDER_SCALE).to_pil()
    bbox = prov["bbox"]
    left = max(0, int(bbox["l"] * RENDER_SCALE))
    right = min(image.width, int(bbox["r"] * RENDER_SCALE))
    top = max(0, int((page_height - bbox["t"]) * RENDER_SCALE))
    bottom = min(image.height, int((page_height - bbox["b"]) * RENDER_SCALE))
    return image.crop((left, top, right, bottom)) if right > left and bottom > top else None


def iter_body_children(document: dict, children: list[dict], seen: set[str]):
    for child in children:
        ref = child["$ref"]
        if ref in seen:
            continue
        seen.add(ref)
        yield child
        collection, index = ref_index(ref)
        if collection in {"groups", "tables"}:
            yield from iter_body_children(document, document[collection][index].get("children", []), seen)


def classify_repeated_decorations(records, pdf) -> int:
    groups = defaultdict(list)
    for record in records:
        with Image.open(ROOT / record["image_path"]) as image:
            rgb = image.convert("RGB")
            groups[(rgb.size, hashlib.sha256(rgb.tobytes()).digest())].append(record)
    count = 0
    for repeated in groups.values():
        if len({record["page"] for record in repeated}) < DECORATIVE_REPEAT_MIN_PAGES:
            continue
        if any(record.get("caption") for record in repeated):
            continue
        centers, areas = [], []
        for record in repeated:
            page_width, page_height = pdf[record["page"] - 1].get_size()
            bbox = record["bbox"]
            width, height = bbox["right"] - bbox["left"], bbox["top"] - bbox["bottom"]
            centers.append(((bbox["left"] + bbox["right"]) / (2 * page_width),
                            (bbox["top"] + bbox["bottom"]) / (2 * page_height)))
            areas.append(width * height / (page_width * page_height))
        spread = max(max(x for x, _ in centers) - min(x for x, _ in centers),
                     max(y for _, y in centers) - min(y for _, y in centers))
        if max(areas) <= DECORATIVE_MAX_PAGE_AREA and spread <= DECORATIVE_POSITION_TOLERANCE:
            for record in repeated:
                record["decorative_asset"] = True
                record["classification_reason"] = (
                    "Exact image repeated on multiple pages at the same position; "
                    "small area and no caption. Retained as source, excluded from RAG."
                )
                count += 1
    for record in records:
        record.setdefault("decorative_asset", False)
        record.setdefault("classification_reason", None)
    return count


def extract_figures(pdf_path: Path, input_json: Path, output_dir: Path) -> Path:
    document = json.loads(input_json.read_text(encoding="utf-8"))
    pdf = pdfium.PdfDocument(str(pdf_path))
    image_dir = output_dir / "figures"
    image_dir.mkdir(parents=True, exist_ok=True)
    figures_jsonl = output_dir / "figures.jsonl"
    existing_records = {}
    if figures_jsonl.exists():
        with figures_jsonl.open(encoding="utf-8") as source:
            existing_records = {record["source_ref"]: record for record in map(json.loads, source)}
    records, current_section, seen_refs = [], None, set()
    used_image_paths = set()
    for child in iter_body_children(document, document["body"]["children"], seen_refs):
        collection, index = ref_index(child["$ref"])
        if collection == "texts":
            text_item = document["texts"][index]
            if text_item.get("label") == "section_header":
                current_section = text_item.get("text", "").strip()
            continue
        if collection != "pictures":
            continue
        item = document["pictures"][index]
        prov = item["prov"][0]
        page_no = prov["page_no"]
        previous = existing_records.get(item["self_ref"], {})
        previous_path = ROOT / previous.get("image_path", "")
        if not previous_path.is_file() and previous.get("image_path"):
            previous_path = image_dir / Path(previous["image_path"]).name
        if previous_path.is_file() and previous_path not in used_image_paths:
            image_path = previous_path
            figure_id = previous.get("figure_id", f"{Path(pdf_path).stem}-picture-{index:04d}")
        else:
            image = crop_figure(pdf[page_no - 1], prov)
            if image is None:
                print(f"Warning: could not crop {item['self_ref']}")
                continue
            image_path = image_dir / f"page-{page_no:03d}-picture-{index:04d}.png"
            image.save(image_path, format="PNG")
            figure_id = f"{Path(pdf_path).stem}-picture-{index:04d}"
        used_image_paths.add(image_path)
        captions = texts_from_refs(document, item.get("captions", []))
        extracted_text = texts_from_refs(document, item.get("children", []))
        caption = " ".join(captions) or None
        records.append({
            "figure_id": figure_id,
            "document_id": Path(pdf_path).stem,
            "content_type": "figure",
            "image_path": image_path.resolve().relative_to(ROOT).as_posix(),
            "caption": caption,
            "extracted_text": extracted_text,
            "content": "\n".join(filter(None, [caption, *extracted_text])),
            "section": current_section,
            "page": page_no,
            "bbox": bbox_of(prov),
            "source_ref": item["self_ref"],
            "description": None,
        })
    count = classify_repeated_decorations(records, pdf)
    with figures_jsonl.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} figure records to {figures_jsonl}; marked {count} repeated decorative assets.")
    return figures_jsonl
