import json
import logging
from typing import List, Dict

from src.backend.utils.regex_utils import extract_points
from src.backend.utils.spatial_utils import get_nearby_paragraphs

# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger("chunker")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [Chunker] %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(_handler)

logger.setLevel(logging.DEBUG)


# ─────────────────────────────────────────────────────────────
# Short chunk merger
# ─────────────────────────────────────────────────────────────

def merge_short_chunks(chunks: List[Dict], min_chars: int = 100) -> List[Dict]:
    merged = []
    for chunk in chunks:
        if merged and len(chunk["text"]) < min_chars:
            merged[-1]["text"] += " " + chunk["text"]
        else:
            merged.append(chunk)
    return merged


# ─────────────────────────────────────────────────────────────
# Build multimodal chunks
# ─────────────────────────────────────────────────────────────

def build_chunks(parsed_data: Dict) -> List[Dict]:
    text_elements = parsed_data["text_elements"]
    images        = parsed_data["images"]
    image_map     = {img["index"]: img for img in images}

    chunks = []

    for para in text_elements:
        meta          = para["metadata"]
        text          = para["text"]
        image_indices = meta.get("image_indices", [])

        image_paths       = []
        image_caption     = None
        image_description = []

        for idx in image_indices:
            img = image_map.get(idx)
            if not img:
                continue

            if "path" in img:
                image_paths.append(img["path"])

            nearby_paras = get_nearby_paragraphs(img, text_elements)

            if nearby_paras:
                image_caption = nearby_paras[0]["text"]
                for p in nearby_paras:
                    refined = extract_points(p["text"])
                    image_description.extend(refined)

        chunk = {
            "text":              text,
            "image_paths":       image_paths,
            "image_caption":     image_caption,
            "image_description": image_description,
            "metadata":          meta,
        }
        chunks.append(chunk)

    # ── Merge short chunks ──────────────────────────────────────
    chunks = merge_short_chunks(chunks, min_chars=100)

    return chunks


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        input_file  = r"output\\parsed\\GMMDC-Media-Articles-2019.json"
        output_file = r"output\\chunks\\chunks.json"
    else:
        input_file  = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else r"output\\chunks\\chunks.json"

    print(f"[Chunker] Input : {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = build_chunks(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[Chunker] Total chunks : {len(chunks)}")
    print(f"[Chunker] Saved to    : {output_file}")