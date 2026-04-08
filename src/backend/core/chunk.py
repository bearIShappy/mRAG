# import json
# import math
# import re
# from typing import List, Dict


# # ─────────────────────────────────────────────────────────────
# # Regex extractor (optional refinement)
# # ─────────────────────────────────────────────────────────────

# def extract_points(text: str) -> List[str]:
#     extracted_matches = []

#     pattern = r'\d+\.\s(.*?)(?=\n\d+\.\s|$)'
#     matches = re.findall(pattern, text, re.DOTALL)

#     for match in matches:
#         extracted_matches.append(match.strip())

#     if not extracted_matches:
#         return [text]

#     return extracted_matches


# # ─────────────────────────────────────────────────────────────
# # Spatial helpers
# # ─────────────────────────────────────────────────────────────

# def is_left_of(img_bbox, para_bbox):
#     return para_bbox["x2"] <= img_bbox["x1"]


# def is_right_of(img_bbox, para_bbox):
#     return para_bbox["x1"] >= img_bbox["x2"]


# def is_above(img_bbox, para_bbox):
#     return para_bbox["y2"] <= img_bbox["y1"]


# def bbox_distance(a, b):
#     ax = (a["x1"] + a["x2"]) / 2
#     ay = (a["y1"] + a["y2"]) / 2
#     bx = (b["x1"] + b["x2"]) / 2
#     by = (b["y1"] + b["y2"]) / 2

#     return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


# # ─────────────────────────────────────────────────────────────
# # Find nearby paragraphs for an image
# # ─────────────────────────────────────────────────────────────

# def get_nearby_paragraphs(image, text_elements, max_neighbors=3):
#     img_bbox = image["metadata"].get("bbox")
#     page     = image["metadata"].get("page_number")

#     if not img_bbox:
#         return []

#     candidates = []

#     for para in text_elements:
#         if para["metadata"].get("page_number") != page:
#             continue

#         para_bbox = para["metadata"].get("bbox")
#         if not para_bbox:
#             continue

#         # spatial relation
#         if (
#             is_left_of(img_bbox, para_bbox) or
#             is_right_of(img_bbox, para_bbox) or
#             is_above(img_bbox, para_bbox)
#         ):
#             dist = bbox_distance(img_bbox, para_bbox)
#             candidates.append((dist, para))

#     candidates.sort(key=lambda x: x[0])

#     return [c[1] for c in candidates[:max_neighbors]]


# # ─────────────────────────────────────────────────────────────
# # Build multimodal chunks
# # ─────────────────────────────────────────────────────────────

# def build_chunks(parsed_data: Dict) -> List[Dict]:
#     text_elements = parsed_data["text_elements"]
#     images = parsed_data["images"]

#     image_map = {img["index"]: img for img in images}
#     chunks = []

#     for para in text_elements:
#         meta = para["metadata"]
#         text = para["text"]
#         image_indices = meta.get("image_indices", [])

#         image_paths = []
#         image_caption = None
#         image_description = []

#         for idx in image_indices:
#             img = image_map.get(idx)
#             if not img:
#                 continue

#             # image path
#             if "path" in img:
#                 image_paths.append(img["path"])

#             # 🔥 spatial paragraph retrieval
#             nearby_paras = get_nearby_paragraphs(img, text_elements)

#             if nearby_paras:
#                 # Closest paragraph = caption
#                 image_caption = nearby_paras[0]["text"]

#                 # Full nearby paragraphs = description
#                 for p in nearby_paras:
#                     refined = extract_points(p["text"])
#                     image_description.extend(refined)

#         chunk = {
#             "text": text,
#             "image_paths": image_paths,
#             "image_caption": image_caption,
#             "image_description": image_description,
#             "metadata": meta
#         }

#         chunks.append(chunk)

#     return chunks


# # ─────────────────────────────────────────────────────────────
# # CLI usage
# # ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) < 2:
#         input_file = "drylab_parsed.json"
#         output_file = "chunks.json"
#     else:
#         input_file = sys.argv[1]
#         output_file = sys.argv[2] if len(sys.argv) > 2 else "chunks.json"

#     print(f"[Chunker] Input : {input_file}")

#     with open(input_file, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     chunks = build_chunks(data)

#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(chunks, f, indent=2, ensure_ascii=False)

#     print(f"[Chunker] Total chunks created : {len(chunks)}")
#     print(f"[Chunker] Saved to            : {output_file}")

import json
from typing import List, Dict

from src.backend.utils.regex_utils import extract_points
from src.backend.utils.spatial_utils import get_nearby_paragraphs


def build_chunks(parsed_data: Dict) -> List[Dict]:
    text_elements = parsed_data["text_elements"]
    images = parsed_data["images"]

    image_map = {img["index"]: img for img in images}
    chunks = []

    for para in text_elements:
        meta = para["metadata"]
        text = para["text"]
        image_indices = meta.get("image_indices", [])

        image_paths = []
        image_caption = None
        image_description = []

        for idx in image_indices:
            img = image_map.get(idx)
            if not img:
                continue

            if "path" in img:
                image_paths.append(img["path"])

            # 🔥 spatial retrieval
            nearby_paras = get_nearby_paragraphs(img, text_elements)

            if nearby_paras:
                image_caption = nearby_paras[0]["text"]

                for p in nearby_paras:
                    refined = extract_points(p["text"])
                    image_description.extend(refined)

        chunk = {
            "text": text,
            "image_paths": image_paths,
            "image_caption": image_caption,
            "image_description": image_description,
            "metadata": meta
        }

        chunks.append(chunk)

    return chunks


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        input_file = "GMMDC-Media-Articles-2019_parsed.json"
        output_file = "output\\chunks\\chunks.json"
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "chunks.json"

    print(f"[Chunker] Input : {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = build_chunks(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[Chunker] Total chunks : {len(chunks)}")
    print(f"[Chunker] Saved to    : {output_file}")