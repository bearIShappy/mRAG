import math


def is_left_of(img_bbox, para_bbox):
    return para_bbox["x2"] <= img_bbox["x1"]


def is_right_of(img_bbox, para_bbox):
    return para_bbox["x1"] >= img_bbox["x2"]


def is_above(img_bbox, para_bbox):
    return para_bbox["y2"] <= img_bbox["y1"]


def bbox_distance(a, b):
    ax = (a["x1"] + a["x2"]) / 2
    ay = (a["y1"] + a["y2"]) / 2
    bx = (b["x1"] + b["x2"]) / 2
    by = (b["y1"] + b["y2"]) / 2

    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def get_nearby_paragraphs(image, text_elements, max_neighbors=3):
    img_bbox = image["metadata"].get("bbox")
    page     = image["metadata"].get("page_number")

    if not img_bbox:
        return []

    candidates = []

    for para in text_elements:
        if para["metadata"].get("page_number") != page:
            continue

        para_bbox = para["metadata"].get("bbox")
        if not para_bbox:
            continue

        if (
            is_left_of(img_bbox, para_bbox) or
            is_right_of(img_bbox, para_bbox) or
            is_above(img_bbox, para_bbox)
        ):
            dist = bbox_distance(img_bbox, para_bbox)
            candidates.append((dist, para))

    candidates.sort(key=lambda x: x[0])

    return [c[1] for c in candidates[:max_neighbors]]