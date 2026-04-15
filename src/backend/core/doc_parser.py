"""
Document Parser — PARAGRAPH-based partitioning with spatial image association
Using Unstructured.io + Tesseract OCR for scanned PDFs

Partitioning basis:
  - One element = one paragraph (split on blank lines)
  - ONLY paragraphs that have an image spatially beside them
    will have a non-empty `image_indices` list in their metadata.
  - Paragraphs with no adjacent image get `image_indices: []`

Install:
    pip install "unstructured[all-docs]" unstructured-inference pytesseract pillow pymupdf pypdf

System deps:
    sudo apt-get install -y tesseract-ocr poppler-utils
"""

import os
import io
import json
import base64
import re
import hashlib
from pathlib import Path
from typing import Optional

import fitz
import pytesseract
from PIL import Image
from pypdf import PdfReader

from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    NarrativeText, Title, ListItem, Table,
    Image as UnstructuredImage, FigureCaption,
    Header, Footer, CompositeElement,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paragraph splitter
# ─────────────────────────────────────────────────────────────────────────────

def split_into_paragraphs(text: str, min_chars: int = 20) -> list:
    """
    Split raw text into paragraphs by blank lines.
    Collapses internal line-wraps inside a paragraph into a single space.
    Drops chunks shorter than min_chars (noise / lone page numbers).
    """
    raw_chunks = re.split(r"\n{2,}", text)
    paragraphs = []
    for chunk in raw_chunks:
        flat = re.sub(r"(?<!\n)\n(?!\n)", " ", chunk)
        flat = re.sub(r" {2,}", " ", flat).strip()
        if len(flat) >= min_chars:
            paragraphs.append(flat)
    return paragraphs


# ─────────────────────────────────────────────────────────────────────────────
# Bounding-box helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_vertical_overlap(bbox_a: dict, bbox_b: dict, threshold: float = 0.3) -> bool:
    """
    Return True if two bounding boxes share significant vertical overlap.
    Used to detect side-by-side layout: text on LEFT, image on RIGHT,
    occupying the same vertical band on the page.

    Only paragraphs and images that actually share vertical space will match.
    Most paragraphs will NOT match any image and return False.
    """
    if not bbox_a or not bbox_b:
        return False

    top     = max(bbox_a["y1"], bbox_b["y1"])
    bottom  = min(bbox_a["y2"], bbox_b["y2"])
    overlap = max(0, bottom - top)

    height_a = bbox_a["y2"] - bbox_a["y1"]
    height_b = bbox_b["y2"] - bbox_b["y1"]
    shorter  = min(height_a, height_b)

    if shorter <= 0:
        return False

    return (overlap / shorter) >= threshold


def _extract_bbox(metadata: dict) -> Optional[dict]:
    """
    Pull a normalised bbox dict from Unstructured element metadata.
    Returns None if coordinates are not available.
    """
    coords = metadata.get("coordinates")
    if not coords:
        return None
    try:
        points = coords.get("points") or coords.get("layout_bbox")
        if not points or len(points) < 4:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}
    except Exception:
        return None


def _fitz_bbox(rect) -> dict:
    """Convert a fitz.Rect to our normalised bbox dict."""
    return {"x1": rect.x0, "y1": rect.y0, "x2": rect.x1, "y2": rect.y1}


# ─────────────────────────────────────────────────────────────────────────────
# Spatial association: link images to the paragraphs beside them
# ─────────────────────────────────────────────────────────────────────────────

def associate_images_to_paragraphs(text_elements: list, images: list) -> tuple:
    """
    For each image, check every paragraph on the same page.
    If their vertical bands overlap → they are side-by-side → link them.
    """
    for img in images:
        img_bbox = img["metadata"].get("bbox")
        img_page = img["metadata"].get("page_number")

        for para in text_elements:
            if para["metadata"].get("page_number") != img_page:
                continue  # different page — skip

            para_bbox = para["metadata"].get("bbox")

            # Only link if they genuinely overlap vertically (side-by-side check)
            if _bbox_vertical_overlap(img_bbox, para_bbox):
                para["metadata"]["image_indices"].append(img["index"])
                img["metadata"]["associated_paragraph_indices"].append(
                    para["metadata"]["paragraph_index"]
                )

    return text_elements, images


# ─────────────────────────────────────────────────────────────────────────────
# Scanned PDF: OCR → paragraph split → image extraction with bbox
# ─────────────────────────────────────────────────────────────────────────────

class ScannedPDFOCR:

    def __init__(
        self,
        dpi: int = 300,
        languages: list = None,
        image_output_dir: str = "output/images", # Changed default to match base setup
        min_paragraph_chars: int = 20,
    ):
        self.dpi                 = dpi
        self.lang                = "+".join(languages or ["eng"])
        self.image_output_dir    = image_output_dir
        self.min_paragraph_chars = min_paragraph_chars
        # Removed static folder creation here

    def extract(self, file_path: str) -> dict:
        print(f" [OCR] Scanned PDF — rasterising at {self.dpi} DPI")
        
        # 1. SET UP THE DIRECTORY FIRST
        file_name = Path(file_path).stem 
        specific_output_dir = Path(self.image_output_dir) / f"extracted_images_{file_name}"
        specific_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_output_dir = specific_output_dir

        # 2. NOW PERFORM THE EXTRACTION
        text_elements = self._ocr_pages(file_path)
        images        = self._extract_images(file_path)

        # 3. ASSOCIATE AND RETURN
        text_elements, images = associate_images_to_paragraphs(text_elements, images)

        return {"text_elements": text_elements, "images": images}
    
    def _ocr_pages(self, file_path: str) -> list:
        text_elements   = []
        doc             = fitz.open(file_path)
        global_para_idx = 0

        for page_num, page in enumerate(doc, start=1):
            print(f"   [OCR] Page {page_num}/{len(doc)} ...", end="\r")

            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            raw_text = pytesseract.image_to_string(img, lang=self.lang)

            tsv_data = pytesseract.image_to_data(
                img, lang=self.lang, output_type=pytesseract.Output.DICT
            )

            paragraphs  = split_into_paragraphs(raw_text, min_chars=self.min_paragraph_chars)
            para_bboxes = self._compute_paragraph_bboxes(paragraphs, tsv_data)

            for para_idx_on_page, para_text in enumerate(paragraphs):
                bbox = para_bboxes.get(para_idx_on_page)
                text_elements.append({
                    "type": "Paragraph",
                    "text": para_text,
                    "metadata": {
                        "page_number":     page_num,
                        "paragraph_index": global_para_idx,
                        "filename":        os.path.basename(file_path),
                        "source":          "ocr",
                        "bbox":            bbox,
                        "image_indices":   [],
                    },
                })
                global_para_idx += 1

        doc.close()
        print()
        return text_elements

    @staticmethod  #when a function logically belongs to a class but does not need access to self or cls
    def _compute_paragraph_bboxes(paragraphs: list, tsv_data: dict) -> dict:
        bboxes = {}
        words  = [w.strip() for w in tsv_data.get("text", [])]

        for para_idx, para_text in enumerate(paragraphs):
            para_words = para_text.split()
            if not para_words:
                continue

            xs1, ys1, xs2, ys2 = [], [], [], []
            pw_len = len(para_words)

            for i in range(len(words) - pw_len + 1):
                window = [w for w in words[i:i + pw_len] if w]
                if window == para_words[:len(window)]:
                    for j in range(i, min(i + pw_len, len(words))):
                        conf = int(tsv_data["conf"][j]) if tsv_data["conf"][j] != "-1" else 0
                        if conf > 0 and words[j].strip():
                            x, y = tsv_data["left"][j], tsv_data["top"][j]
                            w, h = tsv_data["width"][j], tsv_data["height"][j]
                            xs1.append(x);      ys1.append(y)
                            xs2.append(x + w);  ys2.append(y + h)
                    break

            if xs1:
                bboxes[para_idx] = {
                    "x1": min(xs1), "y1": min(ys1),
                    "x2": max(xs2), "y2": max(ys2),
                }

        return bboxes

    def _extract_images(self, file_path: str) -> list:
        images = []
        doc    = fitz.open(file_path)
        idx    = 0

        # Track content hashes so repeated images (e.g. logo on every page)
        # are saved to disk once and skipped on subsequent pages.
        seen_hashes: dict = {}   # md5_hex → first image index that had it

        for page_num, page in enumerate(doc, start=1):
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.colorspace and pix.colorspace.n > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    if pix.width < 100 or pix.height < 100:
                        continue

                    # ── Dedup: hash raw pixel bytes before saving ──────────
                    raw_bytes  = pix.tobytes("png")
                    img_hash   = hashlib.md5(raw_bytes).hexdigest()

                    if img_hash in seen_hashes:
                        print(f"   [Parser] Duplicate image skipped on page {page_num} "
                              f"(same as image #{seen_hashes[img_hash]})")
                        continue   # do NOT save to disk, do NOT add to list
                    seen_hashes[img_hash] = idx
                    # ──────────────────────────────────────────────────────

                    out_path = os.path.join(
                        self.current_output_dir, f"image_{idx:04d}_p{page_num}.png"
                    )
                    # Write the already-computed bytes (avoids second encode)
                    Path(out_path).write_bytes(raw_bytes)

                    img_rects = page.get_image_rects(xref)
                    bbox      = _fitz_bbox(img_rects[0]) if img_rects else None

                    images.append({
                        "index": idx,
                        "path":  out_path,
                        "metadata": {
                            "page_number":                  page_num,
                            "width":                        pix.width,
                            "height":                       pix.height,
                            "bbox":                         bbox,
                            "associated_paragraph_indices": [],
                        },
                    })
                    idx += 1
                except Exception as e:
                    print(f"   [OCR] Warning — image xref={xref}: {e}")

        doc.close()
        return images


# ─────────────────────────────────────────────────────────────────────────────
# Normal PDFs / DOCX / PPTX via Unstructured → paragraph + association
# ─────────────────────────────────────────────────────────────────────────────

PARAGRAPH_TYPES = (
    NarrativeText, Title, ListItem,
    FigureCaption, Header, Footer, CompositeElement,
)


def _unstructured_to_paragraphs(elements: list, source_file: str) -> tuple:
    text_elements   = []
    images          = []
    img_idx         = 0
    global_para_idx = 0

    for el in elements:
        meta = el.metadata.to_dict() if hasattr(el.metadata, "to_dict") else {}

        if isinstance(el, UnstructuredImage):
            record = {
                "index": img_idx,
                "metadata": {
                    "page_number":                  meta.get("page_number"),
                    "bbox":                         _extract_bbox(meta),
                    "associated_paragraph_indices": [],
                },
            }
            img_path = meta.get("image_path")
            if img_path and os.path.exists(str(img_path)):
                record["path"] = img_path
            elif getattr(el.metadata, "image_base64", None):
                record["base64"] = el.metadata.image_base64
            images.append(record)
            img_idx += 1

        elif isinstance(el, Table):
            text_elements.append({
                "type": "Table",
                "text": el.text.strip(),
                "metadata": {
                    "page_number":     meta.get("page_number"),
                    "paragraph_index": global_para_idx,
                    "filename":        meta.get("filename", os.path.basename(source_file)),
                    "bbox":            _extract_bbox(meta),
                    "image_indices":   [],
                },
            })
            global_para_idx += 1

        elif isinstance(el, PARAGRAPH_TYPES):
            raw = el.text.strip()
            if not raw:
                continue

            sub_paragraphs = split_into_paragraphs(raw, min_chars=10) or [raw]

            for sub in sub_paragraphs:
                text_elements.append({
                    "type": type(el).__name__,
                    "text": sub,
                    "metadata": {
                        "page_number":     meta.get("page_number"),
                        "paragraph_index": global_para_idx,
                        "filename":        meta.get("filename", os.path.basename(source_file)),
                        "bbox":            _extract_bbox(meta),
                        "image_indices":   [],
                    },
                })
                global_para_idx += 1

    text_elements, images = associate_images_to_paragraphs(text_elements, images)

    return text_elements, images


# ─────────────────────────────────────────────────────────────────────────────
# Scanned PDF detection
# ─────────────────────────────────────────────────────────────────────────────

def _is_scanned_pdf(file_path: str, sample_pages: int = 3) -> bool:
    try:
        reader = PdfReader(file_path)
        for i in range(min(sample_pages, len(reader.pages))):
            if (reader.pages[i].extract_text() or "").strip():
                return False
        return True
    except Exception:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Main DocumentParser
# ─────────────────────────────────────────────────────────────────────────────

class DocumentParser:

    def __init__(
        self,
        extract_images:      bool          = True,
        image_output_dir:    Optional[str] = None,
        strategy:            str           = "fast", # "fast" | "hi_res" (Unstructured PDF partitioning strategy)
        languages:           list          = None,
        ocr_dpi:             int           = 300,
        force_ocr:           bool          = False,
        min_paragraph_chars: int           = 20,
        bbox_overlap_threshold: float      = 0.3,
    ):
        self.extract_images         = extract_images
        self.image_output_dir       = image_output_dir or "output/images"
        self.strategy               = strategy
        self.languages              = languages or ["eng"]
        self.ocr_dpi                = ocr_dpi
        self.force_ocr              = force_ocr
        self.min_paragraph_chars    = min_paragraph_chars
        self.bbox_overlap_threshold = bbox_overlap_threshold
        # Removed static folder creation here

    def parse(self, file_path: str) -> dict:
        file_path = str(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext    = Path(file_path).suffix.lower()
        is_pdf = ext == ".pdf"

        print(f"\n[Parser] File     : {file_path}")
        print(f"[Parser] Strategy : {self.strategy}")

        use_ocr = self.force_ocr or (is_pdf and _is_scanned_pdf(file_path))
        
        # Create the dynamic path logic to pass to OCR or Unstructured
        file_name = Path(file_path).stem
        specific_output_dir = str(Path(self.image_output_dir) / f"extracted_images_{file_name}")

        if use_ocr:
            ocr = ScannedPDFOCR(
                dpi=self.ocr_dpi,
                languages=self.languages,
                image_output_dir=self.image_output_dir,
                min_paragraph_chars=self.min_paragraph_chars,
            )
            result        = ocr.extract(file_path)
            text_elements = result["text_elements"]
            images        = result["images"] if self.extract_images else []
        else:
            raw_elements  = self._partition(file_path, specific_output_dir)
            text_elements, images = _unstructured_to_paragraphs(raw_elements, file_path)

        full_text = "\n\n".join(e["text"] for e in text_elements)

        paras_with_image = sum(
            1 for e in text_elements if e["metadata"].get("image_indices")
        )

        print(f"[Parser] Total paragraphs          : {len(text_elements)}")
        print(f"[Parser] Paragraphs with image     : {paras_with_image}")
        print(f"[Parser] Paragraphs without image  : {len(text_elements) - paras_with_image}")
        print(f"[Parser] Total images              : {len(images)}")
        print(f"[Parser] OCR used                  : {use_ocr}")

        return {
            "file":                    file_path,
            "text_elements":           text_elements,
            "images":                  images,
            "full_text":               full_text,
            "ocr_used":                use_ocr,
            "total_paragraphs":        len(text_elements),
            "total_images":            len(images),
            "paragraphs_with_image":   paras_with_image,
        }

    def parse_to_json(self, file_path: str, output_path: Optional[str] = None) -> str:
        result      = self.parse(file_path)
        output_path = output_path or Path(file_path).stem + "_parsed.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"[Parser] Saved to : {output_path}")
        return output_path

    def _partition(self, file_path: str, specific_output_dir: str) -> list:
        ext    = Path(file_path).suffix.lower()
        common = dict(strategy=self.strategy, languages=self.languages)

        if ext == ".pdf":
            # Make sure the folder exists for Unstructured
            if self.extract_images:
                Path(specific_output_dir).mkdir(parents=True, exist_ok=True)
                
            return partition_pdf(
                filename=file_path,
                extract_images_in_pdf=self.extract_images,
                extract_image_block_output_dir=(
                    specific_output_dir if self.extract_images else None
                ),
                extract_image_block_types=(
                    ["Image", "Table"] if self.extract_images else []
                ),
                **common,
            )

        return partition(filename=file_path, **common)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_images_from_base64(images: list, output_dir: str = "extracted_images") -> list:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved = []
    for img in images:
        if "base64" in img:
            out_path = os.path.join(output_dir, f"image_{img['index']:04d}.png")
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(img["base64"]))
            saved.append(out_path)
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        file_path = r"documents\GMMDC-Media-Articles-2019.pdf"
        output_path = r"output/parsed/GMMDC-Media-Articles-2019.json"
        force_ocr = True
    else:
        file_path = sys.argv[1]
        output_path = None
        force_ocr = "--force-ocr" in sys.argv

        for arg in sys.argv[2:]:
            if not arg.startswith("--"):
                output_path = arg

    parser = DocumentParser(
        extract_images=True,
        image_output_dir="output/images", # Set to output/images base
        strategy="fast",
        ocr_dpi=300,
        force_ocr=force_ocr,
        min_paragraph_chars=20,
        bbox_overlap_threshold=0.3,
    )

    result_path = parser.parse_to_json(file_path, output_path)

    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n── Paragraphs WITH image beside them ────────────────────────────")
    for el in data["text_elements"]:
        if el["metadata"].get("image_indices"):
            m = el["metadata"]
            print(f"  Para #{m['paragraph_index']} | page={m.get('page_number')}"
                  f" | image_indices={m['image_indices']}")
            print(f"  text: {el['text'][:100]}")

    print(f"\n── Image → paragraph mapping ────────────────────────────────────")
    for img in data["images"]:
        m = img["metadata"]
        assoc = m.get("associated_paragraph_indices")
        label = f"beside para(s) {assoc}" if assoc else "no paragraph beside it"
        print(f"  Image #{img['index']} | page={m.get('page_number')} | {label}")

    print(f"\n── Totals ───────────────────────────────────────────────────────")
    print(f"  Total paragraphs         : {data['total_paragraphs']}")
    print(f"  Paragraphs with image    : {data['paragraphs_with_image']}")
    print(f"  Paragraphs without image : {data['total_paragraphs'] - data['paragraphs_with_image']}")
    print(f"  Total images             : {data['total_images']}")