"""
embedder.py
-----------
Converts chunks into dense vectors.

Strategies (set EMBEDDING_STRATEGY in config/settings.py):
  "text"       → mixedbread-ai/mxbai-embed-large-v1  (1024-dim)
  "multimodal" → clip-ViT-B-32                        (512-dim, text + image)

mxbai prefix rules
------------------
  Passages (index time) → NO prefix  (PASSAGE_INSTRUCTION = "")
  Queries  (query time) → prepend QUERY_INSTRUCTION

Usage:
    from backend.core.embedder import Embedder
    embedder = Embedder()
    chunks   = embedder.embed_chunks(chunks)
    q_vec    = embedder.embed_query("What is GMMDC?")
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer

from src.backend.config.settings import settings


# ─────────────────────────────────────────────────────────────
# Logger setup
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger("embedder")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [Embedder] %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(_handler)

logger.setLevel(logging.DEBUG)   # flip to INFO in production


# ─────────────────────────────────────────────────────────────
# Passage text builder
# ─────────────────────────────────────────────────────────────

def _build_embed_text(chunk: Dict, passage_instruction: str = "") -> str:
    """
    Combine paragraph text + image caption + image description
    into the string that gets embedded at index time.

    passage_instruction is prepended when the model requires it
    (e.g. E5 uses "passage: "). mxbai leaves it empty.
    """
    parts = []

    text = (chunk.get("text") or "").strip()
    if text:
        parts.append(text)
    else:
        logger.debug("Chunk has no 'text' field — will produce partial embed_text")

    caption = (chunk.get("image_caption") or "").strip()
    if caption and caption != text:
        parts.append(f"[Image Caption]: {caption}")
        logger.debug("Added image caption: '%s'", caption[:60])

    descriptions = chunk.get("image_description") or []
    if descriptions:
        joined = " | ".join(d.strip() for d in descriptions if d.strip())
        if joined:
            parts.append(f"[Image Context]: {joined}")
            logger.debug("Added image description (%d points): '%s'", len(descriptions), joined[:60])

    raw = "\n".join(parts) if parts else "empty"

    if raw == "empty":
        logger.warning("Chunk produced empty embed_text — metadata: %s",
                       chunk.get("metadata", {}))

    result = f"{passage_instruction}{raw}" if passage_instruction else raw
    logger.debug("embed_text built (%d chars): '%s'", len(result), result[:80].replace("\n", " ↵ "))
    return result


# ─────────────────────────────────────────────────────────────
# Embedder
# ─────────────────────────────────────────────────────────────

class Embedder:
    """
    Reads all config from settings.py.
    Override strategy via constructor arg if needed.
    """

    def __init__(self, strategy: str = None):
        logger.info("Initialising Embedder ...")
        self.strategy = strategy or settings.embedding_strategy

        cfg = settings.image_embedding if self.strategy == "multimodal" else settings.text_embedding

        self.model_name           = cfg.MODEL_NAME
        self.vector_dim           = cfg.VECTOR_DIM
        self.normalize            = cfg.NORMALIZE
        self.batch_size           = cfg.BATCH_SIZE
        self._cache_dir           = cfg.CACHE_DIR
        self._query_instruction   = getattr(cfg, "QUERY_INSTRUCTION",    "")
        self._passage_instruction = getattr(cfg, "PASSAGE_INSTRUCTION",  "")

        logger.info("Strategy           : %s", self.strategy)
        logger.info("Model              : %s", self.model_name)
        logger.info("Vector dim         : %d", self.vector_dim)
        logger.info("Normalize          : %s", self.normalize)
        logger.info("Batch size         : %d", self.batch_size)
        logger.info("Cache dir          : %s", self._cache_dir)
        logger.info("Query instruction  : '%s'", self._query_instruction)
        logger.info("Passage instruction: '%s'", self._passage_instruction)

        # Point HuggingFace to project's models/ folder
        os.environ["TRANSFORMERS_CACHE"]         = self._cache_dir
        os.environ["HF_HOME"]                    = self._cache_dir
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = self._cache_dir
        logger.debug("HuggingFace env vars set to: %s", self._cache_dir)

        logger.info("Loading SentenceTransformer model ...")
        t0 = time.time()
        try:
            self.model = SentenceTransformer(self.model_name, cache_folder=self._cache_dir)
        except Exception as e:
            logger.error("Failed to load model '%s': %s", self.model_name, e)
            raise
        logger.info("Model loaded in %.2fs", time.time() - t0)

    # ── Public API ────────────────────────────────────────────

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Adds to each chunk:
            chunk["embed_text"] → passage string that was embedded
            chunk["vector"]     → List[float]
        """
        if not chunks:
            logger.warning("embed_chunks called with empty list — nothing to do.")
            return chunks

        logger.info("Building embed_text for %d chunks ...", len(chunks))

        # Build passage strings (passage_instruction="" for mxbai)
        empty_count = 0
        for i, chunk in enumerate(chunks):
            chunk["embed_text"] = _build_embed_text(chunk, self._passage_instruction)
            if chunk["embed_text"] == "empty":
                empty_count += 1
                logger.warning("Chunk #%d has empty embed_text | metadata: %s",
                               i, chunk.get("metadata", {}))

        if empty_count:
            logger.warning("%d / %d chunks produced empty embed_text", empty_count, len(chunks))
        else:
            logger.debug("All %d chunks have non-empty embed_text", len(chunks))

        logger.info("Encoding %d chunks with strategy='%s' ...", len(chunks), self.strategy)
        t0 = time.time()

        if self.strategy == "multimodal":
            self._embed_multimodal(chunks)
        else:
            self._embed_text_only(chunks)

        elapsed = time.time() - t0
        logger.info("Encoding done in %.2fs  (%.1f chunks/sec)",
                    elapsed, len(chunks) / elapsed if elapsed > 0 else 0)

        # Sanity-check: verify all chunks got a vector
        missing = [i for i, c in enumerate(chunks) if not c.get("vector")]
        if missing:
            logger.error("Chunks missing vectors after encoding: indices %s", missing)
        else:
            logger.debug("Vector shape check — dim=%d, sample norm=%.4f",
                         len(chunks[0]["vector"]),
                         float(np.linalg.norm(chunks[0]["vector"])))

        logger.info("embed_chunks complete. %d chunks embedded.", len(chunks))
        return chunks

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a user query.
        mxbai: prepends QUERY_INSTRUCTION to improve retrieval.
        """
        if not query or not query.strip():
            logger.warning("embed_query called with empty query string!")

        prefixed = f"{self._query_instruction}{query}" if self._query_instruction else query
        logger.debug("embed_query input  : '%s'", query[:80])
        logger.debug("embed_query prefixed: '%s'", prefixed[:120])

        t0  = time.time()
        vec = self.model.encode(
            [prefixed],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        logger.debug("Query encoded in %.3fs | norm=%.4f",
                     time.time() - t0, float(np.linalg.norm(vec[0])))

        return vec[0].tolist()

    # ── Internal ──────────────────────────────────────────────

    def _embed_text_only(self, chunks: List[Dict]):
        logger.debug("_embed_text_only: encoding %d texts in batches of %d",
                     len(chunks), self.batch_size)

        texts   = [c["embed_text"] for c in chunks]
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            chunk["vector"] = vec.tolist()
            if i < 3 or i == len(chunks) - 1:   # log first 3 + last
                logger.debug("  chunk #%d | norm=%.4f | embed_text='%s'",
                             i, float(np.linalg.norm(vec)),
                             chunk["embed_text"][:60].replace("\n", " ↵ "))

    def _embed_multimodal(self, chunks: List[Dict]):
        """
        CLIP strategy: fuse text vector + image vectors by averaging.
        """
        from PIL import Image as PILImage

        text_only_count   = 0
        fused_count       = 0
        image_error_count = 0

        for i, chunk in enumerate(chunks):
            logger.debug("Multimodal chunk #%d | image_paths=%s",
                         i, chunk.get("image_paths", []))

            text_vec = self.model.encode(
                chunk["embed_text"],
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )

            valid_imgs = [p for p in (chunk.get("image_paths") or []) if Path(p).exists()]
            missing_imgs = [p for p in (chunk.get("image_paths") or []) if not Path(p).exists()]

            if missing_imgs:
                logger.warning("Chunk #%d — %d image path(s) not found on disk: %s",
                               i, len(missing_imgs), missing_imgs)

            if valid_imgs:
                img_vecs = []
                for img_path in valid_imgs:
                    try:
                        img = PILImage.open(img_path).convert("RGB")
                        iv  = self.model.encode(
                            img,
                            normalize_embeddings=self.normalize,
                            convert_to_numpy=True,
                        )
                        img_vecs.append(iv)
                        logger.debug("  Image encoded: %s | norm=%.4f",
                                     img_path, float(np.linalg.norm(iv)))
                    except Exception as e:
                        image_error_count += 1
                        logger.error("  Image encoding failed (%s): %s", img_path, e)

                if img_vecs:
                    all_vecs = np.stack([text_vec] + img_vecs)
                    fused    = np.mean(all_vecs, axis=0)
                    norm     = np.linalg.norm(fused)
                    chunk["vector"] = (fused / norm if norm > 0 else fused).tolist()
                    fused_count += 1
                    logger.debug("  Fused %d image vec(s) + text vec | final norm=%.4f",
                                 len(img_vecs), float(np.linalg.norm(chunk["vector"])))
                    continue

            # Fallback: text only
            chunk["vector"] = text_vec.tolist()
            text_only_count += 1

        logger.info("Multimodal summary — fused: %d | text-only: %d | image errors: %d",
                    fused_count, text_only_count, image_error_count)
        
if __name__ == "__main__":
    # Quick test
    embedder = Embedder()
    test_chunks = [
        {"text": "This is a test chunk with no images.", "metadata": {"id": 1}},
        {"text": "This chunk has an image.", "image_caption": "A cat on a sofa.", "metadata": {"id": 2}},
        {"text": "", "image_caption": "An empty text chunk with an image.", "metadata": {"id": 3}},
    ]
    embedded_chunks = embedder.embed_chunks(test_chunks)
    for c in embedded_chunks:
        logger.info("Chunk ID %s | vector norm=%.4f", c["metadata"]["id"], float(np.linalg.norm(c["vector"])))  