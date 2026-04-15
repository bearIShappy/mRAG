# """
# embedder.py
# -----------
# Converts chunks into dense vectors.

# Strategies (set EMBEDDING_STRATEGY in config/settings.py):
#   "text"       → mixedbread-ai/mxbai-embed-large-v1  (1024-dim)
#   "multimodal" → clip-ViT-B-32                        (512-dim, text + image)

# mxbai prefix rules
# ------------------
#   Passages (index time) → NO prefix  (PASSAGE_INSTRUCTION = "")
#   Queries  (query time) → prepend QUERY_INSTRUCTION

# Usage:
#     from backend.core.embedder import Embedder
#     embedder = Embedder()
#     chunks   = embedder.embed_chunks(chunks)
#     q_vec    = embedder.embed_query("What is GMMDC?")
# """

# import json
# import os
# import time
# import logging
# import numpy as np
# from pathlib import Path
# from typing import List, Dict

# from sentence_transformers import SentenceTransformer

# from src.backend.config.settings import settings

# # CLIP's text encoder hard-truncates at 77 tokens (BPE).
# # ~4 chars/token on average → 300 chars is a safe ceiling that
# # keeps the most semantically dense content (the opening sentences)
# # while avoiding silent mid-word cuts.
# # Increase to 350 if you find the tail content matters for your queries.
# _CLIP_MAX_CHARS = 300


# # ─────────────────────────────────────────────────────────────
# # Logger setup
# # ─────────────────────────────────────────────────────────────

# logger = logging.getLogger("embedder")

# if not logger.handlers:
#     _handler = logging.StreamHandler()
#     _handler.setFormatter(logging.Formatter(
#         "[%(asctime)s] [%(levelname)s] [Embedder] %(message)s",
#         datefmt="%H:%M:%S"
#     ))
#     logger.addHandler(_handler)

# logger.setLevel(logging.DEBUG)   # flip to INFO in production


# # ─────────────────────────────────────────────────────────────
# # Passage text builder
# # ─────────────────────────────────────────────────────────────

# def _build_embed_text(chunk: Dict, passage_instruction: str = "") -> str:
#     """
#     Combine paragraph text + image caption + image description
#     into the string that gets embedded at index time.

#     passage_instruction is prepended when the model requires it
#     (e.g. E5 uses "passage: "). mxbai leaves it empty.
#     """
#     parts = []

#     text = (chunk.get("text") or "").strip()
#     if text:
#         parts.append(text)
#     else:
#         logger.debug("Chunk has no 'text' field — will produce partial embed_text")

#     caption = (chunk.get("image_caption") or "").strip()
#     if caption and caption != text:
#         parts.append(f"[Image Caption]: {caption}")
#         logger.debug("Added image caption: '%s'", caption[:60])

#     descriptions = chunk.get("image_description") or []
#     if descriptions:
#         joined = " | ".join(d.strip() for d in descriptions if d.strip())
#         if joined:
#             parts.append(f"[Image Context]: {joined}")
#             logger.debug("Added image description (%d points): '%s'", len(descriptions), joined[:60])

#     raw = "\n".join(parts) if parts else "empty"

#     if raw == "empty":
#         logger.warning("Chunk produced empty embed_text — metadata: %s",
#                        chunk.get("metadata", {}))

#     result = f"{passage_instruction}{raw}" if passage_instruction else raw

#     # ── CLIP truncation ───────────────────────────────────────
#     # CLIP's text encoder silently ignores anything beyond token 77
#     # (~300 chars). We truncate here so what gets embedded matches
#     # what we stored, and we log a warning so large chunks are visible.
#     from src.backend.config.settings import settings as _s
#     if getattr(_s, "embedding_strategy", "") == "multimodal" and len(result) > _CLIP_MAX_CHARS:
#         logger.warning(
#             "embed_text truncated from %d → %d chars for CLIP (para_idx=%s). "
#             "Consider splitting this chunk upstream in chunk.py.",
#             len(result), _CLIP_MAX_CHARS,
#             chunk.get("metadata", {}).get("paragraph_index", "?"),
#         )
#         result = result[:_CLIP_MAX_CHARS]
#     # ─────────────────────────────────────────────────────────

#     logger.debug("embed_text built (%d chars): '%s'", len(result), result[:80].replace("\n", " ↵ "))
#     return result


# # ─────────────────────────────────────────────────────────────
# # Embedder
# # ─────────────────────────────────────────────────────────────

# class Embedder:
#     """
#     Reads all config from settings.py.
#     Override strategy via constructor arg if needed.
#     """

#     def __init__(self, strategy: str = None):
#         logger.info("Initialising Embedder ...")
#         self.strategy = strategy or settings.embedding_strategy

#         cfg = settings.image_embedding if self.strategy == "multimodal" else settings.text_embedding

#         self.model_name           = cfg.MODEL_NAME
#         self.vector_dim           = cfg.VECTOR_DIM
#         self.normalize            = cfg.NORMALIZE
#         self.batch_size           = cfg.BATCH_SIZE
#         self._cache_dir           = cfg.CACHE_DIR
#         self._query_instruction   = getattr(cfg, "QUERY_INSTRUCTION",    "")
#         self._passage_instruction = getattr(cfg, "PASSAGE_INSTRUCTION",  "")

#         logger.info("Strategy           : %s", self.strategy)
#         logger.info("Model              : %s", self.model_name)
#         logger.info("Vector dim         : %d", self.vector_dim)
#         logger.info("Normalize          : %s", self.normalize)
#         logger.info("Batch size         : %d", self.batch_size)
#         logger.info("Cache dir          : %s", self._cache_dir)
#         logger.info("Query instruction  : '%s'", self._query_instruction)
#         logger.info("Passage instruction: '%s'", self._passage_instruction)

#         # Point HuggingFace to project's models/ folder
#         os.environ["TRANSFORMERS_CACHE"]         = self._cache_dir
#         os.environ["HF_HOME"]                    = self._cache_dir
#         os.environ["SENTENCE_TRANSFORMERS_HOME"] = self._cache_dir
#         logger.debug("HuggingFace env vars set to: %s", self._cache_dir)

#         logger.info("Loading SentenceTransformer model ...")
#         t0 = time.time()
#         try:
#             self.model = SentenceTransformer(self.model_name, cache_folder=self._cache_dir)
#         except Exception as e:
#             logger.error("Failed to load model '%s': %s", self.model_name, e)
#             raise
#         logger.info("Model loaded in %.2fs", time.time() - t0)

#     # ── Public API ────────────────────────────────────────────

#     def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
#         """
#         Adds to each chunk:
#             chunk["embed_text"] → passage string that was embedded
#             chunk["vector"]     → List[float]
#         """
#         if not chunks:
#             logger.warning("embed_chunks called with empty list — nothing to do.")
#             return chunks

#         logger.info("Building embed_text for %d chunks ...", len(chunks))

#         # Build passage strings (passage_instruction="" for mxbai)
#         empty_count = 0
#         for i, chunk in enumerate(chunks):
#             chunk["embed_text"] = _build_embed_text(chunk, self._passage_instruction)
#             if chunk["embed_text"] == "empty":
#                 empty_count += 1
#                 logger.warning("Chunk #%d has empty embed_text | metadata: %s",
#                                i, chunk.get("metadata", {}))

#         if empty_count:
#             logger.warning("%d / %d chunks produced empty embed_text", empty_count, len(chunks))
#         else:
#             logger.debug("All %d chunks have non-empty embed_text", len(chunks))

#         logger.info("Encoding %d chunks with strategy='%s' ...", len(chunks), self.strategy)
#         t0 = time.time()

#         if self.strategy == "multimodal":
#             self._embed_multimodal(chunks)
#         else:
#             self._embed_text_only(chunks)

#         elapsed = time.time() - t0
#         logger.info("Encoding done in %.2fs  (%.1f chunks/sec)",
#                     elapsed, len(chunks) / elapsed if elapsed > 0 else 0)

#         # Sanity-check: verify all chunks got a vector
#         missing = [i for i, c in enumerate(chunks) if not c.get("vector")]
#         if missing:
#             logger.error("Chunks missing vectors after encoding: indices %s", missing)
#         else:
#             logger.debug("Vector shape check — dim=%d, sample norm=%.4f",
#                          len(chunks[0]["vector"]),
#                          float(np.linalg.norm(chunks[0]["vector"])))

#         logger.info("embed_chunks complete. %d chunks embedded.", len(chunks))
#         return chunks

#     def embed_query(self, query: str) -> List[float]:
#         """
#         Embed a user query.
#         mxbai: prepends QUERY_INSTRUCTION to improve retrieval.
#         """
#         if not query or not query.strip():
#             logger.warning("embed_query called with empty query string!")

#         prefixed = f"{self._query_instruction}{query}" if self._query_instruction else query
#         logger.debug("embed_query input  : '%s'", query[:80])
#         logger.debug("embed_query prefixed: '%s'", prefixed[:120])

#         t0  = time.time()
#         vec = self.model.encode(
#             [prefixed],
#             normalize_embeddings=self.normalize,
#             convert_to_numpy=True,
#         )
#         logger.debug("Query encoded in %.3fs | norm=%.4f",
#                      time.time() - t0, float(np.linalg.norm(vec[0])))

#         return vec[0].tolist()

#     # ── Internal ──────────────────────────────────────────────

#     def _embed_text_only(self, chunks: List[Dict]):
#         logger.debug("_embed_text_only: encoding %d texts in batches of %d",
#                      len(chunks), self.batch_size)

#         texts   = [c["embed_text"] for c in chunks]
#         vectors = self.model.encode(
#             texts,
#             batch_size=self.batch_size,
#             show_progress_bar=True,
#             convert_to_numpy=True,
#             normalize_embeddings=self.normalize,
#         )

#         for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
#             chunk["vector"] = vec.tolist()
#             if i < 3 or i == len(chunks) - 1:   # log first 3 + last
#                 logger.debug("  chunk #%d | norm=%.4f | embed_text='%s'",
#                              i, float(np.linalg.norm(vec)),
#                              chunk["embed_text"][:60].replace("\n", " ↵ "))

#     def _embed_multimodal(self, chunks: List[Dict]):
#         """
#         CLIP strategy: fuse text vector + image vectors by averaging.
#         """
#         from PIL import Image as PILImage

#         text_only_count   = 0
#         fused_count       = 0
#         image_error_count = 0

#         for i, chunk in enumerate(chunks):
#             logger.debug("Multimodal chunk #%d | image_paths=%s",
#                          i, chunk.get("image_paths", []))

#             text_vec = self.model.encode(
#                 chunk["embed_text"],
#                 normalize_embeddings=self.normalize,
#                 convert_to_numpy=True,
#             )

#             valid_imgs = [p for p in (chunk.get("image_paths") or []) if Path(p).exists()]
#             missing_imgs = [p for p in (chunk.get("image_paths") or []) if not Path(p).exists()]

#             if missing_imgs:
#                 logger.warning("Chunk #%d — %d image path(s) not found on disk: %s",
#                                i, len(missing_imgs), missing_imgs)

#             if valid_imgs:
#                 img_vecs = []
#                 for img_path in valid_imgs:
#                     try:
#                         img = PILImage.open(img_path).convert("RGB")
#                         iv  = self.model.encode(
#                             img,
#                             normalize_embeddings=self.normalize,
#                             convert_to_numpy=True,
#                         )
#                         img_vecs.append(iv)
#                         logger.debug("  Image encoded: %s | norm=%.4f",
#                                      img_path, float(np.linalg.norm(iv)))
#                     except Exception as e:
#                         image_error_count += 1
#                         logger.error("  Image encoding failed (%s): %s", img_path, e)

#                 if img_vecs:
#                     all_vecs = np.stack([text_vec] + img_vecs)
#                     fused    = np.mean(all_vecs, axis=0)
#                     norm     = np.linalg.norm(fused)
#                     chunk["vector"] = (fused / norm if norm > 0 else fused).tolist()
#                     fused_count += 1
#                     logger.debug("  Fused %d image vec(s) + text vec | final norm=%.4f",
#                                  len(img_vecs), float(np.linalg.norm(chunk["vector"])))
#                     continue

#             # Fallback: text only
#             chunk["vector"] = text_vec.tolist()
#             text_only_count += 1

#         logger.info("Multimodal summary — fused: %d | text-only: %d | image errors: %d",
#                     fused_count, text_only_count, image_error_count)
        
# if __name__ == "__main__":
#     # Quick test
#     embedder = Embedder()
    
#     # Navigate up 4 levels to project root, then into output/chunks
#     chunks_path = Path(__file__).parents[3] / "output" / "chunks" / "chunks.json"

#     with open(chunks_path, "r", encoding="utf-8") as f:
#         test_chunks = json.load(f)
#     # [
#     #     {"text": "This is a test chunk with no images.", "metadata": {"id": 1}},
#     #     {"text": "This chunk has an image.", "image_caption": "A cat on a sofa.", "metadata": {"id": 2}},
#     #     {"text": "", "image_caption": "An empty text chunk with an image.", "metadata": {"id": 3}},
#     # ]
#     embedded_chunks = embedder.embed_chunks(test_chunks)
#     for c in embedded_chunks:
#         logger.info("Chunk ID %s | vector norm=%.4f", c["metadata"]["paragraph_index"], float(np.linalg.norm(c["vector"])))
"""
embedder.py
-----------
Converts chunks into dense vectors using the RIGHT model per chunk type.

Routing logic
-------------
  text-only chunks  (no image_paths)  → mxbai-embed-large-v1  (1024-dim)
  image+text chunks (has image_paths) → clip-ViT-B-32          (512-dim, fused)

Why split?
  CLIP was trained on image-text pairs — great for multimodal fusion,
  poor for semantic text retrieval. mxbai is purpose-built for retrieval.

mxbai prefix rules
------------------
  Passages (index time) → NO prefix  (PASSAGE_INSTRUCTION = "")
  Queries  (query time) → prepend QUERY_INSTRUCTION

Each embedded chunk gets:
    chunk["vector"]          → List[float]
    chunk["embed_text"]      → the string that was embedded
    chunk["embedding_model"] → "text" | "multimodal"  (for VectorStore routing)

Usage:
    embedder = Embedder()
    chunks   = embedder.embed_chunks(chunks)
    q_vecs   = embedder.embed_query("What is GMMDC?")
    # returns {"text": [...], "multimodal": [...]}
"""

import json
import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer

from src.backend.config.settings import settings

# CLIP's text encoder hard-truncates at 77 BPE tokens (~300 chars).
# We truncate before encoding so stored embed_text matches what was embedded.
_CLIP_MAX_CHARS = 300

# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger("embedder")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [Embedder] %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(_handler)

logger.setLevel(logging.DEBUG)


# ─────────────────────────────────────────────────────────────
# Passage text builder
# ─────────────────────────────────────────────────────────────

def _build_embed_text(
    chunk: Dict,
    passage_instruction: str = "",
    clip_truncate: bool = False,
) -> str:
    """
    Combine paragraph text + image caption + image description
    into the string that gets embedded at index time.
    clip_truncate=True enforces _CLIP_MAX_CHARS limit.
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
            logger.debug("Added image description (%d points): '%s'",
                         len(descriptions), joined[:60])

    raw    = "\n".join(parts) if parts else "empty"
    result = f"{passage_instruction}{raw}" if passage_instruction else raw

    if raw == "empty":
        logger.warning("Chunk produced empty embed_text — metadata: %s",
                       chunk.get("metadata", {}))

    # ── CLIP truncation ───────────────────────────────────────
    if clip_truncate and len(result) > _CLIP_MAX_CHARS:
        logger.warning(
            "embed_text truncated %d → %d chars for CLIP (para_idx=%s). "
            "Consider splitting upstream in chunk.py.",
            len(result), _CLIP_MAX_CHARS,
            chunk.get("metadata", {}).get("paragraph_index", "?"),
        )
        result = result[:_CLIP_MAX_CHARS]
    # ─────────────────────────────────────────────────────────

    logger.debug("embed_text built (%d chars): '%s'",
                 len(result), result[:80].replace("\n", " ↵ "))
    return result


# ─────────────────────────────────────────────────────────────
# Embedder
# ─────────────────────────────────────────────────────────────

class Embedder:
    """
    Loads BOTH models on init and routes each chunk to the correct one:
      - text-only  → mxbai  (self.text_model)
      - image+text → CLIP   (self.clip_model)

    embed_query() returns vectors for BOTH models so VectorStore
    can search both collections simultaneously.
    """

    def __init__(self):
        logger.info("Initialising Embedder (dual-model) ...")

        text_cfg = settings.text_embedding
        clip_cfg = settings.image_embedding

        # ── Text model config ─────────────────────────────────
        self._text_model_name     = text_cfg.MODEL_NAME
        self._text_dim            = text_cfg.VECTOR_DIM
        self._text_normalize      = text_cfg.NORMALIZE
        self._text_batch_size     = text_cfg.BATCH_SIZE
        self._text_cache_dir      = text_cfg.CACHE_DIR
        self._query_instruction   = getattr(text_cfg, "QUERY_INSTRUCTION",   "")
        self._passage_instruction = getattr(text_cfg, "PASSAGE_INSTRUCTION", "")

        # ── CLIP model config ─────────────────────────────────
        self._clip_model_name  = clip_cfg.MODEL_NAME
        self._clip_dim         = clip_cfg.VECTOR_DIM
        self._clip_normalize   = clip_cfg.NORMALIZE
        self._clip_batch_size  = clip_cfg.BATCH_SIZE
        self._clip_cache_dir   = clip_cfg.CACHE_DIR

        logger.info("Text model : %s  (dim=%d)", self._text_model_name, self._text_dim)
        logger.info("CLIP model : %s  (dim=%d)", self._clip_model_name, self._clip_dim)

        os.environ["TRANSFORMERS_CACHE"]         = self._text_cache_dir
        os.environ["HF_HOME"]                    = self._text_cache_dir
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = self._text_cache_dir

        logger.info("Loading text model (mxbai) ...")
        t0 = time.time()
        self.text_model = SentenceTransformer(
            self._text_model_name, cache_folder=self._text_cache_dir
        )
        logger.info("Text model loaded in %.2fs", time.time() - t0)

        logger.info("Loading CLIP model ...")
        t0 = time.time()
        self.clip_model = SentenceTransformer(
            self._clip_model_name, cache_folder=self._clip_cache_dir
        )
        logger.info("CLIP model loaded in %.2fs", time.time() - t0)

    # ── Public API ────────────────────────────────────────────

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Routes each chunk to the correct model based on image_paths.

        Adds to each chunk:
            chunk["embed_text"]      → passage string embedded
            chunk["vector"]          → List[float]
            chunk["embedding_model"] → "text" | "multimodal"
        """
        if not chunks:
            logger.warning("embed_chunks called with empty list.")
            return chunks

        text_chunks = [c for c in chunks if not c.get("image_paths")]
        clip_chunks = [c for c in chunks if c.get("image_paths")]

        logger.info(
            "Routing: %d text-only → mxbai | %d image+text → CLIP",
            len(text_chunks), len(clip_chunks)
        )

        # Build embed_text for each group
        for chunk in text_chunks:
            chunk["embed_text"]      = _build_embed_text(chunk, self._passage_instruction)
            chunk["embedding_model"] = "text"

        for chunk in clip_chunks:
            chunk["embed_text"]      = _build_embed_text(chunk, clip_truncate=True)
            chunk["embedding_model"] = "multimodal"

        # Encode
        t0 = time.time()

        if text_chunks:
            logger.info("Encoding %d text chunks with mxbai ...", len(text_chunks))
            self._encode_text(text_chunks)

        if clip_chunks:
            logger.info("Encoding %d image+text chunks with CLIP ...", len(clip_chunks))
            self._encode_multimodal(clip_chunks)

        elapsed = time.time() - t0
        logger.info("Encoding done in %.2fs (%.1f chunks/sec)",
                    elapsed, len(chunks) / elapsed if elapsed > 0 else 0)

        # Sanity check
        missing = [i for i, c in enumerate(chunks) if not c.get("vector")]
        if missing:
            logger.error("Chunks missing vectors after encoding: %s", missing)
        else:
            logger.debug("Sample norm=%.4f",
                         float(np.linalg.norm(chunks[0]["vector"])))

        logger.info("embed_chunks complete — %d text | %d multimodal",
                    len(text_chunks), len(clip_chunks))
        return chunks

    def embed_query(self, query: str) -> Dict[str, List[float]]:
        """
        Embed query with BOTH models.

        Returns:
            {
                "text":       [...],   # 1024-dim mxbai
                "multimodal": [...],   # 512-dim  CLIP
            }

        VectorStore.similarity_search() uses both to search
        both collections and merges the results.
        """
        if not query or not query.strip():
            logger.warning("embed_query called with empty string!")

        # mxbai: prepend query instruction
        text_input = f"{self._query_instruction}{query}" if self._query_instruction else query
        logger.debug("embed_query [mxbai] : '%s'", text_input[:100])

        t0 = time.time()
        text_vec = self.text_model.encode(
            [text_input],
            normalize_embeddings=self._text_normalize,
            convert_to_numpy=True,
        )[0]
        logger.debug("mxbai query encoded in %.3fs | norm=%.4f",
                     time.time() - t0, float(np.linalg.norm(text_vec)))

        # CLIP: truncate query to 300 chars
        clip_input = query[:_CLIP_MAX_CHARS]
        logger.debug("embed_query [CLIP]  : '%s'", clip_input[:100])

        t0 = time.time()
        clip_vec = self.clip_model.encode(
            [clip_input],
            normalize_embeddings=self._clip_normalize,
            convert_to_numpy=True,
        )[0]
        logger.debug("CLIP  query encoded in %.3fs | norm=%.4f",
                     time.time() - t0, float(np.linalg.norm(clip_vec)))

        return {
            "text":       text_vec.tolist(),
            "multimodal": clip_vec.tolist(),
        }

    # ── Internal ──────────────────────────────────────────────

    def _encode_text(self, chunks: List[Dict]):
        """Batch-encode text-only chunks with mxbai."""
        texts   = [c["embed_text"] for c in chunks]
        vectors = self.text_model.encode(
            texts,
            batch_size=self._text_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self._text_normalize,
        )
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            chunk["vector"] = vec.tolist()
            if i < 3 or i == len(chunks) - 1:
                logger.debug("  [mxbai] chunk #%d | norm=%.4f | '%s'",
                             i, float(np.linalg.norm(vec)),
                             chunk["embed_text"][:60].replace("\n", " ↵ "))

    def _encode_multimodal(self, chunks: List[Dict]):
        """CLIP: fuse text vector + image vectors per chunk."""
        from PIL import Image as PILImage

        fused_count = text_fallback_count = image_error_count = 0

        for i, chunk in enumerate(chunks):
            logger.debug("CLIP chunk #%d | image_paths=%s",
                         i, chunk.get("image_paths", []))

            text_vec = self.clip_model.encode(
                chunk["embed_text"],
                normalize_embeddings=self._clip_normalize,
                convert_to_numpy=True,
            )

            valid_imgs   = [p for p in (chunk.get("image_paths") or []) if Path(p).exists()]
            missing_imgs = [p for p in (chunk.get("image_paths") or []) if not Path(p).exists()]

            if missing_imgs:
                logger.warning("Chunk #%d — %d image(s) not on disk: %s",
                               i, len(missing_imgs), missing_imgs)

            if valid_imgs:
                img_vecs = []
                for img_path in valid_imgs:
                    try:
                        img = PILImage.open(img_path).convert("RGB")
                        iv  = self.clip_model.encode(
                            img,
                            normalize_embeddings=self._clip_normalize,
                            convert_to_numpy=True,
                        )
                        img_vecs.append(iv)
                        logger.debug("  Image encoded: %s | norm=%.4f",
                                     img_path, float(np.linalg.norm(iv)))
                    except Exception as e:
                        image_error_count += 1
                        logger.error("  Image encoding failed (%s): %s", img_path, e)

                if img_vecs:
                    all_vecs        = np.stack([text_vec] + img_vecs)
                    fused           = np.mean(all_vecs, axis=0)
                    norm            = np.linalg.norm(fused)
                    chunk["vector"] = (fused / norm if norm > 0 else fused).tolist()
                    fused_count    += 1
                    logger.debug("  Fused %d image(s) + text | norm=%.4f",
                                 len(img_vecs), float(np.linalg.norm(chunk["vector"])))
                    continue

            # Fallback: text vec only (image paths exist but files missing)
            chunk["vector"]       = text_vec.tolist()
            text_fallback_count  += 1

        logger.info("CLIP summary — fused: %d | text-fallback: %d | image errors: %d",
                    fused_count, text_fallback_count, image_error_count)


# ─────────────────────────────────────────────────────────────
# CLI — quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    embedder    = Embedder()
    chunks_path = Path(__file__).parents[3] / "output" / "chunks" / "chunks.json"

    with open(chunks_path, "r", encoding="utf-8") as f:
        test_chunks = json.load(f)

    embedded    = embedder.embed_chunks(test_chunks)
    text_chunks = [c for c in embedded if c["embedding_model"] == "text"]
    clip_chunks = [c for c in embedded if c["embedding_model"] == "multimodal"]

    logger.info("── Text chunks (mxbai) ──")
    for c in text_chunks[:3]:
        logger.info("  para_idx=%s | norm=%.4f | dim=%d | '%s'",
                    c["metadata"]["paragraph_index"],
                    float(np.linalg.norm(c["vector"])),
                    len(c["vector"]),
                    c["text"][:60])

    logger.info("── Multimodal chunks (CLIP) ──")
    for c in clip_chunks[:3]:
        logger.info("  para_idx=%s | norm=%.4f | dim=%d | '%s'",
                    c["metadata"]["paragraph_index"],
                    float(np.linalg.norm(c["vector"])),
                    len(c["vector"]),
                    c["text"][:60])
        

# Now I'll update all three files. The plan:

# embedder.py — load both models, route text-only chunks → mxbai, image+text chunks → CLIP, tag each chunk with embedding_model so vector_store knows which collection to use
# vector_store.py — two collections (text_chunks 1024-dim, image_chunks 512-dim), quality filter before upsert, merged search
# rag_qa.py — query both collections, merge + rerank results