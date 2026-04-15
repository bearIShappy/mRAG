# """
# VectorStore
# -----------
# Stores embedded chunks into Qdrant (in-memory or server mode)
# and exposes a similarity_search() method for retrieval.

# Usage:
#     from vector_store import VectorStore

#     store = VectorStore(collection_name="my_docs")
#     store.upsert(embedded_chunks)

#     results = store.similarity_search(query_vector, top_k=5)
# """

# import uuid
# import json, sys
# from src.backend.core.embedder import Embedder
# from typing import List, Dict, Optional

# from qdrant_client import QdrantClient
# from qdrant_client.models import (
#     Distance,
#     VectorParams,
#     PointStruct,
#     Filter,
#     FieldCondition,
#     MatchValue
# )


# # ─────────────────────────────────────────────────────────────
# # VectorStore
# # ─────────────────────────────────────────────────────────────

# class VectorStore:
#     """
#     Thin wrapper around QdrantClient.

#     Parameters
#     ----------
#     collection_name : str
#         Name of the Qdrant collection.
#     vector_dim : int
#         Embedding dimension. If None, inferred automatically from the
#         first chunk passed to upsert(). Collection is recreated if the
#         dim ever changes (e.g. you swap embedding models).
#     host / port : str / int
#         Set to your Qdrant server address.
#         Leave as None to use fast in-memory mode (no server needed).
#     distance : Distance
#         Similarity metric. COSINE works best with normalised embeddings.
#     """

#     def __init__(
#         self,
#         collection_name: str      = "rag_chunks",
#         vector_dim:      Optional[int] = None,       # ← None = infer from data
#         host:            Optional[str] = None,
#         port:            int      = 6333,
#         distance:        Distance = Distance.COSINE,
#     ):
#         self.collection_name = collection_name
#         self.vector_dim      = vector_dim             # may be None until upsert
#         self.distance        = distance

#         if host:
#             print(f"[VectorStore] Connecting to Qdrant at {host}:{port}")
#             self.client = QdrantClient(host=host, port=port)
#         else:
#             print("[VectorStore] Using in-memory Qdrant (no server needed)")
#             self.client = QdrantClient(":memory:")

#         # Only create collection now if dim is already known
#         if self.vector_dim is not None:
#             self._ensure_collection(self.vector_dim)

#     # ── Collection setup ──────────────────────────────────────

#     def _ensure_collection(self, dim: int):
#         """
#         Create the collection with `dim` dimensions.
#         If the collection already exists with a DIFFERENT dim,
#         it is deleted and recreated automatically.
#         """
#         existing = [c.name for c in self.client.get_collections().collections]

#         if self.collection_name in existing:
#             info         = self.client.get_collection(self.collection_name)
#             existing_dim = info.config.params.vectors.size

#             if existing_dim != dim:
#                 print(
#                     f"[VectorStore] Dimension mismatch "
#                     f"(collection={existing_dim}, embedder={dim}). "
#                     f"Recreating collection '{self.collection_name}'."
#                 )
#                 self.client.delete_collection(self.collection_name)
#             else:
#                 print(f"[VectorStore] Collection '{self.collection_name}' already exists (dim={dim}).")
#                 return

#         self.client.create_collection(
#             collection_name=self.collection_name,
#             vectors_config=VectorParams(size=dim, distance=self.distance),
#         )
#         print(f"[VectorStore] Collection '{self.collection_name}' created (dim={dim}).")

#     # ── Upsert ────────────────────────────────────────────────

#     def upsert(self, embedded_chunks: List[Dict], batch_size: int = 64):
#         """
#         Insert or update chunks in Qdrant.

#         Each chunk must already have an "embedding" key (from Embedder).
#         A stable UUID is generated from the chunk's embed_text so re-runs
#         are idempotent (same text → same UUID → overwrite, not duplicate).

#         Payload stored per point (searchable / filterable):
#             - text
#             - embed_text
#             - image_paths
#             - image_caption
#             - image_description
#             - metadata  (page_number, paragraph_index, filename, etc.)
#         """
#         if not embedded_chunks:
#             print("[VectorStore] Warning: nothing to upsert.")
#             return

#         # ── Infer dim from first valid chunk and (re)create collection ──
#         first_vector = None
#         for chunk in embedded_chunks:
#             v = chunk.get("vector")          # ← correct key from Embedder
#             if v:
#                 first_vector = v
#                 break

#         if first_vector is None:
#             print("[VectorStore] Error: no chunk has an 'embedding' key. Aborting upsert.")
#             return

#         inferred_dim = len(first_vector)
#         if self.vector_dim != inferred_dim:
#             self.vector_dim = inferred_dim
#         self._ensure_collection(self.vector_dim)   # auto-recreate if dim changed

#         # ── Build PointStructs ───────────────────────────────
#         points = []
#         for chunk in embedded_chunks:
#             vector = chunk.get("vector")         # ← 
#             if not vector:
#                 print(f"[VectorStore] Skipping chunk with no vector: {chunk.get('text','')[:40]}")
#                 continue

#             # Stable ID from embed_text content
#             point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.get("embed_text", str(chunk))))

#             payload = {
#                 "text":              chunk.get("text", ""),
#                 "embed_text":        chunk.get("embed_text", ""),
#                 "image_paths":       chunk.get("image_paths", []),
#                 "image_caption":     chunk.get("image_caption") or "",
#                 "image_description": chunk.get("image_description", []),
#                 "metadata":          chunk.get("metadata", {}),
#             }

#             points.append(PointStruct(id=point_id, vector=vector, payload=payload))

#         # ── Batch upsert ─────────────────────────────────────
#         total = len(points)
#         for i in range(0, total, batch_size):
#             batch = points[i : i + batch_size]
#             self.client.upsert(collection_name=self.collection_name, points=batch)
#             print(f"[VectorStore] Upserted {min(i + batch_size, total)}/{total} points ...", end="\r")

#         print(f"\n[VectorStore] Done. {total} points stored in '{self.collection_name}'.")

#     # ── Search ────────────────────────────────────────────────

#     def similarity_search(
#         self,
#         query_vector:    List[float],
#         top_k:           int            = 5,
#         score_threshold: Optional[float] = None,
#         filter_page:     Optional[int]   = None,
#     ) -> List[Dict]:
#         """
#         Return top_k most similar chunks to query_vector.

#         Parameters
#         ----------
#         query_vector     : output of Embedder.embed_query()
#         top_k            : number of results to return
#         score_threshold  : optional minimum similarity score (0–1)
#         filter_page      : optional page number filter

#         Returns
#         -------
#         List of dicts with keys:
#             id, score, text, embed_text, image_paths, image_caption,
#             image_description, metadata
#         """
#         search_filter = None
#         if filter_page is not None:
#             search_filter = Filter(
#                 must=[
#                     FieldCondition(
#                         key="metadata.page_number",
#                         match=MatchValue(value=filter_page),
#                     )
#                 ]
#             )

#         hits = self.client.query_points(
#             collection_name=self.collection_name,
#             query=query_vector,
#             limit=top_k,
#             score_threshold=score_threshold,
#             query_filter=search_filter,
#             with_payload=True,
#         ).points                        # unwrap QueryResponse → List[ScoredPoint]

#         results = []
#         for hit in hits:
#             p = hit.payload
#             results.append({
#                 "id":                hit.id,
#                 "score":             round(hit.score, 4),
#                 "text":              p.get("text", ""),
#                 "embed_text":        p.get("embed_text", ""),
#                 "image_paths":       p.get("image_paths", []),
#                 "image_caption":     p.get("image_caption", ""),
#                 "image_description": p.get("image_description", []),
#                 "metadata":          p.get("metadata", {}),
#             })

#         return results

#     # ── Info ──────────────────────────────────────────────────

#     def count(self) -> int:
#         return self.client.count(collection_name=self.collection_name).count

#     def info(self):
#         info = self.client.get_collection(self.collection_name)
#         print(f"[VectorStore] Collection : {self.collection_name}")
#         print(f"[VectorStore] Points     : {info.points_count}")
#         print(f"[VectorStore] Dim        : {info.config.params.vectors.size}")
#         print(f"[VectorStore] Distance   : {info.config.params.vectors.distance}")


# # ─────────────────────────────────────────────────────────────
# # CLI — quick smoke-test
# # ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":

#     input_file = sys.argv[1] if len(sys.argv) > 1 else "output\chunks\chunks.json"

#     with open(input_file, encoding="utf-8") as f:
#         chunks = json.load(f)

#     embedder = Embedder()
#     chunks   = embedder.embed_chunks(chunks)

#     store = VectorStore()                   # ← no vector_dim needed; inferred from data
#     store.upsert(chunks)
#     store.info()

#     # Quick test search
#     q_vec   = embedder.embed_query("What is this document about?")
#     results = store.similarity_search(q_vec, top_k=3)

#     print("\n── Top results ──")
#     for r in results:
#         print(f"  score={r['score']}  text={r['text'][:80]}")
"""
vector_store.py
---------------
Stores embedded chunks into TWO Qdrant collections — one per model:

    "text_chunks"   → mxbai vectors  (1024-dim)  text-only chunks
    "image_chunks"  → CLIP  vectors  (512-dim)   image+text chunks

Why two collections?
  Qdrant requires all vectors in a collection to have the same dimension.
  mxbai=1024, CLIP=512 — they cannot share a collection.

similarity_search() queries BOTH collections with their respective
query vectors and returns a single merged, score-sorted result list.

Usage:
    store = VectorStore()
    store.upsert(embedded_chunks)          # routes automatically by embedding_model

    query_vecs = embedder.embed_query("What is GMMDC?")
    results    = store.similarity_search(query_vecs, top_k=5)
"""

from logging import config
from logging import config
import uuid
import json
import sys
import logging
from typing import List, Dict, Optional
from src.backend.config.settings import QdrantConfig

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

from src.backend.core.embedder import Embedder

# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger("vector_store")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [VectorStore] %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(_handler)

logger.setLevel(logging.DEBUG)


# ─────────────────────────────────────────────────────────────
# Quality filter — keep this before storing in Qdrant
# ─────────────────────────────────────────────────────────────

def _is_low_quality(chunk: Dict, min_chars: int = 40) -> bool:
    """
    Returns True if a chunk should be excluded from the vector store.
    Catches: too-short text, high non-ASCII ratio (garbled OCR),
    high single-char token ratio (OCR fragmentation).
    """
    text = chunk.get("text", "")

    if len(text.strip()) < min_chars:
        return True

    # High non-ASCII → garbled OCR or encoding artefacts
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    if len(text) > 0 and (non_ascii / len(text)) > 0.15:
        return True

    # High single-char token ratio → OCR fragmentation
    words = text.split()
    if words and (sum(1 for w in words if len(w) == 1) / len(words)) > 0.3:
        return True

    return False


# ─────────────────────────────────────────────────────────────
# VectorStore
# ─────────────────────────────────────────────────────────────

class VectorStore:
    """
    Two-collection Qdrant store.

    Collections
    -----------
    text_collection   (default: "text_chunks")   — mxbai 1024-dim
    image_collection  (default: "image_chunks")  — CLIP   512-dim

    Parameters
    ----------
    host / port : str / int
        Leave as None for fast in-memory mode (no server needed).
    distance : Distance
        COSINE works best with normalised embeddings (both models normalize).
    """

    TEXT_DIM  = 1024   # mxbai-embed-large-v1
    CLIP_DIM  = 512    # clip-ViT-B-32

    def __init__(
        self,
        text_collection:  str           = "text_chunks",
        image_collection: str           = "image_chunks",
        host:             Optional[str] = QdrantConfig.HOST,
        port:             int           = QdrantConfig.PORT,
        distance:         Distance      = Distance.COSINE,
    ):
        self.text_collection  = text_collection
        self.image_collection = image_collection
        self.distance         = distance

        if host and not QdrantConfig.IN_MEMORY:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Connected to Qdrant Server at {host}:{port}")
        else:
            self.client = QdrantClient(":memory:")
            logger.warning("Using In-Memory Qdrant - Dashboard will NOT be available")

        self._ensure_collection(self.text_collection,  self.TEXT_DIM)
        self._ensure_collection(self.image_collection, self.CLIP_DIM)

    # ── Collection setup ──────────────────────────────────────

    def _ensure_collection(self, name: str, dim: int):
        existing = [c.name for c in self.client.get_collections().collections]

        if name in existing:
            info         = self.client.get_collection(name)
            existing_dim = info.config.params.vectors.size

            if existing_dim != dim:
                logger.warning(
                    "Dimension mismatch for '%s' (stored=%d, expected=%d). Recreating.",
                    name, existing_dim, dim
                )
                self.client.delete_collection(name)
            else:
                logger.info("Collection '%s' already exists (dim=%d).", name, dim)
                return

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=self.distance),
        )
        logger.info("Collection '%s' created (dim=%d).", name, dim)

    # ── Upsert ────────────────────────────────────────────────

    def upsert(self, embedded_chunks: List[Dict], batch_size: int = 64):
        """
        Route each chunk to the correct collection based on embedding_model tag:
            "text"       → text_collection  (mxbai, 1024-dim)
            "multimodal" → image_collection (CLIP,  512-dim)

        Applies quality filter before storing.
        A stable UUID from embed_text makes re-runs idempotent.
        """
        if not embedded_chunks:
            logger.warning("upsert called with empty list.")
            return

        # ── Quality filter ────────────────────────────────────
        clean   = [c for c in embedded_chunks if not _is_low_quality(c)]
        removed = len(embedded_chunks) - len(clean)
        if removed:
            logger.info("Quality filter removed %d/%d low-quality chunks.",
                        removed, len(embedded_chunks))

        # ── Split by model ────────────────────────────────────
        text_points  = []
        image_points = []

        for chunk in clean:
            vector = chunk.get("vector")
            if not vector:
                logger.warning("Skipping chunk with no vector: '%s'",
                               chunk.get("text", "")[:40])
                continue

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS,
                                      chunk.get("embed_text", str(chunk))))
            payload  = {
                "text":              chunk.get("text", ""),
                "embed_text":        chunk.get("embed_text", ""),
                "image_paths":       chunk.get("image_paths", []),
                "image_caption":     chunk.get("image_caption") or "",
                "image_description": chunk.get("image_description", []),
                "metadata":          chunk.get("metadata", {}),
                "embedding_model":   chunk.get("embedding_model", "text"),
            }

            point = PointStruct(id=point_id, vector=vector, payload=payload)

            if chunk.get("embedding_model") == "multimodal":
                image_points.append(point)
            else:
                text_points.append(point)

        # ── Batch upsert to each collection ───────────────────
        self._batch_upsert(self.text_collection,  text_points,  batch_size)
        self._batch_upsert(self.image_collection, image_points, batch_size)

        logger.info("Upsert complete — %d text points | %d image points",
                    len(text_points), len(image_points))

    def _batch_upsert(self, collection: str, points: List, batch_size: int):
        if not points:
            return
        total = len(points)
        for i in range(0, total, batch_size):
            batch = points[i: i + batch_size]
            self.client.upsert(collection_name=collection, points=batch)
            logger.debug("  Upserted %d/%d into '%s' ...",
                         min(i + batch_size, total), total, collection)
        logger.info("Stored %d points in '%s'.", total, collection)

    # ── Search ────────────────────────────────────────────────

    def similarity_search(
        self,
        query_vectors:   Dict[str, List[float]],
        top_k:           int            = 5,
        score_threshold: Optional[float] = None,
        filter_page:     Optional[int]   = None,
    ) -> List[Dict]:
        """
        Search BOTH collections using their respective query vectors,
        then merge and return top_k results sorted by score.

        Parameters
        ----------
        query_vectors : dict from embedder.embed_query()
            {"text": [...1024...], "multimodal": [...512...]}
        top_k         : total results to return (across both collections)
        score_threshold : minimum score to include
        filter_page   : optional page number filter

        Returns
        -------
        List of dicts sorted by score descending.
        """
        search_filter = None
        if filter_page is not None:
            search_filter = Filter(must=[
                FieldCondition(
                    key="metadata.page_number",
                    match=MatchValue(value=filter_page),
                )
            ])

        all_results = []

        # ── Search text collection (mxbai) ────────────────────
        text_vec = query_vectors.get("text")
        if text_vec:
            hits = self.client.query_points(
                collection_name=self.text_collection,
                query=text_vec,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
            ).points
            for hit in hits:
                all_results.append(self._hit_to_dict(hit, source="text"))
            logger.debug("Text collection returned %d hits.", len(hits))

        # ── Search image collection (CLIP) ────────────────────
        clip_vec = query_vectors.get("multimodal")
        if clip_vec:
            hits = self.client.query_points(
                collection_name=self.image_collection,
                query=clip_vec,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
            ).points
            for hit in hits:
                all_results.append(self._hit_to_dict(hit, source="multimodal"))
            logger.debug("Image collection returned %d hits.", len(hits))

        # ── Merge: sort by score, take top_k ─────────────────
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    @staticmethod
    def _hit_to_dict(hit, source: str) -> Dict:
        p = hit.payload
        return {
            "id":                hit.id,
            "score":             round(hit.score, 4),
            "source_collection": source,
            "text":              p.get("text", ""),
            "embed_text":        p.get("embed_text", ""),
            "image_paths":       p.get("image_paths", []),
            "image_caption":     p.get("image_caption", ""),
            "image_description": p.get("image_description", []),
            "metadata":          p.get("metadata", {}),
        }

    # ── Info ──────────────────────────────────────────────────

    def count(self) -> Dict[str, int]:
        return {
            "text":  self.client.count(self.text_collection).count,
            "image": self.client.count(self.image_collection).count,
        }

    def info(self):
        for name in [self.text_collection, self.image_collection]:
            info = self.client.get_collection(name)
            logger.info("Collection : %-20s | Points : %4d | Dim : %d | Distance : %s",
                        name,
                        info.points_count,
                        info.config.params.vectors.size,
                        info.config.params.vectors.distance)


# ─────────────────────────────────────────────────────────────
# CLI — quick smoke-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else r"output\chunks\chunks.json"

    with open(input_file, encoding="utf-8") as f:
        chunks = json.load(f)

    embedder = Embedder()
    chunks   = embedder.embed_chunks(chunks)

    store = VectorStore()
    store.upsert(chunks)
    store.info()

    q_vecs  = embedder.embed_query("What is this document about?")
    results = store.similarity_search(q_vecs, top_k=5)

    print("\n── Top results ──")
    for r in results:
        print(f"  score={r['score']}  [{r['source_collection']}]  {r['text'][:80]}")