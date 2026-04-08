"""
settings.py
-----------
Single source of truth for ALL configurable values:
  - Model names & local cache paths
  - Directory paths
  - Qdrant settings
  - RAG hyperparameters
  - Prompt template paths

Import anywhere:
    from backend.config.settings import settings
"""

from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Root paths (all relative to project root)
# ─────────────────────────────────────────────────────────────

ROOT_DIR          = Path(__file__).resolve().parents[3]   # mRAG/
SRC_DIR           = ROOT_DIR / "src"
BACKEND_DIR       = SRC_DIR / "backend"

DOCUMENTS_DIR     = ROOT_DIR / "documents"
EXTRACTED_IMAGES  = ROOT_DIR / "extracted_images"
MODEL_CACHE_DIR   = ROOT_DIR / "models"                   # HuggingFace cache lives here
OUTPUTS_DIR       = ROOT_DIR / "outputs"
LOGS_DIR          = ROOT_DIR / "logs"

# Ensure dirs exist at import time
for _d in [DOCUMENTS_DIR, EXTRACTED_IMAGES, MODEL_CACHE_DIR, OUTPUTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Embedding models
# ─────────────────────────────────────────────────────────────

class TextEmbeddingConfig:
    """
    mixedbread-ai/mxbai-embed-large-v1 (via sentence-transformers)
    -----------------------
    • 1024-dim vectors  
    • Beats BGE-large and E5-large on MTEB retrieval benchmarks
    • Trained with AnglE loss — better semantic separation
    • Needs ~1.3 GB RAM
    • PREFIX RULES (different from BGE/E5):
        query   → prepend QUERY_INSTRUCTION
        passage → NO prefix (PASSAGE_INSTRUCTION = "")
    """

    MODEL_NAME          = "mixedbread-ai/mxbai-embed-large-v1"
    VECTOR_DIM          = 1024
    CACHE_DIR           = str(MODEL_CACHE_DIR / "mxbai-embed-large-v1")
    NORMALIZE           = True
    BATCH_SIZE          = 32
    QUERY_INSTRUCTION   = "Represent this sentence for retrieving relevant passages: "
    PASSAGE_INSTRUCTION = ""          # mxbai: NO prefix on passages
 

class ImageTextEmbeddingConfig:
    """
    clip-ViT-B-32  (via sentence-transformers)
    ------------------------------------------
    • 512-dim vectors
    • Embeds BOTH images and text in the SAME vector space
    • Perfect for your multimodal chunks (text + image_paths)
    • Much lighter than BGE (~350 MB)
    • Use when chunks have images; fallback to text model otherwise
    """
    MODEL_NAME = "clip-ViT-B-32"
    VECTOR_DIM = 512
    CACHE_DIR  = str(MODEL_CACHE_DIR / "clip-ViT-B-32")
    NORMALIZE  = True
    BATCH_SIZE = 32


# Active embedding strategy: "text" | "multimodal"
# Switch to "multimodal" when your chunks have real image content
EMBEDDING_STRATEGY = "text"

# Shortcut — used by Embedder and VectorStore
ACTIVE_EMBEDDING = (
    TextEmbeddingConfig
    if EMBEDDING_STRATEGY == "text"
    else ImageTextEmbeddingConfig
)


# ─────────────────────────────────────────────────────────────
# Qdrant
# ─────────────────────────────────────────────────────────────

class QdrantConfig:
    COLLECTION_NAME  = "rag_chunks"
    IN_MEMORY        = True           # False → use HOST/PORT for real server
    HOST             = "localhost"
    PORT             = 6333
    DISTANCE         = "Cosine"       # "Cosine" | "Dot" | "Euclid"
    UPSERT_BATCH     = 64


# ─────────────────────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────────────────────

class ChunkConfig:
    MAX_NEIGHBORS       = 3      # spatial nearby-paragraph count
    MIN_PARAGRAPH_CHARS = 20     # from doc_parser


# ─────────────────────────────────────────────────────────────
# RAG retrieval
# ─────────────────────────────────────────────────────────────

class RetrievalConfig:
    TOP_K            = 5
    SCORE_THRESHOLD  = 0.30      # minimum cosine similarity to include
    MAX_CONTEXT_CHARS = 4000     # max chars sent to LLM


# ─────────────────────────────────────────────────────────────
# LLM (Claude)
# ─────────────────────────────────────────────────────────────

class LLMConfig:
    MODEL      = "claude-sonnet-4-6"
    MAX_TOKENS = 1024


# ─────────────────────────────────────────────────────────────
# Prompts directory
# ─────────────────────────────────────────────────────────────

PROMPTS_DIR = BACKEND_DIR / "prompts"


# ─────────────────────────────────────────────────────────────
# Convenience bundle (optional — import this if you want one object)
# ─────────────────────────────────────────────────────────────

class Settings:
    root              = ROOT_DIR
    documents         = DOCUMENTS_DIR
    extracted_images  = EXTRACTED_IMAGES
    model_cache       = MODEL_CACHE_DIR
    outputs           = OUTPUTS_DIR

    embedding         = ACTIVE_EMBEDDING
    text_embedding    = TextEmbeddingConfig
    image_embedding   = ImageTextEmbeddingConfig
    embedding_strategy = EMBEDDING_STRATEGY

    qdrant            = QdrantConfig
    chunk             = ChunkConfig
    retrieval         = RetrievalConfig
    llm               = LLMConfig
    prompts_dir       = PROMPTS_DIR


settings = Settings()

# EMBEDDING_STRATEGY = "text"        # → BAAI/bge-large-en-v1.5 (1024-dim)
# EMBEDDING_STRATEGY = "multimodal"  # → clip-ViT-B-32 (512-dim)

# Model                   Dim       Best for                              Size
# BAAI/bge-large-en-v1.5  1024      Newspaper articles, pure text         ~1.3 GB
# clip-ViT-B-32           512       Chunks with real image content        ~350 MB