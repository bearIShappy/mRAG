"""
rag_qa.py
---------
Full RAG QA pipeline — retrieval + LLM answer generation.

Usage:
    from backend.core.rag_qa import RAGQA
    qa     = RAGQA(store=store, embedder=embedder)
    result = qa.ask("What did newspapers say about GMMDC?")
    print(result["answer"])
"""

import os
from typing import List, Dict, Optional

from backend.config.settings  import settings
from backend.prompts.rag_prompts import RAGPrompts


# ─────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────

def _build_context(retrieved: List[Dict]) -> str:
    max_chars = settings.retrieval.MAX_CONTEXT_CHARS
    parts     = []
    total     = 0

    for i, r in enumerate(retrieved, 1):
        chunk_str = RAGPrompts.build_context_chunk(
            index       = i,
            page        = r.get("metadata", {}).get("page_number", "?"),
            score       = r.get("score", 0),
            text        = r.get("text", ""),
            caption     = r.get("image_caption", ""),
            description = r.get("image_description", []),
        )
        if total + len(chunk_str) > max_chars:
            break
        parts.append(chunk_str)
        total += len(chunk_str)

    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────

def _call_claude(prompt: str) -> str:
    try:
        import anthropic
    except ImportError:
        return "[ERROR] Run: pip install anthropic"

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERROR] ANTHROPIC_API_KEY not set."

    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model      = settings.llm.MODEL,
        max_tokens = settings.llm.MAX_TOKENS,
        messages   = [{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ─────────────────────────────────────────────────────────────
# RAGQA
# ─────────────────────────────────────────────────────────────

class RAGQA:

    def __init__(self, store, embedder):
        self.store    = store
        self.embedder = embedder

    def ask(self, question: str, verbose: bool = False) -> Dict:
        # 1. Embed query
        query_vector = self.embedder.embed_query(question)

        # 2. Retrieve
        retrieved = self.store.similarity_search(query_vector)

        if not retrieved:
            return {
                "question": question,
                "answer":   "No relevant content found in the document.",
                "sources":  [],
                "context":  "",
            }

        if verbose:
            print(f"\n[RAG] {len(retrieved)} chunks retrieved:")
            for r in retrieved:
                print(f"  score={r['score']}  text={r['text'][:70]}")

        # 3. Build context using prompt templates
        context = _build_context(retrieved)

        # 4. Build full prompt and call LLM
        prompt = RAGPrompts.build_qa_prompt(question=question, context=context)
        answer = _call_claude(prompt)

        return {
            "question": question,
            "answer":   answer,
            "sources":  retrieved,
            "context":  context,
        }


# ─────────────────────────────────────────────────────────────
# CLI — interactive loop
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys
    from backend.core.embedder     import Embedder
    from backend.core.vector_store import VectorStore

    input_file = sys.argv[1] if len(sys.argv) > 1 else "chunks.json"

    with open(input_file, encoding="utf-8") as f:
        chunks = json.load(f)

    embedder = Embedder()
    chunks   = embedder.embed_chunks(chunks)

    store = VectorStore()
    store.upsert(chunks)
    store.info()

    qa = RAGQA(store=store, embedder=embedder)
    print("\n[RAG] Ready. Type your question (or 'quit').\n")

    while True:
        q = input("Q: ").strip()
        if not q or q.lower() in {"quit", "exit", "q"}:
            break
        result = qa.ask(q, verbose=True)
        print(f"\nA: {result['answer']}\n{'─'*60}")