"""
rag_prompts.py
--------------
All LLM prompt templates used in the RAG pipeline.

Keep every prompt string here — never hardcode them inside logic files.
Import with:
    from backend.prompts.rag_prompts import RAGPrompts
"""


class RAGPrompts:

    # ── QA system instruction ─────────────────────────────────
    QA_SYSTEM = """You are a precise document QA assistant.
Answer ONLY using the provided context chunks.
If the context does not contain enough information to answer, say:
"I don't have enough information in the document to answer that."
Be concise and factual. Cite source numbers (e.g. [Source 1]) when relevant."""

    # ── QA user turn template ─────────────────────────────────
    # Slots: {context}  {question}
    QA_USER = """\
=== DOCUMENT CONTEXT ===
{context}

=== QUESTION ===
{question}

=== ANSWER ==="""

    # ── Context chunk formatter ───────────────────────────────
    # Slots: {index}  {page}  {score}  {text}  {caption}  {description}
    CONTEXT_CHUNK = """\
[Source {index}: page {page} | similarity {score}]
{text}{caption_block}{description_block}"""

    CAPTION_BLOCK     = "\n[Image Caption]: {caption}"
    DESCRIPTION_BLOCK = "\n[Image Description]: {description}"

    # ── Summarisation (future use) ────────────────────────────
    SUMMARISE_USER = """\
Summarise the following document content in 3-5 bullet points.
Focus on key facts, names, dates, and decisions.

=== CONTENT ===
{content}

=== SUMMARY ==="""

    # ─────────────────────────────────────────────────────────
    # Builder helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_context_chunk(index: int, page, score: float,
                            text: str, caption: str = "",
                            description: list = None) -> str:
        """Format one retrieved chunk into context text."""
        cap_block  = RAGPrompts.CAPTION_BLOCK.format(caption=caption) if caption else ""
        desc_list  = description or []
        desc_block = (
            RAGPrompts.DESCRIPTION_BLOCK.format(description=" | ".join(desc_list))
            if desc_list else ""
        )
        return RAGPrompts.CONTEXT_CHUNK.format(
            index=index, page=page, score=score,
            text=text.strip(),
            caption_block=cap_block,
            description_block=desc_block,
        )

    @staticmethod
    def build_qa_prompt(question: str, context: str) -> str:
        """Combine system + user turn into a single prompt string."""
        return (
            RAGPrompts.QA_SYSTEM
            + "\n\n"
            + RAGPrompts.QA_USER.format(context=context, question=question)
        )