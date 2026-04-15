"""
rag_prompts.py
--------------
All LLM prompt templates used in the RAG pipeline.

Keep every prompt string here — never hardcode them inside logic files.
Import with:
    from backend.prompts.rag_prompts import RAGPrompts

Gemma 4 token reference (from tokenizer_config.json):
    sot_token  : <|turn>       — start of turn
    eot_token  : <turn|>       — end of turn
    soc_token  : <|channel>    — start of channel (role)
    eoc_token  : <channel|>    — end of channel (role)
"""

from src.backend.config.settings import settings


class RAGPrompts:

    # ── QA system instruction ─────────────────────────────────
    QA_SYSTEM = """You are an expert analytical assistant handling a multimodal document. You will receive context chunks containing standard Text, and optionally [Image Caption] and [Image Description] blocks.

    CRITICAL DIRECTIVES:
    1. EQUAL FOCUS: Give equal weight to the standard Text and the image blocks. A complete answer often requires fusing facts from both.
    2. VISUAL TRANSLATION: You cannot physically 'see' images. However, the [Image Caption] and [Image Description] blocks act as your definitive eyes. Treat their contents as absolute visual facts.
    3. MISSING CAPTIONS: If a user asks about a visual element (photo, diagram, chart) but the [Image Caption] and [Image Description] are missing or empty, you MUST rely on the main Text of the chunk to infer the visual context, as it is spatially linked to the image.
    4. NO HEDGING: Never state that you cannot see an image. Seamlessly answer visual questions (e.g., "who is in the photo?", "what does the diagram show?") using the provided text representations.

    If the provided context does not contain enough information to answer the question, state exactly: "I don't have enough information in the document to answer that."
    Be concise and factual. Always cite page_number and image_indices as figure numbers (e.g., [Page 1][Figure 1]) when relevant."""
    
    # ── QA user turn — generic (used by Claude) ───────────────
    QA_USER = """\
=== DOCUMENT CONTEXT ===
{context}

=== QUESTION ===
{question}

=== ANSWER ==="""

    # ── QA prompt — Gemma 4 chat format ──────────────────────
    # Tokens from tokenizer_config.json:
    #   <|turn>    = sot_token  (start of turn)
    #   <turn|>    = eot_token  (end of turn)
    #   <|channel> = soc_token  (start of channel/role)
    #   <channel|> = eoc_token  (end of channel/role)
    # No closing tag on final model turn — model generates from here
    QA_GEMMA4 = """\
<|turn><|channel>system
{system}<channel|><turn|>
<|turn><|channel>user
=== DOCUMENT CONTEXT ===
{context}

=== QUESTION ===
{question}<channel|><turn|>
<|turn><|channel>model
"""

    # ── Context chunk formatter ───────────────────────────────
    CONTEXT_CHUNK     = "[Source {index}: page {page} | similarity {score}]\n{text}{caption_block}{description_block}"
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
        """
        Build the final prompt string.
        Automatically uses the correct format based on settings.llm.PROVIDER.

        "claude" → plain text (system + user turn)
        "local"  → Gemma 4 chat format with special tokens
        """
        provider = settings.llm.PROVIDER  # "claude" | "local"

        if provider == "local":
            return RAGPrompts.QA_GEMMA4.format(
                system   = RAGPrompts.QA_SYSTEM,
                context  = context,
                question = question,
            )

        # Default: Claude / any plain-text LLM
        return (
            RAGPrompts.QA_SYSTEM
            + "\n\n"
            + RAGPrompts.QA_USER.format(context=context, question=question)
        )