"""Prompts, context formatting, and citation validation for RAG."""

from __future__ import annotations

import re
from typing import Any, Sequence
from genai_platform.vectorstore.qdrant_store import Hit

REFUSAL = "I don't know based on the provided documents."

SYSTEM_INSTRUCTION = f"""You are a precise, grounded AI assistant answering questions based strictly on the provided documents.
Rules:
1. Answer ONLY from the numbered context passages provided.
2. Cite every factual sentence with the passage number in brackets, e.g., [1] or [2][3].
3. If the context does not contain enough information to answer the question, reply EXACTLY with: "{REFUSAL}"
4. Never cite a passage number that does not exist in the context.
5. Conversation history is only for resolving references like "it" or "they", never as factual evidence.
6. Write a complete factual sentence followed by its citation; never output only a standalone citation.
"""

MULTI_QUERY_PROMPT = """You are an AI language model assistant. Your task is to generate {n} different versions of the given user query to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative queries separated by newlines. Do not number them or include any intro/outro text.
Original query: {question}"""

SUMMARY_PROMPT = """Progressively summarize the lines of conversation provided, merging the previous summary with the new conversation turns.
The summary must be dense, factual, retain important entities and constraints, and be strictly under 120 words.
Current summary:
{summary}

New lines of conversation:
{new_lines}

Updated condensed summary:"""


def format_context(hits: Sequence[Hit]) -> str:
    """Format retrieved hits into numbered context passages."""
    if not hits:
        return "No relevant context passages found."
    lines = [f"There are {len(hits)} passage(s), numbered [1] to [{len(hits)}]:\n"]
    for i, h in enumerate(hits, start=1):
        lines.append(f"[{i}] ({h.cite})\n{h.text}\n")
    return "\n".join(lines)


def validate_citations(answer: str, n: int) -> dict[str, Any]:
    """
    Validate citations in an answer.
    Returns:
        cited: list of unique passage numbers cited
        invalid: citations outside [1..n]
        refused: True if answer equals or contains the exact refusal
        grounded: True if refused OR (cited is non-empty and invalid is empty)
    """
    refused = REFUSAL.lower() in answer.lower()
    matches = re.findall(r"\[(\d+)\]", answer)
    cited_nums = sorted(list({int(m) for m in matches}))

    invalid_nums = [x for x in cited_nums if x < 1 or x > n]
    grounded = refused or (len(cited_nums) > 0 and len(invalid_nums) == 0)

    return {
        "cited": cited_nums,
        "invalid": invalid_nums,
        "refused": refused,
        "grounded": grounded,
    }
