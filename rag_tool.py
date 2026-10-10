from langchain_core.tools import tool
from rag import retriever

@tool
def it_kb_retrieve(query: str) -> str:
    """
    Query the university IT policy knowledge base.
    Use this tool for all user IT‑related support questions.
    Only facts returned from this tool can be used to answer.
    If no relevant information is found, state that the information is not specified.
    Do NOT invent facts, numbers or phone numbers.

    Args:
        query: User IT‑support question
    Returns:
        Concatenated policy document fragments
    """
    docs = retriever.invoke(query)
    output_parts = []
    for d in docs:
        meta = d.metadata
        output_parts.append(
            f"[policy_id:{meta['policy_id']}, topic:{meta['topic']}]\n{d.page_content}"
        )
    return "\n---\n".join(output_parts)
