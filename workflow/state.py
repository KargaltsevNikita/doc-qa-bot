from typing_extensions import TypedDict

class RAGState(TypedDict, total=False):
    question: str
    hybrid_results: list
    reranked_results: list
    prompt: str