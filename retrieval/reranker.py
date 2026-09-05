from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
TOP_K = 3

model = CrossEncoder(MODEL_NAME)


def rerank(
    query: str,
    results: list,
    top_k: int = TOP_K
):
    if not query.strip():
        return []

    if not results:
        return []

    pairs = []

    for result in results:
        pairs.append(
            [
                query,
                result["page_content"]
            ]
        )

    scores = model.predict(pairs)

    reranked_results = []

    for result, score in zip(
        results,
        scores
    ):
        result = result.copy()

        result["reranker_score"] = float(
            score
        )

        reranked_results.append(
            result
        )

    reranked_results.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return reranked_results[:top_k]
