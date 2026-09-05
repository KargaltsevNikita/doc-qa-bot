from retrieval.dense import dense_search
from retrieval.sparse import sparse_search

RRF_K = 60
TOP_K = 10

def hybrid_search(query: str, top_k: int = TOP_K):
    dense_results = dense_search(query, top_k=top_k)

    sparse_results = sparse_search(query, top_k=top_k)

    rrf_scores = {}
    chunks = {}

    for rank, result in enumerate(
        dense_results,
        start=1
    ):
        chunk_key = result['page_content']

        rrf_scores[chunk_key] = (
            rrf_scores.get(chunk_key, 0) + 1 / (RRF_K + rank)
        )

        chunks[chunk_key] = result

    for rank, result in enumerate(
        sparse_results,
        start=1
    ):
        chunk_key = result['page_content']

        rrf_scores[chunk_key] = (
            rrf_scores.get(chunk_key, 0) + 1 / (RRF_K + rank)
        )

        chunks[chunk_key] = result

    sorted_keys = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    results = []

    for chunk_key in sorted_keys[:top_k]:
        result = chunks[chunk_key].copy()

        result['rrf_score'] = rrf_scores[chunk_key]

        results.append(result)

    return results