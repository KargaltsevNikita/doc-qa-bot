import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = PROJECT_ROOT / "storage" / "index.faiss"
CHUNKS_PATH = PROJECT_ROOT / "storage" / "chunks.json"

TOP_K = 10

model = SentenceTransformer(MODEL_NAME)


def load_index():
    return faiss.read_index(
        str(INDEX_PATH)
    )


def load_chunks():
    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def dense_search(
    query: str,
    top_k: int = TOP_K
):
    if not query.strip():
        return []

    index = load_index()
    chunks = load_chunks()

    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_vector = np.ascontiguousarray(
        query_vector,
        dtype="float32"
    )

    scores, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):
        if index_id == -1:
            continue

        if index_id >= len(chunks):
            continue

        chunk = chunks[index_id]

        results.append(
            {
                "score": float(score),
                "page_content":
                    chunk["page_content"],
                "metadata":
                    chunk["metadata"],
            }
        )

    return results