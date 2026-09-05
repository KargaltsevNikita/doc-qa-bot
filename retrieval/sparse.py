import json
import re
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = PROJECT_ROOT / "storage" / "chunks.json"

TOP_K = 10


def load_chunks():
    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)

def tokenize(text:str):
    return re.findall(
        r'\w+',
        text.lower()
    )

def sparse_search(
        query: str,
        top_k: int = TOP_K,
        section_title: str | None = None,
):
    chunks = load_chunks()

    if section_title is not None:
        chunks = [
            chunk
            for chunk in chunks
            if chunk['metadata'].get('section_title') == section_title
        ]

    if not chunks:
        return []

    tokenized_corpus = [
        tokenize(chunk['page_content'])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index_id in top_indices:

        if scores[index_id] <= 0:
            continue

        chunk = chunks[index_id]

        results.append(
            {
                'score': float(scores[index_id]),
                'page_content': chunk['page_content'],
                'metadata': chunk['metadata'],
            }
        )

    return results