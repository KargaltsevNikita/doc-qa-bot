import hashlib
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.loaders import load_document
from ingestion.chunker import chunk_documents

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / 'data'
STORAGE_DIR = PROJECT_ROOT / 'storage'

INDEX_PATH = STORAGE_DIR / 'index.faiss'
CHUNKS_PATH = STORAGE_DIR / 'chunks.json'

SUPPORTED_EXTENSIONS = {
    '.pdf',
    '.docx',
    '.md',
    '.html',
    '.htm',
}

def get_chunk_hash(text: str) -> str:
    normalized_text = re.sub(
        r'\s+',
        ' ',
        text.strip()
    )

    return hashlib.sha256(
        normalized_text.encode('utf-8')
    ).hexdigest()

def load_saved_chunks():
    if not CHUNKS_PATH.exists():
        return []

    with open(
        CHUNKS_PATH,
        'r',
        encoding='utf-8'
    ) as file:
        return json.load(file)

def save_chunks(chunks):
    with open(
        CHUNKS_PATH,
        'w',
        encoding='utf-8'
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii= False,
            indent=2
        )

def load_or_create_index(embedding_size: int):
    if INDEX_PATH.exists():
        index = faiss.read_index(
            str(INDEX_PATH)
        )

        if index.d != embedding_size:
            raise ValueError(
                'Размер embedding-модели '
                'не совпадают с размером FAISS index'
            )
        return index

    return faiss.IndexFlatIP(
        embedding_size
    )

def build_index():
    STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    embedding_size = (
        model.get_sentence_embedding_dimension()
    )

    index = load_or_create_index(
        embedding_size
    )

    saved_chunks = load_saved_chunks()

    if index.ntotal != len(saved_chunks):
        raise ValueError(
            'FAISS index и chunks.json'
            'содержат разное количество объектов'
        )

    seen_hashes = {
        chunk['chunk_hash']
        for chunk in saved_chunks
    }

    new_chunks = []

    for file_path in DATA_DIR.iterdir():

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        print(
            f'Обрабатываем: {file_path.name}'
        )

        documents = load_document(
            str(file_path)
        )

        chunks = chunk_documents(
            documents
        )

        for chunk in chunks:

            chunk_hash = get_chunk_hash(
                chunk.page_content
            )

            if chunk_hash in seen_hashes:
                continue

            seen_hashes.add(
                chunk_hash
            )

            new_chunks.append(
                (
                    chunk_hash,
                    chunk
                )
            )

    if not new_chunks:
        print(
            'Новых чанков для индексации нет'
        )
        return index

    texts = [
        chunk.page_content
        for _, chunk in new_chunks
    ]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    vectors = np.asarray(
        vectors,
        dtype='float32'
    )

    index.add(
        vectors
    )

    for chunk_hash, chunk in new_chunks:

        saved_chunks.append(
            {
                'chunk_hash': chunk_hash,
                'page_content': chunk.page_content,
                'metadata': {
                    'source': chunk.metadata.get('source'),
                    'filename': chunk.metadata.get('filename'),
                    'page_number': chunk.metadata.get('page_number'),
                    'section_title': chunk.metadata.get('section_title'),

                }
            }
        )

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    save_chunks(
        saved_chunks
    )

    print(
        f'Всего чанков в FAISS: '
        f'{index.ntotal}'
    )

    return index

if __name__ == "__main__":
    build_index()