from pathlib import Path
import hashlib
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_section_title(text: str):
    lines = text.splitlines()

    for line in lines:
        line = line.strip()

        if 3 <= len(line) <= 120:
            return line

    return None

def add_metadata(documents):
    for document in documents:
        source = document.metadata.get('source', '')

        document.metadata['filename'] = Path(source).name if source else None

        page = document.metadata.get('page')

        if isinstance(page, int):
            document.metadata['page_number'] = page + 1
        else:
            document.metadata['page_number'] = None

        document.metadata['section_title'] = extract_section_title(
            document.page_content
        )

    return documents

def deduplicate_chunks(chunks):
    unique_chunks = []
    seen_hashes = set()

    for chunk in chunks:
        normalized_text = re.sub(
            r'\s+',
            ' ',
            chunk.page_content.strip()
        )

        chunk_hash = hashlib.sha256(
            normalized_text.encode('utf-8')
        ).hexdigest()

        if chunk_hash not in seen_hashes:
            seen_hashes.add(chunk_hash)
            unique_chunks.append(chunk)

    return unique_chunks

def chunk_documents(documents):
    documents = add_metadata(documents)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    chunks = deduplicate_chunks(chunks)

    return chunks