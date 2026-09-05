from pathlib import Path

from langchain_community.document_loaders import(
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)

def load_document(file_path: str):
    extension = Path(file_path).suffix.lower()

    if extension == '.pdf':
        loader = PyPDFLoader(file_path)

    elif extension == '.docx':
        loader = Docx2txtLoader(file_path)

    elif extension == '.md':
        loader = UnstructuredMarkdownLoader(file_path)

    elif extension in {'.html', '.htm'}:
        loader = UnstructuredHTMLLoader(file_path)

    else:
        raise ValueError(
            f'Unsupported file format: {extension}'
        )

    return loader.load()

