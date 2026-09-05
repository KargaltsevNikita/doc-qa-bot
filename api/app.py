import json
import shutil
import time

from collections import defaultdict, deque
from pathlib import Path
from threading import Lock

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)

from fastapi.responses import StreamingResponse
from ollama import generate
from pydantic import BaseModel

from ingestion.indexer import build_index
from workflow.rag_graph import rag_graph

from langfuse import propagate_attributes
from observability.langfuse_config import (
    langfuse,
    langfuse_handler,
)

app = FastAPI(
    title='Document Q&A API',
    version='1.0.0'
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

SUPPORTED_EXTENSIONS = {
    '.pdf',
    '.docx',
    '.md',
    '.html',
    '.htm',
}

MODEL_NAME = 'mistral:7b-instruct'

RATE_LIMIT = 10
RATE_WINDOW = 60

request_history = defaultdict(deque)
rate_limit_lock = Lock()

def rate_limit(request: Request):
    if request.client is None:
        client_ip = 'unknown'
    else:
        client_ip = request.client.host

    now = time.monotonic()

    with rate_limit_lock:
        timestamps = request_history[client_ip]

        while timestamps and now - timestamps[0] >= RATE_WINDOW:
            timestamps.popleft()

        if len(timestamps) >= RATE_LIMIT:
            retry_after = int(
                RATE_WINDOW - (now - timestamps[0])
            ) + 1

            raise HTTPException(
                status_code=429,
                detail=(
                    "Rate limit exceeded. " "Maximum 10 requests per minute."
                ),
                headers={
                    "Retry-After": str(retry_after)
                },
            )

        timestamps.append(now)

class AskRequest(BaseModel):
    question: str

def create_sse_event(
        event: str,
        data,
):
    return (
        f'event: {event}\n'
        f'data: {json.dumps(data, ensure_ascii=False)}\n\n'
    )

@app.post('/upload')
def upload_document(
        file: UploadFile = File(...),
        _: None = Depends(rate_limit),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail='Filename is missing.')

    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=(f"Unsupported file format: {extension}"), )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = DATA_DIR / filename

    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"File '{filename}' already exists."
            ),
        )

    try:
        with open(
            destination,
            'wb'
        ) as output_file:
            shutil.copyfileobj(
                file.file,
                output_file
            )

        build_index()

    except Exception as error:
        if destination.exists():
            destination.unlink()

        raise HTTPException(
            status_code=500,
            detail=f'Indexing failed: {error}'
        )

    finally:
        file.file.close()

    return {
        'status': 'success',
        'filename': filename,
        'message': 'Document uploaded and indexed.'
    }

@app.post('/ask')
def ask_question(
        request_data: AskRequest,
        _: None = Depends(rate_limit),
):
    question = request_data.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail='Question cannot be empty.'
        )

    def event_stream():

        with langfuse.start_as_current_observation(
            as_type='span',
            name='rag-request',
            input={
                'question': question
            }
        ) as root_span:

            with propagate_attributes(
                trace_name='Document Q&A',
                tags=[
                    'rag',
                    'langgraph',
                    'fastapi'
                ],
            ):

                try:

                    graph_result = rag_graph.invoke(
                        {
                            'question': question
                        },
                        config={
                            'callbacks': [
                                langfuse_handler
                            ],
                            'run_name': 'rag-graph',
                        }
                    )

                    reranked_results = graph_result[
                        'reranked_results'
                    ]

                    prompt = graph_result[
                        'prompt'
                    ]

                    if not reranked_results:
                        answer = "I don't know"

                        root_span.update(
                            output={
                                'answer': answer,
                                'sources': []
                            }
                        )

                        yield create_sse_event(
                            'token',
                            {
                                'text': answer
                            }
                        )

                        yield create_sse_event(
                            'sources',
                            []
                        )

                        yield create_sse_event(
                            'done',
                            {
                                'status': 'complete'
                            }
                        )

                        return

                    with langfuse.start_as_current_observation(
                        as_type='generation',
                        name='ollama-generation',
                        model=MODEL_NAME,
                        input=prompt,
                    ) as generation:

                        stream = generate(
                            model=MODEL_NAME,
                            prompt=prompt,
                            stream=True,
                        )

                        answer_parts = []

                        for chunk in stream:
                            token = (
                                chunk.response
                                or ''
                            )

                            if token:
                                answer_parts.append(
                                    token
                                )

                                yield create_sse_event(
                                    'token',
                                    {
                                        'text': token
                                    }
                                )

                        answer = ''.join(
                            answer_parts
                        )

                        generation.update(
                            output=answer
                        )

                    sources = []

                    for result in reranked_results:
                        metadata = result[
                            'metadata'
                        ]

                        source = {
                            'filename':
                                metadata.get(
                                    'filename'
                                ),

                            'page_number':
                                metadata.get(
                                    'page_number'
                                ),
                        }

                        if source not in sources:
                            sources.append(
                                source
                            )

                    root_span.update(
                        output={
                            'answer': answer,
                            'sources': sources
                        }
                    )

                    yield create_sse_event(
                        'sources',
                        sources
                    )

                    yield create_sse_event(
                        'done',
                        {
                            'status': 'complete'
                        }
                    )

                except Exception as error:

                    root_span.update(
                        output={
                            'error': str(error)
                        }
                    )

                    yield create_sse_event(
                        'error',
                        {
                            'message': str(error)
                        }
                    )

    return StreamingResponse(
        event_stream(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )
