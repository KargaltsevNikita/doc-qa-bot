# Document Q&A Bot

Система ответов на вопросы по документам на основе подхода **Retrieval-Augmented Generation (RAG)**.

Приложение поддерживает загрузку и обработку документов, гибридный поиск, reranking, генерацию ответов с помощью LLM, трассировку и офлайн-оценку качества.

## Архитектура

```text
Клиент
  ↓
FastAPI
  ↓
LangGraph
  ↓
Гибридный поиск
  ├── Dense Retrieval (FAISS)
  └── Sparse Retrieval (BM25)
  ↓
Reciprocal Rank Fusion (RRF)
  ↓
CrossEncoder Reranker
  ↓
Prompt
  ↓
Mistral 7B Instruct
  ↓
Ollama
  ↓
Потоковый SSE-ответ
```

Наблюдаемость и трассировка:

```text
Langfuse
```

Оценка качества:

```text
RAGAS
```

## Используемые технологии

- Python
- FastAPI
- LangGraph
- FAISS
- BM25
- Reciprocal Rank Fusion (RRF)
- SentenceTransformers
- CrossEncoder
- Ollama
- Mistral 7B Instruct
- Langfuse
- RAGAS
- Docker
- Docker Compose

## Структура проекта

```text
.
├── api/
│   └── app.py
│
├── ingestion/
│   ├── loaders.py
│   ├── chunker.py
│   └── indexer.py
│
├── retrieval/
│   ├── dense.py
│   ├── sparse.py
│   ├── hybrid.py
│   └── reranker.py
│
├── generation/
│   ├── prompt.py
│   └── generator.py
│
├── workflow/
│   ├── state.py
│   └── rag_graph.py
│
├── observability/
│   └── langfuse_config.py
│
├── evaluation/
│   ├── ragas_eval.py
│   └── test_set.json
│
├── data/
├── storage/
│
├── .env
├── .env.example
├── .gitignore
├── .dockerignore
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Переменные окружения

Создайте файл `.env` в корне проекта:

```env
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Файл `.env` содержит секретные ключи и не должен добавляться в Git.

Для этого он должен быть указан в `.gitignore`:

```gitignore
.env
```

## Запуск

Для запуска проекта должен быть установлен и запущен **Docker Desktop**.

Вся система поднимается одной командой:

```bash
docker compose up --build
```

Docker Compose автоматически:

1. Запустит Ollama.
2. Дождётся готовности Ollama.
3. Скачает модель `mistral:7b-instruct`, если она ещё не загружена.
4. Запустит FastAPI-приложение.
5. Подключит приложение к Ollama внутри Docker-сети.

Первый запуск может занять больше времени, так как потребуется загрузить:

- `mistral:7b-instruct`;
- embedding-модель SentenceTransformers;
- CrossEncoder reranker.

При последующих запусках модели сохраняются в Docker volumes и повторно скачиваться не должны.

## Swagger

После успешного запуска приложения откройте:

```text
http://localhost:8000/docs
```

Через Swagger можно протестировать API.

## API

### Загрузка документа

Endpoint:

```text
POST /upload
```

Поддерживаемые форматы:

- PDF
- DOCX
- Markdown
- HTML

Загруженные документы сохраняются в:

```text
data/
```

После загрузки выполняется индексация документа.

Векторный индекс и информация о чанках сохраняются в:

```text
storage/
```

### Задать вопрос

Endpoint:

```text
POST /ask
```

Пример запроса:

```json
{
  "question": "How do dependencies work in FastAPI?"
}
```

Обработка запроса выполняется по следующему pipeline:

```text
Question
   ↓
LangGraph
   ↓
Dense Retrieval + BM25
   ↓
RRF
   ↓
CrossEncoder Reranker
   ↓
Top chunks
   ↓
Prompt
   ↓
Mistral 7B Instruct
   ↓
Answer
```

Ответ передаётся потоково с использованием **Server-Sent Events (SSE)**.

Пример SSE-событий:

```text
event: token
data: {"text": "FastAPI"}

event: token
data: {"text": " uses"}

event: sources
data: [{"filename": "dependencies.pdf", "page_number": 3}]

event: done
data: {"status": "complete"}
```

## LangGraph

LangGraph используется для управления RAG workflow.

Текущий граф:

```text
START
  ↓
retrieve
  ↓
rerank
  ↓
build_prompt
  ↓
END
```

Общее состояние графа хранится в `RAGState` и содержит:

```text
question
hybrid_results
reranked_results
prompt
```

## Retrieval

Система использует два типа поиска.

### Dense Retrieval

Dense retrieval реализован с помощью:

```text
SentenceTransformers
+
FAISS
```

Запрос преобразуется в embedding, после чего FAISS ищет наиболее близкие чанки по векторному сходству.

### Sparse Retrieval

Sparse retrieval реализован с помощью:

```text
BM25
```

BM25 ищет документы по совпадению и статистической важности слов.

### Hybrid Retrieval

Результаты Dense и BM25 объединяются с помощью:

```text
Reciprocal Rank Fusion (RRF)
```

После объединения формируется общий ranking кандидатов.

## Reranking

Для дополнительной сортировки найденных документов используется:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

CrossEncoder получает пары:

```text
question + chunk
```

и присваивает каждому чанку relevance score.

После сортировки лучшие чанки передаются в LLM.

## Генерация ответа

Для генерации используется:

```text
Mistral 7B Instruct
```

Модель запускается через:

```text
Ollama
```

LLM получает:

```text
retrieved context
+
question
+
prompt instructions
```

и генерирует ответ только на основе найденного контекста.

## Langfuse

Langfuse используется для observability и tracing.

Он позволяет отслеживать:

- пользовательский вопрос;
- выполнение LangGraph;
- retrieval;
- reranking;
- сформированный prompt;
- вызов Ollama;
- ответ модели;
- источники;
- latency;
- ошибки.

Один запрос `/ask` формирует отдельный Langfuse trace.

Пример:

```text
Document Q&A
│
└── rag-request
    │
    ├── rag-graph
    │   ├── retrieve
    │   ├── rerank
    │   └── build_prompt
    │
    └── ollama-generation
```

## Оценка качества

Для оценки RAG используется библиотека **RAGAS**.

Основные метрики:

- Faithfulness
- Answer Relevancy
- Context Recall

Тестовый набор находится в:

```text
evaluation/test_set.json
```

Для запуска RAGAS evaluation внутри контейнера:

```bash
docker compose exec app python -m evaluation.ragas_eval
```

Система позволяет сравнивать:

```text
Simple RAG
Dense Retrieval
↓
Generation
```

и:

```text
Advanced RAG
Dense + BM25
↓
RRF
↓
CrossEncoder
↓
Generation
```

## Логи

Для просмотра логов FastAPI-приложения:

```bash
docker compose logs -f app
```

Для просмотра логов Ollama:

```bash
docker compose logs -f ollama
```

Для просмотра всех контейнеров:

```bash
docker compose ps
```

## Остановка

Для остановки системы:

```bash
docker compose down
```

Docker volumes с моделями сохраняются.

Поэтому при следующем запуске:

```bash
docker compose up
```

модели не должны загружаться заново.

Чтобы удалить также постоянные Docker volumes:

```bash
docker compose down -v
```

После этого модели Ollama и Hugging Face при следующем запуске потребуется скачать заново.