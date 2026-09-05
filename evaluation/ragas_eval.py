import json
from pathlib import Path
from time import perf_counter
import os

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import(
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
)

from retrieval.dense import dense_search
from workflow.rag_graph import rag_graph
from generation.generator import generate_answer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_SET_PATH = PROJECT_ROOT / "evaluation" / "test_set.json"
LOG_PATH = PROJECT_ROOT / "evaluation" / "rag_logs.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

client = AsyncOpenAI(
    api_key='ollama',
    base_url=os.getenv(
        'OLLAMA_OPENAI_BASE_URL',
        'http://localhost:11434/v1'
    )
)

evaluator_llm = llm_factory(
    'mistral:7b-instruct',
    provider='openai',
    client=client,
    temperature=0,
    max_tokens=4096,
)

evaluator_embeddings = HuggingFaceEmbeddings(
    model=EMBEDDING_MODEL,
    device='cpu',
    normalize_embeddings=True,
)

faithfulness_metric = Faithfulness(
    llm=evaluator_llm
)

answer_relevancy_metric = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

context_recall_metric = ContextRecall(
    llm=evaluator_llm
)

def load_test_set():
    with open(
        TEST_SET_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)

def log_request(
        system_name,
        query,
        retrieved_chunks,
        answer,
        latency,
):
    log_record = {
        'system': system_name,
        'query': query,
        'retrieved_chunks': retrieved_chunks,
        'answer': answer,
        'latency': latency,
    }

    with open(
        LOG_PATH,
        'a',
        encoding='utf-8'
    ) as file:
        file.write(
            json.dumps(
                log_record,
                ensure_ascii=False
            )
            + '\n'
        )

def run_simple_rag(question):
    start = perf_counter()

    retrieved_results = dense_search(
        question,
        top_k = 3
    )

    generation = generate_answer(
        question,
        retrieved_results
    )

    latency = perf_counter() - start

    log_request(
        system_name='simple_rag',
        query=question,
        retrieved_chunks=retrieved_results,
        answer=generation['answer'],
        latency=latency,
    )

    return {
        'answer': generation['answer'],
        'retrieved_results': retrieved_results,
        'latency': latency,
    }

def run_advanced_rag(question):
    start = perf_counter()

    graph_result = rag_graph.invoke(
        {
            'question': question
        }
    )

    reranked_results = graph_result[
        'reranked_results'
    ]

    generation = generate_answer(
        question,
        reranked_results
    )

    latency = perf_counter() - start

    log_request(
        system_name='hybrid_rag',
        query=question,
        retrieved_chunks=reranked_results,
        answer=generation['answer'],
        latency=latency,
    )

    return {
        'answer': generation['answer'],
        'retrieved_results': reranked_results,
        'latency': latency,
    }

def calculate_metrics(
        question,
        reference,
        answer,
        retrieved_results,
):
    contexts = [
        result['page_content']
        for result in retrieved_results
    ]

    faithfulness_result = faithfulness_metric.score(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    answer_relevancy_result = (
        answer_relevancy_metric.score(
            user_input=question,
            response=answer,
        )
    )

    context_recall_result = context_recall_metric.score(
        user_input=question,
        retrieved_contexts=contexts,
        reference=reference,
    )

    return {
        'faithfulness': faithfulness_result.value,
        'answer_relevancy': answer_relevancy_result.value,
        'context_recall': context_recall_result.value,
    }

def evaluate_system(system_function, test_set):
    results = []

    for item in test_set:
        question = item['question']
        reference = item['reference']

        rag_result = system_function(
            question
        )

        metrics = calculate_metrics(
            question=question,
            reference=reference,
            answer=rag_result['answer'],
            retrieved_results=rag_result['retrieved_results'],
        )

        results.append(
            {
                'question': question,
                'answer': rag_result['answer'],
                'latency': rag_result['latency'],
                **metrics,
            }
        )

    return results

def calculate_average(results):
    count = len(results)

    return {
        "faithfulness": sum(
            result["faithfulness"]
            for result in results
        ) / count,

        "answer_relevancy": sum(
            result["answer_relevancy"]
            for result in results
        ) / count,

        "context_recall": sum(
            result["context_recall"]
            for result in results
        ) / count,

        "latency": sum(
            result["latency"]
            for result in results
        ) / count,
    }

def run_ab_test():
    test_set = load_test_set()

    simple_results = evaluate_system(
        run_simple_rag,
        test_set
    )

    advanced_results = evaluate_system(
        run_advanced_rag,
        test_set
    )

    simple_average = calculate_average(
        simple_results
    )

    advanced_average = calculate_average(
        advanced_results
    )

    print("Simple RAG:")
    print(simple_average)

    print()

    print("Hybrid + Reranking:")
    print(advanced_average)

if __name__ == "__main__":
    run_ab_test()