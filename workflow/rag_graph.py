from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from workflow.state import RAGState
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank
from generation.prompt import build_prompt

def retrieve_node(state: RAGState):
    hybrid_results = hybrid_search(
        state['question'],
        top_k=10
    )

    return {
        'hybrid_results': hybrid_results
    }

def rerank_node(state: RAGState):
    reranked_results = rerank(
        state['question'],
        state['hybrid_results'],
        top_k=3
    )

    return {
        'reranked_results': reranked_results
    }

def build_prompt_node(state: RAGState):
    if not state['reranked_results']:
        return {
            'prompt': ''
        }

    context = '\n\n'.join(
        result['page_content']
        for result in state['reranked_results']
    )

    prompt = build_prompt(
        context=context,
        question=state['question']
    )

    return {
        'prompt': prompt
    }

graph_builder = StateGraph(RAGState)

graph_builder.add_node(
    'retrieve',
    retrieve_node
)

graph_builder.add_node(
    'rerank',
    rerank_node
)

graph_builder.add_node(
    'build_prompt',
    build_prompt_node
)

graph_builder.add_edge(
    START,
    'retrieve'
)

graph_builder.add_edge(
    'retrieve',
    'rerank'
)

graph_builder.add_edge(
    'rerank',
    'build_prompt'
)

graph_builder.add_edge(
    'build_prompt',
    END
)

rag_graph = graph_builder.compile()