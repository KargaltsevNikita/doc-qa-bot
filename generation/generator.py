from ollama import generate
from generation.prompt import build_prompt


MODEL_NAME = 'mistral:7b-instruct'


def generate_answer(question: str, retrieved_results: list):
    if not retrieved_results:
        return {
            'answer': "I don't know",
            'sources': []
        }

    context = '\n\n'.join(
        result['page_content']
        for result in retrieved_results
    )

    prompt = build_prompt(
        context=context,
        question=question
    )

    response = generate(
        model=MODEL_NAME,
        prompt=prompt
    )

    answer = response['response']

    sources = []

    for result in retrieved_results:
        metadata = result['metadata']

        source = {
            'filename': metadata.get('filename'),
            'page_number': metadata.get('page_number'),
        }

        if source not in sources:
            sources.append(source)

    return {
        'answer': answer,
        'sources': sources
    }
