def build_prompt(context: str, question: str) -> str:
    prompt = f"""
You are a helpful assistant for FastAPI documentation.

Answer the user's question using only the provided context.
Do not use information that is not present in the context.
Do not make up facts.

If the answer cannot be found in the context, reply:
"I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt