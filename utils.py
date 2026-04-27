import requests

def ask_llm(context, question):
    prompt = f"""
    Answer ONLY from the context below.
    If the answer is not in the context, say "I don't know".

    Context:
    {context}

    Question:
    {question}
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]