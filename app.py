from rag_pipeline import load_and_process_pdf, create_vector_store, retrieve
from utils import ask_llm

# Load PDF
texts = load_and_process_pdf("data/your_file.pdf")

# Create DB
index, texts = create_vector_store(texts)

print("PDF loaded and indexed!")

while True:
    query = input("\nAsk a question (or 'exit'): ")

    if query.lower() == "exit":
        break

    docs = retrieve(query, index, texts)
    context = "\n".join(docs)

    answer = ask_llm(context, query)

    print("\nAnswer:", answer)