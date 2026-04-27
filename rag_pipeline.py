from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


def load_and_process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)
    texts = [doc.page_content for doc in chunks]

    return texts


def create_vector_store(texts):
    embeddings = embed_model.encode(texts)

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, texts


def retrieve(query, index, texts, k=5):
    q_embedding = embed_model.encode([query])
    distances, indices = index.search(q_embedding, k)

    return [texts[i] for i in indices[0]]