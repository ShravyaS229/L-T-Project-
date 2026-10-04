"""Interface for Person 2.

    from src.retriever import search_documents, get_retriever

    results = search_documents("What is the attendance requirement?", k=3)
    for r in results:
        print(r["text"], r["source"], r["page"], r["score"])

`score` is a distance: LOWER means MORE similar.
`get_retriever()` returns a LangChain retriever for use in a RAG chain.
"""
from functools import lru_cache

from langchain_chroma import Chroma

try:
    from config import VECTOR_DB_PATH, COLLECTION_NAME
    from embeddings import get_embedding_model
except ImportError:          # when imported as src.retriever
    from src.config import VECTOR_DB_PATH, COLLECTION_NAME
    from src.embeddings import get_embedding_model


@lru_cache(maxsize=1)
def get_vector_db():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_PATH,
        embedding_function=get_embedding_model(),
    )


def get_retriever(k=3):
    return get_vector_db().as_retriever(search_kwargs={"k": k})


def search_documents(question, k=3):
    """Return the k most relevant chunks as plain dictionaries."""
    pairs = get_vector_db().similarity_search_with_score(question, k=k)
    return [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "score": float(score),
        }
        for doc, score in pairs
    ]
