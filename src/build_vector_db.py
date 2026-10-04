from langchain_chroma import Chroma

from config import VECTOR_DB_PATH, COLLECTION_NAME
from document_loader import load_documents
from text_splitter import split_documents
from embeddings import get_embedding_model


def build_vector_database():
    print("Loading documents...")
    documents = load_documents()
    if not documents:
        print("No documents to index. Put PDFs in data/documents and try again.")
        return

    print("Splitting documents...")
    chunks = split_documents(documents)
    print("Total pages:", len(documents))
    print("Total chunks:", len(chunks))

    print("Loading embedding model...")
    embeddings = get_embedding_model()

    print("Creating vector database...")
    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )

    # Start from an empty collection so re-running never creates duplicates
    vector_db.reset_collection()

    ids = vector_db.add_documents(chunks)
    stored = vector_db._collection.count()
    print("Chunks added:", len(ids))
    print("Stored in database:", stored)

    if stored != len(chunks):
        print("\nERROR: stored count does not match chunk count.")
    else:
        print("\nVector database created successfully.")


if __name__ == "__main__":
    build_vector_database()
