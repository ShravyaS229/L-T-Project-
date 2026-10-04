"""Shared settings for the whole knowledge-base module.

Person 2 only needs: VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL.
"""
from pathlib import Path

# Project root = folder that contains src/, data/ and vector_db/
BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENT_FOLDER = BASE_DIR / "data" / "documents"
VECTOR_DB_PATH = str(BASE_DIR / "vector_db")

COLLECTION_NAME = "college_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
