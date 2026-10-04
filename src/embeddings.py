from langchain_huggingface import HuggingFaceEmbeddings

try:
    from config import EMBEDDING_MODEL
except ImportError:          # when imported as src.embeddings
    from src.config import EMBEDDING_MODEL


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
