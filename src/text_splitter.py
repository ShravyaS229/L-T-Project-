from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from document_loader import load_documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    print("Total documents/pages:", len(documents))
    print("Total chunks:", len(chunks))

    if chunks:
        print("\nFirst chunk:")
        print(chunks[0].page_content)
        print("\nMetadata:", chunks[0].metadata)
