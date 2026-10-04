import re
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import DOCUMENT_FOLDER


def clean_text(text):
    """Light cleaning: remove page-number lines and extra whitespace."""
    text = text.replace("\x00", "")
    # lines that are only a page number, e.g. "12", "Page 12", "Page 12 of 80"
    text = re.sub(r"^\s*(page\s*)?\d+(\s*of\s*\d+)?\s*$", "", text,
                  flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)       # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)    # collapse blank lines
    return text.strip()


def remove_repeated_lines(pages):
    """Remove headers/footers: short lines repeated on at least half the pages."""
    if len(pages) < 3:
        return pages

    counts = Counter()
    for page in pages:
        for line in {l.strip() for l in page.splitlines() if l.strip()}:
            counts[line] += 1

    threshold = max(3, int(len(pages) * 0.5))
    repeated = {line for line, n in counts.items()
                if n >= threshold and len(line) < 80}

    return ["\n".join(l for l in page.splitlines() if l.strip() not in repeated)
            for page in pages]


def load_documents():
    """Load every PDF in data/documents, clean it, keep source + page metadata."""
    documents = []
    pdf_files = sorted(DOCUMENT_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print(f"WARNING: no PDF files found in {DOCUMENT_FOLDER}")
        return documents

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        pages = PyPDFLoader(str(pdf_file)).load()
        texts = remove_repeated_lines([p.page_content for p in pages])

        empty_pages = 0
        for page_number, text in enumerate(texts, start=1):
            text = clean_text(text)
            if len(text) < 20:
                empty_pages += 1
                continue
            documents.append(Document(
                page_content=text,
                # small, clean metadata: file name and 1-based page number
                metadata={"source": pdf_file.name, "page": page_number},
            ))

        if empty_pages:
            print(f"  WARNING: {empty_pages} page(s) had no readable text "
                  f"(scanned PDF? it would need OCR)")

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print("\nTotal pages loaded:", len(documents))

    if documents:
        print("\nFirst page:")
        print(documents[0].page_content[:1000])
        print("\nMetadata:", documents[0].metadata)
