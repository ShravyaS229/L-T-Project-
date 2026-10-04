from retriever import get_vector_db, search_documents

# (question, keyword that must appear in the top results, file it must come from)
TESTS = [
    ("What are the attendance requirements?", "75%", "academic_regulations.pdf"),
    ("How do I register for examinations?", "15 days", "academic_regulations.pdf"),
    ("How are students graded?", "Outstanding", "academic_regulations.pdf"),
    ("What happens if a student is caught cheating?", "malpractice", "academic_regulations.pdf"),
    ("What documents do I need before starting an internship?", "offer letter", "internship_guidelines.pdf"),
    ("How do I get a duplicate ID card?", "300", "student_faq.pdf"),
    ("How many books can I borrow from the library?", "14 days", "student_faq.pdf"),
]


def normalize(text):
    return " ".join(text.split()).lower()


def test_retrieval():
    print("Chunks in database:", get_vector_db()._collection.count())

    passed = 0
    for question, keyword, source in TESTS:
        results = search_documents(question, k=3)

        combined = normalize(" ".join(r["text"] for r in results))
        keyword_ok = keyword.lower() in combined
        source_ok = any(r["source"] == source for r in results)
        ok = keyword_ok and source_ok
        passed += ok

        print("\n" + "=" * 70)
        print(("PASS" if ok else "FAIL"), "-", question)
        print(f"  expected '{keyword}' from {source}  "
              f"(keyword found: {keyword_ok}, source found: {source_ok})")

        for i, r in enumerate(results, start=1):
            preview = " ".join(r["text"].split())[:150]
            print(f"  {i}. {r['source']} p.{r['page']}  (distance {r['score']:.3f})")
            print(f"     {preview}...")

    print("\n" + "=" * 70)
    print(f"Passed {passed}/{len(TESTS)} retrieval tests")


if __name__ == "__main__":
    test_retrieval()
