import os
import sys

# lets "from src...." imports work when running: python src\test_rag.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_chain import retrieve_context

QUESTIONS = [
    "What is the attendance requirement?",
    "What is the minimum passing marks?",
]

for question in QUESTIONS:
    chunks, context = retrieve_context(question)
    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)
    print(context if context else "NO RESULTS FOUND")
