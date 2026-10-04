import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_chain import ask_question

print("AI Academic Assistant (type 'exit' to quit)")

while True:
    question = input("\nYou: ").strip()
    if question.lower() in ("exit", "quit", "q"):
        break
    if not question:
        continue

    try:
        result = ask_question(question)
    except Exception as e:
        print("Error:", e)
        continue

    print("\nAssistant:", result["answer"])
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print(" -", s)
