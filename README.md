# AI Academic Assistant - Knowledge Base Module (Person 1)

Turns college PDFs into a searchable ChromaDB vector database.

    PDFs -> load + clean -> chunks -> embeddings -> ChromaDB -> retriever (for Person 2)

## Setup (Windows)

    python -m venv venv
    venv\Scripts\activate
    python -m pip install -r requirements.txt

## Build the knowledge base

1. Put the college PDFs in `data/documents/`.
2. Run (this wipes and rebuilds the collection, so it is safe to re-run):

       python src\build_vector_db.py

3. Verify retrieval:

       python src\test_retrieval.py

   Add a test line to `TESTS` in `src/test_retrieval.py` for every new document.

## Project layout

| File | Purpose |
|---|---|
| `src/config.py` | Paths, collection name, embedding model, chunk size |
| `src/document_loader.py` | Loads PDFs, cleans text, keeps `source` and `page` metadata |
| `src/text_splitter.py` | Splits pages into 1000-character chunks (200 overlap) |
| `src/embeddings.py` | Embedding model (`all-MiniLM-L6-v2`) |
| `src/build_vector_db.py` | Builds the ChromaDB collection in `vector_db/` |
| `src/retriever.py` | Interface for Person 2 |
| `src/test_retrieval.py` | Automated retrieval checks |

## For Person 2 (RAG)

Run from the project root, after `build_vector_db.py` has been run once:

    from src.retriever import search_documents, get_retriever

    results = search_documents("What is the attendance requirement?", k=3)
    # each result: {"text", "source" (file name), "page" (starts at 1), "score"}

    retriever = get_retriever(k=3)   # LangChain retriever for a RAG chain

Notes:
- `score` is a distance: lower = more similar.
- Show citations as `source` + `page`, e.g. "academic_regulations.pdf, Page 2".
- Do not change the embedding model or collection name without rebuilding the database.
- `vector_db/` is not committed to Git; every teammate runs `build_vector_db.py` themselves.

## Limitations

- Scanned (image-only) PDFs have no text layer and need OCR; the loader prints a warning for them.
- Tables in PDFs may lose their layout when extracted.

## For Person 3 (LangGraph + Study Planner)

Files: `src/planner.py`, `src/assistant.py`

- **Study planner (`src/planner.py`):** build `create_study_plan(subjects, exam_dates, hours_per_day)`. This is plain Python and does not depend on any other module, so it can be built and tested right away.
- **LangGraph workflow (`src/assistant.py`):** route each student message to the right node: a document question (Person 2's `answer_question`), a study plan (`create_study_plan`), or a tool (Person 4). Expose `run_assistant(message, session_id)`, which is the single function the frontend calls.
- Keep the `sources` list from document answers so the UI can show citations.

Agreed interfaces:

    # Person 3
    create_study_plan(subjects, exam_dates, hours_per_day) -> dict
    run_assistant(message, session_id) -> {"reply": str, "sources": list}

    # from Person 2
    answer_question(question, chat_history=None) -> {"answer": str, "sources": [{"source": str, "page": int}]}

Until the other modules are ready, use a stub that returns a sample result in the same shape.

## For Person 4 (External Tools + Conversation)

Files: `src/tools.py`, optionally `src/memory.py`

- **Calculator:** `calculate_cgpa(grades_and_credits) -> float` and similar helpers. The grade points must match the college's real regulations once the real documents are added.
- **Calendar:** simple helpers to add and list dates such as exams and deadlines.
- **Chat memory:** store the conversation history per `session_id` so follow-up questions like "and what about labs?" work. Pass the history to Person 2's `answer_question(question, chat_history)`.
- Write the tools as plain Python functions so Person 3 can plug them into the LangGraph workflow.

## For Person 5 (Frontend + Integration + Testing)

Files: `app.py` (or a `frontend/` folder), `tests/`

- **Frontend:** build the chat UI against `run_assistant(message, session_id)`. Use a mock version of that function until Person 3's real one exists:

      def run_assistant(message, session_id):
          return {"reply": "Sample reply", "sources": [{"source": "academic_regulations.pdf", "page": 2}]}

- **Citations:** show each source under the answer as `source, Page N`.
- **Integration:** when the real modules are ready, replace the stubs and run the whole flow end to end.
- **Testing:** write test questions that cover every module: document questions, study plan, CGPA calculation and follow-ups. Keep a short checklist for the demo.
- **Demo:** make sure the vector database has been built (`python src\build_vector_db.py`) with the final college documents before presenting.

## Team workflow (Git)

- Work on your own branch, for example `person2-rag`, `person3-langgraph`, `person4-tools`, `person5-ui`.
- Each person owns their own files. Do not edit another person's file without telling them.
- `requirements.txt` is shared: add your packages at the end and pull often.
- Never commit API keys. Keep them in a `.env` file (already in `.gitignore`).
- Merge into `main` through pull requests.
