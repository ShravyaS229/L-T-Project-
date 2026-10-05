# Five-Minute Demo Script

## Before the Demo

- Start the app with `streamlit run app.py`.
- Confirm `.env` has a working `LLM_API_KEY` and `LLM_MODEL`.
- Build the local index with `python src\build_vector_db.py`.
- Keep the app open on **Chat Assistant**. Choose a future exam date for the planner demo.

## Timed Flow

| Time | Screen/action | Prompt and presenter cue |
|---|---|---|
| 0:00-0:20 | Introduce | “This assistant combines college-document search with planning and academic tools. The chat answers from the supplied PDFs and shows citations when retrieval is used.” |
| 0:20-1:05 | Direct question and follow-up | Ask: **“What is the minimum attendance required in each course?”** Point out the 75% requirement. Then ask: **“What happens if I fall short?”** Show the condonation range, fee, and below-65% outcome. |
| 1:05-1:35 | RAG answer with sources | Ask: **“Which documents must I submit before starting an internship?”** Open the source expander and point to `internship_guidelines.pdf` and its page citation. |
| 1:35-1:55 | Unknown question | Ask: **“Who won the cricket world cup?”** Show that the assistant says the answer is not in the college documents instead of presenting it as college policy. |
| 1:55-2:30 | Tool usage | Ask: **“Calculate my CGPA: O (4 credits), A (3 credits), B+ (3 credits).”** The configured example grade mapping gives 8.5. Mention that official grade points must be verified for the institution. |
| 2:30-3:15 | Create a study plan | Open **Study Planner**. Enter **Maths, Physics, Chemistry, Biology**, set **3 hours per day**, choose an exam date a few weeks ahead, and select **Generate Plan**. Point out the schedule table and daily totals. |
| 3:15-3:50 | Modify the plan | Request: **“Add extra revision for Maths and increase study time to 4 hours per day.”** Select **Update Plan** and show the revised schedule. Be clear that the current planner represents subjects, exam dates, and daily hours; it does not yet store a separate revision-intensity value per subject. |
| 3:50-4:35 | RAG vs Basic LLM | Open **RAG vs Basic LLM** and compare: **“What is the minimum attendance required in each course?”** Contrast the general answer with the document-grounded answer and its source citation. |
| 4:35-5:00 | Close | Recap: document-specific answers with citations, a clear fallback for unsupported questions, and connected planning/calculation features. Note that answer quality depends on the indexed documents and configured model. |

## If Time Allows

From Chat Assistant, demonstrate a calendar tool request such as **“Add exam Maths Final on 2026-11-10”**, then **“Show my calendar events.”** Use a future date appropriate to the demo and remove the event afterward if desired.
