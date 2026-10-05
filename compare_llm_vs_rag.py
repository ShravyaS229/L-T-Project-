"""Compare direct LLM responses with answers grounded in college documents.

Run from the project root with ``python compare_llm_vs_rag.py``. The script
calls only the frontend-facing functions in ``backend_api.py`` and writes
``comparison_results.csv`` and ``comparison_report.md`` beside this file.
"""

from __future__ import annotations

import csv
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import backend_api


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "comparison_results.csv"
MARKDOWN_PATH = PROJECT_ROOT / "comparison_report.md"

SAMPLE_QUESTIONS = (
    {
        "question": "What minimum attendance is required to sit for end-semester exams?",
        "expected_terms": ("75%",),
        "expected_source": "academic_regulations.pdf",
    },
    {
        "question": "How early should I register for exams, and what happens with late registration?",
        "expected_terms": ("15 days", "5 days", "1000"),
        "expected_source": "academic_regulations.pdf",
    },
    {
        "question": "What are the passing requirements for a theory course?",
        "expected_terms": ("40%", "total"),
        "expected_source": "academic_regulations.pdf",
    },
    {
        "question": "How many points are grades O and A+ worth?",
        "expected_terms": ("O", "10", "A+", "9"),
        "expected_source": "academic_regulations.pdf",
    },
    {
        "question": "What is the minimum internship duration for credit?",
        "expected_terms": ("8 weeks",),
        "expected_source": "internship_guidelines.pdf",
    },
    {
        "question": "Which documents must I submit before starting an internship?",
        "expected_terms": ("offer letter", "approval form", "consent"),
        "expected_source": "internship_guidelines.pdf",
    },
    {
        "question": "How much does a duplicate ID card cost, and how long does it take?",
        "expected_terms": ("300", "5 working days"),
        "expected_source": "student_faq.pdf",
    },
    {
        "question": "How many library books may I borrow, and what is the loan period?",
        "expected_terms": ("4", "14 days"),
        "expected_source": "student_faq.pdf",
    },
)

CSV_COLUMNS = (
    "question",
    "basic LLM answer",
    "RAG answer",
    "sources used",
    "accuracy observation",
)


def _contains_expected_terms(answer: str, terms: tuple[str, ...]) -> bool:
    """Use simple reference-term matching as a transparent accuracy heuristic."""
    normalized = " ".join(answer.casefold().split())

    def matches(term: str) -> bool:
        normalized_term = term.casefold()
        if len(normalized_term) <= 2 and normalized_term.isalnum():
            return re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])",
                normalized,
            ) is not None
        return normalized_term in normalized

    return all(matches(term) for term in terms)


def _format_sources(source_documents: Any) -> str:
    """Format backend source metadata into a compact citation string."""
    if not isinstance(source_documents, list):
        return ""

    labels = []
    for document in source_documents:
        if isinstance(document, dict):
            source = str(document.get("source", "Unknown document"))
            page = document.get("page")
            labels.append(f"{source}, page {page}" if page is not None else source)
        else:
            labels.append(str(document))
    return "; ".join(dict.fromkeys(labels))


def _accuracy_observation(
    basic_answer: str,
    rag_answer: str,
    sources: str,
    expected_terms: tuple[str, ...],
    expected_source: str,
) -> str:
    """Describe matches against document-derived facts and expected citation."""
    basic_matches = _contains_expected_terms(basic_answer, expected_terms)
    rag_matches = _contains_expected_terms(rag_answer, expected_terms)
    source_matches = expected_source.casefold() in sources.casefold()

    basic_status = "matches" if basic_matches else "does not match"
    if rag_matches and source_matches:
        rag_status = f"matches reference facts and cites {expected_source}"
    elif rag_matches:
        rag_status = f"matches reference terms but does not cite {expected_source}"
    elif source_matches:
        rag_status = f"cites {expected_source} but misses one or more reference terms"
    else:
        rag_status = "does not match reference terms or cite the expected document"

    return (
        f"RAG {rag_status}. Basic LLM {basic_status} the reference terms; "
        "the direct answer has no document citations."
    )


def build_comparison_rows(
    basic_llm: Callable[[str], str] | None = None,
    rag_ask: Callable[[str, str], dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Call both backend APIs for each sample and return comparison rows."""
    basic_llm = basic_llm or backend_api.ask_basic_llm
    rag_ask = rag_ask or backend_api.ask_question
    session_id = "llm-rag-comparison-" + uuid.uuid4().hex
    rows = []

    try:
        for sample in SAMPLE_QUESTIONS:
            question = sample["question"]
            try:
                basic_answer = str(basic_llm(question))
            except Exception:
                logger.exception("Basic LLM failed for comparison question")
                basic_answer = "The basic LLM request failed."

            try:
                rag_result = rag_ask(question, session_id)
                if not isinstance(rag_result, dict):
                    raise TypeError("ask_question must return a dictionary.")
                rag_answer = str(rag_result.get("answer", ""))
                sources = _format_sources(rag_result.get("source_documents", []))
            except Exception:
                logger.exception("RAG request failed for comparison question")
                rag_answer = "The RAG request failed."
                sources = ""

            rows.append({
                "question": question,
                "basic LLM answer": basic_answer,
                "RAG answer": rag_answer,
                "sources used": sources,
                "accuracy observation": _accuracy_observation(
                    basic_answer,
                    rag_answer,
                    sources,
                    sample["expected_terms"],
                    sample["expected_source"],
                ),
            })
    finally:
        try:
            backend_api.reset_session(session_id)
        except Exception:
            logger.warning("Could not clear temporary comparison session", exc_info=True)

    return rows


def _markdown_cell(value: str) -> str:
    """Escape text for a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")


def write_reports(rows: list[dict[str, str]]) -> None:
    """Write the requested CSV table and Markdown report."""
    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    table_lines = [
        "| " + " | ".join(CSV_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in CSV_COLUMNS) + " |",
    ]
    for row in rows:
        table_lines.append(
            "| " + " | ".join(_markdown_cell(row[column]) for column in CSV_COLUMNS) + " |"
        )

    report = [
        "# Basic LLM vs RAG Comparison",
        "",
        f"Generated: {generated_at}",
        "",
        "Accuracy observations use document-derived reference terms and the expected source file. "
        "They are a lightweight check, not a complete human accuracy review.",
        "",
        *table_lines,
        "",
        "## Conclusion",
        "",
        "For college-specific questions, RAG is preferable because it grounds responses in the "
        "provided college documents and can cite the relevant source, making answers more "
        "document-specific than a general LLM response. This grounding reduces the risk of "
        "unsupported hallucinations; it does not guarantee that every generated answer is "
        "error-free, so citations and important claims should still be checked.",
        "",
    ]
    MARKDOWN_PATH.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    """Generate and save both comparison reports."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rows = build_comparison_rows()
    write_reports(rows)
    print(f"Compared {len(rows)} questions.")
    print(f"CSV report: {CSV_PATH}")
    print(f"Markdown report: {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()