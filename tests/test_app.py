"""Live college-assistant scenarios shared with ``run_test_report.py``.

Question scenarios require a built Chroma database and configured LLM. Plan
creation is deterministic; plan modification also requires the configured LLM.
When prerequisites are missing, pytest skips the affected live scenarios with
the reason, while the report runner records them as FAIL / not run.
"""

from __future__ import annotations

import importlib.util
import os
import re
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

import backend_api


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Scenario:
    """One realistic live evaluation case."""

    category: str
    name: str
    input: str
    expected_behavior: str
    mode: str
    required_terms: tuple[str, ...] = ()
    expected_source: str | None = None
    prior_question: str | None = None
    prior_required_terms: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()
    hours_per_day: float = 3.0
    exam_days_from_today: int = 35
    instruction: str | None = None
    expected_subjects: tuple[str, ...] = ()
    forbidden_subjects: tuple[str, ...] = ()
    max_hours_per_day: float | None = None
    require_revision_increase: bool = False


SCENARIOS = (
    Scenario(
        "Direct questions", "attendance minimum",
        "What is the minimum attendance required in each course to sit the end-semester exam?",
        "Answer with the attendance threshold and cite academic_regulations.pdf.",
        "qa", ("75%",), "academic_regulations.pdf",
    ),
    Scenario(
        "Direct questions", "duplicate ID card",
        "How do I get a duplicate ID card, what does it cost, and how long does it take?",
        "Explain the request process, Rs. 300 fee, and five-working-day issue time; cite student_faq.pdf.",
        "qa", ("300", "5 working days"), "student_faq.pdf",
    ),
    Scenario(
        "Direct questions", "library borrowing",
        "How many library books may I borrow, and for how long?",
        "State the four-book limit and 14-day loan period; cite student_faq.pdf.",
        "qa", ("4", "14 days"), "student_faq.pdf",
    ),
    Scenario(
        "Follow-up questions", "attendance shortage",
        "What happens if I fall short?",
        "Use the preceding attendance question to explain condonation and the below-65% outcome; cite academic_regulations.pdf.",
        "follow_up", ("65%", "74.99%", "500"), "academic_regulations.pdf",
        prior_question="What is the minimum attendance required in each course?",
        prior_required_terms=("75%",),
    ),
    Scenario(
        "Follow-up questions", "late exam registration",
        "What if I miss that registration deadline?",
        "Use the exam-registration context to state the late deadline and late fee; cite academic_regulations.pdf.",
        "follow_up", ("5 days", "1000"), "academic_regulations.pdf",
        prior_question="How early must I register for end-semester examinations?",
        prior_required_terms=("15 days",),
    ),
    Scenario(
        "Follow-up questions", "internship completion",
        "What must I submit after I return, and by when?",
        "Use the internship context to explain the final report/certificate and 15-day deadline; cite internship_guidelines.pdf.",
        "follow_up", ("15 days", "completion certificate"), "internship_guidelines.pdf",
        prior_question="What documents do I need before starting my internship?",
        prior_required_terms=("offer letter", "consent"),
    ),
    Scenario(
        "RAG-based questions", "exam registration rules",
        "How early must I register for exams, and what is allowed for late registration?",
        "Answer from the exam-registration section, including the normal and late deadlines/fee; cite academic_regulations.pdf.",
        "qa", ("15 days", "5 days", "1000"), "academic_regulations.pdf",
    ),
    Scenario(
        "RAG-based questions", "internship documents",
        "Which documents must I submit to the Training and Placement Cell before an internship starts?",
        "List the offer letter, signed approval form, and consent form; cite internship_guidelines.pdf.",
        "qa", ("offer letter", "approval form", "consent"), "internship_guidelines.pdf",
    ),
    Scenario(
        "RAG-based questions", "lost hall ticket",
        "I lost my hall ticket. What should I do?",
        "Explain the student-portal re-download and Examination Cell alternative; cite student_faq.pdf.",
        "qa", ("student portal", "examination cell"), "student_faq.pdf",
    ),
    Scenario(
        "Unknown questions", "cricket world cup",
        "Who won the cricket world cup?",
        "Say this information is not in the college documents; do not answer from general knowledge.",
        "unknown",
    ),
    Scenario(
        "Unknown questions", "observatory contact",
        "What is the emergency phone extension for the campus observatory?",
        "Say this information is not in the college documents instead of inventing a number.",
        "unknown",
    ),
    Scenario(
        "Unknown questions", "Monday dress code",
        "What color shirt is required by the university dress code on Mondays?",
        "Say this information is not in the college documents instead of guessing a dress code.",
        "unknown",
    ),
    Scenario(
        "Multi-step requests", "extra Maths revision",
        "Create a study plan for Maths, Physics, Chemistry, and Biology at 3 hours per day, then add extra revision for Maths and increase study time to 4 hours per day.",
        "Create and modify a valid plan; keep all four subjects, schedule Maths, and respect the 4-hour daily cap. The current planner has no per-subject revision-intensity field.",
        "multi_step", subjects=("Maths", "Physics", "Chemistry", "Biology"),
        hours_per_day=3.0, instruction="Add extra revision for Maths and increase study time to 4 hours per day.",
        expected_subjects=("Maths", "Physics", "Chemistry", "Biology"),
        max_hours_per_day=4.0, require_revision_increase=True,
    ),
    Scenario(
        "Multi-step requests", "add another subject",
        "Create a 3-hour daily plan for Maths, Physics, Chemistry, and Biology, then add Data Structures and keep the daily limit at 3 hours.",
        "Create and modify a valid plan that includes Data Structures and keeps each day within 3 hours.",
        "multi_step", subjects=("Maths", "Physics", "Chemistry", "Biology"),
        hours_per_day=3.0, instruction="Add Data Structures and keep study time at 3 hours per day.",
        expected_subjects=("Maths", "Physics", "Chemistry", "Biology", "Data Structures"),
        max_hours_per_day=3.0,
    ),
    Scenario(
        "Multi-step requests", "remove a subject",
        "Create a 3-hour daily plan for Maths, Physics, Chemistry, and Biology, then remove Chemistry and increase study time to 5 hours per day.",
        "Create and modify a valid plan without Chemistry and keep each day within 5 hours.",
        "multi_step", subjects=("Maths", "Physics", "Chemistry", "Biology"),
        hours_per_day=3.0, instruction="Remove Chemistry and increase study time to 5 hours per day.",
        expected_subjects=("Maths", "Physics", "Biology"), forbidden_subjects=("Chemistry",),
        max_hours_per_day=5.0,
    ),
)

assert len(SCENARIOS) == 15
assert all(sum(case.category == category for case in SCENARIOS) >= 3 for category in {
    "Direct questions", "Follow-up questions", "RAG-based questions",
    "Unknown questions", "Multi-step requests",
})


def prerequisite_problem(case: Scenario) -> str | None:
    """Return why a live case cannot run, without exposing any secret values."""
    required_settings = ["LLM_API_KEY", "LLM_MODEL"]
    missing_settings = [name for name in required_settings if not os.getenv(name)]
    if missing_settings:
        return "Missing LLM configuration: " + ", ".join(missing_settings)

    required_modules = [
        "openai",
        "langchain_chroma",
        "langchain_huggingface",
        "langgraph",
    ]
    if case.mode != "multi_step":
        required_modules.extend(["chromadb", "sentence_transformers"])
    missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing_modules:
        return "Missing Python packages: " + ", ".join(missing_modules)

    if case.mode != "multi_step":
        database_file = PROJECT_ROOT / "vector_db" / "chroma.sqlite3"
        if not database_file.is_file():
            return "Vector database is missing; run `python src\\build_vector_db.py` first."
    return None


def _contains_terms(answer: str, terms: tuple[str, ...]) -> bool:
    normalized = re.sub(r"\s+", " ", answer.casefold())
    return all(term.casefold() in normalized for term in terms)


def _source_names(result: dict[str, Any]) -> list[str]:
    return [str(document.get("source", "")) for document in result.get("source_documents", [])]


def _plan_counts(result: dict[str, Any]) -> tuple[set[str], dict[str, int], bool]:
    subjects: set[str] = set()
    counts: dict[str, int] = {}
    within_daily_limit = True
    for day in result.get("plan", []):
        allocations = day.get("subjects", [])
        for allocation in allocations:
            subject = str(allocation.get("subject", ""))
            subjects.add(subject.casefold())
            counts[subject.casefold()] = counts.get(subject.casefold(), 0) + 1
        total_hours = float(day.get("total_hours", 0))
        if total_hours > float(result.get("_maximum_daily_hours", 24)) + 0.01:
            within_daily_limit = False
    return subjects, counts, within_daily_limit


def _qa_actual(result: dict[str, Any]) -> str:
    citations = ", ".join(
        f"{document.get('source', 'unknown')} page {document.get('page', '?')}"
        for document in result.get("source_documents", [])
    ) or "no sources"
    return f"Answer: {result.get('answer', '')}\nSources: {citations}"


def evaluate_scenario(case: Scenario) -> dict[str, Any]:
    """Run a scenario against backend_api and return its CSV-ready result."""
    session_id = "scenario-" + uuid.uuid4().hex
    actual = ""
    passed = False

    try:
        if case.mode in {"qa", "unknown"}:
            result = backend_api.ask_question(case.input, session_id)
            answer = str(result.get("answer", ""))
            sources = _source_names(result)
            actual = _qa_actual(result)
            if case.mode == "unknown":
                passed = "couldn't find this information in the college documents" in answer.casefold()
            else:
                passed = _contains_terms(answer, case.required_terms)
                if case.expected_source:
                    passed = passed and any(case.expected_source.casefold() in source.casefold() for source in sources)

        elif case.mode == "follow_up":
            first = backend_api.ask_question(case.prior_question or "", session_id)
            context_query = (
                f"Prior conversation:\nStudent: {case.prior_question}\n"
                f"Assistant: {first.get('answer', '')}\n\n"
                f"Follow-up question: {case.input}"
            )
            result = backend_api.ask_question(context_query, session_id)
            answer = str(result.get("answer", ""))
            sources = _source_names(result)
            actual = "Prior answer: " + str(first.get("answer", "")) + "\n" + _qa_actual(result)
            passed = (
                _contains_terms(str(first.get("answer", "")), case.prior_required_terms)
                and _contains_terms(answer, case.required_terms)
                and bool(case.expected_source)
                and any(case.expected_source.casefold() in source.casefold() for source in sources)
            )

        elif case.mode == "multi_step":
            exam_date = (date.today() + timedelta(days=case.exam_days_from_today)).isoformat()
            initial = backend_api.create_study_plan(
                list(case.subjects), case.hours_per_day, exam_date, session_id
            )
            if initial.get("status") not in {"ok", "warning"}:
                actual = "Initial plan failed: " + str(initial.get("message", initial))
                passed = False
            else:
                initial["_maximum_daily_hours"] = case.hours_per_day
                initial_subjects, initial_counts, _ = _plan_counts(initial)
                updated = backend_api.modify_study_plan(case.instruction or "", session_id)
                updated["_maximum_daily_hours"] = case.max_hours_per_day or 24.0
                updated_subjects, updated_counts, within_limit = _plan_counts(updated)
                expected_subjects = {subject.casefold() for subject in case.expected_subjects}
                forbidden_subjects = {subject.casefold() for subject in case.forbidden_subjects}
                passed = (
                    updated.get("status") in {"ok", "warning"}
                    and expected_subjects.issubset(updated_subjects)
                    and not forbidden_subjects.intersection(updated_subjects)
                    and within_limit
                )
                if case.require_revision_increase:
                    passed = passed and updated_counts.get("maths", 0) > initial_counts.get("maths", 0)
                actual = (
                    f"Initial status: {initial.get('status')}; updated status: {updated.get('status')}; "
                    f"scheduled subjects: {', '.join(sorted(updated_subjects)) or 'none'}; "
                    f"Maths sessions before/after: {initial_counts.get('maths', 0)}/"
                    f"{updated_counts.get('maths', 0)}; "
                    f"updated message: {updated.get('message', '')}"
                )
    except Exception as exc:
        actual = f"ERROR: {type(exc).__name__}: {exc}"
        passed = False
    finally:
        try:
            backend_api.reset_session(session_id)
        except Exception:
            pass

    return {
        "category": case.category,
        "input": case.input,
        "expected_behavior": case.expected_behavior,
        "actual_output": actual,
        "pass/fail": "PASS" if passed else "FAIL",
        "passed": passed,
    }


@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
    ids=[f"{case.category}-{case.name}".lower().replace(" ", "-") for case in SCENARIOS],
)
def test_college_assistant_scenario(scenario: Scenario) -> None:
    """Evaluate 15 documented, realistic user scenarios against the live API."""
    problem = prerequisite_problem(scenario)
    if problem:
        pytest.skip(problem)

    result = evaluate_scenario(scenario)
    assert result["passed"], result["actual_output"]