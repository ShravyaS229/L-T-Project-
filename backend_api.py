"""Frontend-facing API for the AI Academic Assistant.

The functions in this module adapt the existing LangGraph, planner, memory,
and RAG implementations to a small, consistent interface for a frontend.
"""

from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

_SESSION_PLANS: dict[str, dict[str, Any]] = {}
_PLAN_LOCK = RLock()
_NO_INFO_MESSAGE = "I couldn't find this information in the college documents."


def _session_key(session_id: str) -> str:
    """Validate and normalize a frontend session identifier."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("A non-empty session_id is required.")
    return session_id.strip()


def _source_documents(sources: Any) -> list[dict[str, Any]]:
    """Convert existing RAG source labels into frontend-friendly metadata."""
    documents = []
    if not isinstance(sources, list):
        return documents

    for item in sources:
        if isinstance(item, dict):
            documents.append(dict(item))
            continue
        label = str(item)
        match = re.fullmatch(r"(.+), Page (\d+)", label)
        if match:
            documents.append({"source": match.group(1), "page": int(match.group(2))})
        else:
            documents.append({"source": label, "page": None})
    return documents


def ask_question(query: str, session_id: str) -> dict[str, Any]:
    """Ask the LangGraph assistant and return its answer, citations, and nodes.

    The response has ``answer``, ``source_documents`` (source/page metadata),
    and ``nodes`` (LangGraph node names in execution order).
    """
    if not isinstance(query, str) or not query.strip():
        return {
            "answer": "Please enter a question.",
            "source_documents": [],
            "nodes": [],
        }

    try:
        session = _session_key(session_id)
        from src.assistant import _GRAPH

        initial_state = {
            "message": query.strip(),
            "session_id": session,
            "intent": "",
            "reply": "",
            "sources": [],
            "history": [],
        }
        final_state = dict(initial_state)
        nodes = []

        for update in _GRAPH.stream(initial_state, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            for node_name, node_state in update.items():
                nodes.append(node_name)
                if isinstance(node_state, dict):
                    final_state.update(node_state)

        answer = final_state.get("reply") or _NO_INFO_MESSAGE
        return {
            "answer": str(answer),
            "source_documents": _source_documents(final_state.get("sources", [])),
            "nodes": nodes,
        }
    except Exception:
        logger.exception("Assistant request failed")
        return {
            "answer": "I couldn't access the college assistant right now. Please try again.",
            "source_documents": [],
            "nodes": [],
        }


def _plan_result_error(message: str) -> dict[str, Any]:
    """Create the standard planner response for API-layer errors."""
    return {"status": "error", "message": message, "plan": [], "warnings": []}


def _run_planner(
    subjects: list,
    exam_dates: dict[str, str],
    hours_per_day: float,
) -> dict[str, Any]:
    """Call the existing planner and normalize unexpected failures."""
    try:
        from src.planner import create_study_plan as planner_create_study_plan

        return planner_create_study_plan(subjects, exam_dates, hours_per_day)
    except Exception:
        logger.exception("Study plan generation failed")
        return _plan_result_error("Could not create the study plan. Please try again.")


def create_study_plan(
    subjects: list,
    hours_per_day: float,
    exam_date: str,
    session_id: str,
) -> dict:
    """Create a plan using one exam date shared by all subjects.

    Successful and warning plans are retained in process memory by session so
    ``modify_study_plan`` can use them as its starting point.
    """
    try:
        session = _session_key(session_id)
    except ValueError as exc:
        return _plan_result_error(str(exc))

    if not isinstance(subjects, list):
        return _plan_result_error("subjects must be a list of subject names.")

    exam_dates = {subject: exam_date for subject in subjects if isinstance(subject, str)}
    result = _run_planner(subjects, exam_dates, hours_per_day)
    if result.get("status") in {"ok", "warning"}:
        with _PLAN_LOCK:
            _SESSION_PLANS[session] = {
                "subjects": list(subjects),
                "hours_per_day": hours_per_day,
                "exam_dates": dict(exam_dates),
            }
    return result


def _call_llm(messages: list[dict[str, str]]) -> str:
    """Call the configured OpenAI-compatible chat model without retrieval."""
    from src.rag_chain import get_client

    model = os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("LLM_MODEL is not configured.")

    response = get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("The LLM returned an empty response.")
    return content.strip()


def ask_basic_llm(query: str) -> str:
    """Ask the configured LLM directly, without document retrieval."""
    if not isinstance(query, str) or not query.strip():
        return "Please enter a question."

    try:
        return _call_llm([
            {"role": "user", "content": query.strip()},
        ])
    except Exception:
        logger.exception("Direct LLM request failed")
        return "I couldn't reach the language model right now. Please try again."


def _parse_plan_update(content: str) -> dict[str, Any]:
    """Parse the strict JSON structure requested for a plan modification."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        update = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("The requested plan changes could not be interpreted.") from exc

    if not isinstance(update, dict):
        raise ValueError("The requested plan changes could not be interpreted.")
    if not isinstance(update.get("subjects"), list):
        raise ValueError("The updated plan must include a subjects list.")
    if not isinstance(update.get("exam_dates"), dict):
        raise ValueError("The updated plan must include exam_dates by subject.")
    if "hours_per_day" not in update:
        raise ValueError("The updated plan must include hours_per_day.")
    return update


def modify_study_plan(instruction: str, session_id: str) -> dict:
    """Apply a natural-language change to the current session's study plan.

    The configured LLM translates the instruction into planner inputs; the
    existing deterministic planner validates those inputs and creates the
    revised schedule. Plans are process-local and must be created again after
    a backend restart.
    """
    try:
        session = _session_key(session_id)
    except ValueError as exc:
        return _plan_result_error(str(exc))

    if not isinstance(instruction, str) or not instruction.strip():
        return _plan_result_error("Please describe how you want to change the plan.")

    with _PLAN_LOCK:
        current = deepcopy(_SESSION_PLANS.get(session))
    if current is None:
        return _plan_result_error(
            "There is no study plan for this session. Create a plan before modifying it."
        )

    prompt = (
        "Update the student's study-plan inputs according to their instruction. "
        "Return only a JSON object with exactly these fields: subjects (array of "
        "strings), hours_per_day (number), and exam_dates (object mapping every "
        "subject to a YYYY-MM-DD date). Preserve current values unless instructed "
        "to change them. Do not create or explain the schedule.\n\n"
        f"Current inputs: {json.dumps(current)}\n"
        f"Instruction: {instruction.strip()}"
    )

    try:
        update = _parse_plan_update(_call_llm([{"role": "user", "content": prompt}]))
        result = _run_planner(
            update["subjects"],
            update["exam_dates"],
            update["hours_per_day"],
        )
        if result.get("status") in {"ok", "warning"}:
            with _PLAN_LOCK:
                _SESSION_PLANS[session] = {
                    "subjects": list(update["subjects"]),
                    "hours_per_day": update["hours_per_day"],
                    "exam_dates": dict(update["exam_dates"]),
                }
        return result
    except ValueError as exc:
        logger.warning("Study plan modification input rejected: %s", exc)
        return _plan_result_error(str(exc))
    except Exception:
        logger.exception("Study plan modification failed for session %s", session)
        return _plan_result_error(
            "Could not modify the study plan. Check the LLM configuration and try again."
        )


def reset_session(session_id: str) -> None:
    """Clear conversation memory and any in-process study plan for a session."""
    session = _session_key(session_id)
    try:
        from src.memory import clear_history

        clear_history(session)
    except Exception:
        logger.exception("Could not clear conversation memory for session %s", session)
        raise
    finally:
        with _PLAN_LOCK:
            _SESSION_PLANS.pop(session, None)