"""
Person 3 - LangGraph Assistant  (src/assistant.py)

Public interface
----------------
    run_assistant(message, session_id) -> {"reply": str, "sources": list}

Architecture
------------
The LangGraph graph is compiled once at module load.  Once compiled,
the graph holds direct references to the node function objects -- so
patching a node function in tests after compilation has no effect.

To make the logic fully testable, each node is a thin wrapper that
delegates to an internal _exec_* helper.  Python resolves the helper
name through the module's __dict__ at call time, so patching
    src.assistant._exec_rag
works even though the compiled graph holds a reference to node_rag.

Routing
-------
    A. Academic question       -> _exec_rag       (src/rag_chain.ask_question)
    B. Study-plan request      -> _exec_plan      (src/planner.create_study_plan)
    C. CGPA calculation        -> _exec_cgpa      (src/tools.calculate_cgpa)
    D. Calendar operations     -> _exec_calendar  (src/tools.add/list/remove_event)
    E. Conversation memory     -> _exec_load / _exec_save (src/memory)

Safety guarantees
-----------------
- No import-time API calls.
- No hard-coded API keys or secrets.
- All errors caught; safe strings returned to the student.
- reply is always str, sources is always list.
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AssistantState(TypedDict):
    message: str       # original student message
    session_id: str    # conversation session identifier
    intent: str        # "rag" | "plan" | "cgpa" | "calendar"
    reply: str         # final reply string
    sources: list      # source citations from RAG, else []
    history: list      # conversation history from memory


# ---------------------------------------------------------------------------
# Intent classification  (keyword-based, no LLM call)
# ---------------------------------------------------------------------------

_PLAN_PATTERNS = [
    r"\bstudy\s+plan\b", r"\bstudy\s+schedule\b", r"\bplan\s+my\s+stud",
    r"\bexam\s+schedule\b", r"\bprepare\s+for\s+exam", r"\bpreparation\s+plan\b",
    r"\bschedule.*exam", r"\bhow\s+(?:do\s+i\s+)?(?:prepare|study)\b",
    r"\bcreate.*plan\b", r"\bmake.*plan\b", r"\bbuild.*plan\b",
]
_CGPA_PATTERNS = [
    r"\bcgpa\b", r"\bsgpa\b", r"\bgrade\s+point\b", r"\bcalculate.*grade",
    r"\bmy\s+gpa\b", r"\bcompute.*cgpa\b", r"\bwhat.*(?:is|are).*my.*grade",
]
_CALENDAR_PATTERNS = [
    r"\badd\s+(?:an?\s+)?event\b", r"\bschedule\s+(?:an?\s+)?event\b",
    r"\badd\s+(?:an?\s+)?exam\b",            # "add exam ..." or "add an exam ..."
    r"\badd\b.*\bon\s+\d{4}-\d{2}-\d{2}",   # "add ... on 2026-11-15"
    r"\blist\s+events?\b", r"\bshow\s+events?\b", r"\bmy\s+calendar\b",
    r"\bremove\s+event\b", r"\bdelete\s+event\b", r"\bwhat\s+events?\b",
    r"\bupcoming\s+events?\b",
]


def _classify_intent(message: str) -> str:
    """Return 'plan', 'cgpa', 'calendar', or 'rag'."""
    msg = message.lower()
    for p in _PLAN_PATTERNS:
        if re.search(p, msg):
            return "plan"
    for p in _CGPA_PATTERNS:
        if re.search(p, msg):
            return "cgpa"
    for p in _CALENDAR_PATTERNS:
        if re.search(p, msg):
            return "calendar"
    return "rag"


# ---------------------------------------------------------------------------
# Execution helpers  -- these are the real logic; patch these in tests
# ---------------------------------------------------------------------------

def _exec_load_history(state: AssistantState) -> AssistantState:
    """Load conversation history from Person 4's memory module."""
    try:
        from src.memory import get_history
        state["history"] = get_history(state["session_id"], max_messages=10)
    except Exception as exc:
        log.warning("Could not load conversation history: %s", exc)
        state["history"] = []
    return state


def _exec_rag(state: AssistantState) -> AssistantState:
    """
    Call Person 2's RAG chain.

    ask_question(question) returns:
        {"answer": str, "sources": [str, ...]}
    Sources are strings like "filename.pdf, Page 2".
    """
    try:
        from src.rag_chain import ask_question
        result = ask_question(
            state["message"],
            chat_history=state.get("history", []),
        )
        state["reply"] = result.get("answer", "I could not retrieve an answer.")
        state["sources"] = result.get("sources", [])
    except RuntimeError as exc:
        log.error("RAG RuntimeError: %s", exc)
        state["reply"] = (
            "I need an LLM API key to answer document questions. "
            "Please ask your administrator to configure the .env file."
        )
        state["sources"] = []
    except Exception as exc:
        log.error("RAG unexpected error: %s", exc)
        state["reply"] = (
            "I encountered an error while looking up your question. "
            "Please try again in a moment."
        )
        state["sources"] = []
    return state


def _exec_plan(state: AssistantState) -> AssistantState:
    """
    Call the study planner (src/planner.py).

    Parses subjects, exam dates, and hours from the message.
    Asks a follow-up if essential information is missing.
    """
    from src.planner import create_study_plan

    subjects, exam_dates, hours_per_day = _extract_plan_params(state["message"])

    if not subjects or not exam_dates:
        state["reply"] = (
            "I'd love to create a study plan for you! Please tell me:\n\n"
            "1. **Subjects** -- e.g., Maths, Physics, Chemistry\n"
            "2. **Exam dates** -- e.g., Maths on 2026-11-10, Physics on 2026-11-15\n"
            "3. **Study hours per day** -- e.g., 4 hours per day\n\n"
            'Example: "Create a study plan for Maths on 2026-11-10 '
            'and Physics on 2026-11-15, studying 4 hours a day."'
        )
        state["sources"] = []
        return state

    if hours_per_day is None:
        hours_per_day = 4.0
        note = " (I assumed 4 study hours per day -- let me know if you want a different amount.)"
    else:
        note = ""

    try:
        result = create_study_plan(subjects, exam_dates, hours_per_day)
    except Exception as exc:
        log.error("Planner error: %s", exc)
        state["reply"] = (
            "I encountered an error while building your study plan. "
            "Please check your inputs and try again."
        )
        state["sources"] = []
        return state

    if result["status"] == "error":
        state["reply"] = "I could not create a study plan: " + result["message"]
        state["sources"] = []
        return state

    state["reply"] = _format_plan(result) + note
    state["sources"] = []
    return state


def _exec_cgpa(state: AssistantState) -> AssistantState:
    """
    Call Person 4's CGPA calculator.

    Parses grade/credit pairs from the message.
    Asks for clarification if none are found.
    """
    from src.tools import calculate_cgpa

    pairs = _extract_grade_credit_pairs(state["message"])

    if not pairs:
        state["reply"] = (
            "To calculate your CGPA, please list your grades and credits like this:\n\n"
            "  O (4 credits), A+ (3 credits), B+ (3 credits)\n\n"
            "Supported grades: O, A+, A, B+, B, C, P, F"
        )
        state["sources"] = []
        return state

    try:
        cgpa = calculate_cgpa(pairs)
        state["reply"] = (
            "Your CGPA is **" + str(cgpa) + "** "
            "(based on the grades and credits you provided)."
        )
    except ValueError as exc:
        state["reply"] = "CGPA calculation error: " + str(exc)
    except Exception as exc:
        log.error("CGPA error: %s", exc)
        state["reply"] = "An unexpected error occurred during CGPA calculation."

    state["sources"] = []
    return state


def _exec_calendar(state: AssistantState) -> AssistantState:
    """
    Route to Person 4's calendar tools.

    Supports list, add, and remove operations.
    """
    from src.tools import add_event, list_events, remove_event

    msg = state["message"].lower()
    state["sources"] = []

    # -- list ----------------------------------------------------------------
    if re.search(r"\b(?:list|show|upcoming|what)\b.*\bevents?\b", msg):
        try:
            events = list_events()
        except Exception as exc:
            log.error("Calendar list error: %s", exc)
            state["reply"] = "I could not retrieve your calendar events."
            return state

        if not events:
            state["reply"] = "Your academic calendar is currently empty."
        else:
            lines = [
                "- **" + e["title"] + "** on " + e["date"]
                + " (type: " + e["event_type"] + ")"
                for e in events
            ]
            state["reply"] = (
                "Here are your upcoming academic events:\n\n" + "\n".join(lines)
            )
        return state

    # -- remove --------------------------------------------------------------
    if re.search(r"\b(?:remove|delete)\b", msg):
        id_match = re.search(r"\bid\s*[:#=]?\s*(\d+)\b", msg)
        if not id_match:
            state["reply"] = (
                "To remove an event, please tell me the event ID. "
                "You can find IDs by listing your events."
            )
            return state
        event_id = int(id_match.group(1))
        try:
            removed = remove_event(event_id)
        except Exception as exc:
            log.error("Calendar remove error: %s", exc)
            state["reply"] = "I could not remove that event. Please try again."
            return state
        if removed:
            state["reply"] = (
                "Event #" + str(event_id) + " has been removed from your calendar."
            )
        else:
            state["reply"] = (
                "I could not find event #" + str(event_id) + " in your calendar."
            )
        return state

    # -- add -----------------------------------------------------------------
    title, event_date = _extract_event_details(state["message"])

    if not title or not event_date:
        state["reply"] = (
            "To add a calendar event, please say something like:\n\n"
            '  "Add exam Maths Final on 2026-11-10"\n\n'
            "Include the event title and date in YYYY-MM-DD format."
        )
        return state

    try:
        event = add_event(title, event_date, event_type="exam")
        state["reply"] = (
            "Added to your calendar:\n"
            "**" + event["title"] + "** on " + event["date"]
            + " (ID: " + str(event["id"]) + ")"
        )
    except ValueError as exc:
        state["reply"] = "Could not add event: " + str(exc)
    except Exception as exc:
        log.error("Calendar add error: %s", exc)
        state["reply"] = "An unexpected error occurred while adding the event."

    return state


def _exec_save_history(state: AssistantState) -> AssistantState:
    """Save the user message and assistant reply to Person 4's memory."""
    try:
        from src.memory import add_message
        add_message(state["session_id"], "user", state["message"])
        add_message(state["session_id"], "assistant", state["reply"])
    except Exception as exc:
        log.warning("Could not save conversation history: %s", exc)
    return state


# ---------------------------------------------------------------------------
# Graph node wrappers  (thin; delegate to _exec_* for testability)
#
# The compiled LangGraph graph holds direct references to these function
# objects.  Each wrapper resolves its _exec_* helper via the module's
# __dict__ at call time, so patching src.assistant._exec_rag (etc.) in
# tests is effective even after the graph has been compiled.
# ---------------------------------------------------------------------------

def node_load_history(state: AssistantState) -> AssistantState:
    return _exec_load_history(state)


def node_classify(state: AssistantState) -> AssistantState:
    state["intent"] = _classify_intent(state["message"])
    return state


def node_rag(state: AssistantState) -> AssistantState:
    return _exec_rag(state)


def node_plan(state: AssistantState) -> AssistantState:
    return _exec_plan(state)


def node_cgpa(state: AssistantState) -> AssistantState:
    return _exec_cgpa(state)


def node_calendar(state: AssistantState) -> AssistantState:
    return _exec_calendar(state)


def node_save_history(state: AssistantState) -> AssistantState:
    return _exec_save_history(state)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_by_intent(state: AssistantState) -> str:
    return state["intent"]


# ---------------------------------------------------------------------------
# Build graph (once at module load -- no API calls here)
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    graph = StateGraph(AssistantState)

    graph.add_node("load_history", node_load_history)
    graph.add_node("classify",     node_classify)
    graph.add_node("rag",          node_rag)
    graph.add_node("plan",         node_plan)
    graph.add_node("cgpa",         node_cgpa)
    graph.add_node("calendar",     node_calendar)
    graph.add_node("save_history", node_save_history)

    graph.set_entry_point("load_history")
    graph.add_edge("load_history", "classify")

    graph.add_conditional_edges(
        "classify",
        _route_by_intent,
        {"rag": "rag", "plan": "plan", "cgpa": "cgpa", "calendar": "calendar"},
    )

    for node in ("rag", "plan", "cgpa", "calendar"):
        graph.add_edge(node, "save_history")

    graph.add_edge("save_history", END)
    return graph.compile()


_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_assistant(message: str, session_id: str) -> dict[str, Any]:
    """
    Route a student message through the LangGraph workflow.

    Always returns {"reply": str, "sources": list}.
    """
    if not isinstance(message, str) or not message.strip():
        return {
            "reply": "Please enter a message and I'll do my best to help.",
            "sources": [],
        }

    session_id = str(session_id).strip()
    if not session_id:
        return {
            "reply": "A valid session_id is required to track your conversation.",
            "sources": [],
        }

    initial_state: AssistantState = {
        "message":    message.strip(),
        "session_id": session_id,
        "intent":     "",
        "reply":      "",
        "sources":    [],
        "history":    [],
    }

    try:
        final_state = _GRAPH.invoke(initial_state)
    except Exception as exc:
        log.error("Graph execution error for session %s: %s", session_id, exc)
        return {
            "reply": (
                "I encountered an unexpected error. "
                "Please try again or contact support."
            ),
            "sources": [],
        }

    reply = final_state.get("reply", "")
    sources = final_state.get("sources", [])
    if not isinstance(reply, str):
        reply = str(reply)
    if not isinstance(sources, list):
        sources = []

    return {"reply": reply, "sources": sources}


# ---------------------------------------------------------------------------
# Private extraction helpers
# ---------------------------------------------------------------------------

def _extract_plan_params(
    message: str,
) -> tuple[list[str] | None, dict[str, str] | None, float | None]:
    """
    Extract subjects, exam dates, and hours_per_day from free text.

    Returns (subjects, exam_dates, hours_per_day).
    Any component that cannot be extracted is returned as None.
    """
    hours_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\s*(?:per\s*day|a\s*day|daily)?",
        message, re.IGNORECASE,
    )
    hours_per_day = float(hours_match.group(1)) if hours_match else None

    pair_iter = re.finditer(
        r"([A-Za-z][A-Za-z\s]{1,30}?)\s+(?:on|:)\s+(\d{4}-\d{2}-\d{2})",
        message, re.IGNORECASE,
    )

    subjects: list[str] = []
    exam_dates: dict[str, str] = {}
    noise = {"exam", "test", "on", "for", "and", "the", "my", "is"}

    for m in pair_iter:
        raw = m.group(1).strip().rstrip(",")
        words = [w for w in raw.split() if w.lower() not in noise]
        if not words:
            continue
        subject = " ".join(words).title()
        date_str = m.group(2)
        if subject not in exam_dates:
            subjects.append(subject)
            exam_dates[subject] = date_str

    found_dates = re.findall(r"\d{4}-\d{2}-\d{2}", message)
    if not subjects and found_dates:
        return None, None, hours_per_day

    return (
        subjects or None,
        exam_dates or None,
        hours_per_day,
    )


def _extract_grade_credit_pairs(message: str) -> list[tuple[str, float]]:
    """Parse grade/credit pairs like 'O (4 credits), A+ (3)'.

    A+ and B+ must appear before A and B in the alternation so the regex
    engine matches the longer token first.
    """
    valid_grades = {"O", "A+", "A", "B+", "B", "C", "P", "F"}
    pairs: list[tuple[str, float]] = []

    for m in re.finditer(
        r"\b(O|A\+|B\+|A|B|C|P|F)(?=[^A-Za-z+]|$)\s*[:(]?\s*(\d+(?:\.\d+)?)\s*(?:credits?|cr)?[)]?",
        message, re.IGNORECASE,
    ):
        grade = m.group(1).upper()
        if grade in valid_grades:
            try:
                pairs.append((grade, float(m.group(2))))
            except ValueError:
                pass

    return pairs


def _extract_event_details(message: str) -> tuple[str | None, str | None]:
    """Extract an event title and YYYY-MM-DD date from a calendar request."""
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", message)
    event_date = date_match.group(0) if date_match else None

    quoted = re.search(r"['\"](.+?)['\"]", message)
    if quoted:
        title = quoted.group(1).strip()
    else:
        title_match = re.search(
            r"\b(?:add|schedule)\b\s+(?:(?:an?\s+)?(?:exam|event|test|deadline)\s+)?(.+?)"
            r"\s+(?:on|at|for)\s+\d{4}-\d{2}-\d{2}",
            message, re.IGNORECASE,
        )
        title = title_match.group(1).strip() if title_match else None

    return title, event_date


def _format_plan(result: dict) -> str:
    """Format a planner result dict into a readable Markdown string."""
    lines: list[str] = ["Study Plan\n"]

    for w in result.get("warnings", []):
        lines.append("Warning: " + w)
    if result.get("warnings"):
        lines.append("")

    plan = result.get("plan", [])
    if not plan:
        lines.append(result.get("message", "No plan could be generated."))
        return "\n".join(lines)

    for day in plan:
        lines.append("**" + day["date"] + "**")
        for entry in day["subjects"]:
            lines.append(
                "  - " + entry["subject"] + " -- " + str(entry["hours"]) + "h"
                + "  (exam: " + entry["exam_date"] + ")"
            )
        lines.append("")

    lines.append(result.get("message", ""))
    return "\n".join(lines)
