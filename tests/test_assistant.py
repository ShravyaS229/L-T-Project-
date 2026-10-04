"""
tests/test_assistant.py

Unit tests for src/assistant.py (Person 3).

WHY WE PATCH _exec_* HELPERS, NOT node_* WRAPPERS
----------------------------------------------------
LangGraph compiles the graph at module import time (_GRAPH = _build_graph()).
After compilation the graph holds *direct references* to the node function
objects (node_rag, node_cgpa, etc.).  Replacing module attributes with
    @patch("src.assistant.node_rag")
after compilation has no effect on what the compiled graph calls.

The fix: each node is a thin wrapper that calls a module-level _exec_*
helper.  Python resolves those names through the module's __dict__ at the
moment node_rag() runs -- so patching
    @patch("src.assistant._exec_rag")
replaces the entry in that __dict__ and the wrapper picks up the mock.

Label key
---------
  [MOCKED]        -- external service or helper fully replaced by a mock
  [MOCKED-save]   -- only _exec_save_history is mocked (avoids file I/O)
  [OFFLINE]       -- pure-logic test, no mocks needed

Run with:
    python -m pytest tests/test_assistant.py -v
"""

from __future__ import annotations

import sys
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def sid() -> str:
    """Unique session id per test call."""
    return "test-" + uuid.uuid4().hex[:8]


def future(days: int) -> str:
    return (date.today() + timedelta(days=days)).strftime("%Y-%m-%d")


def _noop_save(state: dict) -> dict:
    """Drop-in for _exec_save_history that skips file I/O."""
    return state


def _rag_returning(answer: str, sources: list):
    """Factory: returns a _exec_rag stand-in that injects a fixed response."""
    def _impl(state):
        state["reply"] = answer
        state["sources"] = sources
        return state
    return _impl


def _node_returning(reply: str, sources: list | None = None):
    """Generic _exec_* stand-in that injects a fixed reply."""
    def _impl(state):
        state["reply"] = reply
        state["sources"] = sources if sources is not None else []
        return state
    return _impl


# ---------------------------------------------------------------------------
# Output contract  -- {"reply": str, "sources": list} -- always
# ---------------------------------------------------------------------------

class TestOutputContract:

    def _check(self, result: Any) -> None:
        assert isinstance(result, dict)
        assert isinstance(result.get("reply"), str)
        assert isinstance(result.get("sources"), list)

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    @patch("src.assistant._exec_rag",
           side_effect=_rag_returning("75% attendance required.", ["file.pdf, Page 1"]))
    def test_contract_rag(self, _rag, _save):
        """[MOCKED] RAG route satisfies output contract."""
        from src.assistant import run_assistant
        self._check(run_assistant("What is the attendance requirement?", sid()))

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    def test_contract_plan(self, _save):
        """[MOCKED-save] Planner route satisfies output contract."""
        from src.assistant import run_assistant
        self._check(run_assistant(
            "Create a study plan for Maths on " + future(10) + " studying 4 hours a day",
            sid(),
        ))

    def test_contract_empty_message(self):
        """[OFFLINE] Blank message returns a valid response."""
        from src.assistant import run_assistant
        result = run_assistant("   ", sid())
        self._check(result)
        assert result["reply"]

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    @patch("src.assistant._exec_rag",
           side_effect=_node_returning(
               "I need an LLM API key to answer document questions."
           ))
    def test_contract_on_rag_error(self, _rag, _save):
        """[MOCKED] Error in _exec_rag still satisfies the contract."""
        from src.assistant import run_assistant
        self._check(run_assistant("What are the exam rules?", sid()))

    def test_contract_blank_session_id(self):
        """[OFFLINE] Blank session_id returns a valid response."""
        from src.assistant import run_assistant
        self._check(run_assistant("Hello", ""))


# ---------------------------------------------------------------------------
# RAG routing
# ---------------------------------------------------------------------------

class TestRagRouting:

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    @patch("src.assistant._exec_rag")
    def test_academic_question_calls_exec_rag(self, mock_rag, _save):
        """[MOCKED] Academic question invokes _exec_rag."""
        mock_rag.side_effect = _rag_returning(
            "75% attendance is required.", ["academic_regulations.pdf, Page 2"]
        )
        from src.assistant import run_assistant
        result = run_assistant("What is the attendance requirement?", sid())
        mock_rag.assert_called_once()
        assert "75%" in result["reply"]

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    @patch("src.assistant._exec_rag")
    def test_sources_preserved(self, mock_rag, _save):
        """[MOCKED] Sources returned by _exec_rag pass through unchanged."""
        expected = ["academic_regulations.pdf, Page 3", "student_faq.pdf, Page 1"]
        mock_rag.side_effect = _rag_returning("Some answer.", expected)
        from src.assistant import run_assistant
        result = run_assistant("What is the minimum passing mark?", sid())
        assert result["sources"] == expected

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    @patch("src.assistant._exec_rag")
    def test_no_fabricated_sources(self, mock_rag, _save):
        """[MOCKED] When _exec_rag returns [], sources must be []."""
        mock_rag.side_effect = _rag_returning("Not found.", [])
        from src.assistant import run_assistant
        result = run_assistant("Some obscure question", sid())
        assert result["sources"] == []

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    @patch("src.assistant._exec_rag")
    def test_rag_error_message_hides_env_var(self, mock_rag, _save):
        """[MOCKED] Safe error reply must not contain raw env-var names."""
        mock_rag.side_effect = _node_returning(
            "I need an LLM API key to answer document questions."
        )
        from src.assistant import run_assistant
        result = run_assistant("Tell me about attendance.", sid())
        assert "LLM_API_KEY" not in result["reply"]


# ---------------------------------------------------------------------------
# Study planner routing
# ---------------------------------------------------------------------------

class TestPlannerRouting:

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    def test_plan_request_with_full_info_returns_schedule(self, _save):
        """[MOCKED-save] Full plan request returns a non-empty reply."""
        from src.assistant import run_assistant
        result = run_assistant(
            "Create a study plan for Physics on " + future(14)
            + " studying 4 hours a day",
            sid(),
        )
        assert result["sources"] == []
        assert any(
            kw in result["reply"].lower()
            for kw in ("physics", "study", "plan", future(14)[:7])
        )

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    def test_plan_missing_info_asks_followup(self, _save):
        """[MOCKED-save] Vague plan request prompts for missing details."""
        from src.assistant import run_assistant
        result = run_assistant("Can you make me a study plan?", sid())
        assert result["sources"] == []
        assert any(kw in result["reply"].lower() for kw in ("subject", "exam", "date", "hour"))

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    def test_plan_sources_always_empty(self, _save):
        """[MOCKED-save] Planner responses have no document sources."""
        from src.assistant import run_assistant
        result = run_assistant(
            "Study schedule for Maths on " + future(10) + ", 3 hours a day",
            sid(),
        )
        assert result["sources"] == []


# ---------------------------------------------------------------------------
# CGPA tool routing
# ---------------------------------------------------------------------------

class TestCgpaRouting:

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    @patch("src.assistant._exec_cgpa")
    def test_cgpa_routes_to_exec_cgpa(self, mock_cgpa, _save):
        """[MOCKED] CGPA request invokes _exec_cgpa."""
        mock_cgpa.side_effect = _node_returning("Your CGPA is **8.5**")
        from src.assistant import run_assistant
        result = run_assistant("Calculate my CGPA: O (4 credits), A (3 credits)", sid())
        mock_cgpa.assert_called_once()
        assert "8.5" in result["reply"]

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    def test_cgpa_missing_grades_asks_followup(self, _save):
        """[MOCKED-save] No parseable grades -> clarification message."""
        from src.assistant import run_assistant
        result = run_assistant("What is my CGPA?", sid())
        assert result["sources"] == []
        assert any(kw in result["reply"].lower() for kw in ("grade", "credit", "cgpa", "list"))

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    @patch("src.assistant._exec_cgpa")
    def test_cgpa_error_satisfies_contract(self, mock_cgpa, _save):
        """[MOCKED] _exec_cgpa error reply must satisfy the output contract."""
        mock_cgpa.side_effect = _node_returning("CGPA calculation error: Unknown grade 'XYZ'.")
        from src.assistant import run_assistant
        result = run_assistant("Calculate CGPA: XYZ (3 credits)", sid())
        assert isinstance(result["reply"], str)
        assert isinstance(result["sources"], list)


# ---------------------------------------------------------------------------
# Calendar tool routing
# ---------------------------------------------------------------------------

class TestCalendarRouting:

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    @patch("src.assistant._exec_calendar")
    def test_list_events_invokes_exec_calendar(self, mock_cal, _save):
        """[MOCKED] 'Show my events' routes to _exec_calendar."""
        mock_cal.side_effect = _node_returning(
            "- **Maths Exam** on 2026-11-10 (type: exam)"
        )
        from src.assistant import run_assistant
        result = run_assistant("Show my calendar events", sid())
        mock_cal.assert_called_once()
        assert "Maths Exam" in result["reply"]

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    def test_empty_calendar_reported_politely(self, _save):
        """[MOCKED-save] Empty calendar -> polite message."""
        with patch("src.tools.list_events", return_value=[]):
            from src.assistant import run_assistant
            result = run_assistant("List my upcoming events", sid())
        lower = result["reply"].lower()
        assert "empty" in lower or "no event" in lower or "calendar" in lower

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    @patch("src.assistant._exec_calendar")
    def test_add_event_invokes_exec_calendar(self, mock_cal, _save):
        """[MOCKED] 'Add event' routes to _exec_calendar."""
        mock_cal.side_effect = _node_returning(
            "Added to your calendar: **Physics Final** on 2026-11-15 (ID: 5)"
        )
        from src.assistant import run_assistant
        result = run_assistant("Add exam 'Physics Final' on 2026-11-15", sid())
        mock_cal.assert_called_once()
        assert "Physics Final" in result["reply"]

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    def test_add_event_missing_info_asks_followup(self, _save):
        """[MOCKED-save] Incomplete add-event request asks for details."""
        with patch("src.tools.add_event") as mock_add:
            from src.assistant import run_assistant
            result = run_assistant("Add an event to my calendar", sid())
            mock_add.assert_not_called()
        assert "title" in result["reply"].lower() or "date" in result["reply"].lower()

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    def test_remove_event_calls_remove_event(self, _save):
        """[MOCKED-save] 'Remove event id 3' calls tools.remove_event(3)."""
        with patch("src.tools.remove_event", return_value=True) as mock_remove:
            from src.assistant import run_assistant
            result = run_assistant("Remove event id 3", sid())
            mock_remove.assert_called_once_with(3)
        assert "3" in result["reply"]

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    def test_remove_missing_id_asks_for_id(self, _save):
        """[MOCKED-save] 'Delete event' without an ID asks for the ID."""
        from src.assistant import run_assistant
        result = run_assistant("Delete an event from my calendar", sid())
        assert "id" in result["reply"].lower()

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    def test_calendar_tool_exception_satisfies_contract(self, _save):
        """[MOCKED-save] Exception inside calendar tool -> safe contract reply."""
        with patch("src.tools.add_event", side_effect=Exception("Disk full")):
            from src.assistant import run_assistant
            result = run_assistant("Add exam 'Test' on 2026-11-10", sid())
        assert isinstance(result["reply"], str)
        assert isinstance(result["sources"], list)


# ---------------------------------------------------------------------------
# Session memory
# ---------------------------------------------------------------------------

class TestSessionMemory:

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    @patch("src.assistant._exec_rag")
    def test_get_history_called_with_correct_session(self, mock_rag, _save):
        """[MOCKED] _exec_load_history calls memory.get_history(session_id)."""
        mock_rag.side_effect = _rag_returning("Answer.", [])
        session = "sess-" + uuid.uuid4().hex[:6]
        with patch("src.memory.get_history", return_value=[]) as mock_get:
            from src.assistant import run_assistant
            run_assistant("What is attendance?", session)
            mock_get.assert_called_with(session, max_messages=10)

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    @patch("src.assistant._exec_rag")
    def test_separate_sessions_independent(self, mock_rag, _save):
        """[MOCKED] Two different sessions use separate get_history calls."""
        mock_rag.side_effect = _rag_returning("A.", [])
        s1, s2 = "sess-" + uuid.uuid4().hex[:4], "sess-" + uuid.uuid4().hex[:4]
        with patch("src.memory.get_history", return_value=[]) as mock_get:
            from src.assistant import run_assistant
            run_assistant("Q1", s1)
            run_assistant("Q2", s2)
            call_sessions = [c.args[0] for c in mock_get.call_args_list]
            assert s1 in call_sessions
            assert s2 in call_sessions

    @patch("src.assistant._exec_rag")
    def test_both_user_and_assistant_messages_saved(self, mock_rag):
        """[MOCKED] After a RAG call, add_message is called for user and assistant."""
        mock_rag.side_effect = _rag_returning("Answer.", [])
        session = "save-" + uuid.uuid4().hex[:6]
        with patch("src.memory.get_history", return_value=[]):
            with patch("src.memory.add_message") as mock_add:
                from src.assistant import run_assistant
                run_assistant("Some question", session)
                saved_roles = [c.args[1] for c in mock_add.call_args_list]
                assert "user" in saved_roles
                assert "assistant" in saved_roles


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    @patch("src.assistant._exec_save_history", side_effect=_noop_save)
    @patch("src.assistant._exec_rag")
    def test_greeting_routes_to_rag(self, mock_rag, _save):
        """[MOCKED] Greetings fall through to the RAG default route."""
        mock_rag.side_effect = _rag_returning("Hello! How can I help?", [])
        from src.assistant import run_assistant
        result = run_assistant("Hello!", sid())
        mock_rag.assert_called_once()
        assert isinstance(result["reply"], str)

    def test_very_long_message_does_not_raise(self):
        """[MOCKED] Very long messages are handled without unhandled exceptions."""
        with (
            patch("src.assistant._exec_rag", side_effect=_rag_returning("OK.", [])),
            patch("src.assistant._exec_save_history", side_effect=_noop_save),
        ):
            from src.assistant import run_assistant
            long_msg = "What is " + ("the attendance requirement " * 100)
            result = run_assistant(long_msg, sid())
            assert "reply" in result

    def test_intent_classifier_deterministic(self):
        """[OFFLINE] _classify_intent always returns the same value for the same input."""
        from src.assistant import _classify_intent
        msg = "What is the attendance requirement?"
        assert _classify_intent(msg) == _classify_intent(msg)

    def test_intent_classifier_routing(self):
        """[OFFLINE] Keyword patterns map to the correct intent labels."""
        from src.assistant import _classify_intent
        assert _classify_intent("Make me a study plan") == "plan"
        assert _classify_intent("calculate my cgpa") == "cgpa"
        assert _classify_intent("List my calendar events") == "calendar"
        assert _classify_intent("What is the attendance requirement?") == "rag"

    def test_extract_plan_params_no_dates_returns_none(self):
        """[OFFLINE] A message with no dates returns (None, None, hours)."""
        from src.assistant import _extract_plan_params
        subjects, exam_dates, hours = _extract_plan_params("I want a study plan please")
        assert subjects is None
        assert exam_dates is None

    def test_extract_grade_credit_pairs_valid(self):
        """[OFFLINE] Grade/credit parser picks up valid pairs."""
        from src.assistant import _extract_grade_credit_pairs
        pairs = _extract_grade_credit_pairs("O (4 credits), A+ (3 credits), B (3)")
        assert ("O", 4.0) in pairs
        assert ("A+", 3.0) in pairs
        assert ("B", 3.0) in pairs

    def test_extract_grade_credit_pairs_empty(self):
        """[OFFLINE] No recognisable pairs returns an empty list."""
        from src.assistant import _extract_grade_credit_pairs
        assert _extract_grade_credit_pairs("What is my grade?") == []


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    klasses = [
        TestOutputContract, TestRagRouting, TestPlannerRouting,
        TestCgpaRouting, TestCalendarRouting, TestSessionMemory, TestEdgeCases,
    ]
    passed = failed = 0
    for cls in klasses:
        obj = cls()
        for name in sorted(dir(obj)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(obj, name)()
                print("  PASS  " + cls.__name__ + "." + name)
                passed += 1
            except Exception:
                print("  FAIL  " + cls.__name__ + "." + name)
                traceback.print_exc()
                failed += 1
    print("\n" + str(passed) + " passed, " + str(failed) + " failed.")
