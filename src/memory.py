"""
Person 4 - Conversation memory.

Stores conversation history separately for each session_id.
This is deliberately implemented as plain Python so Person 3 can plug it
into LangGraph without another framework dependency.

Storage:
    data/chat_memory.json

A session contains messages such as:
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "..."}
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_FILE = DATA_DIR / "chat_memory.json"
_LOCK = Lock()


def _load_memory() -> dict[str, list[dict[str, str]]]:
    if not MEMORY_FILE.exists():
        return {}

    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_memory(memory: dict[str, list[dict[str, str]]]) -> None:
    with MEMORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def add_message(session_id: str, role: str, content: str) -> None:
    """
    Add one message to a session.

    role should normally be 'user' or 'assistant'.
    """
    session_id = str(session_id).strip()
    role = str(role).strip().lower()
    content = str(content).strip()

    if not session_id:
        raise ValueError("session_id cannot be empty.")
    if role not in {"user", "assistant", "system"}:
        raise ValueError("role must be user, assistant, or system.")
    if not content:
        raise ValueError("content cannot be empty.")

    with _LOCK:
        memory = _load_memory()
        memory.setdefault(session_id, []).append(
            {"role": role, "content": content}
        )
        _save_memory(memory)


def get_history(session_id: str, max_messages: int | None = None) -> list[dict[str, str]]:
    """
    Return conversation history for a session.

    max_messages limits the returned history from the most recent messages.
    """
    session_id = str(session_id).strip()

    if not session_id:
        raise ValueError("session_id cannot be empty.")

    with _LOCK:
        memory = _load_memory()
        history = list(memory.get(session_id, []))

    if max_messages is not None:
        if max_messages < 0:
            raise ValueError("max_messages cannot be negative.")
        history = history[-max_messages:] if max_messages else []

    return history


def clear_history(session_id: str) -> None:
    """Delete all conversation messages for one session."""
    session_id = str(session_id).strip()

    if not session_id:
        raise ValueError("session_id cannot be empty.")

    with _LOCK:
        memory = _load_memory()
        memory.pop(session_id, None)
        _save_memory(memory)


def build_chat_history(session_id: str, user_message: str) -> list[dict[str, str]]:
    """
    Convenience helper for Person 2.

    It returns previous history and then records the new user message.
    This is useful before calling:
        answer_question(question, chat_history)
    """
    history = get_history(session_id)
    add_message(session_id, "user", user_message)
    return history
