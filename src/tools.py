"""
Person 4 - External Tools for the AI Academic Assistant.

Plain Python functions intended to be called by Person 3's LangGraph workflow.
No LangChain/LangGraph dependency is required here.

Main features:
- CGPA calculation
- Simple academic calendar/event management
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CALENDAR_FILE = DATA_DIR / "calendar_events.json"

# IMPORTANT:
# Replace/verify this mapping using the college's official academic regulations.
# The project specification says the final mapping must match the real college rules.
DEFAULT_GRADE_POINTS = {
    "O": 10.0,
    "A+": 9.0,
    "A": 8.0,
    "B+": 7.0,
    "B": 6.0,
    "C": 5.0,
    "P": 4.0,
    "F": 0.0,
}


def _normalise_grade(grade: str) -> str:
    return str(grade).strip().upper()


def calculate_cgpa(
    grades_and_credits: Iterable[tuple[str, float]],
    grade_points: dict[str, float] | None = None,
) -> float:
    """
    Calculate weighted CGPA.

    Example:
        calculate_cgpa([("A", 4), ("B+", 3), ("O", 3)])

    Returns:
        CGPA rounded to 2 decimal places.

    Args:
        grades_and_credits: iterable of (grade, credit) pairs.
        grade_points: optional mapping to override the default mapping.
    """
    mapping = grade_points or DEFAULT_GRADE_POINTS

    total_points = 0.0
    total_credits = 0.0
    count = 0

    for grade, credits in grades_and_credits:
        grade = _normalise_grade(grade)

        if grade not in mapping:
            raise ValueError(
                f"Unknown grade '{grade}'. Available grades: "
                f"{', '.join(mapping.keys())}"
            )

        try:
            credits = float(credits)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid credit value for grade {grade}") from exc

        if credits <= 0:
            raise ValueError(f"Credits must be greater than 0: {credits}")

        total_points += mapping[grade] * credits
        total_credits += credits
        count += 1

    if count == 0 or total_credits == 0:
        raise ValueError("At least one valid grade/credit pair is required.")

    return round(total_points / total_credits, 2)


def _load_events() -> list[dict]:
    if not CALENDAR_FILE.exists():
        return []

    try:
        with CALENDAR_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_events(events: list[dict]) -> None:
    with CALENDAR_FILE.open("w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)


def add_event(
    title: str,
    date: str,
    event_type: str = "academic",
    description: str = "",
) -> dict:
    """
    Add an academic event.

    date must be YYYY-MM-DD.
    Returns the created event.
    """
    title = title.strip()
    event_type = event_type.strip() or "academic"

    if not title:
        raise ValueError("Event title cannot be empty.")

    try:
        parsed = datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Date must use YYYY-MM-DD format.") from exc

    events = _load_events()

    event = {
        "id": max((int(e.get("id", 0)) for e in events), default=0) + 1,
        "title": title,
        "date": parsed.strftime("%Y-%m-%d"),
        "event_type": event_type,
        "description": description.strip(),
    }

    events.append(event)
    events.sort(key=lambda e: (e["date"], e["id"]))
    _save_events(events)
    return event


def list_events(
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """
    List stored events, optionally within an inclusive date range.
    Dates use YYYY-MM-DD.
    """
    if start_date:
        datetime.strptime(start_date, "%Y-%m-%d")
    if end_date:
        datetime.strptime(end_date, "%Y-%m-%d")

    if start_date and end_date and start_date > end_date:
        raise ValueError("start_date cannot be after end_date.")

    events = _load_events()

    if start_date:
        events = [e for e in events if e["date"] >= start_date]
    if end_date:
        events = [e for e in events if e["date"] <= end_date]

    return events


def remove_event(event_id: int) -> bool:
    """Remove an event by ID. Returns True if an event was removed."""
    events = _load_events()
    new_events = [e for e in events if int(e.get("id", -1)) != int(event_id)]

    if len(new_events) == len(events):
        return False

    _save_events(new_events)
    return True
