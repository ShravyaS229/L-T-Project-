"""
Person 3 – Study Planner  (src/planner.py)

Public interface
----------------
    create_study_plan(subjects, exam_dates, hours_per_day) -> dict

Input
-----
    subjects      : list[str]  – e.g. ["Maths", "Physics", "Chemistry"]
    exam_dates    : dict[str, str]  – subject -> date string "YYYY-MM-DD"
                    Every subject must have an entry.
    hours_per_day : int | float  – study hours available each day (> 0)

Return value
------------
A dict with the following structure:

    {
        "status": "ok" | "error" | "warning",
        "message": str,                   # human-readable summary / error text
        "plan": [                         # list of day entries (empty on error)
            {
                "date": "YYYY-MM-DD",
                "subjects": [
                    {
                        "subject": str,
                        "hours": float,   # hours allocated this day
                        "exam_date": str  # "YYYY-MM-DD"
                    },
                    ...
                ],
                "total_hours": float      # sum of hours for this day
            },
            ...
        ],
        "warnings": [str]                 # non-fatal notices (e.g. tight schedule)
    }

Scheduling approach (deterministic, explained)
----------------------------------------------
1. Parse and validate all inputs.
2. Today is Day 0 (not a study day; only future days are scheduled).
3. For each subject, count the number of full calendar days remaining before
   the exam (exam day itself is NOT a study day for that subject).
4. Sort subjects by ascending days-remaining (nearest exam first).
5. Assign study days to subjects proportionally:
   - Each subject gets at most (days_remaining) study days.
   - Hours per day are split equally among subjects scheduled on that day,
     capped at hours_per_day total.
6. The resulting schedule is built day by day from tomorrow up to the last
   exam date. Each day lists which subjects are studied and for how long.
7. Subjects with 0 days remaining are flagged as warnings, not scheduled.
8. If total required study time exceeds available time, a warning is added.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any

DATE_FORMAT = "%Y-%m-%d"
MIN_HOURS = 0.5   # minimum meaningful hours per subject per day


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def create_study_plan(
    subjects: list[str],
    exam_dates: dict[str, str],
    hours_per_day: float | int,
) -> dict[str, Any]:
    """
    Generate a day-by-day study schedule.

    See module docstring for full contract.
    """
    # ── 1. Validate inputs ──────────────────────────────────────────────────
    validation_error = _validate_inputs(subjects, exam_dates, hours_per_day)
    if validation_error:
        return _error_response(validation_error)

    hours_per_day = float(hours_per_day)
    today = date.today()
    warnings: list[str] = []

    # ── 2. Parse dates and compute days remaining ───────────────────────────
    subject_info: list[dict] = []
    skipped: list[str] = []

    for subj in subjects:
        raw_date = exam_dates[subj]
        try:
            exam_day = datetime.strptime(raw_date, DATE_FORMAT).date()
        except ValueError:
            return _error_response(
                f"Invalid date '{raw_date}' for '{subj}'. Use YYYY-MM-DD format."
            )

        days_remaining = (exam_day - today).days  # 0 = exam is today

        if days_remaining <= 0:
            # Exam today or already passed — no study days possible
            skipped.append(subj)
            if days_remaining == 0:
                warnings.append(
                    f"'{subj}' exam is today ({exam_day}). No study time can be scheduled."
                )
            else:
                warnings.append(
                    f"'{subj}' exam date {exam_day} has already passed. Skipping."
                )
            continue

        subject_info.append(
            {
                "subject": subj,
                "exam_date": exam_day,
                "days_remaining": days_remaining,
            }
        )

    if not subject_info:
        msg = (
            "No subjects with future exam dates could be scheduled. "
            + ("All exams have passed or are today." if skipped else "")
        )
        return {
            "status": "warning",
            "message": msg,
            "plan": [],
            "warnings": warnings,
        }

    # ── 3. Sort by nearest exam first ──────────────────────────────────────
    subject_info.sort(key=lambda s: s["exam_date"])

    # ── 4. Build a per-day assignment map ──────────────────────────────────
    # day_subjects[date] = list of subject names to study that day
    last_exam_day = subject_info[-1]["exam_date"]
    all_study_days = _date_range(today + timedelta(days=1), last_exam_day - timedelta(days=1))

    if not all_study_days:
        warnings.append("All exams are tomorrow or later — very limited study time available.")

    # Each subject studies on days before its own exam
    day_subjects: dict[date, list[str]] = {d: [] for d in all_study_days}

    for info in subject_info:
        subj = info["subject"]
        exam_day = info["exam_date"]
        # Valid study days for this subject: today+1 .. exam_day-1
        valid_days = [d for d in all_study_days if d < exam_day]
        for d in valid_days:
            day_subjects[d].append(subj)

    # ── 5. Compute hours per subject per day ───────────────────────────────
    plan: list[dict] = []
    total_allocated = 0.0
    total_needed_estimate = sum(
        min(s["days_remaining"], 14) * hours_per_day / len(subject_info)
        for s in subject_info
    )

    for study_day in sorted(day_subjects.keys()):
        day_subjs = day_subjects[study_day]
        if not day_subjs:
            continue

        # Nearest exam first on each day as well
        day_subjs.sort(
            key=lambda s: next(i["exam_date"] for i in subject_info if i["subject"] == s)
        )

        n = len(day_subjs)
        hours_each = round(hours_per_day / n, 2)

        if hours_each < MIN_HOURS and n > 1:
            # Too many subjects on one day — only include the most urgent ones
            max_subjects = max(1, math.floor(hours_per_day / MIN_HOURS))
            day_subjs = day_subjs[:max_subjects]
            n = len(day_subjs)
            hours_each = round(hours_per_day / n, 2)

        entries = []
        for s in day_subjs:
            exam_d = next(i["exam_date"] for i in subject_info if i["subject"] == s)
            entries.append(
                {
                    "subject": s,
                    "hours": hours_each,
                    "exam_date": exam_d.strftime(DATE_FORMAT),
                }
            )
            total_allocated += hours_each

        plan.append(
            {
                "date": study_day.strftime(DATE_FORMAT),
                "subjects": entries,
                "total_hours": round(hours_per_day if len(entries) == n else hours_each * n, 2),
            }
        )

    # ── 6. Warn about tight schedules ──────────────────────────────────────
    if total_allocated == 0:
        return {
            "status": "warning",
            "message": "No study days could be generated. Exams may all be tomorrow.",
            "plan": [],
            "warnings": warnings,
        }

    for info in subject_info:
        subj = info["subject"]
        exam_day = info["exam_date"]
        subj_days = sum(
            1 for d_entry in plan
            if any(e["subject"] == subj for e in d_entry["subjects"])
        )
        if subj_days == 0:
            warnings.append(
                f"'{subj}' (exam {exam_day}) has no scheduled study days "
                f"due to limited time or hour constraints."
            )
        elif info["days_remaining"] <= 2:
            warnings.append(
                f"'{subj}' has very little preparation time "
                f"({info['days_remaining']} day(s) until exam)."
            )

    status = "warning" if warnings else "ok"
    message = (
        f"Study plan created for {len(subject_info)} subject(s) "
        f"over {len(plan)} day(s), "
        f"{hours_per_day}h/day."
    )
    if warnings:
        message += " See 'warnings' for notices."

    return {
        "status": status,
        "message": message,
        "plan": plan,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    subjects: Any,
    exam_dates: Any,
    hours_per_day: Any,
) -> str | None:
    """
    Returns an error message string if inputs are invalid, else None.
    """
    # subjects
    if not subjects:
        return "subjects cannot be empty. Provide at least one subject name."
    if not isinstance(subjects, list):
        return "subjects must be a list of strings."
    if not all(isinstance(s, str) for s in subjects):
        return "Every item in subjects must be a string."

    cleaned = [s.strip() for s in subjects]
    if any(s == "" for s in cleaned):
        return "Subject names cannot be blank."

    # Duplicate check (case-insensitive)
    seen: set[str] = set()
    for s in cleaned:
        key = s.lower()
        if key in seen:
            return f"Duplicate subject detected: '{s}'. Remove duplicates before planning."
        seen.add(key)

    # exam_dates
    if not exam_dates:
        return "exam_dates cannot be empty. Provide a date for each subject."
    if not isinstance(exam_dates, dict):
        return "exam_dates must be a dict mapping subject name -> 'YYYY-MM-DD' date string."

    missing = [s for s in subjects if s not in exam_dates]
    if missing:
        return (
            f"Missing exam date(s) for: {', '.join(missing)}. "
            "Every subject must have a corresponding entry in exam_dates."
        )

    # hours_per_day
    try:
        h = float(hours_per_day)
    except (TypeError, ValueError):
        return f"hours_per_day must be a number, got: {hours_per_day!r}."

    if h <= 0:
        return f"hours_per_day must be greater than 0, got {h}."
    if h > 24:
        return f"hours_per_day cannot exceed 24, got {h}."

    return None


def _date_range(start: date, end: date) -> list[date]:
    """Inclusive list of dates from start to end."""
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _error_response(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "message": message,
        "plan": [],
        "warnings": [],
    }
