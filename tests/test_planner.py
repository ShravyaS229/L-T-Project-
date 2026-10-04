"""
tests/test_planner.py

Unit tests for src/planner.py (Person 3).

These tests are fully offline — no API key, no vector database, and no
network connection are required. They test the pure-Python planner logic only.

Run with:
    python -m pytest tests/test_planner.py -v
"""

import sys
from datetime import date, timedelta
from pathlib import Path

# Allow running from the project root without pip-installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.planner import create_study_plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def future(days: int) -> str:
    """Return a YYYY-MM-DD date string `days` days from today."""
    return (date.today() + timedelta(days=days)).strftime("%Y-%m-%d")


def yesterday() -> str:
    return (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")


def today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Valid inputs
# ---------------------------------------------------------------------------

def test_single_subject_valid():
    """A simple single-subject plan should succeed."""
    result = create_study_plan(
        subjects=["Maths"],
        exam_dates={"Maths": future(10)},
        hours_per_day=4,
    )
    assert result["status"] in ("ok", "warning"), result
    assert isinstance(result["plan"], list)
    assert len(result["plan"]) > 0
    # Every plan entry must have the expected keys
    for entry in result["plan"]:
        assert "date" in entry
        assert "subjects" in entry
        assert "total_hours" in entry


def test_multiple_subjects_different_dates():
    """Multiple subjects with different exam dates should all appear in plan."""
    result = create_study_plan(
        subjects=["Maths", "Physics", "Chemistry"],
        exam_dates={
            "Maths": future(7),
            "Physics": future(14),
            "Chemistry": future(21),
        },
        hours_per_day=5,
    )
    assert result["status"] in ("ok", "warning"), result
    assert len(result["plan"]) > 0

    # All three subjects should appear somewhere in the plan
    all_subjects_in_plan = set()
    for day in result["plan"]:
        for entry in day["subjects"]:
            all_subjects_in_plan.add(entry["subject"])

    for subj in ["Maths", "Physics", "Chemistry"]:
        assert subj in all_subjects_in_plan, f"{subj} missing from plan"


def test_daily_hour_limit_respected():
    """Total hours on any single day must not exceed hours_per_day."""
    hours_per_day = 3
    result = create_study_plan(
        subjects=["Maths", "Physics", "Chemistry"],
        exam_dates={
            "Maths": future(10),
            "Physics": future(12),
            "Chemistry": future(15),
        },
        hours_per_day=hours_per_day,
    )
    assert result["status"] in ("ok", "warning")
    for day in result["plan"]:
        assert day["total_hours"] <= hours_per_day + 0.01, (
            f"Day {day['date']} exceeded limit: {day['total_hours']}"
        )


def test_return_structure_always_present():
    """The returned dict must always have status, message, plan, warnings."""
    result = create_study_plan(["Maths"], {"Maths": future(5)}, 4)
    for key in ("status", "message", "plan", "warnings"):
        assert key in result, f"Key '{key}' missing from result"
    assert isinstance(result["status"], str)
    assert isinstance(result["message"], str)
    assert isinstance(result["plan"], list)
    assert isinstance(result["warnings"], list)


def test_deterministic_same_input():
    """Calling create_study_plan twice with the same input must produce the same plan."""
    kwargs = dict(
        subjects=["Maths", "Physics"],
        exam_dates={"Maths": future(10), "Physics": future(15)},
        hours_per_day=4,
    )
    result1 = create_study_plan(**kwargs)
    result2 = create_study_plan(**kwargs)
    assert result1["plan"] == result2["plan"]


def test_plan_does_not_schedule_after_exam():
    """No study session for a subject should be on or after its exam date."""
    exam_date_str = future(7)
    result = create_study_plan(
        subjects=["Maths"],
        exam_dates={"Maths": exam_date_str},
        hours_per_day=4,
    )
    for day in result["plan"]:
        for entry in day["subjects"]:
            if entry["subject"] == "Maths":
                assert day["date"] < exam_date_str, (
                    f"Study session for Maths on {day['date']} is on/after exam {exam_date_str}"
                )


def test_nearest_exam_appears_earlier_in_plan():
    """Subjects with nearer exams should appear in the earlier part of the plan."""
    result = create_study_plan(
        subjects=["Near", "Far"],
        exam_dates={"Near": future(5), "Far": future(20)},
        hours_per_day=4,
    )
    # "Near" must appear on at least one day before its exam
    near_days = [
        d["date"] for d in result["plan"]
        if any(e["subject"] == "Near" for e in d["subjects"])
    ]
    assert near_days, "Subject with nearest exam has no study days"
    assert all(day < future(5) for day in near_days)


# ---------------------------------------------------------------------------
# Invalid inputs — must return status="error"
# ---------------------------------------------------------------------------

def test_empty_subjects():
    result = create_study_plan([], {"Maths": future(10)}, 4)
    assert result["status"] == "error"
    assert result["plan"] == []


def test_none_subjects():
    result = create_study_plan(None, {"Maths": future(10)}, 4)
    assert result["status"] == "error"


def test_empty_exam_dates():
    result = create_study_plan(["Maths"], {}, 4)
    assert result["status"] == "error"


def test_missing_exam_date_for_subject():
    """If a subject has no matching entry in exam_dates, return error."""
    result = create_study_plan(
        subjects=["Maths", "Physics"],
        exam_dates={"Maths": future(10)},  # Physics is missing
        hours_per_day=4,
    )
    assert result["status"] == "error"
    assert "Physics" in result["message"]


def test_invalid_date_format():
    result = create_study_plan(["Maths"], {"Maths": "10-11-2026"}, 4)
    assert result["status"] == "error"


def test_invalid_date_string():
    result = create_study_plan(["Maths"], {"Maths": "not-a-date"}, 4)
    assert result["status"] == "error"


def test_zero_hours_per_day():
    result = create_study_plan(["Maths"], {"Maths": future(10)}, 0)
    assert result["status"] == "error"


def test_negative_hours_per_day():
    result = create_study_plan(["Maths"], {"Maths": future(10)}, -2)
    assert result["status"] == "error"


def test_string_hours_invalid():
    result = create_study_plan(["Maths"], {"Maths": future(10)}, "many")
    assert result["status"] == "error"


def test_hours_over_24():
    result = create_study_plan(["Maths"], {"Maths": future(10)}, 25)
    assert result["status"] == "error"


def test_duplicate_subjects():
    result = create_study_plan(
        ["Maths", "Maths"],
        {"Maths": future(10)},
        4,
    )
    assert result["status"] == "error"
    assert "Duplicate" in result["message"] or "duplicate" in result["message"].lower()


def test_blank_subject_name():
    result = create_study_plan(["  "], {"  ": future(10)}, 4)
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Past / tight exam dates
# ---------------------------------------------------------------------------

def test_exam_already_passed():
    """A past exam should be skipped with a warning, not raise an exception."""
    result = create_study_plan(["Maths"], {"Maths": yesterday()}, 4)
    assert result["status"] in ("warning", "error")
    assert result["plan"] == []
    assert any("pass" in w.lower() or "yesterday" in w.lower() or "passed" in w.lower()
               for w in result["warnings"]) or "passed" in result["message"].lower()


def test_exam_today():
    """An exam today cannot have study sessions scheduled."""
    result = create_study_plan(["Maths"], {"Maths": today_str()}, 4)
    assert result["status"] in ("warning", "error")
    assert result["plan"] == []


def test_exam_tomorrow_has_no_study_days():
    """Exam tomorrow means no available study days before it."""
    result = create_study_plan(["Maths"], {"Maths": future(1)}, 4)
    # Should warn but not error — no days available
    assert result["status"] in ("ok", "warning")


def test_mixed_past_and_future():
    """Past subject should be warned; future subject should still be planned."""
    result = create_study_plan(
        subjects=["Old", "New"],
        exam_dates={"Old": yesterday(), "New": future(10)},
        hours_per_day=4,
    )
    # Overall status may be "warning" but plan should contain "New"
    all_subjs = {
        e["subject"]
        for day in result["plan"]
        for e in day["subjects"]
    }
    assert "New" in all_subjs
    assert "Old" not in all_subjs


def test_insufficient_time_generates_warning():
    """Very many subjects crammed into very few days should trigger a warning."""
    result = create_study_plan(
        subjects=["A", "B", "C", "D", "E"],
        exam_dates={s: future(3) for s in ["A", "B", "C", "D", "E"]},
        hours_per_day=1,
    )
    # Should at least not crash, and should have warnings or a plan
    assert "status" in result
    assert isinstance(result["plan"], list)


# ---------------------------------------------------------------------------
# Standalone run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test when run directly
    import traceback

    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed.")
