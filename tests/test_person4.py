import sys
from pathlib import Path

# Allows tests to run from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools import add_event, calculate_cgpa, list_events, remove_event
from src.memory import add_message, clear_history, get_history


def test_cgpa():
    result = calculate_cgpa([
        ("O", 4),
        ("A", 3),
        ("B+", 3),
    ])
    expected = round((10 * 4 + 8 * 3 + 7 * 3) / 10, 2)
    assert result == expected


def test_invalid_grade():
    try:
        calculate_cgpa([("XYZ", 3)])
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_memory():
    session = "test-session-123"
    clear_history(session)

    add_message(session, "user", "What is the attendance requirement?")
    add_message(session, "assistant", "The requirement is 75%.")
    history = get_history(session)

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert "attendance" in history[0]["content"]

    clear_history(session)
    assert get_history(session) == []


def test_calendar():
    title = "Person 4 Test Event"
    event = add_event(title, "2099-01-01", "exam", "Test event")

    events = list_events("2099-01-01", "2099-01-01")
    assert any(e["id"] == event["id"] for e in events)

    assert remove_event(event["id"]) is True
    assert not any(e["id"] == event["id"] for e in list_events())


if __name__ == "__main__":
    test_cgpa()
    test_invalid_grade()
    test_memory()
    test_calendar()
    print("All Person 4 tests passed.")
