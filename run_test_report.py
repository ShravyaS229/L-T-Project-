"""Run live assistant scenarios and save a CSV test report."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TESTS_DIR = PROJECT_ROOT / "tests"
sys.path.insert(0, str(TESTS_DIR))

from test_app import SCENARIOS, evaluate_scenario, prerequisite_problem


FIELDNAMES = ["category", "input", "expected behavior", "actual output", "pass/fail"]


def main() -> int:
    """Execute every case, print outcomes, and write test_results.csv."""
    report_rows = []

    for scenario in SCENARIOS:
        problem = prerequisite_problem(scenario)
        if problem:
            result = {
                "category": scenario.category,
                "input": scenario.input,
                "expected_behavior": scenario.expected_behavior,
                "actual_output": "NOT RUN: " + problem,
                "pass/fail": "FAIL",
                "passed": False,
            }
        else:
            result = evaluate_scenario(scenario)
        report_rows.append(result)
        print(
            f"{result['pass/fail']:4} | {scenario.category} | {scenario.name}\n"
            f"       {result['actual_output'].replace(chr(10), ' | ')}"
        )

    report_path = PROJECT_ROOT / "test_results.csv"
    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for result in report_rows:
            writer.writerow({
                "category": result["category"],
                "input": result["input"],
                "expected behavior": result["expected_behavior"],
                "actual output": result["actual_output"],
                "pass/fail": result["pass/fail"],
            })

    passed = sum(result["passed"] for result in report_rows)
    failed = len(report_rows) - passed
    print(f"\nResults: {passed} passed, {failed} failed; report saved to {report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())