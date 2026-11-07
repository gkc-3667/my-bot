"""Measure the deterministic reasoning engine on held-out JSONL cases."""
import argparse
import json
from pathlib import Path

from brain.reasoning import solve_question


def load_cases(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at line {line_number}: {error}") from error
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
            raise ValueError(f"Line {line_number}: prompt must be a string.")
        expected = row.get("expected")
        if isinstance(expected, str):
            expected = [expected]
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"Line {line_number}: expected must be a non-empty list.")
        if not all(isinstance(value, str) for value in expected):
            raise ValueError(f"Line {line_number}: every expected value must be a string.")
        rows.append((row["prompt"], expected))
    if not rows:
        raise ValueError("Evaluation data is empty.")
    return rows


def evaluate_cases(cases):
    results = []
    for prompt, expected in cases:
        result = solve_question(prompt)
        output = result.answer if result else ""
        normalized = output.casefold()
        passed = bool(result) and any(
            target.casefold() in normalized for target in expected
        )
        results.append(
            {
                "prompt": prompt,
                "expected": expected,
                "output": output,
                "passed": passed,
                "method": result.method if result else "",
                "steps": list(result.steps) if result else [],
                "confidence": result.confidence if result else 0.0,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data/evaluation/reasoning_smoke.jsonl"
    )
    args = parser.parse_args()
    results = evaluate_cases(load_cases(args.data))
    for index, item in enumerate(results, 1):
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status} {index}: {item['prompt']}")
        if item["output"]:
            print(f"  Answer: {item['output']}")
            print(f"  Method: {item['method']} (confidence {item['confidence']:.0%})")
        else:
            print("  No supported derivation found.")
    passed = sum(item["passed"] for item in results)
    print(f"\nSymbolic reasoning score: {passed}/{len(results)} = {passed / len(results):.1%}")
    print("This measures only the explicit rule patterns in this engine.")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
