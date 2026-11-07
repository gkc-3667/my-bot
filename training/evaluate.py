"""Evaluate a checkpoint on held-out prompt/expected-answer JSONL data.

This is a small exact-answer smoke test, not a comprehensive intelligence
benchmark. Keep evaluation questions out of all training datasets.
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch

from brain.generate import generate
from brain.tokenizer import ByteTokenizer
from brain.transformer import AIConfig, TKCAI


def normalize_answer(text):
    text = unicodedata.normalize("NFC", str(text).casefold())
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def load_cases(path):
    cases = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Line {line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
            raise ValueError(f"Line {line_number}: expected a string prompt")
        expected = row.get("expected")
        if isinstance(expected, str):
            expected = [expected]
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(value, str) and value.strip() for value in expected)
        ):
            raise ValueError(
                f"Line {line_number}: expected must be a non-empty string or string list"
            )
        cases.append({"prompt": row["prompt"], "expected": expected})
    if not cases:
        raise ValueError("Evaluation dataset is empty.")
    return cases


def answer_matches(output, expected_values):
    normalized_output = " " + normalize_answer(output) + " "
    for value in expected_values:
        normalized_expected = normalize_answer(value)
        if normalized_expected and (
            " " + normalized_expected + " "
        ) in normalized_output:
            return True
    return False


@torch.no_grad()
def evaluate(model, tokenizer, cases, device, max_new_tokens=100):
    results = []
    for index, case in enumerate(cases, 1):
        prompt = "Người dùng: " + case["prompt"] + "\nTrợ lý:"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = prompt_ids[-model.position_embedding.num_embeddings:]
        generated_ids = generate(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=1,
            device=device,
        )
        output = tokenizer.decode(generated_ids[len(prompt_ids):]).strip()
        passed = answer_matches(output, case["expected"])
        results.append(
            {
                "index": index,
                "passed": passed,
                "expected": case["expected"],
                "output": output,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tkc_ai_instruct.pt")
    parser.add_argument(
        "--data", default="data/evaluation/reasoning_smoke.jsonl"
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = AIConfig()
    for key, value in state["config"].items():
        if hasattr(config, key):
            setattr(config, key, value)
    model = TKCAI(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    cases = load_cases(args.data)
    results = evaluate(
        model, ByteTokenizer(), cases, device, args.max_new_tokens
    )
    passed = sum(item["passed"] for item in results)
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(
            f"{status} {item['index']}: expected={item['expected']}; "
            f"generated={item['output']!r}"
        )
    print(f"Accuracy: {passed}/{len(results)} = {passed / len(results):.1%}")
    print("This smoke score is not a measure of broad reasoning ability.")


if __name__ == "__main__":
    main()
