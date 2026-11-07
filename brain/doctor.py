"""Run local diagnostics for GKC-AI environment and model readiness."""
import argparse
import importlib.util
import platform
from pathlib import Path

from brain.knowledge import KnowledgeBase
from brain.llm import OpenAICompatibleProvider
from brain.reasoning import solve_question


def run_diagnostics(
    checkpoint="checkpoints/gkc_ai_instruct.pt",
    corpus_dir="data/training_data",
    memory_path="data/knowledge.sqlite3",
):
    checks = []

    def add(name, ok, detail, level="info"):
        checks.append(
            {"name": name, "ok": bool(ok), "detail": str(detail), "level": level}
        )

    add("Python", platform.python_version_tuple() >= ("3", "10"), platform.python_version())
    try:
        import torch
        add("PyTorch", True, torch.__version__)
        add(
            "CUDA",
            torch.cuda.is_available(),
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU mode; no CUDA device detected",
            "info",
        )
    except ImportError:
        add("PyTorch", False, "Install the PyTorch version appropriate for this machine.", "error")

    checkpoint_path = Path(checkpoint)
    checkpoint_ready = checkpoint_path.is_file()
    add(
        "Local language-model checkpoint",
        checkpoint_ready,
        str(checkpoint_path) if checkpoint_ready else "Not found; the scratch Transformer cannot answer general questions.",
        "info" if checkpoint_ready else "warning",
    )

    provider = None
    try:
        provider = OpenAICompatibleProvider.from_env()
        add(
            "External language-model API",
            provider is not None,
            (
                f"Configured model '{provider.model}' at {provider.base_url}; live connectivity has not been tested."
                if provider is not None
                else "GKC_AI_MODEL is not set; configure an OpenAI-compatible model to enable general chat."
            ),
            "info" if provider is not None else "warning",
        )
    except ValueError as error:
        add("External language-model API", False, str(error), "error")

    model_ready = checkpoint_ready or provider is not None
    add(
        "General language model configured",
        model_ready,
        "A local checkpoint or external model is configured."
        if model_ready
        else "No trained local checkpoint or external language model is configured.",
        "info" if model_ready else "warning",
    )

    corpus_path = Path(corpus_dir)
    corpus_files = sorted(corpus_path.rglob("*.txt")) if corpus_path.exists() else []
    corpus_bytes = sum(path.stat().st_size for path in corpus_files)
    add(
        "Pretraining corpus",
        bool(corpus_files and corpus_bytes > 0),
        f"{len(corpus_files)} UTF-8 .txt file(s), {corpus_bytes} bytes" if corpus_files else "No local .txt training data yet.",
        "warning",
    )

    reasoning_cases = (
        ("Tính 15 trừ 3 nhân 4.", "3"),
        ("Tính 15% của 240.", "36"),
        ("Giải phương trình 2x + 3 = 11.", "4"),
        (
            "Nếu trời mưa thì đường ướt. Nếu đường ướt thì xe dễ trượt. Trời mưa. Có thể kết luận gì?",
            "xe dễ trượt",
        ),
    )
    reasoning_passed = 0
    for question, expected in reasoning_cases:
        result = solve_question(question)
        if result is not None and expected.casefold() in result.answer.casefold():
            reasoning_passed += 1
    add(
        "Symbolic reasoning",
        reasoning_passed == len(reasoning_cases),
        f"{reasoning_passed}/{len(reasoning_cases)} built-in smoke cases passed.",
        "error" if reasoning_passed != len(reasoning_cases) else "info",
    )

    try:
        from ddgs import DDGS  # noqa: F401
        add("Web-search package", True, "ddgs is installed; a live network search has not been tested.")
    except ImportError:
        add("Web-search package", False, "Install ddgs to enable live web search.", "warning")

    try:
        memory = KnowledgeBase(memory_path)
        add("SQLite knowledge memory", True, f"Database ready; {memory.count()} saved note(s).")
    except Exception as error:
        add("SQLite knowledge memory", False, f"{type(error).__name__}: {error}", "error")

    learned_path = Path("data/learned_examples.jsonl")
    learned_count = (
        sum(1 for line in learned_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if learned_path.is_file()
        else 0
    )
    add("Taught instruction examples", True, f"{learned_count} example(s) awaiting supervised fine-tuning.")

    return checks


def main():
    parser = argparse.ArgumentParser(description="Check GKC-AI capabilities and local prerequisites.")
    parser.add_argument("--checkpoint", default="checkpoints/gkc_ai_instruct.pt")
    parser.add_argument("--corpus-dir", default="data/training_data")
    parser.add_argument("--memory", default="data/knowledge.sqlite3")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if no general language model is configured or any error check fails.",
    )
    args = parser.parse_args()

    checks = run_diagnostics(args.checkpoint, args.corpus_dir, args.memory)
    print("GKC-AI DIAGNOSTICS")
    print("=" * 56)
    for check in checks:
        status = "OK" if check["ok"] else (
            "ERROR" if check["level"] == "error" else "NOT READY"
        )
        print(f"[{status}] {check['name']}: {check['detail']}")
    print("=" * 56)
    print("Rule-based reasoning can solve only the patterns it explicitly supports.")
    print("A trained local checkpoint or configured external language model is required for general language generation.")
    if args.strict and (
        not any(check["name"] == "General language model configured" and check["ok"] for check in checks)
        or any(not check["ok"] and check["level"] == "error" for check in checks)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
