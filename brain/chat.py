"""Interactive chat entry point for the staged TKC-AI agent."""
import argparse

import torch

from brain.agent import ResearchAgent
from brain.knowledge import KnowledgeBase
from brain.transformer import AIConfig, TKCAI


def load_agent(checkpoint_path, memory_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model" not in state or "config" not in state:
        raise ValueError("Checkpoint must contain model and config entries.")

    config = AIConfig()
    for key, value in state["config"].items():
        if hasattr(config, key):
            setattr(config, key, value)

    model = TKCAI(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return ResearchAgent(
        model=model,
        device=device,
        memory=KnowledgeBase(memory_path),
        max_search_results=5,
    )


def print_response(response):
    print("\n" + (response.answer or "[Mô hình chưa tạo được câu trả lời.]"))
    if response.sources:
        print("\nNguồn tham khảo:")
        for index, source in enumerate(response.sources, 1):
            print(f"[{index}] {source.title}\n    {source.url}")
    if response.warnings:
        print("\nLưu ý:")
        for warning in response.warnings:
            print(f"- {warning}")


def main():
    parser = argparse.ArgumentParser(
        description="Chat with TKC-AI using planning, web search, memory and self-review."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/tkc_ai_instruct.pt",
        help="A checkpoint trained by this project.",
    )
    parser.add_argument(
        "--memory",
        default="data/knowledge.sqlite3",
        help="SQLite file for user-taught knowledge.",
    )
    parser.add_argument("--question", help="Ask one question and exit.")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    try:
        agent = load_agent(args.checkpoint, args.memory)
    except FileNotFoundError:
        parser.error(
            f"Checkpoint not found: {args.checkpoint}. "
            "Train TKC-AI from scratch before starting chat."
        )

    if args.question:
        print_response(agent.reply(args.question, args.max_new_tokens))
        return

    print("TKC-AI: nhập câu hỏi; gõ 'thoát' để kết thúc.")
    while True:
        try:
            question = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã kết thúc.")
            break
        if question.casefold() in {"thoát", "exit", "quit"}:
            print("TKC-AI: Tạm biệt.")
            break
        if not question:
            continue
        try:
            print_response(agent.reply(question, args.max_new_tokens))
        except Exception as error:
            print(f"TKC-AI gặp lỗi: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
