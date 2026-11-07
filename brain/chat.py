"""Interactive chat entry point for TKC-AI, including pretraining fallback."""
import argparse
from pathlib import Path

import torch

from brain.agent import ResearchAgent
from brain.knowledge import KnowledgeBase
from brain.transformer import AIConfig, TKCAI


def load_agent(checkpoint_path, memory_path):
    memory = KnowledgeBase(memory_path)
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        return ResearchAgent(
            model=None,
            device="cpu",
            memory=memory,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
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
        memory=memory,
        max_search_results=5,
    )


def print_response(response):
    print("\n" + (response.answer or "[Chưa có câu trả lời.]"))
    if response.reasoning_method:
        print(f"\nPhương pháp suy luận: {response.reasoning_method}")
        print(f"Độ tin cậy của bộ giải quy tắc: {response.reasoning_confidence:.0%}")
        print("Các bước có thể kiểm tra:")
        for index, step in enumerate(response.reasoning_steps, 1):
            print(f"  {index}. {step}")
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
        description="Chat with TKC-AI using symbolic reasoning, web search and learned memory."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/tkc_ai_instruct.pt",
        help="A checkpoint trained by this project. Missing checkpoint enables tool-only mode.",
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
    except FileNotFoundError as error:
        parser.error(str(error))
    except (ValueError, RuntimeError, KeyError) as error:
        parser.error(f"Cannot load checkpoint: {error}")

    if agent.model is None:
        print(
            "TKC-AI đang chạy chế độ suy luận quy tắc + bộ nhớ + tra cứu web. "
            "Chưa có checkpoint ngôn ngữ đã huấn luyện; các câu hỏi tổng quát "
            "sẽ được báo rõ là ngoài khả năng hiện tại."
        )
    else:
        print("TKC-AI đã nạp checkpoint. Có thể nhập câu hỏi hoặc gõ 'thoát'.")

    if args.question:
        print_response(agent.reply(args.question, args.max_new_tokens))
        return

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
