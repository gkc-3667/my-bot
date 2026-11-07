"""Interactive chat entry point for GKC-AI, including pretraining fallback."""
import argparse
from pathlib import Path

import torch

from brain.agent import ResearchAgent
from brain.knowledge import KnowledgeBase
from brain.llm import OpenAICompatibleProvider
from brain.transformer import AIConfig, TKCAI


def load_agent(checkpoint_path, memory_path):
    memory = KnowledgeBase(memory_path)
    provider = OpenAICompatibleProvider.from_env()
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        return ResearchAgent(
            model=None,
            device="cpu",
            memory=memory,
            llm_provider=provider,
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
        llm_provider=provider,
    )


def handle_learning_command(agent, question):
    """Handle explicit, reviewable teaching commands in interactive chat."""
    if question.startswith("/teach "):
        payload = question[len("/teach "):]
        parts = [part.strip() for part in payload.split("|", 2)]
        if len(parts) < 2 or not parts[1]:
            print("Cách dùng: /teach Chủ đề | Nội dung | Nguồn (tùy chọn)")
            return True
        topic, fact = parts[0], parts[1]
        source = parts[2] if len(parts) > 2 else ""
        item_id = agent.memory.add(
            fact, topic=topic, source=source
        )
        print(f"Đã lưu ghi chú #{item_id} vào bộ nhớ truy xuất.")
        print(f"Nguồn: {source or 'chưa ghi nguồn'}")
        print("Ghi chú này chưa cập nhật trọng số mô hình.")
        return True

    if question == "/status":
        provider = getattr(agent, "llm_provider", None)
        checkpoint_state = (
            f"đã kết nối mô hình API: {getattr(provider, 'model', 'không rõ tên')}"
            if provider is not None
            else (
                "đã nạp checkpoint cục bộ"
                if agent.model is not None
                else "chưa có mô hình ngôn ngữ; chỉ dùng công cụ và bộ suy luận quy tắc"
            )
        )
        web_state = (
            "đã nạp thư viện tìm kiếm; vẫn cần mạng khi tra cứu"
            if agent.search_provider is not None
            else "chưa khả dụng (" + (agent.search_setup_error or "thiếu thư viện") + ")"
        )
        memory_count = agent.memory.count()
        example_path = Path("data/learned_examples.jsonl")
        example_count = 0
        if example_path.is_file():
            example_count = sum(
                1 for line in example_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        print("\nTrạng thái GKC-AI:")
        print(f"- Mô hình ngôn ngữ: {checkpoint_state}.")
        print("- Suy luận quy tắc: đang bật cho các dạng có dữ kiện kiểm chứng được.")
        print(
            "- Suy luận có kiểm chứng: số học, phương trình bậc nhất, "
            "phần trăm, trung bình, so sánh, tỉ lệ, dãy số, lịch và một số luật logic."
        )
        if provider is not None:
            print("- Câu hỏi mở: chuyển tới mô hình API đã cấu hình.")
            print("- Lịch sử chat: giữ 6 lượt gần nhất trong RAM; dùng /reset để xóa.")
        elif agent.model is not None:
            print("- Câu hỏi mở: chuyển tới checkpoint ngôn ngữ cục bộ.")
        else:
            print("- Câu hỏi ngoài các mẫu đã hỗ trợ vẫn cần mô hình ngôn ngữ.")

        print(f"- Tìm kiếm web: {web_state}.")
        print(f"- Ghi chú đã dạy: {memory_count}.")
        print(f"- Ví dụ chờ fine-tune: {example_count}.")
        if not (provider is not None or agent.model is not None):
            print("- Viết mã: chưa có mô hình ngôn ngữ để sinh mã tổng quát.")
        else:
            print("- Viết mã: mô hình có thể tạo mã; chưa có sandbox thực thi hoặc kiểm thử tự động.")
        return True

    if question == "/reset":
        history = getattr(agent, "conversation_history", None)
        if history is not None:
            history.clear()
        print("Đã xóa ngữ cảnh hội thoại đang giữ trong RAM.")
        return True

    if question == "/doctor":
        from brain.doctor import run_diagnostics
        print("\nChẩn đoán GKC-AI:")
        for check in run_diagnostics():
            status = "OK" if check["ok"] else "NOT READY"
            print(f"- [{status}] {check['name']}: {check['detail']}")
        return True

    if question in {"/help", "help"}:
        print("Các lệnh GKC-AI:")
        print("- /status: xem năng lực hiện tại.")
        print("- /reset: xóa ngữ cảnh hội thoại hiện tại.")
        print("- /doctor: kiểm tra môi trường và các thành phần đã sẵn sàng.")
        print("- /teach Chủ đề | Nội dung | Nguồn: lưu ghi chú.")
        print("- /example Câu hỏi | Câu trả lời: lưu ví dụ cho lần fine-tune sau.")
        print("- thoát: kết thúc chat.")
        return True

    if question.startswith("/example "):
        payload = question[len("/example "):]
        prompt, separator, answer = payload.partition("|")
        if not separator or not prompt.strip() or not answer.strip():
            print("Cách dùng: /example Câu hỏi | Câu trả lời đã kiểm chứng")
            return True
        added = agent.memory.add_training_example(
            prompt.strip(), answer.strip(), "data/learned_examples.jsonl"
        )
        if added:
            print("Đã lưu ví dụ vào data/learned_examples.jsonl.")
            print("Mô hình chỉ học ví dụ sau lần fine-tune tiếp theo.")
        else:
            print("Ví dụ này đã tồn tại; không lưu trùng.")
        return True
    return False


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
        description="Chat with GKC-AI using symbolic reasoning, web search and learned memory."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/gkc_ai_instruct.pt",
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

    if agent.llm_provider is not None:
        print(
            f"GKC-AI đã kết nối mô hình ngôn ngữ API: {agent.llm_provider.model}. "
            "Có thể trò chuyện nhiều lượt; gõ 'thoát' để kết thúc."
        )
    elif agent.model is not None:
        print("GKC-AI đã nạp checkpoint cục bộ. Có thể nhập câu hỏi hoặc gõ 'thoát'.")
    else:
        print(
            "GKC-AI đang chạy chế độ suy luận quy tắc + bộ nhớ + tra cứu web. "
            "Chưa có mô hình ngôn ngữ; câu hỏi tổng quát sẽ bị giới hạn."
        )

    if args.question:
        if not handle_learning_command(agent, args.question):
            print_response(agent.reply(args.question, args.max_new_tokens))
        return

    print("Lệnh: /status, /doctor, /help, /reset, /teach Chủ đề | Nội dung | Nguồn, /example Câu hỏi | Câu trả lời")
    while True:
        try:
            question = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã kết thúc.")
            break
        if question.casefold() in {"thoát", "exit", "quit"}:
            print("GKC-AI: Tạm biệt.")
            break
        if not question:
            continue
        if handle_learning_command(agent, question):
            continue
        try:
            print_response(agent.reply(question, args.max_new_tokens))
        except Exception as error:
            print(f"GKC-AI gặp lỗi: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
