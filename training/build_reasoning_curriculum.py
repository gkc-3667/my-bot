"""Generate a reproducible synthetic reasoning curriculum from verified solvers.

The examples are useful for teaching narrow task formats. They are not a
replacement for a large language pretraining corpus or human-reviewed data.
"""
import argparse
import json
import random
from pathlib import Path

from brain.reasoning import solve_question


WEEKDAYS = (
    "thứ hai", "thứ ba", "thứ tư", "thứ năm",
    "thứ sáu", "thứ bảy", "chủ nhật",
)
PEOPLE = ("An", "Bình", "Chi", "Dũng", "Hà", "Linh", "Minh", "Nam")
CATEGORIES = (
    ("con mèo", "mèo", "động vật"),
    ("con cá", "cá", "động vật"),
    ("bông hoa", "hoa", "thực vật"),
    ("xe đạp", "xe đạp", "phương tiện"),
    ("quả táo", "táo", "trái cây"),
)
CONDITIONALS = (
    ("trời mưa", "đường ướt"),
    ("đèn bật", "phòng sáng"),
    ("nhiệt độ xuống dưới 0 độ C", "nước có thể đóng băng"),
    ("máy hết pin", "máy có thể tắt"),
)


def _answer_for(prompt):
    result = solve_question(prompt)
    if result is None:
        raise RuntimeError(f"No verified solver result for generated prompt: {prompt}")
    explanation = "\n".join(f"- {step}" for step in result.steps)
    return (
        f"{result.answer}\n"
        f"Phương pháp: {result.method}.\n"
        f"Các bước kiểm tra:\n{explanation}"
    )


def build_examples(count_per_type=100, seed=42):
    if count_per_type < 1:
        raise ValueError("count_per_type must be positive.")
    rng = random.Random(seed)
    examples = []

    def add(prompt):
        examples.append({"prompt": prompt, "answer": _answer_for(prompt)})

    for _ in range(count_per_type):
        a, b, c = (rng.randint(1, 30) for _ in range(3))
        operator_word = rng.choice(("cộng", "trừ", "nhân"))
        add(f"Tính {a} {operator_word} {b}.")
        add(f"Tính {a} cộng {b} nhân {c} theo thứ tự phép tính chuẩn.")

        groups = rng.randint(2, 12)
        each = rng.randint(2, 15)
        add(
            f"Một giỏ có {groups} túi, mỗi túi có {each} quả cam. "
            "Có tất cả bao nhiêu quả cam?"
        )

        total = rng.randint(2, 12) * rng.randint(2, 12)
        people_count = rng.randint(2, 12)
        total = total - (total % people_count)
        if total == 0:
            total = people_count
        add(
            f"Có {total} chiếc bánh chia đều cho {people_count} người. "
            "Mỗi người nhận mấy chiếc?"
        )

        coefficient = rng.randint(1, 9)
        solution = rng.randint(-10, 10)
        constant = rng.randint(-10, 10)
        result = coefficient * solution + constant
        if constant < 0:
            equation = f"{coefficient}x - {abs(constant)} = {result}"
        else:
            equation = f"{coefficient}x + {constant} = {result}"
        add(f"Giải phương trình {equation}.")

        start = rng.randint(-20, 30)
        difference = rng.choice(tuple(value for value in range(-8, 9) if value))
        sequence = [start + difference * index for index in range(4)]
        add(
            f"Dãy số {', '.join(map(str, sequence))}, "
            "số tiếp theo là bao nhiêu?"
        )

        start = rng.randint(1, 8)
        ratio = rng.randint(2, 4)
        sequence = [start * (ratio ** index) for index in range(4)]
        add(
            f"Dãy số {', '.join(map(str, sequence))}, "
            "số tiếp theo là bao nhiêu?"
        )

        category, identity, super_category = rng.choice(CATEGORIES)
        person = rng.choice(PEOPLE)
        add(
            f"Mọi {category} đều là {super_category}. "
            f"{person} là {identity}. "
            f"{person} có phải là {super_category} không?"
        )

        antecedent, consequent = rng.choice(CONDITIONALS)
        add(
            f"Nếu {antecedent} thì {consequent}. "
            f"{consequent.capitalize()}. Có chắc chắn {antecedent} không?"
        )
        add(
            f"Nếu {antecedent} thì {consequent}. "
            f"{consequent.capitalize()} không. Có thể kết luận gì?"
        )

        tomorrow_index = rng.randrange(len(WEEKDAYS))
        add(
            f"Ngày mai là {WEEKDAYS[tomorrow_index].title()}. "
            "Hôm nay là thứ mấy?"
        )

    # A deterministic seed plus prompt de-duplication makes output reproducible.
    unique = {}
    for example in examples:
        unique[example["prompt"]] = example
    return list(unique.values())


def write_dataset(examples, output_path):
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-per-type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="data/reasoning_curriculum.jsonl"
    )
    args = parser.parse_args()
    examples = build_examples(args.count_per_type, args.seed)
    target = write_dataset(examples, args.output)
    print(f"Generated {len(examples)} verified-by-solver examples at {target}.")
    print(
        "These are narrow synthetic reasoning examples. "
        "They do not replace a broad pretraining corpus or independent evaluation."
    )


if __name__ == "__main__":
    main()
