"""Small, deterministic reasoning tools with checkable derivations.

This is genuine but deliberately narrow symbolic reasoning, not a general
language model. It solves only patterns whose assumptions it can parse.
"""
import ast
import math
import operator
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningResult:
    answer: str
    method: str
    steps: tuple
    confidence: float = 1.0


_BINARY_OPERATORS = {
    ast.Add: (operator.add, "+"),
    ast.Sub: (operator.sub, "-"),
    ast.Mult: (operator.mul, "*"),
    ast.Div: (operator.truediv, "/"),
    ast.FloorDiv: (operator.floordiv, "//"),
    ast.Mod: (operator.mod, "%"),
    ast.Pow: (operator.pow, "**"),
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_WEEKDAYS = {
    "thứ hai": 0,
    "thứ ba": 1,
    "thứ tư": 2,
    "thứ năm": 3,
    "thứ sáu": 4,
    "thứ bảy": 5,
    "chủ nhật": 6,
}


def _format_number(value):
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value)


def _safe_ast_value(node, steps):
    if isinstance(node, ast.Expression):
        return _safe_ast_value(node.body, steps)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("Number is outside the supported range.")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        value = _safe_ast_value(node.operand, steps)
        result = _UNARY_OPERATORS[type(node.op)](value)
        steps.append(f"Áp dụng dấu một ngôi: {_format_number(result)}.")
        return result
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        calculate, symbol = _BINARY_OPERATORS[type(node.op)]
        left = _safe_ast_value(node.left, steps)
        right = _safe_ast_value(node.right, steps)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("Exponent is too large for this calculator.")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        result = calculate(left, right)
        if not math.isfinite(result) or abs(result) > 1e100:
            raise ValueError("Result is outside the supported range.")
        steps.append(
            f"{_format_number(left)} {symbol} {_format_number(right)} "
            f"= {_format_number(result)}."
        )
        return result
    raise ValueError("Expression contains an unsupported operation.")


def safe_calculate(expression):
    """Evaluate numeric arithmetic without eval(), names, calls or attributes."""
    expression = str(expression).strip()
    if not expression or len(expression) > 200:
        raise ValueError("Expression must contain 1 to 200 characters.")
    tree = ast.parse(expression, mode="eval")
    steps = []
    value = _safe_ast_value(tree, steps)
    return value, tuple(steps)


def _solve_word_problem(text):
    # Multiplication: 3 bags, each containing 4 oranges.
    match = re.search(
        r"(\d+)\s+(?:nhóm|túi|hộp|giỏ|thùng|gói)\s*[, ]{0,6}"
        r"mỗi\s+(?:nhóm|túi|hộp|giỏ|thùng|gói)\s+(?:đều\s+)?(?:có\s+)?(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        count, each = map(int, match.groups())
        total = count * each
        return ReasoningResult(
            answer=f"Có tất cả {total} món.",
            method="Mô hình hóa bài toán nhân",
            steps=(
                f"Có {count} nhóm, mỗi nhóm có {each} món.",
                f"Tổng số món = {count} × {each} = {total}.",
            ),
        )

    # Equal sharing: 12 cakes shared equally by 3 people.
    match = re.search(
        r"(\d+)\s+(?:chiếc\s+|cái\s+|quả\s+|món\s+)?"
        r"[\wÀ-ỹ]+(?:\s+[\wÀ-ỹ]+){0,2}\s+chia\s+đều\s+cho\s+(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        total, groups = map(int, match.groups())
        if groups > 0 and total % groups == 0:
            each = total // groups
            return ReasoningResult(
                answer=f"Mỗi người nhận {each} món.",
                method="Mô hình hóa phép chia đều",
                steps=(
                    f"Tổng cộng có {total} món và {groups} người.",
                    f"Mỗi người nhận = {total} ÷ {groups} = {each}.",
                ),
            )
    return None


def _solve_arithmetic(text):
    normalized = text.casefold()
    replacements = (
        (r"\bchia\s+lấy\s+dư\b", "%"),
        (r"\bchia\s+cho\b", "/"),
        (r"\bchia\b", "/"),
        (r"\bnhân\s+với\b", "*"),
        (r"\bnhân\b", "*"),
        (r"\bcộng\s+(?:thêm\s+)?", "+"),
        (r"\btrừ\s+(?:đi\s+)?", "-"),
        (r"\bmũ\b", "**"),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)

    pattern = re.compile(
        r"(?<![\w.])\d+(?:[.,]\d+)?"
        r"(?:\s*(?:\*\*|[+\-*/%])\s*\d+(?:[.,]\d+)?)+"
    )
    for match in pattern.finditer(normalized):
        expression = match.group(0).replace(",", ".")
        try:
            value, calculation_steps = safe_calculate(expression)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            continue
        steps = (f"Biểu thức được nhận diện: {expression}.",) + calculation_steps
        return ReasoningResult(
            answer=f"Kết quả là {_format_number(value)}.",
            method="Tính toán số học có kiểm tra cú pháp",
            steps=steps,
        )
    return None


def _solve_sequence(text):
    normalized = text.casefold()
    if not any(
        phrase in normalized
        for phrase in ("dãy số", "số tiếp theo", "quy luật", "chuỗi số")
    ):
        return None
    values = [
        int(value)
        for value in re.findall(r"(?<![\w])\d+(?![\w])", normalized)
    ]
    if len(values) < 3 or len(values) > 12:
        return None

    differences = [b - a for a, b in zip(values, values[1:])]
    if len(set(differences)) == 1:
        next_value = values[-1] + differences[-1]
        return ReasoningResult(
            answer=f"Số tiếp theo là {next_value}.",
            method="Nhận diện cấp số cộng",
            steps=(
                f"Các số đã cho: {', '.join(map(str, values))}.",
                f"Hiệu liên tiếp đều bằng {differences[0]}.",
                f"Số tiếp theo = {values[-1]} + {differences[0]} = {next_value}.",
            ),
        )

    if all(value != 0 for value in values[:-1]):
        ratios = [b / a for a, b in zip(values, values[1:])]
        if max(ratios) - min(ratios) < 1e-12:
            ratio = ratios[-1]
            next_value = values[-1] * ratio
            if float(next_value).is_integer():
                next_value = int(next_value)
            return ReasoningResult(
                answer=f"Số tiếp theo là {_format_number(next_value)}.",
                method="Nhận diện cấp số nhân",
                steps=(
                    f"Các số đã cho: {', '.join(map(str, values))}.",
                    f"Tỉ số liên tiếp đều bằng {_format_number(ratio)}.",
                    f"Số tiếp theo = {values[-1]} × {_format_number(ratio)} "
                    f"= {_format_number(next_value)}.",
                ),
            )
    return None


def _solve_syllogism(text):
    statements = re.split(r"[.!?;\n]+", text.casefold())
    universal = None
    identity = None
    for statement in statements:
        match = re.search(
            r"(?:mọi|tất cả)\s+(.+?)\s+(?:đều\s+)?là\s+(.+)",
            statement.strip(),
        )
        if match:
            universal = tuple(part.strip() for part in match.groups())
            continue
        match = re.search(r"^([\wÀ-ỹ]+)\s+là\s+(.+)$", statement.strip())
        if match:
            identity = tuple(part.strip() for part in match.groups())

    if not universal or not identity:
        return None
    category, super_category = universal
    subject, subject_category = identity
    category_word = category.split()[-1]
    subject_word = subject_category.split()[-1]
    if category_word != subject_word:
        return None

    question_match = re.search(
        r"([\wÀ-ỹ]+)\s+có\s+phải\s+là\s+(.+?)\s+không",
        text.casefold(),
    )
    if not question_match:
        return None
    asked_subject, asked_category = question_match.groups()
    if asked_subject != subject:
        return None
    if asked_category.strip().split()[-1] != super_category.split()[-1]:
        return None

    return ReasoningResult(
        answer=f"Có. Theo hai tiền đề, {subject} thuộc nhóm {category}, "
               f"mà mọi {category} đều là {super_category}.",
        method="Tam đoạn luận (suy diễn tập hợp)",
        steps=(
            f"Tiền đề 1: Mọi {category} đều là {super_category}.",
            f"Tiền đề 2: {subject} là {subject_category}.",
            f"Kết luận: {subject} là {super_category}.",
        ),
    )


def _solve_conditional(text):
    normalized = text.casefold()
    match = re.search(r"nếu\s+(.+?)\s+thì\s+(.+?)(?:[.!?]|$)", normalized)
    if not match:
        return None
    antecedent, consequent = (part.strip() for part in match.groups())
    consequent_terms = [
        word for word in re.findall(r"[\wÀ-ỹ]+", consequent)
        if word not in {"đang", "thì", "là", "được", "bị", "có"}
    ]
    if not consequent_terms:
        return None

    question_text = re.sub(r"\b(?:có|chắc chắn|phải|đang|là)\b", " ", normalized)
    words = re.findall(r"[\wÀ-ỹ]+", question_text)
    has_consequent = all(term in words for term in consequent_terms)
    if not has_consequent:
        return None

    consequent_start = min(
        (words.index(term) for term in consequent_terms if term in words),
        default=-1,
    )
    # A negative statement of the consequent supports modus tollens.
    negative_nearby = False
    if consequent_start >= 0:
        nearby = words[max(0, consequent_start - 2):consequent_start + len(consequent_terms) + 2]
        negative_nearby = "không" in nearby
    if negative_nearby:
        return ReasoningResult(
            answer=f"Có thể kết luận {antecedent} không xảy ra, "
                   "nếu mệnh đề điều kiện ban đầu là đúng.",
            method="Phản chứng bằng phủ định hệ quả (modus tollens)",
            steps=(
                f"Nếu {antecedent} thì {consequent}.",
                f"Quan sát cho biết hệ quả '{consequent}' không xảy ra.",
                f"Vì vậy suy ra '{antecedent}' không xảy ra.",
            ),
        )

    return ReasoningResult(
        answer=f"Không thể kết luận chắc chắn rằng {antecedent}. "
               f"Biết '{consequent}' đúng không chứng minh được điều kiện '{antecedent}'.",
        method="Kiểm tra lỗi khẳng định hệ quả",
        steps=(
            f"Tiền đề chỉ nói: nếu {antecedent} thì {consequent}.",
            f"Ta quan sát thấy '{consequent}', nhưng điều đó có thể có nguyên nhân khác.",
            f"Vì vậy không đủ dữ kiện để khẳng định '{antecedent}'.",
        ),
        confidence=0.98,
    )


def _solve_weekday(text):
    normalized = re.sub(r"\s+", " ", text.casefold())
    match = re.search(
        r"ngày mai\s+là\s+(thứ hai|thứ ba|thứ tư|thứ năm|thứ sáu|thứ bảy|chủ nhật)",
        normalized,
    )
    if not match or not re.search(r"hôm nay\s+là\s+thứ mấy", normalized):
        return None
    tomorrow = _WEEKDAYS[match.group(1)]
    today = (tomorrow - 1) % 7
    today_name = next(name for name, index in _WEEKDAYS.items() if index == today)
    return ReasoningResult(
        answer=f"Hôm nay là {today_name.title()}.",
        method="Suy luận lịch theo chu kỳ 7 ngày",
        steps=(
            f"Ngày mai là {match.group(1).title()}.",
            "Hôm nay đứng ngay trước ngày mai trong tuần.",
            f"Lùi một ngày: hôm nay là {today_name.title()}.",
        ),
    )


def solve_question(question):
    """Return a checkable solution when a supported reasoning pattern is found."""
    text = str(question).strip()
    if not text:
        return None
    solvers = (
        _solve_word_problem,
        _solve_syllogism,
        _solve_conditional,
        _solve_sequence,
        _solve_weekday,
        _solve_arithmetic,
    )
    for solver in solvers:
        result = solver(text)
        if result is not None:
            return result
    return None
