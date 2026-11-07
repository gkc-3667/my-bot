"""Additional deterministic numeric reasoning with auditable derivations.

Only narrowly parsed templates are accepted. Unsupported or ambiguous text
returns None instead of being guessed.
"""
import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ExtendedReasoningResult:
    answer: str
    method: str
    steps: tuple
    confidence: float = 0.99


_NUMBER = r"-?\d+(?:[.,]\d+)?"


def _number(value):
    return float(value.replace(",", "."))


def _fmt(value):
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-10):
        return str(int(round(value)))
    return f"{value:.10g}"


def solve_extended_question(question):
    text = re.sub(r"\s+", " ", str(question).casefold()).strip()
    if not text or len(text) > 1000:
        return None

    # Percentage of a base quantity: "Tính 15% của 240".
    match = re.search(
        rf"(?:tính\s+)?({_NUMBER})\s*%\s*(?:của|trong)\s*({_NUMBER})",
        text,
    )
    if match:
        percent, base = map(_number, match.groups())
        value = base * percent / 100.0
        if not math.isfinite(value) or abs(value) > 1e100:
            return None
        return ExtendedReasoningResult(
            answer=f"{_fmt(percent)}% của {_fmt(base)} là {_fmt(value)}.",
            method="Tính phần trăm của một đại lượng",
            steps=(
                f"Đổi phần trăm thành số thập phân: {_fmt(percent)}% = {_fmt(percent / 100)}.",
                f"Lấy {_fmt(base)} × {_fmt(percent / 100)} = {_fmt(value)}.",
            ),
        )

    # Increase/decrease by a percentage.
    match = re.search(
        rf"(?:tăng|giảm)\s+({_NUMBER})\s+(?:lên|đi|thêm|bớt)\s+({_NUMBER})\s*%",
        text,
    )
    if match:
        base, percent = map(_number, match.groups())
        increasing = text.startswith("tăng")
        delta = base * percent / 100.0
        result = base + delta if increasing else base - delta
        if not all(math.isfinite(value) and abs(value) <= 1e100 for value in (delta, result)):
            return None
        operation = "+" if increasing else "-"
        verb = "tăng" if increasing else "giảm"
        return ExtendedReasoningResult(
            answer=f"Kết quả sau khi {verb} {_fmt(percent)}% là {_fmt(result)}.",
            method="Điều chỉnh đại lượng theo phần trăm",
            steps=(
                f"Phần thay đổi = {_fmt(base)} × {_fmt(percent)} ÷ 100 = {_fmt(delta)}.",
                f"Giá trị mới = {_fmt(base)} {operation} {_fmt(delta)} = {_fmt(result)}.",
            ),
        )

    # Percentage increase/decrease from an old value to a new value.
    match = re.search(
        rf"(?:từ\s+)?({_NUMBER})\s+(?:lên|thành)\s+({_NUMBER}).*?"
        r"(?:tăng|giảm)\s+(?:bao nhiêu\s*)?(?:phần trăm|%)",
        text,
    )
    if match:
        old, new = map(_number, match.groups())
        if old == 0:
            return ExtendedReasoningResult(
                answer="Không xác định được phần trăm thay đổi vì giá trị ban đầu bằng 0.",
                method="Kiểm tra mẫu số của phần trăm thay đổi",
                steps=("Công thức phần trăm thay đổi cần chia cho giá trị ban đầu.",),
                confidence=1.0,
            )
        change = (new - old) / abs(old) * 100.0
        if not math.isfinite(change):
            return None
        direction = "tăng" if change >= 0 else "giảm"
        return ExtendedReasoningResult(
            answer=f"Giá trị đã {direction} {_fmt(abs(change))}%.",
            method="Tính phần trăm thay đổi",
            steps=(
                f"Mức thay đổi = {_fmt(new)} - {_fmt(old)} = {_fmt(new - old)}.",
                f"Phần trăm thay đổi = ({_fmt(new - old)} ÷ {_fmt(abs(old))}) × 100 = {_fmt(change)}%.",
            ),
        )

    # Arithmetic mean of an explicitly listed short set.
    if any(term in text for term in ("trung bình", "trung bình cộng", "giá trị trung bình")):
        values_match = re.search(
            rf"(?<!\w){_NUMBER}(?:\s*[,;]\s*{_NUMBER}){{1,19}}",
            text,
        )
        if values_match:
            values = [_number(item) for item in re.findall(_NUMBER, values_match.group(0))]
            if len(values) >= 2 and all(math.isfinite(value) for value in values):
                total = sum(values)
                average = total / len(values)
                if not math.isfinite(average):
                    return None
                return ExtendedReasoningResult(
                    answer=f"Giá trị trung bình là {_fmt(average)}.",
                    method="Tính trung bình cộng",
                    steps=(
                        f"Cộng các giá trị: {' + '.join(_fmt(value) for value in values)} = {_fmt(total)}.",
                        f"Chia tổng cho {len(values)} giá trị: {_fmt(total)} ÷ {len(values)} = {_fmt(average)}.",
                    ),
                )

    # Divide a total into parts according to a positive integer ratio.
    match = re.search(
        rf"(?:chia\s+({_NUMBER})\s+theo\s+tỉ\s+lệ\s+|"
        rf"tổng\s+({_NUMBER})\s+(?:được\s+)?chia\s+theo\s+tỉ\s+lệ\s+)"
        r"(\d+)\s*[:/]\s*(\d+)",
        text,
    )
    if match:
        total_text = match.group(1) or match.group(2)
        total = _number(total_text)
        first_ratio, second_ratio = int(match.group(3)), int(match.group(4))
        ratio_sum = first_ratio + second_ratio
        if total >= 0 and first_ratio > 0 and second_ratio > 0 and ratio_sum > 0:
            unit = total / ratio_sum
            first_part = unit * first_ratio
            second_part = unit * second_ratio
            if all(math.isfinite(value) for value in (unit, first_part, second_part)):
                return ExtendedReasoningResult(
                    answer=(
                        f"Hai phần lần lượt là {_fmt(first_part)} và "
                        f"{_fmt(second_part)}."
                    ),
                    method="Chia đại lượng theo tỉ lệ",
                    steps=(
                        f"Tổng số phần tỉ lệ = {first_ratio} + {second_ratio} = {ratio_sum}.",
                        f"Một phần = {_fmt(total)} ÷ {ratio_sum} = {_fmt(unit)}.",
                        f"Phần thứ nhất = {first_ratio} × {_fmt(unit)} = {_fmt(first_part)}.",
                        f"Phần thứ hai = {second_ratio} × {_fmt(unit)} = {_fmt(second_part)}.",
                        f"Kiểm tra: {_fmt(first_part)} + {_fmt(second_part)} = {_fmt(total)}.",
                    ),
                )

    # Compare exactly two explicitly named numbers.
    if "so sánh" in text:
        match = re.search(rf"({_NUMBER})\s+(?:và|với)\s+({_NUMBER})", text)
        if match:
            left, right = map(_number, match.groups())
            if left == right:
                answer = f"Hai số bằng nhau: {_fmt(left)} = {_fmt(right)}."
            elif left > right:
                answer = (
                    f"{_fmt(left)} lớn hơn {_fmt(right)}; "
                    f"chênh lệch là {_fmt(left - right)}."
                )
            else:
                answer = (
                    f"{_fmt(right)} lớn hơn {_fmt(left)}; "
                    f"chênh lệch là {_fmt(right - left)}."
                )
            return ExtendedReasoningResult(
                answer=answer,
                method="So sánh số học",
                steps=(
                    f"Số thứ nhất: {_fmt(left)}.",
                    f"Số thứ hai: {_fmt(right)}.",
                    answer,
                ),
            )

    return None
