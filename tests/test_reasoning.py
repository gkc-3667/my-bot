import unittest

from brain.reasoning import safe_calculate, solve_question


class ArithmeticReasoningTests(unittest.TestCase):
    def test_operator_precedence(self):
        result = solve_question("Tính 15 trừ 3 nhân 4 theo thứ tự phép tính chuẩn.")
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Kết quả là 3.")
        self.assertIn("3 * 4 = 12.", result.steps)

    def test_safe_calculator_rejects_function_calls(self):
        with self.assertRaises(ValueError):
            safe_calculate("__import__('os').system('echo unsafe')")

    def test_division_by_zero_is_rejected(self):
        with self.assertRaises(ZeroDivisionError):
            safe_calculate("10 / 0")

    def test_dates_are_not_mistaken_for_division(self):
        self.assertIsNone(solve_question("Sự kiện diễn ra ngày 2026/11/07 phải không?"))


class LinearEquationTests(unittest.TestCase):
    def test_solves_and_substitutes_linear_equation(self):
        result = solve_question("Giải phương trình 2x + 3 = 11.")
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Nghiệm là x = 4.")
        self.assertIn("vế trái = 11", result.steps[-1])

    def test_does_not_claim_to_solve_quadratic_equation(self):
        result = solve_question("Giải phương trình x^2 = 4.")
        self.assertIsNone(result)

    def test_solves_parenthesized_linear_equation(self):
        result = solve_question("Giải phương trình 2(x + 1) = 6.")
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Nghiệm là x = 2.")


class WordProblemTests(unittest.TestCase):
    def test_multiplication_word_problem(self):
        result = solve_question(
            "Một giỏ có 3 túi, mỗi túi có 4 quả cam. Có tất cả bao nhiêu quả cam?"
        )
        self.assertIsNotNone(result)
        self.assertIn("12", result.answer)
        self.assertEqual(result.method, "Mô hình hóa bài toán nhân")

    def test_equal_sharing(self):
        result = solve_question("Có 12 chiếc bánh chia đều cho 3 người. Mỗi người nhận mấy chiếc?")
        self.assertIsNotNone(result)
        self.assertIn("4", result.answer)


class LogicReasoningTests(unittest.TestCase):
    def test_syllogism(self):
        result = solve_question(
            "Mọi con mèo đều là động vật. Miu là mèo. Miu có phải là động vật không?"
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.answer.startswith("Có."))

    def test_syllogism_with_multiword_subject(self):
        result = solve_question(
            "Mọi loài chim đều là động vật. Chim cánh cụt là chim. "
            "Chim cánh cụt có phải là động vật không?"
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.answer.startswith("Có."))

    def test_affirming_the_consequent_is_not_valid(self):
        result = solve_question(
            "Nếu trời mưa thì đường ướt. Đường đang ướt. Có chắc chắn trời mưa không?"
        )
        self.assertIsNotNone(result)
        self.assertIn("Không thể kết luận", result.answer)

    def test_modus_tollens(self):
        result = solve_question(
            "Nếu trời mưa thì đường ướt. Đường không ướt. Có thể kết luận gì?"
        )
        self.assertIsNotNone(result)
        self.assertIn("không xảy ra", result.answer)

    def test_parity_from_remainder(self):
        result = solve_question(
            "Một số chia cho 2 dư 1. Số đó có phải số chẵn không?"
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.answer.startswith("Không."))


class PatternReasoningTests(unittest.TestCase):
    def test_geometric_sequence(self):
        result = solve_question("Dãy số 2, 4, 8, 16, số tiếp theo là bao nhiêu?")
        self.assertIsNotNone(result)
        self.assertIn("32", result.answer)

    def test_arithmetic_sequence_with_negative_values(self):
        result = solve_question("Dãy số -3, -1, 1, 3, số tiếp theo là bao nhiêu?")
        self.assertIsNotNone(result)
        self.assertIn("5", result.answer)

    def test_weekday(self):
        result = solve_question("Ngày mai là thứ Năm. Hôm nay là thứ mấy?")
        self.assertIsNotNone(result)
        self.assertIn("Thứ Tư", result.answer)

    def test_unsupported_question_is_not_guessed(self):
        self.assertIsNone(solve_question("Hãy giải thích lịch sử triết học phương Tây."))


if __name__ == "__main__":
    unittest.main()
