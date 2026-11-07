import unittest

from brain.reasoning import solve_question


class PercentageReasoningTests(unittest.TestCase):
    def test_percentage_of_quantity(self):
        result = solve_question("Tính 15% của 240.")
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "15% của 240 là 36.")
        self.assertEqual(result.method, "Tính phần trăm của một đại lượng")
        self.assertEqual(len(result.steps), 2)

    def test_increase_by_percentage(self):
        result = solve_question("Tăng 150 lên 20%.")
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Kết quả sau khi tăng 20% là 180.")

    def test_decrease_by_percentage(self):
        result = solve_question("Giảm 150 đi 20%.")
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Kết quả sau khi giảm 20% là 120.")

    def test_percentage_change(self):
        result = solve_question("Từ 80 lên 100 tăng bao nhiêu phần trăm?")
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Giá trị đã tăng 25%.")

    def test_percentage_change_from_zero_is_not_divided(self):
        result = solve_question("Từ 0 lên 10 tăng bao nhiêu phần trăm?")
        self.assertIsNotNone(result)
        self.assertIn("Không xác định", result.answer)

    def test_arithmetic_mean(self):
        result = solve_question("Tính trung bình cộng của 4, 6, 8.")
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Giá trị trung bình là 6.")

    def test_divide_total_by_ratio(self):
        result = solve_question("Chia 60 theo tỉ lệ 2:3.")
        self.assertIsNotNone(result)
        self.assertEqual(
            result.answer,
            "Hai phần lần lượt là 24 và 36.",
        )
        self.assertIn("Kiểm tra:", result.steps[-1])

    def test_ratio_rejects_zero_parts(self):
        self.assertIsNone(solve_question("Chia 60 theo tỉ lệ 0:3."))

    def test_compare_two_values(self):
        result = solve_question("So sánh 17 và 23.")
        self.assertIsNotNone(result)
        self.assertIn("23 lớn hơn 17", result.answer)
        self.assertIn("chênh lệch là 6", result.answer)

    def test_ambiguous_general_question_is_not_guessed(self):
        self.assertIsNone(solve_question("Bạn nghĩ phần trăm quan trọng thế nào?"))


if __name__ == "__main__":
    unittest.main()
