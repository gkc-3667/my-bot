import unittest

from brain.reasoning import solve_question


class ParityReasoningTests(unittest.TestCase):
    def test_remainder_one_means_odd_not_even(self):
        result = solve_question(
            "Một số chia cho 2 dư 1. Số đó có phải số chẵn không?"
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.answer.startswith("Không."))
        self.assertIn("số lẻ", result.answer)

    def test_remainder_zero_means_even(self):
        result = solve_question(
            "Một số chia cho 2 dư 0. Số đó có phải số chẵn không?"
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.answer.startswith("Có."))
        self.assertIn("số chẵn", result.answer)

    def test_remainder_one_means_odd(self):
        result = solve_question(
            "Một số chia cho 2 dư 1. Số đó có phải số lẻ không?"
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.answer.startswith("Có."))
        self.assertIn("số lẻ", result.answer)


if __name__ == "__main__":
    unittest.main()
