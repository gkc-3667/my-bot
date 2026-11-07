import unittest

from brain.reasoning import solve_question


class ConditionalChainReasoningTests(unittest.TestCase):
    def test_chains_two_rules_from_a_known_fact(self):
        result = solve_question(
            "Nếu trời mưa thì đường ướt. "
            "Nếu đường ướt thì xe dễ trượt. "
            "Trời mưa. Có thể kết luận gì?"
        )
        self.assertIsNotNone(result)
        self.assertIn("đường ướt", result.answer)
        self.assertIn("xe dễ trượt", result.answer)
        self.assertEqual(
            result.method,
            "Suy diễn chuỗi bằng luật kéo theo (forward chaining)",
        )
        self.assertEqual(len(result.steps), 2)

    def test_rule_order_does_not_change_the_conclusion(self):
        result = solve_question(
            "Nếu đường ướt thì xe dễ trượt. "
            "Nếu trời mưa thì đường ướt. "
            "Trời mưa. Có thể kết luận gì?"
        )
        self.assertIsNotNone(result)
        self.assertIn("xe dễ trượt", result.answer)

    def test_does_not_infer_the_converse(self):
        result = solve_question(
            "Nếu trời mưa thì đường ướt. "
            "Nếu đường ướt thì xe dễ trượt. "
            "Đường ướt. Có thể kết luận trời mưa không?"
        )
        self.assertIsNotNone(result)
        self.assertIn("Chưa đủ dữ kiện", result.answer)

    def test_cycle_terminates_without_repeating_conclusions(self):
        result = solve_question(
            "Nếu A thì B. Nếu B thì A. A. Có thể kết luận gì?"
        )
        self.assertIsNotNone(result)
        self.assertIn("b", result.answer)
        self.assertEqual(len(result.steps), 1)

    def test_chain_solver_does_not_guess_without_supporting_facts(self):
        result = solve_question(
            "Nếu trời mưa thì đường ướt. "
            "Nếu đường ướt thì xe dễ trượt. "
            "Có thể kết luận gì?"
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
