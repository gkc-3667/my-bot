import unittest

from training.build_reasoning_curriculum import build_examples


class ReasoningCurriculumTests(unittest.TestCase):
    def test_generator_is_reproducible(self):
        first = build_examples(count_per_type=2, seed=17)
        second = build_examples(count_per_type=2, seed=17)
        self.assertEqual(first, second)

    def test_generated_examples_have_prompts_and_answers(self):
        examples = build_examples(count_per_type=2, seed=17)
        self.assertGreaterEqual(len(examples), 16)
        for row in examples:
            self.assertTrue(row["prompt"].strip())
            self.assertTrue(row["answer"].strip())
            self.assertIn("Phương pháp:", row["answer"])

    def test_generated_answers_cover_supported_task_families(self):
        examples = build_examples(count_per_type=1, seed=31)
        answers = "\n".join(row["answer"] for row in examples)
        self.assertIn("phương trình", answers.casefold())
        self.assertIn("Tam đoạn luận", answers)
        self.assertIn("cấp số", answers.casefold())


if __name__ == "__main__":
    unittest.main()
