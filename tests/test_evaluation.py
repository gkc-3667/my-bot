import unittest

from training.evaluate import answer_matches, normalize_answer


class EvaluationTests(unittest.TestCase):
    def test_normalizes_case_and_punctuation(self):
        self.assertEqual(normalize_answer("  Thứ Tư! "), "thứ tư")

    def test_matches_expected_phrase_as_whole_words(self):
        self.assertTrue(answer_matches("Kết quả là 12 quả.", ["12"]))
        self.assertFalse(answer_matches("Kết quả là 112 quả.", ["12"]))

    def test_supports_alternative_expected_answers(self):
        self.assertTrue(answer_matches("Đáp án: Đúng.", ["Có", "Đúng"]))


if __name__ == "__main__":
    unittest.main()
