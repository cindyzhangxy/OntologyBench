from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ontologybench.generative.scoring import (  # noqa: E402
    PROMPT_SHA256,
    build_messages,
    validate_score,
)


class GenerativeScoringTests(unittest.TestCase):
    def test_messages_enforce_appendix_f_text_only_compatibility(self):
        messages = build_messages(
            ["Increased lower facial height", "Laryngeal stenosis", "Long philtrum"],
            "Autosomal dominant prognathism.",
        )

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Use ONLY the information provided", messages[0]["content"])
        self.assertIn("Do NOT use external medical knowledge", messages[0]["content"])
        self.assertIn("complete phenotype set", messages[1]["content"])
        self.assertIn("0-19: incompatible", messages[1]["content"])
        self.assertIn("80-100: near-complete match", messages[1]["content"])
        self.assertIn('{"score": 0}', messages[1]["content"])

    def test_prompt_identifier_matches_the_released_appendix_f_template(self):
        self.assertEqual(
            PROMPT_SHA256,
            "11b242f710f212f0e37e21db54c704c32f7153f01c0991f3a939151438fce08a",
        )

    def test_score_validation_accepts_only_one_integer_between_zero_and_one_hundred(self):
        self.assertEqual(validate_score({"score": 0}), 0)
        self.assertEqual(validate_score({"score": 100}), 100)

        invalid_values = (
            {"score": -1},
            {"score": 101},
            {"score": 50.0},
            {"score": True},
            {"score": 50, "explanation": "extra"},
            {},
            [50],
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_score(value)


if __name__ == "__main__":
    unittest.main()
