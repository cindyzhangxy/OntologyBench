import math
import unittest

from ontologybench.evaluation import metrics as metric_module
from ontologybench.evaluation.metrics import evaluate_run, ndcg_at_k


class RetrievalMetricsTests(unittest.TestCase):
    def test_single_gold_at_rank_one_has_perfect_ndcg(self):
        self.assertEqual(ndcg_at_k(["A", "B"], ["A"], k=2), 1.0)

    def test_unretrieved_gold_document_contributes_to_idcg(self):
        score = ndcg_at_k(["A", "B"], ["A", "C"], k=2)

        self.assertAlmostEqual(score, 0.6131471927654584)

    def test_two_golds_retrieved_ideally_have_perfect_ndcg(self):
        self.assertEqual(ndcg_at_k(["A", "C"], ["A", "C"], k=2), 1.0)

    def test_two_golds_at_ranks_one_and_three_use_standard_discounts(self):
        score = ndcg_at_k(["A", "B", "C"], ["A", "C"], k=3)

        self.assertAlmostEqual(score, 0.9197207891481876)

    def test_no_relevant_retrieval_has_zero_ndcg_and_hit(self):
        self.assertEqual(ndcg_at_k(["B", "D"], ["A", "C"], k=2), 0.0)
        self.assertTrue(hasattr(metric_module, "hit_at_k"))
        if hasattr(metric_module, "hit_at_k"):
            self.assertEqual(
                metric_module.hit_at_k(["B", "D"], ["A", "C"], k=2),
                0.0,
            )

    def test_duplicate_retrieved_id_is_removed_before_top_k_cutoff(self):
        score = ndcg_at_k(["A", "A", "B"], ["A", "C"], k=2)

        self.assertAlmostEqual(score, 0.6131471927654584)

    def test_duplicate_gold_ids_count_once(self):
        with_duplicate = ndcg_at_k(["A", "B"], ["A", "A", "C"], k=2)
        without_duplicate = ndcg_at_k(["A", "B"], ["A", "C"], k=2)

        self.assertAlmostEqual(with_duplicate, 0.6131471927654584)
        self.assertEqual(with_duplicate, without_duplicate)

    def test_missing_run_query_contributes_zero_to_macro_average(self):
        metrics = evaluate_run(
            {"q1": ["A"]},
            {"q1": ["A"], "q2": ["B"]},
            k=1,
        )

        self.assertEqual(
            metrics,
            {"MRR@1": 0.5, "Hit@1": 0.5, "nDCG@1": 0.5},
        )

    def test_unknown_run_query_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "absent from qrels"):
            evaluate_run(
                {"q1": ["A"], "unknown": ["B"]},
                {"q1": ["A"]},
                k=1,
            )

    def test_nonpositive_k_raises_value_error(self):
        for k in (0, -1):
            with self.subTest(k=k):
                with self.assertRaisesRegex(ValueError, "k must be positive"):
                    ndcg_at_k(["A"], ["A"], k=k)

    def test_empty_gold_set_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "empty gold set"):
            evaluate_run({"q1": []}, {"q1": []}, k=1)

    def test_empty_qrels_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "qrels must not be empty"):
            evaluate_run({}, {}, k=1)

    def test_single_gold_rank_positions_match_standard_ndcg(self):
        cases = (
            (["A", "B", "C"], 1.0),
            (["B", "A", "C"], 1.0 / math.log2(3)),
            (["B", "C", "A"], 0.5),
            (["B", "C", "D"], 0.0),
        )

        for retrieved, expected in cases:
            with self.subTest(retrieved=retrieved):
                self.assertAlmostEqual(ndcg_at_k(retrieved, ["A"], k=3), expected)


if __name__ == "__main__":
    unittest.main()
