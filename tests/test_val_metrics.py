"""Small protocol tests; no artifact loading or original-data writes."""
import unittest
from evaluation.validation import ranking_metrics


class ValidationProtocolTests(unittest.TestCase):
    def test_unreachable_target_stays_in_denominator(self):
        result = ranking_metrics([1,2],[2,1],{1,99})
        self.assertEqual(result['lgb_recall50'],0.5)
        self.assertEqual(result['target_count'],2)
        self.assertAlmostEqual(result['lgb_ndcg50'],1/(1+1/__import__('math').log2(3)))

    def test_zero_candidates_retains_zero_metrics(self):
        result = ranking_metrics([],[],{99})
        self.assertEqual(result['pool_hits'],0)
        self.assertEqual(result['lgb_recall50'],0)
        self.assertEqual(result['lgb_ndcg50'],0)

    def test_rank_order_changes_ndcg(self):
        result = ranking_metrics([7,1],[1,7],{1})
        self.assertLess(result['lgb_ndcg50'],result['fusion_ndcg50'])
        self.assertEqual(result['lgb_recall50'],result['fusion_recall50'])

    def test_empty_targets_rejected(self):
        with self.assertRaises(ValueError):
            ranking_metrics([1],[1],set())

    def test_duplicate_candidate_rejected(self):
        with self.assertRaises(ValueError):
            ranking_metrics([1,1],[1,1],{1})


if __name__ == '__main__':
    unittest.main()
