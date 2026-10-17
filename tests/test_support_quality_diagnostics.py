"""CPU checks for paired audit metrics, ties, and content-order invariance."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnose_support_quality import analyze_prediction, aggregate, balanced_select


class QualityDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.target = np.array([[1, 0], [0, 0]], bool)
        self.logits = np.full((4, 2, 2), -10., np.float32)
        self.logits[0, 0, 0] = 10.

    def test_equal_weights_have_no_position_advantage_and_no_fusion_change(self):
        row = analyze_prediction(self.logits, np.zeros(4), np.full(4, .25), self.target)
        self.assertEqual(row['dice'], [100., 0., 0., 0.])
        self.assertEqual(row['selected_single_dice'], 25.)
        self.assertEqual(row['random_single_dice'], 25.)
        self.assertEqual(row['top1_hit'], .25)
        self.assertEqual(row['random_top1_hit'], .25)
        self.assertEqual(row['pair_correct'] / row['pair_count'], .5)
        self.assertEqual(row['weighted_minus_uniform'], 0.)
        self.assertEqual(row['changed_pixels'], 0)

    def test_correct_content_ranking_changes_fusion_and_is_order_invariant(self):
        weights = np.array([.7, .1, .1, .1], np.float32)
        first = analyze_prediction(self.logits, np.log(weights), weights, self.target)
        permutation = [3, 2, 1, 0]
        second = analyze_prediction(self.logits[permutation], np.log(weights[permutation]),
                                    weights[permutation], self.target)
        for key in ['selected_single_dice', 'random_single_dice', 'weighted_dice',
                    'uniform_dice', 'top1_hit', 'pair_correct']:
            self.assertEqual(first[key], second[key])
        self.assertEqual(first['selected_minus_random'], 75.)
        self.assertEqual(first['weighted_dice'], 100.)
        self.assertEqual(first['uniform_dice'], 0.)
        self.assertEqual(first['pair_correct'], first['pair_count'])

    def test_all_zero_has_uniform_teacher_and_is_excluded_from_ranking(self):
        row = analyze_prediction(np.full_like(self.logits, -10.), np.zeros(4),
                                 np.full(4, .25), self.target)
        self.assertTrue(row['all_shots_zero'])
        self.assertEqual(row['teacher_weights'], [.25] * 4)
        self.assertEqual(row['uniform_teacher_kl'], 0.)
        result = aggregate([dict(row, patient='one')])
        self.assertEqual(result['informative_n'], 0)
        self.assertIsNone(result['informative_top1_hit'])
        self.assertIsNone(result['pairwise_accuracy'])
        self.assertEqual(result['fusion_delta_ci'], [None, None])

    def test_selection_spreads_patients_and_preserves_unique_queries(self):
        rows = [dict(patient=p, query_name=f'{p}_{i}', gt_area512=i)
                for p in ('a', 'b', 'c') for i in range(10)]
        selected = balanced_select(rows, 6)
        self.assertEqual(len({r['query_name'] for r in selected}), 6)
        self.assertEqual([r['patient'] for r in selected], ['a', 'b', 'c', 'a', 'b', 'c'])
        self.assertEqual({r['gt_area512'] for r in selected}, {0, 9})
        self.assertEqual(len(balanced_select(rows, 100)), len(rows))

    def test_one_query_per_patient_does_not_always_select_smallest_target(self):
        rows = [dict(patient=f'p{p:02d}', query_name=f'p{p:02d}_{i}', gt_area512=i)
                for p in range(16) for i in range(101)]
        selected = balanced_select(rows, 12)
        self.assertEqual(len({r['patient'] for r in selected}), 12)
        self.assertLess(min(r['gt_area512'] for r in selected), 10)
        self.assertGreater(max(r['gt_area512'] for r in selected), 90)
        self.assertEqual(sorted(r['gt_area512'] for r in selected),
                         [round((i+.5)/12*100) for i in range(12)])


if __name__ == '__main__':
    unittest.main()
