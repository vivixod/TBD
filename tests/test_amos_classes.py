"""AMOS rare-class exclusion across episode sampling and metric aggregation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.logger import AverageMeter
from data.amos import DatasetAMOS


class AMOSClassTests(unittest.TestCase):
    def make_dataset(self, split):
        names = ['x___amos_0001_0', 'x___amos_0002_0']
        pools = {1: [names[0]], 13: [names[1]], 14: names, 15: names}
        with tempfile.TemporaryDirectory() as directory:
            labels = {str(c): f'organ_{c}' for c in range(1, 16)}
            (Path(directory) / 'dataset.json').write_text(json.dumps({'labels': labels}))
            with mock.patch.object(DatasetAMOS, 'build_img_metadata', return_value=names), \
                    mock.patch.object(DatasetAMOS, 'build_img_metadata_classwise', return_value=pools):
                dataset = DatasetAMOS(directory, 0, transforms.Compose([]), split, 4, False,
                                      neighbor_prob=0.0)
        dataset._label_classes_cache = {names[0]: [0, 1, 14, 15], names[1]: [0, 13, 14, 15]}
        return dataset

    def test_active_classes_and_names_preserve_original_label_ids(self):
        for split in ('trn', 'val', 'test'):
            dataset = self.make_dataset(split)
            self.assertEqual(dataset.class_ids, list(range(13)))
            self.assertEqual(dataset.class_names, [f'organ_{c}' for c in range(1, 14)])
            self.assertEqual(dataset.nclass, 15)  # Storage and joint offsets keep their IDs.

    def test_original_and_balanced_episodes_never_target_excluded_organs(self):
        for split in ('trn', 'val', 'test'):
            for probability in (0, .5, 1):
                dataset = self.make_dataset(split)
                dataset.set_class_balance_prob(probability)
                np.random.seed(321)
                sampled = set()
                for index in range(200):
                    query, supports, class_id = dataset.sample_episode(index % 2)
                    sampled.add(class_id)
                    self.assertIn(class_id + 1, dataset._label_classes(query))
                    self.assertEqual(len(supports), 4)
                    self.assertTrue(all(class_id + 1 in dataset._label_classes(name)
                                        for name in supports))
                self.assertEqual(sampled, {0, 12})
                self.assertTrue(set(dataset.balanced_class_ids).isdisjoint({13, 14}))

    def test_metrics_exclude_rare_classes_from_all_averages_and_reports(self):
        dataset = self.make_dataset('val')
        # Exercise the actual metric code on CPU; excluded buffer slots contain failures.
        with mock.patch.object(torch.Tensor, 'cuda', lambda tensor, *args, **kwargs: tensor):
            meter = AverageMeter(dataset)
        meter.intersection_buf[:, :13] = 2
        meter.union_buf[:, :13] = 2
        meter.union_buf[:, 13:] = 1000
        self.assertAlmostEqual(float(meter.compute_dice()), 100, places=4)
        self.assertEqual(tuple(meter.compute_per_class_dice().shape), (13,))
        miou, fb_iou = meter.compute_iou()
        self.assertEqual(float(miou), 100)
        self.assertEqual(float(fb_iou), 100)
        self.assertEqual(tuple(meter.compute_per_class_iou().shape), (13,))


if __name__ == '__main__':
    unittest.main()
