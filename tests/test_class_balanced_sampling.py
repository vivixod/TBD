"""CPU tests for class-first query sampling, dataset weights, and validation stability."""
from collections import Counter
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.btcv import DatasetBTCV2
from data.amos import DatasetAMOS
from data.adam import DatasetAdam
from data.brats import DatasetBraTS
from data.dataset import FSSDataset
from data.joint import DatasetJoint
from data.medical_base import build_patient_slice_index


DATASET_TYPES = (DatasetBTCV2, DatasetAMOS, DatasetAdam, DatasetBraTS)


def populate(dataset, split='trn', shot=1):
    dataset.split = split
    dataset.shot = shot
    dataset.neighbor_prob = 0.0
    dataset.benchmark = 'synthetic'
    dataset.nclass = 3
    dataset.class_ids = [0, 1, 2]
    dataset.class_names = ['common', 'rare', 'missing']
    common = [f'x___patient_{i}' for i in range(100)]
    rare = ['x___patient_100']
    dataset.img_metadata = common + rare
    dataset.img_metadata_classwise = {1: common, 2: rare}
    dataset._label_classes_cache = {name: [0, 1] for name in common}
    dataset._label_classes_cache.update({name: [0, 2] for name in rare})
    dataset._patient2slices, dataset._patient2order = build_patient_slice_index(dataset.img_metadata)
    return dataset


def synthetic(cls=DatasetBTCV2, split='trn', shot=1):
    return populate(cls.__new__(cls), split, shot)


class TinyDataset(DatasetBTCV2):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize,
                 neighbor_prob=0.0):
        populate(self, split, shot)
        self.benchmark = datapath

    def __getitem__(self, idx):
        q, supports, c = self.sample_episode(idx % len(self.img_metadata))
        return {'query_name': q, 'support_names': supports, 'class_id': torch.tensor(c)}


class ClassBalancedSamplingTests(unittest.TestCase):
    def test_class_frequency_and_positive_query_for_all_four_datasets(self):
        for cls in DATASET_TYPES:
            for probability, expected_rare in [(0.0, 1 / 101), (.5, .5 / 101 + .25), (1.0, .5)]:
                with self.subTest(dataset=cls.__name__, probability=probability):
                    ds = synthetic(cls)
                    ds.set_class_balance_prob(probability)
                    np.random.seed(321)
                    counts = Counter()
                    for i in range(8000):
                        query, supports, c = ds.sample_episode(i % len(ds.img_metadata))
                        self.assertIn(c + 1, ds._label_classes(query))
                        self.assertTrue(all(c + 1 in ds._label_classes(s) for s in supports))
                        counts[c] += 1
                    self.assertAlmostEqual(counts[1] / 8000, expected_rare, delta=.02)
                    self.assertEqual(counts[2], 0)

    def test_disabled_and_validation_sampling_do_not_consume_extra_rng(self):
        for split, probability in [('trn', 0), ('val', 1)]:
            ds = synthetic(split=split)
            ds.set_class_balance_prob(probability)
            np.random.seed(321)
            before = np.random.get_state()
            self.assertIsNone(ds.sample_class_balanced_query())
            after = np.random.get_state()
            self.assertEqual(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2:], after[2:])
        for cls in DATASET_TYPES:
            original = synthetic(cls, split='val')
            configured = copy.deepcopy(original)
            configured.set_class_balance_prob(1)
            np.random.seed(321)
            before = [original.sample_episode(i) for i in range(101)]
            np.random.seed(321)
            after = [configured.sample_episode(i) for i in range(101)]
            self.assertEqual(before, after)

    def test_class_pools_filter_split_membership_deduplicate_and_validate_probability(self):
        ds = synthetic()
        ds.img_metadata_classwise[2] += ['validation_only', 'x___patient_100']
        ds.set_class_balance_prob(1)
        self.assertEqual(ds.class_query_pools[1], ('x___patient_100',))
        self.assertEqual(ds.balanced_class_ids, (0, 1))
        for invalid in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                ds.set_class_balance_prob(invalid)
        ds.img_metadata_classwise = {}
        with self.assertRaises(ValueError):
            ds.set_class_balance_prob(1)

    def test_rare_support_pool_smaller_than_shot_does_not_hang(self):
        for cls in DATASET_TYPES:
            ds = synthetic(cls, shot=4)
            rare = ['x___patient_100', 'x___patient_101']
            ds.img_metadata = rare
            ds.img_metadata_classwise = {2: rare}
            ds.class_ids = [1]
            ds._label_classes_cache = {name: [0, 2] for name in rare}
            ds.set_class_balance_prob(1)
            query, supports, c = ds.sample_episode(0)
            self.assertEqual(c, 1)
            self.assertEqual(len(supports), 4)
            self.assertEqual(set(supports), set(rare))

    def test_joint_sampling_preserves_dataset_weights_and_class_offsets(self):
        registry = {name: TinyDataset for name in ('btcv2', 'amos', 'adam', 'brats')}
        with mock.patch.object(FSSDataset, 'datasets', registry, create=True):
            joint = DatasetJoint(
                0, None, 'trn', 1, False,
                joint_subdatasets=[(name, name) for name in registry],
                n_episodes=8000, sub_weights=[1, 1, 2, 1.5], class_balance_prob=.5,
            )
        expected = np.array([1, 1, 2, 1.5]) / 5.5
        np.testing.assert_allclose(joint._sub_probs, expected)
        np.random.seed(321)
        counts = Counter()
        for i in range(8000):
            batch = joint[i]
            dataset_idx = batch['dataset_idx']
            counts[dataset_idx] += 1
            local_class = int(batch['class_id']) - joint._class_id_offsets[dataset_idx]
            self.assertIn(local_class, (0, 1))
        np.testing.assert_allclose([counts[i] / 8000 for i in range(4)], expected, atol=.025)
        self.assertTrue(all(ds.class_balance_prob == .5 for ds in joint.sub_datasets))

    def test_dataloader_routes_balance_flag_only_to_training(self):
        FSSDataset.initialize(32, 'synthetic', False)
        with mock.patch.dict(FSSDataset.datasets, {'btcv2': TinyDataset}), \
                mock.patch('torch.utils.data.distributed.DistributedSampler', return_value=None):
            train = FSSDataset.build_dataloader('btcv2', 1, 0, 0, 'trn',
                                                class_balance_prob=.5)
            val = FSSDataset.build_dataloader('btcv2', 1, 0, 0, 'val',
                                              class_balance_prob=.5)
        self.assertEqual(train.dataset.class_balance_prob, .5)
        self.assertEqual(val.dataset.class_balance_prob, 0)

    def test_real_btcv_validation_matches_recorded_prechange_sampling(self):
        datapath = ROOT.parent / 'BTCV'
        if not (datapath / 'mask').is_dir():
            self.skipTest('Local BTCV masks are not available')
        previous_cwd = Path.cwd()
        try:
            os.chdir(ROOT)
            ds = DatasetBTCV2(str(datapath), 0, None, 'val', 4, False, 1)
            ds.set_class_balance_prob(.5)
            counts = Counter()
            for rank in range(3):
                np.random.seed(321)
                for i in (list(range(1000)) + [0, 1])[rank::3]:
                    _, _, c = ds.sample_episode(i % len(ds.img_metadata))
                    counts[int(c)] += 1
        finally:
            os.chdir(previous_cwd)
        self.assertEqual(dict(counts), {0: 58, 1: 54, 2: 64, 3: 13, 4: 113, 5: 119,
                                       6: 76, 7: 220, 8: 191, 9: 20, 10: 37, 11: 10, 12: 27})


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
