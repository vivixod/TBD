"""CPU checks for patient separation, nested targets and reproducible episodes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.dataset import FSSDataset
from data.joint import DatasetJoint
from data.papila import DatasetPAPILA, papila_patient_id
from data.splits.gen_papila_splits import build


class PAPILATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / 'PAPILA'
        (cls.root / 'image').mkdir(parents=True)
        (cls.root / 'label').mkdir()
        records = []
        for patient in range(12):
            for eye in ('OD', 'OS'):
                name = f'RET{patient:03d}{eye}'
                image = np.zeros((16, 20, 3), dtype=np.uint8)
                image[:] = [30, 90, 150]
                Image.fromarray(image).save(cls.root / 'image' / (name + '.png'))
                masks = np.zeros((2, 16, 20), dtype=np.uint8)
                masks[0, 3:13, 4:16] = 1
                masks[1, 6:10, 8:12] = 1
                # Exercise both exclusive and overlapping input channels.
                if patient % 2 == 0:
                    masks[0, masks[1].astype(bool)] = 0
                label = f'label/{name}.npz'
                sp.save_npz(cls.root / label, sp.csr_matrix(masks.reshape(2, -1)))
                records.append({'image': f'image/{name}.png', 'label': label})
        # Deliberately put OD in source training and OS in source test.
        source = {'labels': {'0': 'background', '1': 'optic_disc', '2': 'optic_cup'},
                  'training': records[::2], 'test': records[1::2]}
        (cls.root / 'dataset.json').write_text(json.dumps(source))
        cls.lists = Path(cls.tmp.name) / 'lists'
        cls.manifest = build(cls.root, cls.lists, ratio=.5, seed=321)
        cls.transform = transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        patch = mock.patch('data.papila.SPLIT_ROOT', self.lists)
        patch.start()
        self.addCleanup(patch.stop)

    def dataset(self, split='val', shot=4, **kwargs):
        return DatasetPAPILA(str(self.root), 0, self.transform, split, shot, False, **kwargs)

    def test_source_overlap_is_repaired_and_both_eyes_stay_together(self):
        self.assertEqual(self.manifest['source_train_test_patient_overlap'], 12)
        train, val = self.dataset('trn'), self.dataset()
        train_p = {papila_patient_id(n) for n in train.img_metadata}
        val_p = {papila_patient_id(n) for n in val.img_metadata}
        self.assertFalse(train_p & val_p)
        self.assertEqual(len(train_p | val_p), 12)
        for ds in (train, val):
            for patient in {papila_patient_id(n) for n in ds.img_metadata}:
                self.assertIn(patient + 'OD', ds.img_metadata)
                self.assertIn(patient + 'OS', ds.img_metadata)

    def test_supports_are_other_distinct_patients_and_stay_in_partition(self):
        for split in ('trn', 'val'):
            ds = self.dataset(split, neighbor_prob=1)
            ds.set_class_balance_prob(.5)
            self.assertEqual(ds.neighbor_prob, 0)
            for index in range(200):
                q, supports, c = ds.sample_episode(index)
                patients = [papila_patient_id(s) for s in supports]
                self.assertEqual(len(set(patients)), 4)
                self.assertNotIn(papila_patient_id(q), patients)
                self.assertTrue(set(supports) <= set(ds.img_metadata))
                self.assertTrue(all(s in ds.img_metadata_classwise[c + 1] for s in supports))

    def test_validation_covers_both_classes_once_with_fixed_order_independent_supports(self):
        ds = self.dataset()
        expected = {(n, c) for n in ds.img_metadata for c in (0, 1)}
        episodes = [ds.sample_episode(i) for i in range(len(ds))]
        self.assertEqual({(q, c) for q, _, c in episodes}, expected)
        self.assertEqual(len(episodes), len(expected))
        np.random.seed(12)
        before = np.random.get_state()
        reverse = [ds.sample_episode(i) for i in reversed(range(len(ds)))]
        after = np.random.get_state()
        self.assertEqual(episodes, list(reversed(reverse)))
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        # Simulate rank-strided access with different world sizes.
        for world in (2, 3):
            merged = {i: ds.sample_episode(i) for rank in range(world)
                      for i in range(rank, len(ds), world)}
            self.assertEqual(episodes, [merged[i] for i in range(len(ds))])

    def test_disc_contains_cup_for_both_input_layouts_without_mutating_labels(self):
        ds = self.dataset()
        for name in ('RET000OD', 'RET001OD'):
            label = ds.read_mask(name)
            original = label.clone()
            disc, _ = ds.extract_ignore_idx(label.float(), 0)
            cup, _ = ds.extract_ignore_idx(label.float(), 1)
            self.assertEqual(disc.sum(), 120)
            self.assertEqual(cup.sum(), 16)
            self.assertTrue(torch.all(cup <= disc))
            torch.testing.assert_close(label, original)

    def test_batch_shapes_rgb_targets_and_native_size(self):
        ds = self.dataset()
        batch = next(iter(DataLoader(ds, batch_size=2, num_workers=0)))
        self.assertEqual(batch['query_img'].shape, (2, 3, 32, 32))
        self.assertEqual(batch['query_mask'].shape, (2, 32, 32))
        self.assertEqual(batch['support_imgs'].shape, (2, 4, 3, 32, 32))
        self.assertEqual(batch['support_masks'].shape, (2, 4, 32, 32))
        self.assertEqual(set(batch['class_id'].tolist()), {0, 1})
        self.assertFalse(torch.equal(batch['query_img'][:, 0], batch['query_img'][:, 1]))
        self.assertEqual(set(batch['query_mask'].unique().tolist()), {0, 1})
        ds.use_original_imgsize = True
        self.assertEqual(ds[0]['query_mask'].shape, (16, 20))

    def test_class_balance_is_available_only_for_training(self):
        for split in ('trn', 'val'):
            ds = self.dataset(split)
            ds.set_class_balance_prob(1)
            self.assertEqual(ds.class_balance_prob, 1 if split == 'trn' else 0)
            if split == 'trn':
                classes = [ds.sample_episode(i)[2] for i in range(1000)]
                self.assertAlmostEqual(np.mean(classes), .5, delta=.07)

    def test_insufficient_support_patients_and_unsupported_fold_fail(self):
        with self.assertRaisesRegex(ValueError, 'only 5 available'):
            self.dataset(shot=6)
        with self.assertRaisesRegex(ValueError, 'fold 0'):
            DatasetPAPILA(str(self.root), 1, self.transform, 'trn', 4, False)

    def test_registry_joint_class_offsets_and_validation_loader(self):
        FSSDataset.initialize(32, str(self.root), False)
        self.assertIs(FSSDataset.datasets['papila'], DatasetPAPILA)
        ds = DatasetJoint(0, self.transform, 'trn', 4, False,
                          joint_subdatasets=[('papila', str(self.root))],
                          n_episodes=5, class_balance_prob=.5, neighbor_prob=1)
        batch = ds[0]
        self.assertEqual(ds.nclass_total, 2)
        self.assertEqual(batch['dataset_name'], 'papila')
        self.assertIn(int(batch['class_id']), (0, 1))
        val = DatasetJoint.build_val_dataloaders(2, 0, [('papila', str(self.root))], shot=4)
        loader = next(iter(val.values()))
        self.assertEqual(len(loader.dataset), 24)
        self.assertEqual(next(iter(loader))['support_masks'].shape, (2, 4, 32, 32))

    def test_generator_is_reproducible_and_refuses_to_overwrite_another_split(self):
        source_before = (self.root / 'dataset.json').read_bytes()
        self.assertEqual(build(self.root, self.lists, ratio=.5, seed=321), self.manifest)
        with self.assertRaises(FileExistsError):
            build(self.root, self.lists, ratio=.5, seed=999)
        self.assertEqual((self.root / 'dataset.json').read_bytes(), source_before)

    def test_papila_launcher_routes_dataset_weights_and_retains_quality_options(self):
        # Capture arguments with a stub; never start distributed training/GPU work.
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / 'python3'
            python.write_text(f'#!{sys.executable}\n'
                              'import json, sys\n'
                              'if sys.argv[1] == "-c": print(23456)\n'
                              'else: print(json.dumps(sys.argv[1:]))\n')
            python.chmod(0o755)
            env = dict(os.environ, PATH=tmp + os.pathsep + os.environ['PATH'],
                       GPU_ID='1,2', CLASS_BALANCE_PROB='0.5',
                       SUPPORT_MASK_POOLING='max', SUPPORT_FUSION='quality',
                       BBOX_SUPPORT_ATTN_MASK='1', ICL_SUPPORT_ATTN_MASK='1')
            for key in ('JOINT_SUBDATASETS', 'JOINT_WEIGHTS', 'EXPERIMENT_PREFIX', 'EXPERIMENT_NAME'):
                env.pop(key, None)
            result = subprocess.run(['sh', str(ROOT / 'train_joint_papila.sh')],
                                    env=env, cwd=tmp, capture_output=True, text=True, check=True)
            args = json.loads(result.stdout.splitlines()[-1])
            self.assertIn('--nproc_per_node=2', args)
            datasets = args[args.index('--joint-subdatasets') + 1]
            self.assertIn(f'papila:{ROOT.parent}/PAPILA', datasets)
            self.assertNotIn('adam:', datasets)
            self.assertEqual(args[args.index('--joint-weights') + 1], '1,1,1,1.5')
            self.assertEqual(args[args.index('--support-fusion') + 1], 'quality')
            self.assertIn('_papila_', args[args.index('--logpath') + 1])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
