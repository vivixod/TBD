"""CPU tests for retrieval correctness, patient/split isolation and integration."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.amos import DatasetAMOS
from data.btcv import DatasetBTCV2
from data.brats import DatasetBraTS
from data.dataset import FSSDataset
from data.joint import DatasetJoint
from data.support_retrieval import (SimilaritySupportIndex, bank_path, configure_support_sampling,
                                    dataset_spec, rank_descriptors, save_bank)
from prepare_support_retrieval import ImageDescriptors, prepare_dataset


def fixture(klass=DatasetBTCV2, split='val'):
    ds = klass.__new__(klass)
    ds.benchmark = {DatasetBTCV2: 'btcv2', DatasetAMOS: 'amos', DatasetBraTS: 'brats'}[klass]
    ds.split, ds.fold, ds.shot, ds.neighbor_prob = split, 0, 4, 1.
    ds.transform = transforms.Compose([transforms.Normalize([.4]*3, [.2]*3)])
    ds.class_ids = [0, 1]
    ds.img_metadata = [f'x___{patient}_{i}' for patient in 'ABC' for i in range(4)]
    ds.img_metadata_classwise = {1: ds.img_metadata + ['x___OTHER_SPLIT_0'],
                                2: [n for n in ds.img_metadata if '___B_' not in n]}
    ds._label_classes_cache = {name: ([0, 1] if '___B_' in name else [0, 1, 2])
                               for name in ds.img_metadata}
    # Ranking is deliberately dominated by B for A queries: all four supports
    # MUST be allowed to come from B, instead of adding an unrequested quota.
    features = np.array([[1.,0.]]*4 + [[1.,.01*i] for i in range(1,5)]
                        + [[.6,.8]]*4, dtype=np.float32)
    return ds, features


def write_fixture(ds, features, directory):
    spec = dataset_spec(ds)
    indices, scores = rank_descriptors(spec, features, max_shot=4, chunk_size=3)
    save_bank(bank_path(ds, directory), spec, indices, scores, 'fixture-weights')
    return indices, scores


class SupportRetrievalTests(unittest.TestCase):
    def test_global_top4_excludes_query_patient_and_keeps_class_and_split(self):
        ds, features = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(ds, features, tmp)
            index = SimilaritySupportIndex(ds, tmp)
            self.assertEqual(index.select('x___A_0', 0, 4), [f'x___B_{i}' for i in range(4)])
            self.assertEqual(index.select('x___A_0', 1, 4), [f'x___C_{i}' for i in range(4)])
            self.assertEqual(index.select('x___A_0', 0, 1), ['x___B_0'])
            self.assertTrue(all('OTHER_SPLIT' not in n for n in index.names))
            before = np.random.get_state()
            for _ in range(5):
                self.assertEqual(index.select('x___A_0', 0, 4), [f'x___B_{i}' for i in range(4)])
            after = np.random.get_state()
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2:], after[2:])

    def test_cosine_not_magnitude_and_ties_are_chunk_independent(self):
        ds, _ = fixture()
        features = np.tile([1., 1.], (len(ds.img_metadata),1)).astype(np.float32)
        features[5] *= 100
        spec = dataset_spec(ds)
        first, scores = rank_descriptors(spec, features, chunk_size=1)
        second, _ = rank_descriptors(spec, features, chunk_size=7)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0,0], [4,5,6,7])
        np.testing.assert_allclose(scores, 1., atol=2e-7)

    def test_real_episode_samplers_route_all_three_datasets_train_and_val(self):
        for klass in (DatasetBTCV2, DatasetAMOS, DatasetBraTS):
            for split in ('trn', 'val'):
                with self.subTest(klass=klass,split=split), tempfile.TemporaryDirectory() as tmp:
                    ds, features = fixture(klass,split)
                    write_fixture(ds, features, tmp)
                    configure_support_sampling(ds, 'similarity', tmp)
                    self.assertEqual(ds.neighbor_prob, 0)
                    for index in range(12):
                        q, supports, cls = ds.sample_episode(index)
                        self.assertEqual(supports, ds.support_retriever.select(q, cls, 4))
                        self.assertTrue(all(s.split('___')[1].split('_')[0] != q.split('___')[1].split('_')[0]
                                            for s in supports))

    def test_missing_stale_and_corrupt_cache_fail_without_random_fallback(self):
        ds, features = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                configure_support_sampling(ds, 'similarity', tmp)
            neighbors, scores = write_fixture(ds, features, tmp)
            for change in ('split','classes','normalization'):
                altered = copy.deepcopy(ds)
                if change == 'split':
                    altered.img_metadata = altered.img_metadata[:-1]
                elif change == 'classes':
                    altered.class_ids = [0]
                else:
                    altered.transform = transforms.Compose([transforms.Normalize([.1]*3,[.3]*3)])
                with self.assertRaises(ValueError):
                    SimilaritySupportIndex(altered, tmp)
            for bad_index in (1,8):
                corrupt=neighbors.copy()
                if bad_index==1:
                    corrupt[0,0,0]=1  # Same query patient.
                else:
                    corrupt[0,1,0]=4  # B does not contain class2.
                save_bank(bank_path(ds,tmp),dataset_spec(ds),corrupt,scores,'fixture')
                with self.assertRaises(ValueError):
                    SimilaritySupportIndex(ds,tmp)

    def test_too_few_other_patient_slices_and_invalid_descriptors_fail(self):
        ds, features=fixture()
        spec=dataset_spec(ds)
        spec['pools']['0']=spec['names'][:4]
        with self.assertRaises(ValueError):
            rank_descriptors(spec,features)
        features[0,0]=float('nan')
        with self.assertRaises(ValueError):
            rank_descriptors(dataset_spec(ds),features)

    def test_papila_retains_native_sampler_and_random_keeps_legacy_state(self):
        ds=SimpleNamespace(benchmark='papila',neighbor_prob=0.,sample_episode=object())
        original=ds.sample_episode
        configure_support_sampling(ds,'similarity','does-not-exist')
        self.assertIs(ds.sample_episode,original)
        self.assertFalse(hasattr(ds,'support_retriever'))
        legacy,_=fixture()
        before=copy.deepcopy(legacy.__dict__)
        configure_support_sampling(legacy,'random')
        self.assertEqual(set(before),set(legacy.__dict__))
        self.assertEqual(legacy.neighbor_prob,1.)

    def test_strict_random_control_uses_same_class_pool_and_excludes_query_patient(self):
        ds,_=fixture()
        configure_support_sampling(ds,'cross-patient-random')
        for _ in range(20):
            supports=ds.support_retriever.select('x___A_0',1,4)
            self.assertEqual(set(supports),{f'x___C_{i}' for i in range(4)})

    def test_preparation_uses_only_images_and_writes_a_loadable_cache(self):
        ds,_=fixture()
        ds.read_mask=mock.Mock(side_effect=AssertionError('No GT reads during retrieval'))
        ds.read_img=mock.Mock(side_effect=lambda name:Image.new('RGB',(30,40),(80+ord(name[-3]),110,150)))
        class Encoder(torch.nn.Module):
            def forward(self,images):
                return images.mean((2,3))
        with tempfile.TemporaryDirectory() as tmp:
            prepare_dataset(ds,Encoder(),torch.device('cpu'),tmp,'test',batch_size=4,workers=0)
            selector=SimilaritySupportIndex(ds,tmp)
            self.assertEqual(len(selector.select('x___A_0',0,4)),4)
            ds.read_mask.assert_not_called()
            self.assertEqual(ds.read_img.call_count,len(ds.img_metadata))

    def test_descriptor_encoder_is_frozen_eval_and_has_spatial_features(self):
        from model.base.resnet import resnet50
        state=resnet50(pretrained=False).state_dict()
        with mock.patch('prepare_support_retrieval.torch.load',return_value=state):
            encoder=ImageDescriptors('mock')
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
        self.assertTrue(all(not m.training for m in encoder.modules()))
        with torch.inference_mode():
            features=encoder(torch.randn(2,3,64,64))
        self.assertEqual(features.shape,(2,7680))
        torch.testing.assert_close(features.norm(dim=1),torch.ones(2))

    def test_joint_train_and_validation_loaders_both_apply_retrieval(self):
        ds,features=fixture(DatasetAMOS)
        class FakeAMOS(DatasetAMOS):
            def __init__(self,datapath,fold,transform,split,shot,use_original_imgsize,neighbor_prob=0.):
                value,_=fixture(DatasetAMOS,split)
                self.__dict__.update(value.__dict__)
                self.nclass=2
                self.class_names=['one','two']
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(ds,features,tmp)
            trn,features=fixture(DatasetAMOS,'trn')
            write_fixture(trn,features,tmp)
            FSSDataset.initialize(512,'',False)
            with mock.patch.dict(FSSDataset.datasets,{'amos':FakeAMOS}), \
                    mock.patch('torch.utils.data.distributed.DistributedSampler',return_value=None):
                loader=FSSDataset.build_dataloader('joint',1,0,0,'trn',shot=4,
                        joint_subdatasets=[('amos','fake')],n_episodes=8,
                        support_sampling='similarity',support_retrieval_dir=tmp)
                self.assertIsInstance(loader.dataset.sub_datasets[0].support_retriever,SimilaritySupportIndex)
                loaders=DatasetJoint.build_val_dataloaders(1,0,[('amos','fake')],shot=4,
                        support_sampling='similarity',support_retrieval_dir=tmp)
                self.assertIsInstance(next(iter(loaders.values())).dataset.support_retriever,SimilaritySupportIndex)

    def test_launcher_prepares_before_training_and_can_disable_retrieval(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub=Path(tmp)/'python3'
            stub.write_text(f'#!{sys.executable}\nimport json,sys,os\n'
                            'if sys.argv[1]=="-c": print(23456)\n'
                            'else:\n print(json.dumps(sys.argv[1:]))\n'
                            ' if os.environ.get("FAIL_PREPARE") and sys.argv[1]=="prepare_support_retrieval.py": sys.exit(17)\n')
            stub.chmod(0o755)
            env=dict(os.environ,PATH=tmp+os.pathsep+os.environ['PATH'],GPU_ID='1,2',LAYER3_MODE='frozen')
            for name in ('SUPPORT_SAMPLING','EXPERIMENT_NAME','NEIGHBOR_PROB'):
                env.pop(name,None)
            def run():
                value=subprocess.run(['sh',str(ROOT/'train_joint.sh')],env=env,cwd=tmp,
                                     capture_output=True,text=True)
                calls=[json.loads(line) for line in value.stdout.splitlines() if line.startswith('[')]
                return value,calls
            result,calls=run()
            self.assertEqual(result.returncode,0)
            self.assertEqual(calls[0][0],'prepare_support_retrieval.py')
            training=calls[-1]
            self.assertEqual(training[training.index('--support-sampling')+1],'similarity')
            self.assertEqual(training[training.index('--neighbor-prob')+1],'0.0')
            self.assertIn('_supportsimilarity_',training[training.index('--logpath')+1])
            env['FAIL_PREPARE']='1'
            result,calls=run()
            self.assertEqual(result.returncode,17)
            self.assertEqual(len(calls),1)
            env.pop('FAIL_PREPARE')
            env['SUPPORT_SAMPLING']='cross-patient-random'
            result,calls=run()
            self.assertEqual(result.returncode,0)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][0],'-m')


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
