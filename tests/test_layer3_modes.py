"""Layer3 ablations: running statistics, gradients, checkpoints, and launcher."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'segment-anything'))
from model.VRP_encoder import VRP_encoder
from model.base import resnet


def make_encoder(mode=None):
    args = SimpleNamespace(num_query=50)
    if mode is not None:
        args.layer3_mode = mode
    backbone = resnet.resnet50(pretrained=False)
    with mock.patch('model.VRP_encoder.models.resnet50', return_value=backbone):
        return VRP_encoder(args, 'resnet50', False)


def snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


class Layer3ModeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(321)

    def test_frozen_layer3_is_unchanged_while_feature_projection_learns(self):
        model = make_encoder('frozen')
        self.assertTrue(all(not p.requires_grad for p in model.layer3.parameters()))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) not in optimized for p in model.layer3.parameters()))
        # Validation and subsequent epoch setup must not reopen the frozen BN.
        model.eval()
        model.train_mode()
        self.assertFalse(model.layer3.training)
        before = snapshot(model.layer3)
        projection_before = model.downsample_query[0].weight.detach().clone()
        query = torch.randn(2, 512, 8, 8)
        support = torch.randn_like(query) + 2
        for inputs in (query, support):
            features = model.layer3(inputs)
            self.assertFalse(features.requires_grad)
            projected = model.downsample_query(torch.cat([inputs, features], dim=1))
            projected.square().mean().backward()
        optimizer.step()
        for name, value in model.layer3.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.layer3.parameters()))
        self.assertFalse(torch.equal(projection_before, model.downsample_query[0].weight))

    def test_fixed_bn_allows_convolution_and_bn_affine_updates(self):
        model = make_encoder('finetune-frozen-bn')
        model.train_mode()
        self.assertTrue(model.layer3.training)
        bn = [m for m in model.layer3.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        self.assertEqual(len(bn), 19)
        self.assertTrue(all(not m.training for m in bn))
        buffers_before = {name: value.clone() for name, value in model.layer3.named_buffers()}
        conv = model.layer3[0].conv1.weight
        affine = model.layer3[0].bn1.weight
        conv_before, affine_before = conv.detach().clone(), affine.detach().clone()
        optimizer = torch.optim.SGD(model.layer3.parameters(), lr=1e-3)
        model.layer3(torch.randn(2, 512, 8, 8) + 2).square().mean().backward()
        self.assertIsNotNone(conv.grad)
        self.assertIsNotNone(affine.grad)
        optimizer.step()
        self.assertFalse(torch.equal(conv_before, conv))
        self.assertFalse(torch.equal(affine_before, affine))
        for name, value in model.layer3.named_buffers():
            torch.testing.assert_close(value, buffers_before[name], rtol=0, atol=0)

    def test_default_preserves_legacy_two_bn_updates_and_checkpoint_keys(self):
        model = make_encoder()
        model.train_mode()
        bn = model.layer3[0].bn1
        before_mean = bn.running_mean.clone()
        count = bn.num_batches_tracked.item()
        with torch.no_grad():
            model.layer3(torch.randn(2, 512, 8, 8))
            model.layer3(torch.randn(2, 512, 8, 8) + 2)
        self.assertTrue(bn.training)
        self.assertEqual(bn.num_batches_tracked.item(), count + 2)
        self.assertFalse(torch.equal(before_mean, bn.running_mean))
        state = snapshot(model)
        del model
        frozen = make_encoder('frozen').eval()
        frozen.load_state_dict(state, strict=True)
        self.assertEqual(set(state), set(frozen.state_dict()))
        self.assertTrue(all(not p.requires_grad for p in frozen.layer3.parameters()))

    def test_sync_bn_keeps_each_modes_flags_across_epoch_transitions(self):
        for mode in ('finetune', 'finetune-frozen-bn', 'frozen'):
            with self.subTest(mode=mode):
                model = nn.SyncBatchNorm.convert_sync_batchnorm(make_encoder(mode))
                model.eval()
                model.train_mode()
                bn = [m for m in model.layer3.modules() if isinstance(m, nn.SyncBatchNorm)]
                self.assertEqual(len(bn), 19)
                self.assertTrue(all(m.training == (mode == 'finetune') for m in bn))
                self.assertTrue(all(p.requires_grad == (mode != 'frozen')
                                    for p in model.layer3.parameters()))
                for layer in (model.layer0, model.layer1, model.layer2, model.layer4):
                    self.assertTrue(all(not m.training for m in layer.modules()
                                        if isinstance(m, nn.SyncBatchNorm)))
                del model

    def test_launcher_forwards_mode_and_uses_distinct_log_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / 'python3'
            python.write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then\n'
                              '  echo 12345\nelse\n  printf "%s\\n" "$@"\nfi\n')
            python.chmod(0o755)
            env = os.environ.copy()
            env.update(PATH=tmp + os.pathsep + env.get('PATH', ''), GPU_ID='1,2',
                       SUPPORT_SAMPLING='random', EPOCHS='100')
            env.pop('EXPERIMENT_NAME', None)
            for mode in ('finetune', 'finetune-frozen-bn', 'frozen'):
                with self.subTest(mode=mode):
                    env['LAYER3_MODE'] = mode
                    run = subprocess.run(['sh', 'train_joint.sh'], cwd=ROOT, env=env,
                                         text=True, capture_output=True, check=True)
                    args = run.stdout.splitlines()
                    self.assertEqual(args[args.index('--layer3-mode') + 1], mode)
                    logpath = args[args.index('--logpath') + 1]
                    self.assertTrue(logpath.endswith('_ep100'))
                    self.assertEqual(args[args.index('--epochs') + 1], '100')
                    self.assertEqual(logpath.removesuffix('_ep100').endswith('_l3' + mode),
                                     mode != 'finetune')
            env['LAYER3_MODE'] = 'invalid'
            run = subprocess.run(['sh', 'train_joint.sh'], cwd=ROOT, env=env,
                                 text=True, capture_output=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn('LAYER3_MODE', run.stderr)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
