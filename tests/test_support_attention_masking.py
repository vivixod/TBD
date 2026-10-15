"""Check background exclusion, empty supports, and independent ablation wiring."""
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

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'segment-anything'))
from model.base.attention_mask import support_background_mask
from model.base.bbox_transfer import BboxTransferModule
from model.base.transformer_decoder import transformer_decoder
from model.VRP_encoder import VRP_encoder
from model.base import resnet


class SupportAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(321)
        self.query = torch.randn(3, 32, 2, 3)
        self.support = torch.randn(3, 32, 2, 3)
        self.mask = torch.zeros(3, 1, 2, 3)
        self.mask[0, 0, 1, 2] = 1  # Sparse foreground, empty sample, then full mask.
        self.mask[2] = 1
        self.changed = self.support + (1 - self.mask) * torch.randn_like(self.support) * 100

    def test_mask_shapes_and_polarity(self):
        expected = torch.tensor([[True] * 5 + [False], [True] * 6, [False] * 6])
        for mask in (self.mask, self.mask[:, 0]):
            torch.testing.assert_close(support_background_mask(mask, self.support), expected)
        for bad in (None, self.mask[:, :, :, :2], self.mask[:2],
                    self.mask.expand(-1, 2, -1, -1)):
            with self.assertRaises(ValueError):
                support_background_mask(bad, self.support)

    def test_bbox_ignores_background_and_keeps_bbox_keys_for_empty_support(self):
        module = BboxTransferModule(32, 4, num_layers=2, dropout=0,
                                    mask_support_attention=True).eval()
        boxes = torch.randn(3, 2, 32)
        captured = []
        def capture(module, args, kwargs):
            captured.append(kwargs.get('key_padding_mask'))
        handles = [module.supp_cross_attn[0].register_forward_pre_hook(capture, with_kwargs=True),
                   module.qry_cross_attn[0].register_forward_pre_hook(capture, with_kwargs=True)]
        support = self.support.clone().requires_grad_()
        try:
            result = module(boxes, support, self.query, supp_mask=self.mask)
        finally:
            for h in handles:
                h.remove()
        expected = torch.cat([torch.zeros(3, 2, dtype=torch.bool),
                              support_background_mask(self.mask, support)], dim=1)
        torch.testing.assert_close(captured[0], expected)
        self.assertIsNone(captured[1])  # Query attention remains unrestricted.
        reference = module(boxes, self.changed, self.query, supp_mask=self.mask)
        torch.testing.assert_close(result, reference)
        self.assertTrue(torch.isfinite(result).all())
        (result * torch.randn_like(result)).sum().backward()
        self.assertTrue(torch.isfinite(support.grad).all())
        self.assertEqual((support.grad * (1 - self.mask)).abs().sum().item(), 0)
        self.assertGreater((support.grad * self.mask).abs().sum().item(), 0)

    def test_icl_ignores_background_and_only_unmasks_empty_samples(self):
        for pre_norm in (False, True):
            with self.subTest(pre_norm=pre_norm):
                module = transformer_decoder(SimpleNamespace(icl_support_attn_mask=1),
                                             5, 32, 64, nheads=4, num_layers=2,
                                             pre_norm=pre_norm).eval()
                captured = []
                def capture(module, args, kwargs):
                    captured.append(kwargs.get('memory_key_padding_mask'))
                handles = [module.transformer_cross_attention_layers_0[0]
                           .register_forward_pre_hook(capture, with_kwargs=True),
                           module.transformer_cross_attention_layers[0]
                           .register_forward_pre_hook(capture, with_kwargs=True)]
                support = self.support.clone().requires_grad_()
                try:
                    result = module(self.query, support, self.mask)
                finally:
                    for h in handles:
                        h.remove()
                expected = support_background_mask(self.mask, support)
                expected[1] = False
                torch.testing.assert_close(captured[0], expected)
                self.assertIsNone(captured[1])
                reference = module(self.query, self.changed, self.mask)
                torch.testing.assert_close(result[[0, 2]], reference[[0, 2]])
                self.assertTrue(torch.isfinite(result).all())
                (result * torch.randn_like(result)).sum().backward()
                self.assertTrue(torch.isfinite(support.grad).all())
                self.assertEqual((support.grad[0] * (1 - self.mask[0])).abs().sum().item(), 0)
                self.assertGreater(support.grad[0].abs().sum().item(), 0)
                self.assertGreater(support.grad[1].abs().sum().item(), 0)

    def test_disabled_modules_accept_legacy_calls_and_strict_checkpoint_loading(self):
        bbox = BboxTransferModule(32, 4, dropout=0).eval()
        enabled_bbox = BboxTransferModule(32, 4, dropout=0, mask_support_attention=True).eval()
        enabled_bbox.load_state_dict(bbox.state_dict(), strict=True)
        boxes = torch.randn(3, 2, 32)
        torch.testing.assert_close(bbox(boxes, self.support, self.query),
                                   bbox(boxes, self.support, self.query, supp_mask=self.mask),
                                   rtol=0, atol=0)
        legacy = transformer_decoder(SimpleNamespace(), 5, 32, 64).eval()
        explicit_off = transformer_decoder(SimpleNamespace(icl_support_attn_mask=0),
                                            5, 32, 64).eval()
        explicit_off.load_state_dict(legacy.state_dict(), strict=True)
        torch.testing.assert_close(legacy(self.query, self.support, self.mask),
                                   explicit_off(self.query, self.support, self.mask),
                                   rtol=0, atol=0)
        self.assertFalse(bbox.mask_support_attention)
        self.assertFalse(legacy.mask_support_attention)

    def test_full_encoder_switches_are_independent_and_retrieval_is_unchanged(self):
        backbone = resnet.resnet50(pretrained=False)
        with mock.patch('model.VRP_encoder.models.resnet50', return_value=backbone):
            model = VRP_encoder(SimpleNamespace(num_query=50, bbox_support_attn_mask=1,
                                                icl_support_attn_mask=1), 'resnet50', False)
        self.assertTrue(model.bbox_transfer.mask_support_attention)
        self.assertTrue(model.transformer_decoder.mask_support_attention)
        model.eval()
        query, support = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
        mask = torch.zeros(2, 32, 32)
        mask[0, 1, 1] = 1
        baseline_state = copy.deepcopy(model.state_dict())
        outputs = {}
        with torch.no_grad():
            for bbox, icl in ((0, 0), (1, 0), (0, 1), (1, 1)):
                model.bbox_transfer.mask_support_attention = bool(bbox)
                model.transformer_decoder.mask_support_attention = bool(icl)
                outputs[bbox, icl] = model('mask', query, support, mask, False)
                for output in outputs[bbox, icl]:
                    self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(outputs[bbox, icl][2], outputs[0, 0][2],
                                           rtol=0, atol=0)
        torch.testing.assert_close(outputs[1, 0][0], outputs[0, 0][0], rtol=0, atol=0)
        torch.testing.assert_close(outputs[0, 1][1], outputs[0, 0][1], rtol=0, atol=0)
        self.assertFalse(torch.allclose(outputs[1, 0][1], outputs[0, 0][1]))
        self.assertFalse(torch.allclose(outputs[0, 1][0], outputs[0, 0][0]))
        model.load_state_dict(baseline_state, strict=True)
        model.train_mode()
        tokens, boxes, pseudo = model('mask', query, support, mask, True)
        (tokens * torch.randn_like(tokens)).mean().backward(retain_graph=True)
        (boxes * torch.randn_like(boxes)).mean().backward()
        for parameters in (model.bbox_transfer.parameters(), model.transformer_decoder.parameters()):
            grads = [p.grad for p in parameters if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))


class AttentionLauncherTests(unittest.TestCase):
    def test_four_ablation_modes_route_flags_and_name_logs_without_starting_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / 'python3'
            python.write_text(f'#!{sys.executable}\n'
                              'import json, sys\n'
                              'if sys.argv[1] == "-c": print(23456)\n'
                              'else: print(json.dumps(sys.argv[1:]))\n')
            python.chmod(0o755)
            for bbox, icl in ((0, 0), (1, 0), (0, 1), (1, 1)):
                with self.subTest(bbox=bbox, icl=icl):
                    env = dict(os.environ, PATH=tmp + os.pathsep + os.environ['PATH'],
                               BBOX_SUPPORT_ATTN_MASK=str(bbox), ICL_SUPPORT_ATTN_MASK=str(icl),
                               CLASS_BALANCE_PROB='0.5')
                    env.pop('EXPERIMENT_NAME', None)
                    result = subprocess.run(['sh', str(ROOT / 'train_joint.sh')],
                                            env=env, capture_output=True, text=True, check=True)
                    args = json.loads(result.stdout.splitlines()[-1])
                    self.assertEqual(args[args.index('--bbox-support-attn-mask') + 1], str(bbox))
                    self.assertEqual(args[args.index('--icl-support-attn-mask') + 1], str(icl))
                    expected = 'trn_joint4_shot4_cb0.5'
                    if bbox or icl:
                        expected += f'_bboxm{bbox}_iclm{icl}'
                    self.assertEqual(args[args.index('--logpath') + 1], expected)
                    self.assertEqual(args[args.index('--epochs') + 1], '300')
                    self.assertEqual(args[args.index('--n-episodes') + 1], '8000')
            env['BBOX_SUPPORT_ATTN_MASK'] = '2'
            result = subprocess.run(['sh', str(ROOT / 'train_joint.sh')],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
