"""Exercise tiny/empty support masks through the actual prompt encoder."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'segment-anything'))
from model.VRP_encoder import VRP_encoder
from model.base import resnet


class SupportMaskPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(321)
        # Avoid pretrained downloads; retain the real backbone and both branches.
        backbone = resnet.resnet50(pretrained=False)
        with mock.patch('model.VRP_encoder.models.resnet50', return_value=backbone):
            cls.model = VRP_encoder(SimpleNamespace(num_query=50), 'resnet50', False)

    def test_small_and_empty_supports_reach_both_branches_in_train_and_eval(self):
        query = torch.randn(3, 3, 32, 32)
        support = torch.randn(3, 3, 32, 32)
        mask = torch.zeros(3, 32, 32)
        mask[0, 1, 1] = 1  # Falls between the old nearest sample locations.
        mask[1, -1, -1] = 1  # Include the last row/column.
        legacy = F.interpolate(mask[:, None], size=(4, 4), mode='nearest')
        self.assertEqual(legacy.sum().item(), 0)

        for training in (False, True):
            with self.subTest(training=training):
                captured = {}
                def capture_icl(module, args):
                    captured['mask'] = args[2].detach().clone()
                def capture_bbox(module, args):
                    captured['bbox_features'] = args[1].detach().clone()
                handles = [
                    self.model.eicl_interaction.register_forward_pre_hook(capture_icl),
                    self.model.bbox_transfer.register_forward_pre_hook(capture_bbox),
                ]
                self.model.train(training)
                self.model.zero_grad(set_to_none=True)
                try:
                    tokens, boxes, pseudo = self.model(
                        'mask', query, support, mask, training)
                finally:
                    for handle in handles:
                        handle.remove()

                reduced = captured['mask']
                self.assertEqual(tuple(reduced.shape), (3, 1, 4, 4))
                torch.testing.assert_close(reduced.sum((1, 2, 3)),
                                           torch.tensor([1., 1., 0.]))
                self.assertEqual(reduced[0, 0, 0, 0].item(), 1)
                self.assertEqual(reduced[1, 0, -1, -1].item(), 1)
                features = captured['bbox_features']
                self.assertTrue((features[:2].abs().sum((1, 2, 3)) > 0).all())
                self.assertEqual(features[2].abs().sum().item(), 0)
                self.assertEqual((features * (1 - reduced)).abs().sum().item(), 0)
                self.assertTrue((pseudo[:2].flatten(1).amax(1) > 0).all())
                self.assertEqual(pseudo[2].abs().sum().item(), 0)
                self.assertEqual(tuple(tokens.shape), (3, 50, 256))
                self.assertEqual(tuple(boxes.shape), (3, 2, 256))
                for output in (tokens, boxes, pseudo):
                    self.assertTrue(torch.isfinite(output).all())
                if training:
                    (tokens.square().mean() + boxes.square().mean()
                     + pseudo.square().mean()).backward()
                    grad = self.model.downsample_query[0].weight.grad
                    self.assertIsNotNone(grad)
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)

    def test_non_divisible_input_keeps_border_foreground(self):
        self.model.eval()
        mask = torch.zeros(1, 1, 33, 35)
        mask[0, 0, -1, -1] = 1
        captured = []
        handle = self.model.eicl_interaction.register_forward_pre_hook(
            lambda module, args: captured.append(args[2].detach().clone()))
        try:
            with torch.no_grad():
                outputs = self.model('mask', torch.randn(1, 3, 33, 35),
                                     torch.randn(1, 3, 33, 35), mask, False)
        finally:
            handle.remove()
        self.assertEqual(captured[0][0, 0, -1, -1].item(), 1)
        for output in outputs:
            self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
