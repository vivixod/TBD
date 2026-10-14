"""Numerical and checkpoint compatibility tests for the training speed changes."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'segment-anything'))
from model.base.transformer_decoder import SelfAttentionLayer, CrossAttentionLayer
from model.base.bbox_transfer import BboxTransferModule
from model.VRP_encoder import VRP_encoder
from model.base import resnet


def attention_with_weights(original):
    def forward(module, *args, **kwargs):
        kwargs['need_weights'] = True
        return original(module, *args, **kwargs)
    return forward


class SpeedupTests(unittest.TestCase):
    def test_attention_outputs_and_input_gradients_match_legacy_path(self):
        original = nn.MultiheadAttention.forward
        torch.manual_seed(321)
        for pre_norm in (False, True):
            for kind in ('self', 'cross', 'bbox'):
                with self.subTest(pre_norm=pre_norm, kind=kind):
                    if kind == 'self':
                        layer = SelfAttentionLayer(32, 4, normalize_before=pre_norm)
                        inputs = [torch.randn(3, 2, 32)]
                    elif kind == 'cross':
                        layer = CrossAttentionLayer(32, 4, normalize_before=pre_norm)
                        inputs = [torch.randn(3, 2, 32), torch.randn(6, 2, 32),
                                  torch.randn(6, 2, 32)]
                    else:
                        layer = BboxTransferModule(hidden_dim=32, nhead=4, dropout=0)
                        inputs = [torch.randn(2, 2, 32), torch.randn(2, 32, 4, 4),
                                  torch.randn(2, 32, 4, 4)]
                    layer.eval()
                    reference_layer = copy.deepcopy(layer)
                    current_inputs = [x.clone().requires_grad_() for x in inputs]
                    reference_inputs = [x.clone().requires_grad_() for x in inputs]
                    current = layer(*current_inputs)
                    with mock.patch.object(nn.MultiheadAttention, 'forward',
                                           attention_with_weights(original)):
                        reference = reference_layer(*reference_inputs)
                    torch.testing.assert_close(current, reference, rtol=2e-4, atol=2e-5)
                    current.square().sum().backward()
                    reference.square().sum().backward()
                    for x, y in zip(current_inputs, reference_inputs):
                        torch.testing.assert_close(x.grad, y.grad, rtol=1e-3, atol=2e-5)
                    for x, y in zip(layer.parameters(), reference_layer.parameters()):
                        if x.grad is not None:
                            torch.testing.assert_close(x.grad, y.grad, rtol=1e-3, atol=2e-5)

    def test_skipping_layer4_preserves_vrp_outputs_and_checkpoint_keys(self):
        torch.manual_seed(321)
        backbone = resnet.resnet50(pretrained=False)
        with mock.patch('model.VRP_encoder.models.resnet50', return_value=backbone):
            model = VRP_encoder(SimpleNamespace(num_query=50), 'resnet50', False)
        model.eval()
        query = torch.randn(1, 3, 32, 32)
        support = torch.randn(1, 3, 32, 32)
        mask = torch.zeros(1, 32, 32)
        mask[:, 8:24, 8:24] = 1
        calls = []
        hook = model.layer4.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            current = model('mask', query, support, mask, False)
            self.assertEqual(calls, [])
            # Reproduce the discarded layer4 calls in the former forward path.
            legacy_hook = model.layer3.register_forward_hook(
                lambda module, args, output: (model.layer4(output), None)[1])
            try:
                reference = model('mask', query, support, mask, False)
            finally:
                legacy_hook.remove()
        hook.remove()
        self.assertEqual(len(calls), 2)
        for x, y in zip(current, reference):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertTrue(any(k.startswith('layer4.') for k in model.state_dict()))
        self.assertTrue(all(not p.requires_grad for p in model.layer4.parameters()))
        model.load_state_dict(model.state_dict(), strict=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
