"""Quality supervision, optimizer updates, checkpoint state, and valid parity."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'segment-anything'))
from model.base.support_fusion import (SupportQualityHead, checkpoint_fusion_mode,
                                      fuse_support_logits, fusion_mix_for_epoch,
                                      support_quality_loss, support_quality_targets)
from model.VRP_encoder import VRP_encoder
from model.base import resnet
from test_support_fusion import NonlinearSAM, Prompt, make_batch


class QualityPrompt(Prompt):
    set_fusion_epoch = VRP_encoder.set_fusion_epoch

    def __init__(self):
        super().__init__()
        self.support_fusion = 'quality'
        self.support_quality_head = SupportQualityHead(4)
        self.fusion_warmup_epochs, self.fusion_ramp_epochs = 2, 3
        self.scale = nn.Parameter(torch.tensor(1.))

    def train_mode(self):
        self.train()

    def forward(self, condition, query, support, mask, training, dataset_names=None,
                return_quality=False):
        self.names = dataset_names
        value = support[:, 0, 0, 0] * self.scale
        features = torch.stack([value, value.square(), value * 0 + 1, value * 0], dim=1)
        tokens = features[:, None].expand(-1, 3, -1)
        boxes = features[:, None].expand(-1, 2, -1)
        if return_quality:
            grid = F.adaptive_max_pool2d(mask[:, None], (2, 2))
            scores = self.support_quality_head(tokens, boxes, mask, grid)
            return tokens, boxes, None, scores
        return tokens, boxes, None


class SupportQualityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(321)

    def test_head_starts_uniform_detaches_features_and_learns_quality(self):
        head = SupportQualityHead()
        self.assertEqual(sum(p.numel() for p in head.parameters()), 50945)
        tokens = torch.randn(4, 50, 256, requires_grad=True)
        boxes = torch.randn(4, 2, 256, requires_grad=True)
        mask = torch.ones(4, 1, 8, 8, requires_grad=True)
        grid = torch.ones(4, 1, 2, 2, requires_grad=True)
        scores = head(tokens, boxes, mask, grid).reshape(1, 4)
        logits = torch.full((1, 4, 2, 2), -10., requires_grad=True)
        with torch.no_grad():
            logits[:, 0, 0, 0] = 10
        target = torch.zeros(1, 2, 2); target[:, 0, 0] = 1
        _, weights = fuse_support_logits(logits, scores)
        torch.testing.assert_close(weights, torch.full((1, 4), .25), rtol=0, atol=0)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
        for _ in range(4):
            scores = head(tokens, boxes, mask, grid).reshape(1, 4)
            loss, _, _ = support_quality_loss(scores, logits, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        for tensor in [tokens, boxes, mask, grid, logits]:
            self.assertIsNone(tensor.grad)
        learned = head(tokens, boxes, mask, grid).softmax(0)
        self.assertGreater(learned[0].item(), .25)

    def test_teacher_handles_empty_queries_and_loss_reaches_only_score_branch(self):
        logits = torch.full((2, 4, 2, 2), -10., requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 0, 0] = 10
        target = torch.zeros(2, 2, 2); target[0, 0, 0] = 1
        beta, dice, valid = support_quality_targets(logits, target)
        torch.testing.assert_close(dice[0], torch.tensor([1., 0., 0., 0.]))
        self.assertEqual(valid.tolist(), [True, False])
        self.assertFalse(beta.requires_grad)
        scores = torch.randn(2, 4, requires_grad=True)
        loss, _, _ = support_quality_loss(scores, logits, target)
        loss.backward()
        self.assertGreater(scores.grad[0].abs().sum().item(), 0)
        self.assertEqual(scores.grad[1].abs().sum().item(), 0)
        self.assertIsNone(logits.grad)
        empty_scores = torch.randn(2, 4, requires_grad=True)
        empty_loss, _, _ = support_quality_loss(empty_scores, logits, target * 0)
        self.assertEqual(empty_loss.item(), 0)
        empty_loss.backward()
        self.assertTrue(torch.isfinite(empty_scores.grad).all())
        for temperature in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                support_quality_targets(logits, target, temperature)

    def test_schedule_weighted_recovery_shot_one_and_permutation(self):
        self.assertEqual([fusion_mix_for_epoch(e) for e in range(6)],
                         [0, 0, 1/3, 2/3, 1, 1])
        self.assertEqual(fusion_mix_for_epoch(0, 0, 0), 1)
        logits = torch.tensor([[[[3.]], [[-2.]], [[-2.]], [[-2.]]]], requires_grad=True)
        scores = torch.tensor([[4., 0., 0., 0.]], requires_grad=True)
        uniform, _ = fuse_support_logits(logits)
        warm, warm_weights = fuse_support_logits(logits, scores, mix=0)
        torch.testing.assert_close(warm, uniform, rtol=0, atol=0)
        torch.testing.assert_close(warm_weights, torch.full((1, 4), .25))
        fused, weights = fuse_support_logits(logits, scores)
        self.assertLess(uniform.item(), 0)
        self.assertGreater(fused.item(), 0)
        fused.sum().backward()
        self.assertTrue((logits.grad.abs() > 0).all())
        self.assertTrue((scores.grad.abs() > 0).all())
        order = [3, 0, 2, 1]
        permuted, _ = fuse_support_logits(logits[:, order], scores[:, order])
        torch.testing.assert_close(permuted, fused)
        single, weights = fuse_support_logits(logits[:, :1], scores[:, :1], mix=.5)
        torch.testing.assert_close(single, logits[:, 0])
        torch.testing.assert_close(weights, torch.ones(1, 1))

    def test_train_updates_quality_head_and_valid_matches_label_free_inference(self):
        import train as runner

        class Wrapper(nn.Module):
            def __init__(self, module):
                super().__init__(); self.module = module
            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        batch = make_batch()
        class Loader:
            dataset = SimpleNamespace(benchmark='joint')
            def __iter__(self): return iter([batch])
            def __len__(self): return 1

        model, sam = QualityPrompt(), NonlinearSAM()
        args = SimpleNamespace(seed=321, condition='mask', benchmark='joint', bbox_loss_weight=.5,
                               support_fusion='quality', fusion_target_temperature=.2,
                               fusion_quality_weight=.1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        meter = mock.Mock(); meter.loss_buf = []
        meter.update.side_effect = lambda inter, union, ids, loss: meter.loss_buf.append(loss)
        meter.compute_iou.return_value = (torch.zeros(()), torch.zeros(()))
        meter.compute_dice.return_value = torch.zeros(())
        initial = model.support_quality_head.mlp[-1].weight.detach().clone()
        with mock.patch.object(runner.utils, 'to_cuda', side_effect=lambda x: x), \
             mock.patch.object(runner.utils, 'is_main_process', return_value=False), \
             mock.patch.object(runner, 'AverageMeter', return_value=meter), \
             mock.patch.object(runner.Evaluator, 'classify_prediction',
                               return_value=(torch.zeros(2, 2), torch.zeros(2, 2))):
            # Quality loss must update the head even during uniform warmup.
            runner.train(args, 0, Wrapper(model), sam, Loader(), optimizer, mock.Mock(), True)
            self.assertEqual(model.support_quality_head.mix.item(), 0)
            self.assertFalse(torch.equal(initial, model.support_quality_head.mlp[-1].weight))
            model.set_fusion_epoch(2)
            label_free = dict(batch); label_free.pop('query_mask')
            model.eval()
            expected = model.predict_mask_nshot(args, label_free, sam, 4)[2][:, 0]
            runner.train(args, 99, Wrapper(model), sam, Loader(), None, None, False)
            torch.testing.assert_close(model.observed_logits, expected, rtol=0, atol=0)
            self.assertAlmostEqual(model.support_quality_head.mix.item(), 1/3)

    def test_real_encoder_quality_forward_and_checkpoint_restore(self):
        args = SimpleNamespace(num_query=50, benchmark='joint', support_fusion='quality',
                               support_mask_pooling='max', bbox_support_attn_mask=1,
                               icl_support_attn_mask=1)
        factory = resnet.resnet50
        with mock.patch('model.VRP_encoder.models.resnet50', side_effect=lambda **kw: factory(pretrained=False)):
            model = VRP_encoder(args, 'resnet50', False).eval()
        query, support = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
        mask = torch.zeros(4, 32, 32); mask[:3, 1, 1] = 1
        with torch.no_grad():
            old_interface = model('mask', query, support, mask, False)
            outputs = model('mask', query, support, mask, False, return_quality=True)
        self.assertEqual(len(old_interface), 3)
        self.assertEqual(len(outputs), 4)
        for a, b in zip(old_interface, outputs):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(tuple(outputs[3].shape), (4,))
        self.assertTrue(torch.isfinite(outputs[3]).all())
        model.set_fusion_epoch(2)
        state = copy.deepcopy(model.state_dict())
        checkpoint = {'model': {'module.' + k: v for k, v in state.items()},
                      'config': {'support_fusion': 'quality'}}
        self.assertEqual(checkpoint_fusion_mode(checkpoint), 'quality')
        with self.assertRaises(ValueError): checkpoint_fusion_mode(checkpoint, 'mean')
        model.set_fusion_epoch(5)
        model.load_state_dict(state, strict=True)
        self.assertAlmostEqual(model.support_quality_head.mix.item(), 1/3)
        mean_state = {k: v for k, v in state.items() if not k.startswith('support_quality_head.')}
        self.assertEqual(checkpoint_fusion_mode(mean_state), 'mean')
        with self.assertRaises(ValueError):
            checkpoint_fusion_mode({'model': mean_state, 'config': {'support_fusion': 'quality'}})
        with mock.patch('model.VRP_encoder.models.resnet50', side_effect=lambda **kw: factory(pretrained=False)):
            mean_model = VRP_encoder(SimpleNamespace(num_query=50), 'resnet50', False)
        mean_model.load_state_dict(mean_state, strict=True)

    def test_logger_checkpoint_persists_config_epoch_and_mix(self):
        from common.logger import Logger
        model = QualityPrompt(); model.set_fusion_epoch(2)
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(Logger, 'logpath', directory, create=True), \
             mock.patch.object(Logger, 'run_config', {'support_fusion': 'quality'}, create=True):
            Logger.save_model_dice(model, nn.Linear(2, 2), 2, 35.5)
            saved = torch.load(Path(directory) / 'best_model.pt', weights_only=True)
        self.assertEqual(saved['epoch'], 2)
        self.assertEqual(saved['config']['support_fusion'], 'quality')
        self.assertAlmostEqual(saved['model']['support_quality_head.mix'].item(), 1/3)
        self.assertEqual(checkpoint_fusion_mode(saved), 'quality')


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
