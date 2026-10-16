"""Cross-entry regression checks for per-shot decoding and logit fusion."""
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
from model.base.support_fusion import decode_support_logits, mean_support_logits


class Prompt(nn.Module):
    predict_mask_nshot = VRP_encoder.predict_mask_nshot

    def __init__(self):
        super().__init__()
        self.use_original_imgsize = False
        self.names = None
        self.observed_logits = None

    def forward(self, condition, query, support, mask, training, dataset_names=None):
        self.names = dataset_names
        value = support[:, 0, 0, 0]
        tokens = value[:, None, None].expand(-1, 3, 4)
        boxes = value[:, None, None].expand(-1, 2, 4)
        return tokens, boxes, None

    def compute_bbox_token_loss(self, boxes, masks):
        return boxes.new_zeros(()), boxes.new_ones(())

    def compute_objective(self, logits, mask):
        self.observed_logits = logits.clone()
        return logits.square().mean()


class NonlinearSAM(nn.Module):
    """Nonlinearity makes mean(tokens)->decode observably different."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, images, names, prompts, image_repeats=1):
        self.calls.append((len(images), list(names), image_repeats, prompts.shape[0]))
        value = prompts.mean((1, 2)).square() - 5
        logits = value[:, None, None, None].expand(-1, 1, *images.shape[-2:])
        return logits, (logits[:, 0] > 0).float()


def make_batch():
    values = torch.tensor([[-3., -1., 1., 3.], [1., 2., 3., 4.]])
    return {
        'query_img': torch.zeros(2, 3, 3, 5),
        'query_name': ['q1', 'q2'],
        'query_mask': torch.ones(2, 3, 5),
        'support_imgs': values[:, :, None, None, None].expand(-1, -1, 3, 3, 5),
        'support_masks': torch.ones(2, 4, 3, 5),
        'dataset_name': ['adam', 'amos'],
        'class_id': torch.zeros(2, dtype=torch.long),
    }


class SupportFusionTests(unittest.TestCase):
    def setUp(self):
        self.batch = make_batch()
        self.args = SimpleNamespace(condition='mask', benchmark='joint', seed=321,
                                    bbox_loss_weight=0.5)

    def test_independent_prediction_decodes_each_shot_in_query_order_without_query_gt(self):
        model, sam = Prompt(), NonlinearSAM()
        batch = dict(self.batch)
        batch.pop('query_mask')
        aggregate, pred, logits = model.predict_mask_nshot(self.args, batch, sam, 4)
        values = batch['support_imgs'][:, :, 0, 0, 0]
        expected = (values.square() - 5).mean(dim=1)[:, None, None, None].expand_as(logits)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        self.assertFalse(torch.allclose(logits[:, 0, 0, 0], values.mean(dim=1).square() - 5))
        self.assertEqual(model.names, ['adam'] * 4 + ['amos'] * 4)
        self.assertEqual(sam.calls, [(2, ['q1', 'q2'], 4, 8)])
        self.assertEqual(tuple(pred.shape), (2, 1, 3, 5))
        torch.testing.assert_close(aggregate, pred[:, 0])
        # q1's averaged logit is exactly zero: use train/valid's strict threshold.
        self.assertEqual(pred[0].sum().item(), 0)

    def test_shot_one_subset_and_support_permutation(self):
        for shot in [1, 2, 4]:
            with self.subTest(shot=shot):
                logits = Prompt().predict_mask_nshot(self.args, self.batch, NonlinearSAM(), shot)[2]
                values = self.batch['support_imgs'][:, :shot, 0, 0, 0]
                torch.testing.assert_close(logits[:, 0, 0, 0], (values.square() - 5).mean(1))
        baseline = Prompt().predict_mask_nshot(self.args, self.batch, NonlinearSAM(), 4)[2]
        shuffled = dict(self.batch)
        order = [3, 0, 2, 1]
        shuffled['support_imgs'] = shuffled['support_imgs'][:, order]
        shuffled['support_masks'] = shuffled['support_masks'][:, order]
        actual = Prompt().predict_mask_nshot(self.args, shuffled, NonlinearSAM(), 4)[2]
        torch.testing.assert_close(actual, baseline)
        for shot in [0, -1, 5, 1.5, True]:
            with self.assertRaises(ValueError):
                Prompt().predict_mask_nshot(self.args, self.batch, NonlinearSAM(), shot)

    def test_original_size_restoration_preserves_return_contract(self):
        batch = {k: v[:1] for k, v in self.batch.items()}
        batch['org_query_imsize'] = [torch.tensor([9]), torch.tensor([7])]
        model = Prompt()
        small = model.predict_mask_nshot(self.args, batch, NonlinearSAM(), 4)[2]
        model.use_original_imgsize = True
        aggregate, pred, actual = model.predict_mask_nshot(self.args, batch, NonlinearSAM(), 4)
        expected = F.interpolate(small, (7, 9), mode='bilinear', align_corners=True)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (1, 1, 7, 9))
        self.assertEqual(tuple(aggregate.shape), (1, 7, 9))
        self.assertEqual(tuple(pred.shape), (1, 1, 7, 9))

    def test_actual_train_validation_loop_matches_independent_prediction(self):
        import train as runner

        class Wrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        class Loader:
            dataset = SimpleNamespace(benchmark='joint')
            def __iter__(loader):
                return iter([self.batch])
            def __len__(loader):
                return 1

        model = Prompt()
        expected = model.predict_mask_nshot(self.args, self.batch, NonlinearSAM(), 4)[2][:, 0]
        meter = mock.Mock()
        meter.loss_buf = []
        meter.update.side_effect = lambda inter, union, ids, loss: meter.loss_buf.append(loss)
        meter.compute_iou.return_value = (torch.zeros(()), torch.zeros(()))
        meter.compute_dice.return_value = torch.zeros(())
        with mock.patch.object(runner.utils, 'to_cuda', side_effect=lambda batch: batch), \
             mock.patch.object(runner, 'AverageMeter', return_value=meter), \
             mock.patch.object(runner.Evaluator, 'classify_prediction',
                               return_value=(torch.zeros(2, 2), torch.zeros(2, 2))):
            runner.train(self.args, 1, Wrapper(model), NonlinearSAM(), Loader(), None, None, False)
        torch.testing.assert_close(model.observed_logits, expected, rtol=0, atol=0)

    def test_shared_decoder_keeps_gradients_and_rejects_malformed_batches(self):
        images = self.batch['query_img']
        prompts = torch.randn(8, 5, 4, requires_grad=True)
        shots = decode_support_logits(NonlinearSAM(), images, ['q1', 'q2'], prompts, 4)
        torch.testing.assert_close(mean_support_logits(shots), shots.mean(1), rtol=0, atol=0)
        mean_support_logits(shots).sum().backward()
        self.assertTrue(torch.isfinite(prompts.grad).all())
        self.assertTrue((prompts.grad.flatten(1).abs().sum(1) > 0).all())
        with self.assertRaises(ValueError):
            decode_support_logits(NonlinearSAM(), images, ['q1', 'q2'], prompts[:4], 4)
        for invalid in [0, 1.5, True]:
            with self.assertRaises(ValueError):
                decode_support_logits(NonlinearSAM(), images, ['q1', 'q2'], prompts, invalid)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
