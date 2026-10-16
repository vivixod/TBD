"""Dataset-specific support pooling and train/validation episode routing."""
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
from model.base.mask_pooling import resize_support_mask, repeat_dataset_names
from model.VRP_encoder import VRP_encoder
from model.base import resnet


class DatasetMaskPoolingTests(unittest.TestCase):
    def test_policies_and_multishot_order(self):
        mask = torch.zeros(4, 1, 32, 32)
        mask[:, :, 1, 1] = 1  # nearest misses this pixel in every sample.
        nearest = F.interpolate(mask, size=(4, 4), mode='nearest')
        maximum = F.adaptive_max_pool2d(mask, (4, 4))
        torch.testing.assert_close(resize_support_mask(mask, (4, 4), 'max'), maximum)
        torch.testing.assert_close(resize_support_mask(mask, (4, 4), 'nearest'), nearest)
        names = repeat_dataset_names(['adam', 'btcv2'], 2, repeats=2)
        self.assertEqual(names, ['adam', 'adam', 'btcv2', 'btcv2'])
        actual = resize_support_mask(mask, (4, 4), 'adam-only', names)
        torch.testing.assert_close(actual[:2], maximum[:2], rtol=0, atol=0)
        torch.testing.assert_close(actual[2:], nearest[2:], rtol=0, atol=0)
        for name in ['adam', 'btcv', 'btcv2', 'amos', 'brats']:
            expected = maximum if name == 'adam' else nearest
            torch.testing.assert_close(resize_support_mask(mask, (4, 4), 'adam-only', name),
                                       expected, rtol=0, atol=0)
        for names in (None, 'joint', ['adam']):
            with self.assertRaises(ValueError):
                resize_support_mask(mask, (4, 4), 'adam-only', names)
        with self.assertRaises(ValueError):
            resize_support_mask(mask, (4, 4), 'unknown')

    def test_real_encoder_routes_same_mask_to_icl_and_bbox_for_mixed_datasets(self):
        torch.manual_seed(321)
        backbone = resnet.resnet50(pretrained=False)
        with mock.patch('model.VRP_encoder.models.resnet50', return_value=backbone):
            model = VRP_encoder(SimpleNamespace(num_query=50, benchmark='joint',
                                                support_mask_pooling='adam-only'),
                                'resnet50', False).eval()
        captured = {}
        def capture_icl(module, args):
            captured['icl'] = args[2].clone()
        def capture_bbox(module, args, kwargs):
            captured['bbox'] = kwargs['supp_mask'].clone()
        handles = [model.eicl_interaction.register_forward_pre_hook(capture_icl),
                   model.bbox_transfer.register_forward_pre_hook(capture_bbox, with_kwargs=True)]
        mask = torch.zeros(2, 32, 32)
        mask[:, 1, 1] = 1
        try:
            with torch.no_grad():
                model('mask', torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32),
                      mask, False, dataset_names=['adam', 'amos'])
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(captured['icl'].sum((1, 2, 3)), torch.tensor([1., 0.]))
        torch.testing.assert_close(captured['icl'], captured['bbox'], rtol=0, atol=0)

    def test_train_loop_and_per_dataset_validation_repeat_names_in_support_order(self):
        import train as runner

        class Prompt(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))
                self.names = []
                self.use_original_imgsize = False
            def train_mode(self):
                self.train()
            def forward(self, condition, query, support, mask, training, dataset_names=None):
                self.names.append(dataset_names)
                value = self.weight.expand(query.shape[0], 2, 4)
                return value, value, None
            def compute_bbox_token_loss(self, boxes, mask):
                return boxes.square().mean(), torch.ones(())
            def compute_objective(self, logits, mask):
                return logits.square().mean()

        class Wrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        class Decoder(nn.Module):
            def forward(self, image, names, protos, image_repeats):
                logits = protos.mean((1, 2))[:, None, None, None]
                return logits.expand(-1, 1, *image.shape[-2:]), None

        class Loader:
            def __init__(self, benchmark, names):
                self.dataset = SimpleNamespace(benchmark=benchmark)
                self.batch = {'query_img': torch.ones(2, 3, 8, 8),
                              'query_mask': torch.ones(2, 8, 8),
                              'query_name': ['q1', 'q2'],
                              'support_imgs': torch.ones(2, 4, 3, 8, 8),
                              'support_masks': torch.ones(2, 4, 8, 8),
                              'class_id': torch.zeros(2, dtype=torch.long)}
                if names is not None:
                    self.batch['dataset_name'] = names
            def __iter__(self):
                return iter([self.batch])
            def __len__(self):
                return 1

        args = SimpleNamespace(seed=321, condition='mask', bbox_loss_weight=.5)
        for training, benchmark, names, expected in [
                (True, 'joint', ['adam', 'btcv2'], ['adam'] * 4 + ['btcv2'] * 4),
                (False, 'adam', None, ['adam'] * 8),
                (False, 'amos', None, ['amos'] * 8)]:
            with self.subTest(training=training, benchmark=benchmark):
                prompt = Prompt()
                model = Wrapper(prompt)
                meter = mock.Mock()
                meter.loss_buf = []
                meter.update.side_effect = lambda inter, union, ids, loss: meter.loss_buf.append(loss)
                meter.compute_iou.return_value = (torch.zeros(()), torch.zeros(()))
                meter.compute_dice.return_value = torch.zeros(())
                with mock.patch.object(runner.utils, 'to_cuda', side_effect=lambda batch: batch), \
                     mock.patch.object(runner, 'AverageMeter', return_value=meter), \
                     mock.patch.object(runner.Evaluator, 'classify_prediction',
                                       return_value=(torch.zeros(2, 2), torch.zeros(2, 2))):
                    runner.train(args, 1, model, Decoder(), Loader(benchmark, names),
                                 torch.optim.SGD(model.parameters(), lr=.01), mock.Mock(), training)
                self.assertEqual(prompt.names, [expected])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
