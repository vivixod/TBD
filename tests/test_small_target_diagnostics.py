"""CPU checks for diagnostic metrics, crop controls, and shared-query inference."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnose_small_targets import (binary_metrics, localization_metrics, context_box,
                                    grid_box, restore_crop, restrict_to_box, log_config,
                                    predict_view)
from model.base.eicl_interaction import EICLInteraction


class DiagnosticTests(unittest.TestCase):
    def test_empty_prediction_and_exact_overlap(self):
        target = np.zeros((8, 8), bool);target[2:4, 3:5] = True
        empty = binary_metrics(np.zeros_like(target), target)
        self.assertEqual((empty['dice'], empty['recall'], empty['precision']), (0, 0, 0))
        perfect = binary_metrics(target, target)
        self.assertEqual((perfect['dice'], perfect['recall'], perfect['precision']), (100, 100, 100))

    def test_localization_keeps_single_pixel_gt_and_rejects_constant_maps(self):
        target = np.zeros((512, 512), bool);target[1, 1] = True
        heat = np.zeros((64, 64), np.float32);heat[0, 0] = 1
        metrics = localization_metrics(heat, target)
        self.assertTrue(metrics['peak_hit'])
        self.assertEqual(metrics['top1pct_gt_grid_coverage'], 100)
        constant = localization_metrics(np.zeros_like(heat), target)
        self.assertTrue(constant['constant_heatmap'])
        self.assertIsNone(constant['peak_hit'])

    def test_native_crop_contains_border_gt_and_restores_to_full_grid(self):
        target = np.zeros((512, 512), bool);target[-3:, -5:] = True
        box = context_box(target, min_size=64, scale=2)
        l, t, r, b = box
        self.assertEqual(int(target[t:b, l:r].sum()), int(target.sum()))
        self.assertEqual((r-l, b-t), (64, 64))
        restored = restore_crop(np.ones((512, 512), np.float32), box)
        expected = np.zeros_like(target);expected[t:b, l:r] = True
        np.testing.assert_array_equal(restored > 0, expected)
        np.testing.assert_array_equal(restrict_to_box(np.ones_like(target), box), expected)
        self.assertEqual(grid_box((128, 256, 256, 384), (1024, 1024)), (64, 128, 128, 192))

    def test_log_configuration_matches_each_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'best_model.pt'
            log = checkpoint.parent / 'log.txt'
            for text, expected in [('', 'nearest'),
                    ('[Support mask] downsample=adaptive_max_pool2d', 'max'),
                    ('| support_mask_pooling: adam-only\n| bbox_support_attn_mask: 1\n', 'adam-only')]:
                log.write_text(text)
                config, _ = log_config(checkpoint, {})
                self.assertEqual(config['support_mask_pooling'], expected)

    def test_restore_uses_fractional_native_coordinates(self):
        native_size = (528, 528)
        box = (177, 186, 243, 252)
        # Horizontal ramp: transformed values should follow the exact pixel-centre map.
        ramp = np.tile(np.arange(512, dtype=np.float32), (512, 1))
        result = restore_crop(ramp, grid_box(box, native_size),
                              native_box=box, native_size=native_size)
        x = (200.5) * 528 / 512
        expected = (x-box[0]) / (box[2]-box[0]) * 512 - .5
        self.assertAlmostEqual(float(result[200,200]), expected, places=3)
        self.assertEqual(result[0,0], -30)

    def test_predictions_share_image_encoding_and_use_mean_logits(self):
        class Prompt(nn.Module):
            def __init__(self):
                super().__init__();self.eicl_interaction = EICLInteraction(4)
            def forward(self, condition, query, supports, masks, training, dataset_names,
                        return_quality=False):
                b = supports.shape[0]
                features = torch.arange(b*16, dtype=torch.float32).reshape(b,4,2,2) + 1
                reduced = F.adaptive_max_pool2d(masks[:, None], (2,2))
                pseudo, _ = self.eicl_interaction(features, features.flip(-1), reduced)
                values = torch.tensor([-3., 1.])[:b, None, None]
                outputs = values.expand(b,50,256), values.expand(b,2,256), pseudo
                return outputs + (torch.tensor([9., 1.]).log(),) if return_quality else outputs
        class SAM:
            def __init__(self):self.calls=0
            def forward_img_encoder(self, query):
                self.calls+=1;return torch.ones(1,256,64,64)
            def forward_prompt_encoder(self, protos):return protos,None
            def forward_mask_decoder(self, features, sparse, dense, ori_size):
                return sparse.mean((1,2))[:,None,None,None].expand(-1,1,*ori_size),None
        masks = torch.zeros(2,512,512);masks[:,1,1]=1
        view = dict(query=torch.ones(1,3,512,512), supports=torch.ones(2,3,512,512),
                    support_masks=masks)
        with tempfile.TemporaryDirectory() as tmp:
            model, sam = Prompt(), SAM();path=Path(tmp)/'feature.npy'
            result=predict_view(model,sam,view,{'dataset':'adam'},path,torch.device('cpu'))
            self.assertEqual(sam.calls,1)
            np.testing.assert_array_equal(result['mean_logits'],np.full((512,512),-1.))
            self.assertEqual(result['raw_pseudo'].shape,(2,2,2))
            np.testing.assert_array_equal(result['fused_logits'], result['mean_logits'])
            self.assertIsNone(model.eicl_interaction.last_A)
            predict_view(model,sam,view,{'dataset':'adam'},path,torch.device('cpu'))
            self.assertEqual(sam.calls,1)
            predict_view(model,sam,view,{'dataset':'adam'},Path(tmp)/'crop.npy',torch.device('cpu'))
            self.assertEqual(sam.calls,2)
            self.assertFalse(model.eicl_interaction._forward_hooks)
            model.support_fusion = 'quality'
            model.support_quality_head = nn.Module()
            model.support_quality_head.register_buffer('mix', torch.tensor(.5))
            weighted = predict_view(model,sam,view,{'dataset':'adam'},path,torch.device('cpu'))
            np.testing.assert_allclose(weighted['fusion_weights'], [.7, .3], atol=1e-7)
            np.testing.assert_allclose(weighted['fused_logits'], -1.8, atol=1e-6)
            np.testing.assert_array_equal(weighted['mean_logits'], np.full((512,512),-1.))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
