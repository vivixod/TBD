"""CPU regression tests; substitute only the expensive pretrained SAM model."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'segment-anything'))
from SAM2pred import SAM_pred, sam_model_registry


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.batch_sizes = []

    def forward(self, image):
        self.batch_sizes.append(image.shape[0])
        values = image.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1)
        return (values * self.weight).expand(-1, 256, 64, 64)


class PromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, points, boxes, protos, masks):
        return protos * self.weight, torch.zeros(protos.shape[0], 256, 64, 64)

    def get_dense_pe(self):
        return torch.zeros(1, 256, 64, 64)


class MaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                dense_prompt_embeddings, multimask_output, attn_bias):
        values = image_embeddings.mean(dim=(1, 2, 3))
        values = values + sparse_prompt_embeddings.mean(dim=(1, 2)) * self.weight
        return values.view(-1, 1, 1, 1).expand(-1, 1, 4, 4), None


class SmallSAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.prompt_encoder = PromptEncoder()
        self.mask_decoder = MaskDecoder()


class CacheAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {'TBD_FEATS': self.tmp.name})
        env.start()
        self.addCleanup(env.stop)
        with mock.patch.dict(sam_model_registry, {'vit_h': lambda _: SmallSAM()}):
            self.model = SAM_pred()
        self.image = torch.ones(1, 3, 8, 8)
        self.protos = torch.zeros(1, 2, 256)
        self.saved = Path(self.tmp.name) / 'eval_unaugmented_v1/query.npy'

    def test_training_bypasses_cache_for_same_name_with_changed_image(self):
        self.model.eval()
        self.model(self.image, ['query'], self.protos)
        saved = self.saved.read_bytes()
        self.model.train()
        logits, _ = self.model(self.image * 3, ['query'], self.protos)
        torch.testing.assert_close(logits, torch.full_like(logits, 3))
        self.assertEqual(self.saved.read_bytes(), saved)
        self.assertEqual(self.model.sam_model.image_encoder.batch_sizes, [1, 1])
        self.assertFalse(self.model.sam_model.image_encoder.training)

    def test_direct_cache_method_also_bypasses_disk_in_training(self):
        self.model.train()
        self.model.get_feat_from_np(self.image, ['query'], self.protos)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_eval_isolates_old_cache_and_writes_an_atomic_cache(self):
        np.save(Path(self.tmp.name) / 'query.npy', np.zeros((1, 256, 64, 64)))
        self.model.eval()
        first, _ = self.model(self.image, ['query'], self.protos)
        second, _ = self.model(self.image, ['query'], self.protos)
        torch.testing.assert_close(first, torch.ones_like(first))
        torch.testing.assert_close(first, second)
        self.assertEqual(self.model.sam_model.image_encoder.batch_sizes, [1])
        self.assertEqual(np.load(self.saved).shape, (1, 256, 64, 64))
        self.assertEqual(list(self.saved.parent.glob('*.tmp*')), [])

    def test_corrupt_or_wrong_shape_cache_is_recomputed(self):
        self.model.eval()
        self.saved.parent.mkdir()
        for payload in ('broken', np.zeros((1, 256, 2, 2)),
                        np.full((1, 256, 64, 64), np.nan)):
            if isinstance(payload, str):
                self.saved.write_text(payload)
            else:
                np.save(self.saved, payload)
            logits, _ = self.model(self.image, ['query'], self.protos)
            torch.testing.assert_close(logits, torch.ones_like(logits))
        self.assertEqual(self.model.sam_model.image_encoder.batch_sizes, [1, 1, 1])

    def test_multishot_encodes_unique_queries_and_preserves_order_and_gradients(self):
        self.model.train()
        image = torch.cat([self.image, self.image * 2])
        protos = torch.ones(8, 2, 256, requires_grad=True)
        logits, _ = self.model(image, ['q1', 'q2'], protos, image_repeats=4)
        values = logits.mean(dim=(1, 2, 3))
        torch.testing.assert_close(values, torch.tensor([2.] * 4 + [3.] * 4))
        self.assertEqual(self.model.sam_model.image_encoder.batch_sizes, [2])
        logits.sum().backward()
        self.assertGreater(protos.grad.abs().sum().item(), 0)
        self.assertIsNotNone(self.model.sam_model.mask_decoder.weight.grad)
        self.assertIsNotNone(self.model.sam_model.prompt_encoder.weight.grad)
        self.assertIsNone(self.model.sam_model.image_encoder.weight.grad)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_disabled_cache_and_failed_cache_write_still_predict(self):
        self.model.eval()
        self.model.use_cache = False
        self.model(self.image, ['query'], self.protos)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])
        self.model.use_cache = True
        with mock.patch('SAM2pred.os.replace', side_effect=OSError('disk unavailable')):
            logits, _ = self.model(self.image, ['query'], self.protos)
        torch.testing.assert_close(logits, torch.ones_like(logits))
        self.assertFalse(self.saved.exists())
        self.assertEqual(list(self.saved.parent.glob('*.tmp*')), [])

    def test_real_sam_decoder_accepts_repeated_embeddings_and_backpropagates(self):
        from segment_anything.modeling import Sam, PromptEncoder, MaskDecoder, TwoWayTransformer
        sam = Sam(
            image_encoder=ImageEncoder(),
            prompt_encoder=PromptEncoder(256, (64, 64), (1024, 1024), 16),
            mask_decoder=MaskDecoder(
                transformer_dim=256,
                transformer=TwoWayTransformer(2, 256, 8, 2048),
                num_multimask_outputs=3,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
        )
        with mock.patch.dict(sam_model_registry, {'vit_h': lambda _: sam}):
            model = SAM_pred()
        model.train()
        protos = torch.randn(4, 52, 256, requires_grad=True)
        logits, masks = model(torch.cat([self.image, self.image * 2]),
                              ['q1', 'q2'], protos, image_repeats=2)
        self.assertEqual(logits.shape, (4, 1, 8, 8))
        self.assertEqual(masks.shape, (4, 8, 8))
        self.assertTrue(torch.isfinite(logits).all())
        logits.square().mean().backward()
        self.assertGreater(protos.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None for p in sam.mask_decoder.parameters()))
        self.assertEqual(sam.image_encoder.batch_sizes, [2])

    def test_invalid_prompt_batch_is_rejected_before_encoding(self):
        with self.assertRaises(ValueError):
            self.model(self.image, ['query'], self.protos, image_repeats=4)
        self.assertEqual(self.model.sam_model.image_encoder.batch_sizes, [])

    def test_precision_auto_uses_native_bf16_and_falls_back_to_fp32(self):
        self.model.train()
        with mock.patch('torch.cuda.get_device_capability', return_value=(8, 0)), \
                mock.patch('torch.cuda.is_bf16_supported', return_value=True):
            self.assertEqual(self.model.resolve_encoder_precision('cuda:0'), 'bf16')
        with mock.patch('torch.cuda.get_device_capability', return_value=(7, 0)):
            self.assertEqual(self.model.resolve_encoder_precision('cuda:0'), 'fp32')
            self.model.encoder_precision = 'bf16'
            with self.assertRaises(RuntimeError):
                self.model.resolve_encoder_precision('cuda:0')

    def test_evaluation_stays_fp32_with_bf16_requested(self):
        self.model.encoder_precision = 'bf16'
        self.model.eval()
        with mock.patch('torch.cuda.get_device_capability', side_effect=AssertionError):
            self.assertEqual(self.model.resolve_encoder_precision('cuda:0'), 'fp32')
        self.model(self.image, ['query'], self.protos)
        self.assertEqual(np.load(self.saved).dtype, np.float32)

    def test_real_sam_image_encoder_autocast_returns_finite_fp32_features(self):
        from segment_anything.modeling.image_encoder import ImageEncoderViT
        self.model.sam_model.image_encoder = ImageEncoderViT(
            img_size=1024, patch_size=128, embed_dim=32, depth=1,
            num_heads=4, mlp_ratio=2, out_chans=256, use_rel_pos=True,
        ).requires_grad_(False)
        self.model.train()
        with mock.patch.object(self.model, 'resolve_encoder_precision', return_value='fp32'):
            reference = self.model.forward_img_encoder(self.image)
        # Exercise the real autocast path on CPU; CUDA throughput is measured separately.
        with mock.patch.object(self.model, 'resolve_encoder_precision', return_value='bf16'):
            features = self.model.forward_img_encoder(self.image)
        self.assertEqual(features.dtype, torch.float32)
        self.assertFalse(features.requires_grad)
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(features, reference, rtol=.1, atol=.08)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
