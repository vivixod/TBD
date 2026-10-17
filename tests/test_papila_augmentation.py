"""Fundus geometry/color checks with known synthetic targets; no GPU needed."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.papila import DatasetPAPILA
from data.papila_augment import letterbox_fundus, papila_augment


def example(height=96, width=160):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (width // 2, height // 2), 18, 1, -1)
    cv2.circle(mask, (width // 2, height // 2), 8, 2, -1)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[mask > 0] = [180, 110, 65]
    return Image.fromarray(image), torch.from_numpy(mask)


class FundusAugmentationTests(unittest.TestCase):
    def test_letterbox_preserves_circle_shape_rgb_and_padding(self):
        image, mask = example()
        # Exact half-size scaling isolates aspect preservation from raster rounding.
        prepared, target = letterbox_fundus(image, mask, 80)
        self.assertEqual(prepared.size, (80, 80))
        self.assertEqual(prepared.mode, 'RGB')
        self.assertEqual(target.shape, (80, 80))
        ys, xs = np.where(target.numpy() > 0)
        self.assertLessEqual(abs((xs.max() - xs.min()) - (ys.max() - ys.min())), 1)
        pixels = np.asarray(prepared)
        self.assertFalse(pixels[:16].any())
        self.assertFalse(pixels[-16:].any())
        self.assertEqual(set(target.unique().tolist()), {0, 1, 2})

    def test_letterbox_image_geometry_does_not_depend_on_gt_location(self):
        image, mask = example()
        first, _ = letterbox_fundus(image, mask, 128)
        moved = torch.roll(mask, shifts=12, dims=1)
        second, _ = letterbox_fundus(image, moved, 128)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_horizontal_flip_is_shared_exactly_by_image_and_label(self):
        image, mask = example()
        output, labels = papila_augment(
            image, mask, angle=0, translate=0, scale=(1, 1),
            hflip_prob=1, color_prob=0, gamma_prob=0)
        np.testing.assert_array_equal(np.asarray(output), np.asarray(image)[:, ::-1])
        np.testing.assert_array_equal(labels.numpy(), mask.numpy()[:, ::-1])

    def test_random_geometry_aligns_image_and_keeps_nested_foregrounds(self):
        image, mask = example()
        image, mask = letterbox_fundus(image, mask, 128)
        changed = 0
        for seed in range(30):
            np.random.seed(seed)
            output, labels = papila_augment(image, mask, color_prob=0, gamma_prob=0)
            # Threshold the red channel at half its foreground intensity.
            predicted = np.asarray(output)[:, :, 0] >= 90
            expected = labels.numpy() > 0
            intersection = np.logical_and(predicted, expected).sum()
            union = np.logical_or(predicted, expected).sum()
            self.assertGreater(intersection / union, .94)
            disc, _ = DatasetPAPILA.extract_ignore_idx(None, labels.float(), 0)
            cup, _ = DatasetPAPILA.extract_ignore_idx(None, labels.float(), 1)
            self.assertTrue(torch.all(cup <= disc))
            self.assertGreater(cup.sum(), 0)
            self.assertEqual(set(labels.unique().tolist()), {0, 1, 2})
            changed += not torch.equal(labels, mask)
        self.assertGreater(changed, 20)

    def test_unsafe_geometry_falls_back_without_cropping_foreground(self):
        image, mask = example()
        outside = np.array([[1., 0., 500.], [0., 1., 500.]])
        with mock.patch('data.papila_augment.cv2.getRotationMatrix2D', return_value=outside):
            output, labels = papila_augment(
                image, mask, hflip_prob=0, color_prob=0, gamma_prob=0, max_trials=2)
        np.testing.assert_array_equal(np.asarray(output), np.asarray(image))
        torch.testing.assert_close(labels, mask)

    def test_photometric_changes_rgb_without_changing_labels_or_black_padding(self):
        image, mask = example()
        image, mask = letterbox_fundus(image, mask, 128)
        source_image, source_mask = np.asarray(image).copy(), mask.clone()
        np.random.seed(321)
        output, labels = papila_augment(
            image, mask, angle=0, translate=0, scale=(1, 1), hflip_prob=0,
            color_prob=1, gamma_prob=1)
        output = np.asarray(output)
        self.assertFalse(np.array_equal(source_image, output))
        self.assertTrue((output[source_image.max(2) == 0] == 0).all())
        self.assertFalse(np.array_equal(output[:, :, 0], output[:, :, 1]))
        self.assertEqual(output.dtype, np.uint8)
        torch.testing.assert_close(labels, source_mask, rtol=0, atol=0)
        np.testing.assert_array_equal(np.asarray(image), source_image)
        torch.testing.assert_close(mask, source_mask)

    def test_augmentation_repeats_with_worker_numpy_seed(self):
        image, mask = example()
        np.random.seed(321)
        image_a, mask_a = papila_augment(image, mask)
        np.random.seed(321)
        image_b, mask_b = papila_augment(image, mask)
        np.testing.assert_array_equal(np.asarray(image_a), np.asarray(image_b))
        torch.testing.assert_close(mask_a, mask_b)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
