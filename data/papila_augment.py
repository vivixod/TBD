"""RGB fundus preprocessing and training augmentation for PAPILA only.

All label maps use 0=background, 1=disc/rim, 2=cup. Geometry is shared by
image/label; color/exposure changes apply only to the image, before Normalize.
"""
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF


PREPROCESS_VERSION = 'papila_fundus_v2'


def letterbox_fundus(image, mask, size):
    """Fit the complete image into a square without cropping or stretching it."""
    width, height = image.size
    if size < 1 or tuple(mask.shape) != (height, width):
        raise ValueError('PAPILA image/mask sizes must match and output size must be positive')
    ratio = size / max(width, height)
    out_w = max(1, min(size, int(round(width * ratio))))
    out_h = max(1, min(size, int(round(height * ratio))))
    left, top = (size - out_w) // 2, (size - out_h) // 2
    resized_image = image.convert('RGB').resize((out_w, out_h), Image.Resampling.BILINEAR)
    resized_mask = Image.fromarray(mask.numpy().astype(np.uint8)).resize(
        (out_w, out_h), Image.Resampling.NEAREST)
    canvas = Image.new('RGB', (size, size), color=0)
    canvas.paste(resized_image, (left, top))
    labels = np.zeros((size, size), dtype=np.uint8)
    labels[top:top + out_h, left:left + out_w] = np.asarray(resized_mask)
    for value in (1, 2):
        if bool((mask == value).any()) and not np.any(labels == value):
            raise ValueError(f'PAPILA label {value} disappears at input size {size}')
    return canvas, torch.from_numpy(labels)


def papila_augment(image, mask, *, angle=15.0, translate=.03, scale=(.9, 1.1),
                   hflip_prob=.5, color_prob=.8, brightness=.15, contrast=.15,
                   saturation=.10, hue=.01, gamma_prob=.3, gamma=(.9, 1.1),
                   max_trials=8):
    """Mild fundus augmentation with complete foreground retained in the canvas.

    GT is used only to reject training transforms that crop a target or erase a
    label through nearest interpolation. Query/support draws are independent.
    No GT-dependent operation is used in validation preprocessing.
    """
    image_array = np.asarray(image.convert('RGB')).copy()
    labels = mask.numpy().astype(np.uint8, copy=True)
    height, width = labels.shape
    if image_array.shape[:2] != labels.shape:
        raise ValueError('PAPILA augmentation requires aligned image and mask')
    present = [c for c in (1, 2) if np.any(labels == c)]

    if np.random.rand() < hflip_prob:
        image_array = np.ascontiguousarray(image_array[:, ::-1])
        labels = np.ascontiguousarray(labels[:, ::-1])

    ys, xs = np.where(labels > 0)
    corners = None
    if len(xs):
        corners = np.array([[xs.min(), ys.min(), 1], [xs.max(), ys.min(), 1],
                            [xs.max(), ys.max(), 1], [xs.min(), ys.max(), 1]], dtype=float)
    # Applying the same scale on x/y preserves disc/cup geometry.
    for _ in range(max_trials):
        matrix = cv2.getRotationMatrix2D(
            ((width - 1) / 2, (height - 1) / 2),
            float(np.random.uniform(-angle, angle)), float(np.random.uniform(*scale)))
        matrix[0, 2] += np.random.uniform(-translate, translate) * width
        matrix[1, 2] += np.random.uniform(-translate, translate) * height
        if corners is not None:
            moved = corners @ matrix.T
            if (np.any(moved < 1) or np.any(moved[:, 0] > width - 2)
                    or np.any(moved[:, 1] > height - 2)):
                continue
        warped = cv2.warpAffine(labels, matrix, (width, height),
                                flags=cv2.INTER_NEAREST, borderValue=0)
        if any(not np.any(warped == c) for c in present):
            continue
        image_array = cv2.warpAffine(image_array, matrix, (width, height),
                                     flags=cv2.INTER_LINEAR, borderValue=0)
        labels = warped
        break
    # If no safe affine is found, retain the image/mask after the optional flip.
    black = image_array.max(axis=2) == 0
    augmented = Image.fromarray(image_array)
    if np.random.rand() < color_prob:
        changes = [
            (TF.adjust_brightness, np.random.uniform(1 - brightness, 1 + brightness)),
            (TF.adjust_contrast, np.random.uniform(1 - contrast, 1 + contrast)),
            (TF.adjust_saturation, np.random.uniform(1 - saturation, 1 + saturation)),
            (TF.adjust_hue, np.random.uniform(-hue, hue)),
        ]
        for index in np.random.permutation(len(changes)):
            operation, factor = changes[int(index)]
            augmented = operation(augmented, float(factor))
    if np.random.rand() < gamma_prob:
        augmented = TF.adjust_gamma(augmented, float(np.random.uniform(*gamma)))
    # Contrast adjustment must not turn padded/background black into a gray frame.
    output = np.asarray(augmented).copy()
    output[black] = 0
    return Image.fromarray(output), torch.from_numpy(labels)
