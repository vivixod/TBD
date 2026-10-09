""" Bbox prediction + encoding + ground-truth extraction.

Provides:
  - PositionEmbeddingRandom:  SAM-native Random Fourier Feature PE
  - CornerTypeEmbedding:      learnable TL/BR corner type embeddings
  - encode_support_bbox:      bbox → RFF-PE + corner type → [B, 2, C]
  - perturb_bbox:             random expand/shrink for robustness
  - giou_loss:                Generalized IoU loss
  - extract_gt_bbox:          get normalised GT bbox from a binary mask
  - BboxHead:                 predict normalised [cx, cy, w, h] from a token
  - BboxEncoder:              encode 4-D bbox into a 256-D embedding
  - bbox_to_sam_format:       normalised bbox → SAM pixel coordinates
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
#  SAM-native Random Fourier Feature Positional Encoding
# ============================================================================

class PositionEmbeddingRandom(nn.Module):
    """Exact replica of SAM's PositionEmbeddingRandom.

    Uses random spatial frequencies instead of sinusoidal functions.
    """

    def __init__(self, num_pos_feats: int = 128, scale: float = 1.0):
        super().__init__()
        if scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            'positional_encoding_gaussian_matrix',
            scale * torch.randn((2, num_pos_feats)),
        )
        self.num_pos_feats = num_pos_feats

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points normalised to [0, 1].

        Args:
            coords:  [..., 2]  values in [0, 1]^2
        Returns:
            [..., 2 * num_pos_feats]  (sin/cos interleaved)
        """
        coords = 2 * coords - 1                                    # → [-1, 1]
        coords = coords @ self.positional_encoding_gaussian_matrix  # [..., num_pos_feats]
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: tuple
    ) -> torch.Tensor:
        """Positionally encode points that are NOT normalised.

        Args:
            coords_input:  [B, N, 2]  pixel coordinates
            image_size:    (H, W)     image dimensions
        Returns:
            [B, N, 2 * num_pos_feats]
        """
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.float())


# ============================================================================
#  Learnable corner-type embeddings  (mimics SAM point_embeddings[2:4])
# ============================================================================

class CornerTypeEmbedding(nn.Module):
    """Learnable embeddings distinguishing top-left vs bottom-right corners."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        # index 0 → top-left,  index 1 → bottom-right
        self.tl_embed = nn.Embedding(1, embed_dim)
        self.br_embed = nn.Embedding(1, embed_dim)

    def forward(self) -> tuple:
        """Returns (tl_embed, br_embed), each [1, embed_dim]."""
        zero = torch.zeros(1, dtype=torch.long,
                           device=self.tl_embed.weight.device)
        return self.tl_embed(zero), self.br_embed(zero)


# ============================================================================
#  Core: support bbox → SAM-style [B, 2, C] prompt embedding
# ============================================================================

def encode_support_bbox(
    bbox_pixel: torch.Tensor,
    pe_layer: PositionEmbeddingRandom,
    corner_emb: CornerTypeEmbedding,
    image_size: tuple = (1024, 1024),
) -> torch.Tensor:
    """Convert support bbox (pixel coords) to SAM-style box prompt embedding.

    Matches SAM's ``_embed_boxes`` exactly:
        1.  boxes + 0.5  (shift to pixel centre)
        2.  RFF-PE for each corner
        3.  add learnable TL / BR corner-type embeddings

    Args:
        bbox_pixel:  [B, 4]  xyxy pixel coords  (x1, y1, x2, y2)
        pe_layer:    PositionEmbeddingRandom instance
        corner_emb:  CornerTypeEmbedding instance
        image_size:  (H, W) for normalisation inside PE

    Returns:
        embedding:   [B, 2, C]  where C = 2 * pe_layer.num_pos_feats
    """
    boxes = bbox_pixel + 0.5                          # shift to pixel centre
    coords = boxes.reshape(-1, 2, 2)                   # [B, 2, 2]
    corner_embedding = pe_layer.forward_with_coords(coords, image_size)  # [B, 2, C]
    tl_type, br_type = corner_emb()                    # [1, C] each
    corner_embedding[:, 0, :] += tl_type
    corner_embedding[:, 1, :] += br_type
    return corner_embedding                              # [B, 2, C]


# ============================================================================
#  Bbox perturbation for training robustness
# ============================================================================

def perturb_bbox(
    bbox_pixel: torch.Tensor,
    img_h: int = 1024,
    img_w: int = 1024,
    max_shift: float = 3.0,
    training: bool = True,
) -> torch.Tensor:
    """Randomly expand or shrink the bbox by up to ±max_shift pixels.

    During training this forces the model to be robust to imperfect support
    bboxes. Each corner is independently perturbed.

    Args:
        bbox_pixel:  [B, 4]  xyxy in pixel coords
        img_h, img_w: image dimensions (for clamping)
        max_shift:   maximum perturbation in pixels (default 3)
        training:    if False, returns bbox unchanged

    Returns:
        perturbed:   [B, 4]  xyxy, clamped to valid range
    """
    if not training or max_shift <= 0:
        return bbox_pixel

    B = bbox_pixel.shape[0]
    noise = (torch.rand(B, 4, device=bbox_pixel.device) * 2 - 1) * max_shift
    perturbed = bbox_pixel.clone()
    perturbed[:, 0] = perturbed[:, 0] - noise[:, 0]    # x1: left
    perturbed[:, 1] = perturbed[:, 1] - noise[:, 1]    # y1: top
    perturbed[:, 2] = perturbed[:, 2] + noise[:, 2]    # x2: right
    perturbed[:, 3] = perturbed[:, 3] + noise[:, 3]    # y2: bottom

    # Clamp and ensure x1 < x2, y1 < y2
    perturbed[:, 0].clamp_(0, img_w - 1)
    perturbed[:, 1].clamp_(0, img_h - 1)
    perturbed[:, 2].clamp_(0, img_w - 1)
    perturbed[:, 3].clamp_(0, img_h - 1)
    too_narrow = perturbed[:, 2] <= perturbed[:, 0]
    too_short = perturbed[:, 3] <= perturbed[:, 1]
    if too_narrow.any():
        perturbed[too_narrow, 2] = perturbed[too_narrow, 0] + 1
    if too_short.any():
        perturbed[too_short, 3] = perturbed[too_short, 1] + 1
    return perturbed


# ============================================================================
#  Ground-truth bbox extraction
# ============================================================================

def extract_gt_bbox(mask: torch.Tensor):
    """Extract the tight bounding box from a binary mask, normalised to [0,1].

    Args:
        mask:  [B, H, W]  or  [B, 1, H, W]  binary mask (values 0/1).

    Returns:
        bbox:  [B, 4]  in (cx, cy, w, h) format, all in [0, 1].
    """
    if mask.dim() == 4:
        mask = mask.squeeze(1)                  # [B, H, W]
    B, H, W = mask.shape
    bbox = torch.zeros(B, 4, device=mask.device, dtype=torch.float32)

    for i in range(B):
        ys, xs = torch.where(mask[i] > 0.5)
        if ys.numel() == 0:
            bbox[i] = torch.tensor([0.5, 0.5, 1.0, 1.0], device=mask.device)
        else:
            y1, y2 = ys.min().float(), ys.max().float()
            x1, x2 = xs.min().float(), xs.max().float()
            # normalise to [0, 1]
            cx = ((x1 + x2) / 2.0) / W
            cy = ((y1 + y2) / 2.0) / H
            w  = (x2 - x1 + 1.0) / W
            h  = (y2 - y1 + 1.0) / H
            bbox[i] = torch.stack([cx, cy, w, h])

    return bbox


# ============================================================================
#  GIoU loss
# ============================================================================

def giou_loss(pred_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
    """Generalized IoU loss between predicted and ground-truth bboxes.

    Both inputs in (cx, cy, w, h) normalised format.  Loss = 1 − GIoU.

    Args:
        pred_bbox:  [B, 4]  (cx, cy, w, h)
        gt_bbox:    [B, 4]  (cx, cy, w, h)
    Returns:
        scalar loss
    """
    def to_xyxy(b):
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

    pred_xy = to_xyxy(pred_bbox)
    gt_xy   = to_xyxy(gt_bbox)

    # intersection
    ix1 = torch.max(pred_xy[:, 0], gt_xy[:, 0])
    iy1 = torch.max(pred_xy[:, 1], gt_xy[:, 1])
    ix2 = torch.min(pred_xy[:, 2], gt_xy[:, 2])
    iy2 = torch.min(pred_xy[:, 3], gt_xy[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # union
    area_pred = (pred_xy[:, 2] - pred_xy[:, 0]).clamp(min=0) * \
                (pred_xy[:, 3] - pred_xy[:, 1]).clamp(min=0)
    area_gt   = (gt_xy[:, 2] - gt_xy[:, 0]).clamp(min=0) * \
                (gt_xy[:, 3] - gt_xy[:, 1]).clamp(min=0)
    union = area_pred + area_gt - inter
    iou = inter / (union + 1e-6)

    # smallest enclosing box
    cx1 = torch.min(pred_xy[:, 0], gt_xy[:, 0])
    cy1 = torch.min(pred_xy[:, 1], gt_xy[:, 1])
    cx2 = torch.max(pred_xy[:, 2], gt_xy[:, 2])
    cy2 = torch.max(pred_xy[:, 3], gt_xy[:, 3])
    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)

    giou = iou - (c_area - union) / (c_area + 1e-6)
    return (1 - giou).mean()


# ============================================================================
#  Bbox predictor & encoder
# ============================================================================

class BboxHead(nn.Module):
    """Predict normalised bbox [cx, cy, w, h] from a semantic token."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),                     # output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, hidden_dim] → bbox: [B, 4]"""
        return self.fc(x)


class BboxEncoder(nn.Module):
    """Encode a 4-D normalised bbox into a 256-D embedding."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        """bbox: [B, 4] → emb: [B, hidden_dim]"""
        return self.fc(bbox)


# ============================================================================
#  Utility: normalised bbox → SAM pixel coordinates
# ============================================================================

def bbox_to_sam_format(bbox_norm: torch.Tensor, img_h: int = 1024, img_w: int = 1024):
    """Convert normalised (cx, cy, w, h) to SAM xyxy pixel coords.

    Args:
        bbox_norm: [B, 4]  in [0, 1]
    Returns:
        boxes:     [B, 1, 4]  xyxy in pixel space
    """
    cx, cy, w, h = bbox_norm[:, 0], bbox_norm[:, 1], bbox_norm[:, 2], bbox_norm[:, 3]
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    boxes = torch.stack([x1, y1, x2, y2], dim=1).clamp(0, max(img_w - 1, img_h - 1))
    return boxes.unsqueeze(1)    # [B, 1, 4]
