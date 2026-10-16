"""Per-support decoding, supervised reliability scoring, and logit fusion."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class SupportQualityHead(nn.Module):
    """A shared scalar scorer; query labels are never inputs to this module."""
    def __init__(self, hidden_dim=256, mlp_dim=64):
        super().__init__()
        self.icl_norm = nn.LayerNorm(hidden_dim)
        self.bbox_norm = nn.LayerNorm(2 * hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(3 * hidden_dim + 2, mlp_dim),
                                 nn.SiLU(), nn.Linear(mlp_dim, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer('mix', torch.tensor(0.0))

    def forward(self, icl_tokens, bbox_tokens, support_mask, grid_mask):
        # Supervision of quality must not reshape the original prompt features.
        icl = self.icl_norm(icl_tokens.detach().float().mean(dim=1))
        bbox = self.bbox_norm(bbox_tokens.detach().float().flatten(1))
        mask = support_mask.detach().float().flatten(1)
        grid = grid_mask.detach().float().flatten(1)
        area = mask.sum(1).log1p() / math.log1p(mask.shape[1])
        cells = grid.sum(1).log1p() / math.log1p(grid.shape[1])
        features = torch.cat([icl, bbox, area[:, None], cells[:, None]], dim=1)
        return self.mlp(features).squeeze(-1)


def fusion_mix_for_epoch(epoch, warmup_epochs=2, ramp_epochs=3):
    if epoch < 0 or warmup_epochs < 0 or ramp_epochs < 0:
        raise ValueError('Fusion epoch and schedule lengths must be nonnegative')
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return 1.0
    return min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)


def fuse_support_logits(shot_logits, scores=None, mix=1.0):
    """Return [B,H,W] fused logits and optional [B,S] learned weights."""
    if scores is None:
        return mean_support_logits(shot_logits), None
    if shot_logits.ndim != 4 or scores.shape != shot_logits.shape[:2]:
        raise ValueError('Quality scores must match the [B,S] logit dimensions')
    weights = scores.float().softmax(dim=1)
    weights = (1 - mix) / scores.shape[1] + mix * weights
    return (shot_logits * weights[:, :, None, None]).sum(dim=1), weights


@torch.no_grad()
def support_quality_targets(shot_logits, query_gt, temperature=0.2):
    """Detached per-query Dice teacher; empty-query labels receive no loss."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Fusion target temperature must be finite and positive')
    if query_gt.ndim == 4 and query_gt.shape[1] == 1:
        query_gt = query_gt[:, 0]
    if (shot_logits.ndim != 4 or query_gt.ndim != 3
            or query_gt.shape[0] != shot_logits.shape[0]):
        raise ValueError('Expected logits [B,S,H,W] and query GT [B,H,W]')
    if shot_logits.shape[-2:] != query_gt.shape[-2:]:
        shot_logits = F.interpolate(shot_logits, size=query_gt.shape[-2:],
                                    mode='bilinear', align_corners=True)
    pred = shot_logits.detach().float().sigmoid() > 0.5
    target = query_gt[:, None] > 0
    intersection = (pred & target).sum((-2, -1)).float()
    denominator = pred.sum((-2, -1)) + target.sum((-2, -1))
    dice = 2 * intersection / denominator.clamp_min(1)
    valid = target.flatten(1).any(dim=1)
    return (dice / temperature).softmax(dim=1), dice, valid


def support_quality_loss(scores, shot_logits, query_gt, temperature=0.2):
    targets, dice, valid = support_quality_targets(shot_logits, query_gt, temperature)
    if scores.shape != targets.shape:
        raise ValueError('Quality scores must match the [B,S] teacher targets')
    per_query = F.kl_div(scores.float().log_softmax(1), targets, reduction='none').sum(1)
    # Average over all queries, with invalid ones contributing zero. This also
    # gives the correct global-query average under DDP's gradient averaging.
    return (per_query * valid).mean(), dice, valid


def checkpoint_fusion_mode(checkpoint, requested=None):
    """Infer the architecture and reject accidental mean/quality mismatch."""
    state = checkpoint.get('model', checkpoint)
    has_head = any(k.removeprefix('module.').startswith('support_quality_head.') for k in state)
    inferred = 'quality' if has_head else 'mean'
    saved = checkpoint.get('config', {}).get('support_fusion', inferred)
    if saved != inferred:
        raise ValueError('Checkpoint fusion configuration does not match its parameters')
    if requested is not None and requested != inferred:
        raise ValueError(f'Checkpoint uses support-fusion={inferred}, requested {requested}')
    return inferred


def decode_support_logits(sam_model, query_images, query_names, prompts, shot):
    """Decode query-major prompts and return logits shaped [B, S, H, W].

    Prompt order must match support.flatten(0, 1) and
    query.repeat_interleave(shot, dim=0). SAM reuses each query embedding.
    """
    if not isinstance(shot, int) or isinstance(shot, bool) or shot < 1:
        raise ValueError('shot must be a positive integer')
    batch_size = query_images.shape[0]
    if prompts.shape[0] != batch_size * shot:
        raise ValueError('Expected one prompt set per query/support pair')
    logits, _ = sam_model(query_images, query_names, prompts, image_repeats=shot)
    if logits.ndim != 4 or logits.shape[:2] != (batch_size * shot, 1):
        raise ValueError('SAM must return one logit channel per query/support pair')
    return logits.reshape(batch_size, shot, *logits.shape[-2:])


def mean_support_logits(shot_logits):
    """Average decoded logits, preserving the existing train/valid protocol."""
    if shot_logits.ndim != 4 or shot_logits.shape[1] < 1:
        raise ValueError('Expected nonempty support logits shaped [B, S, H, W]')
    return shot_logits.mean(dim=1)
