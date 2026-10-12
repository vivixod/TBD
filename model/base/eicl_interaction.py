""" EICSeg-style Explicit In-Context Learning interaction module.

Replaces prototype-based query-support interaction with patch-level
cross-attention, as described in:
  EICSeg: Universal Medical Image Segmentation via Explicit In-Context Learning
  (IEEE TMI, 2026)

Core formula:
  pseudo   = softmax(Q_f @ K_f^T / τ) @ M        →  粗略定位图
  attended = softmax(Q_f @ K_f^T / τ) @ K_f      →  加权 support 特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EICLInteraction(nn.Module):
    """Patch-level explicit retrieval between query and support features.

    For each query patch, computes cosine similarity with all support patches,
    then uses the resulting attention weights to aggregate the support mask
    (→ pseudo_mask) and support features (→ attended_supp).

        A = softmax(Q K^T / τ)     # [B, HW, HW] query-support 对应关系
        pseudo_mask  = A @ M_s     # [B, 1, H, W] min-max 归一化
        attended_supp = A @ K      # [B, C, H, W] 加权 support 特征

    Args:
        hidden_dim: feature channel dimension (default 256)
        temperature: softmax temperature τ (lower = sharper attention)
    """

    def __init__(self, hidden_dim: int = 256, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.hidden_dim = hidden_dim

    def forward(self, query_feat, supp_feat, supp_mask):
        """Run patch-level in-context retrieval.

        Args:
            query_feat:      [B, C, H, W]  query feature map
            supp_feat:       [B, C, H, W]  support feature map
            supp_mask:       [B, 1, H, W]  support binary mask  (values 0/1)

        Returns:
            pseudo_mask:     [B, 1, H, W]  coarse query mask, min-max normalised
            attended_supp:   [B, C, H, W]  query-attended support features
        """
        B, C, H, W = query_feat.shape

        # ---- 1. flatten spatial dims -------------------------------------------
        Q = query_feat.view(B, C, -1)          # [B, C, HW]
        K = supp_feat.view(B, C, -1)           # [B, C, HW]
        V_mask = supp_mask.view(B, 1, -1)      # [B, 1, HW]

        # ---- 2. L2-normalise → cosine similarity --------------------------------
        Q = F.normalize(Q, dim=1)              # [B, C, HW]
        K = F.normalize(K, dim=1)              # [B, C, HW]

        # ---- 3. attention: A[b, i, j] = similarity(query_patch_i, supp_patch_j) -
        A = torch.bmm(Q.transpose(1, 2), K)    # [B, HW, HW]
        A = F.softmax(A / self.temperature, dim=-1)
        self.last_A = A.detach()               # 调试用: 供外部可视化/分析

        # ---- 4. pseudo mask (support mask propagated to query) -----------------
        V_mask_t = V_mask.transpose(1, 2)      # [B, HW, 1]
        pseudo = torch.bmm(A, V_mask_t)         # [B, HW, 1]
        pseudo = pseudo.view(B, 1, H, W)

        # ---- 5. attended support features --------------------------------------
        attended = torch.bmm(A, K.transpose(1, 2))   # [B, HW, C]
        attended = attended.transpose(1, 2).contiguous().view(B, C, H, W)

        # ---- 6. min-max normalise pseudo_mask per sample -----------------------
        pseudo_flat = pseudo.view(B, -1)
        p_min = pseudo_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        p_max = pseudo_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        pseudo = (pseudo - p_min) / (p_max - p_min + 1e-5)

        return pseudo, attended
