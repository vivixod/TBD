""" EICSeg-style Explicit In-Context Learning interaction module.

Replaces prototype-based query-support interaction with patch-level
cross-attention, as described in:
  EICSeg: Universal Medical Image Segmentation via Explicit In-Context Learning
  (IEEE TMI, 2026)

Core formula:  Ŷ = softmax(Q_f @ P_f^T / τ) @ M

Extended with dual-path (foreground / background) value retrieval:
  the same correspondence A reads two disentangled values,
    R_fg = A @ (M_s ⊙ K)   (positive target evidence)
    R_bg = A @ ((1-M_s) ⊙ K) (negative anatomical context)
  which exactly decompose the original mixed retrieval A @ K.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EICLInteraction(nn.Module):
    """Patch-level explicit retrieval between query and support features.

    Dual-path (foreground / background decoupled) in-context retrieval:
    the correspondence A = softmax(Q K^T / τ) is shared by both paths,

        coarse     = A @ M_s                  → 粗略定位图（min-max 归一化）
        R_fg       = A @ (M_s ⊙ K)           → 前景语义证据（positive）
        R_bg       = A @ ((1-M_s) ⊙ K)       → 背景解剖上下文证据（negative）

    R_fg + R_bg = A @ K，两条流恰好分解原始混合检索（无信息损失，只是解耦）。

    Args:
        hidden_dim: feature channel dimension (default 256)
        temperature: softmax temperature τ (lower = sharper attention)
    """

    def __init__(self, hidden_dim: int = 256, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.hidden_dim = hidden_dim

    def forward(self, query_feat, supp_feat, supp_mask):
        """Run dual-path (foreground / background) in-context retrieval.

        The correspondence A is computed over the FULL support (fg + bg patches),
        but the retrieved value is disentangled into two streams:

            R_fg = A @ (M_s ⊙ K)   → positive target semantic evidence
            R_bg = A @ ((1-M_s) ⊙ K) → negative anatomical context evidence

        Note: R_fg + R_bg = A @ K, so the two streams exactly decompose the
        original mixed retrieval (no information loss, only disentanglement).

        Args:
            query_feat:      [B, C, H, W]  query feature map
            supp_feat:       [B, C, H, W]  support feature map
            supp_mask:       [B, 1, H, W]  support binary mask  (values 0/1)

        Returns:
            coarse_mask:     [B, 1, H, W]  coarse query mask, min-max normalised
            attended_fg:     [B, C, H, W]  foreground evidence (positive)
            attended_bg:     [B, C, H, W]  background evidence (negative)
        """
        B, C, H, W = query_feat.shape

        # ---- 1. flatten spatial dims -------------------------------------------
        Q = query_feat.view(B, C, -1)          # [B, C, HW]
        K = supp_feat.view(B, C, -1)           # [B, C, HW]
        V_mask = supp_mask.view(B, 1, -1)      # [B, 1, HW]

        # ---- 2. L2-normalise → cosine similarity --------------------------------
        Q_norm = F.normalize(Q, dim=1)              # [B, C, HW]
        K_norm = F.normalize(K, dim=1)              # [B, C, HW]

        # ---- 3. attention: A[b, i, j] = similarity(query_patch_i, supp_patch_j) -
        A = torch.bmm(Q_norm.transpose(1, 2), K_norm)    # [B, HW, HW]
        A = F.softmax(A / self.temperature, dim=-1)

        # ---- 4. coarse mask (support mask propagated to query) -----------------
        V_mask_t = V_mask.transpose(1, 2)      # [B, HW, 1]
        coarse = torch.bmm(A, V_mask_t)         # [B, HW, 1]
        coarse = coarse.view(B, 1, H, W)

        # ---- 5. dual-path retrieval: foreground / background evidence ----------
        # 正路 R_fg：只聚合 support 前景特征（M_s ⊙ K），
        #           幅度天然携带 foreground confidence C_q（= AM_s）
        attended_fg = torch.bmm(A, (K * V_mask).transpose(1, 2))       # [B, HW, C]
        attended_fg = attended_fg.transpose(1, 2).contiguous().view(B, C, H, W)

        # 负路 R_bg：只聚合 support 背景特征（(1-M_s) ⊙ K）
        attended_bg = torch.bmm(A, (K * (1 - V_mask)).transpose(1, 2))  # [B, HW, C]
        attended_bg = attended_bg.transpose(1, 2).contiguous().view(B, C, H, W)

        # ---- 6. min-max normalise per sample -----------------------------------
        coarse_flat = coarse.view(B, -1)
        c_min = coarse_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        c_max = coarse_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        coarse = (coarse - c_min) / (c_max - c_min + 1e-5)

        return coarse, attended_fg, attended_bg
