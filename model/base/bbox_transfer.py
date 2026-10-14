""" Bbox Token Module: learnable query bbox tokens → support → query → bbox token.

Pipeline:
    1. Learnable query bbox tokens [2, C] (TL/BR) expand to [2, B, C]
    2. Cross-attend to support foreground features
       → absorb "what is the target object"   Z_s [2, B, C]
    3. Cross-attend to query features (with 2D positional encoding)
       → locate the target in query            Z_q [2, B, C]
    4. Output query bbox token [B, 2, C], directly usable as SAM sparse prompt

The final token is supervised in token space (MSE against the SAM-style
encoding of the GT query bbox), no explicit 4-coordinate regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_decoder import PositionEmbeddingSine


class BboxTransferModule(nn.Module):
    """Learnable query-bbox tokens → support → query → final bbox token [B,2,C].

    Args:
        hidden_dim:      feature dimension (default 256)
        nhead:           number of attention heads (default 8)
        num_layers:      number of cross-attention layers per stage (default 1)
        dropout:         attention dropout (default 0.1)
        pre_norm:        use pre-LayerNorm (default True)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ---- learnable query bbox tokens --------------------------------------
        # token 0 → 左上角, token 1 → 右下角
        self.query_bbox_tokens = nn.Parameter(
            torch.randn(2, hidden_dim) * 0.02
        )

        # ---- 2D positional encoding for query features ------------------------
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # ---- Stage 1: bbox tokens attend to support features ------------------
        self.supp_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout,
                                  batch_first=False)
            for _ in range(num_layers)
        ])
        self.supp_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.supp_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(inplace=False),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.supp_ffn_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.supp_dropout = nn.Dropout(dropout)

        # ---- Stage 2: enriched tokens attend to query features ---------------
        self.qry_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout,
                                  batch_first=False)
            for _ in range(num_layers)
        ])
        self.qry_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.qry_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(inplace=False),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.qry_ffn_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.qry_dropout = nn.Dropout(dropout)

    def _cross_attn_block(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        cross_attn: nn.ModuleList,
        norm: nn.ModuleList,
        ffn: nn.ModuleList,
        ffn_norm: nn.ModuleList,
        dropout: nn.Dropout,
        memory_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply num_layers of cross-attention + FFN.

        Args:
            query:      [num_tokens, B, C]  query tokens
            memory:     [HW, B, C]          key/value features
            memory_pos: [HW, B, C] or None  positional encoding for memory
        Returns:
            [num_tokens, B, C]
        """
        # Prepare key/value with optional positional encoding
        k = v = memory if memory_pos is None else memory + memory_pos

        for i in range(self.num_layers):
            # cross-attention
            attn_out, _ = cross_attn[i](query, k, v, need_weights=False)
            query = query + dropout(attn_out)
            query = norm[i](query)
            # FFN
            ffn_out = ffn[i](query)
            query = query + dropout(ffn_out)
            query = ffn_norm[i](query)
        return query

    def forward(
        self,
        supp_bbox_emb: torch.Tensor,
        supp_feat: torch.Tensor,
        query_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Run bbox token transfer pipeline.

        Args:
            supp_bbox_emb:  [B, 2, C]   SAM-style support bbox token（RFF-PE+corner）
            supp_feat:      [B, C, H, W]  support feature map（建议为 mask 前景特征）
            query_feat:     [B, C, H, W]  query feature map

        Returns:
            query_bbox_token:  [B, 2, C]  可直接作为 SAM sparse prompt 的框 token
        """
        B, C, H, W = supp_feat.shape
        device = supp_feat.device

        # ---- preparation: reshape to Transformer format ----------------------
        # learnable query bbox tokens（独立初始化，不从 support token 演化）: [2, B, C]
        bbox_tokens = self.query_bbox_tokens.unsqueeze(1).expand(-1, B, -1)  # [2, B, C]

        # support memory: 支持图前景特征 + 支持框 token 一起作为 K/V
        supp_memory = supp_feat.flatten(2).permute(2, 0, 1)   # [HW, B, C]
        supp_bbox_tokens = supp_bbox_emb.permute(1, 0, 2)     # [2, B, C]
        supp_memory = torch.cat([supp_bbox_tokens, supp_memory], dim=0)  # [HW+2, B, C]

        # query features as memory + positional encoding
        qry_memory = query_feat.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        qry_pos = self.pe_layer(query_feat, None)            # [B, C, H, W]
        qry_pos = qry_pos.flatten(2).permute(2, 0, 1)        # [HW, B, C]

        # ---- Stage 1: learnable tokens → 从支持框 token + 支持图特征读取目标信息 ----
        Z_s = self._cross_attn_block(
            query=bbox_tokens,
            memory=supp_memory,
            cross_attn=self.supp_cross_attn,
            norm=self.supp_norm,
            ffn=self.supp_ffn,
            ffn_norm=self.supp_ffn_norm,
            dropout=self.supp_dropout,
        )  # [2, B, C]

        # ---- Stage 2: enriched tokens → locate in query image ----------------
        Z_q = self._cross_attn_block(
            query=Z_s,
            memory=qry_memory,
            cross_attn=self.qry_cross_attn,
            norm=self.qry_norm,
            ffn=self.qry_ffn,
            ffn_norm=self.qry_ffn_norm,
            dropout=self.qry_dropout,
            memory_pos=qry_pos,
        )  # [2, B, C]

        # ---- 输出：框 token（不再回归 4 坐标） --------------------------------
        query_bbox_token = Z_q.permute(1, 0, 2)   # [B, 2, C]

        return query_bbox_token
