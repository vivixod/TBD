import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionedPromptGenerator(nn.Module):
    def __init__(self, hidden_dim=256, layer2_ch=512, layer4_ch=2048):
        super().__init__()
        self.reduce_layer2 = nn.Conv2d(layer2_ch, hidden_dim, 1, bias=False)
        self.reduce_layer4 = nn.Conv2d(layer4_ch, hidden_dim, 1, bias=False)

        self.deconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2)

        # 融合模块：注意输入通道数必须与实际拼接的特征数量一致
        self.fuse1 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1, bias=False),   # 512 -> 256
            nn.ReLU(inplace=False))
        self.fuse2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1, bias=False),   # 768 -> 256
            nn.ReLU(inplace=False))
        self.fuse3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1, bias=False),   # 768 -> 256
            nn.ReLU(inplace=False))

        self.mask_head = nn.Conv2d(hidden_dim, 1, 1, bias=False)

    def forward(self, feat_layer2, feat_layer4, query_feat_down, supp_vec):
        B, _, H2, W2 = feat_layer2.shape
        _, _, Hd, Wd = query_feat_down.shape
        _, _, H4, W4 = feat_layer4.shape

        # 广播支持原型到各尺度
        proto_2 = supp_vec.view(B, -1, 1, 1).repeat(1, 1, H2, W2)
        proto_d = supp_vec.view(B, -1, 1, 1).repeat(1, 1, Hd, Wd)
        proto_4 = supp_vec.view(B, -1, 1, 1).repeat(1, 1, H4, W4)

        # 最深尺度 (1/32 或 1/16)
        f4 = self.reduce_layer4(feat_layer4)                # [B, 256, H4, W4]
        f4 = torch.cat([f4, proto_4], dim=1)               # [B, 512, H4, W4]
        f4 = self.fuse1(f4)                                 # [B, 256, H4, W4]

        # 上采样到 query_feat_down 尺寸
        f3_up = F.interpolate(f4, size=(Hd, Wd), mode='bilinear', align_corners=True)
        f3 = torch.cat([f3_up, query_feat_down, proto_d], dim=1)  # [B, 768, Hd, Wd]
        f3 = self.fuse2(f3)                                 # [B, 256, Hd, Wd]

        # 上采样到 layer2 尺寸
        f2_up = F.interpolate(f3, size=(H2, W2), mode='bilinear', align_corners=True)
        f2_r = self.reduce_layer2(feat_layer2)              # [B, 256, H2, W2]
        f2 = torch.cat([f2_up, f2_r, proto_2], dim=1)      # [B, 768, H2, W2]
        f2 = self.fuse3(f2)                                 # [B, 256, H2, W2]

        # 输出伪掩码
        pseudo_mask = self.mask_head(f2)                    # [B, 1, H2, W2]
        return torch.sigmoid(pseudo_mask)