r""" Visual Prompt Encoder of VRP-SAM """
from functools import reduce
from operator import add
import random
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import model.base.resnet as models
import model.base.vgg as vgg_models
from model.base.eicl_interaction import EICLInteraction
from model.base.bbox_tokenizer import (bbox_to_sam_format,
    PositionEmbeddingRandom, CornerTypeEmbedding, encode_support_bbox,
    extract_gt_bbox, perturb_bbox)
from model.base.bbox_transfer import BboxTransferModule
from torch.nn import BatchNorm2d as BatchNorm
from common.utils import get_stroke_preset, get_random_points_from_mask, get_mask_by_input_strokes

import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import numpy as np
import os
from .base.transformer_decoder import transformer_decoder

# # copy from SEEM
# def get_bounding_boxes(mask):
#         """
#         Returns:
#             Boxes: tight bounding boxes around bitmasks.
#             If a mask is empty, it's bounding box will be all zero.
#         """
#         boxes = torch.zeros(mask.shape[0], 4, dtype=torch.float32).to(mask.device)
#         box_mask = torch.zeros_like(mask).to(mask.device)
#         x_any = torch.any(mask, dim=1)
#         y_any = torch.any(mask, dim=2)
#         for idx in range(mask.shape[0]):
#             x = torch.where(x_any[idx, :])[0].int()
#             y = torch.where(y_any[idx, :])[0].int()
#             if len(x) > 0 and len(y) > 0:
#                 boxes[idx, :] = torch.as_tensor(
#                     [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
#                 )
#                 x1, y1, x2, y2 = x[0], y[0], x[-1] + 1, y[-1] + 1
        
#                 box_mask[idx, y1:y2, x1:x2]=1
#         return boxes, box_mask

# def get_point_mask(mask, training, max_points=20):
#         """
#         Returns:
#             Point_mask: random 20 point for train and test.
#             If a mask is empty, it's Point_mask will be all zero.
#         """
#         max_points = min(max_points, mask.sum().item())
#         if training:
#             num_points = random.Random().randint(1, max_points) # get a random number of points 
#         else:
#             num_points = max_points 
#         b,h,w = mask.shape
#         point_masks = []

#         for idx in range(b):
#             view_mask = mask[idx].view(-1)
#             non_zero_idx = view_mask.nonzero()[:,0] # get non-zero index of mask
#             selected_idx = torch.randperm(len(non_zero_idx))[:num_points] # select id
#             non_zero_idx = non_zero_idx[selected_idx] # select non-zero index
#             rand_mask = torch.zeros(view_mask.shape).to(mask.device) # init rand mask
#             rand_mask[non_zero_idx] = 1 # get one place to zero
#             point_masks.append(rand_mask.reshape(h, w).unsqueeze(0))
#         return torch.cat(point_masks, 0)

# def get_scribble_mask(mask, training, stroke_preset=['rand_curve', 'rand_curve_small'], stroke_prob=[0.5, 0.5]):
#         """
#         Returns:
#             Scribble_mask: random 20 point for train and test.
#             If a mask is empty, it's Scribble_mask will be all zero.
#         """
#         if training:
#             stroke_preset_name = random.Random().choices(stroke_preset, weights=stroke_prob, k=1)[0]
#             nStroke = random.Random().randint(1, min(20, mask.sum().item()))
#         else:
#             stroke_preset_name = random.Random(321).choices(stroke_preset, weights=stroke_prob, k=1)[0]
#             nStroke = random.Random(321).randint(1, min(20, mask.sum().item()))
#         preset = get_stroke_preset(stroke_preset_name)
        
#         b,h,w = mask.shape
        
#         scribble_masks = []
#         for idx in range(b):
#             points = get_random_points_from_mask(mask[idx].bool(), n=nStroke)  
#             rand_mask = get_mask_by_input_strokes(init_points=points, imageWidth=w, imageHeight=h, nStroke=min(nStroke, len(points)), **preset)
#             rand_mask = (~torch.from_numpy(rand_mask)) * mask[idx].bool().cpu()
#             scribble_masks.append(rand_mask.float().unsqueeze(0))
#         return torch.cat(scribble_masks, 0).to(mask.device)



def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs, targets = inputs.flatten(1), targets.flatten(1)
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

class VRP_encoder(nn.Module):
    def __init__(self, args, backbone, use_original_imgsize):
        super(VRP_encoder, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=True)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=True)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        elif backbone == 'convnext_tiny':
            # 1. 创建空的 ConvNeXt-Tiny 模型
            convnext = timm.create_model('convnext_tiny', pretrained=False)
            
            # 2. 加载你的本地权重文件
            model_path = '/vcl/2025zhutaotao/LanGuideMedSeg-ysm/lib/convnext-tiny-224/pytorch_model.bin'
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 兼容处理：检查权重是存在字典的最外层还是 'model'/'state_dict' 键中
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # strict=False 防止分类头不匹配导致报错
            convnext.load_state_dict(state_dict, strict=False)
            print("✅ 成功加载本地 ConvNeXt-Tiny 模型权重！")
            
            # 3. 偷天换日：完美映射到 ResNet 的 layer0 ~ layer4 结构
            self.layer0 = convnext.stem       # 对应 ResNet stem，输出通道: 96,  Stride: 4
            self.layer1 = convnext.stages[0]  # 对应 ResNet layer1，输出通道: 96,  Stride: 4
            self.layer2 = convnext.stages[1]  # 对应 ResNet layer2，输出通道: 192, Stride: 8
            self.layer3 = convnext.stages[2]  # 对应 ResNet layer3，输出通道: 384, Stride: 16
            self.layer4 = convnext.stages[3]  # 对应 ResNet layer4，输出通道: 768, Stride: 32
        elif backbone == 'RadImageNet':
            # 1. 实例化你本地代码库魔改过的 ResNet50 (不再导入 torchvision)
            resnet = models.resnet50(pretrained=False)
            
            # 2. 读取 RadImageNet 本地权重
            rad_weight_path = '/vcl/2025zhutaotao/VRP-SAM/weights/ResNet50.pt'
            checkpoint = torch.load(rad_weight_path, map_location='cpu')
            
            # 3. 权重字典清洗 
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            clean_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '') 
                if 'fc' not in new_k: # 丢弃分类头
                    clean_state_dict[new_k] = v
                    
            # 4. strict=False解决shallow layer的尺寸冲突
            resnet.load_state_dict(clean_state_dict, strict=False)
            
            # 5. 恢复成最原始的 layer0 结构
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.layer0.eval(), self.layer1.eval(), self.layer2.eval(), self.layer3.eval(), self.layer4.eval()
        if backbone == 'vgg16':
            fea_dim = 512 + 256
        # else:
        #     # fea_dim = 1024 + 512
        #     fea_dim = 2048 # 注意：你原来是 concat 了 layer2 和 3，这里如果你只用 layer4，维度是 2048
        elif backbone == 'convnext_tiny':
            vrp_fea_dim = 192 + 384  # 576 (convnext_tiny的layer2输出192通道,layer3输出384)
            iris_fea_dim = 768       # 768 (convnext_tiny的layer4输出768通道)
        else:
            vrp_fea_dim = 1024 + 512  # 1536 (原 VRP 分支拼接 layer2和3 的维度)
            iris_fea_dim = 2048       # 2048 (Iris 分支单吃最深层 layer4 的维度)

        hidden_dim = 256
        self.hidden_dim = hidden_dim
        
        # ==================================================
        # E-ICL: patch-level query-support interaction (替代 prototype + ConditionedPromptGenerator)
        # ==================================================
        self.eicl_interaction = EICLInteraction(hidden_dim=self.hidden_dim, temperature=0.1)
        
        # ==================================================
        # Bbox Transfer Module（ChatGPT 方案）:
        #   supp_bbox → SAM-RFF-PE + corner type → [B,2,256]
        #   → cross-attn(supp_feat) → "框里是什么"
        #   → cross-attn(query_feat+2D PE) → "查询图哪里"
        #   → concat TL+BR → MLP → (cx,cy,w,h)
        # ==================================================
        self.bbox_pe_layer = PositionEmbeddingRandom(
            num_pos_feats=self.hidden_dim // 2, scale=1.0
        )
        self.corner_emb = CornerTypeEmbedding(embed_dim=self.hidden_dim)
        self.bbox_transfer = BboxTransferModule(
            hidden_dim=self.hidden_dim,
            nhead=8,
            num_layers=1,
            dropout=0.1,
        )
        
        # ==================================================
        # 特征拼接 + Transformer Decoder → 50 tokens（恢复）
        # 单路检索后 query/supp 输入均为:
        #   [自身特征 256, attended_supp 256, pseudo_mask 1] = 513 通道
        # ==================================================
        self.merge_1 = nn.Sequential(
            nn.Conv2d(hidden_dim*2+1, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
        )
        self.proto_dropout = nn.Dropout(0.1)
        self.num_query = args.num_query
        self.transformer_decoder = transformer_decoder(args, args.num_query, hidden_dim, hidden_dim*2)

        # 特征降维（保留）
        self.downsample_query = nn.Sequential(
            nn.Conv2d(vrp_fea_dim, self.hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=0.5))

        # Loss 函数
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    # def forward(self, condition, query_img, support_img, support_mask, training):

    #     # if training:
    #     #     condition = random.Random().choices(['scribble', 'point', 'box', 'mask'], weights=[0.25,0.25,0.25,0.25], k=1)[0]  
    #     # else:
    #     #     condition = condition

    #     # if condition == 'scribble':
    #     #     support_mask_ori = get_scribble_mask(support_mask,training) # scribble_mask
    #     # elif condition == 'point':
    #     #     support_mask_ori = get_point_mask(support_mask,training) # point_mask
    #     # elif condition == 'box':
    #     #     boxes, support_mask_ori = get_bounding_boxes(support_mask) # box_mask
    #     # elif condition == 'mask':
    #     #     support_mask_ori = support_mask
        
    #     # 简化逻辑：无论传入什么 condition，都直接使用完整 mask
    #     # 这样即便以后 train.sh 忘了改参数，代码也不会报错
    #     support_mask_ori = support_mask
        
    #     # 确保 mask 至少是 [B, 1, H, W] 维度
    #     if support_mask.dim() == 3:
    #         mask_expanded = support_mask.unsqueeze(1).float()
    #     else:
    #         mask_expanded = support_mask.float()

    #     # with torch.no_grad():
    #     #     query_feat_0 = self.layer0(query_img)
    #     #     query_feat_1 = self.layer1(query_feat_0)
    #     #     query_feat_2 = self.layer2(query_feat_1)
    #     #     query_feat_3 = self.layer3(query_feat_2)
    #     #     query_feat_4 = self.layer4(query_feat_3)
    #     #     if self.backbone_type == 'vgg16':
    #     #         query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)
    #     #     query_feat = torch.cat([query_feat_2, query_feat_3], 1) 

    #     #     supp_feat_0 = self.layer0(support_img)
    #     #     supp_feat_1 = self.layer1(supp_feat_0)
    #     #     supp_feat_2 = self.layer2(supp_feat_1)
    #     #     supp_feat_3 = self.layer3(supp_feat_2)
    #     #     support_mask = F.interpolate(support_mask_ori.unsqueeze(1).float(), size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='nearest')
    #     #     supp_feat_4 = self.layer4(supp_feat_3*support_mask)
    #     #     if self.backbone_type == 'vgg16':
    #     #         supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)
    #     #     supp_feat = torch.cat([supp_feat_2, supp_feat_3], 1)
    #     #     pseudo_mask = self.get_pseudo_mask(supp_feat_4, query_feat_4, support_mask)
        
    #     # query_feat = self.downsample_query(query_feat)
    #     # supp_feat = self.downsample_query(supp_feat)
    #     # prototype = self.mask_feature(supp_feat, support_mask)
    #     # supp_feat_bin = prototype.repeat(1, 1, query_feat.shape[2], query_feat.shape[3])
    #     # supp_feat_1 = self.merge_1(torch.cat([supp_feat, supp_feat_bin, support_mask*10], 1))                                                                                    
    #     # query_feat_1 = self.merge_1(torch.cat([query_feat, supp_feat_bin, pseudo_mask*10], 1))

    #     # protos = self.transformer_decoder(query_feat_1, supp_feat_1, support_mask)
    #     # return protos, support_mask_ori

    def forward(self, condition, query_img, support_img, support_mask, training):
        support_mask_ori = support_mask
        
        if support_mask.dim() == 3:
            mask_expanded = support_mask.unsqueeze(1).float()
        else:
            mask_expanded = support_mask.float()

        # ==========================================
        # 步骤 1: 骨干网络特征提取 (保持不变)
        # ==========================================
        with torch.no_grad():
            query_feat_0 = self.layer0(query_img)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            
            supp_feat_0 = self.layer0(support_img)
            supp_feat_1 = self.layer1(supp_feat_0)
            supp_feat_2 = self.layer2(supp_feat_1)
            
        query_feat_3 = self.layer3(query_feat_2)
        self.layer4(query_feat_3)            # layer4 可训练，需前向；输出不再被 C.P.G. 消费

        supp_feat_3 = self.layer3(supp_feat_2)
        self.layer4(supp_feat_3)            # layer4 可训练，需前向；输出不再被 Iris 消费
            
        if query_feat_2.shape[-2:] != query_feat_3.shape[-2:]:
            query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)), mode='bilinear', align_corners=True)
        query_feat_cat = torch.cat([query_feat_2, query_feat_3], 1) 
        
        if supp_feat_2.shape[-2:] != supp_feat_3.shape[-2:]:
            supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
        supp_feat_cat = torch.cat([supp_feat_2, supp_feat_3], 1)
        
        supp_mask_down = F.interpolate(mask_expanded, size=supp_feat_3.shape[-2:], mode='nearest')
            
        # 提前降维到 256
        query_feat_down = self.downsample_query(query_feat_cat)
        supp_feat_down = self.downsample_query(supp_feat_cat)
        B, C, H, W = query_feat_down.shape
        
        # ==========================================
        # 步骤 2: E-ICL 单路检索（EICSeg 原式）
        #   A = softmax(Q^T K / τ)    →  patch 级对应关系
        #   pseudo = A @ M            →  粗略定位图
        #   attended = A @ K          →  加权 support 特征
        # ==========================================
        pseudo_mask, attended_supp = self.eicl_interaction(
            query_feat_down, supp_feat_down, supp_mask_down
        )

        # ==========================================
        # 步骤 3a: 50 tokens 路径（Transformer Decoder）
        # ==========================================
        query_feat_1 = self.merge_1(torch.cat([
            query_feat_down,
            attended_supp,
            pseudo_mask
        ], dim=1))

        supp_feat_1 = self.merge_1(torch.cat([
            supp_feat_down,
            supp_feat_down * supp_mask_down,
            supp_mask_down
        ], dim=1))

        protos = self.transformer_decoder(query_feat_1, supp_feat_1, supp_mask_down)  # [B, 50, C]
        if self.training:
            protos = self.proto_dropout(protos)

        # ==========================================
        # 步骤 2b: Bbox Token 生成
        #   ① 从 support mask 提取 GT 框 → SAM 编码 → supp_bbox_emb [B,2,256]
        #   ② learnable query bbox tokens（独立初始化）
        #      → cross-attn(supp_bbox_emb + supp 前景特征) → "目标是什么"
        #      → cross-attn(query_feat + 2D PE) → "目标在查询图哪里"
        #   ③ 输出 query bbox token [B,2,256]（直接给 SAM 当 sparse prompt）
        # ==========================================
        supp_bbox_norm = extract_gt_bbox(support_mask)                     # [B, 4] (cx,cy,w,h)
        supp_bbox_pixel = bbox_to_sam_format(supp_bbox_norm,
                                             img_h=1024, img_w=1024).squeeze(1)  # [B, 4] xyxy
        supp_bbox_pixel = perturb_bbox(supp_bbox_pixel, img_h=1024, img_w=1024,
                                       max_shift=3.0, training=training)
        supp_bbox_emb = encode_support_bbox(supp_bbox_pixel,
                                            self.bbox_pe_layer,
                                            self.corner_emb,
                                            image_size=(1024, 1024))       # [B,2,256]

        supp_feat_fg = supp_feat_down * supp_mask_down   # 只保留支持图前景
        query_bbox_token = self.bbox_transfer(supp_bbox_emb,
                                              supp_feat_fg,
                                              query_feat_down)  # [B, 2, 256]

        # 返回：50 tokens + 2 个框 token + pseudo_mask
        return protos, query_bbox_token, pseudo_mask

    def mask_feature(self, features, support_mask):
        mask = support_mask
        supp_feat = features * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
        return supp_feat

    def predict_mask_nshot(self, args, batch, sam_model, nshot, input_point=None):

        logit_mask_agg = 0
        token_set = []
        bbox_token_set = []
        for s_idx in range(nshot):
            tokens_sub, bbox_token_sub, _ = self(args.condition, batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx], False)
            token_set.append(tokens_sub)
            bbox_token_set.append(bbox_token_sub)
            
        if nshot > 1:
            fused_token = torch.stack(token_set, dim=0).mean(dim=0)
            bbox_token = torch.stack(bbox_token_set, dim=0).mean(dim=0)
        else:
            fused_token = tokens_sub
            bbox_token = bbox_token_sub

        # 拼接 50 tokens + 2 框 tokens → SAM sparse prompt
        # （与 train.py 保持一致：coarse_mask 不传 SAM —— 该 attn bias 为试错后弃用）
        protos_all = torch.cat([fused_token, bbox_token], dim=1)   # [B, 52, C]

        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos_all)
        logit_mask = low_masks
        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        pred_mask = torch.sigmoid(logit_mask) >= 0.5

        pred_mask = pred_mask.float()
            
        logit_mask_agg += pred_mask.squeeze(1).clone()
        return logit_mask_agg, pred_mask, logit_mask
    

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        loss_bce = self.bce_with_logits_loss(logit_mask.squeeze(1), gt_mask.float())
        loss_dice = dice_loss(logit_mask, gt_mask, bsz)
        return loss_bce + loss_dice

    def compute_bbox_token_loss(self, pred_token, query_mask):
        """在 token 空间监督：MSE(预测框 token, GT查询框的SAM编码)。

        Args:
            pred_token: [B, 2, C]  模型输出的 query bbox token
            query_mask: [B, H, W]  查询图 GT mask
        Returns:
            loss: 标量 MSE
            cos_sim: 平均余弦相似度（监控用，越接近 1 越好）
        """
        # 从 GT mask 提取查询框 → 编码成同空间的 GT token
        gt_norm = extract_gt_bbox(query_mask)                              # [B,4] cxcywh
        gt_pixel = bbox_to_sam_format(gt_norm, img_h=1024, img_w=1024).squeeze(1)  # [B,4] xyxy
        gt_token = encode_support_bbox(gt_pixel,
                                       self.bbox_pe_layer,
                                       self.corner_emb,
                                       image_size=(1024, 1024))             # [B,2,C]

        loss_mse = F.mse_loss(pred_token, gt_token)
        cos_sim = F.cosine_similarity(
            pred_token.flatten(1), gt_token.flatten(1), dim=-1
        ).mean()
        return loss_mse, cos_sim
        

    def train_mode(self):
        self.train()
        self.apply(fix_bn)
        self.layer0.eval(), self.layer1.eval(), self.layer2.eval()
        self.layer3.train()
        self.layer4.train()

    def get_pseudo_mask(self, tmp_supp_feat, query_feat_4, mask):
        resize_size = tmp_supp_feat.size(2)
        tmp_mask = F.interpolate(mask, size=(resize_size, resize_size), mode='bilinear', align_corners=True)

        tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
        q = query_feat_4
        s = tmp_supp_feat_4
        bsize, ch_sz, sp_sz, _ = q.size()[:]

        tmp_query = q
        tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s               
        tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1) 
        tmp_supp = tmp_supp.permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

        cosine_eps = 1e-7
        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
        similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)
        corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
        return corr_query

    def debug_visualize_pseudo_mask(self, pseudo_mask, query_img, support_mask_ori, 
                                save_dir='debug_pseudo', step=None, batch_idx=None):
        # 已达上限则直接返回
        if self.pseudo_save_counter >= self.max_pseudo_saves:
            return

        os.makedirs(save_dir, exist_ok=True)

        # 取 batch 中第一张
        mask = pseudo_mask[0].detach().cpu()
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        img = query_img[0].detach().cpu()
        sup_mask = support_mask_ori[0].detach().cpu()
        if sup_mask.dim() == 3:
            sup_mask = sup_mask.squeeze(0)

        # 反归一化 (ImageNet 均值/方差)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_vis = img * std + mean
        img_vis = img_vis.clamp(0, 1)

        # 上采样掩码到查询图像尺寸
        mask = mask.unsqueeze(0).unsqueeze(0)                     # [1,1,H_mask,W_mask]
        mask_resized = F.interpolate(mask, size=img_vis.shape[1:], 
                                    mode='bilinear', align_corners=True)
        mask_resized = mask_resized.squeeze(0).squeeze(0)         # [H_img, W_img]
        mask_np = mask_resized.numpy().astype(np.float32)

        img_np = img_vis.permute(1,2,0).numpy()
        img_np = np.clip(img_np, 0, 1)

        # 红色叠加
        alpha = 0.5
        red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        overlay = img_np * (1 - alpha * mask_np[..., None]) + red * alpha * mask_np[..., None]
        overlay = np.clip(overlay, 0, 1) * 255
        overlay = overlay.astype(np.uint8)

        # 支持掩码上采样
        sup_mask = sup_mask.unsqueeze(0).unsqueeze(0).float()
        sup_mask_resized = F.interpolate(sup_mask, size=img_vis.shape[1:], 
                                        mode='nearest').squeeze(0).squeeze(0).numpy()
        sup_mask_rgb = (np.stack([sup_mask_resized]*3, axis=-1) * 255).astype(np.uint8)

        # 水平拼接
        combined = np.concatenate([overlay, sup_mask_rgb], axis=1)
        combined_pil = Image.fromarray(combined)

        # 使用计数器作为文件名（固定序号，与 step/batch 无关）
        filename = f"{self.pseudo_save_counter}.png"
        combined_pil.save(os.path.join(save_dir, filename))

        # 增加计数
        self.pseudo_save_counter += 1