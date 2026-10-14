from segment_anything import sam_model_registry
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class  SAM_pred(nn.Module):
    def __init__(self, encoder_precision='auto'):
        super().__init__()
        if encoder_precision not in ('auto', 'fp32', 'bf16', 'fp16'):
            raise ValueError(f'Unsupported SAM encoder precision: {encoder_precision}')
        self.encoder_precision = encoder_precision
        # self.sam_model = sam_model_registry['vit_h']('/root/paddlejob/workspace/env_run/vrp_sam/sam_vit_h_4b8939.pth')
        _ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'sam_vit_h_4b8939.pth')
        self.sam_model = sam_model_registry['vit_h'](_ckpt)
        
        for param in self.sam_model.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam_model.prompt_encoder.parameters():
            param.requires_grad = True
        for param in self.sam_model.mask_decoder.parameters():
            param.requires_grad = True
        
        self.sam_model.eval()
        # 只允许 eval 模式使用缓存；训练时在线编码增强后的 query。
        self.use_cache = True
        
    # 覆盖 train() 方法，保证冻结层的安全    
    def train(self, mode=True):
        super().train(mode)
        if mode:
            # 即使整个模型处于 train 模式，也要强行把 image_encoder 摁在 eval 模式
            # 防止内部的 BatchNorm/Dropout 收集错误的统计数据
            self.sam_model.image_encoder.eval()
        return self

    def resolve_encoder_precision(self, device):
        """Only training on CUDA uses lower precision; eval caches remain FP32."""
        device = torch.device(device)
        if not self.training or device.type != 'cuda' or self.encoder_precision == 'fp32':
            return 'fp32'
        if self.encoder_precision == 'fp16':
            return 'fp16'
        native_bf16 = (torch.cuda.get_device_capability(device)[0] >= 8
                       and torch.cuda.is_bf16_supported())
        if native_bf16:
            return 'bf16'
        if self.encoder_precision == 'bf16':
            raise RuntimeError('This GPU does not support native BF16; use '
                               '--sam-encoder-precision fp32 or fp16')
        return 'fp32'

    def forward_img_encoder(self, query_img):
        query_img = F.interpolate(query_img, (1024,1024), mode='bilinear', align_corners=True)
        precision = self.resolve_encoder_precision(query_img.device)
        dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(precision, torch.float32)
        with torch.no_grad(), torch.autocast(
                device_type=query_img.device.type, dtype=dtype, enabled=precision != 'fp32'):
            query_feats = self.sam_model.image_encoder(query_img)
        # 冻结编码器无需 GradScaler；可训练的 prompt/decoder 和所有损失仍用 FP32。
        return query_feats.float()
    
    # def get_feat_from_np(self, query_img, query_name, protos):
    #     # np_feat_path = '/root/paddlejob/workspace/env_run/vrp_sam/feats_np/coco/'
    #     np_feat_path = os.path.join('/vcl/2025zhutaotao/VRP-SAM', 'cache_features')
    #     if not os.path.exists(np_feat_path): os.makedirs(np_feat_path)
    #     files_name = os.listdir(np_feat_path)
    #     query_feat_list = []
    #     for idx, name in enumerate(query_name):
    #         if '/root' in name:
    #             name = os.path.splitext(name.split('/')[-1])[0]
                
    #         if name + '.npy' not in files_name:
    #             query_feats_np = self.forward_img_encoder(query_img[idx, :, :, :].unsqueeze(0))
    #             query_feat_list.append(query_feats_np)
    #             query_feats_np = query_feats_np.detach().cpu().numpy()
    #             np.save(np_feat_path + name + '.npy', query_feats_np)
    #         else:
    #             sub_query_feat = torch.from_numpy(np.load(np_feat_path + name + '.npy')).to(protos.device)
    #             query_feat_list.append(sub_query_feat)
    #             del sub_query_feat
    #     query_feats_np = torch.cat(query_feat_list, dim=0)
    #     return query_feats_np
    
    def get_feat_from_np(self, query_img, query_name, protos):
        # 即使直接调用此方法，训练也不能读取或写入增强图像的磁盘缓存。
        if self.training or not self.use_cache:
            return self.forward_img_encoder(query_img)

        # 独立目录隔离旧缓存：旧文件可能来自历史随机增强，无法保证几何一致。
        cache_root = os.environ.get('TBD_FEATS', '/data2/2025Zhutaotao/TBD/cache_features')
        np_feat_path = os.path.join(cache_root, 'eval_unaugmented_v1')
        os.makedirs(np_feat_path, exist_ok=True)

        query_feat_list = []
        for idx, name in enumerate(query_name):
            if '/root' in name:
                name = os.path.splitext(name.split('/')[-1])[0]
                
            # 使用 os.path.join 来确保路径分隔符正确
            save_path = os.path.join(np_feat_path, name + '.npy')

            cached = None
            if os.path.isfile(save_path):
                try:
                    cached_np = np.load(save_path, allow_pickle=False)
                    if cached_np.shape == (1, 256, 64, 64) and np.isfinite(cached_np).all():
                        cached = torch.from_numpy(cached_np).to(protos.device)
                except Exception:
                    cached = None   # 缓存损坏 → 重算

            if cached is None:
                query_feats_np = self.forward_img_encoder(query_img[idx, :, :, :].unsqueeze(0))
                cached = query_feats_np.to(protos.device)
                # 原子写入：tmp 文件名带 PID，避免多 rank 同时写同一 .tmp 互相 rename 冲突
                tmp_path = f'{save_path}.tmp{os.getpid()}'
                try:
                    # 文件句柄阻止 np.save 自动追加 .npy，保证 replace 的源路径正确。
                    with open(tmp_path, 'wb') as tmp_file:
                        np.save(tmp_file, query_feats_np.detach().cpu().numpy())
                    os.replace(tmp_path, save_path)
                except Exception:
                    # 缓存写失败/竞争不影响训练，直接用刚算的特征继续
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass

            query_feat_list.append(cached)
        
        query_feats_np = torch.cat(query_feat_list, dim=0)
        return query_feats_np

    # def get_pormpt(self, protos, points_mask=None):
    #     if points_mask is not None :
    #         point_mask = points_mask

    #         postivate_pos = (point_mask.squeeze(0).nonzero().unsqueeze(0) + 0.5) * 64 -0.5
    #         postivate_pos = postivate_pos[:,:,[1,0]]
    #         point_label = torch.ones(postivate_pos.shape[0], postivate_pos.shape[1]).to(postivate_pos.device)
    #         point_prompt = (postivate_pos, point_label)
    #     else:
    #         point_prompt = None
    #     protos = protos
    #     return  protos, point_prompt

    def forward_prompt_encoder(self, points=None, boxes=None, protos=None, masks=None):
        q_sparse_em, q_dense_em = self.sam_model.prompt_encoder(
                points=points,
                boxes=boxes,
                protos=protos,
                masks=masks)
        return  q_sparse_em, q_dense_em
    
    def forward_mask_decoder(self, query_feats, q_sparse_em, q_dense_em, ori_size=(512,512), attn_bias=None):
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=query_feats,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=q_sparse_em,
                dense_prompt_embeddings=q_dense_em,
                multimask_output=False,
                attn_bias=attn_bias)
        low_masks = F.interpolate(low_res_masks, size=ori_size, mode='bilinear', align_corners=True)
            
        # from torch.nn.functional import threshold, normalize

        # binary_mask = normalize(threshold(low_masks, 0.0, 0))
        binary_mask = torch.where(low_masks > 0, 1, 0)
        return low_masks, binary_mask

    # ==================== 级联精修（Cascaded Post-refinement，已弃用） ====================
    def _extract_posneg_points(self, low_masks, ori_size=(512, 512)):
        """从 logits 取置信度最高/最低点作为正负点 prompt。
        返回 (coords, labels)：coords [B,2,2] (x,y) 1024 空间，labels [B,2] = [[1,0]]。"""
        h, w = ori_size
        B = low_masks.shape[0]
        scale_x = 1024.0 / w
        scale_y = 1024.0 / h
        flat = low_masks.view(B, -1)
        pos_idx = flat.argmax(dim=1)
        neg_idx = flat.argmin(dim=1)
        pos_pts = torch.stack([(pos_idx % w).float() * scale_x,
                               (pos_idx // w).float() * scale_y], dim=1)  # [B,2] (x,y)
        neg_pts = torch.stack([(neg_idx % w).float() * scale_x,
                               (neg_idx // w).float() * scale_y], dim=1)
        coords = torch.stack([pos_pts, neg_pts], dim=1)                   # [B,2,2]
        labels = torch.tensor([[1, 0]], device=low_masks.device).expand(B, -1)
        return coords, labels

    def _mask_to_bbox(self, binary_mask, ori_size=(512, 512)):
        """binary_mask [B,1,H,W] → [B,4] xyxy（1024 空间）。空 mask 回退到整图。"""
        h, w = ori_size
        B = binary_mask.shape[0]
        scale_x = 1024.0 / w
        scale_y = 1024.0 / h
        boxes = []
        for b in range(B):
            nz = binary_mask[b, 0].nonzero()      # [N,2] (y,x)
            if nz.numel() == 0:
                boxes.append([0.0, 0.0, 1.0, 1.0])
            else:
                y0, x0 = nz.min(dim=0).values.float()
                y1, x1 = nz.max(dim=0).values.float()
                boxes.append([x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y])
        return torch.tensor(boxes, device=binary_mask.device, dtype=torch.float32)

    def _cascade_refine(self, query_feats, protos, low_masks, binary_mask,
                        ori_size=(512, 512)):
        """级联精修：SAM 第一遍输出 mask → 提 bbox → 官方 box 编码 → 拼进 protos 再喂第二遍。
        protos = 第一遍的 52 tokens（50 support tokens + 2 bbox tokens）。
        第二遍 protos2 = Concat(protos, box_embedding) = 54 tokens。
        只跑轻量 decoder，不增加 image encoder 开销。（已确认负效果，默认不启用）"""
        # ---- 第二遍：提 bbox → 官方 box 编码 → 拼进 protos ----
        boxes3 = self._mask_to_bbox(binary_mask, ori_size)        # [B,4] xyxy 1024 空间
        box_embedding, _ = self.forward_prompt_encoder(
            points=None, boxes=boxes3.unsqueeze(1), protos=None, masks=None)
        # box_embedding: [B, 2, 256] 官方 box 编码
        protos2 = torch.cat([protos, box_embedding], dim=1)        # [B, 54, 256]
        q_sparse3, q_dense3 = self.forward_prompt_encoder(
            points=None, boxes=None, protos=protos2, masks=None)
        low_masks3, binary_mask3 = self.forward_mask_decoder(
            query_feats, q_sparse3, q_dense3, ori_size=ori_size, attn_bias=None)

        return low_masks3, binary_mask3
    
    # def forward(self, query_img, query_name, protos, points_mask=None):
    #     B,C, h, w = query_img.shape
        
    #     # query_img = F.interpolate(query_img, (1024,1024), mode='bilinear', align_corners=True)
    #     protos, point_prompt = self.get_pormpt(protos, points_mask)
    #     with torch.no_grad():
    #         #-------------save_sam_img_feat-------------------------
    #         # query_feats = self.forward_img_encoder(query_img)

    #         query_feats = self.get_feat_from_np(query_img, query_name, protos)

    #     q_sparse_em, q_dense_em = self.forward_prompt_encoder(
    #             points=point_prompt,
    #             boxes=None,
    #             protos=protos,
    #             masks=None)
            
    #     low_masks, binary_mask = self.forward_mask_decoder(query_feats, q_sparse_em, q_dense_em, ori_size=(h, w))

    #     return low_masks, binary_mask.squeeze(1)
    
    
    def forward(self, query_img, query_name, protos, boxes=None, use_cascade=False,
                image_repeats=1):
        B, C_img, h, w = query_img.shape
        if not isinstance(image_repeats, int) or image_repeats < 1:
            raise ValueError('image_repeats must be a positive integer')
        if len(query_name) != B or protos.shape[0] != B * image_repeats:
            raise ValueError('query names and prompt batch must match query batch * image_repeats')

        if self.use_cache and not self.training:
            with torch.no_grad():
                query_feats = self.get_feat_from_np(query_img, query_name, protos)
        else:
            query_feats = self.forward_img_encoder(query_img)

        # 同一 query 的多个 support 仅重复 embedding，ViT-H 每张 query 编码一次。
        # 顺序与 train.py 的 query.repeat_interleave(S) / support.flatten(0,1) 对齐。
        if image_repeats > 1:
            query_feats = query_feats.repeat_interleave(image_repeats, dim=0)

        # protos: [B, N, 256] (N=52 tokens)  → SAM sparse prompt
        # boxes:  [B, 1, 4]  (predicted bbox) → SAM box prompt
        q_sparse_em, q_dense_em = self.forward_prompt_encoder(
                points=None,
                boxes=boxes,
                protos=protos,
                masks=None)

        # （PerSAM 式 coarse_mask→attn_bias 已删：试错后弃用的无用设计）
        low_masks, binary_mask = self.forward_mask_decoder(query_feats, q_sparse_em, q_dense_em,
                                                           ori_size=(h, w), attn_bias=None)

        if use_cascade:
            low_masks, binary_mask = self._cascade_refine(
                query_feats, protos, low_masks, binary_mask,
                ori_size=(h, w))

        return low_masks, binary_mask.squeeze(1)

        # low_mask_set = []
        # binary_mask_set = []

        # for idx in range(protos.shape[1]):
        #     q_sparse_em, q_dense_em = self.forward_prompt_encoder(
        #         points=point_prompt,
        #         boxes=None,
        #         protos=protos[:,idx,:].unsqueeze(1),
        #         masks=None)
            
        #     low_masks, binary_mask = self.forward_mask_decoder(query_feats, q_sparse_em, q_dense_em, ori_size=(h, w))

        #     low_mask_set.append(low_masks)
        #     binary_mask_set.append(binary_mask.squeeze(1))

        # return low_mask_set, binary_mask_set
