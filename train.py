r""" Visual Prompt Encoder training (validation) code """
import os
import argparse
import sys
import datetime

import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import csv
import math

from model.VRP_encoder import VRP_encoder
from model.base.mask_pooling import repeat_dataset_names
from model.base.support_fusion import (decode_support_logits, fuse_support_logits,
                                      support_quality_loss)
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM2pred import SAM_pred


def train(args, epoch, model, sam_model, dataloader, optimizer, scheduler, training, split_name=None):
    r""" Train VRP_encoder model """

    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    model.module.train_mode() if training else model.module.eval()
    sam_model.train(training)  # train: 在线编码增强图；eval: 使用未增强图的独立缓存
    average_meter = AverageMeter(dataloader.dataset)
    bbox_cos_buf = []   # 每 batch 的 bbox token cos 相似度，epoch 末打印均值
    quality_mode = getattr(args, 'support_fusion', 'mean') == 'quality'
    quality_kwargs = {'return_quality': True} if quality_mode else {}
    fusion_totals = None
    if quality_mode and training:
        model.module.set_fusion_epoch(epoch)

    for idx, batch in enumerate(dataloader):
        
        batch = utils.to_cuda(batch)
        support_dataset_names = repeat_dataset_names(
            batch.get('dataset_name', dataloader.dataset.benchmark),
            batch_size=batch['query_img'].shape[0],
            repeats=batch['support_imgs'].shape[1])

        # # =====================================================================
        # # 🚨 DEBUG: 可视化 Query 和所有的 Support 图像以检查采样合理性 🚨
        # # =====================================================================
        # if epoch == 0 and idx == 0 and utils.is_main_process():

        #     # 1. 提取第 1 个样本的数据
        #     q_img = batch['query_img'][0].cpu() # [3, H, W]
        #     q_mask = batch['query_mask'][0].cpu().squeeze() # [H, W]
            
        #     # Support 的维度通常是 [B, Shot, C, H, W]，提取出第 1 个 Batch 的所有 Shot
        #     s_imgs = batch['support_imgs'][0].cpu() # [Shot, 3, H, W]
        #     s_masks = batch['support_masks'][0].cpu() # [Shot, H, W] 或者 [Shot, 1, H, W]
            
        #     shot_num = s_imgs.shape[0]
        #     class_id = batch['class_id'][0].item()

        #     # 2. 还原 ImageNet 归一化，方便人眼观察
        #     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        #     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        #     def to_numpy_img(tensor_img):
        #         img = tensor_img * std + mean
        #         return torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()

        #     q_img_vis = to_numpy_img(q_img)

        #     # 3. 动态画板：1 行给 Query，下方每行给 1 个 Support Shot
        #     fig, axs = plt.subplots(1 + shot_num, 2, figsize=(10, 5 * (1 + shot_num)))
            
        #     # 如果只有 1 个 shot，axs 是一维数组，需要统一成二维方便索引
        #     if shot_num == 0: # 极端情况兜底
        #         axs = np.expand_dims(axs, axis=0)
        #     elif shot_num == 1:
        #         axs = np.array([axs[0], axs[1]]) if len(axs.shape)==1 else axs

        #     # --- 画 Query ---
        #     axs[0, 0].imshow(q_img_vis)
        #     axs[0, 0].set_title(f'Query Image (Organ ID: {class_id})')
        #     axs[0, 0].axis('off')

        #     axs[0, 1].imshow(q_mask.numpy(), cmap='gray')
        #     axs[0, 1].set_title(f'Query Mask (Sum: {q_mask.sum().item():.0f})')
        #     axs[0, 1].axis('off')

        #     # --- 循环画所有的 Support ---
        #     for s in range(shot_num):
        #         s_img_vis = to_numpy_img(s_imgs[s])
        #         s_mask_vis = s_masks[s].squeeze().numpy()

        #         row = s + 1
        #         axs[row, 0].imshow(s_img_vis)
        #         axs[row, 0].set_title(f'Support Image (Shot {s+1})')
        #         axs[row, 0].axis('off')

        #         axs[row, 1].imshow(s_mask_vis, cmap='gray')
        #         axs[row, 1].set_title(f'Support Mask (Shot {s+1}, Sum: {s_mask_vis.sum():.0f})')
        #         axs[row, 1].axis('off')

        #     plt.tight_layout()
        #     plt.savefig('check_support_set.png')
        #     print(f"\n=======================================================")
        #     print(f"📸 [DEBUG] 对比图已保存至 check_support_set.png！请下载查看。")
        #     print(f"=======================================================\n")
        # # =====================================================================
        
        # # =====================================================================
        # # 🚨 DEBUG: 将每个 Batch 的图片名称和器官类别保存到 CSV 🚨
        # # =====================================================================
        # # 为了防止文件过大和多卡冲突，只在第 1 个 Epoch 且主进程中记录
        # if epoch == 0 and utils.is_main_process():

        #     csv_filename = 'check_dataset_names2.csv'

        #     # 如果是第 1 个 Batch，先创建文件并写入表头
        #     if idx == 0:
        #         with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        #             writer = csv.writer(f)
        #             writer.writerow(['Batch_Idx', 'Item_in_Batch', 'Class_ID', 'Query_Name', 'Support_Names'])

        #     # 提取当前 Batch 的基本信息
        #     B = batch['query_img'].shape[0]  # Batch Size
        #     class_ids = batch['class_id'].cpu().tolist() # 器官类别 ID 列表
            
        #     # 提取 Query 名称 (通常是 Tuple)
        #     query_names = batch.get('query_name', ["N/A"] * B)

        #     # 追加写入当前 Batch 的数据
        #     with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
        #         writer = csv.writer(f)
        #         for b_i in range(B):
        #             c_id = class_ids[b_i]
        #             q_name = query_names[b_i]

        #             # 灵活解析 Support Names
        #             # Pytorch 通常会将多 Shot 的字符串组装成 List[Tuple(str)]
        #             s_names_for_bi = []
        #             if 'support_names' in batch:
        #                 s_names_raw = batch['support_names']
                        
        #                 # 多 Shot 情况: 包含 S 个 Tuple 的 List，每个 Tuple 包含 B 个字符串
        #                 if isinstance(s_names_raw, list) and len(s_names_raw) > 0 and isinstance(s_names_raw[0], tuple):
        #                     s_names_for_bi = [shot_tuple[b_i] for shot_tuple in s_names_raw]
                        
        #                 # 单 Shot 情况: 直接是一个包含 B 个字符串的 Tuple
        #                 elif isinstance(s_names_raw, tuple) and len(s_names_raw) == B:
        #                     s_names_for_bi = [s_names_raw[b_i]]
                        
        #                 # 兜底情况
        #                 else:
        #                     s_names_for_bi = [str(s_names_raw)]

        #             # 将多个 Support 名称用 "|" 拼接，方便在 CSV 中单格显示
        #             s_names_str = " | ".join(s_names_for_bi)
                    
        #             writer.writerow([idx, b_i, c_id, q_name, s_names_str])
        # # =====================================================================
        
        if training:
            # ==========================================
            # 🚨 Multi-Shot 并行处理逻辑 (仅训练时使用) 🚨
            # ==========================================
            B, S, C, H, W = batch['support_imgs'].shape

            # 1. 拍扁 Support: [B, S, C, H, W] -> [B*S, C, H, W]
            supp_imgs_flat = batch['support_imgs'].flatten(0, 1)
            supp_masks_flat = batch['support_masks'].flatten(0, 1)
            
            # 2. 复制 Query 以对齐并行数量 (必须用 repeat_interleave 保证顺序不乱)
            q_imgs_dummy = batch['query_img'].repeat_interleave(S, dim=0)
            # print(f"Debug: q_imgs_dummy shape after repeat_interleave: {q_imgs_dummy.shape}")  # [B*S, C, H, W]

            # 3. 此时所有输入都是标准的 4D 张量，送入模型
            outputs = model(args.condition, q_imgs_dummy, supp_imgs_flat,
                            supp_masks_flat, training,
                            dataset_names=support_dataset_names, **quality_kwargs)
            p_all, bbox_token = outputs[:2]

            # 3a. 拼接 50 上下文 tokens + 2 bbox tokens → 52 tokens 作为 SAM sparse prompt
            #     （coarse_mask 不传 SAM —— 该 attn bias 为试错后弃用）
            protos_all = torch.cat([p_all, bbox_token], dim=1)   # [B*S, 52, C]

            # 3b. Bbox token 空间监督：MSE(预测框 token, GT查询框的SAM编码)
            #     bbox_token 是 [B*S, 2, C]，需把 query_mask 同样复制 S 次对齐
            q_masks_dummy = batch['query_mask'].repeat_interleave(S, dim=0)
            loss_bbox, cos_sim = model.module.compute_bbox_token_loss(bbox_token, q_masks_dummy)
            bbox_cos_buf.append(cos_sim.item())

            # 4. 调用 SAM 模型（不传 coarse_mask —— cascaded post-refinement 撤掉，结论：负效果）
            shot_logits = decode_support_logits(
                sam_model, batch['query_img'], batch['query_name'], protos_all, S)
            scores = outputs[3].reshape(B, S) if quality_mode else None
            mix = model.module.support_quality_head.mix if quality_mode else 1.0
            logit_mask, fusion_weights = fuse_support_logits(shot_logits, scores, mix)
            
        
        # if training:
        #     # 训练时，由于我们需要反向传播，建议并行处理所有 shot
        #     # 这里我们循环两次获取 protos 并在维度 1 拼接 (参考 VRP-SAM 拼接思想 )
        #     protos_list = []
        #     for s_idx in range(args.shot):
        #         p, _ = model(args.condition,batch['query_img'],batch['support_imgs'][:, s_idx],batch['support_masks'][:, s_idx],training)
        #         protos_list.append(p)
        #     protos = torch.cat(protos_list, dim=1) # [B, num_query * shot, C]
        #     # 训练时调用一次 sam_model
        #     logit_mask, _ = sam_model(batch['query_img'], batch['query_name'], protos)
        else:
            with torch.no_grad():
                B, S, C, H, W = batch['support_imgs'].shape

                # 1. 拍扁 Support: [B, S, C, H, W] -> [B*S, C, H, W]
                supp_imgs_flat = batch['support_imgs'].flatten(0, 1)
                supp_masks_flat = batch['support_masks'].flatten(0, 1)
                
                # 2. 复制 Query 以对齐并行数量 (必须用 repeat_interleave 保证顺序不乱)
                q_imgs_dummy = batch['query_img'].repeat_interleave(S, dim=0)
                # print(f"Debug: q_imgs_dummy shape after repeat_interleave: {q_imgs_dummy.shape}")  # [B*S, C, H, W]

                # 3. 此时所有输入都是标准的 4D 张量，送入模型
                outputs = model(args.condition, q_imgs_dummy, supp_imgs_flat,
                                supp_masks_flat, training,
                                dataset_names=support_dataset_names, **quality_kwargs)
                p_all, bbox_token = outputs[:2]

                # 3a. 拼接 50 + 2 → 52 tokens 作为 SAM sparse prompt（与训练一致）
                protos_all = torch.cat([p_all, bbox_token], dim=1)   # [B*S, 52, C]

                # 3b. 验证阶段同样计算 bbox token loss（仅监控用，不反传）
                q_masks_dummy = batch['query_mask'].repeat_interleave(S, dim=0)
                loss_bbox, cos_sim = model.module.compute_bbox_token_loss(bbox_token, q_masks_dummy)
                bbox_cos_buf.append(cos_sim.item())

                # 4. 调用 SAM 模型（不传 coarse_mask —— cascaded post-refinement 撤掉，结论：负效果）
                shot_logits = decode_support_logits(
                    sam_model, batch['query_img'], batch['query_name'], protos_all, S)
                scores = outputs[3].reshape(B, S) if quality_mode else None
                mix = model.module.support_quality_head.mix if quality_mode else 1.0
                logit_mask, fusion_weights = fuse_support_logits(shot_logits, scores, mix)
                
                if model.module.use_original_imgsize:  # 注意检查你的模型是否用这个属性
                    org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), 
                                            batch['org_query_imsize'][0].item()])
                    logit_mask = F.interpolate(logit_mask.unsqueeze(1), 
                                            org_qry_imsize, 
                                            mode='bilinear', 
                                            align_corners=True).squeeze(1)
            
            
        # low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos)
        # logit_mask = low_masks
        
        pred_mask = torch.sigmoid(logit_mask) > 0.5
        pred_mask = pred_mask.float()

        # 总损失 = 分割损失 (BCE + Dice) + λ * bbox token MSE 损失
        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        loss = loss + args.bbox_loss_weight * loss_bbox
        if quality_mode:
            loss_quality, shot_dice, quality_valid = support_quality_loss(
                scores, shot_logits, batch['query_mask'], args.fusion_target_temperature)
            loss = loss + args.fusion_quality_weight * loss_quality
            with torch.no_grad():
                weights = fusion_weights.detach()
                entropy = -(weights * weights.clamp_min(1e-8).log()).sum(1)
                entropy = entropy / math.log(S) if S > 1 else entropy * 0
                selected_dice = shot_dice.gather(1, weights.argmax(1)[:, None])[:, 0]
                valid = quality_valid.float()
                totals = torch.stack([
                    loss_quality.detach() * B, shot_logits.new_tensor(B),
                    entropy.sum(), weights.max(1).values.sum(),
                    (selected_dice * valid).sum(), (shot_dice.max(1).values * valid).sum(),
                    valid.sum()])
                totals = torch.cat([totals, weights.sum(0)]).double()
                if fusion_totals is None:
                    fusion_totals = totals
                else:
                    fusion_totals += totals
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # 每 5 个 epoch 打印一次 bbox token 监控信息（含权重，便于复现）
    if fusion_totals is not None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(fusion_totals)
        if utils.is_main_process():
            total = fusion_totals.cpu().tolist()
            count, valid_count = max(total[1], 1), max(total[6], 1)
            phase = split_name or ('train' if training else 'valid')
            metrics = dict(quality_loss=total[0] / count,
                           entropy=total[2] / count, max_weight=total[3] / count,
                           selected_dice=100 * total[4] / valid_count,
                           best_shot_dice=100 * total[5] / valid_count)
            weight_means = [v / count for v in total[7:]]
            Logger.info(f'[Support fusion {phase}] epoch={epoch} mode=quality '
                        f'mix={float(model.module.support_quality_head.mix):.3f} '
                        f'quality_loss={metrics["quality_loss"]:.5f} '
                        f'entropy={metrics["entropy"]:.3f} max_weight={metrics["max_weight"]:.3f} '
                        f'mean_weights={[round(v, 3) for v in weight_means]} '
                        f'selected_Dice={metrics["selected_dice"]:.2f} '
                        f'best_shot_Dice={metrics["best_shot_dice"]:.2f} '
                        f'valid_queries={int(total[6])}/{int(total[1])}')
            writer = getattr(Logger, 'tbd_writer', None)
            if writer is not None:
                writer.add_scalars(f'fusion/{phase}', metrics, epoch)

    if epoch % 5 == 0 and utils.is_main_process():
        split_name_ = 'Training' if training else 'Validation'
        if bbox_cos_buf:
            Logger.info(f'[BboxToken {split_name_}] epoch {epoch}: '
                        f'weight={args.bbox_loss_weight:.4f} '
                        f'cos avg={sum(bbox_cos_buf) / len(bbox_cos_buf):.4f}')

    average_meter.write_result(
        split_name or ('Training' if training else 'Validation'), epoch,
        class_names=getattr(dataloader.dataset, 'class_names', None),
        print_per_class=not training,
    )
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()
    dice = average_meter.compute_dice()

    return avg_loss, miou, fb_iou, dice


def validate_joint(args, epoch, model, sam_model, val_loaders):
    r""" joint 模式验证：每个子数据集跑一遍 train(training=False)，分别出 Dice，
    最后算 macro Dice（主指标）；mIoU/FB-IoU 保留作参考。

    val_loaders: OrderedDict[(name, datapath)] -> DataLoader

    返回：
        macro_loss, macro_dice, macro_miou, macro_fb_iou, per_dataset
    """
    per_dataset = {}
    miou_list, fb_list, loss_list, dice_list = [], [], [], []

    for (name, _path), loader in val_loaders.items():
        # btcv2 → btcv 显示别名（两个 key 指向同一个 BTCV 数据集，区别在 split/fold 协议）
        display_name = name.replace('btcv2', 'btcv')
        if utils.is_main_process():
            Logger.info(f'[val] [{display_name}] starting...')
        with torch.no_grad():
            l, m, fb, d = train(args, epoch, model, sam_model, loader,
                                optimizer=None, scheduler=None, training=False,
                                split_name=f'val/{display_name}')
        per_dataset[name] = {'loss': l, 'miou': m, 'fb_iou': fb, 'dice': d}
        miou_list.append(m)
        fb_list.append(fb)
        loss_list.append(l)
        dice_list.append(d)

    macro_loss = sum(loss_list) / max(len(loss_list), 1)
    macro_miou = sum(miou_list) / max(len(miou_list), 1)
    macro_fb_iou = sum(fb_list) / max(len(fb_list), 1)
    macro_dice = sum(dice_list) / max(len(dice_list), 1)

    if utils.is_main_process():
        for name, d in per_dataset.items():
            display_name = name.replace('btcv2', 'btcv')
            Logger.info(f'[val] [{display_name}] Dice={d["dice"]:.2f}  mIoU={d["miou"]:.2f}  FB-IoU={d["fb_iou"]:.2f}  Loss={d["loss"]:.4f}')
        Logger.info(f'[val] [macro] Dice={macro_dice:.2f}  mIoU={macro_miou:.2f}  FB-IoU={macro_fb_iou:.2f}  Loss={macro_loss:.4f}')
        Logger.tbd_writer.add_scalars(
            'val/joint',
            {'loss': macro_loss, 'dice': macro_dice, 'miou': macro_miou, 'fb_iou': macro_fb_iou},
            epoch,
        )

    return macro_loss, macro_dice, macro_miou, macro_fb_iou, per_dataset


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/root/paddlejob/workspace/env_run/datsets/')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'fss', 'btcv', 'btcv2', 'amos', 'papila', 'joint'])
    parser.add_argument('--joint-subdatasets', type=str, default='',
                        help='联合训练子数据集列表，逗号分隔，格式：name:datapath[,name:datapath,...]。仅当 --benchmark joint 时生效。')
    parser.add_argument('--n-episodes', type=int, default=4000,
                        help='joint 训练每个 epoch 的 episode 总数（先按权重选数据集，再从该数据集随机抽样本）')
    parser.add_argument('--joint-weights', type=str, default='',
                        help='joint 各子数据集采样权重，逗号分隔（如 1,1,1,1）；留空则均匀。长度须等于子数据集数。')
    parser.add_argument('--neighbor-prob', type=float, default=0.0,
                        help='support 采样中同患者相邻切片占比（0=关闭；1.0=全部相邻，任务难度下界，诊断用）')
    parser.add_argument('--class-balance-prob', type=float, default=0.0,
                        help='Training-only probability of class-first query sampling: '
                             '0=original, 0.5=mixed, 1=uniform classes. Validation is unchanged.')
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=2) # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--condition', type=str, default='scribble', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101', 'convnext_tiny', 'RadImageNet'])
    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--bbox_loss_weight', type=float, default=0.5,
                        help='weight of the bbox token MSE loss (token-space supervision). 0 disables it.')
    parser.add_argument('--bbox-support-attn-mask', type=int, choices=[0, 1], default=0,
                        help='Exclude support background keys in bbox cross-attention; '
                             'the two support bbox tokens remain valid.')
    parser.add_argument('--support-mask-pooling', choices=['max', 'nearest', 'adam-only'],
                        default='max', help='Support mask resize: max for all datasets, '
                        'nearest for all, or max on ADAM and nearest elsewhere. Train/eval match.')
    parser.add_argument('--icl-support-attn-mask', type=int, choices=[0, 1], default=0,
                        help='Exclude support background keys in the 50-token prompt decoder; '
                             'empty supports fall back to unmasked attention.')
    parser.add_argument('--sam-encoder-precision', choices=['auto', 'fp32', 'bf16', 'fp16'],
                        default='auto',
                        help='Frozen SAM image encoder training precision: auto uses native BF16 '
                             'when supported, otherwise FP32. Validation always uses FP32.')
    parser.add_argument('--support-fusion', choices=['mean', 'quality'], default='mean',
                        help='Fuse per-support SAM logits with uniform or learned reliability weights.')
    parser.add_argument('--fusion-quality-weight', type=float, default=0.1)
    parser.add_argument('--fusion-target-temperature', type=float, default=0.2)
    parser.add_argument('--fusion-warmup-epochs', type=int, default=2)
    parser.add_argument('--fusion-ramp-epochs', type=int, default=3)
    args = parser.parse_args()
    if not 0.0 <= args.class_balance_prob <= 1.0:
        parser.error('--class-balance-prob must be between 0 and 1')
    if not math.isfinite(args.fusion_quality_weight) or args.fusion_quality_weight < 0:
        parser.error('--fusion-quality-weight must be finite and nonnegative')
    if not math.isfinite(args.fusion_target_temperature) or args.fusion_target_temperature <= 0:
        parser.error('--fusion-target-temperature must be finite and positive')
    if min(args.fusion_warmup_epochs, args.fusion_ramp_epochs) < 0:
        parser.error('Fusion warmup and ramp epochs must be nonnegative')
    # Distributed setting
    local_rank = args.local_rank
    
    # 调试
    print(f"Rank {os.environ.get('RANK')} | Local Rank {os.environ.get('LOCAL_RANK')} | Device Count: {torch.cuda.device_count()}")
    
    dist.init_process_group(backend='nccl')
    print('local_rank: ', local_rank)
    torch.cuda.set_device(local_rank)
    # device = torch.device('cuda', local_rank)
    # 如果 local_rank 是 -1，就使用 0 号设备
    device = torch.device('cuda', local_rank if local_rank != -1 else 0)
    
    if utils.is_main_process():
        Logger.initialize(args, training=True)
    utils.fix_randseed(args.seed)
    # Model initialization
    model = VRP_encoder(args, args.backbone, False)
    if utils.is_main_process():
        Logger.log_params(model)

    sam_model = SAM_pred(encoder_precision=args.sam_encoder_precision)
    if utils.is_main_process():
        eval_cache = os.path.join(
            os.environ.get('TBD_FEATS', '/data2/2025Zhutaotao/TBD/cache_features'),
            'eval_unaugmented_v1',
        )
        Logger.info(f'[SAM cache] train=online (augmented query), '
                    f'eval={eval_cache}, image encoder frozen, shared across shots')
        Logger.info(f'[Speed] SAM encoder train={sam_model.resolve_encoder_precision(device)} '
                    f'(requested={args.sam_encoder_precision}), eval=fp32; '
                    f'decoder/loss=fp32, unused layer4 skipped, attention need_weights=False')
    sam_model.to(device)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Device setup
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    sam_model = torch.nn.parallel.DistributedDataParallel(sam_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    for param in model.module.layer0.parameters():
        param.requires_grad = False
    for param in model.module.layer1.parameters():
        param.requires_grad = False
    for param in model.module.layer2.parameters():
        param.requires_grad = False
    for param in model.module.layer3.parameters():
        param.requires_grad = True
    for param in model.module.layer4.parameters():
        param.requires_grad = False
        
    # 收集所有需要优化的参数组
    param_groups = []

    # 1. 已有的特定模块（正常学习率）
    param_groups.append({'params': model.module.transformer_decoder.parameters()})
    param_groups.append({'params': model.module.downsample_query.parameters(), 'lr': args.lr})
    param_groups.append({'params': model.module.merge_1.parameters(), 'lr': args.lr})

    # 2. 微调参与预测的 layer3（低学习率）；layer4 仅保留供旧 checkpoint 加载
    param_groups.append({'params': model.module.layer3.parameters(), 'lr': args.lr * 0.01})

    # 3. SAM 模块（较低学习率）
    param_groups.append({'params': sam_model.module.parameters(), 'lr': args.lr * 0.1})

    # 4. Bbox 分支（bbox_transfer + corner_emb）—— 必须注册，否则 bbox branch 只算梯度不更新
    param_groups.append({'params': model.module.bbox_transfer.parameters(), 'lr': args.lr})
    param_groups.append({'params': model.module.corner_emb.parameters(), 'lr': args.lr})
    if model.module.support_quality_head is not None:
        param_groups.append({'params': model.module.support_quality_head.parameters(), 'lr': args.lr})
    
    # param_groups.append({'params': model.module.reduce_conv.parameters(), 'lr': args.lr * 0.001})
    # param_groups.append({'params': model.module.merge_conv.parameters(), 'lr': args.lr * 0.001})
    # param_groups.append({'params': model.module.query_tokens, 'lr': args.lr * 0.001})

    # prompt_generator 已被 E-ICL 替代（见 VRP_encoder.py:256 注释），无此属性
    # param_groups.append({
    #         'params': model.module.prompt_generator.parameters(),
    #         'lr': args.lr * 0.1   # 与 transformer_decoder 同量级
    #     })

    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    # 【启动检查】requires_grad=True 但未注册进 optimizer 的参数（主进程打印一次）
    if utils.is_main_process():
        optimized_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
        missing = [n for n, p in model.module.named_parameters()
                   if p.requires_grad and id(p) not in optimized_ids]
        print('[Optimizer] 可训练但未注册的参数:',
              missing if missing else '无 (全部已注册)')

    Evaluator.initialize(args)

    # Dataset initialization
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    # 解析 joint 子数据集列表（仅 benchmark=='joint' 时使用）
    joint_sub = None
    joint_weights = None
    if args.benchmark == 'joint':
        joint_sub = []
        for tok in args.joint_subdatasets.split(','):
            tok = tok.strip()
            if not tok:
                continue
            name, path = tok.split(':', 1)
            joint_sub.append((name.strip(), path.strip()))
        assert len(joint_sub) > 0, f"[main] --benchmark joint 需要非空 --joint-subdatasets，得到 {args.joint_subdatasets!r}"
        print(f"[main] joint 训练子数据集 ({len(joint_sub)} 个):", flush=True)
        for n, p in joint_sub:
            print(f"  - {n}: {p}", flush=True)

        # 采样权重：留空则均匀
        if args.joint_weights.strip():
            joint_weights = [float(x) for x in args.joint_weights.split(',') if x.strip()]
            assert len(joint_weights) == len(joint_sub), \
                f'--joint-weights 长度 {len(joint_weights)} != 子数据集数 {len(joint_sub)}'
            assert all(w > 0 for w in joint_weights), f'--joint-weights 必须全为正: {joint_weights}'
        else:
            joint_weights = None
        print(f"[main] joint episode 模式: n_episodes={args.n_episodes}, weights={joint_weights or '均匀'}", flush=True)

    # dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    # dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', shot=args.shot, joint_subdatasets=joint_sub,
                                                 n_episodes=args.n_episodes, sub_weights=joint_weights,
                                                 neighbor_prob=args.neighbor_prob,
                                                 class_balance_prob=args.class_balance_prob)
    if utils.is_main_process():
        pooling_rules = {'max': 'all=adaptive_max_pool2d', 'nearest': 'all=nearest',
                         'adam-only': 'ADAM=adaptive_max_pool2d, others=nearest'}
        Logger.info(f'[Support mask] policy={args.support_mask_pooling}; train/eval; '
                    f'{pooling_rules[args.support_mask_pooling]}')
        Logger.info(f'[Support fusion] mode={args.support_fusion}; per-shot SAM logits; '
                    f'quality_weight={args.fusion_quality_weight} '
                    f'target_temperature={args.fusion_target_temperature} '
                    f'warmup={args.fusion_warmup_epochs} ramp={args.fusion_ramp_epochs}; '
                    'query GT only supervises quality, never generates inference weights')
        Logger.info(f'[Support attention] bbox_mask={args.bbox_support_attn_mask}, '
                    f'icl_mask={args.icl_support_attn_mask}; train/eval; '
                    'bbox tokens always valid; ICL empty support=unmasked; '
                    'query attention and EICL A@M unchanged')
        datasets = getattr(dataloader_trn.dataset, 'sub_datasets', [dataloader_trn.dataset])
        for ds in datasets:
            available = getattr(ds, 'balanced_class_ids', ())
            names = {c: name for c, name in zip(ds.class_ids, getattr(ds, 'class_names', []))}
            pools = getattr(ds, 'class_query_pools', {})
            pool_sizes = ', '.join(f'{names.get(c, c)}={len(pools[c])}' for c in available)
            Logger.info(f'[Sampling] {ds.benchmark}: class-first probability='
                        f'{getattr(ds, "class_balance_prob", 0.0):.2f}, '
                        f'query pools: {pool_sizes or "original query-first"}; validation unchanged')
            if getattr(ds, 'support_sampling', None):
                Logger.info(f'[Support sampling] {ds.benchmark}: {ds.support_sampling}')

    # joint 模式 val 按子数据集分别构造 DataLoader（每个子集单独出 mIoU，最后算 macro）
    if args.benchmark == 'joint':
        from data.joint import DatasetJoint as _DJ
        val_loaders = _DJ.build_val_dataloaders(args.bsz, args.nworker, joint_sub, shot=args.shot,
                                                neighbor_prob=args.neighbor_prob)
    else:
        dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', shot=args.shot, joint_subdatasets=joint_sub, neighbor_prob=args.neighbor_prob)
        val_loaders = None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs * len(dataloader_trn))
    # Training
    best_val_dice = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.epochs):

        trn_loss, trn_miou, trn_fb_iou, trn_dice = train(args, epoch, model, sam_model, dataloader_trn, optimizer, scheduler, training=True)
        if val_loaders is not None:
            # joint 模式：每个子数据集跑一遍 val，返回 macro Dice（主指标）
            val_loss, val_dice, val_miou, val_fb_iou, per_dataset = validate_joint(args, epoch, model, sam_model, val_loaders)
        else:
            with torch.no_grad():
                val_loss, val_miou, val_fb_iou, val_dice = train(args, epoch, model, sam_model, dataloader_val, optimizer, scheduler, training=False)
            per_dataset = None

        # Save the best model（Dice 为主指标）
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            if utils.is_main_process():
                Logger.save_model_dice(model, sam_model, epoch, val_dice)
        if utils.is_main_process():
            trn_dict = {'trn_loss': trn_loss, 'val_loss': val_loss}
            miou_dict = {'trn_miou': trn_miou, 'val_miou': val_miou}
            fb_dict = {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}
            dice_dict = {'trn_dice': trn_dice, 'val_dice': val_dice}
            if per_dataset is not None:
                for name, mdict in per_dataset.items():
                    miou_dict[f'val/{name}_miou'] = mdict['miou']
                    fb_dict[f'val/{name}_fb_iou'] = mdict['fb_iou']
                    dice_dict[f'val/{name}_dice'] = mdict['dice']
                    trn_dict[f'val/{name}_loss'] = mdict['loss']
            Logger.tbd_writer.add_scalars('data/loss', trn_dict, epoch)
            Logger.tbd_writer.add_scalars('data/miou', miou_dict, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', fb_dict, epoch)
            Logger.tbd_writer.add_scalars('data/dice', dice_dict, epoch)
            Logger.tbd_writer.flush()
    if utils.is_main_process():
        Logger.tbd_writer.close()
    beijing_now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)
    Logger.info('[End] 训练结束: %s (北京时间)' % beijing_now.strftime('%Y-%m-%d %H:%M:%S'))
    Logger.info('==================== Finished Training ====================')
