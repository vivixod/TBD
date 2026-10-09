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

from model.VRP_encoder import VRP_encoder
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM2pred import SAM_pred


def train(args, epoch, model, sam_model, dataloader, optimizer, scheduler, training, split_name=None):
    r""" Train VRP_encoder model """

    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)
    bbox_cos_buf = []   # 每 batch 的 bbox token cos 相似度，epoch 末打印均值

    for idx, batch in enumerate(dataloader):
        
        batch = utils.to_cuda(batch)

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
            q_names_dummy = [name for name in batch['query_name'] for _ in range(S)]  # 将名称重复 S 次

            # 3. 此时所有输入都是标准的 4D 张量，送入模型
            p_all, bbox_token, _ = model(args.condition, q_imgs_dummy, supp_imgs_flat, supp_masks_flat, training)

            # 3a. 拼接 50 上下文 tokens + 2 bbox tokens → 52 tokens 作为 SAM sparse prompt
            #     （coarse_mask 不传 SAM —— 该 attn bias 为试错后弃用）
            protos_all = torch.cat([p_all, bbox_token], dim=1)   # [B*S, 52, C]

            # 3b. Bbox token 空间监督：MSE(预测框 token, GT查询框的SAM编码)
            #     bbox_token 是 [B*S, 2, C]，需把 query_mask 同样复制 S 次对齐
            q_masks_dummy = batch['query_mask'].repeat_interleave(S, dim=0)
            loss_bbox, cos_sim = model.module.compute_bbox_token_loss(bbox_token, q_masks_dummy)
            bbox_cos_buf.append(cos_sim.item())

            # 4. 调用 SAM 模型（不传 coarse_mask —— cascaded post-refinement 撤掉，结论：负效果）
            logit_mask, _ = sam_model(q_imgs_dummy, q_names_dummy, protos_all)

            logit_mask = logit_mask.view(B, S, H, W)   # [B, S, H, W]
            logit_mask = logit_mask.mean(dim=1)           # [B, H, W]  简单平均即可
            
        
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
                q_names_dummy = [name for name in batch['query_name'] for _ in range(S)]  # 将名称重复 S 次

                # 3. 此时所有输入都是标准的 4D 张量，送入模型
                p_all, bbox_token, _ = model(args.condition, q_imgs_dummy, supp_imgs_flat, supp_masks_flat, training)

                # 3a. 拼接 50 + 2 → 52 tokens 作为 SAM sparse prompt（与训练一致）
                protos_all = torch.cat([p_all, bbox_token], dim=1)   # [B*S, 52, C]

                # 3b. 验证阶段同样计算 bbox token loss（仅监控用，不反传）
                q_masks_dummy = batch['query_mask'].repeat_interleave(S, dim=0)
                loss_bbox, cos_sim = model.module.compute_bbox_token_loss(bbox_token, q_masks_dummy)
                bbox_cos_buf.append(cos_sim.item())

                # 4. 调用 SAM 模型（不传 coarse_mask —— cascaded post-refinement 撤掉，结论：负效果）
                logit_mask, _ = sam_model(q_imgs_dummy, q_names_dummy, protos_all)

                logit_mask = logit_mask.view(B, S, H, W)   # [B, S, H, W]
                logit_mask = logit_mask.mean(dim=1)           # [B, H, W]  简单平均即可
                
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
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # 每 5 个 epoch 打印一次 bbox token 监控信息（含权重，便于复现）
    if epoch % 5 == 0 and utils.is_main_process():
        split_name_ = 'Training' if training else 'Validation'
        if bbox_cos_buf:
            Logger.info(f'[BboxToken {split_name_}] epoch {epoch}: '
                        f'weight={args.bbox_loss_weight:.4f} '
                        f'cos avg={sum(bbox_cos_buf) / len(bbox_cos_buf):.4f}')

    average_meter.write_result(split_name or ('Training' if training else 'Validation'), epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


def validate_joint(args, epoch, model, sam_model, val_loaders):
    r""" joint 模式验证：每个子数据集跑一遍 train(training=False)，分别出 mIoU，
    最后算 macro mIoU。

    val_loaders: OrderedDict[(name, datapath)] -> DataLoader

    返回：
        macro_loss, macro_miou, macro_fb_iou, per_dataset
    """
    per_dataset = {}
    miou_list, fb_list, loss_list = [], [], []

    for (name, _path), loader in val_loaders.items():
        # btcv2 → btcv 显示别名（两个 key 指向同一个 BTCV 数据集，区别在 split/fold 协议）
        display_name = name.replace('btcv2', 'btcv')
        if utils.is_main_process():
            Logger.info(f'[val] [{display_name}] starting...')
        with torch.no_grad():
            l, m, fb = train(args, epoch, model, sam_model, loader,
                             optimizer=None, scheduler=None, training=False,
                             split_name=f'val/{display_name}')
        per_dataset[name] = {'loss': l, 'miou': m, 'fb_iou': fb}
        miou_list.append(m)
        fb_list.append(fb)
        loss_list.append(l)

    macro_loss = sum(loss_list) / max(len(loss_list), 1)
    macro_miou = sum(miou_list) / max(len(miou_list), 1)
    macro_fb_iou = sum(fb_list) / max(len(fb_list), 1)

    if utils.is_main_process():
        for name, d in per_dataset.items():
            display_name = name.replace('btcv2', 'btcv')
            Logger.info(f'[val] [{display_name}] macro mIoU={d["miou"]:.2f}  FB-IoU={d["fb_iou"]:.2f}  Loss={d["loss"]:.4f}')
        Logger.info(f'[val] [macro] mIoU={macro_miou:.2f}  FB-IoU={macro_fb_iou:.2f}  Loss={macro_loss:.4f}')
        Logger.tbd_writer.add_scalars(
            'val/joint',
            {'loss': macro_loss, 'miou': macro_miou, 'fb_iou': macro_fb_iou},
            epoch,
        )

    return macro_loss, macro_miou, macro_fb_iou, per_dataset


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/root/paddlejob/workspace/env_run/datsets/')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'fss', 'btcv', 'btcv2', 'amos', 'joint'])
    parser.add_argument('--joint-subdatasets', type=str, default='',
                        help='联合训练子数据集列表，逗号分隔，格式：name:datapath[,name:datapath,...]。仅当 --benchmark joint 时生效。')
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
    args = parser.parse_args()
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

    sam_model = SAM_pred()
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
        param.requires_grad = True
        
    # 收集所有需要优化的参数组
    param_groups = []

    # 1. 已有的特定模块（正常学习率）
    param_groups.append({'params': model.module.transformer_decoder.parameters()})
    param_groups.append({'params': model.module.downsample_query.parameters(), 'lr': args.lr})
    param_groups.append({'params': model.module.merge_1.parameters(), 'lr': args.lr})

    # 2. 微调 backbone 最后两层（低学习率）
    param_groups.append({'params': model.module.layer3.parameters(), 'lr': args.lr * 0.01})
    param_groups.append({'params': model.module.layer4.parameters(), 'lr': args.lr * 0.01})

    # 3. SAM 模块（较低学习率）
    param_groups.append({'params': sam_model.module.parameters(), 'lr': args.lr * 0.1})

    # 4. Bbox 分支（bbox_transfer + corner_emb）—— 必须注册，否则 bbox branch 只算梯度不更新
    param_groups.append({'params': model.module.bbox_transfer.parameters(), 'lr': args.lr})
    param_groups.append({'params': model.module.corner_emb.parameters(), 'lr': args.lr})
    
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

    # dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    # dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', shot=args.shot, joint_subdatasets=joint_sub)

    # joint 模式 val 按子数据集分别构造 DataLoader（每个子集单独出 mIoU，最后算 macro）
    if args.benchmark == 'joint':
        from data.joint import DatasetJoint as _DJ
        val_loaders = _DJ.build_val_dataloaders(args.bsz, args.nworker, joint_sub, shot=args.shot)
    else:
        dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', shot=args.shot, joint_subdatasets=joint_sub)
        val_loaders = None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs * len(dataloader_trn))
    # Training
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.epochs):

        trn_loss, trn_miou, trn_fb_iou = train(args, epoch, model, sam_model, dataloader_trn, optimizer, scheduler, training=True)
        if val_loaders is not None:
            # joint 模式：每个子数据集跑一遍 val，返回 macro
            val_loss, val_miou, val_fb_iou, per_dataset = validate_joint(args, epoch, model, sam_model, val_loaders)
        else:
            with torch.no_grad():
                val_loss, val_miou, val_fb_iou = train(args, epoch, model, sam_model, dataloader_val, optimizer, scheduler, training=False)
            per_dataset = None

        # Save the best model (joint 模式用 macro mIoU，单 dataset 模式用 val_miou)
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            if utils.is_main_process():
                Logger.save_model_miou(model, sam_model, epoch, val_miou)
        if utils.is_main_process():
            trn_dict = {'trn_loss': trn_loss, 'val_loss': val_loss}
            miou_dict = {'trn_miou': trn_miou, 'val_miou': val_miou}
            fb_dict = {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}
            if per_dataset is not None:
                for name, mdict in per_dataset.items():
                    miou_dict[f'val/{name}_miou'] = mdict['miou']
                    fb_dict[f'val/{name}_fb_iou'] = mdict['fb_iou']
                    trn_dict[f'val/{name}_loss'] = mdict['loss']
            Logger.tbd_writer.add_scalars('data/loss', trn_dict, epoch)
            Logger.tbd_writer.add_scalars('data/miou', miou_dict, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', fb_dict, epoch)
            Logger.tbd_writer.flush()
    if utils.is_main_process():
        Logger.tbd_writer.close()
    beijing_now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)
    Logger.info('[End] 训练结束: %s (北京时间)' % beijing_now.strftime('%Y-%m-%d %H:%M:%S'))
    Logger.info('==================== Finished Training ====================')