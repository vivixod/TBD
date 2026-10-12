r""" 诊断脚本：attention-guide（coarse_mask → bias）的有效性

1. 统计验证集上 coarse_mask 与 GT mask 的 IoU 分布（判断位置先验本身的质量）
2. 用同一 checkpoint 分别以 bias ON / OFF 跑验证集，对比 mIoU（量化 bias 净贡献）

用法（训练占着 GPU 1,2,3 时用 GPU 0 跑）：
    /home/user/.conda/envs/ztt/bin/python debug_attnguide.py \
        --load logs/trn_btcv_elcseg_attnguide2_shot1.log/best_model.pt
"""
import os
import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F

from model.VRP_encoder import VRP_encoder
from common.logger import AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM2pred import SAM_pred


def iou_of(bin_a, bin_b):
    """bin_a/bin_b: [N,H,W] 0/1"""
    inter = (bin_a * bin_b).sum(dim=(1, 2)).float()
    union = ((bin_a + bin_b) > 0).sum(dim=(1, 2)).float()
    return inter / union.clamp(min=1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='/vcl/2025zhutaotao/BTCV')
    parser.add_argument('--load', type=str,
                        default='/vcl/2025zhutaotao/VRP-SAM/logs/trn_btcv_elcseg_attnguide2_shot1.log/best_model.pt')
    parser.add_argument('--benchmark', type=str, default='btcv2')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--condition', type=str, default='mask')
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)  # 与训练时验证一致
    parser.add_argument('--use_ignore', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=500, help='最多评估的 episode 数')
    args = parser.parse_args()

    # 单卡 process group（DistributedSampler 依赖）
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23457',
                                rank=0, world_size=1)

    device = torch.device('cuda:0')
    utils.fix_randseed(args.seed)

    # ---- 模型与权重 ----
    model = VRP_encoder(args, args.backbone, False)
    sam_model = SAM_pred()
    print(f'==> 加载 {args.load}')
    ckpt = torch.load(args.load, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model'].items()})
        if ckpt.get('sam_model') is not None:
            sam_model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['sam_model'].items()}, strict=False)
            print('==> SAM 微调权重已加载')
        else:
            print('!! 无 sam_model 权重，用官方预训练')
    else:
        model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    model.to(device).eval()
    sam_model.to(device).eval()

    # ---- 数据 ----
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker,
                                             args.fold, 'val', shot=args.shot)
    Evaluator.initialize(args)

    meter_on = AverageMeter(dataloader.dataset)
    meter_off = AverageMeter(dataloader.dataset)
    coarse_ious = []

    print('==> 开始评估（bias ON/OFF + coarse_mask IoU）')
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= args.max_iters:
                break
            batch = utils.to_cuda(batch)
            B, S, C, H, W = batch['support_imgs'].shape
            supp_imgs_flat = batch['support_imgs'].flatten(0, 1)
            supp_masks_flat = batch['support_masks'].flatten(0, 1)
            q_imgs_dummy = batch['query_img'].repeat_interleave(S, dim=0)
            q_names_dummy = [n for n in batch['query_name'] for _ in range(S)]

            protos, bbox_token, coarse_mask = model(args.condition, q_imgs_dummy,
                                                    supp_imgs_flat, supp_masks_flat, False)
            protos_all = torch.cat([protos, bbox_token], dim=1)

            # ---- 1) coarse_mask 与 GT 的 IoU（都在 64×64 上比较）----
            cm = F.interpolate(coarse_mask.float(), size=(64, 64), mode='bilinear',
                               align_corners=True)                      # [B,1,64,64]
            cm_bin = (cm > 0.5).float().squeeze(1)                       # [B,64,64]
            gt = batch['query_mask'].unsqueeze(1).float()                 # [B,1,512,512]
            gt64 = F.interpolate(gt, size=(64, 64), mode='nearest').squeeze(1)
            coarse_ious.append(iou_of(cm_bin, gt64))

            # ---- 2) bias ON ----
            lm_on, _ = sam_model(q_imgs_dummy, q_names_dummy, protos_all, coarse_mask=coarse_mask)
            lm_on = lm_on.view(B, S, H, W).mean(dim=1)
            pred_on = (torch.sigmoid(lm_on) > 0.5).float()
            ai, au = Evaluator.classify_prediction(pred_on.squeeze(1), batch)
            meter_on.update(ai, au, batch['class_id'], loss=None)

            # ---- 3) bias OFF ----
            lm_off, _ = sam_model(q_imgs_dummy, q_names_dummy, protos_all)
            lm_off = lm_off.view(B, S, H, W).mean(dim=1)
            pred_off = (torch.sigmoid(lm_off) > 0.5).float()
            ai2, au2 = Evaluator.classify_prediction(pred_off.squeeze(1), batch)
            meter_off.update(ai2, au2, batch['class_id'], loss=None)

            if (idx + 1) % 50 == 0:
                print(f'  进度 {idx+1}/{args.max_iters}')

    # ---- 汇总 ----
    all_iou = torch.cat(coarse_ious)
    print('\n========== coarse_mask 质量（64×64 上与 GT 的 IoU）==========')
    print(f'  均值: {all_iou.mean().item():.4f}  中位: {all_iou.median().item():.4f}  '
          f'<0.3 占比: {(all_iou < 0.3).float().mean().item()*100:.1f}%  '
          f'>0.5 占比: {(all_iou > 0.5).float().mean().item()*100:.1f}%')

    miou_on, fb_on = meter_on.compute_iou()
    miou_off, fb_off = meter_off.compute_iou()
    print('\n========== bias 净贡献 ==========')
    print(f'  bias ON : mIoU={miou_on.item():.2f}  FB-IoU={fb_on.item():.2f}')
    print(f'  bias OFF: mIoU={miou_off.item():.2f}  FB-IoU={fb_off.item():.2f}')
    print(f'  净提升  : mIoU {(miou_on - miou_off).item():+.2f}')


if __name__ == '__main__':
    main()
