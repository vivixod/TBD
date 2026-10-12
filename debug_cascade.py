r""" 诊断脚本：级联精修（Cascaded Post-refinement）的有效性

级联 = training-free 推理策略，只多跑两次轻量 decoder：
  Round 1: 从 stage-1 预测 mask 取正负点，再喂一次 decoder（attn bias 用当前 mask 更新）
  Round 2: 取 Round 1 精修 mask 的 bbox 作为 box prompt，再喂一次 decoder

用同一 checkpoint 对比：
  baseline : 现有推理（coarse_mask → attn bias，一次 decoder）≈ 47.7 mIoU
  cascade : baseline + 级联精修

用法（GPU 0）：
    /home/user/.conda/envs/ztt/bin/python debug_cascade.py \
        --load logs/trn_btcv_elcseg_attnguide2_shot1.log/best_model.pt
"""
import argparse

import torch
import torch.distributed as dist

from model.VRP_encoder import VRP_encoder
from common.logger import AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM2pred import SAM_pred


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
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--use_ignore', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=500, help='最多评估的 episode 数')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # 单卡 process group（DistributedSampler 依赖）
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23458',
                                rank=0, world_size=1)

    device = torch.device(f'cuda:{args.gpu}')
    # 覆盖所有裸调 .cuda() 默认走 cuda:0 的位置（utils.to_cuda + logger.AverageMeter）
    _orig_tensor_cuda = torch.Tensor.cuda
    def _tensor_cuda(self, *args, **kwargs):
        # 如果用户显式传 device，跳过；否则走目标 device
        if args and isinstance(args[0], (torch.device, int, str)):
            return _orig_tensor_cuda(self, *args, **kwargs)
        return _orig_tensor_cuda(self, device)
    torch.Tensor.cuda = _tensor_cuda
    _orig_to_cuda = utils.to_cuda
    def _to_cuda(batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda(device=device)
        return batch
    utils.to_cuda = _to_cuda
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

    meter_base = AverageMeter(dataloader.dataset)
    meter_cascade = AverageMeter(dataloader.dataset)

    print('==> 开始评估（baseline vs cascade）')
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

            # ---- baseline：现有推理（attn bias，一次 decoder）----
            lm_base, _ = sam_model(q_imgs_dummy, q_names_dummy, protos_all,
                                   coarse_mask=coarse_mask)
            lm_base = lm_base.view(B, S, H, W).mean(dim=1)
            pred_base = (torch.sigmoid(lm_base) > 0.5).float()
            ai, au = Evaluator.classify_prediction(pred_base.squeeze(1), batch)
            meter_base.update(ai, au, batch['class_id'], loss=None)

            # ---- cascade：baseline + 正负点 + bbox 级联精修 ----
            lm_cas, _ = sam_model(q_imgs_dummy, q_names_dummy, protos_all,
                                  coarse_mask=coarse_mask, use_cascade=True)
            lm_cas = lm_cas.view(B, S, H, W).mean(dim=1)
            pred_cas = (torch.sigmoid(lm_cas) > 0.5).float()
            ai2, au2 = Evaluator.classify_prediction(pred_cas.squeeze(1), batch)
            meter_cascade.update(ai2, au2, batch['class_id'], loss=None)

            if (idx + 1) % 50 == 0:
                print(f'  进度 {idx+1}/{args.max_iters}')

    # ---- 汇总 ----
    miou_base, fb_base = meter_base.compute_iou()
    miou_cas, fb_cas = meter_cascade.compute_iou()
    print('\n========== 级联精修 净贡献 ==========')
    print(f'  baseline : mIoU={miou_base.item():.2f}  FB-IoU={fb_base.item():.2f}')
    print(f'  cascade  : mIoU={miou_cas.item():.2f}  FB-IoU={fb_cas.item():.2f}')
    print(f'  净提升   : mIoU {(miou_cas - miou_base).item():+.2f}')


if __name__ == '__main__':
    main()
