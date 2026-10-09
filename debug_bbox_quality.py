"""分析 bbox_head 预测质量：对比 pred_bbox 与 GT bbox 的 IoU、L1 误差。

用法:
    python debug_bbox_quality.py --logpath trn_btcv_elcseg_nobbox --epoch 72
"""

import os, argparse, sys
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))

from model.VRP_encoder import VRP_encoder
from model.base.bbox_tokenizer import extract_gt_bbox, bbox_to_sam_format
from common import utils
from data.dataset import FSSDataset


def bbox_iou(a, b):
    """两个 xyxy bbox 的 IoU。a, b: [4]"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def draw_bbox_on_image(img_pil, bbox_norm, color, width=3):
    """在 PIL 图像上画归一化 bbox。bbox_norm: [cx,cy,w,h] in [0,1]"""
    draw = ImageDraw.Draw(img_pil)
    W, H = img_pil.size
    cx, cy, w, h = bbox_norm.cpu() if torch.is_tensor(bbox_norm) else bbox_norm
    x1 = int((cx - w / 2) * W)
    y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W)
    y2 = int((cy + h / 2) * H)
    for i in range(width):
        draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=color)
    return img_pil


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # ---- 1. 加载模型 ----
    print(f"[1/4] 加载 checkpoint: logs/{args.logpath}/ ...")
    model = VRP_encoder(args, args.backbone, False)
    import glob
    # 兼容 logpath 带或不带 .log 后缀
    logdir = os.path.join('logs', args.logpath)
    if not os.path.isdir(logdir) and os.path.isdir(logdir + '.log'):
        logdir = logdir + '.log'
    candidates = glob.glob(os.path.join(logdir, '*.pt'))
    if not candidates:
        candidates = glob.glob(os.path.join(logdir, '*.pth'))
    if not candidates:
        print(f"❌ 找不到 checkpoint: {logdir}/")
        return
    ckpt_path = candidates[0]
    print(f"  使用: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    # 去掉 module. 前缀 (兼容 DDP)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # ---- 2. 加载数据 ----
    print(f"[2/4] 加载验证集 {args.benchmark} ...")
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:26789',
                            world_size=1, rank=0)
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader = FSSDataset.build_dataloader(args.benchmark, 1, args.nworker, args.fold, 'val', shot=args.shot)

    # ---- 3. 推理 + 分析 ----
    print(f"[3/4] 分析 bbox 质量 (最多 {args.max_samples} 样本) ...")
    iou_list, l1_list = [], []
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            if count >= args.max_samples:
                break
            batch = utils.to_cuda(batch)

            B, S = batch['support_imgs'].shape[:2]
            supp_imgs_flat = batch['support_imgs'].flatten(0, 1)
            supp_masks_flat = batch['support_masks'].flatten(0, 1)
            q_imgs_dummy = batch['query_img'].repeat_interleave(S, dim=0)

            # VRP_encoder forward
            protos, pred_bbox, _ = model(
                args.condition, q_imgs_dummy, supp_imgs_flat, supp_masks_flat, training=False
            )
            pred = pred_bbox[0].cpu()  # [4] normalised (cx,cy,w,h)

            # GT bbox
            gt = extract_gt_bbox(batch['query_mask'])[0].cpu()

            # 转 xyxy 像素坐标
            def to_xyxy(b, w=512, h=512):
                cx, cy, bw, bh = b
                return torch.tensor([
                    (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h
                ])
            iou = bbox_iou(to_xyxy(pred), to_xyxy(gt))
            l1 = F.l1_loss(pred, gt).item()
            iou_list.append(iou)
            l1_list.append(l1)

            # ---- 可视化：GT(绿) + Pred(红) 叠加在 query 图上 ----
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = batch['query_img'][0].cpu()
            img_vis = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            img_vis = (img_vis * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_vis)
            img_pil = draw_bbox_on_image(img_pil, gt, 'green', width=3)
            img_pil = draw_bbox_on_image(img_pil, pred, 'red', width=2)

            name = batch['query_name'][0].replace('/', '_').replace('.png', '')
            img_pil.save(os.path.join(save_dir, f'{count:03d}_{name}_iou{iou:.2f}.png'))

            count += 1
            if count % 10 == 0:
                print(f"  已处理 {count}/{args.max_samples}, "
                      f"avg IoU={np.mean(iou_list):.3f}, avg L1={np.mean(l1_list):.4f}")

    # ---- 4. 汇总 ----
    print(f"\n[4/4] ==== Bbox 质量汇总 (共 {count} 样本) ====")
    print(f"  IoU (pred vs GT):  mean={np.mean(iou_list):.4f}  "
          f"median={np.median(iou_list):.4f}  min={np.min(iou_list):.4f}")
    print(f"  L1  (pred vs GT):  mean={np.mean(l1_list):.4f}  "
          f"median={np.median(l1_list):.4f}")
    print(f"  IoU > 0.5: {sum(1 for x in iou_list if x > 0.5)}/{count} "
          f"({100*sum(1 for x in iou_list if x > 0.5)/count:.1f}%)")
    print(f"  IoU > 0.7: {sum(1 for x in iou_list if x > 0.7)}/{count} "
          f"({100*sum(1 for x in iou_list if x > 0.7)/count:.1f}%)")
    print(f"\n可视化保存到: {save_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bbox Quality Debug')
    parser.add_argument('--logpath', type=str, default='trn_btcv_elcseg_nobbox')
    parser.add_argument('--epoch', type=int, default=72)
    parser.add_argument('--datapath', type=str, default='/vcl/2025zhutaotao/BTCV')
    parser.add_argument('--benchmark', type=str, default='btcv2')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nworker', type=int, default=4)
    parser.add_argument('--condition', type=str, default='mask')
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--max_samples', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='./debug_bbox_output')
    args = parser.parse_args()
    main(args)
