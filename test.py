import os
import argparse
import torch
import torch.nn as nn
from model.VRP_encoder import VRP_encoder
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM2pred import SAM_pred

def test(args, model, sam_model, dataloader):
    r""" Test VRP_encoder model on a new organ (Spleen) """
    model.eval()
    average_meter = AverageMeter(dataloader.dataset)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = utils.to_cuda(batch)
            
            # 使用验证模式的逻辑：n-shot 预测
            # 注意：如果不是分布式训练加载的模型，model 可能没有 .module
            model_to_run = model.module if hasattr(model, 'module') else model
            _, _, logit_mask = model_to_run.predict_mask_nshot(args, batch, sam_model, nshot=args.shot)
            
            # 后处理
            pred_mask = (torch.sigmoid(logit_mask) > 0.5).float()

            # 计算指标 (Spleen 的 class_id 会根据 dataset 定义传入)
            area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
            
            # 每 10 个 batch 打印一次进度
            average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=10)

    # 打印最终结果
    average_meter.write_result('Test_Spleen', epoch=-1)
    miou, fb_iou = average_meter.compute_iou()
    return miou, fb_iou

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test on Spleen with Pretrained Liver Model')
    # 路径配置
    parser.add_argument('--datapath', type=str, default='/vcl/2025zhutaotao/BTCV')
    parser.add_argument('--load', type=str, default='/vcl/2025zhutaotao/VRP-SAM/logs/trn_btcv_elcseg_bbox_aug2.log/best_model.pt', help='Path to best_model.pt')
    parser.add_argument('--benchmark', type=str, default='btcv2')
    parser.add_argument('--bsz', type=int, default=1) # 测试建议用 1，方便观察结果
    parser.add_argument('--nworker', type=int, default=4)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--condition', type=str, default='mask')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_ignore', type=bool, default=True)
    parser.add_argument('--num_query', type=int, default=50)
    args = parser.parse_args()

    # 环境设置 (单卡测试)
    device = torch.device('cuda:0')
    utils.fix_randseed(args.seed)

    # 1. 模型初始化
    model = VRP_encoder(args, args.backbone, False)
    sam_model = SAM_pred()
    
    # 2. 加载权重（兼容 {model, sam_model} 打包格式与旧的裸 state_dict 格式）
    print(f"==> Loading weights from {args.load}")
    ckpt = torch.load(args.load, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model_ckpt, sam_ckpt = ckpt['model'], ckpt.get('sam_model')
    else:
        model_ckpt, sam_ckpt = ckpt, None
    new_state_dict = {k.replace('module.', ''): v for k, v in model_ckpt.items()}
    model.load_state_dict(new_state_dict)
    if sam_ckpt is not None:
        sam_state_dict = {k.replace('module.', ''): v for k, v in sam_ckpt.items()}
        sam_model.load_state_dict(sam_state_dict)
        print("==> SAM 微调权重已加载")
    else:
        print("==> 未找到 sam_model 权重，使用 SAM 官方预训练权重（结果会偏低）")
    
    model.to(device)
    sam_model.to(device)
    
    # 3. 数据集初始化（与训练保持一致）
    # 单卡测试也需要初始化 process group（DistributedSampler 依赖），测试结束进程即退出
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='gloo',
            init_method='tcp://127.0.0.1:23456',
            rank=0,
            world_size=1
        )
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    
    # 这里我们通过传入不同的 split 或 benchmark 参数来强制读取脾脏数据
    # 如果你的 Dataset 逻辑还没写好 spleen，请看下方的修改建议
    dataloader_test = FSSDataset.build_dataloader(
        benchmark=args.benchmark, 
        bsz=args.bsz, 
        nworker=args.nworker, 
        fold=0, 
        split='val', # 使用 val 逻辑
        shot=args.shot
    )

    Evaluator.initialize(args)
    Logger.initialize(args, training=False)

    # 4. 执行测试
    print("==> Starting Inference on Spleen...")
    miou, fb_iou = test(args, model, sam_model, dataloader_test)
    
    print(f"\n[Final Results on Spleen] mIoU: {miou:.2f} | FB-IoU: {fb_iou:.2f}")