# 三项小目标诊断

独立脚本 `diagnose_small_targets.py`，不修改模型或训练过程，不替换 bbox tokens。
范围：ADAM 未治疗动脉瘤、BTCV 双侧肾上腺及门静脉。

## 固定样本与训练配置

脚本从 checkpoint 同目录的 `log.txt` 读取池化策略、两处 attention 开关、shot、
neighbor_prob 和 seed。重建三卡验证的抽样顺序，每个 query/class 只取一个实际抽到的
support 组合，优先排除 query 自身作为 support 的组合，并按目标面积分层选取。
默认 ADAM 最多 17 张 query，BTCV 每类最多 8 张；结果是诊断子集，不代表完整验证集成绩。
生成的 `manifest.json` 可通过 `--manifest` 用于其他 checkpoint，保证 query/support 相同。

```bash
conda activate vrpsam
cd /home/zhutaotao/zhutaotao/TBD

# 只准备样本清单，无需 GPU；输出目录必须不存在。
python diagnose_small_targets.py \
  --checkpoint logs/trn_joint4_shot4_cb0.5_pooladam-only_bboxm1_iclm1.log/best_model.pt \
  --output analysis/small_targets_manifest --prepare-only

# 先每类一张（前四个样本），检查输出；只选一张 GPU。
CUDA_VISIBLE_DEVICES=0 python diagnose_small_targets.py \
  --checkpoint logs/trn_joint4_shot4_cb0.5_pooladam-only_bboxm1_iclm1.log/best_model.pt \
  --manifest analysis/small_targets_manifest/manifest.json \
  --output analysis/small_targets_smoke --limit 4

# 完整诊断。每个 episode 的三类结果都完成后立即写出，可查看进度。
CUDA_VISIBLE_DEVICES=0 python diagnose_small_targets.py \
  --checkpoint logs/trn_joint4_shot4_cb0.5_pooladam-only_bboxm1_iclm1.log/best_model.pt \
  --manifest analysis/small_targets_manifest/manifest.json \
  --output analysis/small_targets_full
```

GPU 默认显存上限为总显存的 25%，可用 `--gpu-memory-fraction` 调整。评估使用 FP32。
原图可读取已有 `eval_unaugmented_v1` 缓存；裁剪视图使用本次输出目录里的独立缓存，
重新编码 query，不读取原图缓存。模型权重加载后保存一份固定快照，避免训练覆盖 best_model
影响本次诊断。

## 输出

- `episodes.csv`：完整图、两档局部裁剪，以及各裁剪对应的原图 ROI 限制对照。
- `shots.csv`：每个 support 的 Dice/precision/recall、前景面积和有效格数。
- `localization.csv`：原始 pseudo 范围、峰值命中、到 GT 中心的距离、top 1% 格子的覆盖率。
  GT 用 max pooling 投影到特征网格；完全常数热图的定位指标留空。
- `summary.csv` / `report.md`：各类的平均 Dice、汇总像素 Dice、召回率、空预测率。
- 每个 episode 的 `overview.png`、`supports.png`、`crops.png` 和 `arrays.npz`。

正常推理保留所有 50 个上下文 tokens 和两个由 query cross-attention 产生的 bbox tokens，
四个 support 独立预测后平均 **logits**，与训练内验证一致。

局部放大使用 GT query/support 框，默认边长为目标框最大边的 2 倍，最小为模型输入尺度
的 64 / 128 像素，保留周围上下文；从原始 PNG 裁剪后再缩到 512，重新运行完整模型和
SAM image encoder。预测映回原图再计算指标，同时输出只屏蔽原图预测 ROI 外区域的对照。
该诊断同时改变目标尺度及局部上下文，不能把收益完全归因于 query 分辨率，也不能作为
自动定位的正式成绩。单个 support 最优值同样只用于诊断聚合损失，不是正式预测策略。
