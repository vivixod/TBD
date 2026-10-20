# 四数据集 support 评分诊断

`diagnose_support_quality.py` 只做诊断，不改变模型、优化器或正在运行的训练。
它强制使用 CPU FP32 和已经存在的 `eval_unaugmented_v1` SAM 特征，
严格加载 checkpoint 中的 prompt 模型、SAM prompt encoder 和 mask decoder。
不初始化 CUDA，不重新计算或修改 SAM 图像缓存。

```bash
conda activate vrpsam
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 \
python diagnose_support_quality.py \
  --checkpoint logs/trn_joint4_shot4_cb0.5_poolmax_bboxm1_iclm1_fusionquality_qw0.1_qt0.2_fw2_fr3.log/best_model.pt \
  --output analysis/support_quality_audit \
  --validation-ranks 2 --queries-per-class 12 --workers 4 --threads 6
```

输出目录必须为空。启动时保存固定 checkpoint 快照和 SHA256，
从 checkpoint 配置读取池化、attention、shot、seed、相邻 support 等参数。
按当前验证 rank 数回放每个数据集的 1000 episodes，保存实际抽到的 support。
随后去重 query/class，并在推理之前按患者和目标大小选取每类最多 12 个 query。
覆盖 BTCV 13 类、AMOS 13 类、ADAM 2 类、BraTS 3 类。
AMOS 按当前数据集的有效类别列表采样，排除膀胱和前列腺/子宫；历史 15 类诊断需单独报告。

明确排除 query 自己作为 support、缩放后 query GT 为空、SAM 缓存缺失的候选；
每类的排除数量、实际样本和患者数记录在 `coverage.csv`。
当前验证入口在长列表上仅访问前 1000 个索引，这个诊断保留该覆盖范围，
不代表完整验证患者或全量类别均衡评测。稀有类可能只有少量 query、一个患者。

每个 episode 只运行一次四路预测，基于同一组 FP32 logits 比较：

- 最高评分一路的 Dice，与随机选一路的期望 Dice（四路 Dice 的均值）。
- 学习权重融合，与四路等权平均 logits 后的 Dice。
- 四路 Dice 的最大减最小差距、四路全失败比例、实际最大权重及权重熵。
- 当前 teacher 的权重分布，以及预测分布和均匀分布各自对 teacher 的 KL。

Dice 数值和差值使用百分数/百分点。四路差距至少 1 个百分点时才统计有区分度的
top1 命中；差距至少 10 个百分点作为预先指定的明显差异子集。
pairwise 排序只比较 Dice 差距至少 1 个百分点的 support 对，随机基准为 50%。
预测评分并列时平均这些候选的单路 Dice，不通过位置打破平局；
GT 最优 Dice 并列时，随机 top1 基准按并列数量修正。
四路全部失败保留在融合结果中，但不计入有区分度的排序指标。

`dataset_summary.csv` 和报告中的 Dice 是逐 query 的均值，
不是训练日志中先汇总像素再按类别平均的 Dice。
`class_summary.csv` 另存每类 pooled Dice，但同样仅对应本次固定子集。
患者聚类 bootstrap 给出加权减等权、选中减随机的 95% 区间；
只有一个患者的组不报告区间，不能据此判断泛化效果。
GT 最好单路只用于诊断，不是多路融合性能的上限。

主要输出：

- `manifest.json` / `run.json`：固定 query/support、配置、checkpoint 来源与快照摘要。
- `shots.csv`：四路各自的 Dice、原始评分、实际融合权重及 teacher 权重。
- `episodes.csv`：单个 query 内的质量差距、排序、加权和等权的配对结果。
- `class_summary.csv` / `dataset_summary.csv`：逐类和逐数据集统计。
- `results.json`：完整数值；`report.md` / `support_quality.png`：汇总表和散点图。

分开准备、推理、重建报告：

```bash
# 相同命令加 --prepare-only：只固定 checkpoint 和样本清单。
python diagnose_support_quality.py --output analysis/support_quality_audit \
  --run-prepared --workers 4 --threads 6
python diagnose_support_quality.py --output analysis/support_quality_audit --summarize-only
```
