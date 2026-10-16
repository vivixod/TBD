# Support attention 屏蔽消融

所有组均保留 max pooling 前景掩码、class_balance_prob=0.5、原数据集权重、
4-shot、seed=321、8000 episodes/epoch、300 轮余弦学习率调度和逐轮验证。
两个新开关默认均为 0，继续使用当前 max-pooling 基线。

当前推荐的池化设置恢复为所有数据集 `max`。沿用最近实验的两处 attention
屏蔽设置时，从 TBD 目录运行：

```bash
SUPPORT_MASK_POOLING=max BBOX_SUPPORT_ATTN_MASK=1 ICL_SUPPORT_ATTN_MASK=1 bash train_joint.sh
```

新日志名明确包含池化策略：`trn_joint4_shot4_cb0.5_poolmax_bboxm1_iclm1.log`。
启动日志应显示 `[Support mask] policy=max; train/eval; all=adaptive_max_pool2d`。
显式指定 `max` 会覆盖终端之前导出的 `adam-only`。训练、验证共用该策略；
独立 `test.py` 的默认策略同样为 `max`。修改脚本不改变已经运行的训练进程。

| 组别 | BBOX_SUPPORT_ATTN_MASK | ICL_SUPPORT_ATTN_MASK |
|---|---:|---:|
| max pooling 基线 | 0 | 0 |
| bbox 分支屏蔽 | 1 | 0 |
| ICL prompt 分支屏蔽 | 0 | 1 |
| 两处均屏蔽 | 1 | 1 |

从 TBD 目录运行，按需选择一组：

```bash
# 当前 max pooling 基线；已有这组实验时直接使用其同 epoch 结果。
BBOX_SUPPORT_ATTN_MASK=0 ICL_SUPPORT_ATTN_MASK=0 bash train_joint.sh

# 仅 bbox 分支
BBOX_SUPPORT_ATTN_MASK=1 ICL_SUPPORT_ATTN_MASK=0 bash train_joint.sh

# 仅 ICL prompt 分支
BBOX_SUPPORT_ATTN_MASK=0 ICL_SUPPORT_ATTN_MASK=1 bash train_joint.sh

# 两处分支
BBOX_SUPPORT_ATTN_MASK=1 ICL_SUPPORT_ATTN_MASK=1 bash train_joint.sh
```

新实验日志名均包含 `_poolmax`（或实际池化策略），非基线组再追加
`_bboxm1_iclm0` 等后缀，可用 `EXPERIMENT_NAME` 自定义。既有日志目录保持原名。
日志参数及 `[Support attention]` 行记录两个开关，训练和验证共用同一设置。

实现范围：bbox 分支的 Support Cross Attention 排除背景空间 K/V，始终保留
两个 support bbox tokens；空前景时仅读取这两个 token。ICL prompt decoder
的 support Cross Attention 同样排除背景；空前景样本回退到原来的未屏蔽 attention，
避免所有位置被屏蔽导致非有限值。query Cross Attention、self-attention 和
EICL 相似度传播 `pseudo=A@M` 维持原计算，bbox token MSE 监督维持原设置。

没有增加模型参数，原 state_dict 可严格加载。开关属于运行配置，不保存在参数张量中；
用 `test.py` 加载权重时，需显式传入与训练一致的
`--bbox-support-attn-mask 0/1 --icl-support-attn-mask 0/1`。
`test.py` 的多 shot 聚合已统一为与训练内验证相同的逐support解码后平均logits。
当前消融仍统一使用 `train.py` 的valid分支，既有日志的评估含义保持一致。

建议先对比相同 epoch 0–19 的 ADAM 未治疗动脉瘤 Dice、最近五轮均值及 BTCV 小器官；
保留 `--epochs 300`，在第 20 轮取中间结果，以保持相同学习率进程。
以上命令各启动一个训练进程，GPU 和进程数使用 `train_joint.sh` 中的当前配置，请分别运行。

## 仅 ADAM 使用 max pooling

新增独立变量 `SUPPORT_MASK_POOLING`，默认 `max` 保持上述实验配置：

| 设置 | ADAM | BTCV / AMOS / BraTS |
|---|---|---|
| `max` | max pooling | max pooling |
| `nearest` | nearest | nearest |
| `adam-only` | max pooling | nearest |

先单独验证按数据集选择池化，两个 attention 开关均关闭：

```bash
SUPPORT_MASK_POOLING=adam-only BBOX_SUPPORT_ATTN_MASK=0 ICL_SUPPORT_ATTN_MASK=0 bash train_joint.sh
```

日志名为 `trn_joint4_shot4_cb0.5_pooladam-only`（Logger 会追加 `.log`，已有同名日志时加时间戳）。
后续可组合 attention 消融，例如：

```bash
SUPPORT_MASK_POOLING=adam-only BBOX_SUPPORT_ATTN_MASK=1 ICL_SUPPORT_ATTN_MASK=0 bash train_joint.sh
```

联合训练按每个 episode 的 `dataset_name` 选择策略；多 shot 展开时顺序与
`repeat_interleave(shot)` 一致。分数据集验证使用验证 dataset 的 benchmark，
因此 ADAM 验证同样使用 max，其余使用 nearest，混合 batch 也按样本选择。
独立 `test.py` 需带相同的 `--support-mask-pooling adam-only`。
其他数据集恢复 nearest 只恢复了掩码缩放规则；联合模型共享参数，其 Dice 不保证恢复到旧实验数值。
