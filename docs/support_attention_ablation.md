# Support attention 屏蔽消融

所有组均保留 max pooling 前景掩码、class_balance_prob=0.5、原数据集权重、
4-shot、seed=321、8000 episodes/epoch、300 轮余弦学习率调度和逐轮验证。
两个新开关默认均为 0，继续使用当前 max-pooling 基线。

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

非基线组日志名自动追加 `_bboxm1_iclm0` 等后缀，可用 `EXPERIMENT_NAME` 自定义。
日志参数及 `[Support attention]` 行记录两个开关，训练和验证共用同一设置。

实现范围：bbox 分支的 Support Cross Attention 排除背景空间 K/V，始终保留
两个 support bbox tokens；空前景时仅读取这两个 token。ICL prompt decoder
的 support Cross Attention 同样排除背景；空前景样本回退到原来的未屏蔽 attention，
避免所有位置被屏蔽导致非有限值。query Cross Attention、self-attention 和
EICL 相似度传播 `pseudo=A@M` 维持原计算，bbox token MSE 监督维持原设置。

没有增加模型参数，原 state_dict 可严格加载。开关属于运行配置，不保存在参数张量中；
用 `test.py` 加载权重时，需显式传入与训练一致的
`--bbox-support-attn-mask 0/1 --icl-support-attn-mask 0/1`。
`test.py` 原有的多 shot 聚合方式与训练内验证不同，消融指标应统一取训练内验证，
或先单独统一独立测试协议。

建议先对比相同 epoch 0–19 的 ADAM 未治疗动脉瘤 Dice、最近五轮均值及 BTCV 小器官；
保留 `--epochs 300`，在第 20 轮取中间结果，以保持相同学习率进程。
以上命令各启动一个三卡训练进程，请按自己的 GPU 排期分别运行。
