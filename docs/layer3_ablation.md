# layer3 权重与 BN 统计消融

`LAYER3_MODE` 控制 ResNet layer3 的训练行为，普通联合训练和 PAPILA 联合训练共用此选项。

| 设置 | layer3 权重（含 BN γ/β） | BN 运行统计 |
|---|---|---|
| `finetune`（默认，复现旧实验） | 微调，学习率为主学习率的 0.01 倍 | 更新 |
| `finetune-frozen-bn` | 同上 | 固定 |
| `frozen` | 固定，从优化器排除 | 固定 |

冻结参数在 DDP 包装之前设置；每轮训练的 `train_mode()` 会重新设置正确的 BN 模式。
冻结 layer3 后，特征降维、prompt/bbox 分支、可选 support 评分头与 SAM 可训练组件仍继续学习。
模型参数键和总参数量不变，旧 checkpoint 可严格加载。

保持当前全局 max pooling、两处 attention 屏蔽和 quality 融合，单独启动冻结实验：

```bash
GPU_ID=1,2 LAYER3_MODE=frozen SUPPORT_MASK_POOLING=max \
BBOX_SUPPORT_ATTN_MASK=1 ICL_SUPPORT_ATTN_MASK=1 SUPPORT_FUSION=quality \
bash train_joint.sh
```

只固定 BN、继续微调 layer3 权重时，将 `LAYER3_MODE=frozen` 换成 `LAYER3_MODE=finetune-frozen-bn`。
复现原 layer3 训练行为使用 `LAYER3_MODE=finetune`。
`train_joint.sh` 和 `train_joint_papila.sh` 默认均使用 BTCV + AMOS MR + PAPILA + BraTS。
新模式日志名分别追加 `_l3frozen` / `_l3finetune-frozen-bn`；自定义 `EXPERIMENT_NAME` 时使用指定名称。
启动日志 `[Backbone]` 记录实际模式、layer3 可学习参数量和 BN 统计行为，checkpoint 配置同时保存模式。

建议先从相同预训练初始化、相同随机种子和其余配置比较这些模式。
从旧 checkpoint 冻结只会保留其已有权重和运行统计，不会恢复预训练统计。
评估使用训练文件内的 valid 分支，自动沿用模型设置；独立 eval 前向在三种模式下均使用保存的统计。
冻结可减少 layer3 的反向计算和同步，但整轮加速幅度需实测；验证频率不变。
