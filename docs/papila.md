# PAPILA 接入与实验协议

已处理的输入位于 `/data/2025Zhutaotao/PAPILA`，也可把完整数据目录复制到另一台机器。
数据读取使用 `dataset.json` 里的相对路径，保留 RGB，不依赖不完整的 `imask/`。
本次只生成项目内的 split 列表，没有修改原始图片、标签或 JSON，也没有启动训练。

## 患者划分

本地共有 244 位患者、488 张眼底图片，每位患者左右眼各一张。
原 JSON 中 `training`/`test` 是图片级划分，有 74 位患者同时出现在两组。
因此合并 JSON 的全部有标签记录后，按患者重新划分；原 JSON 只用于查找文件路径。

固定 seed=321、患者比例 8:2、fold=0：

| 划分 | 患者 | 图片 | 视盘 query | 视杯 query |
| --- | ---: | ---: | ---: | ---: |
| train | 195 | 390 | 390 | 390 |
| val | 49 | 98 | 98 | 98 |

左右眼保持在同一组，训练与验证患者无交集。当前没有另设第三个独立测试集；
`split='test'` 与项目已有医学数据接口一致，读取 val 划分，不能作为额外独立测试结果。
患者名单、输入 JSON SHA256、类别数量和协议记录在
`data/splits/lists/papila/manifest.json`，另有 train/val 的查询列表和类别候选列表。

重现列表：

```bash
python data/splits/gen_papila_splits.py --datapath /data/2025Zhutaotao/PAPILA
```

脚本检查所有 488 张图片与标签的尺寸、二值通道和类别存在性。相同设置可重复执行；
若新设置会覆盖既有划分，脚本报错，可先用 `--output /tmp/papila_new_split` 检查。
训练使用项目内固定路径的列表，不会自动读取 `--output` 指定的试验目录。

## 独立 support 采样

1. query 采用现有 query-first / class-first 混合策略，支持 `CLASS_BALANCE_PROB`。
2. support 从 query 所在划分的同类别候选池采样：训练从 train，验证从 val。
3. 排除 query 的整位患者，包括另一只眼。
4. 4-shot 先无放回抽 4 位不同患者，再在每位患者中随机选一只含目标结构的眼睛。
5. 训练持续随机采样；验证按 seed/query/class 固定组合，与调用顺序和分配到哪张卡无关。
6. PAPILA 不使用相邻切片逻辑；即使联合训练传入 `--neighbor-prob 1.0`，PAPILA 也忽略它。
7. 不足 shot 位其他患者时直接报错，不复制 support，不以 query 自己补齐。

验证 support 属于验证患者池，是带标注 support 的 episodic 验证协议。
它没有使用 train 患者作为验证 support，也没有借用 query 自己/另一只眼的标注。
后续 baseline 需要复用相同患者划分、support 来源及固定组合。

验证逐一遍历 98 张图片的两个类别，共 196 个 episode，不截取前 1000 条或随机漏评某类。
每类 Dice 沿用训练代码的累计交并统计方式。分布式 sampler 在 episode 总数不能整除卡数时
仍按现有实现补样；196 能被当前使用的两张卡整除。

## 视盘和视杯标签

所有 488 个本地稀疏标签的两个通道互不重叠。直接把通道 0 当完整视盘会漏掉视杯。
读取时约定：

- `class_id=0`，`optic_disc`：通道 0 与通道 1 的并集，即完整视盘。
- `class_id=1`，`optic_cup`：通道 1。

两者是两个独立的二分类 episode，不要求互斥。也兼容原始两个通道已重叠的输入。
图片保留 RGB，使用 ImageNet mean/std；训练沿用小角度旋转、平移、对比度扰动，关闭高斯噪声。
现有 JSON 没有记录采用哪位专家或何种合并标注，本次沿用下载包提供的标签，不另行假定。

## 训练入口

已注册 `papila`，支持 `--benchmark papila` 和联合训练里的 `papila:/path/to/PAPILA`。
`train_joint_papila.sh` 提供 BTCV + AMOS MR + PAPILA + BraTS 的独立入口，使用已有训练选项，
原 `train_joint.sh` 的默认四数据集组合保留。新入口默认权重 `1,1,1,1.5`，PAPILA 不继承 ADAM 的权重 2。
可用 `JOINT_SUBDATASETS`、`JOINT_WEIGHTS` 覆盖。

若延续当前全局 max pooling、两处 attention mask、quality 融合设置：

```bash
conda activate vrpsam
cd /data/2025Zhutaotao/TBD
GPU_ID=1,2 SUPPORT_MASK_POOLING=max \
BBOX_SUPPORT_ATTN_MASK=1 ICL_SUPPORT_ATTN_MASK=1 SUPPORT_FUSION=quality \
bash train_joint_papila.sh
```

不传 attention / fusion 环境变量时，复用现有默认值（两处 mask 关闭、mean 融合）。
新日志名前缀含 `papila`，其余参数仍为 shot=4、8000 episode/epoch、300 epoch、每 epoch 验证。
更换数据集后的四数据集宏平均需单独报告。

## 验证

CPU 单元测试覆盖患者分离、同类异患者 support、验证固定组合、完整视盘包含视杯、
RGB/batch 输出、类别均衡、联合注册与启动参数，不会启动 GPU 训练：

```bash
python -m unittest discover -s tests -p test_papila.py -v
```
