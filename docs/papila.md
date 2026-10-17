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
图片保留 RGB，使用 ImageNet mean/std；PAPILA 独立的增强和预处理见下节。
现有 JSON 没有记录采用哪位专家或何种合并标注，本次沿用下载包提供的标签，不另行假定。

## 眼底专用预处理与增强（fundus_v2）

PAPILA 覆盖自己的 `_augment` 和 episode 图像处理流程，使用 `data/papila_augment.py`，
不再调用共享的 `mr_augment`。以下幅度是本项目的保守起始设置，尚未经过性能消融。
原始 [PAPILA 论文](https://www.nature.com/articles/s41597-022-01388-1)也使用过缩放、水平翻转和平移等增强；
本实现的颜色操作基于 [Torchvision functional/ColorJitter](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html) 提供的变换。

训练与验证都先按原始宽高比缩放，再居中补黑边至 `512×512`，图像用双线性插值、标签用最近邻。
本地 `2576×1934` 图片会缩为 `512×384`，上下各补 64 像素。整个过程保留全图，
不根据 query GT 定位或裁剪；输出像素尺寸取整会有正常的亚像素比例误差。
随后在这个输入尺度上做训练增强，避免在 500 万像素原图上重复进行颜色/仿射变换。

| 操作 | 参数 | 应用方式 |
| --- | --- | --- |
| 水平翻转 | 概率 0.5 | 图像、标签同步；分割类别没有左右眼类别之分 |
| 旋转 | ±15° | 与下述平移、缩放合成一次仿射 |
| 平移 | 宽/高的 ±3% | 图像、标签同步 |
| 缩放 | 0.9–1.1 倍 | x/y 使用相同尺度 |
| 亮度 | 0.85–1.15 倍 | 颜色增强整体以 0.8 概率启用 |
| 对比度 | 0.85–1.15 倍 | 与其他颜色操作随机排序 |
| 饱和度 | 0.9–1.1 倍 | 只改图像 |
| 色相 | ±0.01 | 只改图像；在 Normalize 前执行 |
| Gamma | 0.9–1.1 | 概率 0.3，只改图像 |

query 与每个 support 独立增强，每张图像和自己的标签严格共享几何变换。
训练时使用标签检查变换后的前景框是否完整位于画布内，以及原来存在的视盘/视杯标签是否仍然存在；
最多尝试 8 次，全部不合格时回退到仅翻转后的图像，继续应用颜色增强。
颜色操作不修改 mask，并保持原来纯黑的背景和填充区域为零。
不启用高斯噪声、垂直翻转、随机裁剪、擦除或弹性形变。

验证只进行确定性的等比例缩放、补边及 RGB 归一化，不执行随机增强。
由于评估坐标包含补边，PAPILA 使用 `use_original_imgsize=False`，不支持直接切换为原图坐标 GT。
原始文件读取 `read_img/read_mask` 仍保留原始尺寸，可用于离线分析。

**SAM 缓存隔离**：batch 中 query 名带预处理版本和输入尺寸，如
`papila_fundus_v2_s512__RET002OD`。现有 SAM 接口会自然写入对应的独立缓存文件，
不会读到旧版 `RET002OD.npy` 的直接拉伸特征。原始 split 和 support 名字保持不变，
`read_img/read_mask/papila_patient_id` 也支持识别版本化 query 名。
训练继续在线编码实际增强后的图像。缓存首次生成的成本由新预处理对应的 query 数量决定。

这是一次预处理和训练增强协议更新；新启动的 PAPILA 实验会自动使用它。
已经在运行的 Python 进程不会自动载入这些修改，旧 checkpoint 的输入预处理也需要按对应版本区分。

## 训练入口

已注册 `papila`，支持 `--benchmark papila` 和联合训练里的 `papila:/path/to/PAPILA`。
`train_joint.sh` 默认使用 BTCV + AMOS MR + PAPILA + BraTS；`train_joint_papila.sh` 仍可作为同组合的入口。
两者默认权重均为 `1,1,1,1.5`，PAPILA 不继承 ADAM 的权重 2。
可用 `JOINT_SUBDATASETS`、`JOINT_WEIGHTS` 覆盖。

若延续当前全局 max pooling、两处 attention mask、quality 融合设置：

```bash
conda activate vrpsam
cd /data/2025Zhutaotao/TBD
GPU_ID=1,2 SUPPORT_MASK_POOLING=max \
BBOX_SUPPORT_ATTN_MASK=1 ICL_SUPPORT_ATTN_MASK=1 SUPPORT_FUSION=quality \
bash train_joint.sh
```

不传 attention / fusion 环境变量时，复用现有默认值（两处 mask 关闭、mean 融合）。
新日志名前缀含 `papila`，其余参数仍为 shot=4、8000 episode/epoch、300 epoch、每 epoch 验证。
更换数据集后的四数据集宏平均需单独报告。

## 验证

CPU 单元测试覆盖患者分离、同类异患者 support、验证固定组合、完整视盘包含视杯、
RGB/batch 输出、类别均衡、联合注册与启动参数，不会启动 GPU 训练：

```bash
python -m unittest discover -s tests -p 'test_papila*.py' -v
python -m unittest discover -s tests -p test_sam_cache_alignment.py -v
```

2026-10-17 的 CPU 检查：PAPILA/专用增强 19 项、SAM 缓存对齐 12 项、原四数据集采样回归 7 项全部通过。
遍历本地全部 488 张图片进行 512 输入预处理，原有视盘/视杯标签均未消失；
对 65 张训练图片抽取 325 次随机增强，未出现视盘或视杯消失，二者包含关系保持成立。
全量预处理后最小完整视盘为 3224 像素，最小视杯为 56 像素。
检查明细与三组全图预览位于 `analysis/papila_augmentation_20261017/audit.json`、`preview.png`。
这些验证检查数据与几何一致性，分割精度收益仍需训练实验确认。
