# VRP-SAM: Medical Few-Shot Segmentation with E-ICL

基于 [VRP-SAM](https://arxiv.org/abs/2402.17726)（CVPR 2024）的医学图像少样本分割扩展实现。

## 主要扩展

- **跨患者 support 检索**：体数据集排除 query 患者后选图像特征最相似的4张，PAPILA 保持异患者采样；训练前自动准备索引，详见 [使用说明](docs/support_retrieval.md)。

- **E-ICL 显式检索**：query-support 做 patch 级匹配，生成 `pseudo=A@M_s` 与图像检索特征 `A@normalized(Fs)`（`model/base/eicl_interaction.py`）
- **52 tokens 喂 SAM**：50 个上下文 token + 2 个 bbox token 作为 SAM sparse prompt；bbox token 在 token 空间用 MSE 监督（`--bbox_loss_weight`）
- **联合训练（joint）**：支持 N 个子数据集联合训练，每个 episode 的 support/query 只来自同一子数据集（`data/joint.py`）
- **patient-level 划分**：8:2 按患者切分；稀有类（出现患者数 ≤2）保证 train/val 各至少 1 患者

## 数据准备

已处理数据集格式（统一）：`image/{x,y}/*.png`（RGB 切片）+ `label/*.npz`（CSR 稀疏，`shape=[n_class, H*W]`）+ `dataset.json`。

| 数据集 | 路径示例 | 类数 | patients |
|---|---|---|---|
| BTCV | `image/x/` + `mask/` PNG | 13 | — |
| AMOS2022_MR | `image/{x,y}/` + `label/*.npz` | 13（原始标签 15） | 36 |
| ADAM2020 | `image/x/` + `label/*.npz` | 2 | 14 |
| BraTS2019 | `image/x/` + `label/*.npz` | 3 | 302 |

AMOS 仅使用原始标签 1–13（零基类别 ID 0–12）。膀胱（标签 14）和前列腺/子宫
（标签 15）在训练、验证分区各只有一位患者，不再作为 episode 目标，也不计入
Dice、mIoU、FB-IoU 或逐类报告。原始标注、患者 split 和其他类别编号保持不变。
修改对新启动的进程生效；历史 AMOS 15 类结果需与当前 13 类结果区分报告。

生成 split 列表（patient-level 8:2，seed=321）：

```bash
python data/splits/gen_splits.py --datasets adam,amos,brats --ratio 0.8 --seed 321
```

## 训练

联合训练（BTCV + AMOS2022_MR）：

```bash
bash train_joint.sh
# 脚本内配置：JOINT_SUBDATASETS="btcv2:/path/BTCV,amos:/path/AMOS2022_MR"
```

或直接命令：

```bash
python -m torch.distributed.launch --nproc_per_node=1 train.py \
    --benchmark joint \
    --joint-subdatasets "btcv2:/path/BTCV,amos:/path/AMOS2022_MR" \
    --backbone resnet50 --shot 1 --fold 0 --condition mask \
    --num_query 50 --epochs 100 --lr 1e-4 --bsz 1
```

SAM 特征与 query 增强保持一致：训练时在线编码增强后的 query，同一 query 的多张
support 共用一次图像编码；验证/测试在 `eval()` 模式下缓存未增强图像特征。
新验证缓存位于 `${TBD_FEATS:-/data2/2025Zhutaotao/TBD/cache_features}/eval_unaugmented_v1/`，
首次验证会生成缓存，后续复用；旧目录根部的历史缓存不再读取。
训练/验证入口会设置 SAM 的 `train()`/`eval()` 模式，图像编码器始终冻结。

训练加速：`--sam-encoder-precision auto`（默认）在支持原生 BF16 的 GPU 上对冻结的
SAM 图像编码器使用 BF16，其他 GPU 回退到 FP32；输出特征转换回 FP32，prompt/decoder、
损失和验证缓存都保持 FP32。`layer4` 不参与当前预测，已跳过其前向并冻结参数，但保留
checkpoint 中的参数结构；context/bbox attention 不再输出未使用的注意力权重。
完整验证仍然每轮执行，episode 数量、shot、batch size 和学习率设置不变。

类别均衡采样：`train_joint.sh` 默认 `CLASS_BALANCE_PROB=0.5`，四个子数据集内部
各有 50% 的训练 episodes 先均匀选一个有 query 候选的类别，再从本训练 split 内
含该类的切片中选 query；其余沿用 query-first 采样。support 继续按原有相邻/同类
策略选取。四数据集权重不变，验证采样和频率不变。稀有类候选不足 shot 时允许重复
填充，避免无限循环。启动日志的 `[Sampling]` 会列出均衡概率和每类 query 池大小。

`CLASS_BALANCE_PROB=0` 关闭均衡，`1` 使用纯类别均衡；直接运行 `train.py` 时通过
`--class-balance-prob` 指定（默认 0，保持旧命令行为）。新日志默认使用
`trn_joint4_shot4_cb0.5.log` 前缀，可用 `EXPERIMENT_NAME` 覆盖。

启动与精度对照：

```bash
bash train_joint.sh
SAM_ENCODER_PRECISION=fp32 bash train_joint.sh
CLASS_BALANCE_PROB=0 bash train_joint.sh
CLASS_BALANCE_PROB=1 bash train_joint.sh
```

在空闲 GPU 上测 SAM 编码器的实际速度（不训练、不写缓存）：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_sam_encoder.py --steps 20 --warmup 5
```

编码器测速结果不等于完整训练提速比例，完整耗时以训练日志/TensorBoard 为准。

CPU 回归检查（使用项目 Python 环境，无需加载 SAM 权重）：

```bash
python tests/test_sam_cache_alignment.py -v
python tests/test_training_speedups.py -v
python tests/test_class_balanced_sampling.py -v
```

单数据集训练：

```bash
python -m torch.distributed.launch --nproc_per_node=1 train.py \
    --benchmark btcv2 --datapath /path/BTCV --shot 1
```

新增第 N 个数据集：实现 `DatasetXxx`（协议同 `data/amos.py`），在 `data/dataset.py` 注册，再往 `--joint-subdatasets` 追加 `xxx:/path` 即可。

## 测试

```bash
python test.py --load logs/<exp>/best_model.pt --benchmark btcv2 --shot 1
```

## 主要文件

```
train.py                      训练入口（joint / 单数据集）
data/joint.py                 N 数据集联合 wrapper
data/amos.py                  AMOS2022_MR dataset（MR 归一化 + CSR npz 解码）
data/splits/gen_splits.py     split 列表生成
model/VRP_encoder.py          E-ICL + 52 token 编码器
model/base/eicl_interaction.py 双路检索模块
model/base/bbox_transfer.py   bbox token 生成
SAM2pred.py                   SAM 适配
```

## Requirements

当前训练环境为 Conda `vrpsam`：Python 3.9.25、PyTorch 2.8.0、
torchvision 0.23.0，PyTorch CUDA 构建版本为 12.8。
`requirements.txt` 固定项目直接依赖的已安装版本，SAM 使用仓库内的源码。

在仓库根目录安装：

```bash
conda create -n vrpsam python=3.9.25
conda activate vrpsam
python -m pip install -r requirements.txt
```

SAM 单独以 editable 模式安装：

```bash
cd segment-anything
pip install -v -e .
cd ..
```

已有 `vrpsam` 环境时，跳过创建步骤。
`match_abd_to_img.py` 的 NIfTI 转换额外需要 `nibabel`（当前训练环境未安装）：
`python -m pip install nibabel`。

## BibTeX

```BibTeX
@inproceedings{sun2024vrp,
    title={VRP-SAM: SAM with Visual Reference Prompt},
    author={Sun, Yanpeng and Chen, Jiahui and Zhang, Shan and Zhang, Xinyu and Chen, Qiang and Zhang, Gang and Ding, Errui and Wang, Jingdong and Li, Zechao},
    booktitle={Conference on Computer Vision and Pattern Recognition 2024},
    year={2024}
}
```
