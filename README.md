# VRP-SAM: Medical Few-Shot Segmentation with E-ICL

基于 [VRP-SAM](https://arxiv.org/abs/2402.17726)（CVPR 2024）的医学图像少样本分割扩展实现。

## 主要扩展

- **E-ICL 双路检索**：query-support 做 patch 级显式检索，前景证据 `R_fg = A @ (M_s ⊙ K)` 与背景解剖上下文 `R_bg = A @ ((1-M_s) ⊙ K)` 解耦（`model/base/eicl_interaction.py`）
- **52 tokens 喂 SAM**：50 个上下文 token + 2 个 bbox token 作为 SAM sparse prompt；bbox token 在 token 空间用 MSE 监督（`--bbox_loss_weight`）
- **联合训练（joint）**：支持 N 个子数据集联合训练，每个 episode 的 support/query 只来自同一子数据集（`data/joint.py`）
- **patient-level 划分**：8:2 按患者切分；稀有类（出现患者数 ≤2）保证 train/val 各至少 1 患者

## 数据准备

已处理数据集格式（统一）：`image/{x,y}/*.png`（RGB 切片）+ `label/*.npz`（CSR 稀疏，`shape=[n_class, H*W]`）+ `dataset.json`。

| 数据集 | 路径示例 | 类数 | patients |
|---|---|---|---|
| BTCV | `image/x/` + `mask/` PNG | 13 | — |
| AMOS2022_MR | `image/{x,y}/` + `label/*.npz` | 15 | 36 |
| ADAM2020 | `image/x/` + `label/*.npz` | 2 | 14 |
| BraTS2019 | `image/x/` + `label/*.npz` | 3 | 302 |

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
    --num_query 50 --epochs 300 --lr 1e-4 --bsz 1
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

- Python 3.9+
- PyTorch 1.12+（训练环境：conda env `ztt`）
- segment-anything: `cd segment-anything && pip install -v -e .`

## BibTeX

```BibTeX
@inproceedings{sun2024vrp,
    title={VRP-SAM: SAM with Visual Reference Prompt},
    author={Sun, Yanpeng and Chen, Jiahui and Zhang, Shan and Zhang, Xinyu and Chen, Qiang and Zhang, Gang and Ding, Errui and Wang, Jingdong and Li, Zechao},
    booktitle={Conference on Computer Vision and Pattern Recognition 2024},
    year={2024}
}
```
