"""DatasetJoint: 通用联合训练包装器（不写死数据集组合）。

可以把任意 N 个子数据集（BTCV + AMOS + 未来 X/Y/Z）联合训练，每个子数据集的
episode 池内部完全独立（support/query 不会跨数据集）。设计通用化，新增数据集
不需要改这个文件，只在 train.py 的 --joint-subdatasets 列表里加一项。

设计要点：
  - 子 dataset 由元数据列表配置：[(name, datapath), ...]
  - 每个子 dataset 用 FSSDataset.datasets[name] 实例化
  - **episode 模式**：__len__ = n_episodes；每次 __getitem__ 先按 sub_weights
    掷骰子选子数据集，再从该数据集随机抽一个内部索引构建 episode
    （避免大数据集（如 BraTS）在 epoch 中占比过高）
  - 每个 episode 增广由子 dataset 自己的 augment 函数负责（不强加统一）

用法（train.py argparse）：
  --benchmark joint
  --joint-subdatasets 'btcv2:/path/to/btcv,amos:/path/to/amos'
  --n-episodes 4000            # 每个 epoch 的 episode 数（默认 4000）
  --joint-weights 1,1,1,1      # 数据集采样权重（默认均匀，逗号分隔）
  --fold 0 --shot 1 --epochs 50

未来加新数据集：
  1) 在 data/xxx.py 写 DatasetXxx（继承 btcv.py:359 协议）
  2) data/dataset.py: FSSDataset.datasets['xxx'] = DatasetXxx
  3) train.py 命令行加 ',xxx:/path/to/xxx' 即可，joint.py 不需要改
"""
import os
import numpy as np
from torch.utils.data import Dataset

# 避免循环 import：FSSDataset 仅在方法内部延迟导入
# （FSSDataset.datasets 也在方法内通过 'data.dataset.FSSDataset' 路径访问）


class DatasetJoint(Dataset):
    r""" 联合训练 dataset：把多个子 dataset 的 episode 池拼成一个长的 episode 池。

    每个子 dataset 内部完全独立（episode = query + K 张 support 都来自同一子数据集），
    不支持跨数据集 sample。
    """

    def __init__(self, fold, transform, split, shot,
                 use_original_imgsize,
                 sub_datasets_config=None, joint_subdatasets=None,
                 n_episodes=None, sub_weights=None, neighbor_prob=0.0):
        """
        Args:
            sub_datasets_config (or joint_subdatasets):
                list of (name, datapath) tuples，
                例如 [('btcv2', '/data/btcv'), ('amos', '/data/amos')]
            fold, transform, split, shot, use_original_imgsize: 透传给每个子 dataset
            n_episodes: 每个 epoch 的 episode 总数（episode 模式）；None 则回退到
                        旧行为（sum(各子集长度)，idx 路由）
            sub_weights: 各子数据集采样权重（长度 = n_sub）；None 则均匀
        """
        # 兼容两种 kw 名：sub_datasets_config（更内部）/ joint_subdatasets（外部接口）
        cfg = sub_datasets_config if sub_datasets_config is not None else joint_subdatasets
        assert cfg is not None, "DatasetJoint 需要 sub_datasets_config 或 joint_subdatasets"
        self.sub_datasets_config = list(cfg)
        if not self.sub_datasets_config:
            raise ValueError("sub_datasets_config must be non-empty")

        # 实例化每个子 dataset（每个子 dataset 独立的 episode 池）
        # 延迟导入避免与 data.dataset 的循环 import
        from data.dataset import FSSDataset
        self.sub_datasets = []
        for name, datapath in self.sub_datasets_config:
            if name not in FSSDataset.datasets:
                raise KeyError(
                    f"Unknown dataset '{name}'. Available: {list(FSSDataset.datasets.keys())}"
                )
            ds_cls = FSSDataset.datasets[name]
            ds = ds_cls(
                datapath=datapath,
                fold=fold,
                transform=transform,
                split=split,
                shot=shot,
                use_original_imgsize=use_original_imgsize,
                neighbor_prob=neighbor_prob,
            )
            self.sub_datasets.append(ds)

        self.n_sub = len(self.sub_datasets)
        self.fold = fold
        self.split = split
        self.shot = shot

        # 每个子 dataset 的 train/val 长度（验证阶段保持各子数据集 1000 episodes 独立评估）
        self._sub_lens = [len(ds) for ds in self.sub_datasets]

        # ---- episode 模式配置 ----
        self.n_episodes = n_episodes          # None → 旧行为（idx 路由）
        if sub_weights is not None:
            assert len(sub_weights) == self.n_sub, \
                f'sub_weights 长度 {len(sub_weights)} != n_sub {self.n_sub}'
            w = np.asarray(sub_weights, dtype=np.float64)
        else:
            w = np.ones(self.n_sub, dtype=np.float64)
        assert (w > 0).all(), f'sub_weights 必须全为正: {sub_weights}'
        self._sub_probs = w / w.sum()

        # class_id 全局重映射：第 i 个子 dataset 的 class_id 整体偏移到前面子 dataset 之后
        # 例如 BTCV2 (nclass=13) → [0..12], AMOS (nclass=15) → [13..27]
        # 这样 AverageMeter 单一 [2, nclass_total] buffer 不会撞 ID
        self._class_id_offsets = []
        running = 0
        for ds in self.sub_datasets:
            self._class_id_offsets.append(running)
            running += ds.nclass
        self.nclass_total = running    # 总类数 (joint 全局 buffer 大小)

        # 仿照单 dataset 协议，暴露给 AverageMeter：
        self.benchmark = 'joint'
        # class_ids = 各子 dataset 的 class_ids 拼接（已含偏移）
        self.class_ids = []
        for off, ds in zip(self._class_id_offsets, self.sub_datasets):
            self.class_ids.extend([c + off for c in ds.class_ids])

    def __len__(self):
        # episode 模式：固定 episode 数；None 则回退到 sum(各子集长度)
        if self.n_episodes is not None:
            return self.n_episodes
        return sum(self._sub_lens)

    def __getitem__(self, idx):
        if self.n_episodes is not None:
            # episode 模式：先按权重选数据集，再从该数据集随机抽内部样本
            sub_idx = int(np.random.choice(self.n_sub, p=self._sub_probs))
            internal_idx = int(np.random.randint(len(self.sub_datasets[sub_idx])))
        else:
            # 旧行为：idx 路由（idx % n_sub 选子集，idx // n_sub 取内部）
            sub_idx = idx % self.n_sub
            internal_idx = (idx // self.n_sub) % len(self.sub_datasets[sub_idx])
        sub_ds = self.sub_datasets[sub_idx]
        batch = sub_ds[internal_idx]
        # 加 dataset_idx / dataset_name 字段（不影响 train.py 现有逻辑）
        batch['dataset_idx'] = sub_idx
        batch['dataset_name'] = self.sub_datasets_config[sub_idx][0]
        # class_id 重映射：避免不同子数据集撞 ID 写入同一个 IoU buffer
        offset = self._class_id_offsets[sub_idx]
        batch['class_id'] = batch['class_id'] + offset
        return batch

    @property
    def nclass_list(self):
        r""" 每个子 dataset 的 nclass 列表（用于评估时分数据集统计）"""
        return [ds.nclass for ds in self.sub_datasets]

    @property
    def class_ids_list(self):
        r""" 每个子 dataset 的 class_ids（仿 DatasetBTCV2 协议）"""
        return [ds.class_ids for ds in self.sub_datasets]

    def build_class_ids(self):
        # 兼容 btcv.py:225 build_class_ids 接口（如果需要）
        # 这里返回所有子 dataset 的 class_ids 拼接
        out = []
        for ds in self.sub_datasets:
            out.extend(ds.class_ids)
        return out

    # ===================== 透传到 FSSDataset 的类方法（仿 btcv.py 协议） =====================
    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):
        # 单 dataset 模式直接走 FSSDataset.initialize（不在 joint 里重做）
        # joint 的 transform 由 FSSDataset.initialize 设置好后透传给子 dataset
        # 延迟导入避免与 data.dataset 的循环 import
        from data.dataset import FSSDataset
        FSSDataset.initialize(
            img_size=img_size,
            datapath=datapath,        # 兼容旧接口，实际联合训练用 --joint-subdatasets 覆盖
            use_original_imgsize=use_original_imgsize,
        )
    @classmethod
    def build_val_dataloaders(cls, bsz, nworker, joint_subdatasets, shot=1, neighbor_prob=0.0):
        r""" 验证阶段按子数据集分别出 val DataLoader。

        返回 OrderedDict[(name, datapath)] -> DataLoader，
        用于 train.py 验证阶段对每个子数据集单独评测，最后算 macro mIoU。

        用例：
            val_loaders = DatasetJoint.build_val_dataloaders(
                bsz=2, nworker=4, joint_subdatasets=joint_sub, shot=1)
            for (name, path), loader in val_loaders.items():
                # 跑一遍这个子数据集的 val，得到 AverageMeter（含 nclass/class_ids）
        """
        from collections import OrderedDict
        import torch
        from torch.utils.data import DataLoader
        from data.dataset import FSSDataset

        cls._validate_sub_config(joint_subdatasets)

        loaders = OrderedDict()
        for name, datapath in joint_subdatasets:
            assert name in FSSDataset.datasets, \
                f"Unknown sub-dataset '{name}'. Available: {list(FSSDataset.datasets.keys())}"
            ds_cls = FSSDataset.datasets[name]
            ds = ds_cls(
                datapath=datapath,
                fold=0,                     # fold 仅对 PASCAL-5i 协议有效
                transform=FSSDataset.transform,
                split='val',
                shot=shot,
                use_original_imgsize=FSSDataset.use_original_imgsize,
                neighbor_prob=neighbor_prob,
            )
            sampler = None
            try:
                sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False)
            except (ValueError, RuntimeError):
                # 单卡/未初始化 distributed 时退化到 None（顺序采样）
                pass
            loader = DataLoader(
                ds, batch_size=bsz, shuffle=False, pin_memory=True,
                num_workers=0, sampler=sampler,
            )
            loaders[(name, datapath)] = loader
        return loaders

    @classmethod
    def _validate_sub_config(cls, joint_subdatasets):
        assert joint_subdatasets is not None and len(joint_subdatasets) > 0, \
            "[joint] joint_subdatasets 必须非空"
        for item in joint_subdatasets:
            assert isinstance(item, tuple) and len(item) == 2, \
                f"[joint] 子数据集项必须是 (name, datapath) tuple，得到 {item!r}"
