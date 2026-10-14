r""" Dataloader builder for few-shot semantic segmentation dataset  """
import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.btcv import DatasetBTCV,DatasetBTCV2
from data.amos import DatasetAMOS
from data.adam import DatasetAdam
from data.brats import DatasetBraTS
from data.joint import DatasetJoint
from data.episode_sampling import configure_class_balance
# from data.coco2pascal import DatasetCOCO2PASCAL


def worker_init_fn(worker_id):
    """每个 DataLoader worker 用独立随机种子（基于 torch 分发的 worker seed）。

    episode 模式下 __getitem__ 依赖 np.random 随机选数据集/样本，fork 出来的
    worker 会继承相同 RNG 状态导致采样序列重复，这里按 worker 重播种。
    """
    seed = torch.initial_seed() % (2 ** 31)
    np.random.seed(seed)
    random.seed(seed)


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'btcv': DatasetBTCV,
            'btcv2': DatasetBTCV2,
            'amos': DatasetAMOS,
            'adam': DatasetAdam,
            'brats': DatasetBraTS,
            'joint': DatasetJoint,
            # 'coco2pascal': DatasetCOCO2PASCAL
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize
        
        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1,
                         joint_subdatasets=None, n_episodes=None, sub_weights=None,
                         neighbor_prob=0.0, class_balance_prob=0.0):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        if benchmark == 'joint':
            assert joint_subdatasets is not None and len(joint_subdatasets) > 0, \
                "[FSSDataset] benchmark='joint' 必须提供非空的 joint_subdatasets"
            dataset = cls.datasets['joint'](
                joint_subdatasets=joint_subdatasets,
                fold=fold, transform=cls.transform,
                split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize,
                n_episodes=n_episodes if split == 'trn' else None,
                sub_weights=sub_weights if split == 'trn' else None,
                neighbor_prob=neighbor_prob,
                class_balance_prob=class_balance_prob if split == 'trn' else 0.0,
            )
        else:
            dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize, neighbor_prob=neighbor_prob)
            configure_class_balance(dataset, class_balance_prob if split == 'trn' else 0.0)
        # print(f"[DEBUG-dataset] build_dataloader: {benchmark} {split} shot={shot} "
        #       f"dataset构建完成 len={len(dataset)}", flush=True)
        if split == 'trn':
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=shuffle)
            shuffle = False
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=shuffle)
            pin_memory = True
        # 【抗过拟合】每个 rank 用独立的 generator 生成 worker base_seed。
        # 否则两个 rank 主进程 torch RNG 相同 → worker seed 相同 → 采样序列重复，
        # 训练集有效多样性减半。
        gen = torch.Generator()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        gen.manual_seed(20250818 + rank * 1000003)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, pin_memory=True,
                                num_workers=nworker, sampler=sampler, generator=gen,
                                worker_init_fn=worker_init_fn)

        return dataloader
