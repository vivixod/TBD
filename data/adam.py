"""ADAM2020 few-shot dataset（TOF-MRA 颅内动脉瘤）。

2 前景类：1=aneurysm（未治疗动脉瘤）、2=treated_aneurysm（已治疗）。
患者命名 100xxB / 100xxF 为同一患者的基线/随访，split 已按患者归并。

个性化：TOF-MRA 血管图像对比度低（实测 std≈0.088），
增广用较小的旋转角 + 更强的对比度扰动 + 更弱的噪声（血管细节敏感）。
"""
from data.medical_base import MedicalDataset


class DatasetAdam(MedicalDataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, neighbor_prob=0.0):
        super().__init__(
            datapath=datapath,
            fold=fold,
            transform=transform,
            split=split,
            shot=shot,
            use_original_imgsize=use_original_imgsize,
            name='adam',
            nclass=2,
            neighbor_prob=neighbor_prob,
            # 实测（63 张抽样，灰度转 RGB）：
            img_mean=[0.116, 0.116, 0.116],
            img_std=[0.088, 0.088, 0.088],
            # TOF-MRA 个性化增广：小旋转、强对比度、弱噪声
            aug_params=dict(
                angle=8.0,
                trans=0.03,
                contrast=0.15,
                noise_sigma=(0.5, 2.0),
                noise_p=0.3,
            ),
        )
