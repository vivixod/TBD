"""BraTS2019 few-shot dataset（MRI FLAIR 脑肿瘤）。

3 前景类：1=necrotic_tumor_core（坏死核心）、2=peritumoral_edema（瘤周水肿）、
3=enhancing_tumor（增强肿瘤）。

个性化：FLAIR 对比度较高（实测 std≈0.188），增广用常规参数即可；
旋转角度可以稍大（脑部肿瘤方向无先验）。
"""
from data.medical_base import MedicalDataset


class DatasetBraTS(MedicalDataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, neighbor_prob=0.0):
        super().__init__(
            datapath=datapath,
            fold=fold,
            transform=transform,
            split=split,
            shot=shot,
            use_original_imgsize=use_original_imgsize,
            name='brats',
            nclass=3,
            neighbor_prob=neighbor_prob,
            # 实测（61 张抽样，灰度转 RGB）：
            img_mean=[0.100, 0.100, 0.100],
            img_std=[0.188, 0.188, 0.188],
            # FLAIR 脑肿瘤个性化增广：旋转可稍大、噪声与 AMOS 一致
            aug_params=dict(
                angle=12.0,
                trans=0.03,
                contrast=0.10,
                noise_sigma=(1.0, 3.0),
                noise_p=0.3,
            ),
        )
