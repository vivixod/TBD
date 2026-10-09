"""AMOS2022 few-shot medical semantic segmentation dataset (MR modaltiy).

与 VRP-SAM 完全相同的 episodic few-shot 协议（参考 data/btcv.py: DatasetBTCV2），
但支持：
  - MR 友好增广（不假设 CT 窗宽窗位）
  - 跨子目录 image/x/ + image/y/（patient 共享 train/val 池）
  - 稀疏 CSR npz label 直接在 __getitem__ 内解码，不写中间 PNG
"""
import os
import glob
import numpy as np
import scipy.sparse as sp
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2


# ===================== AMOS MR 友好增广 =====================
def amos_augment(img_pil, mask_tensor):
    """AMOS MR 几何+强度增广（不假设 CT HU 窗宽窗位）。

    - 几何：旋转 [-10°, 10°] + 平移 ±3%（img 双线性 / mask 最近邻）
    - 强度：对比度扰动 ±10%（无窗位假设）+ 高斯噪声 σ∈[1,3]，p=0.3
    - 截断到 [0, 255]
    """
    img = np.array(img_pil).astype(np.float32)
    mask = mask_tensor.numpy().astype(np.int64)
    h, w = img.shape[:2]

    # ---- 1. 几何：旋转 + 平移 ----
    angle = np.random.uniform(-10, 10)
    tx = np.random.uniform(-0.03, 0.03) * w
    ty = np.random.uniform(-0.03, 0.03) * h
    if abs(angle) > 0.5 or abs(tx) > 1 or abs(ty) > 1:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # ---- 2. 强度：对比度扰动（无窗位假设） ----
    if np.random.rand() < 0.5:
        contrast = np.random.uniform(0.9, 1.1)
        mean = np.mean(img)
        img = (img - mean) * contrast + mean

    # ---- 3. 高斯噪声 ----
    if np.random.rand() < 0.3:
        sigma = np.random.uniform(1.0, 3.0)
        img = img + np.random.normal(0, sigma, img.shape)

    img = np.clip(img, 0, 255)
    return Image.fromarray(img.astype(np.uint8)), torch.from_numpy(mask)


# ===================== DatasetAMOS =====================
class DatasetAMOS(Dataset):
    r""" AMOS2022 MR 多器官 few-shot dataset（不继承 DatasetBTCV）。

    数据布局：
      /vcl/2025zhutaotao/AMOS2022_MR/
        image/
          x/amos_0507_5.png       # (290, 320, 3) uint8
          y/amos_0508_10.png      # (320, 260, 3) uint8
        label/
          x___amos_0507_5.(15, 290, 320, 1).npz    # CSR 15 通道
          y___amos_0508_10.(15, 320, 260, 1).npz
      data/splits/lists/amos/fss_list/
        train/data_list_0.txt                 # 每行一个 image_name (如 'x___amos_0507_5')
        train/sub_class_file_list_0.txt       # {class_id: [[image_name], ...]}
        val/data_list_0.txt
        val/sub_class_file_list_0.txt

    Split 设计：x+y 混合 36 patients 按 8:2 patient-level 划分（seed=321），
    稀有类（膀胱/前列腺）固定 1 患者进 val（生成脚本 data/splits/gen_splits.py）。
    """

    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        # 规范化 split：'trn' -> 'train'，'val'/'test' -> 'val'（对齐 btcv.py）
        self.split_norm = 'val' if split in ['val', 'test'] else 'train'
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 15                     # AMOS 0~14 共 15 类
        self.benchmark = 'amos'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        self.img_path = datapath             # 指向 image/ 目录
        self.label_path = datapath           # 指向 label/ 目录

                # AMOS 是 MR 图像，PNG mean ≈ 0.13 / std ≈ 0.15（实测 50 张）
        # 与 ImageNet mean=[0.485,...] 差距过大；替换 Normalize 层用 AMOS 自有 mean/std
        self.transform = self._build_amos_transform(transform)

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        # 缓存每个 image_name 的 in-plane shape（避免 __getitem__ 重复 glob）
        self._shape_cache = {}
        # 缓存每张图的器官类别集合（避免 sample_episode 重复解码 npz）
        self._label_classes_cache = {}

    @staticmethod
    def _build_amos_transform(transform):
        """把传入的 torchvision Compose 里 Normalize 层替换成 AMOS MR 自有 mean/std。

        实测（新 AMOS2022_MR，x 1415 + y 1740 slices pooled）：
        mean ≈ [0.133, 0.133, 0.133]，std ≈ [0.184, 0.184, 0.184]
        归一化后 mean≈0, std≈1，对齐 BTCV2（其 PNG 已预归一化到 ImageNet 中性灰度）。
        """
        from torchvision import transforms
        amos_mean = [0.133, 0.133, 0.133]
        amos_std  = [0.184, 0.184, 0.184]
        new_steps = []
        for step in transform.transforms:
            if isinstance(step, transforms.Normalize):
                new_steps.append(transforms.Normalize(amos_mean, amos_std))
            else:
                new_steps.append(step)
        return transforms.Compose(new_steps)

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)

        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(
            query_name, support_names
        )

        if self.split == 'trn':
            query_img, query_cmask = amos_augment(query_img, query_cmask)
            supp_aug = [amos_augment(si, sm) for si, sm in zip(support_imgs, support_cmasks)]
            support_imgs, support_cmasks = [a[0] for a in supp_aug], [a[1] for a in supp_aug]

        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(
                query_cmask.unsqueeze(0).unsqueeze(0).float(),
                size=query_img.size()[-2:], mode='nearest',
            ).squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)

        support_imgs = torch.stack([self.transform(s) for s in support_imgs])
        support_masks, support_ignore_idxs = [], []
        for scmask in support_cmasks:
            scmask = F.interpolate(
                scmask.unsqueeze(0).unsqueeze(0).float(),
                size=support_imgs.size()[-2:], mode='nearest',
            ).squeeze()
            sm, si = self.extract_ignore_idx(scmask.float(), class_sample)
            support_masks.append(sm)
            support_ignore_idxs.append(si)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {
            'query_img': query_img,
            'query_mask': query_mask,
            'query_name': query_name,
            'query_ignore_idx': query_ignore_idx,
            'org_query_imsize': org_qry_imsize,
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'support_names': support_names,
            'support_ignore_idxs': support_ignore_idxs,
            'class_id': torch.tensor(class_sample),
        }
        return batch

    # ===================== 协议层（与 btcv.py:92 一致） =====================
    def extract_ignore_idx(self, mask, class_id):
        r""" 多类 mask（值 0~14）转 binary mask：保留 class_id+1 的前景，其余置 0。"""
        boundary = torch.zeros_like(mask)
        target_pixel_val = class_id + 1
        mask[mask != target_pixel_val] = 0
        mask[mask == target_pixel_val] = 1
        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(n) for n in support_names]
        support_masks = [self.read_mask(n) for n in support_names]
        org_qry_imsize = query_img.size
        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    # ===================== AMOS specific =====================
    def _split_prefix(self, image_name):
        """'x___amos_0507_5' → ('x', 'amos_0507_5')"""
        # 前 4 个字符形如 'x___' 或 'y___'
        for prefix in ('x', 'y'):
            if image_name.startswith(f'{prefix}___'):
                return prefix, image_name[4:]
        raise ValueError(f"Unknown prefix in image_name: {image_name}")

    def read_img(self, img_name):
        prefix, clean = self._split_prefix(img_name)
        img_p = os.path.join(self.img_path, 'image', prefix, clean + '.png')
        return Image.open(img_p).convert('RGB')

    def read_mask(self, img_name):
        """读 npz label → 多通道 CSR → 单通道 (H, W)，值 0~15（0=背景，1~15=器官）。"""
        prefix, clean = self._split_prefix(img_name)
        # 文件名带 shape 标记，如 'x___amos_0507_5.(15, 290, 320, 1).npz'
        npz_glob = glob.glob(
            os.path.join(self.label_path, 'label', f'{prefix}___{clean}.*.npz')
        )
        if not npz_glob:
            raise FileNotFoundError(f"No npz for {img_name}")
        npz = np.load(npz_glob[0])
        csr = sp.csr_matrix(
            (npz['data'], npz['indices'], npz['indptr']),
            shape=tuple(npz['shape']),
        )
        n_class = int(npz['shape'][0])
        n_hxw = int(npz['shape'][1])
        dense = csr.toarray().reshape(n_class, n_hxw)        # (15, H*W)
        # 取前景类（argmax），0 = 背景
        label = dense.argmax(axis=0) + 1                      # (H*W,) 值 1~15
        label[dense.sum(axis=0) == 0] = 0                    # 全背景位置
        # 用 PNG image 拿 (H, W)（PIL size = (W, H)），并缓存避免重复读图
        if img_name in self._shape_cache:
            H, W = self._shape_cache[img_name]
        else:
            img = self.read_img(img_name)
            H, W = img.size[1], img.size[0]
            self._shape_cache[img_name] = (H, W)
        label_2d = label.reshape(H, W)
        return torch.tensor(label_2d.astype(np.int64))

    def _label_classes(self, img_name):
        r"""返回某张图包含的器官像素值列表（带缓存）。"""
        if img_name not in self._label_classes_cache:
            self._label_classes_cache[img_name] = self.read_mask(img_name).unique().tolist()
        return self._label_classes_cache[img_name]

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        # 读 mask 拿含前景的器官集（值 1~15）——用缓存避免每个 batch 重复解码 npz
        label_class = self._label_classes(query_name)
        if 0 in label_class:
            label_class.remove(0)
        label_class_0 = [c - 1 for c in label_class]              # 0-indexed
        valid_classes = [c for c in label_class_0 if c in self.class_ids]
        if len(valid_classes) == 0:
            class_sample = self.class_ids[0]
        else:
            class_sample = int(np.random.choice(valid_classes))

        # 抽 K 张 support（同类器官）
        candidates = self.img_metadata_classwise.get(class_sample + 1, [])
        support_names = []
        loop_iter = 0
        while True:
            loop_iter += 1
            if len(candidates) == 0:
                support_name = query_name
            else:
                support_name = np.random.choice(candidates)
            if query_name != support_name or len(candidates) <= 1:
                support_names.append(support_name)
            if len(support_names) == self.shot:
                break
        return query_name, support_names, class_sample

    def build_class_ids(self):
        # AMOS 不分 fold 类，全 15 类都参与训练+验证
        return list(range(self.nclass))

    def build_img_metadata(self):
        # 从 train/val 子目录里的 data_list_0.txt 读 2D 切片名
        list_file = f'data/splits/lists/{self.benchmark}/fss_list/{self.split_norm}/data_list_{self.fold}.txt'
        with open(list_file, 'r') as f:
            metadata = [line.strip().split(' ')[0].split('/')[-1].split('.')[0]
                        for line in f if line.strip()]
        return metadata

    def build_img_metadata_classwise(self):
        subclass_file = f'data/splits/lists/{self.benchmark}/fss_list/{self.split_norm}/sub_class_file_list_{self.fold}.txt'
        with open(subclass_file, 'r') as f:
            sub_data = f.read()
        sub_class_file_list = eval(sub_data)
        img_metadata_classwise = {}
        for sub_cls in sub_class_file_list.keys():
            img_metadata_classwise[sub_cls] = [
                data[0].split('/')[-1].split('.')[0]
                for data in sub_class_file_list[sub_cls]
            ]
        return img_metadata_classwise