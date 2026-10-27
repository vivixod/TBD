"""医学图像 few-shot dataset 共享基类（MR/CT 通用）。

从 data/amos.py 的 DatasetAMOS 提炼而来，数据集间差异全部参数化：
  - name: benchmark 名（决定 split 列表目录 data/splits/lists/<name>/）
  - nclass: 前景类数
  - img_mean / img_std: 归一化常量（各数据集实测）
  - 增广参数：旋转/平移/对比度/噪声强度（各数据集个性化）

子类示例见 data/adam.py、data/brats.py。
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
from data.episode_sampling import ClassBalancedQueryMixin


# ===================== MR 友好增广（无 CT 窗位假设） =====================
def mr_augment(img_pil, mask_tensor, angle=10.0, trans=0.03, contrast=0.1,
               noise_sigma=(1.0, 3.0), noise_p=0.3):
    """MR 友好几何+强度增广。

    - 几何：旋转 [-angle, angle]° + 平移 ±trans（img 双线性 / mask 最近邻）
    - 强度：对比度扰动 ±contrast（无窗位假设）+ 高斯噪声 σ∈noise_sigma, p=noise_p
    - 截断到 [0, 255]
    """
    img = np.array(img_pil).astype(np.float32)
    mask = mask_tensor.numpy().astype(np.int64)
    h, w = img.shape[:2]

    a = np.random.uniform(-angle, angle)
    tx = np.random.uniform(-trans, trans) * w
    ty = np.random.uniform(-trans, trans) * h
    if abs(a) > 0.5 or abs(tx) > 1 or abs(ty) > 1:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if np.random.rand() < 0.5:
        c = np.random.uniform(1 - contrast, 1 + contrast)
        mean = np.mean(img)
        img = (img - mean) * c + mean

    if np.random.rand() < noise_p:
        sigma = np.random.uniform(*noise_sigma)
        img = img + np.random.normal(0, sigma, img.shape)

    img = np.clip(img, 0, 255)
    return Image.fromarray(img.astype(np.uint8)), torch.from_numpy(mask)


# ===================== 同患者相邻切片 support 采样 =====================
def parse_patient_slice(img_name):
    r"""从 image_name 解析 (patient, slice_no)。

    各数据集命名统一为 'x___<患者>_<切片号>'：
      x___ABD_001_128                -> ('ABD_001', 128)
      x___amos_0507_0                -> ('amos_0507', 0)
      x___10027_26                   -> ('10027', 26)
      x___HGG__BraTS19_2013_10_1_100 -> ('HGG__BraTS19_2013_10_1', 100)
    """
    clean = img_name.replace('x___', '')
    if '_' not in clean:
        return None, None
    patient, slice_str = clean.rsplit('_', 1)
    try:
        slice_no = int(slice_str)
    except ValueError:
        return None, None
    return patient, slice_no


def build_patient_slice_index(metadata):
    r"""按患者聚合切片。返回 ({patient: {slice_no: img_name}}, {patient: [sorted slice_no]})。

    注意：切片号可能是 stride 采样后的原始索引（如 btcv），
    “相邻”必须从排序列表中取邻居，而不是数值 ±1。
    """
    p2slices, p2order = {}, {}
    for name in metadata:
        patient, slice_no = parse_patient_slice(name)
        if patient is None:
            continue
        p2slices.setdefault(patient, {})[slice_no] = name
        p2order.setdefault(patient, []).append(slice_no)
    for patient in p2order:
        p2order[patient] = sorted(set(p2order[patient]))
    return p2slices, p2order


class MedicalDataset(ClassBalancedQueryMixin, Dataset):
    r"""通用医学 2D few-shot dataset（episode 协议与 data/amos.py 完全一致）。

    数据布局（datapath 下）：
      image/{x,y}/*.png            # RGB 切片
      label/{x,y}___*.npz          # CSR sparse, shape=[nclass, H*W]
      data/splits/lists/<name>/fss_list/{train,val}/data_list_0.txt
                                    sub_class_file_list_0.txt
    """

    def __init__(self, datapath, fold, transform, split, shot,
                 use_original_imgsize, name, nclass, img_mean, img_std,
                 aug_params=None, neighbor_prob=0.0):
        self.split_norm = 'val' if split in ['val', 'test'] else 'train'
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = nclass
        self.benchmark = name
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        self.img_path = datapath
        self.label_path = datapath

        self.transform = self._build_transform(transform, img_mean, img_std)

        # 个性化增广参数（None 用默认）
        self.aug_params = aug_params or {}

        # 相邻切片 support 采样比例（0=关闭；1.0=全部相邻，难度下界诊断用）
        self.neighbor_prob = float(neighbor_prob)

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        # 同患者切片索引（相邻 support 采样用）
        self._patient2slices, self._patient2order = \
            build_patient_slice_index(self.img_metadata)

        # 类名（与 class_ids 对齐，val 打印 per-class 指标用）
        self.class_names = self._load_class_names()

        self._shape_cache = {}
        self._label_classes_cache = {}

    def _load_class_names(self):
        """从 dataset.json 的 labels 读类名（按 class_ids 顺序）。"""
        import json
        try:
            j = json.load(open(os.path.join(self.img_path, 'dataset.json')))
            labels = j.get('labels', {})
            all_names = [labels.get(str(c), f'c{c}') for c in range(1, self.nclass + 1)]
        except Exception:
            all_names = [f'c{c}' for c in range(1, self.nclass + 1)]
        return [all_names[c] for c in self.class_ids]

    @staticmethod
    def _build_transform(transform, img_mean, img_std):
        """把传入 Compose 的 Normalize 层替换为数据集自有的 mean/std。"""
        from torchvision import transforms
        new_steps = []
        for step in transform.transforms:
            if isinstance(step, transforms.Normalize):
                new_steps.append(transforms.Normalize(img_mean, img_std))
            else:
                new_steps.append(step)
        return transforms.Compose(new_steps)

    def _augment(self, img_pil, mask_tensor):
        return mr_augment(img_pil, mask_tensor, **self.aug_params)

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)

        query_name, support_names, class_sample = self.sample_episode(idx)
        return self._load_episode(query_name, support_names, class_sample)

    def _load_episode(self, query_name, support_names, class_sample):
        """Load/augment an episode; datasets may define their own episode indexing."""
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = \
            self.load_frame(query_name, support_names)

        if self.split == 'trn':
            query_img, query_cmask = self._augment(query_img, query_cmask)
            supp_aug = [self._augment(si, sm) for si, sm in zip(support_imgs, support_cmasks)]
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

    # ===================== 协议层（与 data/amos.py 一致） =====================
    def extract_ignore_idx(self, mask, class_id):
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

    # ===================== 数据读取 =====================
    def _split_prefix(self, image_name):
        """'x___amos_0507_5' → ('x', 'amos_0507_5')"""
        for prefix in ('x', 'y'):
            if image_name.startswith(f'{prefix}___'):
                return prefix, image_name[4:]
        raise ValueError(f"Unknown prefix in image_name: {image_name}")

    def read_img(self, img_name):
        prefix, clean = self._split_prefix(img_name)
        img_p = os.path.join(self.img_path, 'image', prefix, clean + '.png')
        return Image.open(img_p).convert('RGB')

    def read_mask(self, img_name):
        """CSR npz → 单通道 (H, W)，值 0~nclass（0=背景）。"""
        prefix, clean = self._split_prefix(img_name)
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
        dense = csr.toarray().reshape(n_class, n_hxw)
        label = dense.argmax(axis=0) + 1
        label[dense.sum(axis=0) == 0] = 0
        if img_name in self._shape_cache:
            H, W = self._shape_cache[img_name]
        else:
            img = self.read_img(img_name)
            H, W = img.size[1], img.size[0]
            self._shape_cache[img_name] = (H, W)
        label_2d = label.reshape(H, W)
        return torch.tensor(label_2d.astype(np.int64))

    def _label_classes(self, img_name):
        if img_name not in self._label_classes_cache:
            self._label_classes_cache[img_name] = self.read_mask(img_name).unique().tolist()
        return self._label_classes_cache[img_name]

    def sample_episode(self, idx):
        balanced = self.sample_class_balanced_query()
        if balanced is not None:
            query_name, class_sample = balanced
        else:
            query_name = self.img_metadata[idx]
            label_class = self._label_classes(query_name)
            if 0 in label_class:
                label_class.remove(0)
            valid_classes = [c - 1 for c in label_class if c - 1 in self.class_ids]
            class_sample = int(np.random.choice(valid_classes)) if valid_classes else self.class_ids[0]

        if getattr(self, 'support_retriever', None) is not None:
            return query_name, self.support_retriever.select(query_name, class_sample, self.shot), class_sample

        # 相邻切片 support（易任务）：不足 shot 再退回跨患者候选池补齐
        support_names = []
        if self.neighbor_prob > 0 and np.random.rand() < self.neighbor_prob:
            support_names = self._sample_neighbor_supports(query_name, class_sample)

        candidates = self.img_metadata_classwise.get(class_sample + 1, [])
        while len(support_names) < self.shot:
            if len(candidates) == 0:
                support_names.append(query_name)
            elif len(set(candidates)) <= 1 and candidates[0] in support_names:
                # 同类候选只有 1 个且已被相邻逻辑选中：允许重复以凑满 shot
                support_names.append(candidates[0])
            elif set(candidates).issubset(support_names):
                support_names.append(np.random.choice(candidates))
            else:
                support_name = np.random.choice(candidates)
                if support_name not in support_names:
                    support_names.append(support_name)
        return query_name, support_names, class_sample

    def _sample_neighbor_supports(self, query_name, class_sample):
        r"""从 query 同患者相邻切片（排序列表 ±3 窗口）选含该类的切片做 support。"""
        patient, slice_no = parse_patient_slice(query_name)
        if patient is None:
            return []
        order = self._patient2order.get(patient)
        if not order or slice_no not in order:
            return []
        pos = order.index(slice_no)
        cand_nos = [order[i] for i in range(max(0, pos - 6), min(len(order), pos + 7))
                    if i != pos]
        cand_names = [self._patient2slices[patient][n] for n in cand_nos]
        valid = [n for n in cand_names
                 if (class_sample + 1) in self._label_classes(n)]
        if not valid:
            return []
        k = min(self.shot, len(valid))
        return list(np.random.choice(valid, size=k, replace=False))

    def build_class_ids(self):
        return list(range(self.nclass))

    def build_img_metadata(self):
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
