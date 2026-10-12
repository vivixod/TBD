r""" BTCV few-shot medical semantic segmentation dataset """
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import cv2


def joint_augment(img_pil, mask_tensor):
    """腹部 CT 多器官分割的训练增强。

    - 几何：随机旋转 [-10°, 10°] + 平移 ±3%（img 双线性 / mask 最近邻，同步）
    - 强度：对比度扰动（0.9~1.1，均值中心）+ 亮度平移 [-10, 10]
    - 噪声：高斯噪声 σ∈[1, 3]，概率 0.3

    刻意不做：翻转（解剖学不一致）、弹性形变（破坏器官关系）。
    """
    img = np.array(img_pil).astype(np.float32)
    mask = mask_tensor.numpy().astype(np.int64)
    h, w = img.shape[:2]

    # ---- 1. 小幅度仿射变换：旋转 + 平移 ----
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

    # ---- 2. CT 强度增强 ----
    if np.random.rand() < 0.5:   # 对比度（窗宽）扰动
        contrast = np.random.uniform(0.9, 1.1)
        mean = np.mean(img)
        img = (img - mean) * contrast + mean
    if np.random.rand() < 0.5:   # 亮度（窗位）平移
        img = img + np.random.uniform(-10, 10)

    # ---- 3. 高斯噪声 ----
    if np.random.rand() < 0.3:
        sigma = np.random.uniform(1.0, 3.0)
        img = img + np.random.normal(0, sigma, img.shape)

    # ---- 4. 截断到 [0, 255] ----
    img = np.clip(img, 0, 255)

    return Image.fromarray(img.astype(np.uint8)), torch.from_numpy(mask)

# ---- DEBUG 工具：每个进程最多打印 30 条，防止刷屏 ----
_DEBUG_N = 0

def _dbg(msg):
    global _DEBUG_N
    if _DEBUG_N < 30:
        _DEBUG_N += 1
        print(msg, flush=True)

class DatasetBTCV(Dataset):
    BTCV_CLASS_NAMES = ['spleen', 'right_kidney', 'left_kidney', 'gallbladder',
                        'esophagus', 'liver', 'stomach', 'aorta', 'IVC',
                        'portal_vein', 'pancreas', 'right_adrenal', 'left_adrenal']
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        # 规范化 split 命名，对齐 PASCAL 的习惯
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        # BTCV 官方有 13 个腹部器官
        self.nclass = 13 
        self.benchmark = 'btcv'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        # 你的数据实体路径
        self.img_path = os.path.join(datapath, 'image')
        self.ann_path = os.path.join(datapath, 'mask')
        self.transform = transform

        # 构建 Fold 的类别 ID (决定哪些器官用于训练，哪些用于验证)
        self.class_ids = self.build_class_ids()
        # 类名与 class_ids 对齐（val 打印 per-class 指标用）
        self.class_names = [self.BTCV_CLASS_NAMES[c] for c in self.class_ids]
        
        # 构建情景训练的“双重索引库”
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        # 训练时按实际切片数，验证时仿照 PASCAL 固定测试 1000 个 Episode
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata) 
        
        # 1. 核心：情景采样 (抽 Query -> 定器官 -> 抽 Support)
        query_name, support_names, class_sample = self.sample_episode(idx)
        
        # 2. 读取原始数据
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)

        # 2.5 训练集联合增强：img/mask 同步做翻转 + 旋转 + 弹性形变
        if self.split == 'trn':
            query_img, query_cmask = joint_augment(query_img, query_cmask)
            supp_aug = [joint_augment(si, sm) for si, sm in zip(support_imgs, support_cmasks)]
            support_imgs, support_cmasks = [a[0] for a in supp_aug], [a[1] for a in supp_aug]

        # 3. 处理 Query
        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), size=query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)

        # 4. 处理 Supports
        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        support_masks = []
        support_ignore_idxs = []
        
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), size=support_imgs.size()[-2:], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask.float(), class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
            
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        # 5. 严格对齐 VRP-SAM 的 Batch 格式
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
            'class_id': torch.tensor(class_sample)
        }

        return batch

    def extract_ignore_idx(self, mask, class_id):
        r""" 二值化处理 """
        boundary = torch.zeros_like(mask) 
        
        # 【关键修复】把网络用的 0-12 ID 加 1，变回寻找的真实像素值
        target_pixel_val = class_id + 1
        
        # 把非当前器官的值全变 0，当前器官的值变 1
        mask[mask != target_pixel_val] = 0
        mask[mask == target_pixel_val] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size
        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r""" 返回 Mask 张量 (必须保证像素值是真实的 1-13) """
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name + '.png'))))
        return mask

    def read_img(self, img_name):
        r""" 返回 RGB 图像 """
        # img_name 目前是 "x___ABD_021_95"
        # 1. 扒掉前缀得到真实的图片名
        clean_name = img_name.replace('x___', '')
        
        # 2. 拼接正确的图片路径 (进入 x 子文件夹)
        real_img_path = os.path.join(self.img_path, 'x', clean_name + '.png')
        
        return Image.open(real_img_path).convert('RGB')

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        
        # 获取 Query 图里包含的所有器官真实像素值 (1-13)
        label_class = self.read_mask(query_name).unique().tolist()
        if 0 in label_class:
            label_class.remove(0) # 剔除背景
            
        # 【关键修复】将像素值 (1-13) 平移成网络用的类别 ID (0-12)
        label_class_0_indexed = [c - 1 for c in label_class]
        
        # 过滤出只属于当前 Split 的类别
        valid_classes = [c for c in label_class_0_indexed if c in self.class_ids]
        
        if len(valid_classes) == 0:
            class_sample = self.class_ids[0]
        else:
            class_sample = np.random.choice(valid_classes)
            
        support_names = []
        # 【关键修复】去 txt 字典抽样时，要把 0-12 的 ID 加 1 变回 1-13 的真实像素值
        candidates = self.img_metadata_classwise.get(class_sample + 1, [])
        # _dbg(f"[DEBUG-btcv] sample_episode: idx={idx} query={query_name} "
        #      f"class={class_sample} n_candidates={len(candidates)} shot={self.shot}")
        loop_iter = 0
        while True:
            loop_iter += 1
            if len(candidates) == 0:
                support_name = query_name # 极端兜底：没有同类图，用自己
            else:
                # 换回 np.random，恢复修改前的采样序列
                support_name = np.random.choice(candidates)
            # if loop_iter <= 3:
            #     _dbg(f"[DEBUG-btcv] while iter={loop_iter} 选中={support_name} "
            #          f"已收集={len(support_names)}")
                
            # 修复：candidates 为空时也必须 append，否则死循环
            if query_name != support_name or len(candidates) <= 1: 
                support_names.append(support_name)
            if len(support_names) == self.shot: 
                break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        r""" 交叉验证器官划分 (4-Fold, 严格使用 0-12 索引) """
        nclass_trn = self.nclass // self.nfolds  # 13 // 4 = 3
        
        # 严格从 0 开始索引
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)] 
        
        # 最后一个 fold 包揽剩下的
        if self.fold == self.nfolds - 1:
            class_ids_val = [i for i in range(self.fold * nclass_trn, self.nclass)]
            
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):
        split = 'train' if self.split == 'trn' else 'val'
        # 兼容 PASCAL 的列表存放习惯
        list_file = os.path.join(f'data/splits/lists/btcv/fss_list/{split}/data_list_{self.fold}.txt')
        
        with open(list_file, 'r') as f:
            # 兼容有可能带路径或空格的写法
            metadata = [line.strip().split(' ')[0].split('/')[-1].split('.')[0] for line in f if line.strip()]
        return metadata

    def build_img_metadata_classwise(self):
        split = 'train' if self.split == 'trn' else 'val'
        subclass_file = os.path.join(f'data/splits/lists/btcv/fss_list/{split}/sub_class_file_list_{self.fold}.txt')
            
        with open(subclass_file, 'r') as f:
            subclass_data = f.read()
            
        sub_class_file_list = eval(subclass_data)
        img_metadata_classwise = {}
        for sub_cls in sub_class_file_list.keys():
            # 取出干净的文件名
            img_metadata_classwise[sub_cls] = [data[0].split('/')[-1].split('.')[0] for data in sub_class_file_list[sub_cls]]
        
        # _dbg(f"[DEBUG-btcv] classwise: {subclass_file} → {len(img_metadata_classwise)} 个类别键 {sorted(img_metadata_classwise.keys())}")
        return img_metadata_classwise



class DatasetBTCV2(Dataset):
    BTCV_CLASS_NAMES = ['spleen', 'right_kidney', 'left_kidney', 'gallbladder',
                        'esophagus', 'liver', 'stomach', 'aorta', 'IVC',
                        'portal_vein', 'pancreas', 'right_adrenal', 'left_adrenal']
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        # 规范化 split 命名，对齐 PASCAL 的习惯
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        # BTCV 官方有 13 个腹部器官
        self.nclass = 13 
        self.benchmark = 'btcv2'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        # 你的数据实体路径
        self.img_path = os.path.join(datapath, 'image')
        self.ann_path = os.path.join(datapath, 'mask')
        self.transform = transform

        # 构建 Fold 的类别 ID (决定哪些器官用于训练，哪些用于验证)
        self.class_ids = self.build_class_ids()
        # 类名与 class_ids 对齐（val 打印 per-class 指标用）
        self.class_names = [self.BTCV_CLASS_NAMES[c] for c in self.class_ids]
        
        # 构建情景训练的“双重索引库”
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        # 训练时按实际切片数，验证时仿照 PASCAL 固定测试 1000 个 Episode
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata) 
        
        # 1. 核心：情景采样 (抽 Query -> 定器官 -> 抽 Support)
        query_name, support_names, class_sample = self.sample_episode(idx)
        
        # 2. 读取原始数据
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)

        # 2.5 训练集联合增强：img/mask 同步做翻转 + 旋转 + 弹性形变
        if self.split == 'trn':
            query_img, query_cmask = joint_augment(query_img, query_cmask)
            supp_aug = [joint_augment(si, sm) for si, sm in zip(support_imgs, support_cmasks)]
            support_imgs, support_cmasks = [a[0] for a in supp_aug], [a[1] for a in supp_aug]

        # 3. 处理 Query
        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), size=query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)

        # 4. 处理 Supports
        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        support_masks = []
        support_ignore_idxs = []
        
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), size=support_imgs.size()[-2:], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask.float(), class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
            
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        # 5. 严格对齐 VRP-SAM 的 Batch 格式
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
            'class_id': torch.tensor(class_sample)
        }

        return batch

    def extract_ignore_idx(self, mask, class_id):
        r""" 二值化处理 """
        boundary = torch.zeros_like(mask) 
        
        # 【关键修复】把网络用的 0-12 ID 加 1，变回寻找的真实像素值
        target_pixel_val = class_id + 1
        
        # 把非当前器官的值全变 0，当前器官的值变 1
        mask[mask != target_pixel_val] = 0
        mask[mask == target_pixel_val] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size
        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r""" 返回 Mask 张量 (必须保证像素值是真实的 1-13) """
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name + '.png'))))
        return mask

    def read_img(self, img_name):
        r""" 返回 RGB 图像 """
        # img_name 目前是 "x___ABD_021_95"
        # 1. 扒掉前缀得到真实的图片名
        clean_name = img_name.replace('x___', '')
        
        # 2. 拼接正确的图片路径 (进入 x 子文件夹)
        real_img_path = os.path.join(self.img_path, 'x', clean_name + '.png')
        
        return Image.open(real_img_path).convert('RGB')

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        
        # 获取 Query 图里包含的所有器官真实像素值 (1-13)
        label_class = self.read_mask(query_name).unique().tolist()
        if 0 in label_class:
            label_class.remove(0) # 剔除背景
            
        # 【关键修复】将像素值 (1-13) 平移成网络用的类别 ID (0-12)
        label_class_0_indexed = [c - 1 for c in label_class]
        
        # 过滤出只属于当前 Split 的类别
        valid_classes = [c for c in label_class_0_indexed if c in self.class_ids]
        
        if len(valid_classes) == 0:
            class_sample = self.class_ids[0]
        else:
            class_sample = np.random.choice(valid_classes)
            
        support_names = []
        # 【关键修复】去 txt 字典抽样时，要把 0-12 的 ID 加 1 变回 1-13 的真实像素值
        candidates = self.img_metadata_classwise.get(class_sample + 1, [])
        # _dbg(f"[DEBUG-btcv2] sample_episode: idx={idx} query={query_name} "
        #      f"class={class_sample} n_candidates={len(candidates)} shot={self.shot}")
        loop_iter = 0
        while True:
            loop_iter += 1
            if len(candidates) == 0:
                support_name = query_name # 极端兜底：没有同类图，用自己
            else:
                # 换回 np.random，恢复修改前的采样序列
                support_name = np.random.choice(candidates)
            # if loop_iter <= 3:
            #     _dbg(f"[DEBUG-btcv2] while iter={loop_iter} 选中={support_name} "
            #          f"已收集={len(support_names)}")
                
            # 修复：candidates 为空时也必须 append，否则死循环
            if query_name != support_name or len(candidates) <= 1: 
                support_names.append(support_name)
            if len(support_names) == self.shot: 
                break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        # 直接返回所有类别，不再按 fold/split 划分
        return list(range(self.nclass))

    def build_img_metadata(self):
        split = 'train' if self.split == 'trn' else 'val'
        # 兼容 PASCAL 的列表存放习惯
        list_file = os.path.join(f'data/splits/lists/btcv2/fss_list/{split}/data_list_{self.fold}.txt')
        
        with open(list_file, 'r') as f:
            # 兼容有可能带路径或空格的写法
            metadata = [line.strip().split(' ')[0].split('/')[-1].split('.')[0] for line in f if line.strip()]
        return metadata

    def build_img_metadata_classwise(self):
        split = 'train' if self.split == 'trn' else 'val'
        subclass_file = os.path.join(f'data/splits/lists/btcv2/fss_list/{split}/sub_class_file_list_{self.fold}.txt')
            
        with open(subclass_file, 'r') as f:
            subclass_data = f.read()
            
        sub_class_file_list = eval(subclass_data)
        img_metadata_classwise = {}
        for sub_cls in sub_class_file_list.keys():
            # 取出干净的文件名
            img_metadata_classwise[sub_cls] = [data[0].split('/')[-1].split('.')[0] for data in sub_class_file_list[sub_cls]]
        
        # _dbg(f"[DEBUG-btcv2] classwise: {subclass_file} → {len(img_metadata_classwise)} 个类别键 {sorted(img_metadata_classwise.keys())}")
        return img_metadata_classwise