r"""生成 fss_list split 列表（patient-level 划分）。

支持的已处理数据集（格式统一：image/{x,y}/*.png + label/*.npz + dataset.json）:
  - ADAM2020   (/vcl/2025zhutaotao/ADAM2020)       2 前景类
  - AMOS2022_MR(/vcl/2025zhutaotao/AMOS2022_MR)    15 前景类
  - BraTS2019  (/vcl/2025zhutaotao/BraTS2019)      3 前景类

协议（与 data/amos.py 对齐）:
  - patient-level 9:1 划分，seed=321（ratio/seed 可调）
  - ADAM 的 baseline(B)/follow-up(F) 归同一 patient，避免泄漏
  - data_list_0.txt:      每行一个 image_name（含 x___/y___ 前缀）
  - sub_class_file_list_0.txt: dict literal，key 为 1-based 像素类别
      {1: [['x___a'], ...], 2: [...]}   ← sample_episode 用 class_sample+1 查询

用法:
  python data/splits/gen_splits.py                 # 生成全部三个
  python data/splits/gen_splits.py --datasets adam,amos_mr,brats --ratio 0.9 --seed 321
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

DATASETS = {
    'adam': '/vcl/2025zhutaotao/ADAM2020',
    'amos': '/vcl/2025zhutaotao/AMOS2022_MR',
    'brats': '/vcl/2025zhutaotao/BraTS2019',
}

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lists')


def split_prefix(image_name):
    """'x___amos_0507_0' -> ('x', 'amos_0507_0')"""
    m = re.match(r'^([xy])___(.+)$', image_name)
    assert m, f'unknown prefix: {image_name}'
    return m.group(1), m.group(2)


def patient_id(image_name):
    """提取 patient id。

    - AMOS : x___amos_0507_0      -> amos_0507
    - ADAM : x___10059B_61        -> 10059   (B/F 基线/随访归同一患者)
    - BraTS: x___HGG__BraTS19_2013_10_1_100 -> HGG__BraTS19_2013_10_1
    """
    _, clean = split_prefix(image_name)
    core = re.sub(r'_\d+$', '', clean)          # 去掉尾部 slice 号
    if core and core[-1] in 'BF':               # ADAM baseline/follow-up 同患者
        core = core[:-1]
    return core


def image_name_from_label(fname):
    """'x___10027_26.(2, 560, 560, 1).npz' -> 'x___10027_26'"""
    return re.sub(r'\.\(.*?\)\.npz$', '', fname)


def has_classes(npz_path):
    """返回该 npz 包含的前景类别（1-based 像素值列表）。

    CSR indptr: 通道 c 非空 ⇔ indptr[c] < indptr[c+1]，通道 c 对应像素类别 c+1。
    只读 indptr，不解压 data。
    """
    z = np.load(npz_path, mmap_mode='r')
    indptr = z['indptr']
    n_class = int(z['shape'][0])
    return [c for c in range(1, n_class + 1) if indptr[c - 1] < indptr[c]]


def build(name, datapath, ratio, seed, rare_thresh):
    print(f'\n===== {name} ({datapath}) =====')
    j = json.load(open(os.path.join(datapath, 'dataset.json')))

    # ---- 1. 收集 (image_name, patient, classes)，npz 只读一遍 ----
    records = []
    for t in j['training']:
        fname = os.path.basename(t['label'])
        image_name = image_name_from_label(fname)
        npz_path = os.path.join(datapath, t['label'])
        records.append((image_name, patient_id(image_name), has_classes(npz_path)))
    assert len(records) == j['numTraining'], f"{len(records)} != numTraining {j['numTraining']}"

    patients = sorted(set(p for _, p, _ in records))
    patients_of = defaultdict(set)
    for _, p, cls in records:
        patients_of[p].update(cls)
    print(f'  slices={len(records)}  patients={len(patients)}')

    rng = np.random.RandomState(seed)

    # ---- 2. 稀有类覆盖约束：出现患者数 <= rare_thresh 的类，固定 1 个患者进 val
    #         （"一边放一个"：保证 train 仍保留该类的另一个患者）
    class_pat = defaultdict(list)
    for p, cs in patients_of.items():
        for c in cs:
            class_pat[c].append(p)
    rare_classes = [c for c, ps in class_pat.items() if len(ps) <= rare_thresh]
    V_req = set()
    for c in sorted(rare_classes):
        if any(c in patients_of[p] for p in V_req):
            continue
        cands = [p for p in class_pat[c] if p not in V_req]
        pick = cands[int(rng.randint(len(cands)))]
        V_req.add(pick)
        print(f'  [coverage] 稀有类 {c} 仅 {len(class_pat[c])} 患者 → 固定 {pick} 进 val')

    n_val = max(len(V_req), int(round(len(patients) * (1 - ratio))))
    rest = [p for p in patients if p not in V_req]
    order = rng.permutation(len(rest))
    extra = [rest[i] for i in order[:n_val - len(V_req)]]
    val_pat = set(V_req) | set(extra)
    trn_recs = [r for r in records if r[1] not in val_pat]
    val_recs = [r for r in records if r[1] in val_pat]
    print(f'  train: {len(set(p for _, p, _ in trn_recs))} patients / {len(trn_recs)} slices | '
          f'val: {len(set(p for _, p, _ in val_recs))} patients / {len(val_recs)} slices')

    # ---- 3. 类候选列表（1-based key）----
    def classwise(recs):
        cls_map = {}
        for image_name, _, cls in recs:
            for c in cls:
                cls_map.setdefault(c, []).append([image_name])
        if cls_map:
            for c in range(1, max(cls_map.keys()) + 1):
                cls_map.setdefault(c, [])
        return {k: cls_map[k] for k in sorted(cls_map)}

    trn_cls = classwise(trn_recs)
    val_cls = classwise(val_recs)
    print('  per-class candidates (train):',
          {k: len(v) for k, v in trn_cls.items()})
    print('  per-class candidates (val)  :',
          {k: len(v) for k, v in val_cls.items()})

    # ---- 4. 写文件 ----
    for split, recs, cls_map in [('train', trn_recs, trn_cls),
                                 ('val', val_recs, val_cls)]:
        out_dir = os.path.join(OUT_ROOT, name, 'fss_list', split)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'data_list_0.txt'), 'w') as f:
            for image_name, _, _ in recs:
                f.write(image_name + '\n')
        with open(os.path.join(out_dir, 'sub_class_file_list_0.txt'), 'w') as f:
            f.write(repr(cls_map))
        print(f'  wrote {out_dir}/data_list_0.txt ({len(recs)} lines) + sub_class_file_list_0.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='adam,amos,brats',
                        help='逗号分隔，可选: adam, amos, brats')
    parser.add_argument('--ratio', type=float, default=0.8, help='训练 patient 比例（默认 8:2）')
    parser.add_argument('--rare-thresh', type=int, default=2,
                        help='稀有类阈值：出现患者数 <= 此值的类，固定 1 个患者进 val（一边放一个）')
    parser.add_argument('--seed', type=int, default=321)
    args = parser.parse_args()

    for name in [s.strip() for s in args.datasets.split(',') if s.strip()]:
        assert name in DATASETS, f'unknown dataset {name}, available: {list(DATASETS)}'
        build(name, DATASETS[name], args.ratio, args.seed, args.rare_thresh)
    print('\nDone.')
