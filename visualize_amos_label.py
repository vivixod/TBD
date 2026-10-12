"""可视化 AMOS2022_MR 的一张标签。

标签 npz 内是 scipy 稀疏矩阵(15 通道 one-hot, 通道 i 对应类别 i+1)。
输出:
  - vis_amos/amos_0507_0_label_gray.png   黑白(灰度)标签图
  - vis_amos/amos_0507_0_label_color.png  彩色标签图(所有器官一张图)
  - vis_amos/amos_0507_0_image.png        原始图像(参考)
"""
import os
import json
import shutil
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

BASE = '/vcl/2025zhutaotao/AMOS2022_MR'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vis_amos')
os.makedirs(OUT, exist_ok=True)

# 要处理的切片 ID 列表 (对应 image/x/<id>.png, 标签从 dataset.json 自动查找)
SLICE_IDS = ['amos_0507_21', 'amos_0507_23']

# 类别定义 (dataset.json)
LABELS = {
    0: 'background',
    1: 'spleen',
    2: 'kidney_right',
    3: 'kidney_left',
    4: 'gallbladder',
    5: 'esophagus',
    6: 'liver',
    7: 'stomach',
    8: 'aorta',
    9: 'inferior_vena_cava',
    10: 'pancreas',
    11: 'adrenal_gland_right',
    12: 'adrenal_gland_left',
    13: 'duodenum',
    14: 'bladder',
    15: 'prostate_and_uterus',
}


def load_label(path):
    z = np.load(path, allow_pickle=True)
    fmt = str(z['format'])
    shape = tuple(int(s) for s in z['shape'])
    if 'csr' in fmt:
        m = sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=shape)
    elif 'csc' in fmt:
        m = sp.csc_matrix((z['data'], z['indices'], z['indptr']), shape=shape)
    else:
        raise ValueError(f'unknown sparse format {fmt}')
    return m.toarray(), shape, fmt


def find_slice(slice_id):
    """从 dataset.json 查找切片对应的 image/label 相对路径."""
    with open(os.path.join(BASE, 'dataset.json')) as f:
        d = json.load(f)
    for item in d['training'] + d['test']:
        if item['image'].endswith(f'/{slice_id}.png'):
            return item['image'], item['label']
    raise FileNotFoundError(f'no entry for {slice_id} in dataset.json')


def process_slice(slice_id):
    img_rel, lbl_rel = find_slice(slice_id)
    img_name = os.path.basename(img_rel)
    lbl_name = os.path.basename(lbl_rel)
    save_tag = slice_id
    img_path = os.path.join(BASE, img_rel.replace('/', os.sep))
    lbl_path = os.path.join(BASE, lbl_rel.replace('/', os.sep))

    arr, shape, fmt = load_label(lbl_path)
    print('label npz ->', fmt, 'saved shape', shape, 'dense shape', arr.shape)

    # npz 内为展平后的 (15, H*W); 原始形状从文件名 (15, H, W, 1) 自动解析
    import re
    m = re.search(r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', lbl_name)
    assert m, f'无法从文件名解析形状: {lbl_name}'
    n_cls, H, W, D = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
    print(f'parsed shape from filename: ({n_cls}, {H}, {W}, {D})')
    onehot = arr.reshape(n_cls, H, W, D)
    onehot = onehot[:, :, :, 0]  # (15, H, W)

    # 通道 i -> 类别 i+1; 全 0 为背景
    label_map = (np.argmax(np.concatenate(
        [np.zeros((1, H, W), dtype=np.float32), onehot.astype(np.float32)],
        axis=0), axis=0))
    print('label_map shape:', label_map.shape)
    print('present classes:', sorted(np.unique(label_map).tolist()))
    for c in sorted(np.unique(label_map).tolist()):
        print(f'  class {c:2d} {LABELS[c]:20s} pixel count {int((label_map == c).sum())}')

    img = plt.imread(img_path)
    print('image shape:', img.shape, 'dtype:', img.dtype)

    # ---- 1. 黑白标签图: 二值(器官=白, 背景=黑) ----
    binary = ((label_map > 0) * 255).astype(np.uint8)
    plt.imsave(os.path.join(OUT, f'{save_tag}_label_binary.png'), binary,
               cmap='gray', vmin=0, vmax=255)

    # ---- 1b. 灰度标签图: 出现的类别映射到可区分的灰度级 ----
    present = [c for c in sorted(np.unique(label_map).tolist()) if c != 0]
    gray = np.zeros_like(label_map, dtype=np.float32)
    if present:
        vals = np.linspace(60, 255, len(present))
        for c, v in zip(present, vals):
            gray[label_map == c] = v
    plt.imsave(os.path.join(OUT, f'{save_tag}_label_gray.png'), gray,
               cmap='gray', vmin=0, vmax=255)

    # ---- 2. 彩色标签图: 每个器官一种颜色 ----
    n_all = len(LABELS)
    cmap = plt.cm.get_cmap('tab20', n_all)
    cmap_use = ListedColormap([cmap(i)[:3] for i in range(n_all)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original image')
    axes[0].axis('off')

    im1 = axes[1].imshow(binary, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Label (black-white, binary)')
    axes[1].axis('off')

    im2 = axes[2].imshow(label_map, cmap=cmap_use, vmin=0, vmax=n_all - 1,
                         interpolation='nearest')
    axes[2].set_title('Label (colored)')
    axes[2].axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'{save_tag}_label_color.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)

    # 单独保存一张纯彩色标签大图(不含图例/标题, 方便直接查看或当切片用)
    fig2, ax2 = plt.subplots(figsize=(W / 100, H / 100))
    ax2.imshow(label_map, cmap=cmap_use, vmin=0, vmax=n_all - 1,
               interpolation='nearest')
    ax2.axis('off')
    fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig2.savefig(os.path.join(OUT, f'{save_tag}_label_color_only.png'),
                 dpi=100, pad_inches=0)
    plt.close(fig2)

    # ---- 3. 数据集级颜色图例(所有切片共用一份, 覆盖全部 16 类) ----
    legend_path = os.path.join(OUT, 'amos_class_legend.png')
    if not os.path.exists(legend_path):
        handles = []
        for c in range(n_all):
            color = cmap_use(c / float(n_all - 1)) if n_all > 1 else cmap_use(0)
            handles.append(mpatches.Patch(color=color, label=f'{c}: {LABELS[c]}'))
        fig3 = plt.figure(figsize=(3, 4))
        fig3.legend(handles=handles, loc='center', fontsize=9, title='AMOS classes')
        fig3.savefig(legend_path, bbox_inches='tight', dpi=150)
        plt.close(fig3)
        print('saved dataset-level legend:', legend_path)

    # ---- 4. 复制原始图像到输出目录 ----
    img_copy = os.path.join(OUT, f'{save_tag}_image.png')
    shutil.copy(img_path, img_copy)

    print('saved:', save_tag)


def main():
    for sid in SLICE_IDS:
        print('=' * 60)
        print('processing slice:', sid)
        process_slice(sid)
    print('\nall done. output dir:', OUT)


if __name__ == '__main__':
    main()
