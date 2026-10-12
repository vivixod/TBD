"""从 BTCV 中挑选: 1 张器官最多的 query 切片 + 4 张合适的参考集(support)切片.

规则:
  - Query 取自 test 集, 选择包含器官类别最多的切片(平局时选前景面积最大的)
  - 目标器官 = query 中面积最大的器官
  - Support 取自 train 集(与 query 不同扫描), 优先包含目标器官、且目标器官面积大、
    覆盖 query 器官多的切片, 来自 4 个不同扫描
输出到 VRP-SAM/btcv-temp: 每张切片的 image + mask + 信息说明 + 总览图
"""
import os
import json
import shutil
import numpy as np
import PIL.Image as Image

BTCV = '/vcl/2025zhutaotao/BTCV'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'btcv-temp')
os.makedirs(OUT, exist_ok=True)

LABELS = {
    0: 'background', 1: 'spleen', 2: 'kidney_right', 3: 'kidney_left',
    4: 'gallbladder', 5: 'esophagus', 6: 'liver', 7: 'stomach', 8: 'aorta',
    9: 'inferior_vena_cava', 10: 'portal_vein_and_splenic_vein',
    11: 'pancreas', 12: 'adrenal_gland_right', 13: 'adrenal_gland_left',
}


def scan_of(name):
    """ABD_001_100 -> ABD_001"""
    return name.split('_')[0] + '_' + name.split('_')[1]


def read_mask_vals(clean_name):
    p = os.path.join(BTCV, 'mask', f'x___{clean_name}.png')
    m = np.array(Image.open(p))
    return m


def slice_stats(clean_name):
    m = read_mask_vals(clean_name)
    vals, cnts = np.unique(m, return_counts=True)
    organs = {int(v): int(c) for v, c in zip(vals, cnts) if v != 0}
    return organs, int((m > 0).sum())


def main():
    d = json.load(open(os.path.join(BTCV, 'dataset.json')))
    train_names = [os.path.basename(it['image'])[:-4] for it in d['training']]
    test_names = [os.path.basename(it['image'])[:-4] for it in d['test']]
    print(f'train={len(train_names)} test={len(test_names)}')

    # ---- 1. Query: test 集中器官数最多的切片 ----
    best = None
    for name in test_names:
        organs, fg = slice_stats(name)
        score = (len(organs), fg)
        if best is None or score > best[1]:
            best = (name, score, organs)
    q_name, (n_org, fg), q_organs = best
    q_fg = fg
    print(f'query: {q_name}  organs={n_org}  fg={fg}')
    print('  organ composition:', {k: LABELS[k] for k in q_organs})

    # 目标器官 = query 中面积最大的器官
    target = max(q_organs, key=q_organs.get)
    print(f'target organ: {target} ({LABELS[target]}) area={q_organs[target]}')

    # ---- 2. Support: train 集中找 4 张合适的参考切片 ----
    cands = []
    for name in train_names:
        organs, fg = slice_stats(name)
        if target not in organs:
            continue
        overlap = len(set(organs) & set(q_organs))
        # 评分: 目标器官面积为主, 覆盖 query 器官数 + 总前景为辅
        score = (organs[target], overlap, fg)
        cands.append((name, score, organs))
    cands.sort(key=lambda x: x[1], reverse=True)

    # 按扫描去重, 取 4 个不同扫描
    picks = []
    used_scans = {scan_of(q_name)}
    for name, score, organs in cands:
        sc = scan_of(name)
        if sc in used_scans:
            continue
        picks.append((name, organs))
        used_scans.add(sc)
        if len(picks) == 4:
            break
    print(f'\n4 supports:')
    for i, (name, organs) in enumerate(picks):
        print(f'  support{i + 1}: {name}  organs={len(organs)} '
              f'{ {k: LABELS[k] for k in sorted(organs)} }')

    # ---- 3. 复制文件到 btcv-temp ----
    def copy_pair(src_clean, dst_prefix):
        shutil.copy(os.path.join(BTCV, 'image', 'x', src_clean + '.png'),
                    os.path.join(OUT, f'{dst_prefix}_image.png'))
        shutil.copy(os.path.join(BTCV, 'mask', f'x___{src_clean}.png'),
                    os.path.join(OUT, f'{dst_prefix}_mask.png'))

    copy_pair(q_name, 'query')
    for i, (name, _) in enumerate(picks):
        copy_pair(name, f'support{i + 1}')

    # 信息说明
    lines = [
        'BTCV few-shot 演示切片 (query + 4 supports)',
        '',
        f'query  : {q_name}  ({n_org} organs, fg={q_fg})',
        f'target : class {target} = {LABELS[target]} (query 中面积最大)',
        '',
        'support 选取规则: train 集中含目标器官、面积大、与 query 器官重叠多, 来自不同扫描',
    ]
    for i, (name, organs) in enumerate(picks):
        lines.append(f'support{i + 1}: {name}  organs={sorted(organs)}')
    with open(os.path.join(OUT, 'info.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\nfiles saved to', OUT)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
