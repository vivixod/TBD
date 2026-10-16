"""Snapshot the experiment logs and compare only matching completed epochs."""
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'nearest': 'trn_joint4_shot4_cb0.5.log',
    'max': 'trn_joint4_shot4_cb0.5.log_1015_151028',
    'adam_only_masks': 'trn_joint4_shot4_cb0.5_pooladam-only_bboxm1_iclm1.log',
    'quality': 'trn_joint4_shot4_cb0.5_poolmax_bboxm1_iclm1_fusionquality_qw0.1_qt0.2_fw2_fr3.log',
}
LABELS = dict(nearest='nearest，无屏蔽、等权', max='全局max，无屏蔽、等权',
              adam_only_masks='ADAM-only max，双屏蔽、等权', quality='全局max，双屏蔽、质量加权')
DATASETS = ['btcv', 'amos', 'adam', 'brats']


def parse(path):
    text = path.read_text()
    config = dict(re.findall(r'^\|\s*(\w+):\s*(.*?)\s*$', text, re.M))
    metrics, classes, fusion = {}, {}, []
    pending = None
    epoch = None
    for line in text.splitlines():
        match = re.search(r'\*\*\* val/(\w+) \[@Epoch (\d+)\].*?Dice:\s*([\d.]+)', line)
        if match:
            dataset, epoch = match[1], int(match[2])
            metrics.setdefault(epoch, {})[dataset] = float(match[3])
            pending = dataset, epoch
        elif line.startswith('[per-class Dice]') and pending:
            dataset, current_epoch = pending
            classes.setdefault(current_epoch, {})[dataset] = {
                k: float(v) for k, v in re.findall(r'([\w/]+)=([\d.]+)', line)}
            pending = None
        macro = re.search(r'^\[val\] \[macro\] Dice=([\d.]+)', line)
        if macro and epoch is not None:
            metrics[epoch]['macro'] = float(macro[1])
        monitor = re.match(r'^\[Support fusion ([^\]]+)\] epoch=(\d+)', line)
        if monitor:
            values = dict(re.findall(r'(mix|quality_loss|entropy|max_weight|selected_Dice|best_shot_Dice)=([\d.]+)', line))
            fusion.append(dict(phase=monitor[1], epoch=int(monitor[2]),
                               **{k: float(v) for k, v in values.items()}))
    config['pooling_inferred'] = config.get('support_mask_pooling',
        'max' if '[Support mask] downsample=adaptive_max_pool2d' in text else 'nearest')
    config['batches_per_epoch'] = int(re.search(r'\[Batch: 0001/(\d+)\]', text)[1])
    complete = sorted(e for e, values in metrics.items() if all(d in values for d in DATASETS + ['macro']))
    saves = re.findall(r'Model saved @(\d+) w/ val\. Dice:\s*([\d.]+)', text)
    return dict(path=str(path), source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                config=config, metrics=metrics, classes=classes, fusion=fusion,
                complete_epochs=complete, last_save=saves[-1] if saves else None)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    experiments = {name: parse(ROOT / 'logs' / folder / 'log.txt') for name, folder in SOURCES.items()}
    common = sorted(set.intersection(*(set(run['complete_epochs']) for run in experiments.values())))
    last = max(common)
    prefix = ROOT / 'analysis' / f'quality_same_epoch_20261016_ep{last}'
    epoch_rows, class_rows, means = [], [], []
    for name, run in experiments.items():
        for epoch in common:
            epoch_rows.append(dict(experiment=name, epoch=epoch, **run['metrics'][epoch]))
            for dataset, values in run['classes'][epoch].items():
                for label, value in values.items():
                    class_rows.append(dict(experiment=name, epoch=epoch, dataset=dataset,
                                           class_name=label, dice=value))
        means.append(dict(experiment=name, first_epoch=min(common), last_epoch=last,
                          **{d: sum(run['metrics'][e][d] for e in common) / len(common)
                             for d in DATASETS + ['macro']}))
    write_csv(prefix.with_suffix('.csv'), epoch_rows)
    write_csv(Path(str(prefix) + '_classes.csv'), class_rows)
    write_csv(Path(str(prefix) + '_means.csv'), means)
    prefix.with_suffix('.json').write_text(json.dumps(dict(common_epochs=common, experiments=experiments), indent=2))

    lines = [f'# 质量融合实验：同epoch对比（最新完整epoch {last}）', '',
             '对比同一epoch，包含不同池化、attention和GPU卡数设置，不作评分头的单因素因果归因。', '',
             f'## Epoch {last}', '', '|设置|BTCV|AMOS|ADAM|BraTS|Macro|', '|---|---:|---:|---:|---:|---:|']
    for name, run in experiments.items():
        m = run['metrics'][last]
        lines.append('|' + LABELS[name] + '|' + '|'.join(f'{m[d]:.2f}' for d in DATASETS + ['macro']) + '|')
    lines += ['', f'## 共同epoch {min(common)}–{last}均值', '',
              '|设置|BTCV|AMOS|ADAM|BraTS|Macro|', '|---|---:|---:|---:|---:|---:|']
    for row in means:
        lines.append('|' + LABELS[row['experiment']] + '|' + '|'.join(f'{row[d]:.2f}' for d in DATASETS + ['macro']) + '|')
    lines += ['', '## 重点类别：最新共同epoch', '',
              '|数据集 / 类别|nearest|全局max等权|ADAM-only双屏蔽等权|质量加权|', '|---|---:|---:|---:|---:|']
    targets = {'adam': ['aneurysm', 'treated_aneurysm'],
               'btcv': ['right_adrenal', 'left_adrenal', 'portal_vein', 'pancreas', 'esophagus'],
               'amos': ['bladder', 'prostate_and_uterus', 'adrenal_gland_right', 'adrenal_gland_left', 'duodenum'],
               'brats': ['necrotic_tumor_core', 'peritumoral_edema', 'enhancing_tumor']}
    for dataset, labels in targets.items():
        for label in labels:
            values = [run['classes'][last][dataset][label] for run in experiments.values()]
            lines.append('|' + dataset + '/' + label + '|' + '|'.join(f'{v:.1f}' for v in values) + '|')
    lines += ['', '## 比较口径与评分头', '',
              '旧三组均2667个batch/epoch，当前两卡组4000个batch/epoch；每卡batch=1。',
              '相同epoch时，新组优化步数约多50%；验证的类别和support抽样也会随rank数变化。',
              'AMOS稀有类只有一位验证患者，且每类只有少量query被随机选中；单轮结果波动很大。',
              'mean_weights接近均匀本身不能说明评分无效，因为不同query可能偏好不同shot；',
              'max_weight和归一化entropy更能反映每个query内部是否拉开权重。',
              'selected_Dice和best_shot_Dice是逐query均值，不能与类别汇总的验证Dice直接相减。',
              '当前只做日志分析，逐support的评分与Dice对应关系留待后续训练更充分时检查。', '',
              '## 本次快照的评分头监控（含未完成整轮验证的分数据集结果）', '',
              '|phase|epoch|mix|max weight|entropy|selected Dice|best-shot Dice|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in experiments['quality']['fusion']:
        lines.append(f'|{row["phase"]}|{row["epoch"]}|{row["mix"]:.3f}|{row["max_weight"]:.3f}'
                     f'|{row["entropy"]:.3f}|{row["selected_Dice"]:.2f}|{row["best_shot_Dice"]:.2f}|')
    prefix.with_suffix('.md').write_text('\n'.join(lines) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    colors = ['#888888', '#cf9736', '#447cbb', '#b84e58']
    for ax, dataset in zip(axes.flat, DATASETS + ['macro']):
        for (name, run), color in zip(experiments.items(), colors):
            ax.plot(common, [run['metrics'][e][dataset] for e in common], marker='o', label=name, color=color)
        ax.set_title(dataset.upper()); ax.set_xlabel('Epoch (0-based)'); ax.set_ylabel('Validation Dice (%)')
        ax.set_xticks(common); ax.grid(alpha=.2)
    for dataset, color in zip(DATASETS, colors):
        rows = [r for r in experiments['quality']['fusion'] if r['phase'] == 'val/' + dataset]
        axes.flat[5].plot([r['epoch'] for r in rows], [r['max_weight'] for r in rows], marker='o', label=dataset, color=color)
    axes.flat[5].axhline(.25, color='black', linestyle='--', linewidth=1)
    axes.flat[5].set_title('Mean maximum support weight'); axes.flat[5].set_xlabel('Epoch (0-based)')
    axes.flat[5].set_ylim(.245, .35); axes.flat[5].legend(fontsize=8); axes.flat[5].grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Matched epochs; settings and GPU counts differ')
    fig.savefig(prefix.with_suffix('.png'), dpi=170)
    print(prefix.with_suffix('.md'))
    print('\n'.join(lines[:24]))


if __name__ == '__main__':
    main()
