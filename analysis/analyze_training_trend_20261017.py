"""Snapshot logs: matched-window per-class comparisons and quality-weight trends.

No model inference or checkpoint loading. All epoch numbers are zero-based log
indices, and only completed four-dataset validations enter comparisons.
"""
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
from zoneinfo import ZoneInfo

from compare_quality_same_epoch_20261016 import parse, SOURCES, DATASETS, LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/training_trend_20261017_ep33'
METRICS = DATASETS + ['macro']


def average(run, start, end):
    assert all(e in run['complete_epochs'] for e in range(start, end + 1))
    return {d: statistics.mean(run['metrics'][e][d] for e in range(start, end + 1))
            for d in METRICS}


def class_average(run, dataset, label, start, end):
    return statistics.mean(run['classes'][e][dataset][label] for e in range(start, end + 1))


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    runs = {name: parse(ROOT / 'logs' / folder / 'log.txt') for name, folder in SOURCES.items()}
    q = runs['quality']
    assert max(q['complete_epochs']) >= 33
    # Record the actual logs read by the parser, not a second evolving read.
    epochs, classes = [], []
    for name, run in runs.items():
        for e in run['complete_epochs']:
            for dataset in DATASETS:
                values = run['classes'][e][dataset]
                assert len(values) == {'btcv': 13, 'amos': 15, 'adam': 2, 'brats': 3}[dataset]
                assert abs(statistics.mean(values.values()) - run['metrics'][e][dataset]) < .065
                classes += [dict(experiment=name, epoch=e, dataset=dataset, class_name=label, dice=value)
                            for label, value in values.items()]
            assert abs(statistics.mean(run['metrics'][e][d] for d in DATASETS) - run['metrics'][e]['macro']) < .015
            epochs.append(dict(experiment=name, epoch=e, **run['metrics'][e]))
    csv_write(OUT / 'epoch_metrics.csv', epochs)
    csv_write(OUT / 'class_metrics.csv', classes)
    comparisons, class_comparisons = [], []
    for label, start, end, names in [
        ('all_common', 19, 23, list(runs)),
        ('closest_configuration', 26, 30, ['nearest', 'adam_only_masks', 'quality']),
        ('recent_matched', 29, 33, ['nearest', 'quality']),
    ]:
        for name in names:
            comparisons.append(dict(comparison=label, experiment=name, first_epoch=start,
                                    last_epoch=end, **average(runs[name], start, end)))
    csv_write(OUT / 'dataset_comparisons.csv', comparisons)
    for old_name, start, end in [('adam_only_masks', 26, 30), ('nearest', 29, 33), ('max', 19, 23)]:
        for dataset in DATASETS:
            for label in q['classes'][end][dataset]:
                old = class_average(runs[old_name], dataset, label, start, end)
                new = class_average(q, dataset, label, start, end)
                class_comparisons.append(dict(reference=old_name, first_epoch=start, last_epoch=end,
                                              dataset=dataset, class_name=label, old_mean=old,
                                              new_matched_mean=new, delta=new-old,
                                              new_recent_mean=class_average(q, dataset, label, 29, 33),
                                              new_epoch33=q['classes'][33][dataset][label]))
    csv_write(OUT / 'class_comparisons.csv', class_comparisons)
    trend_windows = [(0, 4), (5, 9), (10, 14), (15, 19), (19, 23), (24, 28), (29, 33)]
    trends = [dict(first_epoch=a, last_epoch=b, **average(q, a, b)) for a, b in trend_windows]
    csv_write(OUT / 'quality_trend_windows.csv', trends)
    fusion_rows = [{k: v for k, v in r.items() if k not in ('selected_Dice', 'best_shot_Dice')}
                   for r in q['fusion'] if r['epoch'] <= 33]
    # Preserve the rounded shot-position means separately; they are not a
    # measure of within-query separation and are not used to judge efficacy.
    log_text = (ROOT / 'logs' / SOURCES['quality'] / 'log.txt').read_text()
    for line in log_text.splitlines():
        match = re.match(r'^\[Support fusion ([^\]]+)\] epoch=(\d+)', line)
        weights = re.search(r'mean_weights=(\[[^\]]+\])', line)
        if match and weights:
            for row in fusion_rows:
                if row['phase'] == match[1] and row['epoch'] == int(match[2]):
                    row['mean_weights'] = json.loads(weights[1]); break
    csv_write(OUT / 'fusion_epoch_metrics.csv', fusion_rows)
    fusion_windows = []
    for phase in ['train'] + ['val/' + d for d in DATASETS]:
        for start, end in [(8, 12), (19, 23), (24, 28), (29, 33)]:
            rows = [r for r in fusion_rows if r['phase'] == phase and start <= r['epoch'] <= end]
            fusion_windows.append(dict(phase=phase, first_epoch=start, last_epoch=end,
                                       **{k: statistics.mean(r[k] for r in rows)
                                          for k in ['max_weight', 'entropy', 'quality_loss']}))
    csv_write(OUT / 'fusion_window_metrics.csv', fusion_windows)
    training = []
    pattern = r'^\*\*\* Training \[@Epoch (\d+)\] Avg L:\s*([\d.]+)\s+Dice:\s*([\d.]+)'
    for m in re.finditer(pattern, log_text, re.M):
        if int(m[1]) <= 33:
            training.append(dict(epoch=int(m[1]), loss=float(m[2]), dice=float(m[3])))
    csv_write(OUT / 'training_metrics.csv', training)
    data = dict(snapshot_time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                reporting_cutoff_epoch=33, experiments=runs, comparisons=comparisons,
                class_comparisons=class_comparisons, trends=trends,
                fusion_windows=fusion_windows, fusion_epoch_metrics=fusion_rows, training=training)
    (OUT / 'snapshot.json').write_text(json.dumps(data, indent=2))

    lines = ['# ReQ-SAM 联合训练：epoch 33 趋势与逐类比较', '',
             '仅分析日志，不做模型推理或单路 Dice 诊断。epoch 使用日志中的编号；当前已完成 epoch 33 验证。', '',
             '当前配置：全局 max pooling、双 support attention 屏蔽、quality 加权、类别均衡概率 0.5、4-shot。',
             '主要旧配置：仅 ADAM max pooling、双屏蔽、mean 融合，完整到 epoch 30。',
             '其他参考：nearest/无屏蔽/mean 完整到45；全局max/无屏蔽/mean完整到23。', '',
             '**旧三组每轮2667步，当前两卡每轮4000步；同epoch的新组更新步数约多50%，全局batch不同，验证类别/support抽样也随rank数变化。以下是组合设置对比，不能单独归因于评分头。**', '',
             '## 当前阶段', '', '|区间|BTCV|AMOS|ADAM|BraTS|四集Macro|', '|---|---:|---:|---:|---:|---:|']
    for row in trends:
        lines.append('|' + f'{row["first_epoch"]}–{row["last_epoch"]}' + '|' + '|'.join(f'{row[d]:.2f}' for d in METRICS) + '|')
    best = int(q['last_save'][0])
    lines += ['', f'最佳 checkpoint 为 epoch {best}，Macro={q["metrics"][best]["macro"]:.2f}；epoch33 Macro={q["metrics"][33]["macro"]:.2f}。',
              '最近29–33与不重叠的24–28比较：BTCV +1.35、AMOS +0.33、ADAM -1.04、BraTS +0.02、Macro +0.16。',
              '整体上升明显放缓，BTCV仍提升；AMOS与BraTS趋于平台；ADAM未治疗类仍波动。epoch28的低点之后已恢复，不宜仅凭单轮判定整体退化。', '',
              '## 同epoch五轮均值比较']
    for group in dict.fromkeys(r['comparison'] for r in comparisons):
        subset = [r for r in comparisons if r['comparison'] == group]
        lines += ['', f'### {group}: epoch {subset[0]["first_epoch"]}–{subset[0]["last_epoch"]}', '',
                  '|设置|BTCV|AMOS|ADAM|BraTS|Macro|', '|---|---:|---:|---:|---:|---:|']
        for row in subset:
            lines.append('|' + LABELS[row['experiment']] + '|' + '|'.join(f'{row[d]:.2f}' for d in METRICS) + '|')
    lines += ['', '## 每一类：与前一版双屏蔽实验比较', '',
              '旧值与新值均为相同 epoch26–30 的五轮均值；末列是新实验最近29–33五轮均值，仅用于看后续走向，不与旧26–30直接相减。']
    for dataset in DATASETS:
        lines += ['', '### ' + dataset.upper(), '',
                  '|类别|旧26–30|新26–30|同epoch变化 pp|新29–33|', '|---|---:|---:|---:|---:|']
        for row in class_comparisons:
            if row['reference'] == 'adam_only_masks' and row['dataset'] == dataset:
                lines.append(f'|{row["class_name"]}|{row["old_mean"]:.2f}|{row["new_matched_mean"]:.2f}|{row["delta"]:+.2f}|{row["new_recent_mean"]:.2f}|')
    lines += ['', '## 各数据集的实际变化', '',
              '- BTCV：同26–30比前一版总体+0.56，主要增益为胆囊+10.80、左肾上腺+9.56；右肾上腺-3.90、IVC-3.42、肝-2.92、食管-2.56。门静脉同期基本持平，最近五轮才进一步升至42.26。',
              '- AMOS：同26–30总体+4.74，但膀胱1.92→39.72、前列腺/子宫14.66→54.44主导均值改善；剔除这两类，其余13类70.14→69.65。右肾上腺+17.74、胰腺+1.84，但胆囊-10.06、主动脉-10.88、十二指肠-7.96。',
              '- ADAM：同26–30未治疗动脉瘤5.04→14.56，已治疗89.42→89.10。最近五轮未治疗均值15.88，范围10.3–20.4；有改善但仍很低，未出现持续稳定突破。已治疗稳定在约89。',
              '- BraTS：同26–30总体仅+0.16，坏死核-1.22、水肿-0.16、增强肿瘤+1.88；相对更早nearest，最近五轮分别+0.40/-1.78/+3.58。主要改善是增强类，水肿仍偏弱。', '',
              'AMOS当前膀胱仅2个不同验证query、前列腺/子宫仅4个，均来自一位患者。当前实验内每轮验证固定seed，波动不是每轮重新换患者，而是极少数固定query的预测随模型更新变化；跨不同GPU卡数实验，query/class/support清单又有差别。',
              '膀胱最近五轮为70.5、25.4、0、6.1、54.9；前列腺/子宫为44.3、58.7、52.9、46.1、57.7。前者尚未稳定，后者近期更稳定，但均不足以判断患者泛化。', '',
              '## 评分权重是否变化', '',
              '只读取权重监控，不分析单路预测。下面为每个query最大权重的均值，不是四个shot位置的平均权重。', '',
              '|阶段|epoch8–12|maxW epoch29–33|变化|最新epoch33|最新entropy|',
              '|---|---:|---:|---:|---:|---:|']
    for phase in ['train'] + ['val/' + d for d in DATASETS]:
        early = next(r for r in fusion_windows if r['phase'] == phase and r['first_epoch'] == 8)
        late = next(r for r in fusion_windows if r['phase'] == phase and r['first_epoch'] == 29)
        latest = next(r for r in fusion_rows if r['phase'] == phase and r['epoch'] == 33)
        lines.append(f'|{phase}|{early["max_weight"]:.4f}|{late["max_weight"]:.4f}|{late["max_weight"]-early["max_weight"]:+.4f}|{latest["max_weight"]:.3f}|{latest["entropy"]:.3f}|')
    lines += ['', '训练quality_loss均值从epoch8–12的0.08151降至29–33的0.06644，但验证端没有一致明显下降；teacher随主模型变化，因此不能用loss下降直接证明评分质量提高。',
              '最新epoch33的位置均值：BTCV [0.250,0.249,0.250,0.250]，AMOS [0.250,0.250,0.251,0.250]，ADAM [0.244,0.251,0.252,0.253]，BraTS [0.251,0.250,0.250,0.249]。',
              '最大权重略有上升，但仍接近0.25，熵仍0.998–0.999，没有明显变尖。整体分割提升不能据此归因于评分头；此前epoch12的配对诊断也不能自动代表epoch33。', '',
              '## 产物', '',
              '- class_comparisons.csv：全部33类，与三个旧实验分别在共同窗口对比。',
              '- epoch_metrics.csv / class_metrics.csv：所有已完成轮次的总指标和逐类指标。',
              '- fusion_epoch_metrics.csv / fusion_window_metrics.csv：评分权重、熵、quality_loss。',
              '- snapshot.json：解析快照、日志路径与SHA256；trends.png：曲线图。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    make_plot(runs, fusion_rows)
    print(OUT / 'report_zh.md')
    print('\n'.join(lines[:22]))


def make_plot(runs, fusion_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(nearest='#808080', max='#b98b32', adam_only_masks='#3575a5', quality='#b63d49')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for ax, dataset in zip(axes.flat, METRICS):
        for name, run in runs.items():
            es = [e for e in run['complete_epochs'] if e <= 33]
            vals = [run['metrics'][e][dataset] for e in es]
            ax.plot(es, vals, color=colors[name], alpha=.14, linewidth=.8)
            if len(es) >= 5:
                smooth = [statistics.mean(vals[i-4:i+1]) for i in range(4, len(vals))]
                ax.plot(es[4:], smooth, label=name, color=colors[name], linewidth=1.8)
        ax.set(title=dataset.upper(), xlabel='Epoch (log index)', ylabel='Validation Dice (%)')
        ax.grid(alpha=.2)
    for dataset, color in zip(DATASETS, ['#3575a5', '#b98b32', '#b63d49', '#5f8d5d']):
        rows = [r for r in fusion_rows if r['phase'] == 'val/' + dataset and r['epoch'] >= 4]
        axes.flat[5].plot([r['epoch'] for r in rows], [r['max_weight'] for r in rows],
                          label=dataset, color=color, linewidth=1.2)
    axes.flat[5].axhline(.25, color='black', ls='--', linewidth=1)
    axes.flat[5].set(title='Mean maximum support weight', xlabel='Epoch (log index)', ylabel='Weight', ylim=(.248, .28))
    axes.flat[5].legend(fontsize=8); axes.flat[5].grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Full validation logs; bold = trailing 5 epochs; settings and GPU counts differ')
    fig.savefig(OUT / 'trends.png', dpi=165); plt.close(fig)


if __name__ == '__main__':
    main()
