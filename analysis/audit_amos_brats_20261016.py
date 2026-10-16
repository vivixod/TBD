"""Read-only dataset/sampler audit; writes only analysis artifacts beside this file."""
import ast
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.amos import DatasetAMOS
from data.brats import DatasetBraTS
from data.medical_base import build_patient_slice_index


def patient(name):
    return name.split('___', 1)[-1].rsplit('_', 1)[0]


def load_metadata(dataset, split):
    folder = ROOT / 'data/splits/lists' / dataset / 'fss_list' / split
    names = [line.split()[0].split('/')[-1].split('.')[0]
             for line in (folder / 'data_list_0.txt').read_text().splitlines() if line.strip()]
    raw = ast.literal_eval((folder / 'sub_class_file_list_0.txt').read_text())
    pools = {int(c): list(dict.fromkeys(row[0].split('/')[-1].split('.')[0] for row in entries))
             for c, entries in raw.items()}
    presence = {name: [0] for name in names}
    for c, entries in pools.items():
        assert set(entries).issubset(presence), (dataset, split, c)
        for name in entries:
            presence[name].append(c)
    return names, pools, presence


def sampling_object(dataset, path, split, names, pools, presence):
    klass = DatasetAMOS if dataset == 'amos' else DatasetBraTS
    ds = klass.__new__(klass)
    ds.split = 'trn' if split == 'train' else 'val'
    ds.img_metadata, ds.img_metadata_classwise = names, pools
    ds.class_ids = list(range(15 if dataset == 'amos' else 3))
    ds.shot, ds.neighbor_prob = 4, 1.0
    ds.img_path = ds.label_path = str(path)
    ds._shape_cache = {}
    ds._label_classes_cache = presence
    ds._patient2slices, ds._patient2order = build_patient_slice_index(names)
    return ds


def sample_validation(ds, labels):
    records = []
    padded_indices = list(range(len(ds)))
    padded_indices += padded_indices[:(-len(ds)) % 3]
    for rank in range(3):
        np.random.seed(321)
        for index in padded_indices[rank::3]:
            q, ss, c = ds.sample_episode(index % len(ds.img_metadata))
            records.append(dict(dataset='amos' if len(ds.class_ids) == 15 else 'brats',
                                rank=rank, index=index, query=q, patient=patient(q),
                                class_id=int(c) + 1, class_name=labels[str(c + 1)],
                                supports=list(ss), self_support=q in ss,
                                unique_supports=len(set(ss)),
                                support_patients=len({patient(s) for s in ss})))
    result = []
    for c in ds.class_ids:
        selected = [r for r in records if r['class_id'] == c + 1]
        result.append(dict(class_id=c + 1, class_name=labels[str(c + 1)],
                           episodes=len(selected), unique_queries=len({r['query'] for r in selected}),
                           patients=len({r['patient'] for r in selected}),
                           self_support_episodes=sum(r['self_support'] for r in selected),
                           repeated_support_episodes=sum(r['unique_supports'] < 4 for r in selected)))
    return dict(episodes=len(records), unique_queries=len({r['query'] for r in records}),
                patients=len({r['patient'] for r in records}), classes=result), records


def mask_audit(ds, pools, class_ids, limit):
    results = []
    rng = np.random.RandomState(321)
    for c in class_ids:
        candidates = sorted(pools[c])
        if limit and len(candidates) > limit:
            candidates = [candidates[i] for i in sorted(rng.choice(len(candidates), limit, replace=False))]
        areas, nearest_cells, max_cells = [], [], []
        for name in candidates:
            mask = ds.read_mask(name)
            actual_classes = set(int(v) for v in mask.unique().tolist()) - {0}
            cached_classes = set(ds._label_classes_cache[name]) - {0}
            assert actual_classes == cached_classes, (name, actual_classes, cached_classes)
            binary = (mask == c).float()[None, None]
            scaled = F.interpolate(binary, size=(512, 512), mode='nearest')
            areas.append(int(scaled.sum()))
            nearest_cells.append(int(F.interpolate(scaled, size=(64, 64), mode='nearest').sum()))
            max_cells.append(int(F.adaptive_max_pool2d(scaled, (64, 64)).sum()))
        results.append(dict(class_id=c, sampled_slices=len(candidates),
                            all_positive_slices=len(pools[c]),
                            area512_min=min(areas), area512_median=float(np.median(areas)),
                            area512_max=max(areas),
                            nearest_cells_median=float(np.median(nearest_cells)),
                            nearest_empty=sum(n == 0 for n in nearest_cells),
                            max_empty=sum(n == 0 for n in max_cells)))
    return results


def parse_trends(log):
    rows, current = [], None
    pattern = re.compile(r'\*\*\* val/(amos|brats) \[@Epoch (\d+)\].*?Dice:\s*([\d.]+)')
    for line in log.read_text().splitlines():
        match = pattern.search(line)
        if match:
            current = (match[1], int(match[2]), float(match[3]))
        elif line.startswith('[per-class Dice]') and current:
            ds, epoch, mean = current
            for piece in line.split('] ', 1)[1].split(', '):
                name, value = piece.split('=')
                rows.append(dict(dataset=ds, epoch=epoch, class_name=name,
                                 dice=float(value), dataset_dice=mean))
            current = None
    windows = []
    for ds in ['amos', 'brats']:
        latest = max(r['epoch'] for r in rows if r['dataset'] == ds)
        for first, last in [(0, 4), (10, 14), (20, 24), (latest - 4, latest)]:
            selected = [r for r in rows if r['dataset'] == ds and first <= r['epoch'] <= last]
            by_class = defaultdict(list)
            for r in selected:
                by_class[r['class_name']].append(r['dice'])
            windows.append(dict(dataset=ds, window=f'{first}-{last}',
                                classes={c: dict(mean=float(np.mean(v)), min=min(v), max=max(v), n=len(v))
                                         for c, v in by_class.items()}))
    return rows, windows


def main():
    torch.set_num_threads(1)
    result, validation_records, counts = {}, [], []
    for dataset, directory, weight in [('amos', 'AMOS2022_MR', 1), ('brats', 'BraTS2019', 1.5)]:
        path = ROOT.parent / directory
        labels = json.loads((path / 'dataset.json').read_text())['labels']
        ds_info, patient_sets = {}, {}
        for split in ['train', 'val']:
            names, pools, presence = load_metadata(dataset, split)
            patient_sets[split] = {patient(n) for n in names}
            ds = sampling_object(dataset, path, split, names, pools, presence)
            classes = []
            for c, positives in sorted(pools.items()):
                pcounts = Counter(patient(n) for n in positives)
                query_probability = sum(1 / (len(presence[n]) - 1) for n in positives) / len(names)
                row = dict(dataset=dataset, split=split, class_id=c, class_name=labels[str(c)],
                           positive_slices=len(positives), positive_patients=len(pcounts))
                if split == 'train':
                    row['expected_episodes_per_epoch_cb05'] = 8000 * weight / 5.5 * (0.5 / len(pools) + 0.5 * query_probability)
                counts.append(row)
                classes.append(dict(row, patient_slices=dict(sorted(pcounts.items()))))
            ds_info[split] = dict(slices=len(names), patients=len(patient_sets[split]), classes=classes)
            ids = [14, 15] if dataset == 'amos' else [1, 2, 3]
            ds_info[split]['mask_audit'] = mask_audit(ds, pools, ids, None if dataset == 'amos' else 128)
            if split == 'val':
                summary, records = sample_validation(ds, labels)
                ds_info[split]['fixed_validation_sampling'] = summary
                validation_records.extend(records)
            print(dataset, split, 'finished', flush=True)
        ds_info['patient_overlap'] = sorted(patient_sets['train'] & patient_sets['val'])
        result[dataset] = ds_info
    log = ROOT / 'logs/trn_joint4_shot4_cb0.5_pooladam-only_bboxm1_iclm1.log/log.txt'
    trends, windows = parse_trends(log)
    result['log'] = str(log)
    result['trend_windows'] = windows
    result['notes'] = [
        'Validation replay: actual dataset sample_episode with label presence cached from class metadata; 3 ranks, seed321 per rank, 1000 episodes padded to1002.',
        'Mask audit: all AMOS rare-class positive slices; BraTS fixed random128 positives per class/split; no augmentation.',
        'Training episode counts are analytical expectations, not observations; current8000 episodes, weights1:1:2:1.5, class-balance probability0.5.',
        'Patient IDs strip x/y directory prefixes; counts describe prepared PNG/NPZ dataset, not missing anatomy in the original scans.',
    ]
    prefix = Path(__file__).with_suffix('')
    prefix.with_suffix('.json').write_text(json.dumps(result, indent=2))
    for suffix, rows in [('_counts.csv', counts), ('_trends.csv', trends)]:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with Path(str(prefix) + suffix).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    Path(str(prefix) + '_val_manifest.json').write_text(json.dumps(validation_records, indent=2))
    print('Saved', prefix.with_suffix('.json'), flush=True)


if __name__ == '__main__':
    main()
