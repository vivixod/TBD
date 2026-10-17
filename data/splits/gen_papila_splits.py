"""Audit prepared PAPILA files and generate reproducible patient-level split lists.

The source dataset.json eye-level training/test split is intentionally replaced
in the generated lists only. Images, labels and dataset.json are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.papila import SPLIT_ROOT, load_papila_records, papila_patient_id


def build(datapath, output=SPLIT_ROOT, ratio=0.8, seed=321):
    if not 0 < ratio < 1:
        raise ValueError('Training patient ratio must be between 0 and 1')
    root, output = Path(datapath), Path(output)
    records = load_papila_records(root)
    classes = {}
    exclusive, overlapping = 0, 0
    for name, entry in sorted(records.items()):
        with Image.open(root / entry['image']) as image:
            width, height = image.size
            if image.mode != 'RGB':
                raise ValueError(f'{name}: expected an RGB image, got {image.mode}')
        label = sp.load_npz(root / entry['label']).tocsr()
        if label.shape != (2, height * width):
            raise ValueError(f'{name}: expected two sparse mask channels matching the image')
        if not np.isin(label.data, [0, 1]).all():
            raise ValueError(f'{name}: expected binary sparse mask channels')
        label.eliminate_zeros()
        disc_present, cup_present = (label.getrow(c).nnz > 0 for c in range(2))
        classes[name] = ([1] if disc_present or cup_present else []) + ([2] if cup_present else [])
        if not classes[name]:
            raise ValueError(f'{name}: no disc or cup foreground')
        if label.getrow(0).multiply(label.getrow(1)).nnz:
            overlapping += 1
        else:
            exclusive += 1
    patients = sorted({papila_patient_id(name) for name in records})
    n_val = int(round(len(patients) * (1 - ratio)))
    if not 1 <= n_val < len(patients):
        raise ValueError('Requested ratio leaves an empty train or validation partition')
    rng = np.random.RandomState(seed)
    val_patients = {patients[int(i)] for i in rng.permutation(len(patients))[:n_val]}
    train_patients = set(patients) - val_patients
    source = json.loads((root / 'dataset.json').read_text())
    source_patients = {
        key: {papila_patient_id(Path(e['image']).stem) for e in source.get(key, [])}
        for key in ('training', 'test')}
    manifest = {
        'version': 1, 'seed': seed, 'train_patient_ratio': ratio,
        'source_json_sha256': hashlib.sha256((root / 'dataset.json').read_bytes()).hexdigest(),
        'source_train_test_patient_overlap': len(source_patients['training'] & source_patients['test']),
        'patients': {'train': sorted(train_patients), 'val': sorted(val_patients)},
        'targets': {'1': 'optic_disc = channel_0 OR channel_1', '2': 'optic_cup = channel_1'},
        'source_channel_layout': {'exclusive_images': exclusive, 'overlapping_images': overlapping},
        'support_pool': 'same partition; exclude query patient; one image per support patient',
        'validation': 'each nonempty image/class pair once; supports fixed by seed/query/class',
        'counts': {},
    }
    prepared = {}
    for split, group in [('train', train_patients), ('val', val_patients)]:
        names = sorted(n for n in records if papila_patient_id(n) in group)
        classwise = {c: [[n] for n in names if c in classes[n]] for c in (1, 2)}
        manifest['counts'][split] = {
            'patients': len(group), 'images': len(names),
            'class_images': {str(c): len(v) for c, v in classwise.items()},
            'class_patients': {str(c): len({papila_patient_id(e[0]) for e in v})
                               for c, v in classwise.items()},
            'image_class_pairs': sum(len(v) for v in classwise.values()),
        }
        prepared[split] = (names, classwise)
    # Refuse a silent protocol change once lists have been generated.
    files = {output / 'manifest.json': json.dumps(manifest, indent=2, ensure_ascii=False) + '\n'}
    for split, (names, classwise) in prepared.items():
        folder = output / 'fss_list' / split
        files[folder / 'data_list_0.txt'] = '\n'.join(names) + '\n'
        files[folder / 'sub_class_file_list_0.txt'] = repr(classwise) + '\n'
    for path, content in files.items():
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f'{path} already contains a different protocol; '
                                  'choose a new --output directory to inspect the new split')
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datapath', type=Path, default=ROOT.parent / 'PAPILA')
    parser.add_argument('--output', type=Path, default=SPLIT_ROOT)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=321)
    args = parser.parse_args()
    print(json.dumps(build(args.datapath, args.output, args.ratio, args.seed),
                     indent=2, ensure_ascii=False))
