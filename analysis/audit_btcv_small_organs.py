"""CPU-only audit of BTCV sampling, mask downsampling, and geometric augmentation.

Reads masks and split lists without importing the model or touching caches/checkpoints.
Run with the project's Python environment; outputs JSON and CSV beside this script.
"""
import argparse
import ast
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


NAMES = ['spleen', 'right_kidney', 'left_kidney', 'gallbladder',
         'esophagus', 'liver', 'stomach', 'aorta', 'IVC', 'portal_vein',
         'pancreas', 'right_adrenal', 'left_adrenal']


def audit(data_root, split_root, seed):
    rng = np.random.RandomState(seed)
    result = {}
    for split in ('train', 'val'):
        names = [line.split()[0].split('/')[-1].split('.')[0]
                 for line in (split_root / split / 'data_list_0.txt').read_text().splitlines()
                 if line.strip()]
        declared = ast.literal_eval(
            (split_root / split / 'sub_class_file_list_0.txt').read_text())
        declared = {int(c): {r[0].split('/')[-1].split('.')[0] for r in rows}
                    for c, rows in declared.items()}
        areas, counts8, counts16, aug_dice = ([[] for _ in NAMES] for _ in range(4))
        probability = np.zeros(13)
        empty_query_count = 0
        mismatches = 0
        patient_classes = [set() for _ in NAMES]
        for name in names:
            mask = np.asarray(Image.open(data_root / 'mask' / (name + '.png')))
            if mask.shape != (512, 512):
                raise ValueError(f'Unexpected mask shape {mask.shape}: {name}')
            present = np.unique(mask)
            present = present[(present >= 1) & (present <= 13)]
            if not len(present):
                empty_query_count += 1
                continue
            probability[present - 1] += 1.0 / len(present) / len(names)
            # This reproduces the geometric part of joint_augment, not its intensity RNG.
            angle = rng.uniform(-10, 10)
            tx, ty = rng.uniform(-.03, .03, size=2) * 512
            matrix = cv2.getRotationMatrix2D((256, 256), angle, 1.0)
            matrix[0, 2] += tx
            matrix[1, 2] += ty
            augmented = cv2.warpAffine(mask, matrix, (512, 512),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            for c in present:
                i = int(c) - 1
                target = mask == c
                areas[i].append(int(target.sum()))
                counts8[i].append(int(target[::8, ::8].sum()))
                counts16[i].append(int(target[::16, ::16].sum()))
                changed = augmented == c
                aug_dice[i].append(float(2 * (target & changed).sum() /
                                         (target.sum() + changed.sum())))
                patient_classes[i].add(name.replace('x___', '').rsplit('_', 1)[0])
                if name not in declared.get(int(c), set()):
                    mismatches += 1
        rows = []
        for i, label in enumerate(NAMES):
            row = {'class_id_1based': i + 1, 'class_name': label,
                   'positive_slices': len(areas[i]), 'positive_patients': len(patient_classes[i]),
                   'episode_probability_pct': float(probability[i] * 100),
                   'expected_episodes': float(probability[i] * (8000 / 5.5 if split == 'train' else 1000))}
            if areas[i]:
                row.update(median_area_px=float(np.median(areas[i])),
                           median_foreground_pct=float(np.median(areas[i]) / 512**2 * 100),
                           median_nearest_tokens_stride8=float(np.median(counts8[i])),
                           empty_nearest_stride8_pct=float(np.mean(np.array(counts8[i]) == 0) * 100),
                           le2_nearest_stride8_pct=float(np.mean(np.array(counts8[i]) <= 2) * 100),
                           empty_nearest_stride16_pct=float(np.mean(np.array(counts16[i]) == 0) * 100),
                           median_geometric_overlap_dice=float(np.median(aug_dice[i]) * 100))
            rows.append(row)
        result[split] = {'slices': len(names),
                         'patients': sorted({n.replace('x___', '').rsplit('_', 1)[0] for n in names}),
                         'empty_query_slices': empty_query_count,
                         'positive_entries_missing_from_class_lists': mismatches,
                         'classes': rows}
        print(split, 'slices=', len(names), 'empty=', empty_query_count, flush=True)
        for r in rows:
            print(r, flush=True)
    result['patient_overlap'] = sorted(set(result['train']['patients']) & set(result['val']['patients']))
    result['notes'] = [
        'Stride 8 is the current dilated ResNet50 grid; stride 16 is a counterfactual.',
        'Expected train episodes use 8000 episodes and BTCV dataset probability 1/5.5.',
        'Episode probabilities use actual mask class presence before augmentation.',
        'Geometric overlap compares a mask with a random transformed copy; it is not model Dice.',
        'Validation counts are expectations across random episodes, not actual per-class log counts.',
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'BTCV')
    parser.add_argument('--split-root', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/splits/lists/btcv2/fss_list')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('btcv_small_organs_audit_20261014.json'))
    parser.add_argument('--seed', type=int, default=321)
    args = parser.parse_args()
    result = audit(args.data_root, args.split_root, args.seed)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with args.output.with_suffix('.csv').open('w', newline='') as handle:
        rows = [{'split': split, **row} for split in ('train', 'val')
                for row in result[split]['classes']]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', args.output, flush=True)


if __name__ == '__main__':
    main()
