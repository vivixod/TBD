"""PAPILA RGB fundus episodes with patient-disjoint, class-matched supports.

Prepared inputs: image/RETxxxOD.png and two-channel sparse label/*.npz.
Channel 0 is disc/rim and channel 1 is cup. The disc target is their union.
Both eyes stay in one partition. Every support comes from a different patient,
including exclusion of the query's fellow eye; neighbor_prob does not apply.
"""
import ast
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import scipy.sparse as sp
import torch

from data.medical_base import MedicalDataset


SPLIT_ROOT = Path(__file__).resolve().parent / 'splits' / 'lists' / 'papila'


def papila_patient_id(image_name):
    match = re.fullmatch(r'(RET\d+)(?:OD|OS)', image_name)
    if match is None:
        raise ValueError(f'Invalid PAPILA image name: {image_name!r}')
    return match.group(1)


def load_papila_records(datapath):
    """Use source JSON for paths only; its eye-level partition is not reused."""
    root = Path(datapath)
    metadata = json.loads((root / 'dataset.json').read_text())
    if metadata.get('labels') != {
            '0': 'background', '1': 'optic_disc', '2': 'optic_cup'}:
        raise ValueError('PAPILA requires label channels 1=optic_disc, 2=optic_cup')
    records = {}
    for key in ('training', 'validation', 'test'):
        for entry in metadata.get(key, []):
            if not isinstance(entry, dict) or not {'image', 'label'} <= entry.keys():
                raise ValueError(f'PAPILA {key} entries must contain image and label paths')
            name = Path(entry['image']).stem
            papila_patient_id(name)
            if name in records:
                raise ValueError(f'Duplicate PAPILA image in dataset.json: {name}')
            records[name] = {'image': entry['image'], 'label': entry['label']}
    if not records:
        raise ValueError('No labeled PAPILA images found in dataset.json')
    return records


class DatasetPAPILA(MedicalDataset):
    support_sampling = ('different patients within the query partition; one eye per '
                        'support patient; validation combinations fixed by seed/query/class; '
                        'neighbor_prob ignored')

    def __init__(self, datapath, fold, transform, split, shot,
                 use_original_imgsize, neighbor_prob=0.0):
        if fold != 0:
            raise ValueError('PAPILA has a fixed patient split; use --fold 0')
        if shot < 1:
            raise ValueError('PAPILA shot must be positive')
        self.records = load_papila_records(datapath)
        manifest_path = SPLIT_ROOT / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError('Generate PAPILA lists first: python '
                                    'data/splits/gen_papila_splits.py --datapath ' + str(datapath))
        self.manifest = json.loads(manifest_path.read_text())
        self.support_seed = int(self.manifest['seed'])
        train_patients = set(self.manifest['patients']['train'])
        val_patients = set(self.manifest['patients']['val'])
        if train_patients & val_patients:
            raise ValueError('PAPILA train/val patient overlap in manifest')
        if train_patients | val_patients != {papila_patient_id(n) for n in self.records}:
            raise ValueError('PAPILA source patients differ from the split manifest; regenerate lists')
        super().__init__(
            datapath, fold, transform, split, shot, use_original_imgsize,
            name='papila', nclass=2,
            # Natural RGB images; retain the existing ImageNet normalization.
            img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225],
            aug_params=dict(angle=10.0, trans=0.03, contrast=0.1, noise_p=0.0),
            neighbor_prob=0.0,
        )
        expected = {n for n in self.records
                    if papila_patient_id(n) in self.manifest['patients'][self.split_norm]}
        if len(self.img_metadata) != len(set(self.img_metadata)) or set(self.img_metadata) != expected:
            raise ValueError('PAPILA query list does not match the patient split manifest')
        self._support_patients = {}
        query_classes = {name: [] for name in self.img_metadata}
        for class_id in self.class_ids:
            groups = {}
            for name in self.img_metadata_classwise.get(class_id + 1, []):
                if name not in query_classes:
                    raise ValueError(f'PAPILA support outside current partition: {name}')
                groups.setdefault(papila_patient_id(name), set()).add(name)
                query_classes[name].append(class_id)
            self._support_patients[class_id] = {
                patient: tuple(sorted(names)) for patient, names in sorted(groups.items())}
        if any(not classes for classes in query_classes.values()):
            raise ValueError('PAPILA contains a query with no foreground class')
        for name, classes in query_classes.items():
            for class_id in classes:
                available = len(self._support_patients[class_id]) - 1
                if available < shot:
                    raise ValueError(f'PAPILA class {class_id}: need {shot} other support '
                                     f'patients for {name}, only {available} available')
        self._query_classes = {n: tuple(sorted(set(c))) for n, c in query_classes.items()}
        # Evaluate every available image/class pair once, covering both disc and cup.
        self._val_episodes = [(n, c) for n in self.img_metadata for c in self._query_classes[n]]

    def build_img_metadata(self):
        path = SPLIT_ROOT / 'fss_list' / self.split_norm / 'data_list_0.txt'
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def build_img_metadata_classwise(self):
        path = SPLIT_ROOT / 'fss_list' / self.split_norm / 'sub_class_file_list_0.txt'
        raw = ast.literal_eval(path.read_text())
        return {int(c): [entry[0] for entry in entries] for c, entries in raw.items()}

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else len(self._val_episodes)

    def __getitem__(self, idx):
        return self._load_episode(*self.sample_episode(idx))

    def sample_episode(self, idx):
        if self.split == 'trn':
            balanced = self.sample_class_balanced_query()
            if balanced is None:
                query_name = self.img_metadata[idx % len(self.img_metadata)]
                class_id = int(np.random.choice(self._query_classes[query_name]))
            else:
                query_name, class_id = balanced
            rng = np.random
        else:
            query_name, class_id = self._val_episodes[idx % len(self._val_episodes)]
            key = f'{self.support_seed}:{self.split_norm}:{query_name}:{class_id}'.encode()
            seed = int.from_bytes(hashlib.sha256(key).digest()[:4], 'little')
            rng = np.random.RandomState(seed)
        groups = self._support_patients[class_id]
        patients = [p for p in groups if p != papila_patient_id(query_name)]
        selected = rng.choice(len(patients), size=self.shot, replace=False)
        supports = []
        for index in selected:
            eyes = groups[patients[int(index)]]
            supports.append(eyes[int(rng.randint(len(eyes)))])
        return query_name, supports, class_id

    def read_img(self, img_name):
        with Image.open(Path(self.img_path) / self.records[img_name]['image']) as image:
            return image.convert('RGB')

    def read_mask(self, img_name):
        sparse = sp.load_npz(Path(self.label_path) / self.records[img_name]['label'])
        if img_name not in self._shape_cache:
            with Image.open(Path(self.img_path) / self.records[img_name]['image']) as image:
                self._shape_cache[img_name] = (image.height, image.width)
        height, width = self._shape_cache[img_name]
        if sparse.shape != (2, height * width):
            raise ValueError(f'Invalid PAPILA sparse label shape for {img_name}: {sparse.shape}')
        channels = sparse.toarray().reshape(2, height, width) > 0
        label = np.zeros((height, width), dtype=np.uint8)
        label[channels[0]] = 1
        label[channels[1]] = 2
        return torch.from_numpy(label)

    def extract_ignore_idx(self, mask, class_id):
        # Cup pixels also belong to the full optic disc, even for exclusive channels.
        foreground = mask > 0 if class_id == 0 else mask == 2
        return foreground.float(), torch.zeros_like(mask)

    def _label_classes(self, img_name):
        return [0] + [c + 1 for c in self._query_classes[img_name]]
