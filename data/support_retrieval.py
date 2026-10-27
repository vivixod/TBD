"""Split-local, cross-patient support selection and cached cosine rankings.

Rankings use frozen image descriptors only. Classwise support metadata defines
the labelled candidate pool; query masks and model predictions are never read.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RETRIEVAL_DIR = ROOT / 'cache_support_retrieval'
RETRIEVAL_VERSION = 'resnet50_l2l3_spatial224_v1'
SUPPORTED_DATASETS = ('btcv2', 'amos', 'brats')
SUPPORT_SAMPLING_MODES = ('random', 'cross-patient-random', 'similarity')


def patient_id(name):
    try:
        patient, number = name.split('___', 1)[-1].rsplit('_', 1)
        int(number)
    except (ValueError, TypeError):
        raise ValueError(f'Cannot determine slice patient: {name!r}')
    if not patient:
        raise ValueError(f'Cannot determine slice patient: {name!r}')
    return patient


def normalization(dataset):
    from torchvision.transforms import Normalize
    steps = [t for t in dataset.transform.transforms if isinstance(t, Normalize)]
    if len(steps) != 1:
        raise ValueError('Support retrieval requires exactly one Normalize transform')
    return dict(mean=list(steps[0].mean), std=list(steps[0].std))


def dataset_spec(dataset):
    names = sorted(set(dataset.img_metadata))
    allowed = set(names)
    pools = {str(c): sorted(set(dataset.img_metadata_classwise.get(c + 1, ())) & allowed)
             for c in dataset.class_ids}
    return dict(version=RETRIEVAL_VERSION, dataset=dataset.benchmark, fold=dataset.fold,
                split='train' if dataset.split == 'trn' else 'val',
                names=names, class_ids=list(dataset.class_ids), pools=pools,
                normalization=normalization(dataset))


def spec_digest(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def bank_path(dataset, directory=None):
    split = 'train' if dataset.split == 'trn' else 'val'
    return Path(directory or DEFAULT_RETRIEVAL_DIR) / f'{dataset.benchmark}_fold{dataset.fold}_{split}.npz'


def rank_descriptors(spec, descriptors, max_shot=4, device='cpu', chunk_size=128):
    """Rank ALL eligible positive slices, with no per-patient quota.

    Names are lexically sorted, and stable sorting breaks exact cosine ties by
    name. Precomputing rankings avoids repeated large matrix products in workers.
    """
    import torch
    if max_shot < 1 or chunk_size < 1:
        raise ValueError('max_shot and chunk_size must be positive')
    features = torch.as_tensor(descriptors, dtype=torch.float32, device='cpu')
    if features.ndim != 2 or features.shape[0] != len(spec['names']):
        raise ValueError('Descriptor rows must match all split image names')
    if not torch.isfinite(features).all() or (features.norm(dim=1) <= 0).any():
        raise ValueError('Descriptors must be finite and nonzero')
    features = torch.nn.functional.normalize(features, dim=1)
    names = spec['names']
    lookup = {name: i for i, name in enumerate(names)}
    patient_names = [patient_id(n) for n in names]
    patient_lookup = {name: i for i, name in enumerate(sorted(set(patient_names)))}
    patients = torch.tensor([patient_lookup[p] for p in patient_names], device=device)
    shape = (len(names), len(spec['class_ids']), max_shot)
    neighbors, similarities = np.empty(shape, np.int32), np.empty(shape, np.float32)
    for column, cls in enumerate(spec['class_ids']):
        candidates = [lookup[name] for name in spec['pools'][str(cls)]]
        if not candidates:
            raise ValueError(f"{spec['dataset']}: class {cls} has no support candidates")
        ids = torch.tensor(candidates, device=device)
        keys = features[candidates].to(device)
        for start in range(0, len(names), chunk_size):
            stop = min(start + chunk_size, len(names))
            eligible = patients[start:stop, None] != patients[ids][None]
            if (eligible.sum(1) < max_shot).any():
                raise ValueError(f"{spec['dataset']}: class {cls} has fewer than {max_shot} "
                                 'distinct positive slices outside at least one query patient')
            scores = features[start:stop].to(device) @ keys.T
            scores.masked_fill_(~eligible, -float('inf'))
            selected = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :max_shot]
            neighbors[start:stop, column] = ids[selected].cpu().numpy()
            similarities[start:stop, column] = scores.gather(1, selected).cpu().numpy()
    return neighbors, similarities


def save_bank(path, spec, neighbors, similarities, encoder_sha256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(version=RETRIEVAL_VERSION, dataset_digest=spec_digest(spec),
                    dataset=spec['dataset'], split=spec['split'], fold=spec['fold'],
                    normalization=spec['normalization'], encoder_sha256=encoder_sha256)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as handle:
            tmp_name = handle.name
            np.savez_compressed(handle, names=np.asarray(spec['names']),
                                class_ids=np.asarray(spec['class_ids'], dtype=np.int32),
                                neighbors=neighbors, similarities=similarities,
                                manifest=np.asarray(json.dumps(manifest)))
        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


class CrossPatientRandom:
    def __init__(self, dataset):
        allowed = set(dataset.img_metadata)
        self.pools = {c: sorted(set(dataset.img_metadata_classwise.get(c + 1, ())) & allowed)
                      for c in dataset.class_ids}

    def select(self, query, cls, shot):
        candidates = [n for n in self.pools[cls] if patient_id(n) != patient_id(query)]
        if len(candidates) < shot:
            raise ValueError(f'Class {cls}: fewer than {shot} eligible cross-patient slices')
        return list(np.random.choice(candidates, size=shot, replace=False))


class SimilaritySupportIndex:
    def __init__(self, dataset, directory=None):
        path = bank_path(dataset, directory)
        if not path.is_file():
            raise FileNotFoundError(f'Missing support retrieval cache: {path}. '
                                    'Run prepare_support_retrieval.py before training.')
        spec = dataset_spec(dataset)
        with np.load(path, allow_pickle=False) as data:
            self.names = data['names'].tolist()
            self.classes = data['class_ids'].tolist()
            self.neighbors = data['neighbors'].copy()
            self.similarities = data['similarities'].copy()
            self.manifest = json.loads(str(data['manifest']))
        if (self.manifest.get('version') != RETRIEVAL_VERSION
                or self.manifest.get('dataset_digest') != spec_digest(spec)
                or self.names != spec['names'] or self.classes != spec['class_ids']):
            raise ValueError(f'Stale support retrieval cache: {path}. '
                             'Re-run prepare_support_retrieval.py for the current split/classes/normalization.')
        n = len(self.names)
        if (self.neighbors.ndim != 3 or self.neighbors.shape[:2] != (n, len(self.classes))
                or self.neighbors.shape != self.similarities.shape
                or self.neighbors.shape[2] < dataset.shot
                or not np.issubdtype(self.neighbors.dtype, np.integer)
                or np.any(self.neighbors < 0) or np.any(self.neighbors >= n)
                or not np.isfinite(self.similarities).all()):
            raise ValueError(f'Invalid/insufficient retrieval rankings: {path}')
        patients = np.asarray([patient_id(name) for name in self.names])
        if np.any(patients[self.neighbors] == patients[:, None, None]):
            raise ValueError(f'Retrieval cache contains query-patient supports: {path}')
        self.lookup = {name: i for i, name in enumerate(self.names)}
        self.columns = {c: i for i, c in enumerate(self.classes)}
        for column, cls in enumerate(self.classes):
            allowed = [self.lookup[name] for name in spec['pools'][str(cls)]]
            chosen = self.neighbors[:, column]
            if (not np.isin(chosen, allowed).all()
                    or np.any(np.diff(np.sort(chosen, axis=1), axis=1) == 0)
                    or np.any(np.diff(self.similarities[:, column], axis=1) > 1e-6)):
                raise ValueError(f'Invalid class candidates/order in retrieval cache: {path}')
        self.path = str(path)

    def select(self, query, cls, shot):
        indices = self.neighbors[self.lookup[query], self.columns[cls], :shot]
        return [self.names[int(i)] for i in indices]


def configure_support_sampling(dataset, mode='random', directory=None):
    if mode not in SUPPORT_SAMPLING_MODES:
        raise ValueError(f'Unknown support sampling mode: {mode}')
    if mode == 'random':
        return  # Preserve legacy RNG consumption and support rules exactly.
    if dataset.benchmark == 'papila':
        dataset.support_sampling_mode = 'papila-native'
        return
    if dataset.benchmark not in SUPPORTED_DATASETS:
        raise ValueError(f'{mode} support sampling supports {SUPPORTED_DATASETS}; '
                         'PAPILA keeps its own patient sampling.')
    dataset.support_retriever = (SimilaritySupportIndex(dataset, directory) if mode == 'similarity'
                                else CrossPatientRandom(dataset))
    dataset.support_sampling_mode = mode
    dataset.neighbor_prob = 0.0
