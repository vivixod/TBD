"""Prepare image-only, cross-patient Top-K support rankings before training.

Uses a fixed pretrained ResNet50, eval BN, unaugmented RGB images resized to224,
and dataset-specific normalization. No query or support mask is decoded.
"""
import argparse
import hashlib
from pathlib import Path
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.dataset import FSSDataset
from data.support_retrieval import (DEFAULT_RETRIEVAL_DIR, ROOT, SUPPORTED_DATASETS,
                                    SimilaritySupportIndex, bank_path, dataset_spec,
                                    normalization, rank_descriptors, save_bank)
from model.base.resnet import resnet50


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class ImageDescriptors(nn.Module):
    def __init__(self, weights):
        super().__init__()
        backbone = resnet50(pretrained=False)
        state = torch.load(weights, map_location='cpu', weights_only=True)
        mismatch = backbone.load_state_dict(state, strict=False)
        missing = [key for key in mismatch.missing_keys if not key.startswith(('fc.', 'layer4.'))]
        if missing:
            raise ValueError(f'Retrieval checkpoint misses backbone weights: {missing[:5]}')
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1,
                                  backbone.conv2, backbone.bn2, backbone.relu2,
                                  backbone.conv3, backbone.bn3, backbone.relu3, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.requires_grad_(False)
        self.eval()

    def forward(self, images):
        second = self.layer2(self.layer1(self.stem(images)))
        third = self.layer3(second)
        pieces = [F.normalize(F.adaptive_avg_pool2d(feature, size).flatten(1), dim=1)
                  for feature in (second, third) for size in (1, 2)]
        return F.normalize(torch.cat(pieces, dim=1), dim=1)


class UnaugmentedImages(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.names = dataset_spec(dataset)['names']
        norm = normalization(dataset)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(norm['mean'], norm['std'])])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image = self.dataset.read_img(self.names[index])
        try:
            return self.transform(image)
        finally:
            image.close()


@torch.inference_mode()
def extract_descriptors(dataset, encoder, device, batch_size=16, workers=4):
    loader = DataLoader(UnaugmentedImages(dataset), batch_size=batch_size,
                        num_workers=workers, shuffle=False, pin_memory=device.type == 'cuda')
    chunks = []
    for index, images in enumerate(loader):
        chunks.append(encoder(images.to(device, non_blocking=True)).cpu())
        if index % 100 == 0 or index + 1 == len(loader):
            print(f'[Retrieval prepare] {dataset.benchmark}/{dataset.split}: '
                  f'features {index + 1}/{len(loader)} batches', flush=True)
    return torch.cat(chunks)


def prepare_dataset(dataset, encoder, device, output_dir, encoder_sha256,
                    max_shot=4, batch_size=16, workers=4, chunk_size=128):
    features = extract_descriptors(dataset, encoder, device, batch_size, workers)
    spec = dataset_spec(dataset)
    print(f'[Retrieval prepare] {dataset.benchmark}/{dataset.split}: '
          f'ranking {len(spec["names"])} images, {len(spec["class_ids"])} classes', flush=True)
    neighbors, similarities = rank_descriptors(spec, features, max_shot, device, chunk_size)
    path = bank_path(dataset, output_dir)
    save_bank(path, spec, neighbors, similarities, encoder_sha256)
    SimilaritySupportIndex(dataset, output_dir)  # Validate the written artifact immediately.
    print(f'[Retrieval prepare] saved {path}', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint-subdatasets', required=True,
                        help='Same name:path,name:path list as train.py; PAPILA is skipped.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights/resnet50_v2.pth')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-shot', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--splits', nargs='+', choices=['train', 'val'], default=['train', 'val'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if min(args.batch_size, args.threads, args.max_shot, args.chunk_size) < 1 or args.workers < 0:
        parser.error('batch/threads/max-shot/chunk must be positive; workers must be nonnegative')
    os.chdir(ROOT)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('Requested CUDA device is unavailable')
    FSSDataset.initialize(512, '', False)
    digest = file_sha256(args.weights)
    encoder = None
    for entry in args.joint_subdatasets.split(','):
        name, path = entry.strip().split(':', 1)
        if name == 'papila':
            print('[Retrieval prepare] PAPILA: retain native different-patient eye sampling', flush=True)
            continue
        if name not in SUPPORTED_DATASETS:
            parser.error(f'Unsupported retrieval dataset: {name}')
        for split in args.splits:
            dataset = FSSDataset.datasets[name](path, 0, FSSDataset.transform,
                        'trn' if split == 'train' else 'val', args.max_shot, False, neighbor_prob=0.)
            if not args.force:
                try:
                    cached = SimilaritySupportIndex(dataset, args.output_dir)
                    if cached.manifest.get('encoder_sha256') == digest:
                        print(f'[Retrieval prepare] reuse {cached.path}', flush=True)
                        continue
                except (FileNotFoundError, ValueError, KeyError, OSError):
                    pass
            if encoder is None:
                encoder = ImageDescriptors(args.weights).to(device).eval()
            prepare_dataset(dataset, encoder, device, args.output_dir, digest,
                            args.max_shot, args.batch_size, args.workers, args.chunk_size)


if __name__ == '__main__':
    main()
