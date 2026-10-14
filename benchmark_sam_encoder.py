"""Measure FP32 vs automatic BF16 frozen SAM encoding on the training GPU.

Run: python benchmark_sam_encoder.py --steps 20 --warmup 5
This reads pretrained weights, does not train, and does not write feature caches.
"""
import argparse
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'segment-anything'))
from SAM2pred import SAM_pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is required for a meaningful GPU speed measurement')
    if args.steps < 1 or args.warmup < 0 or args.bsz < 1:
        parser.error('steps/bsz must be positive and warmup must be nonnegative')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(321)
    model = SAM_pred().to(device).train()
    query = torch.randn(args.bsz, 3, 512, 512, device=device)
    print(f'GPU: {torch.cuda.get_device_name(device)}; batch={args.bsz}; '
          f'PyTorch={torch.__version__}')
    timings = {}
    for requested in ('fp32', 'auto'):
        model.encoder_precision = requested
        resolved = model.resolve_encoder_precision(device)
        for _ in range(args.warmup):
            model.forward_img_encoder(query)
        torch.cuda.synchronize(device)
        base_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        elapsed = []
        for _ in range(args.steps):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            features = model.forward_img_encoder(query)
            end.record()
            end.synchronize()
            if not torch.isfinite(features).all().item():
                raise RuntimeError(f'Nonfinite features with {requested}/{resolved}')
            elapsed.append(start.elapsed_time(end))
            del features
        peak_extra = (torch.cuda.max_memory_allocated(device) - base_memory) / 1024**2
        timings[requested] = statistics.mean(elapsed)
        print(f'{requested} -> {resolved}: mean={timings[requested]:.2f} ms/batch, '
              f'median={statistics.median(elapsed):.2f} ms/batch, '
              f'peak transient memory={peak_extra:.0f} MiB')
    print(f'Encoder speedup: {timings["fp32"] / timings["auto"]:.2f}x '
          '(encoder only; full training speedup must be measured separately)')


if __name__ == '__main__':
    main()
