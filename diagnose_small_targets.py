"""Fixed-episode diagnostics: localization, individual supports, and GT-guided crops.

This script never replaces bbox tokens. Cropped views run the complete prompt
generator, including query cross-attention, on newly encoded images.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'segment-anything'))
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from model.base.support_fusion import checkpoint_fusion_mode, fuse_support_logits


def binary_metrics(pred, target):
    pred, target = np.asarray(pred, bool), np.asarray(target, bool)
    inter = int((pred & target).sum())
    area_pred, area_gt = int(pred.sum()), int(target.sum())
    return dict(intersection=inter, pred_area=area_pred, gt_area=area_gt,
                dice=100 * 2 * inter / max(area_pred + area_gt, 1),
                precision=100 * inter / max(area_pred, 1),
                recall=100 * inter / max(area_gt, 1),
                area_ratio=area_pred / max(area_gt, 1), empty_prediction=area_pred == 0)


def localization_metrics(raw, target):
    """Evaluate heatmaps on an occupancy grid so tiny GT is never nearest-sampled away."""
    raw = np.asarray(raw, np.float32)
    gt = torch.as_tensor(np.asarray(target, np.float32))[None, None]
    occupied = F.adaptive_max_pool2d(gt, raw.shape)[0, 0].numpy() > 0
    dynamic_range = float(raw.max() - raw.min())
    out = dict(raw_min=float(raw.min()), raw_max=float(raw.max()), raw_range=dynamic_range,
               raw_mean_gt=float(raw[occupied].mean()) if occupied.any() else None,
               raw_mean_bg=float(raw[~occupied].mean()) if (~occupied).any() else None,
               constant_heatmap=dynamic_range <= 1e-12)
    if not occupied.any() or out['constant_heatmap']:
        out.update(peak_hit=None, peak_distance_px=None, peak_distance_target_diagonals=None,
                   top1pct_gt_grid_coverage=None)
        return out
    y, x = np.unravel_index(raw.argmax(), raw.shape)
    yy, xx = np.where(target)
    peak_xy = np.array([(x + .5) * target.shape[1] / raw.shape[1],
                        (y + .5) * target.shape[0] / raw.shape[0]])
    center = np.array([xx.mean() + .5, yy.mean() + .5])
    distance = float(np.linalg.norm(peak_xy - center))
    diagonal = math.hypot(xx.max() - xx.min() + 1, yy.max() - yy.min() + 1)
    k = max(1, math.ceil(raw.size * .01))
    top = np.zeros(raw.size, bool)
    top[np.argpartition(raw.ravel(), -k)[-k:]] = True
    out.update(peak_hit=bool(occupied[y, x]), peak_distance_px=distance,
               peak_distance_target_diagonals=distance / diagonal,
               top1pct_gt_grid_coverage=100 * float((top.reshape(raw.shape) & occupied).sum())
               / int(occupied.sum()))
    return out


def context_box(mask, min_size=64, scale=2.):
    """Square crop in native pixels; min_size is expressed on the 512 input grid."""
    yy, xx = np.where(mask)
    if not len(xx):
        raise ValueError('Cannot construct a GT-guided crop from an empty mask')
    h, w = mask.shape
    side = math.ceil(max((xx.max() - xx.min() + 1) * scale,
                         (yy.max() - yy.min() + 1) * scale,
                         min_size * max(h, w) / 512))
    crop_w, crop_h = min(side, w), min(side, h)
    left = min(max(math.floor((xx.min() + xx.max() + 1 - crop_w) / 2), 0), w - crop_w)
    top = min(max(math.floor((yy.min() + yy.max() + 1 - crop_h) / 2), 0), h - crop_h)
    return int(left), int(top), int(left + crop_w), int(top + crop_h)


def grid_box(native_box, native_size, size=512):
    w, h = native_size
    l, t, r, b = native_box
    return (max(0, math.floor(l * size / w)), max(0, math.floor(t * size / h)),
            min(size, math.ceil(r * size / w)), min(size, math.ceil(b * size / h)))


def restore_crop(logits, box, size=512, native_box=None, native_size=None):
    if native_box is not None:
        # Preserve fractional native->512 offsets (e.g. 528px ADAM slices).
        # Rounding a crop to integer 512 coordinates can move a tiny lesion.
        w, h = native_size
        l, t, r, b = native_box
        x = (torch.arange(size, dtype=torch.float32) + .5) * w / size
        y = (torch.arange(size, dtype=torch.float32) + .5) * h / size
        gy, gx = torch.meshgrid(2 * (y-t)/(b-t) - 1, 2 * (x-l)/(r-l) - 1, indexing='ij')
        grid = torch.stack([gx, gy], dim=-1)[None]
        restored = F.grid_sample(torch.as_tensor(logits)[None, None].float(), grid,
                                 mode='bilinear', padding_mode='border', align_corners=False)[0, 0]
        valid = (gx >= -1) & (gx < 1) & (gy >= -1) & (gy < 1)
        return torch.where(valid, restored, torch.full_like(restored, -30.)).numpy()
    l, t, r, b = box
    restored = torch.full((size, size), -30., dtype=torch.float32)
    resized = F.interpolate(torch.as_tensor(logits)[None, None].float(),
                            size=(b - t, r - l), mode='bilinear', align_corners=True)[0, 0]
    restored[t:b, l:r] = resized
    return restored.numpy()


def restrict_to_box(pred, box):
    l, t, r, b = box
    result = np.zeros_like(pred, dtype=bool)
    result[t:b, l:r] = pred[t:b, l:r]
    return result


def log_config(checkpoint, overrides):
    log = checkpoint.parent / 'log.txt'
    text = log.read_text() if log.exists() else ''
    params = dict(re.findall(r'^\|\s*(\w+):\s*(.*?)\s*$', text, re.M))
    sidecar = checkpoint.parent / 'run.json'
    if not text and sidecar.exists():
        params.update(json.loads(sidecar.read_text())['model'])
    pooling = params.get('support_mask_pooling')
    if pooling is None:
        pooling = 'max' if '[Support mask] downsample=adaptive_max_pool2d' in text else 'nearest'
    config = dict(num_query=int(params.get('num_query', 50)), backbone=params.get('backbone', 'resnet50'),
                  shot=int(params.get('shot', 4)), seed=int(params.get('seed', 321)),
                  neighbor_prob=float(params.get('neighbor_prob', 1)), support_mask_pooling=pooling,
                  bbox_support_attn_mask=int(params.get('bbox_support_attn_mask', 0)),
                  icl_support_attn_mask=int(params.get('icl_support_attn_mask', 0)),
                  support_fusion=params.get('support_fusion', 'mean'), benchmark='joint')
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    saved = re.findall(r'Model saved @(\d+) w/ val\. Dice:\s*(\d+(?:\.\d+)?)', text)
    return config, dict(log=str(log), last_logged_save=saved[-1] if saved else None)


def datasets(args, config):
    from data.dataset import FSSDataset
    FSSDataset.initialize(512, args.btcv_root, False)
    result = {}
    for name, path in [('adam', args.adam_root), ('btcv2', args.btcv_root)]:
        result[name] = FSSDataset.datasets[name](
            path, 0, FSSDataset.transform, 'val', config['shot'], False,
            neighbor_prob=config['neighbor_prob'])
    return result


def resized_mask(ds, name, class_id):
    binary = (ds.read_mask(name) == class_id + 1).float()[None, None]
    return F.interpolate(binary, size=(512, 512), mode='nearest')[0, 0].numpy()


def patient_name(name, dataset):
    patient = name.replace('x___', '').replace('y___', '').rsplit('_', 1)[0]
    return re.sub('[BF]$', '', patient) if dataset == 'adam' else patient


def prepare_manifest(ds_map, config, args):
    targets = [('adam', 0, args.adam_queries), ('btcv2', 11, args.btcv_queries),
               ('btcv2', 12, args.btcv_queries), ('btcv2', 9, args.btcv_queries)]
    candidates = {}
    for name, ds in ds_map.items():
        wanted = {c for n, c, _ in targets if n == name}
        for rank in range(args.validation_ranks):
            np.random.seed(config['seed'])
            # DistributedSampler pads [0..999] before rank-striding.
            length = len(ds)
            padded = list(range(length))
            padded += padded[:(-length) % args.validation_ranks]
            for episode_index in padded[rank::args.validation_ranks]:
                q, ss, c = ds.sample_episode(episode_index % len(ds.img_metadata))
                c = int(c)
                if c not in wanted:
                    continue
                key = name, c, q
                includes_query = q in ss
                # Prefer an actual validation draw without query self-support.
                if key in candidates and (not candidates[key]['support_includes_query'] or includes_query):
                    continue
                target = resized_mask(ds, q, c)
                if not target.any():
                    continue
                class_name = ds.class_names[ds.class_ids.index(c)]
                candidates[key] = dict(dataset=name, class_id=c, class_name=class_name,
                                       query_name=q, support_names=list(ss), rank=rank,
                                       episode_index=episode_index, patient=patient_name(q, name),
                                       gt_area512=int(target.sum()), support_includes_query=includes_query)
    groups = []
    for name, c, limit in targets:
        pool = sorted([v for (n, cc, _), v in candidates.items() if n == name and cc == c],
                      key=lambda v: (v['gt_area512'], v['query_name']))
        if not pool:
            raise ValueError(f'No positive validation episodes for {name}, class {c}')
        indices = np.linspace(0, len(pool) - 1, min(limit, len(pool)), dtype=int)
        groups.append([pool[i] for i in indices])
    episodes = []
    for i in range(max(map(len, groups))):
        for group in groups:
            if i < len(group):
                row = dict(group[i])
                row['id'] = f'{len(episodes):03d}_{row["dataset"]}_{row["class_name"]}'
                episodes.append(row)
    return dict(seed=config['seed'], shot=config['shot'], neighbor_prob=config['neighbor_prob'],
                validation_ranks=args.validation_ranks,
                selection='Unique query/class, area-stratified; actual val support draws, prefer no self-support',
                episodes=episodes)


def load_view(ds, episode, crop_min=None, crop_scale=2.):
    images, masks, displays, boxes, native_boxes = [], [], [], [], []
    native_sizes = []
    for name in [episode['query_name']] + episode['support_names']:
        image = ds.read_img(name)
        target = (ds.read_mask(name) == episode['class_id'] + 1).numpy()
        native_sizes.append(image.size)
        box = context_box(target, crop_min, crop_scale) if crop_min is not None else (0, 0, *image.size)
        l, t, r, b = box
        view_image, view_mask = image.crop(box), target[t:b, l:r]
        images.append(ds.transform(view_image))
        masks.append(F.interpolate(torch.from_numpy(view_mask.astype(np.float32))[None, None],
                                   size=(512, 512), mode='nearest')[0, 0])
        displays.append(np.asarray(view_image.resize((512, 512), Image.Resampling.BILINEAR)))
        boxes.append(grid_box(box, image.size))
        native_boxes.append(box)
    return dict(query=images[0][None], supports=torch.stack(images[1:]),
                target=masks[0].numpy() > 0, support_masks=torch.stack(masks[1:]),
                displays=displays, boxes=boxes, native_boxes=native_boxes, native_sizes=native_sizes)


def feature_cache_path(args, output, episode, tag):
    if tag == 'full':
        shared = args.feature_cache / (episode['query_name'] + '.npy')
        if shared.exists():
            return shared
    folder = output / 'features'
    folder.mkdir(exist_ok=True)
    return folder / f'{episode["id"]}_{tag}.npy'


@torch.inference_mode()
def predict_view(model, sam, view, episode, feature_path, device):
    query, supports = view['query'].to(device), view['supports'].to(device)
    masks = view['support_masks'].to(device)
    shot = supports.shape[0]
    capture = {}
    def capture_eicl(module, inputs, outputs):
        reduced = inputs[2]
        attention = module.last_A
        raw = torch.bmm(attention, reduced.flatten(2).transpose(1, 2))
        capture['raw_pseudo'] = raw.transpose(1, 2).reshape_as(reduced).cpu().numpy()[:, 0]
        capture['support_grid_mask'] = reduced.cpu().numpy()[:, 0]
        module.last_A = None  # Never persist the large attention matrix.
    handle = model.eicl_interaction.register_forward_hook(capture_eicl)
    quality = getattr(model, 'support_fusion', 'mean') == 'quality'
    try:
        outputs = model('mask', query.repeat_interleave(shot, 0), supports,
                        masks, False, dataset_names=episode['dataset'],
                        **({'return_quality': True} if quality else {}))
        prompts, boxes, pseudo = outputs[:3]
    finally:
        handle.remove()
    capture['pseudo'] = pseudo.cpu().numpy()[:, 0]
    protos = torch.cat([prompts, boxes], dim=1)
    if feature_path.exists():
        array = np.load(feature_path, allow_pickle=False)
        if array.shape != (1, 256, 64, 64) or not np.isfinite(array).all():
            raise ValueError(f'Invalid cached query embedding: {feature_path}')
        features = torch.from_numpy(array).float().to(device)
    else:
        features = sam.forward_img_encoder(query)
        np.save(feature_path, features.cpu().numpy())
    sparse, dense = sam.forward_prompt_encoder(protos=protos)
    logits, _ = sam.forward_mask_decoder(features.repeat_interleave(shot, 0), sparse, dense,
                                         ori_size=(512, 512))
    capture['logits'] = logits.cpu().numpy()[:, 0]
    capture['mean_logits'] = capture['logits'].mean(axis=0)
    scores = outputs[3].reshape(1, shot) if quality else None
    mix = model.support_quality_head.mix if quality else 1.0
    fused, weights = fuse_support_logits(logits[:, 0].unsqueeze(0), scores, mix)
    capture['fused_logits'] = fused[0].cpu().numpy()
    capture['fusion_weights'] = (weights[0].cpu().numpy() if weights is not None
                                 else np.full(shot, 1 / shot, dtype=np.float32))
    return capture


def save_figures(folder, episode, full, result, crops):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def contour(ax, target, color='lime'):
        if target.any() and not target.all():
            ax.contour(target.astype(float), levels=[.5], colors=[color], linewidths=.65)
    def probability(logits):
        return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    target = full['target']
    pred = result['fused_logits'] > 0
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(full['displays'][0]); contour(axes[0, 0], target)
    axes[0, 0].set_title(f'Query + GT; area={target.sum()} px')
    axes[0, 1].imshow(result['pseudo'].mean(0), vmin=0, vmax=1, cmap='magma')
    coarse_gt = F.adaptive_max_pool2d(torch.from_numpy(target.astype(np.float32))[None, None],
                                     result['pseudo'].shape[-2:])[0, 0].numpy() > 0
    contour(axes[0, 1], coarse_gt)
    axes[0, 1].set_title('Mean normalized pseudo + GT grid')
    im = axes[0, 2].imshow(result['raw_pseudo'].mean(0), cmap='magma')
    fig.colorbar(im, ax=axes[0, 2], fraction=.046)
    contour(axes[0, 2], coarse_gt)
    axes[0, 2].set_title('Mean raw pseudo (own color scale)')
    axes[1, 0].imshow(probability(result['fused_logits']), vmin=0, vmax=1, cmap='magma')
    contour(axes[1, 0], target);axes[1, 0].set_title('Sigmoid(fused logits) + GT')
    error = np.zeros((*target.shape, 3), np.float32)
    error[..., 0] = target & ~pred
    error[..., 1] = target & pred
    error[..., 2] = ~target & pred
    axes[1, 1].imshow(error);axes[1, 1].set_title('Red=FN, blue=FP, green=TP')
    ys, xs = np.where(target)
    l, t, r, b = context_box(target, 64, 2)
    axes[1, 2].imshow(full['displays'][0][t:b, l:r])
    contour(axes[1, 2], target[t:b, l:r]);contour(axes[1, 2], pred[t:b, l:r], 'red')
    axes[1, 2].set_title('Zoom for inspection: GT green, prediction red')
    for ax in axes.flat:ax.axis('off')
    fig.suptitle(f'{episode["id"]}: {episode["query_name"]} | full Dice={binary_metrics(pred,target)["dice"]:.2f}')
    fig.tight_layout();fig.savefig(folder / 'overview.png', dpi=135);plt.close(fig)

    shot = len(result['logits'])
    fig, axes = plt.subplots(3, shot, figsize=(4 * shot, 10), squeeze=False)
    for i in range(shot):
        axes[0, i].imshow(full['displays'][i + 1]);contour(axes[0, i], full['support_masks'][i].numpy() > 0)
        axes[0, i].set_title(f'Support {i+1}: {episode["support_names"][i]}\n'
                             f'area={int(full["support_masks"][i].sum())}; grid={int(result["support_grid_mask"][i].sum())}', fontsize=8)
        im = axes[1, i].imshow(result['raw_pseudo'][i], cmap='magma')
        contour(axes[1, i], coarse_gt);fig.colorbar(im, ax=axes[1, i], fraction=.046)
        axes[1, i].set_title('Raw pseudo; GT green')
        axes[2, i].imshow(probability(result['logits'][i]), vmin=0, vmax=1, cmap='magma')
        contour(axes[2, i], target)
        score = binary_metrics(result['logits'][i] > 0, target)
        axes[2, i].set_title(f'Single-support Dice={score["dice"]:.2f}; recall={score["recall"]:.1f}%\n'
                             f'fusion weight={result["fusion_weights"][i]:.3f}')
    for ax in axes.flat:ax.axis('off')
    fig.tight_layout();fig.savefig(folder / 'supports.png', dpi=135);plt.close(fig)

    fig, axes = plt.subplots(2, len(crops) + 1, figsize=(4 * (len(crops) + 1), 8), squeeze=False)
    axes[0, 0].imshow(full['displays'][0]);contour(axes[0, 0], target)
    axes[0, 0].set_title('Full input')
    axes[1, 0].imshow(probability(result['fused_logits']), vmin=0, vmax=1, cmap='magma')
    contour(axes[1, 0], target);axes[1, 0].set_title(f'Full Dice={binary_metrics(pred,target)["dice"]:.2f}')
    for col, crop in enumerate(crops, 1):
        view, output, restored, control = crop['view'], crop['output'], crop['restored'], crop['control']
        axes[0, col].imshow(view['displays'][0]);contour(axes[0, col], view['target'])
        contour(axes[0, col], output['fused_logits'] > 0, 'red')
        axes[0, col].set_title(f'GT crop min={crop["min_size"]}; GT green, pred red')
        axes[1, col].imshow(probability(restored), vmin=0, vmax=1, cmap='magma')
        contour(axes[1, col], target)
        crop_dice = binary_metrics(restored > 0, target)['dice']
        control_dice = binary_metrics(control, target)['dice']
        axes[1, col].set_title(f'Restored Dice={crop_dice:.2f}\nFull prediction restricted to same ROI={control_dice:.2f}')
    for ax in axes.flat:ax.axis('off')
    fig.suptitle('GT-guided crop diagnostic; full prompt generator retained; not an evaluation score')
    fig.tight_layout();fig.savefig(folder / 'crops.png', dpi=135);plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def summarize(rows, output):
    groups = {}
    for row in rows:
        key = row['dataset'], row['class_name'], row['variant']
        groups.setdefault(key, []).append(row)
    summaries = []
    for (dataset, name, variant), rr in groups.items():
        sums = {k:sum(r[k] for r in rr) for k in ['intersection', 'gt_area', 'pred_area']}
        summaries.append(dict(dataset=dataset, class_name=name, variant=variant, episodes=len(rr),
                              mean_dice=float(np.mean([r['dice'] for r in rr])),
                              pooled_dice=200 * sums['intersection'] / max(sums['gt_area'] + sums['pred_area'], 1),
                              mean_recall=float(np.mean([r['recall'] for r in rr])),
                              mean_precision=float(np.mean([r['precision'] for r in rr])),
                              empty_pct=100 * float(np.mean([r['empty_prediction'] for r in rr]))))
    write_csv(output / 'summary.csv', summaries)
    lines = ['# Small-target diagnostics', '',
             'Fixed unique-query diagnostic subset; scores are not the full validation result.',
             'Crops use GT query/support ROIs and include localization assistance. Compare crops',
             'with both full inference and the full prediction restricted to the same ROI.',
             'No GT bbox token replacement is performed. Per-shot best selection is oracle-only.', '',
             '| Dataset | Class | Variant | N | Mean Dice | Pooled Dice | Recall | Empty % |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f'| {row["dataset"]} | {row["class_name"]} | {row["variant"]} | {row["episodes"]} '
                     f'| {row["mean_dice"]:.2f} | {row["pooled_dice"]:.2f} | {row["mean_recall"]:.2f} | {row["empty_pct"]:.1f} |')
    lines += ['', 'See episodes.csv, shots.csv, localization.csv and each episode folder for raw arrays and figures.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--adam-root', default=str(ROOT.parent / 'ADAM2020'))
    parser.add_argument('--btcv-root', default=str(ROOT.parent / 'BTCV'))
    parser.add_argument('--adam-queries', type=int, default=17)
    parser.add_argument('--btcv-queries', type=int, default=8, help='Per target BTCV class')
    parser.add_argument('--validation-ranks', type=int, default=3)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--crop-min-sizes', default='64,128')
    parser.add_argument('--crop-scale', type=float, default=2.)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.25)
    parser.add_argument('--feature-cache', type=Path, default=Path(os.environ.get(
        'TBD_FEATS', '/data2/2025Zhutaotao/TBD/cache_features')) / 'eval_unaugmented_v1')
    parser.add_argument('--support-mask-pooling', choices=['max', 'nearest', 'adam-only'])
    parser.add_argument('--bbox-support-attn-mask', type=int, choices=[0, 1])
    parser.add_argument('--icl-support-attn-mask', type=int, choices=[0, 1])
    parser.add_argument('--support-fusion', choices=['mean', 'quality'])
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.adam_queries, args.btcv_queries, args.validation_ranks) < 1:
        parser.error('Query counts and validation-ranks must be positive')
    if args.crop_scale < 1 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error('crop-scale must be >=1 and memory fraction must be in (0,1]')
    crop_sizes = [int(x) for x in args.crop_min_sizes.split(',')]
    if not crop_sizes or min(crop_sizes) < 1:
        parser.error('crop-min-sizes must contain positive sizes')
    os.chdir(ROOT)
    torch.set_num_threads(args.threads)
    config, source = log_config(args.checkpoint, {
        'support_mask_pooling':args.support_mask_pooling,
        'bbox_support_attn_mask':args.bbox_support_attn_mask,
        'icl_support_attn_mask':args.icl_support_attn_mask,
        'support_fusion':args.support_fusion})
    torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f'Output directory is not empty: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    ds_map = datasets(args, config)
    manifest = json.loads(args.manifest.read_text()) if args.manifest else prepare_manifest(ds_map, config, args)
    if manifest['shot'] != config['shot'] or manifest['neighbor_prob'] != config['neighbor_prob']:
        raise ValueError('Manifest support protocol does not match the checkpoint log')
    if args.limit is not None:
        manifest['episodes'] = manifest['episodes'][:args.limit]
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    run_config = dict(model=config, source=source, arguments={k:str(v) if isinstance(v, Path) else v
                      for k,v in vars(args).items()}, started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    (args.output / 'run.json').write_text(json.dumps(run_config, indent=2))
    print(f'Prepared {len(manifest["episodes"])} fixed episodes at {args.output}', flush=True)
    if args.prepare_only:
        return
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this process; execute on a GPU-enabled server')
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    from model.VRP_encoder import VRP_encoder
    from model.base import resnet
    from SAM2pred import SAM_pred
    if config['backbone'] != 'resnet50':
        raise ValueError('This diagnostic loader currently supports the experiment\'s resnet50 backbone')
    # The joint checkpoint includes the full backbone: avoid pretrained downloads.
    stat = args.checkpoint.stat()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if args.checkpoint.stat().st_mtime_ns != stat.st_mtime_ns:
        raise RuntimeError('Checkpoint changed during loading; retry with a stable snapshot')
    torch.save(checkpoint, args.output / 'checkpoint_snapshot.pt')
    with (args.output / 'checkpoint_snapshot.pt').open('rb') as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda:f.read(1024 * 1024), b''):
            digest.update(chunk)
    run_config['checkpoint_snapshot_sha256'] = digest.hexdigest()
    run_config['checkpoint_source_mtime_ns'] = stat.st_mtime_ns
    config['support_fusion'] = checkpoint_fusion_mode(checkpoint, args.support_fusion)
    (args.output / 'run.json').write_text(json.dumps(run_config, indent=2))
    factory = resnet.resnet50
    with mock.patch('model.VRP_encoder.models.resnet50', side_effect=lambda **kw: factory(pretrained=False)):
        model = VRP_encoder(SimpleNamespace(**config), 'resnet50', False)
    state = {k.removeprefix('module.'):v for k,v in checkpoint['model'].items()}
    model.load_state_dict(state, strict=True)
    sam = SAM_pred(encoder_precision='fp32')
    sam_state = {k.removeprefix('module.'):v for k,v in checkpoint['sam_model'].items()}
    mismatch = sam.load_state_dict(sam_state, strict=False)
    if mismatch.unexpected_keys or any(not k.startswith('sam_model.image_encoder.') for k in mismatch.missing_keys):
        raise ValueError(f'Unexpected SAM checkpoint mismatch: {mismatch}')
    del checkpoint, state, sam_state
    model.to(device).eval();sam.to(device).eval()
    episode_rows, shot_rows, localization_rows = [], [], []
    for index, episode in enumerate(manifest['episodes']):
        folder = args.output / episode['id'];folder.mkdir()
        ds = ds_map[episode['dataset']]
        full = load_view(ds, episode)
        output = predict_view(model, sam, full, episode,
                              feature_cache_path(args, args.output, episode, 'full'), device)
        target = full['target'];pred = output['fused_logits'] > 0
        identity = {k:episode[k] for k in ['id', 'dataset', 'class_name', 'query_name', 'patient']}
        base_metrics = binary_metrics(pred, target)
        scores = [binary_metrics(logits > 0, target) for logits in output['logits']]
        episode_rows.append(dict(identity, variant='full', **base_metrics,
                                 best_single_dice=max(s['dice'] for s in scores),
                                 best_minus_ensemble=max(s['dice'] for s in scores) - base_metrics['dice']))
        for i, score in enumerate(scores):
            shot_rows.append(dict(identity, support_index=i, support_name=episode['support_names'][i],
                                  support_area=int(full['support_masks'][i].sum()),
                                  fusion_weight=float(output['fusion_weights'][i]),
                                  support_grid_cells=int(output['support_grid_mask'][i].sum()), **score))
            localization_rows.append(dict(identity, support_index=i,
                                           **localization_metrics(output['raw_pseudo'][i], target)))
        localization_rows.append(dict(identity, support_index='mean',
                                       **localization_metrics(output['raw_pseudo'].mean(0), target)))
        arrays = {k:v for k,v in output.items()}
        arrays['target'] = target
        crops = []
        for min_size in crop_sizes:
            view = load_view(ds, episode, min_size, args.crop_scale)
            crop_output = predict_view(model, sam, view, episode,
                feature_cache_path(args, args.output, episode, f'crop{min_size}'), device)
            restored = restore_crop(crop_output['fused_logits'], view['boxes'][0],
                                    native_box=view['native_boxes'][0], native_size=view['native_sizes'][0])
            control = restrict_to_box(pred, view['boxes'][0])
            episode_rows.append(dict(identity, variant=f'crop{min_size}',
                                     query_box=str(view['boxes'][0]),
                                     **binary_metrics(restored > 0, target)))
            episode_rows.append(dict(identity, variant=f'full_roi{min_size}',
                                     query_box=str(view['boxes'][0]), **binary_metrics(control, target)))
            arrays[f'crop{min_size}_restored_logits'] = restored
            arrays[f'crop{min_size}_box'] = np.array(view['boxes'][0])
            arrays[f'crop{min_size}_native_box'] = np.array(view['native_boxes'][0])
            arrays[f'crop{min_size}_view_logits'] = crop_output['logits']
            crops.append(dict(view=view, output=crop_output, restored=restored,
                              control=control, min_size=min_size))
        np.savez_compressed(folder / 'arrays.npz', **arrays)
        save_figures(folder, episode, full, output, crops)
        write_csv(args.output / 'episodes.csv', episode_rows)
        write_csv(args.output / 'shots.csv', shot_rows)
        write_csv(args.output / 'localization.csv', localization_rows)
        summarize(episode_rows, args.output)
        print(f'[{index+1}/{len(manifest["episodes"])}] {episode["id"]} '
              f'full={base_metrics["dice"]:.2f}, best-shot={max(s["dice"] for s in scores):.2f}, '
              + ', '.join(f'crop{c["min_size"]}={binary_metrics(c["restored"]>0,target)["dice"]:.2f}' for c in crops), flush=True)
    print(f'Done: {args.output / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
