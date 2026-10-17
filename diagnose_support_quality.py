"""Read-only, paired four-dataset support-quality audit using frozen FP32 SAM caches.

No CUDA device is used. Selection replays the current validation sampler, then
selects unique query/class pairs before inference, balancing patients and size.
This is a diagnostic subset, not a replacement for complete validation scores.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'segment-anything'))
import numpy as np
import torch
import torch.nn.functional as F
from diagnose_small_targets import load_view, patient_name, resized_mask, write_csv
from model.base.support_fusion import checkpoint_fusion_mode, fuse_support_logits


def analyze_prediction(logits, scores, weights, target, temperature=.2):
    """Metrics in Dice percentage points; ties never favor a support position."""
    logits = np.asarray(logits, np.float32)
    scores, weights = np.asarray(scores, float), np.asarray(weights, float)
    target = np.asarray(target, bool)
    if logits.ndim != 3 or logits.shape[1:] != target.shape:
        raise ValueError('Expected shot logits [S,H,W] and target [H,W]')
    if len(scores) != len(logits) or weights.shape != scores.shape:
        raise ValueError('Expected one score and weight per support')
    if not np.isfinite(logits).all() or not np.isfinite(scores).all():
        raise ValueError('Nonfinite prediction')
    shot = len(logits)
    # Use the exact training/valid threshold, including floating point at zero.
    tensor = torch.from_numpy(logits)
    predictions = tensor.sigmoid().numpy() > .5
    intersection = (predictions & target).sum((1, 2)).astype(float)
    pred_area = predictions.sum((1, 2)).astype(float)
    gt_area = int(target.sum())
    dice = 200 * intersection / np.maximum(pred_area + gt_area, 1)
    uniform_logits = tensor.mean(0)
    # Reuse the same float32 fusion arithmetic as train.py.
    fused_logits = (tensor * torch.as_tensor(weights, dtype=torch.float32)[:, None, None]).sum(0)
    def metrics(values):
        pred = values.sigmoid().numpy() > .5
        inter, area = int((pred & target).sum()), int(pred.sum())
        return dict(dice=200 * inter / max(area + gt_area, 1),
                    intersection=inter, pred_area=area)
    uniform, fused = metrics(uniform_logits), metrics(fused_logits)
    best = np.isclose(dice, dice.max(), atol=1e-6, rtol=0)
    chosen = np.isclose(weights, weights.max(), atol=1e-8, rtol=0)
    pair_correct = pair_count = 0
    for i in range(shot):
        for j in range(i):
            if abs(dice[i] - dice[j]) < 1.:
                continue
            difference = scores[i] - scores[j]
            pair_count += 1
            pair_correct += (.5 if abs(difference) <= 1e-7
                             else float(difference * (dice[i] - dice[j]) > 0))
    d = torch.tensor(dice / 100, dtype=torch.float64)
    beta = (d / temperature).softmax(0).numpy()
    shifted = scores - scores.max()
    predicted = np.exp(shifted); predicted /= predicted.sum()
    teacher_kl = float(np.sum(beta * np.log(np.maximum(beta, 1e-30) / np.maximum(predicted, 1e-30))))
    uniform_kl = float(np.sum(beta * np.log(np.maximum(beta, 1e-30) * shot)))
    entropy = float(-(weights * np.log(np.maximum(weights, 1e-30))).sum() / math.log(shot))
    return dict(
        gt_area=gt_area, valid_query=bool(gt_area), dice=dice.tolist(),
        intersections=intersection.astype(int).tolist(), pred_areas=pred_area.astype(int).tolist(),
        scores=scores.tolist(), weights=weights.tolist(), teacher_weights=beta.tolist(),
        uniform_dice=uniform['dice'], weighted_dice=fused['dice'],
        uniform_intersection=uniform['intersection'], uniform_pred_area=uniform['pred_area'],
        weighted_intersection=fused['intersection'], weighted_pred_area=fused['pred_area'],
        weighted_minus_uniform=fused['dice'] - uniform['dice'],
        random_single_dice=float(dice.mean()), selected_single_dice=float(dice[chosen].mean()),
        selected_minus_random=float(dice[chosen].mean() - dice.mean()),
        best_single_dice=float(dice.max()), worst_single_dice=float(dice.min()),
        dice_spread=float(np.ptp(dice)), all_shots_zero=bool(dice.max() == 0),
        top1_hit=float(best[chosen].mean()), random_top1_hit=float(best.mean()),
        pair_correct=pair_correct, pair_count=pair_count,
        max_weight=float(weights.max()), weight_spread=float(np.ptp(weights)),
        score_spread=float(np.ptp(scores)), normalized_entropy=entropy,
        teacher_max_weight=float(beta.max()), teacher_kl=teacher_kl, uniform_teacher_kl=uniform_kl,
        teacher_normalized_entropy=float(-(beta * np.log(np.maximum(beta, 1e-30))).sum() / math.log(shot)),
        changed_pixels=int(((uniform_logits.sigmoid().numpy() > .5)
                            != (fused_logits.sigmoid().numpy() > .5)).sum()))


def initialize_datasets(config):
    from data.dataset import FSSDataset
    entries = [entry.split(':', 1) for entry in config['joint_subdatasets'].split(',')]
    FSSDataset.initialize(512, entries[0][1], False)
    return {name: FSSDataset.datasets[name](
        path, 0, FSSDataset.transform, 'val', int(config['shot']), False,
        neighbor_prob=float(config['neighbor_prob'])) for name, path in entries}


def balanced_select(rows, limit):
    """Round-robin patients; within each patient spread selected areas evenly."""
    patients = defaultdict(list)
    for row in rows:
        patients[row['patient']].append(row)
    if len(patients) >= limit:
        # When only one query fits per patient, taking the first queue item
        # would select every patient's smallest target. Spread area quantiles
        # across the selected patients instead, before running predictions.
        names = sorted(patients)
        indices = np.linspace(0, len(names) - 1, limit, dtype=int)
        selected = []
        for i, index in enumerate(indices):
            ordered = sorted(patients[names[index]],
                             key=lambda r: (r['gt_area512'], r['query_name']))
            quantile = (i + .5) / limit
            position = round(quantile * (len(ordered) - 1))
            selected.append(ordered[position])
        return selected
    queues = {}
    for patient, group in sorted(patients.items()):
        ordered = sorted(group, key=lambda r: (r['gt_area512'], r['query_name']))
        # Allocate at most limit/patient plus one; retain both small/large cases.
        count = min(len(ordered), max(2, math.ceil(limit / len(patients))))
        indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
        chosen = [ordered[i] for i in indices]
        remaining = [row for row in ordered if row not in chosen]
        queues[patient] = chosen + remaining
    selected = []
    while len(selected) < min(limit, len(rows)):
        for patient in queues:
            if queues[patient] and len(selected) < limit:
                selected.append(queues[patient].pop(0))
    return selected


def prepare_manifest(config, query_limit, validation_ranks, feature_cache):
    datasets = initialize_datasets(config)
    selected, coverage = [], []
    for name, ds in datasets.items():
        candidates = {}
        for rank in range(validation_ranks):
            np.random.seed(int(config['seed']))
            padded = list(range(len(ds)))
            padded += padded[:(-len(ds)) % validation_ranks]
            for index in padded[rank::validation_ranks]:
                q, supports, class_id = ds.sample_episode(index % len(ds.img_metadata))
                class_id = int(class_id)
                key = class_id, q
                row = dict(dataset=name, class_id=class_id,
                           class_name=ds.class_names[ds.class_ids.index(class_id)],
                           query_name=q, support_names=list(supports),
                           patient=patient_name(q, name), rank=rank, episode_index=index,
                           support_includes_query=q in supports)
                if key not in candidates or (candidates[key]['support_includes_query'] and q not in supports):
                    candidates[key] = row
        for class_id, class_name in zip(ds.class_ids, ds.class_names):
            raw = [row for (c, _), row in candidates.items() if c == class_id]
            valid, missing_cache, empty_query, self_support = [], 0, 0, 0
            for row in raw:
                if row['support_includes_query']:
                    self_support += 1
                    continue
                if not (feature_cache / (row['query_name'] + '.npy')).exists():
                    missing_cache += 1
                    continue
                target = resized_mask(ds, row['query_name'], class_id)
                if not target.any():
                    empty_query += 1
                    continue
                row['gt_area512'] = int(target.sum())
                valid.append(row)
            chosen = balanced_select(valid, query_limit)
            coverage.append(dict(dataset=name, class_id=class_id, class_name=class_name,
                                 available_unique_queries=len(raw), eligible_queries=len(valid),
                                 selected_queries=len(chosen), selected_patients=len({r['patient'] for r in chosen}),
                                 excluded_self_support=self_support, excluded_missing_cache=missing_cache,
                                 excluded_empty_query=empty_query))
            selected.extend(chosen)
        print(f'Prepared {name}: {sum(r["dataset"] == name for r in selected)} unique query/class pairs', flush=True)
    for index, row in enumerate(selected):
        row['id'] = f'{index:04d}_{row["dataset"]}_{row["class_name"]}'
    return dict(validation_ranks=validation_ranks, queries_per_class=query_limit,
                selection='Replay validation; unique query/class; exclude self-support and empty query; patient/area stratified before inference',
                episodes=selected, coverage=coverage)


def load_models(checkpoint, config):
    """Load the real prompt model and the exact SAM prompt/mask decoder modules.

    Image encoder is omitted because all embeddings must already be cached from
    unaugmented FP32 validation. Both submodules are loaded strictly, not rebuilt
    from a reduced or approximated model.
    """
    from model.VRP_encoder import VRP_encoder
    from model.base import resnet
    from segment_anything.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer
    if config['backbone'] != 'resnet50':
        raise ValueError('This loader supports the current resnet50 experiment')
    factory = resnet.resnet50
    with mock.patch('model.VRP_encoder.models.resnet50', side_effect=lambda **kw: factory(pretrained=False)):
        model = VRP_encoder(SimpleNamespace(**config), 'resnet50', False)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in checkpoint['model'].items()}, strict=True)
    prompt = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64),
                           input_image_size=(1024, 1024), mask_in_chans=16)
    decoder = MaskDecoder(num_multimask_outputs=3, transformer=TwoWayTransformer(
        depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
        transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256)
    state = {k.removeprefix('module.'): v for k, v in checkpoint['sam_model'].items()}
    for name, module in [('prompt_encoder', prompt), ('mask_decoder', decoder)]:
        prefix = 'sam_model.' + name + '.'
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        module.eval()
    return model.eval(), prompt, decoder


@torch.inference_mode()
def predict(model, prompt, decoder, view, episode, feature_cache):
    shot = view['supports'].shape[0]
    outputs = model('mask', view['query'].repeat_interleave(shot, 0), view['supports'],
                    view['support_masks'], False, dataset_names=episode['dataset'], return_quality=True)
    model.eicl_interaction.last_A = None
    scores = outputs[3].reshape(1, shot)
    features = np.load(feature_cache / (episode['query_name'] + '.npy'), allow_pickle=False)
    if features.shape != (1, 256, 64, 64) or not np.isfinite(features).all():
        raise ValueError('Invalid cached FP32 query embedding')
    sparse, dense = prompt(points=None, boxes=None, masks=None, protos=torch.cat(outputs[:2], 1))
    low_masks, _ = decoder(image_embeddings=torch.from_numpy(features).float().repeat_interleave(shot, 0),
                           image_pe=prompt.get_dense_pe(), sparse_prompt_embeddings=sparse,
                           dense_prompt_embeddings=dense, multimask_output=False)
    logits = F.interpolate(low_masks, size=(512, 512), mode='bilinear', align_corners=True)[:, 0]
    _, weights = fuse_support_logits(logits[None], scores, model.support_quality_head.mix)
    return logits.numpy(), scores[0].numpy(), weights[0].numpy()


def worker(index, workers, output, threads):
    os.chdir(ROOT)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    output = Path(output)
    run = json.loads((output / 'run.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    config = run['config']
    np.random.seed(int(config['seed'])); torch.manual_seed(int(config['seed']))
    datasets = initialize_datasets(config)
    checkpoint = torch.load(output / 'checkpoint_snapshot.pt', map_location='cpu', weights_only=True)
    model, prompt, decoder = load_models(checkpoint, config)
    del checkpoint
    start = time.perf_counter()
    episodes = manifest['episodes'][index::workers]
    with (output / f'worker_{index}.jsonl').open('w') as handle:
        for i, episode in enumerate(episodes):
            view = load_view(datasets[episode['dataset']], episode)
            logits, scores, weights = predict(model, prompt, decoder, view, episode, Path(run['feature_cache']))
            row = dict(episode, **analyze_prediction(logits, scores, weights, view['target'],
                                                   config.get('fusion_target_temperature', .2)))
            handle.write(json.dumps(row) + '\n'); handle.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(episodes):
                print(f'worker {index}: {i+1}/{len(episodes)}; elapsed={time.perf_counter()-start:.0f}s', flush=True)
    return len(episodes)


def clustered_ci(rows, key, seed=321):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row['patient']].append(float(row[key]))
    if len(clusters) < 2:
        return [None, None]
    sums = np.array([sum(v) for v in clusters.values()])
    counts = np.array([len(v) for v in clusters.values()])
    draws = np.random.default_rng(seed).integers(0, len(sums), size=(2000, len(sums)))
    values = sums[draws].sum(1) / counts[draws].sum(1)
    return np.percentile(values, [2.5, 97.5]).tolist()


def aggregate(rows):
    informative = [r for r in rows if r['dice_spread'] >= 1]
    high = [r for r in rows if r['dice_spread'] >= 10]
    standout = [r for r in rows if sorted(r['dice'])[-1] - sorted(r['dice'])[-2] >= 10]
    def avg(key, group=rows):
        return float(np.mean([r[key] for r in group])) if group else None
    pair_count = sum(r['pair_count'] for r in rows)
    out = dict(n=len(rows), patients=len({r['patient'] for r in rows}),
               informative_n=len(informative), high_spread_n=len(high),
               standout_best_n=len(standout),
               all_zero_n=sum(r['all_shots_zero'] for r in rows),
               pairwise_accuracy=sum(r['pair_correct'] for r in rows) / pair_count if pair_count else None,
               informative_top1_hit=avg('top1_hit', informative),
               informative_random_top1_hit=avg('random_top1_hit', informative),
               high_spread_max_weight=avg('max_weight', high),
               high_spread_selected_minus_random=avg('selected_minus_random', high),
               high_spread_fusion_delta=avg('weighted_minus_uniform', high),
               high_spread_top1_hit=avg('top1_hit', high),
               high_spread_random_top1_hit=avg('random_top1_hit', high),
               standout_max_weight=avg('max_weight', standout),
               standout_selected_minus_random=avg('selected_minus_random', standout),
               standout_fusion_delta=avg('weighted_minus_uniform', standout),
               standout_top1_hit=avg('top1_hit', standout),
               fusion_delta_ci=clustered_ci(rows, 'weighted_minus_uniform'),
               selected_delta_ci=clustered_ci(rows, 'selected_minus_random'),
               improved_n=sum(r['weighted_minus_uniform'] > .01 for r in rows),
               worsened_n=sum(r['weighted_minus_uniform'] < -.01 for r in rows),
               changed_query_n=sum(r['changed_pixels'] > 0 for r in rows))
    for key in ['uniform_dice', 'weighted_dice', 'weighted_minus_uniform', 'random_single_dice',
                'selected_single_dice', 'selected_minus_random', 'best_single_dice', 'dice_spread',
                'max_weight', 'weight_spread', 'normalized_entropy', 'teacher_max_weight',
                'teacher_normalized_entropy', 'teacher_kl', 'uniform_teacher_kl', 'changed_pixels']:
        out[key] = avg(key)
    out['max_weight_p95'] = float(np.percentile([r['max_weight'] for r in rows], 95))
    for method in ('uniform', 'weighted'):
        out[method + '_pooled_dice'] = 200 * sum(r[method + '_intersection'] for r in rows) / max(
            sum(r[method + '_pred_area'] + r['gt_area'] for r in rows), 1)
    return out


def summarize(output):
    rows = []
    for path in sorted(output.glob('worker_*.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    rows.sort(key=lambda r: r['id'])
    manifest = json.loads((output / 'manifest.json').read_text())
    expected = {r['id'] for r in manifest['episodes']}
    if {r['id'] for r in rows} != expected or len(rows) != len(expected):
        raise ValueError(f'Incomplete or duplicate results: {len(rows)} vs {len(expected)}')
    datasets, classes = [], []
    for name in dict.fromkeys(r['dataset'] for r in rows):
        group = [r for r in rows if r['dataset'] == name]
        datasets.append(dict(dataset=name, **aggregate(group)))
        for class_name in dict.fromkeys(r['class_name'] for r in group):
            classes.append(dict(dataset=name, class_name=class_name,
                                **aggregate([r for r in group if r['class_name'] == class_name])))
    shots, episodes = [], []
    for row in rows:
        episode = {k: v for k, v in row.items() if not isinstance(v, list)}
        for i, (name, dice, score, weight, teacher) in enumerate(zip(
                row['support_names'], row['dice'], row['scores'], row['weights'], row['teacher_weights'])):
            shots.append(dict(id=row['id'], dataset=row['dataset'], class_name=row['class_name'],
                              query_name=row['query_name'], patient=row['patient'], support_index=i,
                              support_name=name, single_dice=dice, score=score,
                              fusion_weight=weight, teacher_weight=teacher))
        episodes.append(episode)
    write_csv(output / 'episodes.csv', episodes)
    write_csv(output / 'shots.csv', shots)
    write_csv(output / 'dataset_summary.csv', datasets)
    write_csv(output / 'class_summary.csv', classes)
    (output / 'results.json').write_text(json.dumps(dict(datasets=datasets, classes=classes, episodes=rows), indent=2))
    run = json.loads((output / 'run.json').read_text())
    lines = ['# Support-quality audit', '',
             f'Checkpoint epoch {run["epoch"]}; original full-validation macro Dice {run["val_dice"]:.4f}.',
             'CPU FP32, fixed query/support, cached unaugmented SAM embeddings; all model parameters loaded strictly.',
             'Diagnostic subset: patient/area stratified within actual validation draws; not full-validation scores.', '',
             '|Dataset|N|Patients|Mean logits Dice|Weighted Dice|Delta pp|Random single|Selected single|Spread >=10pp|Max weight|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in datasets:
        lines.append(f'|{r["dataset"]}|{r["n"]}|{r["patients"]}|{r["uniform_dice"]:.3f}|{r["weighted_dice"]:.3f}|'
                     f'{r["weighted_minus_uniform"]:+.3f}|{r["random_single_dice"]:.3f}|{r["selected_single_dice"]:.3f}|'
                     f'{r["high_spread_n"]}|{r["max_weight"]:.4f}|')
    lines += ['', '## Definitions and limits', '',
              '- Dice values in this table are per-query means in percent, not the training log class-pooled macro Dice.',
              '- Random single is the exact expected Dice of uniform random support selection (mean of four Dice values).',
              '- Selected single uses the highest learned weight; exact score ties are averaged without position preference.',
              '- Informative queries have max-minus-min single Dice >=1pp; high-spread queries have >=10pp.',
              '- Top1 chance baseline accounts for ties in the best single Dice; pairwise comparisons exclude Dice gaps <1pp.',
              '- Best single is GT-assisted diagnosis, not an upper bound on multi-support fusion.',
              '- Four-shot all-zero Dice queries remain in fusion metrics but are excluded from informative ranking metrics.',
              '- Confidence intervals resample patients, retaining all their query/class rows; single-patient groups have no CI.',
              '- The current validation sampler accesses the first 1000 list indices (wrapping short lists); this audit preserves that coverage.',
              '- Excludes query self-support, empty resized query masks, and missing SAM caches; see coverage.csv.',
              '- Small/rare-class groups with one patient cannot establish generalization or statistical significance.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    make_plot(output, rows)
    print('\n'.join(lines[:12]), flush=True)


def make_plot(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(r['dataset'] for r in rows))
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for col, name in enumerate(names):
        group = [r for r in rows if r['dataset'] == name]
        axes[0, col].scatter([r['dice_spread'] for r in group], [r['max_weight'] for r in group], s=13, alpha=.6)
        axes[0, col].axhline(.25, color='grey', ls='--'); axes[0, col].axvline(10, color='grey', ls=':')
        axes[0, col].set(title=name, xlabel='Single Dice spread (pp)', ylabel='Maximum learned weight')
        axes[1, col].scatter([r['dice_spread'] for r in group], [r['weighted_minus_uniform'] for r in group], s=13, alpha=.6)
        axes[1, col].axhline(0, color='grey', ls='--'); axes[1, col].axvline(10, color='grey', ls=':')
        axes[1, col].set(xlabel='Single Dice spread (pp)', ylabel='Weighted minus mean Dice (pp)')
    fig.suptitle('Fixed query/support audit; CPU FP32; diagnostic subset')
    fig.savefig(output / 'support_quality.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queries-per-class', type=int, default=12)
    parser.add_argument('--validation-ranks', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--feature-cache', type=Path, default=Path(os.environ.get(
        'TBD_FEATS', '/data2/2025Zhutaotao/TBD/cache_features')) / 'eval_unaugmented_v1')
    args = parser.parse_args()
    os.chdir(ROOT)
    if min(args.queries_per_class, args.validation_ranks, args.workers, args.threads) < 1:
        parser.error('Counts must be positive')
    torch.set_num_threads(args.threads)
    if args.summarize_only:
        summarize(args.output); return
    if not args.run_prepared:
        if args.checkpoint is None:
            parser.error('--checkpoint is required when preparing an audit')
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Output directory must be empty')
        args.output.mkdir(parents=True, exist_ok=True)
        stat = args.checkpoint.stat()
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if args.checkpoint.stat().st_mtime_ns != stat.st_mtime_ns:
            raise RuntimeError('Checkpoint changed while loading; retry')
        config = checkpoint['config'].copy()
        if checkpoint_fusion_mode(checkpoint) != 'quality':
            raise ValueError('A learned quality checkpoint is required')
        config['support_fusion'] = 'quality'
        torch.save(checkpoint, args.output / 'checkpoint_snapshot.pt')
        digest = hashlib.sha256((args.output / 'checkpoint_snapshot.pt').read_bytes()).hexdigest()
        run = dict(config=config, epoch=checkpoint['epoch'], val_dice=checkpoint['val_dice'],
                   source=str(args.checkpoint.resolve()), checkpoint_sha256=digest,
                   feature_cache=str(args.feature_cache.resolve()), device='cpu',
                   started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        (args.output / 'run.json').write_text(json.dumps(run, indent=2))
        del checkpoint
        manifest = prepare_manifest(config, args.queries_per_class, args.validation_ranks, args.feature_cache)
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        write_csv(args.output / 'coverage.csv', manifest['coverage'])
        missing = [r for r in manifest['coverage'] if r['selected_queries'] == 0]
        if missing:
            raise ValueError(f'Classes without diagnostic queries: {missing}')
    if args.prepare_only:
        return
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = [executor.submit(worker, i, args.workers, str(args.output.resolve()), args.threads)
                   for i in range(args.workers)]
        for future in as_completed(futures):
            print(f'Finished shard with {future.result()} queries', flush=True)
    summarize(args.output)


if __name__ == '__main__':
    main()
