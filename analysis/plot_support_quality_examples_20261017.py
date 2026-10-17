"""Illustrate two audited failure modes; all Dice values use full images."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/req_sam_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from diagnose_small_targets import load_view, context_box
from diagnose_support_quality import initialize_datasets, load_models, predict, analyze_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'analysis/support_quality_20261017')
    args = parser.parse_args()
    torch.set_num_threads(6)
    run = json.loads((args.root / 'run.json').read_text())
    rows = json.loads((args.root / 'results.json').read_text())['episodes']
    selected = [next(r for r in rows if r['id'] == key)
                for key in ['0324_adam_aneurysm', '0332_adam_treated_aneurysm']]
    checkpoint = torch.load(args.root / 'checkpoint_snapshot.pt', map_location='cpu', weights_only=True)
    model, prompt, decoder = load_models(checkpoint, run['config'])
    del checkpoint
    datasets = initialize_datasets(run['config'])
    folder = args.root / 'examples'; folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 7, figsize=(18, 6), constrained_layout=True)
    metadata = []
    for row_index, episode in enumerate(selected):
        view = load_view(datasets[episode['dataset']], episode)
        logits, scores, weights = predict(model, prompt, decoder, view, episode, Path(run['feature_cache']))
        actual = analyze_prediction(logits, scores, weights, view['target'])
        np.testing.assert_allclose(weights, episode['weights'], atol=2e-5, rtol=1e-4)
        np.testing.assert_allclose(actual['dice'], episode['dice'], atol=.02, rtol=0)
        target = view['target']
        box = context_box(target, 64, 3)
        left, top, right, bottom = box
        image = view['displays'][0][top:bottom, left:right]
        gt = target[top:bottom, left:right]
        mean = logits.mean(0)
        weighted = np.sum(logits * weights[:, None, None], axis=0)
        panels = [None] + [p > 0 for p in logits] + [mean > 0, weighted > 0]
        titles = [episode['class_name'] + '\nQuery + GT']
        titles += [f'Support {i+1}: w={weights[i]:.3f}\nDice={actual["dice"][i]:.2f}' for i in range(4)]
        titles += [f'Mean logits\nDice={actual["uniform_dice"]:.2f}',
                   f'Weighted logits\nDice={actual["weighted_dice"]:.2f}']
        for col, (prediction, title) in enumerate(zip(panels, titles)):
            ax = axes[row_index, col]
            ax.imshow(image)
            if prediction is not None:
                foreground = prediction[top:bottom, left:right]
                rgba = np.zeros((*foreground.shape, 4))
                rgba[foreground] = [1, .15, .1, .35]
                ax.imshow(rgba)
            if gt.any() and not gt.all():
                ax.contour(gt.astype(float), levels=[.5], colors=['lime'], linewidths=1.)
            ax.set_title(title, fontsize=9); ax.axis('off')
        np.savez_compressed(folder / (episode['id'] + '.npz'),
                            logits=logits, target=target, weights=weights, scores=scores)
        metadata.append(dict(id=episode['id'], query_name=episode['query_name'],
                             display_roi=box, metrics=actual))
    fig.suptitle('Green=GT, red=prediction. Display zoom only; inference and Dice use full images.', fontsize=12)
    fig.savefig(folder / 'adam_failure_modes.png', dpi=180)
    plt.close(fig)
    (folder / 'examples.json').write_text(json.dumps(metadata, indent=2))
    print('Saved', folder / 'adam_failure_modes.png', flush=True)


if __name__ == '__main__':
    main()
