"""Archive the biased initial BraTS selection and combine its corrected rerun."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnose_small_targets import write_csv
from diagnose_support_quality import summarize


def main():
    output = ROOT / 'analysis/support_quality_20261017'
    replacement = ROOT / 'analysis/support_quality_20261017_brats_selection_fix'
    previous = json.loads((output / 'results.json').read_text())
    new = json.loads((replacement / 'results.json').read_text())
    kept = [r for r in previous['episodes'] if r['dataset'] != 'brats']
    fixed = new['episodes']
    assert len(kept) == 341 and len(fixed) == 36
    assert all(r['dataset'] == 'brats' for r in fixed)
    archive = output / 'initial_selection'; archive.mkdir(exist_ok=False)
    for path in output.iterdir():
        if path.is_file() and path.suffix in ('.csv', '.json', '.jsonl', '.md', '.png'):
            shutil.copy2(path, archive / path.name)
    (archive / 'README.md').write_text(
        'Superseded initial selection. BraTS sampling selected only the smallest target '
        'per patient when patient count exceeded the query budget. The corrected rerun '
        'spreads within-patient area quantiles before inference. Use ../report.md and '
        '../results.json for final findings. BTCV, AMOS and ADAM records are unchanged.\n')
    manifest = json.loads((output / 'manifest.json').read_text())
    fixed_manifest = json.loads((replacement / 'manifest.json').read_text())
    manifest['episodes'] = [r for r in manifest['episodes'] if r['dataset'] != 'brats'] + fixed_manifest['episodes']
    manifest['coverage'] = [r for r in manifest['coverage'] if r['dataset'] != 'brats'] + fixed_manifest['coverage']
    manifest['selection_correction'] = 'When patient count >= query budget, spread area quantiles across selected patients instead of always selecting smallest targets; BraTS rerun only.'
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    write_csv(output / 'coverage.csv', manifest['coverage'])
    rows = sorted(kept + fixed, key=lambda r: r['id'])
    assert len(rows) == 377
    for index in range(4):
        (output / f'worker_{index}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows[index::4]))
    run = json.loads((output / 'run.json').read_text())
    run['selection_correction'] = dict(reason=manifest['selection_correction'],
                                       initial_archive=str(archive), replacement=str(replacement),
                                       unchanged_datasets=['btcv2', 'amos', 'adam'])
    (output / 'run.json').write_text(json.dumps(run, indent=2))
    summarize(output)


if __name__ == '__main__':
    main()
