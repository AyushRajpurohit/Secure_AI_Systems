"""Scan active first-party Python and a nonexecuted notebook export with Bandit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat
from nbconvert import PythonExporter

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output_dir or ROOT / 'security' / 'sast' / ('scan_' + stamp)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    folders = ['src', 'scripts', 'mnist_adversarial/src',
               'mnist_adversarial/attacks', 'mnist_adversarial/defenses',
               'mnist_adversarial/scripts', 'mnist_adversarial/tests']
    sources = sorted({p for folder in folders
                      for p in (ROOT / folder).rglob('*.py')})
    sources.append(ROOT / 'mnist_adversarial' / 'evaluate_defended.py')
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sources}
    exports = []
    for notebook in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        hashes[str(notebook.relative_to(ROOT))] = hashlib.sha256(notebook.read_bytes()).hexdigest()
        source, _ = PythonExporter().from_notebook_node(nbformat.read(notebook, as_version=4))
        path = output / (notebook.stem + '_export.py')
        path.write_text(source, encoding='utf-8')
        exports.append(path)
    command = [sys.executable, '-m', 'bandit',
               *[str(p) for p in sources + exports],
               '-f', 'json', '-o', str(output / 'bandit.json')]
    result = subprocess.run(command, capture_output=True, text=True, shell=False)
    (output / 'scanner_output.txt').write_text(result.stdout + result.stderr)
    if result.returncode not in (0, 1):
        raise RuntimeError('Bandit execution failed: ' + result.stderr)
    report = json.loads((output / 'bandit.json').read_text())
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'command': command, 'exit_code': result.returncode,
                'source_hashes': hashes,
                'scope': 'active first-party Python, tests, helpers and notebook exports',
                'excluded': ['historical source snapshots and exports', 'third-party code',
                             'datasets', 'weights', 'result data'],
                'limitations': ['IPython shell/magic bodies are not directly audited by Bandit',
                                'run_all.sh requires separate shell review',
                                'raw warnings require manual review']}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    if report.get('errors'):
        raise RuntimeError('Incomplete scan: ' + json.dumps(report['errors']))
    print('Complete scan:', len(sources), 'Python files;', len(exports), 'notebook exports')
    print('Findings:', len(report['results']))
    print('Evidence:', output)


if __name__ == '__main__':
    main()
