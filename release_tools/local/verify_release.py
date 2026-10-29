#!/usr/bin/env python3
"""Verify the portable code/figure-data release and optionally the three data archives.

  python release_tools/local/verify_release.py --repo .
  python release_tools/local/verify_release.py --repo . --packages /path/to/packages
  python release_tools/local/verify_release.py --repo . --write-manifest  # after intentional release edits
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

PUBLIC_DIRS = ('checkpoints', 'code', 'configs', 'data', 'docs', 'extensions', 'figures',
               'lsf_best_validation', 'release_tools', 'tests', 'tools', 'topology', '.github')
SKIP_PARTS = {'__pycache__', '.git', '.pytest_cache', '.ipynb_checkpoints'}
MANIFEST = 'data/RELEASE_FILES.json'
FIGURE_ARCHIVES = {
    'data/aggregate_full/figure_source_data/Figure2/rate_maps_top10_50x50.zip',
    'data/aggregate_full/figure_source_data/Figure2/trajectory_candidates_20step.zip',
}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def public_files(repo):
    files = [p for p in repo.iterdir() if p.is_file() and
             (p.suffix in ('.md', '.txt', '.json', '.cff') or p.name in ('LICENSE', '.gitignore', '.gitattributes'))]
    for name in PUBLIC_DIRS:
        base = repo / name
        if base.is_dir():
            files.extend(p for p in base.rglob('*') if p.is_file())
    return sorted(p for p in set(files) if not (set(p.relative_to(repo).parts) & SKIP_PARTS)
                  and p.relative_to(repo).as_posix() != MANIFEST)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--packages', type=Path)
    parser.add_argument('--write-manifest', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    path = repo / MANIFEST
    if args.write_manifest:
        records = []
        for p in public_files(repo):
            relative = p.relative_to(repo).as_posix()
            if not p.resolve().is_relative_to(repo):
                raise RuntimeError(f'Public file escapes the repository: {p}')
            if (p.stat().st_size > 50_000_000 or p.suffix in ('.pt', '.pth', '.ckpt', '.tar', '.gz')
                    or (p.suffix == '.zip' and relative not in FIGURE_ARCHIVES)):
                raise RuntimeError(f'Model/archive/oversized file in code release: {p}')
            records.append({'path': relative, 'bytes': p.stat().st_size, 'sha256': sha256(p)})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'schema': 'nrt_release_files_v1', 'files': records}, indent=2) + '\n', encoding='utf-8')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    errors, total, python_files = [], 0, 0
    expected = {r['path'] for r in manifest['files']}
    actual = {p.relative_to(repo).as_posix() for p in public_files(repo)}
    for name in sorted(actual - expected):
        errors.append(f'UNLISTED {name}')
    for record in manifest['files']:
        p = repo / record['path']
        if not p.resolve().is_relative_to(repo) or not p.is_file():
            errors.append(f'MISSING/UNSAFE {record["path"]}')
            continue
        if p.stat().st_size != record['bytes'] or sha256(p) != record['sha256']:
            errors.append(f'CHANGED {record["path"]}')
        if p.suffix == '.py':
            python_files += 1
            try:
                ast.parse(p.read_text(encoding='utf-8-sig'), filename=record['path'])
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f'PYTHON {record["path"]}: {exc}')
        total += p.stat().st_size
    packages_checked = 0
    if args.packages:
        for line in (repo / 'data' / 'PACKAGE_SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            expected_sha, name = line.split(None, 1)
            p = args.packages / name.strip()
            if not p.is_file() or sha256(p) != expected_sha:
                errors.append(f'ARCHIVE {name.strip()}')
            packages_checked += 1
    for error in errors:
        print(error, file=sys.stderr)
    print(json.dumps({'status': 'failed' if errors else 'passed', 'public_files': len(expected),
                      'python_files': python_files, 'public_bytes': total,
                      'archives_checked': packages_checked, 'errors': len(errors)}, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
