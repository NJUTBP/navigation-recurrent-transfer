#!/usr/bin/env python3
"""Build the small public code ZIP from a verified release-file manifest.

  python release_tools/local/package_code.py --repo . --out ../code_release.zip
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    subprocess.run([sys.executable, str(repo / 'release_tools/local/verify_release.py'),
                    '--repo', str(repo)], check=True)
    manifest_path = repo / 'data/RELEASE_FILES.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    names = sorted([record['path'] for record in manifest['files']] + ['data/RELEASE_FILES.json'])
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + '.partial')
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in names:
            archive.write(repo / name, 'navigation-recurrent-transfer/' + name)
    with zipfile.ZipFile(partial) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f'ZIP integrity check failed: {bad}')
    partial.replace(out)
    h = hashlib.sha256()
    with out.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    print(json.dumps({'zip': str(out), 'files': len(names), 'bytes': out.stat().st_size,
                      'sha256': h.hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
