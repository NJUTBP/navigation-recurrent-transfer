#!/usr/bin/env python3
"""Build the small public code ZIP from working files tracked by Git.

  python release_tools/local/package_code.py --repo . --out ../code_release.zip
"""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    tracked = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z'])
    names = sorted(name for name in tracked.decode('utf-8').split('\0')
                   if name and (repo / name).is_file())
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
    print(json.dumps({'zip': str(out), 'files': len(names), 'bytes': out.stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
