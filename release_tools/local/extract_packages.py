#!/usr/bin/env python3
"""Checksum and extract P1-P3 into a short repository path; original archives stay unchanged.

  python release_tools/local/extract_packages.py --packages /path/to/packages --repo .
"""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def sha256(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 << 20), b''):
        h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo = args.repo.resolve()
    sums = repo / 'data' / 'PACKAGE_SHA256SUMS.txt'
    plans = []
    # Preflight ALL paths and checksums before writing any extracted data.
    for line in sums.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        archive = args.packages / name.strip()
        with archive.open('rb') as stream:
            if sha256(stream) != expected:
                raise RuntimeError(f'Archive checksum mismatch: {archive}')
        with tarfile.open(archive, 'r:*') as tar:
            for member in tar:
                relative = PurePosixPath(member.name)
                destination = repo / member.name
                if (relative.is_absolute() or '..' in relative.parts or '\\' in member.name
                        or not destination.resolve().is_relative_to(repo)
                        or not (member.isdir() or member.isfile())):
                    raise RuntimeError(f'Unsupported or unsafe archive path: {member.name}')
                if relative.parts[0] not in ('sources', 'rerun_best_validation'):
                    raise RuntimeError(f'Unexpected data-package root: {member.name}')
                if os.name == 'nt' and len(str(destination)) >= 250:
                    raise RuntimeError('Repository path is too long for portable Windows extraction. '
                                       'Move the repository to a short path, for example C:\\nrt, then retry. '
                                       f'Offending path: {destination}')
        plans.append(archive)
    for archive in plans:
        copied = skipped = 0
        with tarfile.open(archive, 'r:*') as tar:
            for member in tar:
                target = repo / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    with target.open('rb') as current, tar.extractfile(member) as incoming:
                        if target.stat().st_size != member.size or sha256(current) != sha256(incoming):
                            raise RuntimeError(f'Existing file differs; inspect before replacing it: {target}')
                    skipped += 1
                    continue
                # A .partial file is never admitted as a complete extracted checkpoint.
                partial = target.with_name(target.name + '.partial')
                with tar.extractfile(member) as source, partial.open('wb') as destination:
                    shutil.copyfileobj(source, destination, length=8 << 20)
                partial.replace(target)
                copied += 1
        print(f'{archive.name}: copied {copied}, already identical {skipped}', flush=True)


if __name__ == '__main__':
    main()
