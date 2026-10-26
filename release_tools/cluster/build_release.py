#!/usr/bin/env python3
"""Assemble the publication data packages on the cluster, read-only on the project.

  1) dry run (default): resolve every manifest entry, report matches, counts and sizes, write
     <out>/inventory/RELEASE_INVENTORY.{csv,txt}. Nothing is copied.
  2) --execute: copy the selected files into <out>/data_release/ (same relative paths as in the project),
     writing weights-only copies of the pretrained checkpoints (optimizer/scheduler states dropped).
  3) --package: write one tar archive per package into <out>/packages/ plus SIZES.txt and SHA256SUMS.txt.

The project tree is never modified: files are only read; all writes go below --out, which must not be
inside an existing result directory. Re-running --execute skips files that already exist in the output.

Usage (on the cluster login or a compute node):
  python build_release.py --project /path/to/multinav_paper_v1 --out /path/to/multinav_paper_v1/publication_release
  python build_release.py --project ... --out ... --execute
  python build_release.py --project ... --out ... --package
"""
import argparse, csv, fnmatch, hashlib, json, os, shutil, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DROP_KEY_PARTS = ('optim', 'scheduler', 'scaler', 'rng_state', 'grad')


def expand_braces(pattern):
    if '{' not in pattern:
        return [pattern]
    pre, rest = pattern.split('{', 1)
    body, post = rest.split('}', 1)
    return [x for opt in body.split(',') for x in expand_braces(pre + opt + post)]


def excluded(path: Path, rel: Path, man) -> bool:
    if any(fnmatch.fnmatch(path.name, pat) for pat in man['exclude_names']):
        return True
    return any(fnmatch.fnmatch(part, pat) for part in rel.parts[:-1] for pat in man['exclude_dirs'])


def resolve(entry, project: Path, man):
    files = []
    for g in entry['globs']:
        for pat in expand_braces(g):
            if entry['mode'] == 'copy_tree_small':
                root = project / pat
                if root.is_dir():
                    for p in root.rglob('*'):
                        if p.is_file() and p.suffix in entry.get('suffixes', []):
                            rel = p.relative_to(project)
                            if not excluded(p, rel, man) and '/checkpoints/' not in rel.as_posix():
                                files.append(p)
            else:
                for p in project.glob(pat):
                    if p.is_file() and not excluded(p, p.relative_to(project), man):
                        files.append(p)
    uniq = sorted(set(files))
    cap = entry.get('max_mb')
    big = [p for p in uniq if cap and p.stat().st_size > cap * 1e6]
    return [p for p in uniq if p not in big], big


def human(n):
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or u == 'TB':
            return f'{n:.1f} {u}'
        n /= 1024


def dry_run(project, out, man):
    inv, lines, totals = [], [], {}
    for e in man['entries']:
        files, big = resolve(e, project, man)
        size = sum(p.stat().st_size for p in files)
        totals[e['package']] = totals.get(e['package'], 0) + size
        exp = e.get('expect')
        flag = '' if exp is None or exp == len(files) else f'  <-- expected {exp}'
        lines.append(f"{e['name']:36s} {len(files):6d} files {human(size):>10s}{flag}")
        if big:
            lines.append(f"{'':36s} skipped (> {e['max_mb']} MB): " + ', '.join(str(p.relative_to(project)) for p in big[:5]))
        if not files:
            lines.append(f"{'':36s} NO MATCH for: {e['globs'][0]}")
        for p in files:
            inv.append({'entry': e['name'], 'package': e['package'], 'mode': e['mode'],
                        'relative_path': p.relative_to(project).as_posix(), 'bytes': p.stat().st_size})
    lines.append('')
    for k, v in totals.items():
        lines.append(f'{k:36s} {human(v):>10s} (before weights-only conversion)')
    (out / 'inventory').mkdir(parents=True, exist_ok=True)
    with open(out / 'inventory' / 'RELEASE_INVENTORY.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(inv[0].keys()) if inv else ['entry'])
        w.writeheader(); w.writerows(inv)
    (out / 'inventory' / 'RELEASE_INVENTORY.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"\nInventory written to {out / 'inventory'}. Review it, then run with --execute.")


def core_sha256(state):
    h = hashlib.sha256()
    for key in ('grid_rnn.cell.in_proj.weight', 'grid_rnn.cell.h_proj.weight'):
        t = state[key].detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(t.shape).encode()); h.update(t.tobytes())
    return h.hexdigest()


def slim_checkpoint(src: Path, dst: Path, project: Path):
    import torch
    payload = torch.load(src, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or 'model_state' not in payload:
        raise RuntimeError(f'{src}: no model_state; copy manually after inspection')
    keep = {k: v for k, v in payload.items() if not any(part in k.lower() for part in DROP_KEY_PARTS)}
    dropped = sorted(set(payload) - set(keep))
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(keep, dst)
    check = torch.load(dst, map_location='cpu', weights_only=False)
    same = core_sha256(check['model_state']) == core_sha256(payload['model_state'])
    rec = {}
    for ident in ('source_identity.json', 'control_identity.json'):
        p = src.parents[2] / ident
        if p.is_file():
            rec[ident] = json.loads(p.read_text()).get('core_sha256')
    return {'source': str(src.relative_to(project)), 'dropped_keys': dropped, 'core_identical_after_save': same,
            'core_sha256': core_sha256(check['model_state']), 'recorded_core_sha256': rec,
            'bytes_before': src.stat().st_size, 'bytes_after': dst.stat().st_size}


def execute(project, out, man):
    inv_path = out / 'inventory' / 'RELEASE_INVENTORY.csv'
    if not inv_path.is_file():
        sys.exit('Run the dry run first and review RELEASE_INVENTORY.csv')
    rows = list(csv.DictReader(open(inv_path)))
    dest_root = out / 'data_release'
    log = []
    for r in rows:
        src = project / r['relative_path']
        dst = dest_root / r['package'] / r['relative_path']
        if dst.exists():
            continue
        if r['mode'] == 'slim_checkpoint':
            log.append(slim_checkpoint(src, dst, project))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if r['mode'] == 'target_checkpoint':
                import torch
                mode = torch.load(dst, map_location='cpu', weights_only=False).get('checkpoint_storage_mode')
                if mode != 'frozen_core_omitted_reconstruct_from_origin_and_seed':
                    log.append({'warning': 'target checkpoint stores its core', 'file': r['relative_path'], 'mode': mode})
    (out / 'inventory' / 'EXECUTE_LOG.json').write_text(json.dumps(log, indent=1))
    bad = [x for x in log if x.get('core_identical_after_save') is False]
    print(f'copied into {dest_root}; checkpoint conversions: {sum(1 for x in log if "dropped_keys" in x)}; problems: {len(bad)}')


def package(out):
    dest_root = out / 'data_release'
    pk = out / 'packages'; pk.mkdir(parents=True, exist_ok=True)
    sizes, sums = [], []
    for d in sorted(p for p in dest_root.iterdir() if p.is_dir()):
        compress = d.name.startswith('P3')          # weights compress poorly; tables compress well
        arc = pk / (d.name + ('.tar.gz' if compress else '.tar'))
        if not arc.exists():
            with tarfile.open(arc, 'w:gz' if compress else 'w') as tar:
                for child in sorted(d.iterdir()):          # archive paths = project-relative paths
                    tar.add(child, arcname=child.name)
        h = hashlib.sha256()
        with open(arc, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 24), b''):
                h.update(chunk)
        sizes.append(f'{arc.name:45s} {human(arc.stat().st_size)}')
        sums.append(f'{h.hexdigest()}  {arc.name}')
    (pk / 'SIZES.txt').write_text('\n'.join(sizes) + '\n')
    (pk / 'SHA256SUMS.txt').write_text('\n'.join(sums) + '\n')
    print('\n'.join(sizes))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, default=HERE / 'release_manifest.json')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--package', action='store_true')
    a = ap.parse_args()
    project, out = a.project.resolve(), a.out.resolve()
    if not project.is_dir():
        sys.exit(f'project not found: {project}')
    for protected in ('rerun_best_validation', 'sources', 'code', 'tools'):
        if (project / protected).resolve() in [out, *out.parents]:
            sys.exit(f'--out must not be inside {protected}/')
    man = json.loads(a.manifest.read_text())
    out.mkdir(parents=True, exist_ok=True)
    if a.package:
        package(out)
    elif a.execute:
        execute(project, out, man)
    else:
        dry_run(project, out, man)


if __name__ == '__main__':
    main()
