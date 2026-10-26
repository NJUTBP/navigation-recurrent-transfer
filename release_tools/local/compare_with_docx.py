#!/usr/bin/env python3
"""Side-by-side sheets: figures embedded in the final Word files vs figures redrawn from the repository.

  python release_tools/local/compare_with_docx.py --docx manuscript.docx --docx si.docx --rendered outputs/figures

The embedded images are listed in document order; each is paired with the redrawn PNG whose name is given in
ORDER below (main text: Figure 1 is author-drawn and only shown). Panels, titles and numbering are judged by eye.
"""
import argparse, re, zipfile
from pathlib import Path
from PIL import Image, ImageDraw

MAIN_ORDER = ['Figure1_external', 'Figure2', 'Figure3', 'Figure4', 'Figure5', 'Figure6', 'Figure7']
SI_ORDER = [f'SupplementaryFigureS{i}' for i in range(1, 13)]


def embedded_images(docx: Path):
    z = zipfile.ZipFile(docx)
    rels = z.read('word/_rels/document.xml.rels').decode('utf8')
    rid2file = dict(re.findall(r'Id="(rId\d+|rIdRev\w+)"[^>]*Target="(media/[^"]+)"', rels))
    rid2file.update({a: b for b, a in re.findall(r'Target="(media/[^"]+)"[^>]*Id="(rId\w+)"', rels)})
    doc = z.read('word/document.xml').decode('utf8')
    order = re.findall(r'r:embed="([^"]+)"', doc)
    return [(rid2file[r], z.read('word/' + rid2file[r])) for r in order if r in rid2file]


def sheet(pairs, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for name, emb_bytes, rendered in pairs:
        from io import BytesIO
        a = Image.open(BytesIO(emb_bytes)).convert('RGB'); a.thumbnail((1400, 1400))
        b = Image.open(rendered).convert('RGB') if rendered and rendered.exists() else Image.new('RGB', a.size, 'white')
        b.thumbnail((1400, 1400))
        w, h = a.width + b.width + 30, max(a.height, b.height) + 40
        c = Image.new('RGB', (w, h), 'white'); c.paste(a, (0, 40)); c.paste(b, (a.width + 30, 40))
        ImageDraw.Draw(c).text((10, 10), f'{name}: left = embedded in Word, right = redrawn', fill='black')
        c.save(out / f'compare_{name}.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--docx', type=Path, action='append', required=True)
    ap.add_argument('--rendered', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('outputs/compare'))
    a = ap.parse_args()
    pairs = []
    for d in a.docx:
        imgs = embedded_images(d)
        is_si = any('Supplementary' in d.name or 'SI' in d.name for _ in [0])
        order = SI_ORDER if is_si else MAIN_ORDER
        if not is_si and len(imgs) > len(order):        # e.g. a graphical abstract precedes Figure 1
            imgs = imgs[len(imgs) - len(order):]
        for name, (_, data) in zip(order, imgs):
            pairs.append((name, data, a.rendered / f'{name}.png'))
    sheet(pairs, a.out)
    print(f'{len(pairs)} comparison sheets written to {a.out}')


if __name__ == '__main__':
    main()
