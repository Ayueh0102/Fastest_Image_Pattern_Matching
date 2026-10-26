"""Round 50 B1: freeze TDCM's synthetic COCO pairs with the authors' own generator.

Runs inside the TDCM environment (``python -I``) and imports the repository's
``utils.get_imgR.rotate_crop`` and ``data.filterData.filter_data``. The sampling
loop copies ``utils.dataSet.CoCo_Dataset.__getitem__`` line for line, but keeps
the raw uint8 arrays instead of normalised tensors, and the seeding copies
``test_single.py`` (random, numpy and torch seeded with 42 before the loop). So
sample i here is the pair test_single.py evaluates as sample i.

Lists (one CSV each, rows "file,x,y,w,h"):
  S2    the published data/S2/val.csv
  S1_5  filter_data(..., scale_range=(0.8, 1.5)) on instances_val2017.json
  S2_5  filter_data(..., scale_range=(0.4, 2.5))
  S1    operational definition of round 50 (the paper defines S1 only as
        "rotation only (scale fixed at 1)"): filter_data with scale_range
        (34/36, 38/36). Its 129 rows are split by image - every fifth image in
        first-appearance order is development - and each split's rows are
        repeated in order until the sample count is reached, so every sample
        is a fresh rotate_crop draw and no object is in both splits.

Per sample the output holds the 224x224 search image, the template crop at its
native size and the same crop resized to 36x36 by torchvision (what
test_single.py feeds the model). Truth stays in truth.json, apart from images:
centre = mean of the rotated box corners, scale_x = w/36, scale_y = h/36, angle
as drawn by rotate_crop (counter-clockwise on screen, cv2.getRotationMatrix2D).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with open(path) as f:
        return [(r[0], tuple(int(v) for v in r[1:5])) for r in csv.reader(f) if r]


def split_by_image(rows, every=5):
    order = []
    for name, _ in rows:
        if name not in order:
            order.append(name)
    dev_images = {name for k, name in enumerate(order) if k % every == every - 1}
    return [r for r in rows if r[0] in dev_images], [r for r in rows if r[0] not in dev_images]


def repeated(rows, count):
    out = []
    while len(out) < count:
        out.extend(rows)
    return out


def generate(rows, count, seed, images_dir, out_dir, rotate_crop, torch, transforms, cv2, np, Image):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resize36 = transforms.Resize((36, 36))
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = []
    for i in range(min(len(rows), count)):
        idx = i
        while True:  # CoCo_Dataset.__getitem__
            name, bbox = rows[idx]
            image = cv2.imread(str(images_dir / name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x, y, w, h = [round(v) for v in bbox]
            template = image[y:y + h, x:x + w]
            imgR, corners, angle = rotate_crop(image, bbox)
            if imgR is None:
                idx = random.randint(0, len(rows) - 1)
                continue
            break
        scale_y, scale_x = h / 36, w / 36
        center = corners.mean(axis=0)
        stem = f'{i:04d}'
        cv2.imwrite(str(out_dir / f'{stem}_search.png'), cv2.cvtColor(imgR, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f'{stem}_template.png'), cv2.cvtColor(np.ascontiguousarray(template), cv2.COLOR_RGB2BGR))
        t36 = np.asarray(resize36(Image.fromarray(np.ascontiguousarray(template))))
        cv2.imwrite(str(out_dir / f'{stem}_template36.png'), cv2.cvtColor(t36, cv2.COLOR_RGB2BGR))
        truth.append(dict(index=i, row=idx, retried=idx != i, name=name, bbox=[x, y, w, h], angle=float(angle),
                          cx=float(center[0]), cy=float(center[1]), scale_x=scale_x, scale_y=scale_y,
                          corners=[[float(a), float(b)] for a, b in corners], search_shape=list(imgR.shape)))
    (out_dir / 'truth.json').write_text(json.dumps(truth, indent=1) + '\n')
    return truth


def check_against_log(truth, log_path):
    """Compare with the authors' res/result-single.txt (S2, first 1000 samples)."""
    import re
    text = Path(log_path).read_text(encoding='utf-8')
    blocks = re.split(r'\n(?=样本 )', text)
    compared = mismatched = 0
    first_mismatch = None
    for b in blocks:
        m = re.match(r'样本 (\d+) -- (\S+)', b)
        if not m:
            continue
        i, name = int(m.group(1)), m.group(2)
        c = re.search(r'center: \(([-\d.]+),([-\d.]+)\)', b)
        a = re.search(r'angle: ([-\d.]+)°', b)
        sx = re.search(r'scale_x: ([-\d.]+)', b)
        sy = re.search(r'scale_y: ([-\d.]+)', b)
        t = truth[i]
        ok = (t['name'] == name and abs(t['cx'] - float(c.group(1))) <= .006 and abs(t['cy'] - float(c.group(2))) <= .006
              and abs(t['angle'] - float(a.group(1))) <= .006 and abs(t['scale_x'] - float(sx.group(1))) <= .0006
              and abs(t['scale_y'] - float(sy.group(1))) <= .0006)
        compared += 1
        mismatched += not ok
        if not ok and first_mismatch is None:
            first_mismatch = dict(index=i, log=b.splitlines()[:6], ours={k: t[k] for k in ('name', 'cx', 'cy', 'angle', 'scale_x', 'scale_y')})
    return dict(compared=compared, mismatched=mismatched, first_mismatch=first_mismatch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--coco-images', type=Path, required=True)
    p.add_argument('--coco-annotations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--settings', nargs='+', default=['S1', 'S1_5', 'S2', 'S2_5'])
    p.add_argument('--splits', nargs='+', default=['eval', 'dev'])
    p.add_argument('--count', type=int, default=1000)
    p.add_argument('--dev-count', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dev-seed', type=int, default=4242)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import cv2
    import numpy as np
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from PIL import Image
    from utils.get_imgR import rotate_crop
    from data.filterData import filter_data

    args.output.mkdir(parents=True, exist_ok=True)
    lists = args.output / 'lists'
    lists.mkdir(exist_ok=True)
    ranges = dict(S1=(34 / 36, 38 / 36), S1_5=(0.8, 1.5), S2_5=(0.4, 2.5))
    record = dict(numpy=np.__version__, torch=torch.__version__, torchvision=torchvision.__version__, opencv=cv2.__version__,
                  seed=args.seed, dev_seed=args.dev_seed, settings={})
    for setting in args.settings:
        if setting == 'S2':
            path = args.repo / 'data/S2/val.csv'
        else:
            path = lists / f'{setting}.csv'
            filter_data(str(args.coco_annotations), str(path), tShape=(36, 36), scale_range=ranges[setting])
        rows = read_rows(path)
        info = dict(list=str(path), list_sha256=sha256(path), rows=len(rows), images=len({r[0] for r in rows}))
        plan = {}
        if setting == 'S1':
            dev_rows, eval_rows = split_by_image(rows)
            info.update(dev_rows=len(dev_rows), eval_rows=len(eval_rows), dev_images=len({r[0] for r in dev_rows}),
                        eval_images=len({r[0] for r in eval_rows}))
            plan = dict(eval=(repeated(eval_rows, args.count), args.count, args.seed),
                        dev=(repeated(dev_rows, args.dev_count), args.dev_count, args.dev_seed))
        else:
            plan = dict(eval=(rows, args.count, args.seed))
        for split, (split_rows, count, seed) in plan.items():
            if split not in args.splits:
                continue
            out_dir = args.output / f'{setting}-{split}'
            truth = generate(split_rows, count, seed, args.coco_images, out_dir, rotate_crop, torch, transforms, cv2, np, Image)
            info[split] = dict(samples=len(truth), retried=sum(t['retried'] for t in truth), seed=seed,
                               truth_sha256=sha256(out_dir / 'truth.json'))
            if setting == 'S2' and split == 'eval':
                info['author_log_check'] = check_against_log(truth, args.repo / 'res/result-single.txt')
        record['settings'][setting] = info
        print(setting, {k: v for k, v in info.items() if k not in ('list',)}, flush=True)
    (args.output / 'fixtures.json').write_text(json.dumps(record, indent=1) + '\n')


if __name__ == '__main__':
    main()
