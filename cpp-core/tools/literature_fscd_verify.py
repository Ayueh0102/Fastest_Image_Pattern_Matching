"""Round 50: verify the third-party FSCD-147 mirror against the official FSC-147 release.

The Counting-DETR links are dead, so the FSCD-147 boxes come from an unofficial
Hugging Face mirror. This script, run with ``python -I`` from python-port/venv,
checks the mirror before any method sees it and writes a JSON record:

* every mirror image is byte-identical to the official FSC-147 Google Drive zip;
* every copy of the FSC-147 annotation and split files equals the author's
  GitHub blobs (compared by SHA-256);
* instances_val/test.json cover exactly the val/test split images, with image
  sizes equal to the files, boxes inside the image, and per-image box counts
  compared with the FSC-147 point counts (Counting-DETR: an object gets a box
  only if it has a dot);
* split sizes and the class count against the Counting-DETR paper's Table 1
  (147 classes; 6,135 images = 3,659 train / 1,286 val / 1,190 test).
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image

PAPER = dict(classes=147, images=6135, train=3659, val=1286, test=1190)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def instances_report(path, split_names, annotation, images_dir):
    d = json.loads(path.read_text())
    images = {im['id']: im for im in d['images']}
    names = Counter(im['file_name'] for im in d['images'])
    boxes = Counter()
    outside = size_mismatch = degenerate = 0
    sizes = {}
    for im in d['images']:
        w, h = Image.open(images_dir / im['file_name']).size
        sizes[im['id']] = (w, h)
        size_mismatch += (w, h) != (im['width'], im['height'])
    for a in d['annotations']:
        x, y, bw, bh = a['bbox']
        w, h = sizes[a['image_id']]
        boxes[images[a['image_id']]['file_name']] += 1
        outside += x < -0.5 or y < -0.5 or x + bw > w + 0.5 or y + bh > h + 0.5
        degenerate += bw <= 0 or bh <= 0
    points = {n: len(annotation[n]['points']) for n in split_names}
    equal = sum(boxes[n] == points[n] for n in split_names)
    diffs = sorted(((boxes[n] - points[n], n) for n in split_names if boxes[n] != points[n]), key=lambda t: -abs(t[0]))
    return dict(file=path.name, sha256=sha256(path), bytes=path.stat().st_size, images=len(d['images']),
                duplicate_file_names=sum(c - 1 for c in names.values() if c > 1),
                images_equal_split=set(names) == set(split_names), annotations=len(d['annotations']),
                categories=[c.get('name') for c in d.get('categories', [])],
                image_size_mismatches=size_mismatch, boxes_outside_image=outside, degenerate_boxes=degenerate,
                split_points=sum(points.values()), images_box_count_equals_points=equal,
                images_box_count_differs=len(diffs), largest_differences=[dict(image=n, boxes_minus_points=v) for v, n in diffs[:10]],
                total_boxes_minus_points=len(d['annotations']) - sum(points.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mirror', type=Path, required=True, help='extracted FSC147/ directory of the mirror zip')
    p.add_argument('--official-images', type=Path, required=True, help='official images_384_VarV2 directory')
    p.add_argument('--official-annotations', type=Path, required=True, help='directory with the GitHub annotation files')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    mirror_images = args.mirror / 'images_384_VarV2'
    official = {f.name: f for f in args.official_images.iterdir() if f.suffix == '.jpg'}
    mine = {f.name: f for f in mirror_images.iterdir() if f.suffix == '.jpg'}
    differing = [n for n in sorted(set(official) & set(mine)) if sha256(official[n]) != sha256(mine[n])]
    annotation_files = {}
    for name in ('annotation_FSC147_384.json', 'Train_Test_Val_FSC_147.json'):
        ref = sha256(args.official_annotations / name)
        copies = sorted(args.mirror.rglob(name))
        annotation_files[name] = dict(official_sha256=ref, copies=[dict(path=str(c.relative_to(args.mirror)), equal=sha256(c) == ref)
                                                                    for c in copies])
    annotation = json.loads((args.official_annotations / 'annotation_FSC147_384.json').read_text())
    splits = json.loads((args.official_annotations / 'Train_Test_Val_FSC_147.json').read_text())
    classes = {}
    for line in (args.official_annotations / 'ImageClasses_FSC147.txt').read_text().splitlines():
        if line.strip():
            name, cls = line.split('\t', 1) if '\t' in line else line.split(None, 1)
            classes[name.strip()] = cls.strip()
    split_classes = {k: len({classes[n] for n in splits[k]}) for k in ('train', 'val', 'test')}
    record = dict(
        mirror_images=len(mine), official_images=len(official),
        images_only_in_mirror=sorted(set(mine) - set(official)), images_only_in_official=sorted(set(official) - set(mine)),
        images_byte_identical=len(set(official) & set(mine)) - len(differing), images_differing=differing,
        annotation_files=annotation_files,
        splits={k: len(v) for k, v in splits.items()}, split_classes=split_classes,
        classes_total=len(set(classes[n] for k in ('train', 'val', 'test') for n in splits[k])),
        paper_table1=PAPER,
        instances={k: instances_report(args.mirror / 'annotations' / f'instances_{k}.json', splits[k], annotation, mirror_images)
                   for k in ('val', 'test')},
        duplicate_instances_files={k: [dict(path=str(c.relative_to(args.mirror)), sha256=sha256(c)) for c in sorted(args.mirror.rglob(f'instances_{k}.json'))]
                                   for k in ('val', 'test')})
    record['checks'] = dict(
        images_identical=not differing and not record['images_only_in_mirror'] and not record['images_only_in_official'],
        annotation_copies_identical=all(c['equal'] for v in annotation_files.values() for c in v['copies']),
        split_sizes_match_paper=all(record['splits'][k] == PAPER[k] for k in ('train', 'val', 'test')),
        classes_match_paper=record['classes_total'] == PAPER['classes'],
        instances_cover_splits=all(v['images_equal_split'] for v in record['instances'].values()),
        instances_geometry_valid=all(v['image_size_mismatches'] == 0 and v['boxes_outside_image'] == 0 and v['degenerate_boxes'] == 0
                                     for v in record['instances'].values()),
        duplicate_instances_identical=all(len({c['sha256'] for c in v}) == 1 for v in record['duplicate_instances_files'].values()))
    args.output.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record['checks'], indent=1))
    for k, v in record['instances'].items():
        print(k, {a: v[a] for a in ('images', 'annotations', 'split_points', 'images_box_count_equals_points',
                                    'images_box_count_differs', 'total_boxes_minus_points', 'categories')})


if __name__ == '__main__':
    main()
