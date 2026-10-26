"""Round 50 B2 gate: does GeCo2 with its official weights reproduce its paper on the FSCD-147 mirror?

Runs in the GeCo2 environment (``python -I``; pycocotools). Reads the GeCo2
worker's batch output (one JSON line per test image, boxes x0, y0, x1, y1 in
source pixels) and scores it the way the authors do (eval_bboxes.py with
detectron2's COCOeval_opt, which equals pycocotools' COCOeval): predictions
converted to xywh and truncated with int() as inference.py writes them,
maxDets 10000, AP over IoU 0.50:0.95 and AP50, area 'all', single class.

Gate fixed in the round-50 protocol: |AP - 46.81| <= 2 and |AP50 - 75.89| <= 2
percentage points against the GeCo2 paper's FSCD147 test row (Table 1). On the
M4 Pro MPS a miss stops the GeCo2 accuracy work and is reported (the gap may
come from MPS numerics or from the mirror); only a CUDA miss as well retires AP50.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

PAPER = dict(ap=46.81, ap50=75.89, mae=7.64, rmse=39.39)
TOLERANCE = 2.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True, help='GeCo2 worker --batch output (.jsonl)')
    p.add_argument('--instances', type=Path, required=True, help='FSCD-147 instances_test.json')
    p.add_argument('--points', type=Path, required=True, help='annotation_FSC147_384.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(args.instances))
    ids = {im['file_name']: im['id'] for im in gt.dataset['images']}
    points = json.loads(args.points.read_text())
    results, counts, header, footer = [], [], None, None
    for line in args.predictions.read_text().splitlines():
        d = json.loads(line)
        if 'header' in d:
            header = d['header']
            continue
        if 'footer' in d:
            footer = d['footer']
            continue
        name = d['id']
        for b in d['detections']:
            w, h = b['x1'] - b['x0'], b['y1'] - b['y0']
            results.append(dict(image_id=ids[name], category_id=1, score=float(b['score']),
                                bbox=[int(b['x0']), int(b['y0']), int(w), int(h)]))
        counts.append((name, len(d['detections'])))
    seen = {n for n, _ in counts}
    with contextlib.redirect_stdout(io.StringIO()) as log:
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.maxDets = [10000, 10000, 10000]
        ev.params.imgIds = sorted(ids[n] for n in seen)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    # pycocotools' summarize() hard-codes maxDets=100 for AP@[.5:.95]; the authors override it (COCOevalMaxDets).
    # precision: [iou thresholds, recall points, classes, area ranges, maxDets]; area 'all' = 0, maxDets 10000 = last.
    precision = ev.eval['precision'][:, :, 0, 0, -1]

    def mean_valid(a):
        a = a[a > -1]
        return float(a.mean() * 100) if a.size else float('nan')
    ap = mean_valid(precision)
    ap50 = mean_valid(precision[int(np.argmin(np.abs(ev.params.iouThrs - .5)))])
    gt_boxes = {n: len(gt.getAnnIds(imgIds=[ids[n]])) for n in seen}
    err_points = [c - len(points[n]['points']) for n, c in counts]
    err_boxes = [c - gt_boxes[n] for n, c in counts]
    record = dict(images=len(seen), expected_images=len(ids), predictions=len(results), ap=ap, ap50=ap50, paper=PAPER,
                  gap=dict(ap=ap - PAPER['ap'], ap50=ap50 - PAPER['ap50']), tolerance=TOLERANCE,
                  gate_pass=len(seen) == len(ids) and abs(ap - PAPER['ap']) <= TOLERANCE and abs(ap50 - PAPER['ap50']) <= TOLERANCE,
                  count_vs_points=dict(mae=float(np.mean(np.abs(err_points))), rmse=float(np.sqrt(np.mean(np.square(err_points))))),
                  count_vs_boxes=dict(mae=float(np.mean(np.abs(err_boxes))), rmse=float(np.sqrt(np.mean(np.square(err_boxes))))),
                  device=(header or {}).get('device'), deform_attn=(header or {}).get('deform_attn'),
                  footer=footer, cocoeval_summary=log.getvalue().splitlines()[-12:])
    args.output.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({k: record[k] for k in ('images', 'ap', 'ap50', 'gap', 'gate_pass', 'count_vs_points', 'count_vs_boxes')}, indent=1))


if __name__ == '__main__':
    main()
