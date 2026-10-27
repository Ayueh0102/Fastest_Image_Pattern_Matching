"""Round 50: run PoseMatch-TDCM with its official weights, in the official or the adapted mode.

Runs inside the TDCM environment (``python -I``), never in python-port/venv. The
repository and weights are passed as paths; nothing is downloaded.

Pipelines (both copy the authors' code):

* ``test_single`` - the evaluation script ``test_single.py`` that produced the
  published ``res/result-single.txt``: RGB input with ImageNet normalisation, the
  template resized to 36x36 by torchvision, the arg-max peak, ``get_center`` over
  score > 0.5, sign/cos angle, the two scale heads, then ``refine_angle_bisection``
  and ``refine_scale_x/y`` (search_range 0.15, step 0.015) on the script's own
  grey conversion (denormalised with *0.5+0.5, COLOR_BGR2GRAY on RGB);
* ``test_py`` - the README usage example ``test.py``: BGR input, ``cv2.resize``
  of the template, ImageNet denormalisation, refinement on the three-channel
  images and a final ``refine_angle`` (initial_range 5).

Modes:

* official - one target per call (the arg-max), search image at native size (the
  evaluation resizes its 224x224 crops to 224x224, a no-op there);
* ``--adapt`` - labelled TDCM-adapted, parameters fixed in the round-50 protocol:
  image and template are resized by k = 36 / max(template w, h) (INTER_AREA when
  k < 1, INTER_LINEAR otherwise) so the template enters at 36 px; every 3x3 local
  maximum of the score map with score >= --peak-threshold (fixed on development
  scenes before the protocol freeze; get_center keeps its official 0.5 support),
  at most 2 x --limit of them by score, becomes a target refined like the single
  target; poses are mapped back by 1/k and suppressed like the fpm NMS (rotated
  intersection over the smaller area > --overlap) down to --limit.

Timing: model loading is reported once; each call's time covers preprocessing,
the forward pass, peak extraction and refinement, never image decoding.
``--warmup`` calls are run first and discarded; ``--repeats`` calls are timed and
their outputs must be identical. ``--batch`` takes a JSON list of
{"id", "image", "template"} and writes one JSON line per item (one call each,
after one discarded warm-up call on the first item).
"""
import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time

TH, TW = 36, 36


def load(args):
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from model import Model
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    t0 = time.perf_counter()
    model = Model(None).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    return model, device, (time.perf_counter() - t0) * 1e3


def read_pair(args, image_path, template_path):
    import cv2
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if image is None or template is None:
        raise ValueError(f'cannot decode {image_path} or {template_path}')
    return image, template


def rotated_overlap(a, b):
    import cv2
    ra = ((a['cx'], a['cy']), (a['width'], a['height']), -a['angle_deg'])
    rb = ((b['cx'], b['cy']), (b['width'], b['height']), -b['angle_deg'])
    kind, points = cv2.rotatedRectangleIntersection(ra, rb)
    if kind == cv2.INTERSECT_NONE or points is None:
        return 0.
    return cv2.contourArea(cv2.convexHull(points)) / min(a['width'] * a['height'], b['width'] * b['height'])


class Runner:
    def __init__(self, args, model, device):
        import cv2
        import torch
        import torchvision.transforms as transforms
        from utils import get_center
        from utils.refine import refine_angle, refine_angle_bisection, refine_scale_x, refine_scale_y
        self.args, self.model, self.device = args, model, device
        self.cv2, self.torch, self.get_center = cv2, torch, get_center
        self.refine = (refine_angle, refine_angle_bisection, refine_scale_x, refine_scale_y)
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        normalise = transforms.Normalize(mean=self.mean, std=self.std)
        self.to_image = transforms.Compose([transforms.ToTensor(), normalise])
        self.to_template_single = transforms.Compose([transforms.Resize((TH, TW)), transforms.ToTensor(), normalise])

    def sync(self):
        if self.device.type == 'mps':
            self.torch.mps.synchronize()

    def tensors(self, image, template):
        from PIL import Image
        cv2, args = self.cv2, self.args
        if args.pipeline == 'test_single':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
            t = self.to_template_single(Image.fromarray(template))
        else:
            t = self.to_image(Image.fromarray(cv2.resize(template, (TW, TH))))
        x = self.to_image(Image.fromarray(image))
        return x.unsqueeze(0).to(self.device), t.unsqueeze(0).to(self.device)

    def refine_images(self, x, t):
        cv2, torch = self.cv2, self.torch
        if self.args.pipeline == 'test_single':
            search = cv2.cvtColor((x[0].cpu() * 0.5 + 0.5).permute(1, 2, 0).numpy(), cv2.COLOR_BGR2GRAY)
            templ = cv2.cvtColor((t[0].cpu() * 0.5 + 0.5).permute(1, 2, 0).numpy(), cv2.COLOR_BGR2GRAY)
        else:
            m, s = torch.tensor(self.mean).view(3, 1, 1), torch.tensor(self.std).view(3, 1, 1)
            search = (x[0].cpu() * s + m).permute(1, 2, 0).numpy()
            templ = (t[0].cpu() * s + m).permute(1, 2, 0).numpy()
        return search, templ

    def pose_at(self, maps, support, py, px, search, templ):
        torch = self.torch
        score, sign_map, cos_map, sx_map, sy_map = maps
        cy, cx = self.get_center((py, px), support)
        sign = 1 if sign_map[0, 0, py, px] > 0.5 else -1
        angle = sign * (torch.arccos(cos_map[0, 0, py, px].clamp(-1, 1)) * 180 / math.pi).item()
        sx, sy = sx_map[0, 0, py, px].item(), sy_map[0, 0, py, px].item()
        refine_angle, refine_angle_bisection, refine_scale_x, refine_scale_y = self.refine
        if self.args.pipeline == 'test_single':
            r, r_score = refine_angle_bisection(search, templ, cx, cy, sx, sy, angle)
            rsx, _ = refine_scale_x(search, templ, cx, cy, sx, sy, r, search_range=0.15, step_size=0.015)
            rsy, _ = refine_scale_y(search, templ, cx, cy, rsx, sy, r, search_range=0.15, step_size=0.015)
        else:
            r, r_score = refine_angle_bisection(search, templ, cx, cy, sx, sy, angle, initial_range=20)
            rsx, _ = refine_scale_x(search, templ, cx, cy, sx, sy, r)
            rsy, _ = refine_scale_y(search, templ, cx, cy, rsx, sy, r)
            r, r_score = refine_angle(search, templ, cx, cy, rsx, rsy, r, initial_range=5)
        return dict(cx=float(cx), cy=float(cy), angle_deg=float(r), angle_unrefined_deg=float(angle),
                    scale_x=float(rsx), scale_y=float(rsy), scale_x_unrefined=float(sx), scale_y_unrefined=float(sy),
                    score=float(score[0, 0, py, px].item()), refine_score=float(r_score))

    def call(self, image, template):
        """One timed call on decoded BGR arrays; returns (detections, info)."""
        torch, cv2, args = self.torch, self.cv2, self.args
        t0 = time.perf_counter()
        k = 1.
        if args.adapt:
            k = 36. / max(template.shape[:2])
            interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR
            image = cv2.resize(image, (max(1, round(image.shape[1] * k)), max(1, round(image.shape[0] * k))), interpolation=interp)
            template = cv2.resize(template, (max(1, round(template.shape[1] * k)), max(1, round(template.shape[0] * k))),
                                  interpolation=interp)
        with torch.no_grad():
            x, t = self.tensors(image, template)
            maps = self.model(x, t)
            self.sync()
            t1 = time.perf_counter()
            pred = maps[0][0, 0].cpu()
            maps = [m.cpu() for m in maps]
            support = torch.nonzero(pred > 0.5).numpy()
            if args.adapt:
                pooled = torch.nn.functional.max_pool2d(pred[None, None], 3, stride=1, padding=1)[0, 0]
                ys, xs = torch.nonzero((pred >= args.peak_threshold) & (pred >= pooled), as_tuple=True)
                order = torch.argsort(pred[ys, xs], descending=True)[:2 * args.limit]
                peaks = [(int(ys[i]), int(xs[i])) for i in order]
            else:
                py, px = torch.nonzero(pred == pred.max())[0].numpy()
                peaks = [(int(py), int(px))]
            search, templ = self.refine_images(x, t)
            poses = [self.pose_at(maps, support, py, px, search, templ) for py, px in peaks]
        detections = []
        for p in poses:
            d = dict(kind='pose', cx=p['cx'] / k, cy=p['cy'] / k, angle_deg=p['angle_deg'], width=p['scale_x'] * TW / k,
                     height=p['scale_y'] * TH / k, scale_x=p['scale_x'], scale_y=p['scale_y'], scale_reference=[TW / k, TH / k],
                     score=p['score'], refine_score=p['refine_score'], angle_unrefined_deg=p['angle_unrefined_deg'],
                     scale_x_unrefined=p['scale_x_unrefined'], scale_y_unrefined=p['scale_y_unrefined'])
            detections.append(d)
        if args.adapt:
            kept = []
            for d in sorted(detections, key=lambda d: -d['score']):
                if len(kept) >= args.limit:
                    break
                if all(rotated_overlap(d, q) <= args.overlap for q in kept):
                    kept.append(d)
            detections = kept
        t2 = time.perf_counter()
        info = dict(call_ms=(t2 - t0) * 1e3, forward_ms=(t1 - t0) * 1e3, post_ms=(t2 - t1) * 1e3, k=k, peaks=len(peaks),
                    input_shape=list(image.shape), score_map_shape=list(pred.shape))
        return detections, info


def memory(device):
    import torch
    out = dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if device.type == 'mps':
        out['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--image', type=Path)
    p.add_argument('--template', type=Path)
    p.add_argument('--batch', type=Path, help='JSON list of {"id", "image", "template"}; writes JSON lines to --output')
    p.add_argument('--device', default='cpu', choices=('cpu', 'mps'))
    p.add_argument('--pipeline', default='test_single', choices=('test_single', 'test_py'))
    p.add_argument('--adapt', action='store_true', help='TDCM-adapted: rescale to a 36 px template, multi-target')
    p.add_argument('--limit', type=int, default=1)
    p.add_argument('--overlap', type=float, default=.2)
    p.add_argument('--peak-threshold', type=float, default=.5, help='TDCM-adapted local-maximum threshold')
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--threads', type=int, default=0, help='torch intra-op threads; 0 keeps the default')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    model, device, load_ms = load(args)
    runner = Runner(args, model, device)
    import cv2
    import numpy as np
    import torch
    head = dict(method='tdcm_adapted' if args.adapt else 'tdcm', pipeline=args.pipeline, device=args.device,
                threads=torch.get_num_threads(), torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__,
                parameters=sum(v.numel() for v in model.parameters()), buffers=sum(v.numel() for v in model.buffers()),
                load_model_ms=load_ms, limit=args.limit, overlap=args.overlap,
                peak_threshold=args.peak_threshold if args.adapt else None)

    if args.batch:
        items = json.loads(args.batch.read_text())
        with open(args.output, 'w') as out:
            out.write(json.dumps(dict(header=head)) + '\n')
            for n, item in enumerate(items):
                image, template = read_pair(args, item['image'], item['template'])
                if n == 0:
                    for _ in range(max(1, args.warmup)):
                        runner.call(image, template)
                detections, info = runner.call(image, template)
                out.write(json.dumps(dict(id=item['id'], timing_ms=info, detections=detections)) + '\n')
            out.write(json.dumps(dict(footer=dict(memory=memory(device), items=len(items)))) + '\n')
        return

    image, template = read_pair(args, args.image, args.template)
    for _ in range(args.warmup):
        runner.call(image, template)
    calls = [runner.call(image, template) for _ in range(args.repeats)]
    detections, info = calls[0]
    result = dict(head, image_shape=list(image.shape), template_shape=list(template.shape),
                  timing_ms=dict(load_model=load_ms, calls=[c[1]['call_ms'] for c in calls],
                                 call_median=float(np.median([c[1]['call_ms'] for c in calls])),
                                 forward=[c[1]['forward_ms'] for c in calls], post=[c[1]['post_ms'] for c in calls]),
                  exact_repeat=all(c[0] == detections for c in calls), call_info=info, memory=memory(device),
                  detections=detections, count=len(detections),
                  deviations=(['TDCM-adapted: resize to a 36 px template, multi-target local maxima, rotated NMS']
                              if args.adapt else ['native-size search image', 'single target per call']))
    text = json.dumps(result, indent=1)
    if args.output:
        args.output.write_text(text + '\n')
    else:
        print(text)


if __name__ == '__main__':
    main()
