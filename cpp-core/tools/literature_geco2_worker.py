"""Round 50: run GeCo2 (AAAI 2026) with official weights on one image and its exemplar boxes.

Runs inside the GeCo2 environment (``python -I``), never in python-port/venv.
Nothing is downloaded: the SAM2 backbone checkpoint that GeCo2 fetches through
torch.hub must already sit in ``<torch-home>/hub/checkpoints``; the worker fails
instead of reaching the network. Steps copy the authors' code:

* model: ``models.counter_infer.build_model`` with the ``utils.arg_parser``
  defaults, weights loaded with strict=False exactly as ``inference.py`` does
  (missing and unexpected keys are counted and reported, not hidden);
* input: ``demo_gradio.py`` preprocessing (RGB / 255, ImageNet normalisation,
  ``utils.data.resize_and_pad`` to 1024 with the exemplar-size rule);
* output: ``inference.py`` post-processing (keep box_v > max(box_v) * threshold,
  NMS 0.5, clamp, drop centres in the padding, rescale to source pixels). The
  FSCD-147 evaluation threshold is 0.11; the Gradio demo default is 0.33.

Deviations, both recorded in the output:

* the official ``MultiScaleDeformableAttention`` extension is CUDA-only, so on
  CPU/MPS the worker installs a module whose forward calls the reference
  ``ms_deform_attn_core_pytorch`` from the same Deformable-DETR source file. Use
  ``--deform-attn cuda`` on a CUDA machine with the compiled extension;
* ``models.query_generator.C_base`` computes its reference points only when its
  CPU-built shape tensors move to another device, so the official code fails on
  a pure CPU run. On CPU the worker sets the three reference-point tensors with
  the class's own ``get_reference_points`` and the same arguments beforehand.

Output: axis-aligned boxes in source pixels with scores, the count, and times
(model load, preprocessing, forward, post-processing) excluding image decoding.
"""
import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import types


def install_reference_deform_attn(repo):
    """Expose Deformable-DETR's ops as models.ops (what install.sh copies) with a PyTorch kernel."""
    ops_dir = repo / 'Deformable-DETR' / 'models' / 'ops'
    shim = types.ModuleType('MultiScaleDeformableAttention')

    def forward(value, shapes, level_start_index, locations, weights, im2col_step):
        from models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
        return ms_deform_attn_core_pytorch(value, [(int(h), int(w)) for h, w in shapes.tolist()], locations, weights)

    def backward(*_):
        raise RuntimeError('reference deformable attention is inference-only')

    shim.ms_deform_attn_forward = forward
    shim.ms_deform_attn_backward = backward
    sys.modules['MultiScaleDeformableAttention'] = shim
    return ops_dir


def register_ops_package(ops_dir):
    import models  # noqa: F401  (runs the hydra config initialisation)
    package = types.ModuleType('models.ops')
    package.__path__ = [str(ops_dir)]
    sys.modules['models.ops'] = package


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--torch-home', type=Path, required=True)
    p.add_argument('--image', type=Path)
    p.add_argument('--exemplars', help='JSON list of [x0, y0, x1, y1] boxes in source pixels')
    p.add_argument('--batch', type=Path, help='JSON list of {"id", "image", "exemplars"}; writes JSON lines to --output')
    p.add_argument('--threshold', type=float, default=.11)
    p.add_argument('--device', default='cpu', choices=('cpu', 'mps', 'cuda'))
    p.add_argument('--deform-attn', default='reference', choices=('reference', 'cuda'))
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--output', type=Path)
    args = p.parse_args()

    os.environ['TORCH_HOME'] = str(args.torch_home.resolve())
    sam2 = args.torch_home / 'hub' / 'checkpoints' / 'sam2_hiera_base_plus.pt'
    if not sam2.is_file():
        raise SystemExit(f'missing {sam2}; place the SAM2 checkpoint there first (no downloads here)')
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    ops_dir = install_reference_deform_attn(repo) if args.deform_attn == 'reference' else repo / 'models' / 'ops'

    import numpy as np
    import torch
    from PIL import Image
    from torchvision import ops
    from torchvision import transforms as T
    if args.deform_attn == 'reference':
        register_ops_package(ops_dir)
    build_model = importlib.import_module('models.counter_infer').build_model
    get_argparser = importlib.import_module('utils.arg_parser').get_argparser
    resize_and_pad = importlib.import_module('utils.data').resize_and_pad

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    def sync():
        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    margs = get_argparser().parse_args([])
    model = build_model(margs)
    state = torch.load(args.weights, map_location='cpu', weights_only=True)['model']
    state = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    deviations = ['reference deformable attention'] if args.deform_attn == 'reference' else []
    if device.type == 'cpu':
        q = model.adapt_features
        q.reference_points1 = q.get_reference_points(q.spatial_shapes1, q.valid_ratios1, device='cpu')
        q.reference_points2 = q.get_reference_points(q.spatial_shapes2, q.valid_ratios2, device='cpu')
        q.reference_points = q.get_reference_points(q.spatial_shapes, q.valid_ratios, device='cpu')
        deviations.append('CPU reference points set before forward (upstream sets them only on a device move)')
    sync()
    load_ms = (time.perf_counter() - t0) * 1e3

    def call(image, exemplars):
        with torch.no_grad():
            t1 = time.perf_counter()
            x = torch.tensor(image).permute(2, 0, 1).float() / 255.
            x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
            boxes = torch.tensor(exemplars, dtype=torch.float32)
            img, boxes, scale = resize_and_pad(x, boxes, size=1024.)
            img, boxes = img.unsqueeze(0).to(device), boxes.unsqueeze(0).to(device)
            sync()
            t2 = time.perf_counter()
            outputs, _, _, _, _ = model(img, boxes)
            sync()
            t3 = time.perf_counter()
            out = outputs[0]
            pred = out['pred_boxes'].reshape(-1, 4)
            v, s = out['box_v'].reshape(-1), out['scores'].reshape(-1)
            keep_boxes, keep_scores = torch.zeros((0, 4)), torch.zeros(0)
            if pred.numel():
                mask = v > v.max() * args.threshold
                keep = ops.nms(pred[mask], v[mask], .5)
                keep_boxes = torch.clamp(pred[mask][keep], 0, 1)
                keep_scores = s[mask][keep]
                # Padding is right/bottom only: centres beyond the scaled image are dropped (inference.py).
                h, w = image.shape[:2]
                centre = (keep_boxes[:, :2] + keep_boxes[:, 2:]) / 2 * img.shape[-1]
                valid = (centre[:, 0] < w * scale) & (centre[:, 1] < h * scale)
                keep_boxes, keep_scores = keep_boxes[valid], keep_scores[valid]
            keep_boxes = (keep_boxes * img.shape[-1] / scale).cpu().tolist()
            keep_scores = keep_scores.cpu().tolist()
            t4 = time.perf_counter()
        detections = [dict(kind='box', x0=b[0], y0=b[1], x1=b[2], y1=b[3], score=float(c))
                      for b, c in zip(keep_boxes, keep_scores)]
        timing = dict(call_ms=(t4 - t1) * 1e3, preprocess_ms=(t2 - t1) * 1e3, forward_ms=(t3 - t2) * 1e3,
                      postprocess_ms=(t4 - t3) * 1e3, input_scale=float(scale))
        return detections, timing

    def memory():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out = dict(peak_rss_bytes=rss if sys.platform == 'darwin' else rss * 1024)  # Linux reports KiB
        if device.type == 'mps':
            out['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
        elif device.type == 'cuda':
            out['cuda_max_allocated_bytes'] = torch.cuda.max_memory_allocated()
            out['cuda_max_reserved_bytes'] = torch.cuda.max_memory_reserved()
        return out

    prefix = lambda keys: dict(Counter(k.split('.')[0] for k in keys))  # noqa: E731
    msda = sys.modules.get('MultiScaleDeformableAttention')
    msda_file = getattr(msda, '__file__', None) or 'reference shim (ms_deform_attn_core_pytorch)'
    head = dict(method='geco2', device=args.device, deform_attn=args.deform_attn, deviations=deviations,
                threads=torch.get_num_threads(), torch=torch.__version__, numpy=np.__version__,
                cuda=torch.version.cuda if device.type == 'cuda' else None,
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                gpu_capability=list(torch.cuda.get_device_capability(device)) if device.type == 'cuda' else None,
                msda_module=msda_file,
                threshold=args.threshold, weights=args.weights.name, checkpoint_keys=len(state),
                missing_keys=len(missing), unexpected_keys=len(unexpected),
                missing_key_prefixes=prefix(missing), unexpected_key_prefixes=prefix(unexpected),
                missing_key_examples=list(missing)[:8], unexpected_key_examples=list(unexpected)[:8], load_model_ms=load_ms)

    def rgb(path):
        return np.asarray(Image.open(path).convert('RGB'))

    if args.batch:
        items = json.loads(args.batch.read_text())
        with open(args.output, 'w') as out:
            out.write(json.dumps(dict(header=head)) + '\n')
            for n, item in enumerate(items):
                image = rgb(item['image'])
                if n == 0:
                    for _ in range(max(1, args.warmup)):
                        call(image, item['exemplars'])
                detections, timing = call(image, item['exemplars'])
                out.write(json.dumps(dict(id=item['id'], timing_ms=timing, detections=detections, count=len(detections))) + '\n')
            out.write(json.dumps(dict(footer=dict(memory=memory(), items=len(items)))) + '\n')
        return

    image = rgb(args.image)
    exemplars = json.loads(args.exemplars)
    for _ in range(args.warmup):
        call(image, exemplars)
    calls = [call(image, exemplars) for _ in range(args.repeats)]
    detections, timing = calls[0]
    result = dict(head, memory=memory(), image_shape=list(image.shape), exemplars=exemplars, input_scale=timing['input_scale'],
                  timing_ms=dict(load_model=load_ms, calls=[c[1]['call_ms'] for c in calls],
                                 call_median=float(np.median([c[1]['call_ms'] for c in calls])),
                                 forward=[c[1]['forward_ms'] for c in calls], preprocess=timing['preprocess_ms'],
                                 postprocess=timing['postprocess_ms']),
                  exact_repeat=all(c[0] == detections for c in calls), detections=detections, count=len(detections))
    text = json.dumps(result, indent=1)
    if args.output:
        args.output.write_text(text + '\n')
    else:
        print(text)


if __name__ == '__main__':
    main()
