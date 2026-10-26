"""Round 50: run the official PoseMatch-TDCM single-target pipeline on one image pair.

Runs inside the TDCM environment (``python -I``), never in python-port/venv. The
repository and weights are passed as paths; nothing is downloaded. The steps copy
the authors' evaluation script ``test_single.py`` (the one that produced their
published ``res/result-single.txt``): RGB input with ImageNet normalisation, the
template resized to 36x36 by torchvision, the arg-max peak, ``get_center`` over
score > 0.5, sign/cos angle, the two scale heads, then ``refine_angle_bisection``,
``refine_scale_x`` and ``refine_scale_y`` (search_range 0.15, step 0.015) on the
script's own grey conversion. Two deviations are explicit and recorded:

* the search image is used at its native size (the script resizes to 224x224,
  which is a no-op only for the authors' 224x224 crops);
* the official code returns one target per call; multi-target extraction is not
  published, so this worker only reports the single peak.

Output JSON (stdout or --output): one pose with centre, angle, scale_x/y and the
model and refinement times. Times are wall clock around each step, excluding
image decoding; model loading is reported separately.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--device', default='cpu', choices=('cpu', 'mps'))
    p.add_argument('--threads', type=int, default=0, help='torch intra-op threads; 0 keeps the default')
    p.add_argument('--output', type=Path)
    args = p.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    import cv2
    import numpy as np
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from model import Model
    from utils import get_center
    from utils.refine import refine_angle_bisection, refine_scale_x, refine_scale_y

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    th, tw = 36, 36
    t0 = time.perf_counter()
    model = Model(None).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1e3
    parameters = sum(v.numel() for v in model.parameters())
    buffers = sum(v.numel() for v in model.buffers())

    # Decoding mirrors CoCo_Dataset: cv2 BGR -> RGB, template cropped image file.
    image = cv2.cvtColor(cv2.imread(str(args.image), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    template = cv2.cvtColor(cv2.imread(str(args.template), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_image = transforms.Compose([transforms.ToTensor(), normalise])
    to_template = transforms.Compose([transforms.Resize((th, tw)), transforms.ToTensor(), normalise])

    with torch.no_grad():
        t1 = time.perf_counter()
        x = to_image(Image.fromarray(image)).unsqueeze(0).to(device)
        t = to_template(Image.fromarray(template)).unsqueeze(0).to(device)
        score, sign_map, cos_map, scale_x_map, scale_y_map = model(x, t)
        pred = score[0, 0].cpu()
        peak = pred.max().item()
        pre_y, pre_x = torch.nonzero(pred == pred.max())[0].numpy()
        pred_y, pred_x = get_center((pre_y, pre_x), torch.nonzero(pred > 0.5).numpy())
        sign = 1 if sign_map[0, 0, pre_y, pre_x] > 0.5 else -1
        angle = sign * (torch.arccos(cos_map[0, 0, pre_y, pre_x].clamp(-1, 1)) * 180 / math.pi).item()
        sx = scale_x_map[0, 0, pre_y, pre_x].item()
        sy = scale_y_map[0, 0, pre_y, pre_x].item()
        if device.type != 'cpu':
            torch.mps.synchronize()
        infer_ms = (time.perf_counter() - t1) * 1e3

        # test_single.py: denormalise with *0.5+0.5 and BGR2GRAY on the RGB tensor, as published.
        t2 = time.perf_counter()
        search = (x[0].cpu() * 0.5 + 0.5).permute(1, 2, 0).numpy()
        templ = (t[0].cpu() * 0.5 + 0.5).permute(1, 2, 0).numpy()
        search = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        templ = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
        refined, refine_score = refine_angle_bisection(search, templ, pred_x, pred_y, sx, sy, angle)
        rsx, _ = refine_scale_x(search, templ, pred_x, pred_y, sx, sy, refined, search_range=0.15, step_size=0.015)
        rsy, _ = refine_scale_y(search, templ, pred_x, pred_y, rsx, sy, refined, search_range=0.15, step_size=0.015)
        refine_ms = (time.perf_counter() - t2) * 1e3

    out = dict(method='tdcm', device=args.device, threads=torch.get_num_threads(), torch=torch.__version__,
               numpy=np.__version__, opencv=cv2.__version__, parameters=parameters, buffers=buffers,
               image_shape=list(image.shape), template_shape=list(template.shape), template_input=[th, tw],
               score_map_shape=list(pred.shape), timing_ms=dict(load_model=load_ms, infer=infer_ms, refine=refine_ms),
               detections=[dict(kind='pose', cx=float(pred_x), cy=float(pred_y), angle_deg=float(refined),
                                angle_unrefined_deg=float(angle), scale_x=float(rsx), scale_y=float(rsy),
                                scale_x_unrefined=float(sx), scale_y_unrefined=float(sy), score=float(peak),
                                refine_score=float(refine_score), scale_reference=[tw, th])],
               deviations=['native-size search image (official eval resizes 224x224 crops, a no-op there)',
                           'single target per call: multi-target extraction is not published'])
    text = json.dumps(out, indent=1)
    if args.output:
        args.output.write_text(text + '\n')
    else:
        print(text)


if __name__ == '__main__':
    main()
