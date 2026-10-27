"""Round 50: the four pre-checks before any GeCo2 CUDA run on the lab PC (WSL2 Ubuntu, RTX PRO 4000).

Runs in the GeCo2 CUDA environment (``python -I``) after literature_geco2_cuda_setup.sh:
1. torch sees the GPU (cuda available, compute capability, sm_120 in the build's arch list).
2. The compiled MultiScaleDeformableAttention extension is the one imported, and its CUDA
   forward agrees with Deformable-DETR's ms_deform_attn_core_pytorch on random inputs.
   GeCo2's ops import the extension unconditionally, so there is no silent fallback; this
   records the module file and the kernel-vs-reference difference.
3. One image through literature_geco2_worker.py (--device cuda --deform-attn cuda) gives
   finite boxes and scores, and the worker header names the compiled module.
4. The checkpoint leaves exactly the 149 sam_mask keys unloaded and every one of them is
   identical across two init seeds, i.e. counter_infer's MaskProcessor loads them from SAM2;
   the 8 unused training heads (class_embed_aux 2, bbox_embed_aux 6) are the same as on the Mac.
Writes one JSON record; the exit status is non-zero when any check fails.
"""
import argparse
import importlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

# literature_geco2_worker.py header on the M4 Pro (CPU and MPS): training-only auxiliary heads.
MAC_UNEXPECTED = {'class_embed_aux': 2, 'bbox_embed_aux': 6}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--torch-home', type=Path, required=True)
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--exemplars', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.environ['TORCH_HOME'] = str(args.torch_home.resolve())
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    import torch
    record = dict(torch=torch.__version__, torch_cuda=torch.version.cuda)

    ok = torch.cuda.is_available()
    record['cuda'] = dict(available=ok, device=torch.cuda.get_device_name(0) if ok else None,
                          capability=list(torch.cuda.get_device_capability(0)) if ok else None,
                          arch_list=torch.cuda.get_arch_list() if ok else [])
    record['cuda']['pass'] = ok and 'sm_120' in record['cuda']['arch_list']
    if not ok:
        return finish(args.output, record)

    import MultiScaleDeformableAttention as MSDA
    from models.ops.functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch
    g = torch.Generator().manual_seed(0)
    shapes = torch.tensor([[64, 64], [32, 32], [16, 16], [8, 8]], dtype=torch.long)
    start = torch.cat((shapes.new_zeros(1), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]))
    n, heads, dim, queries, levels, points = 2, 8, 32, 300, len(shapes), 4
    value = torch.rand(n, int((shapes[:, 0] * shapes[:, 1]).sum()), heads, dim, generator=g)
    loc = torch.rand(n, queries, heads, levels, points, 2, generator=g)
    w = torch.rand(n, queries, heads, levels, points, generator=g)
    w = w / w.sum((-1, -2), keepdim=True)
    with torch.no_grad():
        ref = ms_deform_attn_core_pytorch(value.double(), [tuple(s) for s in shapes.tolist()], loc.double(), w.double())
        got = MSDeformAttnFunction.apply(value.cuda(), shapes.cuda(), start.cuda(), loc.cuda(), w.cuda(), 64).cpu()
    diff = float((got.double() - ref).abs().max())
    module = getattr(MSDA, '__file__', '')
    record['msda'] = dict(module=module, max_abs_diff_vs_reference_fp64=diff, output_shape=list(got.shape))
    record['msda']['pass'] = module.endswith('.so') and math.isfinite(diff) and diff < 1e-4

    run = subprocess.run([sys.executable, '-I', str(args.worker), '--repo', str(repo), '--weights', str(args.weights),
                          '--torch-home', str(args.torch_home), '--image', str(args.image), '--exemplars', args.exemplars,
                          '--device', 'cuda', '--deform-attn', 'cuda', '--warmup', '1', '--repeats', '1'],
                         capture_output=True, text=True)
    record['one_image'] = dict(returncode=run.returncode, stderr_tail=run.stderr.splitlines()[-20:])
    if run.returncode == 0:
        out = json.loads(run.stdout)
        values = [v for d in out['detections'] for v in (d['x0'], d['y0'], d['x1'], d['y1'], d['score'])]
        record['one_image'].update(count=out['count'], finite=all(math.isfinite(v) for v in values), msda_module=out['msda_module'],
                                   gpu=out['gpu'], gpu_capability=out['gpu_capability'], call_median_ms=out['timing_ms']['call_median'],
                                   memory=out['memory'], missing_key_prefixes=out['missing_key_prefixes'],
                                   unexpected_key_prefixes=out['unexpected_key_prefixes'])
        record['one_image']['pass'] = (out['count'] > 0 and record['one_image']['finite']
                                        and out['msda_module'] == module and out['deform_attn'] == 'cuda'
                                        and out['unexpected_key_prefixes'] == MAC_UNEXPECTED and out['missing_key_prefixes'] == {'sam_mask': 149})

    build = importlib.import_module('models.counter_infer').build_model
    margs = importlib.import_module('utils.arg_parser').get_argparser().parse_args([])
    torch.manual_seed(0)
    s1 = build(margs).state_dict()
    torch.manual_seed(1)  # a different init seed: anything not loaded from SAM2 differs
    s2 = build(margs).state_dict()
    ck = torch.load(args.weights, map_location='cpu', weights_only=True)['model']
    ck = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in ck.items()}
    missing = [k for k in s1 if k not in ck]
    sam = [k for k in s1 if k.startswith('sam_mask.')]
    same = [k for k in sam if torch.equal(s1[k], s2[k])]
    prefixes = lambda keys: dict(Counter(k.split('.')[0] for k in keys))  # noqa: E731
    record['sam_mask'] = dict(missing_prefixes=prefixes(missing), unexpected_prefixes=prefixes(k for k in ck if k not in s1),
                              sam_mask_keys=len(sam), identical_across_seeds=len(same), seed_dependent=[k for k in sam if k not in same][:10])
    record['sam_mask']['pass'] = (record['sam_mask']['missing_prefixes'] == {'sam_mask': 149}
                                  and record['sam_mask']['unexpected_prefixes'] == MAC_UNEXPECTED and len(same) == len(sam) == 149)
    return finish(args.output, record)


def finish(path, record):
    checks = [record.get(k, {}).get('pass', False) for k in ('cuda', 'msda', 'one_image', 'sam_mask')]
    record['all_pass'] = all(checks)
    path.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))
    sys.exit(0 if record['all_pass'] else 1)


if __name__ == '__main__':
    main()
