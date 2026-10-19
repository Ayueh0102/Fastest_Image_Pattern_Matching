"""Stage 36: paired full-pipeline Metal resident coarse peak experiment.

Fresh serial processes, alternating method order, independent truth matching.
Both routes use the same build and settings; only coarse_peak_mode differs.
FP32 scores/order are explicitly not required to equal the binary64 control.
No ground truth is given to the native runner. Negative results are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
from types import SimpleNamespace

import benchmark_native_pipeline as native
from many_object_accuracy import evaluate

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = 'controlled6,heldout12,small_fullres,8k_t256_angle180_3scales,many128_pose,many512_pose'
METHODS = {'cpu_peaks': 1, 'metal_resident_peaks': 2}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def job(case, mode, repeats, warmup):
    result = native.native_job(case, 'native_tile', repeats, warmup)
    result.update(coarse_peak_mode=mode, fine_ncc_mode=2, fine_fused=True,
                  fine_angle_reuse=True, pose_resident_mode=0, pose_input_mode=1,
                  pose_storage_mode=1, refinement_backend=1)
    return result


def summarize(manifest, root):
    if not manifest.get('complete'):
        raise ValueError('Incomplete experiment')
    rows = []
    for case in manifest['cases']:
        grouped = {name: [] for name in METHODS}
        for entry in manifest['executions']:
            if entry['case'] != case['name']:
                continue
            for key in ('job', 'result'):
                if sha(root/entry[key]) != entry[key+'_sha256']:
                    raise ValueError('Changed '+key+': '+entry[key])
            grouped[entry['method']].append((entry['block'], json.loads((root/entry['result']).read_text())))
        for name, blocks in grouped.items():
            if sorted(b for b, _ in blocks) != list(range(manifest['blocks'])):
                raise ValueError('Missing or duplicate blocks for '+name)
        reference = grouped['cpu_peaks'][0][1]['first']
        methods = {}
        for name, blocks in grouped.items():
            calls = [r for _, w in blocks for r in [w['first'], *w['runs']]]
            runs = [r for _, w in blocks for r in w['runs']]
            accuracy = [evaluate(r['matches'], case['truth'])['strict'] for r in calls]
            # Work can change after coarse selection. Record it, never assume
            # that a lower candidate count is an equal-work GPU acceleration.
            stages = {key: statistics.median(r[key] for r in runs)
                      for key in ('total_ms', 'prepare_ms', 'proposal_ms', 'refine_ms', 'nms_ms')}
            stages.update({key: statistics.median(sum(w[key] for w in r['work']) for r in runs)
                           for key in ('device_ms', 'wait_ms', 'copy_ms', 'normalization_ms', 'coarse_commands', 'fine_maps')})
            methods[name] = dict(stages=stages,
                times_ms=[r['total_ms'] for r in runs],
                blocks=[dict(block=b, median_ms=statistics.median(r['total_ms'] for r in w['runs'])) for b, w in blocks],
                first_ms=[w['first']['total_ms'] for _, w in blocks],
                construct_ms=[w['construct_ms'] for _, w in blocks],
                peak_rss_bytes=max(w['final_memory']['peak_rss_bytes'] for _, w in blocks),
                frame_coarse_gpu_bytes=max(r['frame_coarse_gpu_bytes'] for r in calls),
                frame_fine_gpu_bytes=max(r['frame_fine_gpu_bytes'] for r in calls),
                accuracy_counts=sorted({(a['tp'], a['fp'], a['fn']) for a in accuracy}),
                all_gt_pass=all(a['fp']==0 and a['fn']==0 for a in accuracy),
                max_errors={k: max((a['max_errors'][k] for a in accuracy if a['max_errors'][k] is not None), default=None)
                            for k in ('position_px', 'angle_deg', 'relative_scale')},
                repeat_exact=all(w['exact_repeat'] for _, w in blocks),
                ordered_matches_equal_control=all(r['matches']==reference['matches'] for r in calls),
                ordered_proposals_equal_control=all(r['proposals']==reference['proposals'] for r in calls),
                proposal_counts=sorted({len(r['proposals']) for r in calls}),
                coarse_work_equal=all([(w['maps'],w['windows']) for w in r['work']]==
                    [(w['maps'],w['windows']) for w in reference['work']] for r in calls),
                no_cpu_dense_normalization=all(all(w['normalization_ms']==0 and w['coarse_host_scoremap_peak_bytes']==0
                    for w in r['work']) for r in calls))
        a, b = methods['cpu_peaks'], methods['metal_resident_peaks']
        memory_ok = b['peak_rss_bytes'] <= a['peak_rss_bytes']+max(8*1024**2,.1*a['peak_rss_bytes'])
        paired = [dict(block=x['block'], delta_percent=100*(y['median_ms']/x['median_ms']-1),
                       within_regression_budget=y['median_ms']<=x['median_ms']+max(2.,.1*x['median_ms']))
                  for x,y in zip(a['blocks'],b['blocks'])]
        rows.append(dict(name=case['name'],truth_count=len(case['truth']),methods=methods,
                         delta_percent=100*(b['stages']['total_ms']/a['stages']['total_ms']-1), paired=paired,
                         memory_within_budget=memory_ok,
                         candidate_for_adoption=b['all_gt_pass'] and b['repeat_exact'] and b['coarse_work_equal'] and
                         b['no_cpu_dense_normalization'] and memory_ok and all(p['within_regression_budget'] for p in paired)))
    return dict(cases=rows, limitation='Coarse NCC/normalization/greedy peaks only; CPU rotation, compact handoff, fine control and final NMS remain.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=ROOT/'build/fourier-gpu')
    p.add_argument('--output', type=Path, default=ROOT/'build/stage36-coarse-resident/formal.json')
    p.add_argument('--cases', default=DEFAULT_CASES)
    p.add_argument('--blocks', type=int, default=4)
    p.add_argument('--repeats', type=int, default=4)
    p.add_argument('--warmup', type=float, default=1.)
    p.add_argument('--summarize', type=Path)
    args = p.parse_args()
    if args.summarize:
        save(args.output, summarize(json.loads(args.summarize.read_text()), args.summarize.parent)); return
    if args.blocks<1 or args.repeats<1 or not 0<=args.warmup<=60:
        p.error('positive blocks/repeats and warmup in 0..60 required')
    root = args.output.resolve().parent
    root.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        raise ValueError('Refusing to overwrite experiment manifest')
    names = args.cases.split(','); inputs = root/'inputs'; inputs.mkdir(exist_ok=True)
    regular = [n for n in names if not n.startswith('many')]
    crowded = [n for n in names if n.startswith('many')]
    cases = native.old.coarse.make_cases(SimpleNamespace(cases=','.join(regular),suite='all'),inputs) if regular else []
    if crowded: cases += native.many.make_cases(inputs,crowded)
    cases.sort(key=lambda c:names.index(c['name']))
    binary = (args.build/'experiments/fpm_native_pipeline_runner').resolve()
    sources = [p for folder in ('src','include','experiments','tools','tests') for p in (ROOT/'cpp-core'/folder).rglob('*')
               if p.is_file() and p.suffix in ('.cpp','.hpp','.h','.mm','.metal','.py','.in')]
    binaries = [binary, *binary.parent.glob('*.dylib')]
    manifest = dict(version=1,platform=platform.platform(), blocks=args.blocks,repeats=args.repeats,
        warmup_seconds=args.warmup,cases=cases,methods=METHODS,executions=[],complete=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        binary_sha256={str(p):sha(p) for p in binaries},
        input_sha256={c[k]:sha(c[k]) for c in cases for k in ('source','template')})
    save(args.output,manifest)
    for case in cases:
        for block in range(args.blocks):
            for name in list(METHODS)[::1 if block%2==0 else -1]:
                label=f"{case['name']}-{name}-b{block}"
                jp=root/'jobs'/(label+'.json'); rp=root/'results'/(label+'.json')
                save(jp,job(case,METHODS[name],args.repeats,args.warmup))
                print('Running '+label,flush=True)
                child=subprocess.run([str(binary),str(jp)],capture_output=True,text=True,timeout=900)
                (jp.with_suffix('.stderr.txt')).write_text(child.stderr)
                if child.returncode: raise RuntimeError(label+': '+child.stderr[-4000:])
                raw=json.loads(child.stdout); save(rp,raw)
                manifest['executions'].append(dict(case=case['name'],method=name,block=block,
                    job=str(jp.relative_to(root)),job_sha256=sha(jp),result=str(rp.relative_to(root)),result_sha256=sha(rp)))
                save(args.output,manifest)
    manifest['complete']=True
    manifest['unchanged']=all(sha(ROOT/k)==v for k,v in manifest['source_sha256'].items()) and all(
        sha(k)==v for group in ('binary_sha256','input_sha256') for k,v in manifest[group].items())
    manifest['summary']=summarize(manifest,root); save(args.output,manifest)
    if not manifest['unchanged']: raise SystemExit('Inputs/source/binaries changed during measurement')


if __name__=='__main__':
    main()
