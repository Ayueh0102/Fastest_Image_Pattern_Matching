"""Validate frozen adaptive NCC workers, hashes and independently rebuild summary.

The native workers independently compare against exhaustive scores after timing.
This checks their contracts/accounting and prevents missing/failed workers or a
hand-edited summary from being reported as a completed successful study.
"""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile

from benchmark_adaptive_ncc import sha, save


def require(value, message):
    if not value: raise ValueError(message)


def finite(value, nonnegative=True):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and (value>=0 or not nonnegative)


def worker(entry, case, method, block, protocol):
    require(entry['case']==case['name'] and entry['method']==method and entry['block']==block,'worker identity')
    require(entry['returncode']==0,'worker failed')
    r=entry['result'];v=r['validation'];s=r['statistics']
    kind=method.rstrip('0123456789');cap=int(method[len(kind):] or 16)
    require(r['method']==kind and r['cap']==cap and r['mode']==case['job']['mode'],'native route')
    require(r['valid'] and r['exact_repeat'] and r['exact_final'] and v['valid'] and v['peak_order_exact'],'native correctness')
    require(v['false_rejects']==v['survivor_bit_mismatches']==v['invalid_values']==0,'oracle failures')
    require(not r['timed_calls_collect_stats'] and not r['rss_includes_oracle'] and not r['fine_path_changed'],'measurement scope')
    require(r['full_search']==(case['job']['mode']=='full'),'mode attribution')
    for group in (s,{k:v[k] for k in ('maps','windows','survivors','rejected','above_threshold','false_rejects','survivor_bit_mismatches','invalid_values')}):
        require(all(finite(x) and int(x)==x for x in group.values()),'invalid counters')
    require(v['windows']==s['windows']==v['survivors']+v['rejected'],'window accounting')
    require(s['rejected']==v['rejected'] and s['adaptive_rejected']<=s['rejected'] and s['rejected']+s['full_dot_products']<=s['windows'],'rejection accounting')
    require(v['above_threshold']<=v['survivors'],'positive-score accounting')
    if kind in ('full','legacy'):require(s['adaptive_checks']==s['adaptive_rejected']==s['adaptive_rectangles']==0,'unexpected adaptive work')
    if kind=='full':require(s['rejected']==0,'exhaustive path pruned')
    require(0<=r['rectangles']<=cap,'rectangle cap')
    require(len(r['samples_ms'])==protocol['repeats'] and len(r['warmups_ms'])>0,'sample completeness')
    require(all(finite(t) for t in r['samples_ms']+r['warmups_ms']+[r['first_ms'],r['model_build_ms'],r['pattern_build_ms']]),'invalid timing')
    require(r['measured_memory']['peak_rss_bytes']>0 and finite(r['measured_memory']['peak_rss_bytes']),'RSS')
    if case['job']['mode']=='full':
        require(v['maps']==r['profile_after_timing']['coarse_maps'] and v['windows']==r['profile_after_timing']['coarse_windows'],'profile work')
    else:require(v['maps']==1,'map count')


def validate(out):
    p=json.loads((out/'protocol.json').read_text())
    for rel,digest in p['source_sha256'].items():require(sha(out/'source'/rel)==digest,'source hash '+rel)
    for rel,digest in p['input_sha256'].items():require(sha(out/rel)==digest,'input hash '+rel)
    expected=set();records=[];windows=maps=0
    for c in p['cases']:
        methods=p['full_methods'] if c['job']['mode']=='full' else p['map_methods']
        for b in range(p['blocks']):
            for m in methods:
                stem=f"{c['name']}_b{b}_{m}";expected.add(stem+'.json')
                entry=json.loads((out/'raw'/(stem+'.json')).read_text());worker(entry,c,m,b,p)
                records.append((entry,c,m,b));maps+=entry['result']['validation']['maps'];windows+=entry['result']['validation']['windows']
                job=json.loads((out/'jobs'/(stem+'.json')).read_text())
                require('truth' not in job and 'warp_roi' not in job,'truth leaked to worker')
                require(job.get('source_crop')==c['job'].get('source_crop'),'crop changed')
                for field in ('source','template'):
                    require(Path(job[field]).name==Path(c['job'][field]).name,'job input changed')
    require(expected=={f.name for f in (out/'raw').glob('*.json')},'raw inventory differs')
    # Load the archived reporting implementation, so a later edit cannot
    # silently alter the meaning of the recorded gates.
    spec=importlib.util.spec_from_file_location('frozen_adaptive_report',out/'source/cpp-core/tools/benchmark_adaptive_ncc.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='adaptive-report-') as temp:
        folder=Path(temp);(folder/'protocol.json').symlink_to(out/'protocol.json');(folder/'raw').symlink_to(out/'raw')
        with contextlib.redirect_stdout(io.StringIO()):module.report(folder)
        require(json.loads((folder/'summary.json').read_text())==json.loads((out/'summary.json').read_text()),'summary does not reconstruct')
    # Deliberate negative controls verify that native success flags cannot hide
    # a false rejection, score mismatch, bad accounting or wrong route.
    original,c,m,b=next(v for v in records if v[2]=='global16')
    mutations=[('false_rejects',lambda r:r['validation'].__setitem__('false_rejects',1)),
        ('score_mismatch',lambda r:r['validation'].__setitem__('survivor_bit_mismatches',1)),
        ('window_count',lambda r:r['statistics'].__setitem__('windows',r['statistics']['windows']+1)),
        ('timed_stats',lambda r:r.__setitem__('timed_calls_collect_stats',True)),
        ('missing_sample',lambda r:r['samples_ms'].pop()),
        ('wrong_cap',lambda r:r.__setitem__('cap',64)),
        ('nonfinite_time',lambda r:r['samples_ms'].__setitem__(0,float('nan')))]
    controls=[]
    for name,mutate in mutations:
        bad=copy.deepcopy(original);mutate(bad['result'])
        try:worker(bad,c,m,b,p)
        except ValueError:controls.append(name)
        else:raise ValueError('negative control accepted '+name)
    summary=json.loads((out/'summary.json').read_text())
    return dict(valid=True,workers=len(records),formal_calls=len(records)*p['repeats'],diagnostic_maps=maps,diagnostic_windows=windows,
        false_rejects=0,survivor_bit_mismatches=0,summary_reconstructed=True,hashes_verified=len(p['source_sha256'])+len(p['input_sha256']),negative_controls_rejected=controls,
        full_gt_pass=summary['all_synthetic_gt'],time_tolerance_failures=len(summary['time_tolerance_failures']),memory_tolerance_failures=len(summary['memory_tolerance_failures']))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('study',type=Path);args=parser.parse_args()
    result=validate(args.study.resolve());save(args.study/'validation.json',result);print(json.dumps(result,indent=2))
