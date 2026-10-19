"""Run bounded persistent native sessions, streaming current RSS and exactness.

Requires completed native benchmark report, which holds GT outside the runner.
Does not claim leak freedom or production-duration endurance from a short soak.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import benchmark_native_pipeline as bench

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--case',default='many128_pose');p.add_argument('--seconds',type=float,default=600)
    p.add_argument('--build',type=Path,default=bench.ROOT/'build/gui-integration');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=json.loads(a.report.read_text());assert report['complete'] and report['unchanged'] and report['summary']['correctness_passed']
    frozen={k:v for group in ('source_sha256','binary_sha256','input_sha256') for k,v in report[group].items()}
    assert all(bench.sha(k)==v for k,v in frozen.items()), 'Formal inputs or implementation changed before soak'
    case=next(c for c in report['cases'] if c['name']==a.case)
    dest=a.output.resolve().parent;dest.mkdir(parents=True,exist_ok=True)
    job=bench.native_job(case,'native_combined',5,1.);job.update(soak_seconds=a.seconds,stream_path=str(dest/(a.output.stem+'.jsonl')))
    jp=dest/(a.output.stem+'.job.json');bench.save(jp,job)
    runner=a.build/'experiments/fpm_native_pipeline_runner';inputs=dict(frozen)
    assert str(runner.resolve()) in report['binary_sha256'], 'Runner was not validated by formal benchmark'
    inputs[str(a.report.resolve())]=bench.sha(a.report)
    print('Starting native combined soak: '+a.case,flush=True)
    child=subprocess.run([str(runner),str(jp)],capture_output=True,text=True)
    (dest/(a.output.stem+'.stderr.txt')).write_text(child.stderr)
    if child.returncode:raise RuntimeError(child.stderr)
    raw=json.loads(child.stdout);bench.save(dest/(a.output.stem+'.raw.json'),raw)
    rows=[json.loads(line) for line in Path(job['stream_path']).read_text().splitlines()]
    capacities=('frame_host_bytes','frame_coarse_gpu_bytes','frame_fine_gpu_bytes','pose_host_bytes','pose_gpu_bytes')
    retained=all(all(r[k]==rows[0][k] for k in capacities) for r in rows)
    gen=True;last_context=None;previous=None
    for r in rows:
        if r['context_index']==last_context:gen &= r['generation']==previous+1
        else:gen &= r['generation']==1 or last_context is None
        last_context=r['context_index'];previous=r['generation']
    work=all([{k:w[k] for k in ('maps','windows','fine_maps')} for w in r['work']]==[{k:w[k] for k in ('maps','windows','fine_maps')} for w in rows[0]['work']] for r in rows)
    rss=[r['memory']['current_rss_bytes'] for r in rows];n=max(1,len(rss)//5)
    # Trend is a bounded observation, not a statistical leak proof. Context
    # recreation may let the driver retain pooled allocations for later work.
    first=statistics.median(rss[:n]);last=statistics.median(rss[-n:]);allowed=max(32*2**20,first*.1)
    ref_entry=next(e for e in report['execution_order'] if e['case']==a.case and e['method']=='native_combined')
    ref=json.loads((a.report.parent/ref_entry['result']).read_text())['first']
    reference_exact=raw['first']['matches']==ref['matches'] and raw['first']['proposals']==ref['proposals']
    d=dict(case=a.case,seconds_requested=a.seconds,calls=raw['calls'],duration_ms=raw['duration_ms'],contexts_recreated=raw['contexts_recreated'],
           all_outputs_exact=raw['exact'] and all(r['exact'] for r in rows) and reference_exact,
           async_drained=raw['drained'],generation_valid=gen,logical_work_constant=work,retained_capacity_constant=retained,
           current_rss_bytes=dict(min=min(rss),max=max(rss),first_fifth_median=first,last_fifth_median=last,median_delta=last-first),
           bounded_rss_gate=dict(allowed_growth_bytes=allowed,passed=last-first<=allowed),
           timing_ms=bench.dist([r['total_ms'] for r in rows]),sha256=dict(inputs,**{str(jp):bench.sha(jp),job['stream_path']:bench.sha(job['stream_path'])}),
           note='RSS includes Qt/image/driver; includes periodic context recreation. 10min/2min is bounded stability evidence, not leak-free or24h certification.')
    d['passed']=all(d[k] for k in ('all_outputs_exact','async_drained','generation_valid','logical_work_constant','retained_capacity_constant')) and d['bounded_rss_gate']['passed'] and all(bench.sha(k)==h for k,h in inputs.items())
    bench.save(a.output,d);print(json.dumps({k:d[k] for k in ['case','calls','contexts_recreated','passed','current_rss_bytes']},indent=2))
    if not d['passed']:raise SystemExit(1)
if __name__=='__main__':main()
