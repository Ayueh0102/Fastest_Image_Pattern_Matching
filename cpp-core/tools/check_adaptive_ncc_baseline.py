"""Fresh-process public CLI regression check against an isolated build of HEAD.

Separate from the warm adaptive study: the CLI serializes 12 significant digits,
so agreement here is serialized-output agreement, not a bit-exact oracle.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
from benchmark_adaptive_ncc import ROOT, sha, save


def run(study, frozen, current):
    out=study/'control';out.mkdir(parents=True,exist_ok=True)
    cases=[c for c in json.loads((study/'protocol.json').read_text())['cases'] if c['job']['mode']=='full']
    protocol=dict(baseline_commit='3793e9ea7e37b4b2d811084431f25d20439a3c91',
        executables={name:dict(path=str(binary),sha256=sha(binary)) for name,binary in [('frozen',frozen),('current',current)]},
        repeats=3,blocks=2,cases=cases,script_sha256=sha(__file__),
        timing='Three fresh CLI processes per paired block; JSON match_ms excludes I/O and learning. No warm-throughput claim.',
        gate='Same ordered JSON output at CLI 12-digit precision; each block current <= frozen + max(10%,2ms).')
    if (out/'protocol.json').exists():raise ValueError('Control protocol exists; preserve the original evidence')
    save(out/'protocol.json',protocol)
    rows=[];valid=True;gates=True
    for c in cases:
        for block in range(2):
            methods=['frozen','current'] if block==0 else ['current','frozen'];samples={}
            for method in methods:
                samples[method]=[]
                for repeat in range(3):
                    job=c['job'];args=[protocol['executables'][method]['path'],'--src',str(study/job['source']),'--tmpl',str(study/job['template']),
                        '--angle',str(job['angle']),'--score','.8','--overlap','.2','--target-num','2000','--min-area','256','--subpixel']
                    call=subprocess.run(args,capture_output=True,text=True,timeout=240)
                    record=dict(case=c['name'],block=block,method=method,repeat=repeat,args=args,returncode=call.returncode,stderr=call.stderr)
                    try:record['result']=json.loads(call.stdout)
                    except json.JSONDecodeError:record['stdout']=call.stdout
                    save(out/f"{c['name']}_b{block}_{method}_{repeat}.json",record)
                    if call.returncode:raise ValueError('CLI failed; evidence retained')
                    samples[method].append(record['result'])
            reference=samples['frozen'][0]['matches']
            same=all(r['matches']==reference for values in samples.values() for r in values);valid &= same
            old,new=[statistics.median([r['match_ms'] for r in samples[m]]) for m in ['frozen','current']]
            gate=new<=old+max(.1*old,2);gates &= gate
            rows.append(dict(case=c['name'],block=block,frozen_ms=old,current_ms=new,serialized_output_identical=same,time_tolerance=gate))
    result=dict(workers=72,serialized_output_identical=valid,time_tolerance_all=gates,rows=rows)
    save(out/'summary.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--frozen',type=Path,default=ROOT/'build/adaptive-ncc-control/build/fpm_cli');p.add_argument('--current',type=Path,default=ROOT/'build/adaptive-ncc/fpm_cli');a=p.parse_args()
    run(a.study.resolve(),a.frozen.resolve(),a.current.resolve())
