"""Generate plots, archive all backend study evidence, and verify offline replay."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import hashlib
from benchmark_sparse_metal_front import summarize, verify_workers, unchanged
from benchmark_fixed_scale import ROOT, sha, save


def plots(report,out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=report['summary']['cases']
    labels={'crop_small360':'Small / 360 deg','crop_medium20':'Six objects','crop_lowcontrast20':'Low contrast','thin_detail20':'Thin detail',
        'scale_violation20':'Scale +4% (negative)','repo_src4':'Photo (no GT)','heldout_crowd128_20':'128 objects','large4096_6':'4096 square / 6 objects'}
    fig,axes=plt.subplots(1,2,figsize=(13,6.5));x=np.arange(len(cases))
    for i,(method,label,color) in enumerate([('cpu','All CPU','#416c86'),('metal_cpu','Metal frontend + CPU refinement','#3c947e'),('metal_metal','Metal frontend + Metal refinement','#cb8a3a')]):
        for ax,field,factor in zip(axes,['total_ms','peak_rss_bytes'],[1,1/2**20]):
            values=[]
            for c in cases:
                item=c['methods'][method].get(field)
                values.append((item['median'] if field=='total_ms' else np.median(item))*factor if item else np.nan)
            bars=ax.barh(x+(i-1)*.25,values,.23,label=label,color=color)
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
    for ax in axes:
        ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False);ax.margins(x=.17)
    axes[0].set_yticks(x,[labels[c['name']] for c in cases],fontsize=9);axes[1].set_yticks(x,[])
    axes[0].set_xlabel('Algorithm time (ms, median of 6 measured calls)');axes[1].set_xlabel('Peak process RSS (MiB, median of 2 workers)')
    fig.suptitle('Sparse full-image search | Metal frontend study',fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.6,.94),ncol=3,fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.89]);fig.text(.015,.02,'Identical ordered proposals and fixed scale NCC settings. Source preparation included; setup and first call recorded separately.',fontsize=9)
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'cpp-core/docs/sparse-metal')
    p.add_argument('--pilot',type=Path,action='append',default=[]);p.add_argument('--validation-log',type=Path,action='append',default=[])
    args=p.parse_args();base=args.results.resolve().parent;report=json.loads(args.results.read_text());out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if not report['complete'] or not report['unchanged'] or not unchanged(report['protocol']):raise ValueError('Incomplete or changed study')
    verify_workers(report,base)
    if sha(args.results.with_suffix('.protocol.json'))!=report['protocol_sha256']:raise ValueError('Changed protocol')
    if summarize(report,base)!=report['summary']:raise ValueError('Summary differs from raw records')
    save(out/'results.json',report['summary']);shutil.copyfile(args.results.with_suffix('.protocol.json'),out/'protocol.json')
    plots(report,out)
    with tempfile.TemporaryDirectory(prefix='fpm-sparse-metal-') as tmp:
        folder=Path(tmp)/'archive';shutil.copytree(base,folder/'formal')
        for pilot in args.pilot:shutil.copytree(pilot.resolve(),folder/'pilots'/pilot.name)
        (folder/'validation').mkdir()
        for log in args.validation_log:shutil.copyfile(log,folder/'validation'/log.name)
        validation={k:v for k,v in report['summary'].items() if k!='cases'}
        validation.update(source_binary_input_hashes_unchanged=True,validation_logs={p.name:sha(p) for p in args.validation_log})
        save(folder/'validation.json',validation);save(out/'validation.json',validation)
        (folder/'README.txt').write_text("""Metal sparse frontend study versus frozen ae0489f CPU.
All routes use compact maps; same-build CPU controls runtime changes.
Raw ordered proposals and work must match across CPU/Metal. Final float poses
may differ, so independentGT accuracy is checked separately. Negative scale and
unlabelled photo outcomes are preserved. CPU deterministic candidate selection remains; this is not all-GPU search.

Offline replay with NumPy/OpenCV, no GPU required:
python3 EXTRACT/formal/source-snapshot/cpp-core/tools/benchmark_sparse_metal_front.py --summarize EXTRACT/formal/results.json --output /tmp/replayed.json

Absolute paths are historical provenance; source snapshots, input fixtures,
protocol, raw worker records and validation logs are archived. Binary hashes
are recorded; executables omitted. All member hashes are in sha256.json.
""")
        manifest={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file()};save(folder/'sha256.json',manifest)
        archive=out/'raw-results.tar.gz'
        with archive.open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz:
                with tarfile.open(fileobj=gz,mode='w') as tar:
                    for path in sorted(folder.rglob('*')):
                        if not path.is_file():continue
                        info=tar.gettarinfo(str(path),arcname=str(path.relative_to(folder)));info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
                        with path.open('rb') as data:tar.addfile(info,data)
        relocated=Path(tmp)/'relocated';relocated.mkdir()
        with tarfile.open(archive) as tar:
            if set(tar.getnames())!=set(manifest)|{'sha256.json'}:raise ValueError('Member list differs')
            for name,digest in manifest.items():
                if hashlib.sha256(tar.extractfile(name).read()).hexdigest()!=digest:raise ValueError('Member hash differs')
            tar.extractall(relocated,filter='data')
        replay=relocated/'replayed.json'
        subprocess.run([sys.executable,str(relocated/'formal/source-snapshot/cpp-core/tools/benchmark_sparse_metal_front.py'),
            '--summarize',str(relocated/'formal/results.json'),'--output',str(replay)],cwd='/tmp',check=True)
        if replay.read_bytes()!=(out/'results.json').read_bytes():raise ValueError('Relocated replay differs')
        save(out/'artifact-sha256.json',dict(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,members=len(manifest),
            all_member_hashes_valid=True,relocated_summary_byte_identical=True,summary_sha256=sha(out/'results.json')))
    print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
