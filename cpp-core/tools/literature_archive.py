"""Round 50 evidence packaging with the round-49 archive functions (deterministic tar.gz, 100 MB cap).

  snapshot COMMIT  source-snapshot.tar.gz + source-sha256.json of the frozen commit
                   (CMakeLists.txt, LICENSE, vcpkg.json, macos-cli GUI sources, cpp-core without docs)
  raw NAME DIR...  one raw-results-NAME.tar.gz per call (manifests, jobs, outputs, stderr)
  archive          archive.json: size and SHA-256 of every evidence file in the output directory
  compact SRC DST  copy a run directory, reducing each fpm_native_pipeline_runner output to its matches,
                   call times, memory and the SHA-256 of the full file (full-output-sha256.json lists every
                   original); used for B2 and development runs, whose full outputs exceed the 100 MB cap
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import stage49_archive as a49


def compact(src, dst):
    hashes = {}
    for path in sorted(src.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        data = path.read_bytes()
        hashes[str(rel)] = hashlib.sha256(data).hexdigest()
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        record = None
        if path.suffix == '.json' and not path.name.endswith('.job.json'):
            try:
                record = json.loads(data)
            except ValueError:
                record = None
        if isinstance(record, dict) and 'runs' in record and 'first' in record:
            reduced = dict(compacted='matches, call times, memory; proposals and per-call diagnostics removed',
                           full_sha256=hashes[str(rel)], construct_ms=record.get('construct_ms'), exact_repeat=record.get('exact_repeat'),
                           final_memory=record.get('final_memory'),
                           first=dict(total_ms=record['first'].get('total_ms'), matches=record['first'].get('matches')),
                           runs=[dict(total_ms=r.get('total_ms'), matches=len(r.get('matches', []))) for r in record['runs']])
            target.write_text(json.dumps(reduced) + '\n')
        else:
            shutil.copyfile(path, target)
    (dst / 'full-output-sha256.json').write_text(json.dumps(hashes, indent=0) + '\n')
    return len(hashes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('snapshot')
    s.add_argument('commit')
    r = sub.add_parser('raw')
    r.add_argument('name')
    r.add_argument('directories', type=Path, nargs='+')
    sub.add_parser('archive')
    c = sub.add_parser('compact')
    c.add_argument('src', type=Path)
    c.add_argument('dst', type=Path)
    args = p.parse_args()
    out = args.output.resolve()
    if args.command == 'compact':
        print(compact(args.src.resolve(), args.dst.resolve()), 'files')
    elif args.command == 'snapshot':
        print(a49.snapshot(args.commit, out), 'files')
    elif args.command == 'raw':
        if (out / f'raw-results-{args.name}.tar.gz').exists():
            raise SystemExit('Refusing to overwrite raw-results-' + args.name)
        print(a49.raw([d.resolve() for d in args.directories], out, 'raw-results-' + args.name))
    else:
        a49.archive(out, {})
        record = json.loads((out / 'archive.json').read_text())
        record['stage'] = 50  # the round-49 function writes its own stage number
        (out / 'archive.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
        print(out / 'archive.json')


if __name__ == '__main__':
    main()
