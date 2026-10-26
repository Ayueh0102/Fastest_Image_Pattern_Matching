"""Round 50 evidence packaging with the round-49 archive functions (deterministic tar.gz, 100 MB cap).

  snapshot COMMIT  source-snapshot.tar.gz + source-sha256.json of the frozen commit
                   (CMakeLists.txt, LICENSE, vcpkg.json, macos-cli GUI sources, cpp-core without docs)
  raw NAME DIR...  one raw-results-NAME.tar.gz per call (manifests, jobs, outputs, stderr)
  archive          archive.json: size and SHA-256 of every evidence file in the output directory
"""
import argparse
import json
from pathlib import Path

import stage49_archive as a49


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
    args = p.parse_args()
    out = args.output.resolve()
    if args.command == 'snapshot':
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
