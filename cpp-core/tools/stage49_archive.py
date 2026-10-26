"""Stage 49 evidence packaging: source snapshot, raw archives and the SHA-256 list.

snapshot: committed bytes of the frozen commit (CMakeLists.txt, LICENSE,
vcpkg.json, macos-cli GUI/reference sources and cpp-core without docs; the
stage-40/48 selection), written as a deterministic tar.gz plus source-sha256.json.
raw: one deterministic tar.gz per run directory (manifest, jobs, results,
stdout/stderr); refuses any archive above 100 MB.
archive: archive.json with the size and SHA-256 of every evidence file.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 100*1000*1000
SNAPSHOT_TOP = ('CMakeLists.txt', 'LICENSE', 'vcpkg.json')
SNAPSHOT_MACOS = ('macos-cli/CMakeLists.txt', 'macos-cli/main.cpp', 'macos-cli/mainwindow.cpp', 'macos-cli/mainwindow.h',
                  'macos-cli/match_core.cpp', 'macos-cli/match_core.h')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def snapshot_files(commit):
    names = subprocess.run(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', commit], capture_output=True,
                           text=True, check=True).stdout.splitlines()
    keep = [name for name in names if name in SNAPSHOT_TOP or name in SNAPSHOT_MACOS or
            (name.startswith('cpp-core/') and not name.startswith('cpp-core/docs/'))]
    return {name: subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{name}'], capture_output=True, check=True).stdout
            for name in sorted(keep)}


def deterministic_tar(entries, path):
    """entries: sorted (name, bytes); fixed mtime/owner, gzip without a timestamp."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w', format=tarfile.PAX_FORMAT) as tar:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode, info.uid, info.gid, info.uname, info.gname = len(data), 0, 0o644, 0, 0, '', ''
            tar.addfile(info, io.BytesIO(data))
    with open(path, 'wb') as out, gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0, compresslevel=9) as stream:
        stream.write(raw.getvalue())
    if path.stat().st_size > LIMIT:
        path.unlink()
        raise ValueError(f'{path.name} exceeds 100 MB')
    return path.stat().st_size


def snapshot(commit, output):
    files = snapshot_files(commit)
    full = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', commit], capture_output=True, text=True, check=True).stdout.strip()
    deterministic_tar([(name, data) for name, data in files.items()], output/'source-snapshot.tar.gz')
    (output/'source-sha256.json').write_text(json.dumps(dict(commit=full, files={name: sha(data) for name, data in files.items()}),
                                                        indent=2)+'\n')
    return len(files)


def raw(directories, output, prefix):
    """directories: run directories under one base; members keep their path below the base's parent."""
    entries = []
    for directory in directories:
        base = directory.parent
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                entries.append((str(path.relative_to(base)), path.read_bytes()))
    return deterministic_tar(entries, output/(prefix+'.tar.gz')), len(entries)


def archive(output, extra):
    artifacts = []
    for path in sorted(output.iterdir()):
        if path.name == 'archive.json' or not path.is_file():
            continue
        data = path.read_bytes()
        row = dict(path=path.name, bytes=len(data), sha256=sha(data))
        if path.name.endswith('.tar.gz'):
            with tarfile.open(path) as tar:
                row['regular_files'] = sum(member.isfile() for member in tar.getmembers())
        artifacts.append(row)
    (output/'archive.json').write_text(json.dumps(dict(stage=49, path_base=str(output.relative_to(ROOT)), artifacts=artifacts,
                                                       **extra), indent=2, ensure_ascii=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('snapshot')
    s.add_argument('--commit', required=True)
    r = sub.add_parser('raw')
    r.add_argument('--name', required=True)
    r.add_argument('directories', type=Path, nargs='+')
    a = sub.add_parser('archive')
    a.add_argument('--extra', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.command == 'snapshot':
        print(snapshot(args.commit, out), 'source files')
    elif args.command == 'raw':
        if (out/(args.name+'.tar.gz')).exists():
            raise ValueError('Refusing to overwrite '+args.name)
        print(*raw([d.resolve() for d in args.directories], out, args.name))
    else:
        archive(out, json.loads(args.extra.read_text()) if args.extra else {})


if __name__ == '__main__':
    main()
