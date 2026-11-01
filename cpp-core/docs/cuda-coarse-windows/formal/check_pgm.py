"""Verify every fixture PGM against the manifest written by export_fixture_pgm.py (stdlib)."""
import hashlib, json, sys
from pathlib import Path
manifest, directory = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
bad = [n for n, e in manifest.items() if hashlib.sha256((directory / n).read_bytes()).hexdigest() != e['sha256']]
print(len(manifest), 'files,', len(bad), 'mismatched', bad)
sys.exit(1 if bad else 0)
