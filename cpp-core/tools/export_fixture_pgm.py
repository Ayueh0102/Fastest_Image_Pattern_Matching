"""Write the 17 fixture images as 8-bit P5 PGMs decoded like fpm_cli.

python-port/venv/bin/python cpp-core/tools/export_fixture_pgm.py OUT_DIR
Each image is read with OpenCV (color, orientation ignored) and converted with
fpm::grayFromRgb, (77R + 150G + 29B + 128) >> 8, exactly as check_backend_identity.py
does. A pure-core fpm_cli (no OpenCV) reads these files directly. OUT_DIR/manifest.json
lists every file with its source image and SHA-256.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cpp-core/tools"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fixture_cases import CASES, pgm_path, read_pgm  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in sorted({n for case in CASES for n in case[:2]}):
        bgr = cv2.imread(str(ROOT / "Test Images" / name), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if bgr is None:
            raise RuntimeError(f"missing fixture: {name}")
        b, g, r = (bgr[..., i].astype(np.uint32) for i in range(3))
        gray = np.ascontiguousarray(((77 * r + 150 * g + 29 * b + 128) >> 8).astype(np.uint8))
        path = pgm_path(out, name)
        path.write_bytes(b"P5\n%d %d\n255\n" % (gray.shape[1], gray.shape[0]) + gray.tobytes())
        if read_pgm(path)[2] != gray.tobytes():
            raise RuntimeError(f"{path}: read-back mismatch")
        manifest[path.name] = {"source": name, "width": gray.shape[1], "height": gray.shape[0],
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote", len(manifest), "images to", out)


if __name__ == "__main__":
    main()
