"""The 17 fixed compare_versions.py cases, importable without OpenCV or NumPy.

Each case is (source, template, max_targets, min_score, tolerance_angle, max_overlap, min_area)
for files in `Test Images/`. export_fixture_pgm.py writes 8-bit P5 copies (<stem>.pgm) decoded
like fpm_cli, so the identity and A/B tools can run with the standard library alone (--pgm-dir).
"""
from __future__ import annotations

from pathlib import Path

CASES = [
    ("Src1.bmp", "Dst1.bmp", 5, .8, 180, .0, 256),
    ("Src2.bmp", "Dst2.bmp", 5, .8, 180, .0, 256),
    ("Src3.bmp", "Dst3.bmp", 38, .8, 0, .0, 256),
    ("Src4.bmp", "Dst4.bmp", 5, .8, 180, .0, 256),
    ("Src6.jpg", "Dst6.bmp", 15, .8, 180, .0, 256),
    ("Src7.bmp", "Dst7.bmp", 5, .8, 180, .0, 256),
    ("Src8.bmp", "Dst8.bmp", 5, .8, 180, .8, 256),
    ("Src9.bmp", "Dst9.bmp", 100, .5, 0, .5, 1024),
    ("Src10.bmp", "Dst10.jpg", 100, .8, 0, .0, 256),
] + [(f"Src5-{angle}.bmp", "Dst5.bmp", 1, .8, 180, .0, 256)
     for angle in (0, 45, 90, 135, 180, 225, 270, 315)]


def pgm_path(directory: Path, name: str) -> Path:
    return Path(directory) / (Path(name).stem + ".pgm")


def read_pgm(path: Path) -> tuple[int, int, bytes]:
    """Width, height and tightly packed pixels of an 8-bit binary (P5) PGM."""
    data = Path(path).read_bytes()
    fields, position = [], 0
    while len(fields) < 4:
        while data[position:position + 1].isspace():
            position += 1
        if data[position:position + 1] == b"#":
            position = data.index(b"\n", position) + 1
            continue
        end = position
        while not data[end:end + 1].isspace():
            end += 1
        fields.append(data[position:end])
        position = end
    if fields[0] != b"P5" or int(fields[3]) != 255:
        raise ValueError(f"{path}: expected an 8-bit P5 PGM")
    width, height = int(fields[1]), int(fields[2])
    pixels = data[position + 1:position + 1 + width * height]
    if len(pixels) != width * height:
        raise ValueError(f"{path}: truncated pixel data")
    return width, height, pixels
