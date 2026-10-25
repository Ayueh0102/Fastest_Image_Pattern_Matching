"""Pilot interleaved timing of several fpm_cli builds (scratch; selection only)."""
import json
import statistics
import subprocess
import sys
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
builds = dict(item.split("=", 1) for item in sys.argv[3:])
sys.path.insert(0, str(root / "cpp-core/tools"))
from compare_versions import CASES  # noqa: E402

pick = {"Src4.bmp", "Src7.bmp", "Src1.bmp", "Src5-0.bmp", "Src3.bmp", "Src8.bmp", "Src10.bmp", "Src2.bmp", "Src6.jpg"}
names = list(builds)
rounds = 8
report = {}
for src, tmpl, count, score, angle, overlap, area in CASES:
    if src not in pick:
        continue
    options = ["--src", str(root / "Test Images" / src), "--tmpl", str(root / "Test Images" / tmpl),
               "--target-num", str(count), "--score", str(score), "--angle", str(angle),
               "--overlap", str(overlap), "--min-area", str(area)]
    times = {n: [] for n in names}
    ref = None
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for n in order:
            res = json.loads(subprocess.check_output([builds[n] + "/fpm_cli", *options], text=True))
            if ref is None:
                ref = res["matches"]
            assert res["matches"] == ref, (src, n)
            if r:
                times[n].append(res["match_ms"])
    med = {n: statistics.median(v) for n, v in times.items()}
    report[src] = {"times_ms": times, "median_ms": med}
    base = med[names[0]]
    print(f"{src:12s} " + " ".join(f"{n}={m:8.3f}({100 * (m / base - 1):+5.1f}%)" for n, m in med.items()), flush=True)
out.write_text(json.dumps(report, indent=2) + "\n")
