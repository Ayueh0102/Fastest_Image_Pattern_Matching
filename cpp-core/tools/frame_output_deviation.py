"""How far an arm's matches are from a reference arm's in benchmark_frame_backend.py output (stdlib only).

python cpp-core/tools/frame_output_deviation.py timing.json --reference cpu --arm fp32 --output deviation.json

Per (round, case, mode) run: a run is identical when every field's bits agree. A reference match is
"found" when an unused arm match has its center within 1 pixel and its angle within 1 degree (each
reference match takes the nearest such arm match by center); for the found pairs the largest absolute
differences of the center, the corners, the angle (degrees, wrapped to [-180, 180)) and the score are
recorded, so a changed order of tied matches is reported as order_changed, not as a distance.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def decode(match: list[str]) -> list[float]:
    return [struct.unpack(">d", bytes.fromhex(h))[0] for h in match]


def angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("timing", type=Path)
    parser.add_argument("--reference", default="cpu")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.timing.read_text())
    runs = {}
    for r in report["records"]:
        if "error" not in r:
            runs[(r["round"], r["case"], r["mode"], r["arm"])] = r
    rows, worst = [], {"center_px": 0.0, "corner_px": 0.0, "angle_deg": 0.0, "score": 0.0}  # over found pairs
    totals = {"runs": 0, "identical_runs": 0, "same_count_runs": 0, "different_count_runs": 0,
              "reference_matches": 0, "matches_found": 0, "extra_matches": 0}
    for (rnd, case, mode, arm), r in sorted(runs.items()):
        if arm != args.arm or (rnd, case, mode, args.reference) not in runs:
            continue
        ref = runs[(rnd, case, mode, args.reference)]
        a, b = [decode(m) for m in ref["matches"]], [decode(m) for m in r["matches"]]
        totals["runs"] += 1
        totals["reference_matches"] += len(a)
        pairs, used = [], set()
        for i, m in enumerate(a):
            near = [(abs(m[8] - n[8]) ** 2 + abs(m[9] - n[9]) ** 2, j) for j, n in enumerate(b)
                    if j not in used and abs(m[8] - n[8]) <= 1 and abs(m[9] - n[9]) <= 1 and angle_diff(m[10], n[10]) <= 1]
            if near:
                j = min(near)[1]
                used.add(j)
                pairs.append((i, j))
        found = len(pairs)
        totals["matches_found"] += found
        totals["extra_matches"] += len(b) - found
        row = {"round": rnd, "case": case, "mode": mode, "reference_count": len(a), "count": len(b), "found": found,
               "identical": ref["matches"] == r["matches"], "order_changed": any(i != j for i, j in pairs)}
        if row["identical"]:
            totals["identical_runs"] += 1
        elif len(a) == len(b):
            totals["same_count_runs"] += 1
        else:
            totals["different_count_runs"] += 1
        totals["order_changed_runs"] = totals.get("order_changed_runs", 0) + row["order_changed"]
        if pairs and not row["identical"]:
            d = {"center_px": max(((a[i][8] - b[j][8]) ** 2 + (a[i][9] - b[j][9]) ** 2) ** 0.5 for i, j in pairs),
                 "corner_px": max(max(abs(a[i][k] - b[j][k]) for k in range(8)) for i, j in pairs),
                 "angle_deg": max(angle_diff(a[i][10], b[j][10]) for i, j in pairs),
                 "score": max(abs(a[i][11] - b[j][11]) for i, j in pairs)}
            row.update(d)
            for k, v in d.items():
                worst[k] = max(worst[k], v)
        rows.append(row)
    out = {"source": args.timing.name, "reference": args.reference, "arm": args.arm, "totals": totals,
           "worst_found_pairs": worst, "runs": rows}
    args.output.write_text(json.dumps(out, indent=1))
    print(json.dumps({"totals": totals, "worst_found_pairs": worst}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
