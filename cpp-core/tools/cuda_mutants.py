"""Compiled mutants of the CUDA backend (stdlib only; run where nvcc and the host compiler work).

python cuda_mutants.py BUILD_DIR SOURCE_ROOT LOG_JSON [LABELS]
LABELS optionally restricts the run to comma-separated label prefixes (e.g. W1,C1). BUILD_DIR is a configured FPM_WITH_CUDA build of SOURCE_ROOT/cpp-core. Each mutant replaces the
`occurrence`-th copy of an anchor (counted from 1; the anchor must occur exactly `total` times),
rebuilds fpm_cuda_kernel_test, fpm_cuda_backend_test and fpm_core_test, runs the tests its file
can affect, and restores the file. KILLED: a test fails or times out. SURVIVED: all pass.
INVALID: it does not build. ANCHOR: the anchor count is not `total`. The unmutated tree runs
first and the restored tree last; both must pass.
"""
import json
import subprocess
import sys
from pathlib import Path

build, root, log = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
only = sys.argv[4].split(',') if len(sys.argv) > 4 else None
CU, NCC = 'cpp-core/experiments/coarse_cuda.cu', 'cpp-core/src/ncc.cpp'
# (label, file, anchor, replacement, occurrence, total)
MUTANTS = [
    ('W1 warp sx re-associated: ax+(c*x-s*y)', CU,
     'const double sx = __dsub_rn(__dadd_rn(m.ax, __dmul_rn(m.c, fx)), __dmul_rn(m.s, fy));',
     'const double sx = __dadd_rn(m.ax, __dsub_rn(__dmul_rn(m.c, fx), __dmul_rn(m.s, fy)));', 1, 1),
    ('W2 warp sy contracted into an FMA', CU,
     'const double sy = __dadd_rn(__dadd_rn(m.ay, __dmul_rn(m.s, fx)), __dmul_rn(m.c, fy));',
     'const double sy = fma(m.c, fy, __dadd_rn(m.ay, __dmul_rn(m.s, fx)));', 1, 1),
    ('W3 warp value not rounded (+0.5 dropped)', CU,
     'floor(__dadd_rn(__dadd_rn(__dmul_rn(top, ry), __dmul_rn(bottom, dy)), 0.5))',
     'floor(__dadd_rn(__dmul_rn(top, ry), __dmul_rn(bottom, dy)))', 1, 1),
    ('W4 warp inside bound w-1 instead of w', CU,
     'if (sx < -1 || sy < -1 || sx >= w || sy >= h) return border;',
     'if (sx < -1 || sy < -1 || sx >= w - 1 || sy >= h) return border;', 1, 1),
    ('N1 normalization: no 0.5 cap on the variance guard', CU,
     'const double limit = bound < 0.5 ? bound : 0.5;', 'const double limit = bound;', 1, 1),
    ('N2 normalization: saturation band 1.25 instead of 1.125', CU,
     '__dmul_rn(denominator, 1.125)', '__dmul_rn(denominator, 1.25)', 1, 1),
    ('N3 normalization: s*(s/area) instead of (s*s)/area', CU,
     '__ddiv_rn(__dmul_rn(s, s), area)', '__dmul_rn(s, __ddiv_rn(s, area))', 1, 1),
    ('C1 fused coarse keeps only scores above the threshold', CU,
     'keep = !(score < threshold);', 'keep = score > threshold;', 1, 2),
    ('C2 fused coarse tile one row short', CU,
     'const int tileW = TILE_X + tw - 1, tileH = TILE_Y + th - 1, tid',
     'const int tileW = TILE_X + tw - 1, tileH = TILE_Y + th - 2, tid', 1, 1),
    ('C3 compaction lane offset counts its own lane', CU,
     'entries[base + __popc(mask & ((1u << lane) - 1))]', 'entries[base + __popc(mask & ((1u << lane)))]', 1, 1),
    ('C4 pack drops the last kept entry', CU,
     'for (unsigned i = threadIdx.x; i < n; i += blockDim.x)', 'for (unsigned i = threadIdx.x; i + 1 < n; i += blockDim.x)', 1, 1),
    ('C5 coarse uniform template not detected', CU,
     'const bool uniform = templ.norm * templ.norm / (double(tw) * th) < DBL_EPSILON;', 'const bool uniform = false;', 1, 3),
    ('C6 unfused coarse canvas warp uses border 0', CU,
     'canvases[canvasOffsets[blockIdx.z] + size_t(y) * job.warp.width + x] = warpPixel(source, sw, sh, x, y, job.warp, border);',
     'canvases[canvasOffsets[blockIdx.z] + size_t(y) * job.warp.width + x] = warpPixel(source, sw, sh, x, y, job.warp, 0);', 1, 1),
    ('F1 fine guard ignored (Q28 result always kept)', CU,
     'if (distance > m.guard) return u8((numerator + half) >> 56);', 'return u8((numerator + half) >> 56);', 1, 1),
    ('F2 fine repairAll ignored', CU, 'if (!m.repairAll) {', 'if (true) {', 1, 1),
    ('F3 fine FP64 fallback samples a border of 1', CU,
     'return warpPixel(s, w, h, x, y, m.exact, 0);', 'return warpPixel(s, w, h, x, y, m.exact, 1);', 1, 1),
    ('F4 fine dot: dx=3 shifted by two bytes', CU,
     'a3 = __dp4a(__funnelshift_r(w0, w1, 24), tv, a3);', 'a3 = __dp4a(__funnelshift_r(w0, w1, 16), tv, a3);', 1, 1),
    ('F5 fine row squares not slid', CU,
     'sum += in - gone; square += in * in - gone * gone;', 'sum += in - gone; square += in * in;', 1, 1),
    ('F6 fine stripe one ROI row short', CU,
     'rows = min(stripeRows, th - r0), roiRows = rows + 2 * PAD;', 'rows = min(stripeRows, th - r0), roiRows = rows + 2 * PAD - 1;', 1, 1),
    ('F7 fine window sums skip the last stripe row', CU,
     'for (int r = 0; r < rows; ++r) {\n            sum += rowStats', 'for (int r = 0; r + 1 < rows; ++r) {\n            sum += rowStats', 1, 1),
    ('F8 fine template padding not zero', CU,
     't[i] = x < tw ? templ[size_t(r0 + y) * tw + x] : 0;', 't[i] = x < tw ? templ[size_t(r0 + y) * tw + x] : 1;', 1, 1),
    ('P1 pyramid rounding bias 127', CU, 'out[size_t(y) * ow + x] = u8((total + 128) / 256);',
     'out[size_t(y) * ow + x] = u8((total + 127) / 256);', 1, 1),
    ('P2 pyramid reflection off by one', CU, 'while (p < 0 || p >= n) p = p < 0 ? -p : 2 * n - p - 2;',
     'while (p < 0 || p >= n) p = p < 0 ? -p - 1 : 2 * n - p - 2;', 1, 1),
    ('S1 sparse peaks: suppression right edge inclusive', NCC,
     'return p.x >= r.x0 && p.x < r.x1 && p.y >= r.y0 && p.y < r.y1; }))', 'return p.x >= r.x0 && p.x <= r.x1 && p.y >= r.y0 && p.y < r.y1; }))', 1, 1),
]
TARGETS = ['fpm_cuda_kernel_test', 'fpm_cuda_backend_test', 'fpm_core_test']
EXE = '.exe' if sys.platform == 'win32' else ''


def run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors='replace')


def tests(rel):
    names = ['experiments/fpm_cuda_kernel_test', 'experiments/fpm_cuda_backend_test']
    if rel == NCC:
        names.append('fpm_core_test')
    return names


def check(rel):
    built = run(['cmake', '--build', str(build), '--target', *TARGETS], 1800)
    if built.returncode:
        return 'INVALID', (built.stdout + built.stderr)[-400:]
    for name in tests(rel):
        try:
            done = run([str(build / (name + EXE))], 900)
        except subprocess.TimeoutExpired:
            return 'KILLED', f'{name}: timeout'
        if done.returncode:
            lines = [x.strip() for x in (done.stdout + done.stderr).splitlines() if 'FAIL' in x]
            return 'KILLED', f'{name} exit {done.returncode}: ' + ' || '.join(lines)[:500]
    return 'SURVIVED', ''


results = []
for rel in (CU, NCC):
    status = check(rel)
    results.append(dict(mutant=f'baseline {rel}', status=status[0], detail=status[1]))
    print('baseline', rel, status, flush=True)
    if status[0] != 'SURVIVED':
        sys.exit('baseline fails')
for label, rel, old, new, occurrence, total in MUTANTS:
    if only and label.split()[0] not in only:
        continue
    path = root / rel
    original = path.read_bytes()  # bytes: no newline translation on Windows
    text = original.decode('utf-8')
    if text.count(old) != total:
        results.append(dict(mutant=label, status='ANCHOR', detail=f'{text.count(old)} copies'))
        print('ANCHOR', label, flush=True)
        continue
    parts = text.split(old)
    mutated = old.join(parts[:occurrence]) + new + old.join(parts[occurrence:])
    path.write_bytes(mutated.encode('utf-8'))
    try:
        status = check(rel)
    finally:
        path.write_bytes(original)
    results.append(dict(mutant=label, status=status[0], detail=status[1]))
    print(f'{status[0]:8s} {label} | {status[1][:200]}', flush=True)
for rel in (CU, NCC):
    status = check(rel)
    results.append(dict(mutant=f'restored {rel}', status=status[0], detail=status[1]))
counts = {}
for r in results:
    if not r['mutant'].startswith(('baseline', 'restored')):
        counts[r['status']] = counts.get(r['status'], 0) + 1
log.write_text(json.dumps(dict(results=results, totals=counts), indent=1))
print('totals', counts)
