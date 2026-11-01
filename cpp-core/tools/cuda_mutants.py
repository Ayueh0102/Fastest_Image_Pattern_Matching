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
     'if (sx < -1 || sy < -1 || sx >= w - 1 || sy >= h) return border;', 1, 2),
    ('N1 normalization: no 0.5 cap on the variance guard', CU,
     'const double limit = bound < 0.5 ? bound : 0.5;', 'const double limit = bound;', 1, 1),
    ('N2 normalization: saturation band 1.25 instead of 1.125', CU,
     '__dmul_rn(denominator, 1.125)', '__dmul_rn(denominator, 1.25)', 1, 1),
    ('N3 normalization: s*(s/area) instead of (s*s)/area', CU,
     '__ddiv_rn(__dmul_rn(s, s), area)', '__dmul_rn(s, __ddiv_rn(s, area))', 1, 1),
    ('C1 fused coarse keeps only scores above the threshold', CU,
     'keep = !(score < threshold);', 'keep = score > threshold;', 1, 6),
    ('C2 fused coarse tile one row short', CU,
     'const int tileW = TILE_X + tw - 1, tileH = TILE_Y + th - 1, tid',
     'const int tileW = TILE_X + tw - 1, tileH = TILE_Y + th - 2, tid', 1, 2),
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
    # Transfer modes (lean prepare, lean calls).
    ('T1 lean coarse counts not cleared', CU,
     'if (threadIdx.x == 0) counts[blockIdx.x] = 0;', '', 1, 1),
    ('T2 lean fine statistics not cleared', CU,
     'if (i < windows) acc[3 * i] = acc[3 * i + 1] = acc[3 * i + 2] = 0;', '', 1, 1),
    ('T3 lean FP64-pixel count not cleared', CU, 'if (i == 0) *repaired = 0;', '', 1, 1),
    ('T4 lean completion signalled by the first block', CU,
     'atomicAdd(done, 1u) == gridDim.x * gridDim.y * gridDim.z - 1', 'atomicAdd(done, 1u) == 0', 1, 1),
    ('T5 lean upload drops the last row of each chunk', CU,
     'check(cudaMemcpyAsync(device + offset, host + offset, size_t(rows) * width, cudaMemcpyHostToDevice, stream_),',
     'check(cudaMemcpyAsync(device + offset, host + offset, size_t(rows - 1) * width, cudaMemcpyHostToDevice, stream_),', 1, 1),
    ('T6 lean copy ignores inversion', CU,
     'if (task.invert) for (int x = 0; x < task.width; ++x) out[x] = u8(255 - in[x]);',
     'if (false) for (int x = 0; x < task.width; ++x) out[x] = u8(255 - in[x]);', 1, 1),
    ('T7 lean prepare ignores a change of height', CU,
     'same = levels[i].width() == shape[i].width && levels[i].height() == shape[i].height;',
     'same = levels[i].width() == shape[i].width;', 1, 1),
    ('T8 lean prepare never copies levels back for the CPU fine path', CU,
     'if (!fineBatchSize_) {', 'if (false) {', 1, 1),
    ('T9 fine block loads the first request', CU, 'request = requests[blockIdx.y];', 'request = requests[0];', 1, 1),
    ('T10 mode change keeps the lean counters marked clean', CU, 'leanCountsZeroed_ = leanAccZeroed_ = 0;', '', 1, 1),
    # Ordering and fresh-allocation mutants: a race or uninitialized memory that
    # the tests need not observe; reported, not gated (protocol).
    ('U1 lean signal without the system fence', CU,
     '    __threadfence_system();\n    __syncthreads();', '    __syncthreads();', 1, 1),
    ('U2 lean prepare does not wait for the previous upload', CU, '        waitSourceRead();', '', 1, 1),
    ('U3 lean coarse counts not zeroed when allocated', CU,
     'if (leanCountsZeroed_ != leanCounts_.capacity()) {', 'if (false) {', 1, 1),
    ('U4 lean fine statistics not zeroed when allocated', CU,
     'if (leanCalls_ && leanAccZeroed_ != fineAcc_.capacity()) {', 'if (false) {', 1, 1),
    # Kernel modes (coarse NCC 2-3, fine NCC 2, copy threads).
    ('K1 Q28 interior path reads past the row end', CU,
     'if (ix >= 0 && iy >= 0 && ix + 1 < w && iy + 1 < h) {', 'if (ix >= 0 && iy >= 0 && ix + 1 <= w && iy + 1 < h) {', 1, 1),
    ('K2 Q28 bilinear weights the bottom row by dx', CU,
     'numerator = (u64(top) << 28) + u64(bottom - top) * u64(dy);',
     'numerator = (u64(top) << 28) + u64(bottom - top) * u64(dx);', 1, 1),
    ('K3 Q28 sample ignores the guard', CU,
     'if (distance > guard) return u8((numerator + half) >> 56);', 'return u8((numerator + half) >> 56);', 1, 1),
    ('K4 Q28 sample outside the image is 0, not the border', CU,
     'u64 numerator = u64(border) << 56;', 'u64 numerator = 0;', 1, 1),
    ('K5 coarse Q28 tile steps 31 columns', CU,
     'const long long stepX = 32 * q.c, stepY = 32 * q.s;', 'const long long stepX = 31 * q.c, stepY = 32 * q.s;', 1, 1),
    ('K6 screen drops windows up to the threshold plus the margin', CU,
     'const double cut = threshold - SCREEN_MARGIN;', 'const double cut = threshold + SCREEN_MARGIN;', 1, 1),
    ('K7 fine mode 2 steps rows with the cosine twice', CU,
     'const long long stepX = 32 * m.c, stepY = 32 * m.s;', 'const long long stepX = 32 * m.c, stepY = 32 * m.c;', 1, 1),
    ('K8 fine mode 2 template padding not zero', CU,
     'for (int x = lane; x < tpitch; x += 32) out[x] = x < tw ? in[x] : 0;',
     'for (int x = lane; x < tpitch; x += 32) out[x] = x < tw ? in[x] : 1;', 1, 1),
    ('K9 copy thread shares skip their last row', CU,
     'copyRows(task_, y0 + share * rows / n, y0 + (share + 1) * rows / n);',
     'copyRows(task_, y0 + share * rows / n, y0 + (share + 1) * rows / n - 1);', 1, 1),
    ('K10 coarse kernel counters not copied to the host', CU,
     'if (counter) hostCounts[gridDim.x + threadIdx.x] = counts[gridDim.x + threadIdx.x];', '', 1, 1),
    ('K11 coarse Q28 uniform template keeps only scores above the threshold', CU,
     'keep = !(score < threshold);', 'keep = score > threshold;', 3, 6),
    ('K12 coarse Q28 binary64 branch keeps only scores above the threshold', CU,
     'keep = !(score < threshold);', 'keep = score > threshold;', 6, 6),
    # Reported, not gated: a degenerate-variance window screened in binary32 needs a
    # near-zero but nonzero variance and a threshold at or below its score; a missing
    # wait for the copy threads is a race.
    ('K13 screen gives zero-variance windows a nonzero score', CU,
     'score = 0;\n                keep = !(score < threshold);', 'score = 1e-300;\n                keep = !(score < threshold);', 1, 1),
    ('U5 screen ignores the zero-variance rule', CU,
     'score32(sum, square, product, n, stats, 4.0f, s32)', 'score32(sum, square, product, n, stats, 0.0f, s32)', 1, 1),
    ('U6 upload queued before the copy threads finish the chunk', CU,
     'while (!copyPool_.chunkDone(c)) {}', '', 1, 1),
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
