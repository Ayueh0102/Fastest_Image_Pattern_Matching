# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rotation-invariant NCC (normalized cross-correlation) template matching. This repo is a fork of DennisLiu1993's Windows MFC/OpenCV tool. The fork's active work is a portable C++17 rewrite plus CPU/Apple Metal research for many-object, large-image, multi-scale matching.

| Directory | Status | Role |
|---|---|---|
| `cpp-core/` | **Active main line** | Dependency-free C++17 core (`fpm_core`), C ABI (`fpm_c`), CLI (`fpm_cli`), Qt6 GUI (`fpm_gui`), opt-in research pipeline in `experiments/` |
| `macos-cli/` | Partly reused | `main.cpp` and `mainwindow.{h,cpp}` are the Qt GUI sources that cpp-core builds. `match_core.{h,cpp}` is only a historical OpenCV/DOTPROD reference for comparison tests |
| `MatchTool/` | Historical | Original MFC + OpenCV implementation; the algorithm reference for porting |
| `python-port/` | Independent | Separate NumPy/OpenCV + Tk version. Its search and NMS definitions differ from cpp-core, so results are not guaranteed equal |

Read these before substantial changes (docs are in Traditional Chinese):
1. `docs/PROJECT_GUIDE_zh-TW.md`: component map and how to read the evidence
2. `cpp-core/docs/PROJECT_STATUS_zh-TW.md`: what is done and the known gaps
3. `cpp-core/docs/CURRENT_BEST_zh-TW.md`: recommended configuration per scenario
4. `cpp-core/docs/EXPERIMENT_REVIEW_zh-TW.md`: index of research rounds 1–40 with adopt/reject reasons

Git: the active branch is `codex/ncc-fourier-gpu-experiments`, built on `codex/portable-cpp-core`. Remote `fork` is Ayueh0102 (push here); `origin` is upstream DennisLiu1993.

## Build and Test (cpp-core)

Run from the repo root. Use Release for anything you time.

```bash
# Pure core: no Qt/OpenCV; fpm_cli then reads 8-bit P5 PGM only (2 tests)
cmake -S cpp-core -B build/core -DCMAKE_BUILD_TYPE=Release
cmake --build build/core -j
ctest --test-dir build/core --output-on-failure

# Full research build: Qt GUI + OpenCV codecs + CPU/Metal experiments (82 tests on macOS)
cmake -S cpp-core -B build/research -DCMAKE_BUILD_TYPE=Release \
  -DFPM_BUILD_GUI=ON -DFPM_BUILD_EXPERIMENTS=ON -DFPM_WITH_OPENCV=ON
cmake --build build/research -j
ctest --test-dir build/research --output-on-failure
ctest --test-dir build/research -R experimental_full_gpu   # subset by regex

# GUI
build/research/fpm_gui.app/Contents/MacOS/fpm_gui --gui --src "Test Images/Src7.bmp" --tmpl "Test Images/Dst7.bmp"

# ASan/UBSan on the scalar path
cmake -S cpp-core -B build/sanitized -DCMAKE_BUILD_TYPE=Debug -DFPM_SANITIZERS=ON -DFPM_DISABLE_SIMD=ON
```

- `cpp-core/CMakeLists.txt` options: `FPM_BUILD_TESTS` (ON), `FPM_WITH_OPENCV`, `FPM_BUILD_GUI`, `FPM_BUILD_EXPERIMENTS`, `FPM_DISABLE_SIMD`, `FPM_SANITIZERS`. The root `CMakeLists.txt` adds `FPM_BUILD_MFC`, which defaults to ON under MSVC.
- On the main development Mac, `build/fourier-gpu` is already configured like the research build.

Python harness and negative-control tests use stdlib `unittest`. The `python-port/venv` interpreter has numpy and opencv:

```bash
FPM_FINE_REUSE_BUILD=build/research FPM_FINE_SEARCH_BUILD=build/research FPM_FINE_FUSED_BUILD=build/research \
  python-port/venv/bin/python -m unittest discover -s cpp-core/tests -p '*.py'
python-port/venv/bin/python cpp-core/tests/color_verify_runner_test.py \
  --runner build/research/experiments/fpm_color_verify_runner --metal
```

- The three `test_fine_*_binding.py` files default to frozen per-round builds (`build/fine-reuse` and similar). Without the env vars, `test_fine_reuse_binding` errors against its stale build.
- `color_verify_runner_test.py` needs `--runner`. Under `discover` its 4 worker tests error with `FileNotFoundError: 'None'`; that is expected, so run the file directly.

Research runner: `build/research/experiments/fpm_native_pipeline_runner job.json` (built on macOS when GUI and experiments are both on). Job keys are the snake_case form of `Options` in `cpp-core/experiments/research_pipeline.hpp`, for example `full_gpu`, `full_gpu_peak_mode`, `coarse_peak_mode`. The `cpp-core/tools/benchmark_*.py` drivers write the jobs, run fresh processes and record protocol and raw results.

## Architecture

### Search algorithm (MFC original and cpp-core)
1. **Learn pattern**: build the template pyramid until the next level would fall below `minReducedArea`; precompute mean, norm and inverse area per level.
2. **Top level**: the angle step is `atan(2 / max(w, h))` of that level's template. For each angle in the tolerance range, rotate the reduced **source** (on a canvas sized to fit) and compute NCC against the upright template. Peak picking switches to the block-partitioned `s_BlockMax` method when the source/template area ratio exceeds 500 and `maxTargets > 10`.
3. **Lower levels**: map each candidate ×2, rotate a small source ROI around it at the candidate angle ±1 step, and keep the best. Continue to level 0, or stop at level 1 when `stopLayer = 1` (legacy FastMode, half resolution).
4. **Subpixel (optional)**: 27-point 3D quadratic fit over x, y and angle, with a 2D fallback for a fixed angle.
5. **NMS**: overlap = rotated-rectangle intersection area / template area (not IoU), compared against `maxOverlap`. Results are sorted by score.

### cpp-core layout
- `include/fpm/matcher.hpp`: public API. `ImageView` (gray8, caller-owned buffer), `Pattern` (learned once, reusable across frames and threads), `MatchConfig`, `fpm::match()`.
- `include/fpm/c_api.h` and `src/c_api.cpp`: stable C ABI. `examples/python_ctypes.py` calls it.
- `src/matcher.cpp` runs the pyramid search; `src/ncc.cpp` holds NCC, the dot products (UDOT when the compiler defines `__ARM_FEATURE_DOTPROD` on aarch64, otherwise NEON or SSE2, plus the scalar reference) and peak selection; `src/image.cpp` holds pyramid levels and rotation canvas/ROI; `src/geometry.cpp` holds rotated-rectangle overlap; `src/subpixel.cpp` holds the quadratic fit.
- `experiments/research_pipeline.{hpp,cpp}`: `Session` owns copies of the source and template, allows one caller at a time, and takes new frames via `updateSource()`. `Options` covers multi-scale search, CPU/Metal backends, continuous pose refinement with bounded rescue, and the full-GPU switches.
- Experiment families in `experiments/`: `coarse_*` (coarse NCC, CPU/Metal), `pose_*` (continuous pose refinement), `full_gpu*` (complete gray8 Metal pipeline, rounds 37–40), `sparse_orientation*` (fixed-scale sparse proposals), `color_verify*` (RGB/chroma verification), `scale_support_refine*`, `adaptive_ncc*`.
- `.metal` shaders are read at configure time into generated `*_source.hpp` headers (templates `*_source.hpp.in`). Editing a shader triggers a reconfigure on the next build.
- `tools/`: native runners (`native_pipeline_runner.cpp`, `color_verify_runner.cpp`, `cli*.cpp`) and the Python benchmark/report/plot drivers. `tests/`: C++/Objective-C++ ctest targets plus the Python harness tests.

### Original MFC code (MatchTool/)
- `MatchToolDlg.h`: `s_TemplData` (template pyramid and stats), `s_MatchParameter` (one match), `s_BlockMax` (block-partitioned peak finder)
- `MatchToolDlg.cpp`: `LearnPattern()` line 453, `Match()` 772, `SubPixEsimation()` 1149, `MatchTemplate()` 1275, `FilterWithRotatedRect()` 1498, `GetNextMaxLoc()` 1558 and 1583 (the `s_BlockMax` overload)
- SIMD: `arm_version.cpp` (Neon), `x86_ubuntu_linux.cpp` (SSE2)

### Mapping
| MFC | cpp-core | python-port |
|---|---|---|
| `LearnPattern()` | `Pattern` constructor, `learnLevel()` (`src/image.cpp`) | inside `pyramid_match()` |
| `Match()` | `fpm::match()` (`src/matcher.cpp`) | `pyramid_match()` |
| `MatchTemplate()` | `src/ncc.cpp` | `ncc_match()` / `ncc_match_fft()` |
| `GetNextMaxLoc()` + `s_BlockMax` | `selectPeaks()` / `selectPeaksFast()` (`src/ncc.cpp`) | `topk_blocked()` |
| `FilterWithRotatedRect()` | `overlap()` (`src/geometry.cpp`) | `nms()` |
| `SubPixEsimation()` | `src/subpixel.cpp` | `refine_peak_subpixel()` |

## Research Workflow and Evidence Rules

Research is organized in numbered rounds (第N輪). A round usually adds:
- a report `cpp-core/docs/<TOPIC>_zh-TW.md`, indexed from `EXPERIMENT_REVIEW_zh-TW.md`, `PROJECT_STATUS_zh-TW.md`, `CURRENT_BEST_zh-TW.md` and `LOG_CATALOG_zh-TW.md`;
- an evidence directory `cpp-core/docs/stageNN-<name>/` with `protocol.json`, `raw-results.tar.gz`, `source-snapshot.tar.gz`, fixtures, `*-summary.json`, `decision-and-failures.json`, `validation.json` and `archive.json` (SHA256 list).

The existing work follows these rules; keep following them:
- Keep failed variants and every gate failure. Never drop slow samples or rerun to get nicer numbers.
- Report time, process RSS, named GPU allocations and accuracy separately. RSS and GPU buffer capacity have different scopes and are never added together.
- "Matches ground truth" and "output identical to the baseline" are separate acceptance checks.
- Don't multiply speedups across rounds; each round compares against its own same-build baseline.
- Formal timings exclude image decoding and `Session` construction and are kept apart from profiling runs. Don't run other builds or experiments while timing GPU work.
- Research results don't change GUI or core defaults; new behavior ships as an opt-in option.
- Reproduce an old round from its frozen `source-snapshot.tar.gz`, not from the current HEAD.

Snapshot as of round 40 (2026-10-19): the full-Metal gray8 stage is closed. The full-GPU research config (`full_gpu=true`, `full_gpu_peak_mode=1`, `full_gpu_fine_mode=4`) roughly ties the CPU/Metal hybrid at 128 objects, is slower at 512 objects and on small scenes, and is much faster on 8K scenes with few objects. Defaults stay on the CPU and hybrid paths. Check `PROJECT_STATUS_zh-TW.md` for anything newer. The 2026-10-19 corrections in the round 37–40 reports add that the hybrid's formal calls reuse a pose context the full-GPU path rebuilds every call (`cpp-core/tools/audit_full_gpu_disclosure.py` recomputes the effect from the archives), and that the 8K scene is a constant-background, upright, scale-1 speed workload. Round 41 (`cpp-core/docs/NEW_FRAME_COST_zh-TW.md`) measured that case directly with every call going through `updateSource`: per new frame, fine4 is 6.9% faster than the hybrid at 128 objects and 13.7% slower at 512, so the archive-only estimate's direction held; it is a measurement round only and changes no defaults. Round 42 (`cpp-core/docs/GPU_PARALLEL_PEAKS_zh-TW.md`) added `full_gpu_peak_mode=2`: exact parallel-round coarse peak selection over batches of score maps (`full_gpu_peak_batch`, default 8), with ordered outputs byte-identical to peak mode 1. With fine4 it cuts full-GPU time by 27.3%/34.4% at 128/512 objects and, per new frame, beats the hybrid at 128 (−31.4%), 512 (−22.7%) and 8K, while the six-object scene stays 1.5% slower; the full-GPU research recommendation is now peak2/fine4, defaults unchanged. Fine search is now about 71% of full-GPU time. Round 43 (`cpp-core/docs/CANDIDATE_DEDUP_zh-TW.md`) adds an opt-in, semantics-changing coarse-candidate dedup before fine search (`candidate_dedup_top_k`, default 0; hybrid and full GPU). With k3 it is GT-neutral on every GT scene family the repo has and cuts 128/512-object time by about 26–37%, but outputs are not bit-identical to the baseline; k1 and the ablations lose objects. Its full-GPU arm was measured on peak1/fine4 (the round-40 job), before round 42 was merged, so the 26–37% figures do not stack with round 42's. Defaults are unchanged. Round 44 (`cpp-core/docs/DEDUP_PEAK2_zh-TW.md`) is a measurement round that runs both together on one build: on full-GPU peak2/fine4, k3 cuts static time by 35.5%/38.9% at 128/512 objects, and per new frame full-GPU peak2+k3 is the fastest of the four combinations (hybrid/full GPU x dedup off/k3) at 128, 512 and six objects; on 8K k3 makes no difference. Every output equals the round-43 archive and no GT case variant regresses; defaults are unchanged. Round 45 (`cpp-core/docs/CPU_PATH_zh-TW.md`, measurement round) adds the full CPU research path (`backend=0`, `refinement_backend=0`, 8 threads, UDOT dot product) per new frame on the same build: it is the slowest arm on every case (3.2 s/10.2 s at 128/512 objects), full-GPU peak2+k3 is 78–89% faster and the hybrid 40–76% faster; k3 is GT-neutral on the CPU and its strict GT counts equal the Metal hybrid's on all 112 GT case variants. Round 46 (`cpp-core/docs/ILLUMINANT_CHROMA_zh-TW.md`) adds the opt-in colour-verify method `chroma_vk`: the template's off-white panel, mapped through each pose, gives per-channel median illuminant estimates for a von Kries gain before masked region chroma; a clipped panel abstains. On test-bed v1 it accepted none of 376 different-colour objects (isogray included), but its pre-registered one-class p95 thresholds were too tight for held-out JPEG, a local warm spot and Blender warm/mixed light (31/88 same-colour accepted), so it failed its gates and is not recommended; the existing chroma, rgb_ncc and gray_ncc outputs are unchanged.

## Coding Conventions

### C++
- C++17. cpp-core uses namespace `fpm`; types are `PascalCase`, functions and members `camelCase`, private members end in `_`, constants are `UPPER_SNAKE_CASE`. C++ sources start with `// SPDX-License-Identifier: BSD-2-Clause`.
- Indentation differs by area, so match the file you are editing. `cpp-core/src`, `cpp-core/include`, `macos-cli` and `MatchTool` use tabs; `cpp-core/experiments` and most newer tools and tests use 4 spaces. There is no clang-format: never mass-reformat, keep diffs surgical.
- The core (`src/`, `include/`) must not depend on Qt or OpenCV. OpenCV is only for CLI codecs and comparison tests; Qt is only for the GUI and the native runners.

### Python
- cpp-core tools are scripts under `cpp-core/tools`, run with the python-port venv; their tests use stdlib `unittest`.
- python-port is formatted with `black` and linted with `ruff`.

## Commit Style
- Concise, imperative, descriptive. Examples: `Optimize exact Metal NCC and close resident GPU experiments`, `Add exact Metal fine ROI row caching and workload counters`, `Fix NCC rotation precision at pyramid top`.
- PRs should include build/run steps, platform and toolchain, result screenshots, and performance deltas with their measurement scope.

## Test Data
- `Test Images/` (Src1–Src10, Dst1–Dst10) and `Result Images/` (upstream screenshots). The boxes in the screenshots are not full ground truth.
- Accuracy studies use generated scenes with independently produced ground truth (`cpp-core/tools/many_object_scenes.py`, `resident_validation_scenes.py`, `stage35_scenes.py`, `blender_challenge_*.py`), plus FSC/VisA real photos in round 35.

## Python Port (python-port/)
```bash
cd python-port
PYTHONPATH=src venv/bin/python -m pytest tests -q   # 13 tests
black src/ && ruff check src/
```
- Entry points `fpm-py` and `fpm-py-gui` (after `pip install -e .[dev]`).
- Key modules: `pyramid.py` (`pyramid_match`), `ncc.py` (`ncc_match`, `ncc_match_fft`, `topk_blocked`), `nms.py`, `subpixel.py`, `gui.py` (Tk).

## Known Quirks and Gotchas
- The root `.gitignore` ignores `*.md` except `CLAUDE.md`, `cpp-core/README_zh-TW.md`, `cpp-core/docs/*.md` and `docs/PROJECT_GUIDE_zh-TW.md`. New Markdown anywhere else needs a whitelist entry or `git add -f`. AGENTS.md is untracked and outdated for cpp-core.
- Evidence archives are large: a single `.tar.gz` reaches about 60 MB and `.git` is about 1.3 GB. GitHub rejects files over 100 MB, so keep each archive below that.
- `build/` (ignored, about 12 GB) holds per-round builds, frozen binaries and results that later replays use. Create a new build directory rather than wiping or reusing those.
- Performance evidence comes from one Apple M4 Pro (48 GiB). Windows/Linux builds have not been run on hardware, and CUDA exists only as a design.
- The stable core's fixed-scale CPU path is still slower than the old OpenCV/DOTPROD build, even after the 2026-10-24 UDOT dot-product kernel (`cpp-core/docs/CORE_DOTPROD_zh-TW.md`; Src7 at the audit config: 122 vs 65 ms, was 138). Most remaining time is in rotation (`warp`), the rest of `computeNcc` and `pyrDown`. Later speedups live in the research pipeline.
- Uniform templates follow the original rule NCC = 1, so they cannot localize anything.
- MFC: build Release and match the OpenCV DLLs to the build configuration. If OpenCV's `RotatedRect` assertion triggers, see the README workaround.
