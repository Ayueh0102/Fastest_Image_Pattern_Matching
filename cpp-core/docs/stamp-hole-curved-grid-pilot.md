# Stamp-hole curved-grid pilot (2026-11-25)

The two original `stem_hole` sample photographs contain a 10-column, 5-row sheet.
This offline Mac experiment adds two measurements that must not be conflated:

1. **Individual hole to whole curved line:** robust cubic fitted to the visible
   holes across a line, followed by a closest-point normal-distance calculation.
   Horizontal positive means down; vertical positive means right. This differs
   from the earlier 12-neighbor residual and can give slightly different values.
2. **A whole line relative to the good sheet:** a smooth degree-3 coordinate map
   aligns the good photograph to the bad photograph. Every point belonging to
   the line under inspection, including intersections, is excluded from fitting
   that map. Mutual nearest-center correspondences, limited to 0.45 hole pitches,
   provide normal and along-line offsets. The median normal offset summarizes a
   whole line; per-stamp-edge medians expose coherent segments.

These are empirical image measurements, **not calibrated lens distortion or
physical millimeters**. Smooth global damage, sheet curvature, and camera geometry
are not independently identifiable from this pair. Fixed low model order and
line exclusion reduce absorption of local defects but do not resolve that
ambiguity. A camera calibration and physical size reference are needed for
calibrated geometry. See the [OpenCV calibration tutorial](https://docs.opencv.org/4.12.0/dc/dbb/tutorial_py_calibration.html).

## Inputs and scope

- Image 1, bad: Git blob `1b83551f9b610c93d3ebedf20c7df79fdca05870`.
- Image 2, good: Git blob `d37b4ffca2dbdfc515c9cc50d30502dac6e48197`.
- Both are 4504 x 4096 originals from the source repository's
  [sample directory](https://github.com/Ayueh0102/stem_hole/tree/ecf0d84edfb0953d24a93a18037f4ad64cd77f01/sample).
- The 19 x 19 template is the earlier pilot's real single-hole green-channel crop.
- Defaults (16 horizontal intervals per stamp, 20 vertical, 8 top-margin
  intervals, radius 3–8 px) are specific to these photographs. They are not
  advertised as a general detector or a production acceptance threshold.
- No new GPU execution, camera control, cloud upload, or GUI integration is used.

## What changed in performance

The earlier contour loop calculated a minimum enclosing circle before rejecting
large contours. Background contours made this very expensive. A necessary
bounding-box condition now runs first: radius < 8 implies integer bounding width
and height <= 16. This preserves the candidates exactly on both original images.
The native FPM template matcher is unchanged. Ring/core contrast and nearest
contour checks are batched rather than dispatched once per hole.

Final fast run: five timed repetitions after one warmup. Both photographs plus
the new whole-line analysis: **0.727 s median**, range **0.720–0.741 s**. The sum
of in-memory detection and geometry stages: **0.362 s median**. These exclude
Python imports, camera acquisition, display, and report generation. PNG reading
and decoding are included in the first number. The original workflow measured
7.489 s on the previous turn; a fresh legacy-contour comparison is saved
separately. This is roughly a tenfold improvement, not proof of optimal speed.
The in-memory figure is a sum of measured stages, not a live-camera FPS claim.

Both fast and legacy runs reproduce all 1,908 / 2,023 circular candidate records
and all 2,099 raw FPM observation records per image from the previous pilot,
including scores, contrasts, and missing values. The original loose acceptance
counts remain 2,051 / 2,050. Geometry applies additional support requirements;
these counts do not imply complete recall or verified real-hole ground truth.

## Results and limits

- All 17 lines' measured median normal offsets after registration are between
  approximately -0.23 and +0.58 px. These small values are near the quantized
  center/model residual scale, not evidence of subpixel metrological accuracy.
- V6 at pitch index 11: **+7.14 px** normal distance to the full curved line.
- H5 at index 72: **-6.69 px**; H3 at index 57: **-6.29 px**.
- The bottom H6 edge is only supported for 112 positions in both photographs.
  An isolated corner beyond obscured holes must not be reported as a 31 px
  defect from curve extrapolation. It is excluded from line measurement.
- V11 has ten re-associated provisional pitch indices near the bottom. The
  previous review's #13 (index 92) and #14 (index 88) are not supported as
  >= 5 px physical displacements by the registered comparison: their reference
  along-line offsets are about +0.87 and +0.95 px. The previous local-grid
  displacement interpretation should not be carried forward for those points.
- Red marks in the new report mean measured normal distance >= 5 px. They are
  a review aid, not a newly established defect ground-truth count. Parallel
  displacement, missing holes, ambiguous correspondence, and unavailable edge
  evidence have separate fields; missing evidence is not a pass or a defect.
- Cubic self-fitting still limits sensitivity to broad, smoothly varying line
  shape changes. Keep reference registration and a fixed calibrated model in
  a future factory workflow; do not simply increase polynomial order to erase
  residuals.

## Run and review

From the repository root with the existing Python environment and native library:

```sh
python-port/venv/bin/python cpp-core/tools/stamp_hole_inspect.py \
  --bad /Users/Ayueh/Downloads/bad.png \
  --good /Users/Ayueh/Downloads/good.png \
  --template build/stamp-hole-pilot/hole-template.png \
  --baseline-dir build/stamp-hole-pilot \
  --out build/stamp-hole-curves --repeats 5

python-port/venv/bin/python cpp-core/tools/validate_stamp_hole_geometry.py \
  --out build/stamp-hole-curves/validation.json

python-port/venv/bin/python cpp-core/tools/report_stamp_hole_lines.py \
  build/stamp-hole-curves --legacy build/stamp-hole-curves-legacy --min-offset 2
```

The optional `--baseline-dir` verifies the old pilot's cached results; it is not
used as detection input. Run the detector with `--legacy-contours` and a different
`--out` directory for an otherwise identical legacy-contour timing comparison.
The detector needs NumPy, SciPy, OpenCV, and `libfpm_c`; the report also needs
Pillow and uses the Mac system Chinese font.

Outputs: `review-lines.html`, `per-hole-lines.csv`, `line-summary.csv`,
`curved-grid-overlay.png`, `curve-deviation-examples.png`, and `results.json`.
The offline HTML contains all 17 residual plots. Raw per-hole CSV includes the
matched reference pitch index so index reassignments remain reviewable. A shared
intersection has one native detection but a measurement on each applicable line.

### Complete annotation audit after the 1 px threshold correction

The first paired-image renderer incorrectly limited its selection to the original
27 bad-image >= 5 px review points, then hid <= 1 px labels within that subset.
That did **not** display all > 1 px measurements. The renderer now selects each
image independently from every supported per-line measurement. No >= 5 px
prefilter or reference-pair gate limits the annotation set.

A fresh detection pass reproduced the original observations and curve models.
The corrected bad-image annotation contains 371 distinct > 1 px hole positions
(372 line measurements because one intersection exceeds the threshold on two
lines). The good image contains 224. These counts include model/center residuals
and are not validated defect counts. Every selected position is drawn; the
renderer asserts that no selected ID is omitted and records computed coverage.

Current complete artifacts are under `build/stamp-hole-audit-all-gt1px/`:

- `01-bad-holes-grid-offsets-all-gt1px.png` and
  `02-good-holes-grid-offsets-all-gt1px.png`: original-resolution images, all
  > 1 px normal offsets labeled; >= 5 px is red, smaller offsets orange.
- `annotated-pair-labels-all-gt1px.csv`: all above-threshold per-line measurements.
  At a shared intersection the image shows the larger component and the CSV keeps
  both. Labels refer to each image's own curve, not a paired displacement.
- `unmeasured-sites.csv` and `annotation-coverage.json`: 58 bad / 50 good nominal
  sites lack reliable geometry. These are gray squares, not confirmed missing
  holes. Bad: 48 native rejections plus 10 accepted centers without adequate
  geometry; good: 49 plus 1. No missing measurement is counted as a pass.

The renderer's `--ignore-offset-at-most 1` compares unrounded magnitudes and
ignores <= 1 px. Values just above 1 px that would round to 1.0 are shown as
`>1` so the visible label does not contradict the threshold.

The current requested threshold is **>= 2 px**, selected with `--min-offset 2`.
This inclusive option overrides `--ignore-offset-at-most`: values below 2 px
are hidden and exactly 2 px is retained. The current pair contains **98 bad-image
and 31 good-image positions** at this threshold, with all >= 5 px positions still
red. The filenames end in `-all-ge2px.png`; the measurement CSV and versioned
coverage JSON have the same `-all-ge2px` suffix. The underlying detections and
curves are unchanged. Counts are checked against all raw per-line measurements,
not against the old 27-point selection.

## Validation

Controlled tests add a known smooth cubic image warp, a 6 px individual-hole
offset, a 6 px whole-line offset, a missing hole, and swapped provisional indices.
Both injected offsets measure approximately 6.00 px; the pure model-compatible
warp leaves no material registered residual. The near-zero synthetic residual is
an algebraic control, **not** measured camera accuracy. Original-image recall,
distortion model validity on new images, and manufacturing tolerances still need
independent labeled captures.

The next low-risk speed improvement is to retain the good reference and learned
template across frames, then measure only new images. Registration from a reduced
image and narrow full-resolution line regions are also candidates, but their
speed/recall tradeoffs have not been measured here.
