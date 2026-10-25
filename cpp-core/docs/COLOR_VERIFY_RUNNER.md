# Known-pose RGB verification worker

`fpm_color_verify_runner` compares already located candidate patches, or runs
the existing grayscale research `Session` to locate them in the same process.
It does not learn thresholds or read evaluation labels.
It is an opt-in research executable; stable matching defaults are unchanged.

Build from the repository root:

```sh
cmake -S cpp-core -B build/color-verify -DCMAKE_BUILD_TYPE=Release \
  -DFPM_BUILD_EXPERIMENTS=ON -DFPM_BUILD_GUI=ON -DFPM_BUILD_TESTS=ON
cmake --build build/color-verify --target fpm_color_verify_runner fpm_color_verify_test
ctest --test-dir build/color-verify -R experimental_color_verify --output-on-failure
build/color-verify/experiments/fpm_color_verify_runner job.json
```

Run the separate CLI contract controls from the repository root:

```sh
python3 cpp-core/tests/color_verify_runner_test.py \
  --runner build/color-verify/experiments/fpm_color_verify_runner
```

On a machine with available Metal hardware, add `--metal` to include the fifth
CPU/Metal RGB score comparison; otherwise that test is explicitly skipped.
The CLI controls use generated tiny fixtures and the Python standard library.
They are separate from CTest. Native memory/undefined-behavior checks can be
built with `-DFPM_SANITIZERS=ON`; sanitizer results are not performance results.

The source algorithms need C++17 only. The runner uses Qt for file decoding and
JSON and is built on Unix because its process RSS probe uses `getrusage`.
CPU/Metal NCC is the existing `fpm_pose_score` implementation, not a replacement.

## Job format

```json
{
  "source": "/absolute/path/source.png",
  "template": "/absolute/path/template.png",
  "poses": [[0, 0, 0, 0], [35, -20, 90, 0]],
  "method": "chroma",
  "backend": "cpu",
  "threads": 1,
  "warmups": 2,
  "repeats": 7,
  "chroma_distance_threshold": 0.04,
  "min_linear_sum": 0.03,
  "min_valid_fraction": 0.95
}
```

`method` is `gray_ncc`, `rgb_ncc`, or `chroma`. NCC supports `cpu` or `metal`;
chroma currently supports one CPU thread. Use a separate process per method and
backend. An optional `mask` path is accepted only for chroma, must have template
dimensions, and includes nonzero pixels. Masks change the evaluated support;
masked/unmasked results must not be represented as equal-work comparisons.

Each pose is `[tx, ty, theta_degrees, log_scale]`, following `pose_ncc.h`:

```
u = template_x - template_width / 2
v = template_height / 2 - template_y
source_x = source_width / 2 + tx + scale * (cos(theta)*u - sin(theta)*v)
source_y = source_height / 2 - ty - scale * (sin(theta)*u + cos(theta)*v)
```

For a scale-one, unrotated crop with top-left `(left,top)`, use
`tx = left + template_width/2 - source_width/2` and
`ty = source_height/2 - top - template_height/2`. Angles are positive CCW in
x-right/y-up coordinates. Integrated mode copies the exact conversion in
`research_pipeline.cpp`: `Match.angle` is passed unchanged to the pose scorer.

## Integrated search mode

Replace `poses` with a `search` object; providing both is an error. All returned
candidates then come from the existing full-image grayscale `Session`:

```json
{
  "source": "/absolute/path/source.png",
  "template": "/absolute/path/template.png",
  "method": "chroma",
  "backend": "cpu",
  "threads": 1,
  "warmups": 2,
  "repeats": 7,
  "search": {
    "backend": 1,
    "refinement_backend": 1,
    "scales": [1],
    "angle": 180,
    "limit": 100,
    "score": 0.65,
    "proposal_score": 0.5,
    "min_area": 256,
    "cpu_threads": 8,
    "pose_resident_mode": 2,
    "pose_storage_mode": 1,
    "pose_input_mode": 1,
    "fine_ncc_mode": 2,
    "fine_fused": true,
    "fine_angle_reuse": true
  }
}
```

The nested backend controls search and refinement, whereas the top-level
backend controls additional verification. `method: "geometry_only"` preserves
the Session's scores and runs no verification or extra color context; its
top-level backend must be `cpu` (no verifier exists), regardless of the nested
search backend. This is the integrated timing/RSS baseline.

The example search values are defaults. If search backend/refinement backend
are CPU, their corresponding GPU defaults automatically become zero/false.
Metal SIMD refinement still requires a template with at most 16,384 pixels;
larger templates must explicitly choose `pose_resident_mode: 0`. Further
supported options are `ncc_mode: 0`, `overlap_enabled: false`, `overlap: 0.2`,
`stop_layer: 1`, `subpixel: false`, `refine: true`, and
`scale_support_refine: false`. Unknown nested keys are rejected. This runner
does not expose source inversion or sparse proposal settings.

Stage35's experiment recipe explicitly overrides `min_area` to `512`,
`proposal_score` to `.25`, and `limit` to `300`; the example above continues to
describe the actual unchanged runner defaults. The original values are kept as
a separate comparison. See [the pyramid diagnosis](STAGE35_PYRAMID_DIAGNOSIS_zh-TW.md)
for the pre-measurement evidence and remaining real-scene failures.

All methods must use identical `search` settings to compare color verification
without changing detection. A color verifier cannot recover objects missed by
the grayscale search. Its scores are reported without filtering detections so
evaluation can distinguish localization failure from color decision failure.

## Three measurements and their limits

Gray NCC uses the repository's integer RGB-to-gray conversion. RGB NCC centers
each channel separately and jointly normalizes their energy:

```
rho = sum_c sum_i [(T_ic - mean(T_c)) (I_ic - mean(I_c))]
      / sqrt(sum_c sum_i (T_ic - mean(T_c))^2
             * sum_c sum_i (I_ic - mean(I_c))^2)
```

Consequently `I_ic = a*T_ic + b_c`, with the same positive `a` across channels,
has correlation one if there is no clipping/quantization. Separate channel
offsets cancel even when they visibly change color. Separate channel gains are
not in general invariant under this joint normalization. NCC alone is not a
color-defect detector; constant templates return the API's invalid score `-2`.

Chroma applies an explicit 256-entry sRGB-to-linear lookup table:

```
L(v) = v/12.92                         if v <= 0.04045
       ((v + 0.055)/1.055)^2.4         otherwise
c = [L(R), L(G), L(B)] / (L(R)+L(G)+L(B))
distance_i = ||c_template_i - c_source_i||_2 / sqrt(2)
```

The source is sampled bilinearly in linear RGB before normalization. For a
positive common linear-light gain `g`, `c(g*RGB) = c(RGB)`: neutral shading is
suppressed, while relative channel changes remain. Encoded input bytes are
assumed to be sRGB; embedded color profiles are not converted. This is not
Lab/DeltaE or a calibrated perceptual color difference. White-balance changes,
colored lighting, reflections, clipping and alignment errors can all change
the result. No model or vendor color library is required.

Template pixels below `min_linear_sum` are excluded. Source pixels outside the
image or below that limit count as missing support. `min_valid_fraction` applies
to the eligible template support. Invalid results keep counts but use `-1` for
distance/fraction, so a mostly missing or dark patch cannot become an apparent
perfect match. Status `0` is valid, `1` has no eligible template pixels, and `2`
has insufficient source support.

`mean_distance` averages the valid distances. `changed_fraction` is the fraction
of valid pixels with distance strictly greater than `chroma_distance_threshold`.
This pixel threshold does not itself decide that an object is anomalous. Any
object threshold must be calibrated using separate normal/training data and
evaluated on held-out data. The default `.04` is only a research control value.

## Illumination-normalized chroma (`chroma_vk`, round 46)

`method: "chroma_vk"` is an opt-in variant implemented separately in
`experiments/color_illuminant.{hpp,cpp}`; `chroma`, `rgb_ncc` and `gray_ncc` are
unchanged. It estimates the source illuminant from a neutral reference, applies a
von Kries diagonal gain that maps source linear RGB into the template's light, and
then computes the chroma distance over `mask` (the colour region) only:

```
W_source,c   = median of nearest source pixels under the reference, linear RGB
W_template,c = the same statistic on the template reference
gain_c = W_template,c / W_source,c
distance = ||c(gain * I) - c(T)|| / sqrt(2)
```

The reference is either `illuminant_reference: "object"` (default), with a
template-size `reference_mask` mapped through every pose (for example a printed
white panel inside the object), or `"source_rect"` with `reference_rect:
[x0,y0,x1,y1)` in source pixels (a fixed calibration card; `template_reference_rgb`
gives that reference's linear RGB under the template light, default `[1,1,1]`).
Medians are per channel over nearest pixels, so clipped bytes can stay in the order
statistic: a median is unchanged while fewer than half of the samples are clipped.

`reference_clip_policy` controls saturation (`saturation_level`, default 255: a
byte at or above it is clipped):

- `rank` (default): clipped bytes stay in the median; the reference is unusable
  when any channel's clipped share exceeds `max_reference_clip_fraction` (.45).
  Colour samples with any clipped bilinear neighbour are missing support.
- `drop`: ablation; reference samples with any clipped channel are discarded
  before the median (a truncated sample biases the estimate low).
- `none`: ablation; no saturation handling anywhere.

The reference is also unusable when fewer than `min_reference_fraction` (.5) of its
samples fall inside the source (for `drop`, after dropping) or its median is dark.
Each score has `status` (0 valid, 1 no eligible template colour pixels, 2
insufficient colour support, 3 reference unusable), `mean_distance` (per-pixel mean,
as `chroma`), `region_distance` (chroma of the summed valid linear RGB against the
template's sum over the same samples, so zero-mean noise adds no floor),
`uncorrected_mean_distance` and `uncorrected_region_distance` (same samples, no
gain), `changed_pixels`/`changed_fraction`, `clipped_pixels`, `reference_eligible`,
`reference_samples`, `reference_fraction`, `reference_clip_fraction` (per channel)
and `gain`. Status 3 is an abstention: corrected distances are `-1` and must never be
read as acceptance. Its uncorrected distances remain for diagnostics. The output adds
`illuminant_config`. Illuminant options are rejected for other methods, and
`chroma_vk` supports one CPU thread.

A diagonal gain cannot remove light that varies inside one object, colored
inter-reflections, or a non-neutral reference, and a clipped reference cannot be
recovered. Thresholds still need separate calibration data. Round 46
([ILLUMINANT_CHROMA_zh-TW.md](ILLUMINANT_CHROMA_zh-TW.md)) evaluates it on test-bed v1.

### Linear reference model (`reference_model`, round 47)

`reference_model: "constant"` (default) is the model above, with byte-identical
scores. `"linear"` (object reference only) models a first-order light gradient
inside the object. The template reference pixels are ranked by template `u` and,
separately, by `v`; the lowest and highest thirds are the two ends of each axis.
Each end is estimated like the whole reference (same clip policy and limits) and
placed at the median template `(u, v)` of the samples behind its median. The two
slopes solve both end pairs jointly, so a pair that is unbalanced in the other
coordinate (rank ties) does not leak that axis's gradient:

```
W_high_u - W_low_u = slope_u (u_high_u - u_low_u) + slope_v (v_high_u - v_low_u)
W_high_v - W_low_v = slope_u (u_high_v - u_low_v) + slope_v (v_high_v - v_low_v)
W_c(u,v)  = W_all,c + slope_u,c (u - u_all) + slope_v,c (v - v_all)
corrected = I_c(u,v) * W_template,c / W_c(u,v)    at every colour sample
```

Every end must be usable, both pairs must be ordered along their own axis with a
positive determinant, and `W` must stay positive at every valid colour sample, or
the score is status 3. `gain` is taken at the reference centre, and linear
scores add `reference_slope_u`/`reference_slope_v` (source linear RGB per template
pixel; zero when unusable). When the colour field lies beside the reference, the
`u` slope is extrapolated, which amplifies reference noise. Round 47
([CHROMA_MARGIN_zh-TW.md](CHROMA_MARGIN_zh-TW.md)) evaluates it.

## Outputs and measurement boundaries

The JSON output has `first`, `warmups_ms`, and `runs`. Each first/run record has
`wall_ms`, `api_wall_ms`, `device_ms`, `scores`, and process memory. For NCC,
`scores` is a number array; for chroma it is an array with `status`,
`eligible_pixels`, `valid_pixels`, `changed_pixels`, `valid_fraction`,
`mean_distance`, and `changed_fraction`. Ordering matches input poses.

Integrated records additionally have `search_matches` (positions, angles,
scales, grayscale scores and corners), `search_ms` (wall),
`search_algorithm_ms` (Session internal total), `verification_ms`, and
`pipeline_ms` (also `wall_ms`). Verification includes conversion of detected
matches to poses and resizing result buffers. `geometry_only` has zero
verification time. Empty search output is supported and produces empty scores.
`search_diagnostics` breaks down Session preparation, proposals, refinement,
NMS and its existing buffer counters. `exact_repeat` compares both complete
search matches and verification scores.

`decode_ms` is separate. `construct_ms` includes conversions, copying and model
creation; gray includes grayscale conversion. Integrated mode further splits
`search_construct_ms` and `color_construct_ms`. Qt decode buffers are released
before the first score. Score timing includes invocation and output-array
updates, but excludes JSON serialization and memory queries. Pose-only mode
allocates input poses and output arrays before scoring; integrated mode includes
pose packing and output resizing in verification time. Warm-up measurements must not be
mixed into steady-state medians. `exact_repeat` compares all returned scores.

`retained_host_bytes` describes only the verification method's owned capacities,
not process RSS or the separate Session capacities in integrated mode.
For chroma this includes the copied byte source, selected template samples and
256-value LUT. It does not include runner pose/results arrays, decoder caches,
Qt, standard-library/runtime allocations, or fragmentation. NCC uses its
existing buffer accounting. Metal shared pages can overlap host and GPU totals;
do not sum those totals as separate RAM allocations. `peak_rss_bytes` is the
whole process high-water mark, including decoding/construction. Current RSS is
provided on macOS and is `-1` when unavailable elsewhere. Neither is a GPU-only
memory measure.

Integrated RSS measures search and verification in the **same process**. Do not
add RSS from separate pose-only and search processes to estimate integrated
memory. All variants decode RGB inputs, including the geometry baseline, so
input decoding is consistent. Search and color contexts retain separate
images in this first opt-in implementation; no shared RGB context is implied.

The small native regression test covers exact/rotated crops, border support,
template masks, ownership, invalid inputs, neutral exposure and the important
case where per-channel offset changes survive chroma but disappear in RGB NCC.
