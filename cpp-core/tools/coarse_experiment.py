"""Standard-library ctypes adapter for opt-in Metal dense coarse NCC.

CoarsePattern.match has the portable Pattern interface, and returns identical
fields. last_metrics identifies coarse GPU work; explicit execution options can
also move source pyramid filtering and coarse angle submissions to Metal.
"""
import ctypes as C
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from python_ctypes import Config, Image, Result


class CoarseMetrics(C.Structure):
    _fields_ = [(name, C.c_double) for name in (
        'wall_ms', 'setup_ms', 'upload_ms', 'device_ms', 'wait_ms',
        'normalization_ms', 'ncc_ms', 'reserved_ms')] + [
        (name, C.c_uint64) for name in (
            'score_maps', 'windows', 'command_buffers', 'integer32_maps',
            'integer64_maps', 'host_bytes', 'gpu_bytes', 'workspace_bytes',
            'source_buffer_bytes', 'template_buffer_bytes', 'scoremap_peak_bytes', 'reserved_bytes')]


class CoarseProfileV1(C.Structure):
    _fields_ = [('version', C.c_uint32), ('struct_bytes', C.c_uint32),
                ('backend', C.c_int32), ('reserved', C.c_int32)] + [
        (name, C.c_double) for name in (
            'api_wall_ms', 'core_wall_ms', 'source_copy_invert_ms', 'pyramid_ms',
            'coarse_rotation_ms', 'coarse_ncc_ms', 'peak_selection_ms',
            'fine_rotation_ms', 'fine_ncc_ms', 'fine_control_ms', 'nms_ms',
            'unclassified_ms', 'api_residual_ms')] + [
        (name, C.c_uint64) for name in (
            'pyramid_levels', 'coarse_angles', 'coarse_maps', 'coarse_windows',
            'coarse_candidates', 'fine_layer_visits', 'fine_passes',
            'fine_rotations', 'fine_maps', 'fine_windows', 'refined_candidates', 'output_matches',
            'source_copy_bytes', 'pyramid_generated_bytes', 'pyramid_live_bytes',
            'coarse_rotation_total_bytes', 'coarse_rotation_peak_bytes',
            'fine_roi_total_bytes', 'fine_roi_peak_bytes',
            'scoremap_peak_bytes', 'scoremap_total_bytes', 'timed_transitions')]


class CoarseExecutionConfigV1(C.Structure):
    _fields_ = [('version', C.c_uint32), ('struct_bytes', C.c_uint32),
                ('pyramid_mode', C.c_int32), ('batch_angles', C.c_int32),
                ('reserved0', C.c_uint64), ('reserved1', C.c_uint64)]


class CoarseExecutionMetricsV1(C.Structure):
    _fields_ = [('version', C.c_uint32), ('struct_bytes', C.c_uint32),
                ('pyramid_mode', C.c_int32), ('batch_angles', C.c_int32)] + [
        (name, C.c_double) for name in (
            'pyramid_wall_ms', 'pyramid_setup_ms', 'pyramid_upload_ms', 'pyramid_encode_ms',
            'pyramid_wait_ms', 'pyramid_device_ms', 'pyramid_copyout_ms', 'batch_encode_ms')] + [
        (name, C.c_uint64) for name in (
            'pyramid_command_buffers', 'pyramid_levels', 'pyramid_input_bytes', 'pyramid_output_bytes',
            'pyramid_scratch_bytes', 'pyramid_buffer_bytes', 'batch_calls', 'batch_maps',
            'max_batch_maps', 'batch_source_arena_bytes', 'batch_source_upload_bytes',
            'batch_template_buffer_bytes', 'batch_statistics_bytes', 'reserved_bytes')]


def library_path(build):
    suffix = '.dylib' if sys.platform == 'darwin' else '.dll' if sys.platform == 'win32' else '.so'
    prefix = '' if sys.platform == 'win32' else 'lib'
    base = Path(build) / 'experiments'
    for path in (base / (prefix+'fpm_coarse_experiment'+suffix),
                 base / 'Release' / (prefix+'fpm_coarse_experiment'+suffix)):
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f'Build with -DFPM_BUILD_EXPERIMENTS=ON: {base}')


class CoarsePattern:
    def __init__(self, build, image, min_area=256, backend='metal', pyramid_mode=0, batch_angles=1):
        if backend not in ('cpu', 'metal'):
            raise ValueError('backend must be cpu or metal')
        self.dll = C.CDLL(str(library_path(build)))
        self.handle = C.c_void_p()
        self.backend = backend
        self.profile = None
        self._execution_config = CoarseExecutionConfigV1(1, C.sizeof(CoarseExecutionConfigV1), 0, 1, 0, 0)
        self.dll.fpm_coarse_create.argtypes = [Image, C.c_int, C.POINTER(C.c_void_p),
                                               C.POINTER(CoarseMetrics), C.POINTER(C.c_char), C.c_size_t]
        self.dll.fpm_coarse_create.restype = C.c_int
        if hasattr(self.dll, 'fpm_coarse_create_backend'):
            self.dll.fpm_coarse_create_backend.argtypes = [Image, C.c_int, C.c_int, C.POINTER(C.c_void_p),
                C.POINTER(CoarseMetrics), C.POINTER(C.c_char), C.c_size_t]
            self.dll.fpm_coarse_create_backend.restype = C.c_int
        self.dll.fpm_coarse_destroy.argtypes = [C.c_void_p]
        self.dll.fpm_coarse_destroy.restype = None
        self.dll.fpm_coarse_match.argtypes = [C.c_void_p, Image, C.POINTER(Config), C.POINTER(Result),
            C.c_size_t, C.POINTER(C.c_size_t), C.POINTER(CoarseMetrics), C.POINTER(C.c_char), C.c_size_t]
        self.dll.fpm_coarse_match.restype = C.c_int
        if hasattr(self.dll, 'fpm_coarse_match_profile'):
            self.dll.fpm_coarse_match_profile.argtypes = [C.c_void_p, Image, C.POINTER(Config), C.POINTER(Result),
                C.c_size_t, C.POINTER(C.c_size_t), C.POINTER(CoarseMetrics), C.POINTER(CoarseProfileV1),
                C.c_size_t, C.POINTER(C.c_char), C.c_size_t]
            self.dll.fpm_coarse_match_profile.restype = C.c_int
        self.dll.fpm_coarse_scoremap.argtypes = [C.c_void_p, Image, C.POINTER(C.c_double), C.c_size_t,
            C.POINTER(C.c_size_t), C.POINTER(CoarseMetrics), C.POINTER(C.c_char), C.c_size_t]
        self.dll.fpm_coarse_scoremap.restype = C.c_int
        if hasattr(self.dll, 'fpm_coarse_set_execution'):
            self.dll.fpm_coarse_execution_default_config.argtypes = [C.POINTER(CoarseExecutionConfigV1)]
            self.dll.fpm_coarse_execution_default_config.restype = None
            self.dll.fpm_coarse_set_execution.argtypes = [C.c_void_p, C.POINTER(CoarseExecutionConfigV1),
                C.c_size_t, C.POINTER(C.c_char), C.c_size_t]
            self.dll.fpm_coarse_set_execution.restype = C.c_int
            self.dll.fpm_coarse_get_execution_metrics.argtypes = [C.c_void_p, C.POINTER(CoarseExecutionMetricsV1),
                C.c_size_t, C.POINTER(C.c_char), C.c_size_t]
            self.dll.fpm_coarse_get_execution_metrics.restype = C.c_int
        self.error = C.create_string_buffer(1024)
        self.metrics = CoarseMetrics()
        if backend == 'cpu':
            if not hasattr(self.dll, 'fpm_coarse_create_backend'):
                raise RuntimeError('CPU coarse profile handles are unavailable; rebuild the experimental library')
            self._check(self.dll.fpm_coarse_create_backend(image, min_area, 0, C.byref(self.handle),
                C.byref(self.metrics), self.error, len(self.error)))
        else:
            self._check(self.dll.fpm_coarse_create(image, min_area, C.byref(self.handle),
                C.byref(self.metrics), self.error, len(self.error)))
        self.setup_metrics = self.last_metrics
        # Preserve frozen-library compatibility and original default setup cost.
        if pyramid_mode != 0 or batch_angles != 1:
            try:
                self.set_execution(pyramid_mode=pyramid_mode, batch_angles=batch_angles)
            except Exception:
                self.close()
                raise

    @property
    def last_metrics(self):
        return {name: getattr(self.metrics, name) for name, _ in self.metrics._fields_}

    @property
    def last_profile(self):
        """Exclusive stages/counters from the latest profile=True call, or None."""
        return None if self.profile is None else {name: getattr(self.profile, name) for name, _ in self.profile._fields_}

    @property
    def execution_config(self):
        return {name: getattr(self._execution_config, name) for name, _ in self._execution_config._fields_}

    @property
    def last_execution_metrics(self):
        """Query outside match timing; None for frozen libraries without this API."""
        if not self.handle:
            raise ValueError('pattern is closed')
        if not hasattr(self.dll, 'fpm_coarse_get_execution_metrics'):
            return None
        metrics = CoarseExecutionMetricsV1()
        self._check(self.dll.fpm_coarse_get_execution_metrics(self.handle, C.byref(metrics), C.sizeof(metrics),
            self.error, len(self.error)))
        return {name: getattr(metrics, name) for name, _ in metrics._fields_}

    def set_execution(self, *, pyramid_mode=0, batch_angles=1):
        if not self.handle:
            raise ValueError('pattern is closed')
        if not hasattr(self.dll, 'fpm_coarse_set_execution'):
            raise RuntimeError('Coarse execution options are unavailable; rebuild the experimental library')
        if not isinstance(pyramid_mode, int) or not isinstance(batch_angles, int):
            raise ValueError('pyramid_mode and batch_angles must be integers')
        if pyramid_mode not in (0, 1, 2) or batch_angles not in (1, 2, 3, 4):
            raise ValueError('pyramid_mode must be 0..2 and batch_angles must be 1..4')
        config = CoarseExecutionConfigV1()
        self.dll.fpm_coarse_execution_default_config(C.byref(config))
        config.pyramid_mode = pyramid_mode
        config.batch_angles = batch_angles
        self._check(self.dll.fpm_coarse_set_execution(self.handle, C.byref(config), C.sizeof(config),
            self.error, len(self.error)))
        self._execution_config = config

    def _check(self, status):
        if status:
            raise RuntimeError(self.error.value.decode('utf-8', errors='replace') or f'FPM status {status}')

    def match(self, image, config, *, prune=False, profile=False):
        if not self.handle:
            raise ValueError('pattern is closed')
        if config.max_targets < 1:
            raise ValueError('max_targets must be positive')
        # Both explicit experiment backends use complete exact scoremaps. This
        # compatibility argument does not enable the stable API's CPU pruning.
        del prune
        results = (Result * config.max_targets)()
        count = C.c_size_t()
        self.profile = None
        if profile:
            if not hasattr(self.dll, 'fpm_coarse_match_profile'):
                raise RuntimeError('Coarse stage profiling is unavailable; rebuild the experimental library')
            self.profile = CoarseProfileV1()
            self._check(self.dll.fpm_coarse_match_profile(self.handle, image, C.byref(config), results,
                len(results), C.byref(count), C.byref(self.metrics), C.byref(self.profile), C.sizeof(self.profile),
                self.error, len(self.error)))
        else:
            self._check(self.dll.fpm_coarse_match(self.handle, image, C.byref(config), results,
                len(results), C.byref(count), C.byref(self.metrics), self.error, len(self.error)))
        return [{'index': i, 'score': r.score, 'angle': r.angle,
                 'pos_x': r.center.x, 'pos_y': r.center.y,
                 'corners': [[p.x, p.y] for p in r.corners]}
                for i, r in enumerate(results[:count.value])]

    def scoremap(self, image):
        if not self.handle:
            raise ValueError('pattern is closed')
        count = C.c_size_t()
        status = self.dll.fpm_coarse_scoremap(self.handle, image, None, 0, C.byref(count),
            C.byref(self.metrics), self.error, len(self.error))
        if status != 2:
            self._check(status)
            raise RuntimeError('unexpected scoremap count query result')
        values = (C.c_double * count.value)()
        self._check(self.dll.fpm_coarse_scoremap(self.handle, image, values, count.value, C.byref(count),
            C.byref(self.metrics), self.error, len(self.error)))
        return values

    def close(self):
        if self.handle:
            self.dll.fpm_coarse_destroy(self.handle)
            self.handle = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FineMetricsV1(C.Structure):
    _fields_ = [('version', C.c_uint32), ('struct_bytes', C.c_uint32)] + [
        (name, C.c_double) for name in (
            'wall_ms', 'setup_ms', 'upload_ms', 'rotation_ms', 'repair_ms',
            'ncc_ms', 'normalization_ms', 'encode_ms', 'wait_ms', 'device_ms')] + [
        (name, C.c_uint64) for name in (
            'calls', 'maps', 'roi_pixels', 'repair_pixels', 'repair_all_maps',
            'command_buffers', 'source_uploads', 'source_upload_bytes',
            'source_buffer_bytes', 'roi_buffer_bytes', 'guard_buffer_bytes',
            'descriptor_buffer_bytes', 'statistics_buffer_bytes', 'template_buffer_bytes',
            'retained_gpu_bytes', 'peak_batch_maps')]


class FineReuseMetricsV1(C.Structure):
    _fields_ = [('version', C.c_uint32), ('struct_bytes', C.c_uint32),
                ('ncc_mode', C.c_int32), ('reserved', C.c_int32)] + [
        (name, C.c_uint64) for name in ('row_buffer_bytes', 'row_dispatches', 'statistics_dispatches')] + [
        ('stats_device_ms', C.c_double)]


class FineFusionMetricsV1(C.Structure):
    _fields_ = [('version', C.c_uint32), ('struct_bytes', C.c_uint32),
                ('enabled', C.c_int32), ('reserved', C.c_int32)] + [
        (name, C.c_double) for name in ('correction_ms', 'fused_wait_ms', 'fused_device_ms')] + [
        (name, C.c_uint64) for name in ('fused_batches', 'fallback_batches', 'dense_maps', 'changed_pixels', 'corrected_windows')]


class FineSearchMetricsV1(C.Structure):
    _fields_ = [('version', C.c_uint32), ('struct_bytes', C.c_uint32),
                ('profiling_enabled', C.c_int32), ('angle_reuse_enabled', C.c_int32)] + [
        (name, C.c_double) for name in ('observe_ms', 'plan_ms', 'descriptor_ms',
            'warp_encode_ms', 'warp_wait_ms', 'warp_device_ms', 'repair_all_ms',
            'guard_scan_ms', 'ncc_encode_ms', 'ncc_wait_ms')] + [
        (name, C.c_uint64) for name in ('calls', 'requests', 'within_call_duplicates',
            'cross_call_duplicates', 'untracked_call_keys', 'tracked_unique_keys',
            'history_limit_keys', 'history_key_payload_bytes', 'history_bucket_count',
            'peak_call_key_capacity_bytes', 'logical_requests', 'reused_maps',
            'cache_peak_bytes', 'warp_batches', 'zero_repair_batches', 'guard_words', 'nonzero_guard_words')]


class CoarseOptimizationConfigV1(C.Structure):
    _fields_ = [('version', C.c_uint32), ('struct_bytes', C.c_uint32),
                ('peak_mode', C.c_int32), ('ncc_mode', C.c_int32),
                ('reserved0', C.c_uint64), ('reserved1', C.c_uint64)]


class CoarseOverlapConfigV1(C.Structure):
    _fields_ = [('version', C.c_uint32), ('struct_bytes', C.c_uint32),
                ('enabled', C.c_int32), ('reserved0', C.c_int32),
                ('reserved1', C.c_uint64), ('reserved2', C.c_uint64)]


class CoarseOverlapMetricsV1(C.Structure):
    _fields_ = [('version', C.c_uint32), ('struct_bytes', C.c_uint32),
                ('enabled', C.c_int32), ('reserved', C.c_int32)] + [
        (name, C.c_double) for name in (
            'submit_ms', 'copy_ms', 'encode_ms', 'finish_wait_ms', 'normalization_ms',
            'device_ms', 'device_span_ms', 'fallback_ms')] + [
        (name, C.c_uint64) for name in (
            'submitted_batches', 'submitted_maps', 'submissions_with_pending', 'max_pending',
            'completed_batches', 'fallback_batches', 'fallback_maps', 'fallback_single',
            'fallback_degenerate', 'fallback_oversized', 'source_upload_bytes', 'template_upload_bytes',
            'slot0_source_bytes', 'slot0_statistics_bytes', 'slot0_template_bytes',
            'slot1_source_bytes', 'slot1_statistics_bytes', 'slot1_template_bytes',
            'retained_gpu_bytes', 'drained_batches')]


class FrameConfigV1(C.Structure):
    _fields_ = [('version', C.c_uint32), ('struct_bytes', C.c_uint32)] + [
        (name, C.c_int32) for name in ('backend', 'pyramid_mode', 'batch_angles', 'fine_batch_size')
    ] + [('reserved', C.c_uint64)]


class FrameMetricsV1(C.Structure):
    _fields_ = [('version', C.c_uint32), ('struct_bytes', C.c_uint32)] + [
        (name, C.c_double) for name in ('prepare_wall_ms', 'source_copy_ms', 'pyramid_ms')] + [
        (name, C.c_uint64) for name in (
            'generation', 'models_matched', 'pyramid_levels', 'host_pyramid_bytes',
            'coarse_gpu_bytes', 'fine_gpu_bytes')]


def _fields_snapshot(value):
    return {name: getattr(value, name) for name, _ in value._fields_}


class SourceFrame:
    """Explicit source snapshot shared by any number of scales/templates.

    Call prepare ONCE per new image (even if its address and size are unchanged)
    and include that cost in frame latency. match reads the immutable snapshot.
    Fine GPU search is opt-in through fine_batch_size=1..1024. Each match saves
    independent diagnostics; capture them immediately after each scale, then
    query frame_metrics once for shared retained memory (never sum per model).
    """
    def __init__(self, build, *, backend='metal', pyramid_mode=None, batch_angles=None, fine_batch_size=0):
        if backend not in ('cpu', 'metal'):
            raise ValueError('backend must be cpu or metal')
        pyramid_mode = (1 if backend == 'metal' else 0) if pyramid_mode is None else pyramid_mode
        batch_angles = (4 if backend == 'metal' else 1) if batch_angles is None else batch_angles
        self._ready = False
        for value in (pyramid_mode, batch_angles, fine_batch_size):
            if not isinstance(value, int):
                raise ValueError('frame execution options must be integers')
        if pyramid_mode not in (0, 1, 2) or batch_angles not in (1, 2, 3, 4) or not 0 <= fine_batch_size <= 1024:
            raise ValueError('invalid frame execution options')
        self.dll = C.CDLL(str(library_path(build)))
        self.handle = C.c_void_p()
        self.error = C.create_string_buffer(1024)
        if not hasattr(self.dll, 'fpm_frame_create'):
            raise RuntimeError('Shared frame API is unavailable; rebuild the experimental library')
        self.dll.fpm_frame_create.argtypes = [C.POINTER(FrameConfigV1), C.c_size_t,
            C.POINTER(C.c_void_p), C.POINTER(C.c_char), C.c_size_t]
        self.dll.fpm_frame_create.restype = C.c_int
        self.dll.fpm_frame_destroy.argtypes = [C.c_void_p]
        self.dll.fpm_frame_destroy.restype = None
        self.dll.fpm_frame_prepare.argtypes = [C.c_void_p, Image, C.POINTER(C.c_void_p), C.c_size_t,
            C.c_int, C.POINTER(FrameMetricsV1), C.POINTER(C.c_char), C.c_size_t]
        self.dll.fpm_frame_prepare.restype = C.c_int
        self.dll.fpm_frame_get_metrics.argtypes = [C.c_void_p, C.POINTER(FrameMetricsV1), C.c_size_t,
            C.POINTER(CoarseExecutionMetricsV1), C.c_size_t, C.POINTER(C.c_char), C.c_size_t]
        self.dll.fpm_frame_get_metrics.restype = C.c_int
        self.dll.fpm_frame_match.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(Config), C.POINTER(Result),
            C.c_size_t, C.POINTER(C.c_size_t), C.POINTER(CoarseMetrics), C.POINTER(CoarseProfileV1), C.c_size_t,
            C.POINTER(CoarseExecutionMetricsV1), C.c_size_t, C.POINTER(FineMetricsV1), C.c_size_t,
            C.POINTER(C.c_char), C.c_size_t]
        self.dll.fpm_frame_match.restype = C.c_int
        self.config = FrameConfigV1(1, C.sizeof(FrameConfigV1), int(backend == 'metal'),
                                   pyramid_mode, batch_angles, fine_batch_size, 0)
        self.last_metrics = self.last_profile = self.last_execution_metrics = self.last_fine_metrics = None
        self.last_overlap_metrics = None
        self._check(self.dll.fpm_frame_create(C.byref(self.config), C.sizeof(self.config),
            C.byref(self.handle), self.error, len(self.error)))

    def _check(self, status):
        if status:
            raise RuntimeError(self.error.value.decode('utf-8', errors='replace') or f'FPM status {status}')

    def _pattern(self, pattern):
        if not self.handle:
            raise ValueError('frame is closed')
        if not isinstance(pattern, CoarsePattern) or not pattern.handle:
            raise ValueError('an open CoarsePattern is required')
        if Path(pattern.dll._name).resolve() != Path(self.dll._name).resolve():
            raise ValueError('frame and pattern must come from the same library')
        return pattern.handle

    @property
    def execution_config(self):
        return _fields_snapshot(self.config)

    def prepare(self, image, patterns, *, invert_source=False):
        if not self.handle:
            raise ValueError('frame is closed')
        self._ready = False
        self.last_metrics = self.last_profile = self.last_execution_metrics = self.last_fine_metrics = None
        self.last_overlap_metrics = None
        models = list(patterns)
        if not models:
            raise ValueError('at least one pattern is required')
        handles = (C.c_void_p * len(models))(*(self._pattern(p).value for p in models))
        metrics = FrameMetricsV1()
        self.last_metrics = self.last_profile = self.last_execution_metrics = self.last_fine_metrics = None
        self._check(self.dll.fpm_frame_prepare(self.handle, image, handles, len(models), int(bool(invert_source)),
            C.byref(metrics), self.error, len(self.error)))
        self._ready = True
        return _fields_snapshot(metrics)

    def set_fine_ncc_mode(self, mode=0):
        if not self.handle:
            raise ValueError('frame is closed')
        if type(mode) is not int or mode not in (0, 1, 2, 3):
            raise ValueError('fine NCC mode must be integer in 0..3')
        if not hasattr(self.dll, 'fpm_frame_set_fine_ncc_mode'):
            raise RuntimeError('Fine reuse API unavailable; rebuild experimental library')
        function = self.dll.fpm_frame_set_fine_ncc_mode
        function.argtypes = [C.c_void_p, C.c_int32, C.POINTER(C.c_char), C.c_size_t]
        function.restype = C.c_int
        self._check(function(self.handle, mode, self.error, len(self.error)))

    def fine_reuse_metrics(self):
        if not self.handle:
            raise ValueError('frame is closed')
        if not hasattr(self.dll, 'fpm_frame_get_fine_reuse_metrics'):
            raise RuntimeError('Fine reuse metrics unavailable; rebuild experimental library')
        function = self.dll.fpm_frame_get_fine_reuse_metrics
        function.argtypes = [C.c_void_p, C.POINTER(FineReuseMetricsV1), C.c_size_t,
                             C.POINTER(C.c_char), C.c_size_t]
        function.restype = C.c_int
        metrics = FineReuseMetricsV1()
        self._check(function(self.handle, C.byref(metrics), C.sizeof(metrics), self.error, len(self.error)))
        return _fields_snapshot(metrics)

    def set_fine_fused(self, enabled=False):
        if not self.handle:
            raise ValueError('frame is closed')
        if type(enabled) is not bool:
            raise ValueError('fine fusion must be boolean')
        function = self.dll.fpm_frame_set_fine_fused
        function.argtypes = [C.c_void_p, C.c_int32, C.POINTER(C.c_char), C.c_size_t]
        function.restype = C.c_int
        self._check(function(self.handle, enabled, self.error, len(self.error)))

    def fine_fusion_metrics(self):
        if not self.handle:
            raise ValueError('frame is closed')
        function = self.dll.fpm_frame_get_fine_fusion_metrics
        function.argtypes = [C.c_void_p, C.POINTER(FineFusionMetricsV1), C.c_size_t, C.POINTER(C.c_char), C.c_size_t]
        function.restype = C.c_int
        metrics = FineFusionMetricsV1()
        self._check(function(self.handle, C.byref(metrics), C.sizeof(metrics), self.error, len(self.error)))
        return _fields_snapshot(metrics)

    def set_fine_search_options(self, *, profiling=False, angle_reuse=False):
        if not self.handle:
            raise ValueError('frame is closed')
        if type(profiling) is not bool or type(angle_reuse) is not bool:
            raise ValueError('fine search options must be booleans')
        if not hasattr(self.dll, 'fpm_frame_set_fine_search_options'):
            raise RuntimeError('Fine search options unavailable; rebuild experimental library')
        function = self.dll.fpm_frame_set_fine_search_options
        function.argtypes = [C.c_void_p, C.c_int32, C.c_int32, C.POINTER(C.c_char), C.c_size_t]
        function.restype = C.c_int
        self._check(function(self.handle, profiling, angle_reuse, self.error, len(self.error)))

    def fine_search_metrics(self):
        if not self.handle:
            raise ValueError('frame is closed')
        if not hasattr(self.dll, 'fpm_frame_get_fine_search_metrics'):
            raise RuntimeError('Fine search metrics unavailable; rebuild experimental library')
        function = self.dll.fpm_frame_get_fine_search_metrics
        function.argtypes = [C.c_void_p, C.POINTER(FineSearchMetricsV1), C.c_size_t,
                             C.POINTER(C.c_char), C.c_size_t]
        function.restype = C.c_int
        metrics = FineSearchMetricsV1()
        self._check(function(self.handle, C.byref(metrics), C.sizeof(metrics), self.error, len(self.error)))
        return _fields_snapshot(metrics)

    def set_coarse_optimization(self, *, peak_mode=0, ncc_mode=0):
        if not self.handle:
            raise ValueError('frame is closed')
        if (not isinstance(peak_mode, int) or peak_mode not in (0, 1, 2) or
                not isinstance(ncc_mode, int) or ncc_mode not in (0, 1, 2, 3)):
            raise ValueError('coarse peak mode must be integer 0..2 and NCC mode integer 0..3')
        if not hasattr(self.dll, 'fpm_frame_set_coarse_optimization'):
            raise RuntimeError('Coarse optimization API unavailable; rebuild experimental library')
        function = self.dll.fpm_frame_set_coarse_optimization
        function.argtypes = [C.c_void_p, C.POINTER(CoarseOptimizationConfigV1), C.c_size_t,
                             C.POINTER(C.c_char), C.c_size_t]
        function.restype = C.c_int
        config = CoarseOptimizationConfigV1(1, C.sizeof(CoarseOptimizationConfigV1),
                                            peak_mode, ncc_mode, 0, 0)
        self._check(function(self.handle, C.byref(config), C.sizeof(config), self.error, len(self.error)))
        self.coarse_optimization = dict(peak_mode=peak_mode, ncc_mode=ncc_mode)

    def set_coarse_overlap(self, *, enabled=True):
        if not self.handle:
            raise ValueError('frame is closed')
        if not isinstance(enabled, (bool, int)) or enabled not in (0, 1):
            raise ValueError('coarse overlap enabled must be boolean or integer 0/1')
        if not hasattr(self.dll, 'fpm_frame_set_coarse_overlap'):
            raise RuntimeError('Coarse overlap API unavailable; rebuild experimental library')
        function = self.dll.fpm_frame_set_coarse_overlap
        function.argtypes = [C.c_void_p, C.POINTER(CoarseOverlapConfigV1), C.c_size_t,
                             C.POINTER(C.c_char), C.c_size_t]
        function.restype = C.c_int
        config = CoarseOverlapConfigV1(1, C.sizeof(CoarseOverlapConfigV1), int(enabled), 0, 0, 0)
        self._check(function(self.handle, C.byref(config), C.sizeof(config), self.error, len(self.error)))
        self.coarse_overlap = dict(enabled=int(enabled))

    def coarse_overlap_metrics(self):
        if not self.handle:
            raise ValueError('frame is closed')
        if not hasattr(self.dll, 'fpm_frame_coarse_overlap_metrics'):
            raise RuntimeError('Coarse overlap API unavailable; rebuild experimental library')
        function = self.dll.fpm_frame_coarse_overlap_metrics
        function.argtypes = [C.c_void_p, C.POINTER(CoarseOverlapMetricsV1), C.c_size_t,
                             C.POINTER(C.c_char), C.c_size_t]
        function.restype = C.c_int
        metrics = CoarseOverlapMetricsV1()
        self._check(function(self.handle, C.byref(metrics), C.sizeof(metrics), self.error, len(self.error)))
        return _fields_snapshot(metrics)

    def _frame_snapshot(self):
        if not self.handle:
            raise ValueError('frame is closed')
        metrics, execution = FrameMetricsV1(), CoarseExecutionMetricsV1()
        self._check(self.dll.fpm_frame_get_metrics(self.handle, C.byref(metrics), C.sizeof(metrics),
            C.byref(execution), C.sizeof(execution), self.error, len(self.error)))
        return _fields_snapshot(metrics), _fields_snapshot(execution)

    @property
    def frame_metrics(self):
        return self._frame_snapshot()[0]

    @property
    def prepare_execution_metrics(self):
        return self._frame_snapshot()[1]

    def match(self, pattern, config, *, profile=False):
        handle = self._pattern(pattern)
        if not self._ready:
            raise ValueError('prepare a source frame successfully before matching')
        if config.max_targets < 1:
            raise ValueError('max_targets must be positive')
        results, count = (Result * config.max_targets)(), C.c_size_t()
        metrics, execution, fine = CoarseMetrics(), CoarseExecutionMetricsV1(), FineMetricsV1()
        stages = CoarseProfileV1() if profile else None
        self.last_metrics = self.last_profile = self.last_execution_metrics = self.last_fine_metrics = None
        self.last_overlap_metrics = None
        self._check(self.dll.fpm_frame_match(self.handle, handle, C.byref(config), results, len(results),
            C.byref(count), C.byref(metrics), C.byref(stages) if stages is not None else None,
            C.sizeof(stages) if stages is not None else 0, C.byref(execution), C.sizeof(execution),
            C.byref(fine), C.sizeof(fine), self.error, len(self.error)))
        self.last_metrics, self.last_execution_metrics, self.last_fine_metrics = map(_fields_snapshot, (metrics, execution, fine))
        self.last_profile = _fields_snapshot(stages) if stages is not None else None
        # Frozen earlier libraries remain usable: the additive getter is only
        # requested after this caller explicitly selected an overlap option.
        if hasattr(self, 'coarse_overlap'):
            self.last_overlap_metrics = self.coarse_overlap_metrics()
        return [dict(index=i, score=r.score, angle=r.angle, pos_x=r.center.x, pos_y=r.center.y,
                     corners=[[p.x, p.y] for p in r.corners]) for i, r in enumerate(results[:count.value])]

    def close(self):
        if self.handle:
            self.dll.fpm_frame_destroy(self.handle)
            self.handle = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
