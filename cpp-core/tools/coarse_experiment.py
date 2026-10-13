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
