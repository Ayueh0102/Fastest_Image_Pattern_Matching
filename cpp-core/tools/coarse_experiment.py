"""Standard-library ctypes adapter for opt-in Metal dense coarse NCC.

CoarsePattern.match has the portable Pattern interface, and returns identical
fields. last_metrics identifies the precise GPU portion; CPU pyramid, rotation,
normalization, peak selection and fine search still execute every call.
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
    def __init__(self, build, image, min_area=256):
        self.dll = C.CDLL(str(library_path(build)))
        self.handle = C.c_void_p()
        self.dll.fpm_coarse_create.argtypes = [Image, C.c_int, C.POINTER(C.c_void_p),
                                               C.POINTER(CoarseMetrics), C.POINTER(C.c_char), C.c_size_t]
        self.dll.fpm_coarse_create.restype = C.c_int
        self.dll.fpm_coarse_destroy.argtypes = [C.c_void_p]
        self.dll.fpm_coarse_destroy.restype = None
        self.dll.fpm_coarse_match.argtypes = [C.c_void_p, Image, C.POINTER(Config), C.POINTER(Result),
            C.c_size_t, C.POINTER(C.c_size_t), C.POINTER(CoarseMetrics), C.POINTER(C.c_char), C.c_size_t]
        self.dll.fpm_coarse_match.restype = C.c_int
        self.dll.fpm_coarse_scoremap.argtypes = [C.c_void_p, Image, C.POINTER(C.c_double), C.c_size_t,
            C.POINTER(C.c_size_t), C.POINTER(CoarseMetrics), C.POINTER(C.c_char), C.c_size_t]
        self.dll.fpm_coarse_scoremap.restype = C.c_int
        self.error = C.create_string_buffer(1024)
        self.metrics = CoarseMetrics()
        self._check(self.dll.fpm_coarse_create(image, min_area, C.byref(self.handle),
            C.byref(self.metrics), self.error, len(self.error)))
        self.setup_metrics = self.last_metrics

    @property
    def last_metrics(self):
        return {name: getattr(self.metrics, name) for name, _ in self.metrics._fields_}

    def _check(self, status):
        if status:
            raise RuntimeError(self.error.value.decode('utf-8', errors='replace') or f'FPM status {status}')

    def match(self, image, config, *, prune=False):
        if not self.handle:
            raise ValueError('pattern is closed')
        if config.max_targets < 1:
            raise ValueError('max_targets must be positive')
        # Metal returns complete exact scoremaps. Conservative CPU pruning is
        # unnecessary for result equivalence and is not a GPU optimization here.
        del prune
        results = (Result * config.max_targets)()
        count = C.c_size_t()
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
