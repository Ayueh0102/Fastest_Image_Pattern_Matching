"""Small ctypes binding for the isolated C++/Metal pose scorer (experimental ABI 1)."""
import ctypes as C
import math
from pathlib import Path
import sys
import time

import numpy as np


class Metrics(C.Structure):
    _fields_ = [("wall_ms", C.c_double), ("device_ms", C.c_double),
                ("host_buffer_bytes", C.c_uint64), ("gpu_buffer_bytes", C.c_uint64),
                ("gpu_allocated_bytes", C.c_uint64)]


FLOAT_P = C.POINTER(C.c_float)


class RefineConfig(C.Structure):
    _fields_ = [("max_iterations", C.c_int), ("optimize_scale", C.c_int),
                ("optimize_angle", C.c_int), ("max_translation", C.c_double),
                ("max_angle_degrees", C.c_double), ("max_log_scale", C.c_double)]


class RefineStats(C.Structure):
    _fields_ = [("iterations", C.c_int), ("evaluations", C.c_int), ("status", C.c_int)]


def library_path(build):
    extension = ".dylib" if sys.platform == "darwin" else ".dll" if sys.platform == "win32" else ".so"
    prefix = "" if sys.platform == "win32" else "lib"
    base = Path(build) / "experiments"
    for path in (base / (prefix + "fpm_pose_experiment" + extension),
                 base / "Release" / (prefix + "fpm_pose_experiment" + extension)):
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Build with -DFPM_BUILD_EXPERIMENTS=ON: {base}")


class Scorer:
    def __init__(self, build, source, template, backend="cpu", threads=1):
        self.lib = C.CDLL(str(library_path(build)))
        self.lib.fpm_pose_create.argtypes = [C.c_int, C.c_int, FLOAT_P, C.c_int, C.c_int,
                                             FLOAT_P, C.c_int, C.c_int, C.c_int, C.c_char_p, C.c_int]
        self.lib.fpm_pose_create.restype = C.c_void_p
        self.lib.fpm_pose_score.argtypes = [C.c_void_p, FLOAT_P, C.c_int, FLOAT_P,
                                            C.POINTER(Metrics), C.c_char_p, C.c_int]
        self.lib.fpm_pose_score.restype = C.c_int
        if hasattr(self.lib, 'fpm_pose_refine'):
            self.lib.fpm_pose_refine.argtypes = [C.c_void_p, FLOAT_P, C.c_int, C.POINTER(RefineConfig),
                FLOAT_P, FLOAT_P, C.POINTER(RefineStats), C.POINTER(Metrics), C.c_char_p, C.c_int]
            self.lib.fpm_pose_refine.restype = C.c_int
        self.lib.fpm_pose_device.argtypes = [C.c_void_p]
        self.lib.fpm_pose_device.restype = C.c_char_p
        self.lib.fpm_pose_memory.argtypes = [C.c_void_p, C.POINTER(Metrics)]
        self.lib.fpm_pose_memory.restype = C.c_int
        self.lib.fpm_pose_destroy.argtypes = [C.c_void_p]
        self.lib.fpm_pose_destroy.restype = None
        source, template = [np.ascontiguousarray(x, dtype=np.float32) for x in (source, template)]
        channels = source.shape[2] if source.ndim == 3 else 1
        if source.ndim not in (2, 3) or template.ndim != source.ndim or channels not in (1, 3):
            raise ValueError("Expected gray or RGB HWC arrays")
        if template.ndim == 3 and template.shape[2] != channels:
            raise ValueError("Channel counts must match")
        if backend not in ("cpu", "metal"):
            raise ValueError("backend must be cpu or metal")
        error = C.create_string_buffer(2048)
        start = time.perf_counter()
        self.handle = self.lib.fpm_pose_create(int(backend == "metal"), threads,
            source.ctypes.data_as(FLOAT_P), source.shape[1], source.shape[0],
            template.ctypes.data_as(FLOAT_P), template.shape[1], template.shape[0], channels, error, len(error))
        self.setup_ms = (time.perf_counter()-start)*1000
        if not self.handle:
            raise RuntimeError(error.value.decode())
        self.device = self.lib.fpm_pose_device(self.handle).decode()

    def score(self, poses):
        poses = np.ascontiguousarray(poses, dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] != 4:
            raise ValueError("Expected N x 4 poses: tx, ty, degrees, log_scale")
        scores = np.empty(len(poses), dtype=np.float32)
        metrics, error = Metrics(), C.create_string_buffer(2048)
        result = self.lib.fpm_pose_score(self.handle, poses.ctypes.data_as(FLOAT_P), len(poses),
            scores.ctypes.data_as(FLOAT_P), C.byref(metrics), error, len(error))
        if result:
            raise RuntimeError(error.value.decode())
        return scores, {key: getattr(metrics, key) for key, _ in Metrics._fields_}

    def close(self):
        if getattr(self, "handle", None):
            self.lib.fpm_pose_destroy(self.handle)
            self.handle = None

    def refine(self, seeds, optimize_scale=True, optimize_angle=True, max_iterations=30,
               max_translation=12., max_angle_degrees=15., max_log_scale=math.log(1.3)):
        """CPU analytic normalized-NCC LM, including when the scorer is Metal.

        Returns (poses, scores, per-seed statistics, timing/memory metrics).
        Bounds are relative to each input seed; this is a local optimizer.
        """
        seeds = np.ascontiguousarray(seeds, dtype=np.float32)
        if seeds.ndim != 2 or seeds.shape[1] != 4:
            raise ValueError('Expected N x 4 poses')
        config = RefineConfig(max_iterations, int(optimize_scale), int(optimize_angle),
                              max_translation, max_angle_degrees, max_log_scale)
        poses, scores = np.empty_like(seeds), np.empty(len(seeds), np.float32)
        stats = (RefineStats * len(seeds))()
        metrics, error = Metrics(), C.create_string_buffer(2048)
        result = self.lib.fpm_pose_refine(self.handle, seeds.ctypes.data_as(FLOAT_P), len(seeds),
            C.byref(config), poses.ctypes.data_as(FLOAT_P), scores.ctypes.data_as(FLOAT_P),
            stats, C.byref(metrics), error, len(error))
        if result:
            raise RuntimeError(error.value.decode())
        return poses, scores, [{key: getattr(row, key) for key, _ in RefineStats._fields_} for row in stats], \
            {key: getattr(metrics, key) for key, _ in Metrics._fields_}

    def memory(self):
        metrics = Metrics()
        if self.lib.fpm_pose_memory(self.handle, C.byref(metrics)):
            raise RuntimeError("Invalid scorer handle")
        return {key: getattr(metrics, key) for key, _ in Metrics._fields_ if key.endswith('bytes')}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
