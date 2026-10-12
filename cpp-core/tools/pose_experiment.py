"""Small ctypes binding for the isolated C++/Metal pose scorer (experimental ABI 1)."""
import ctypes as C
from pathlib import Path
import sys
import time

import numpy as np


class Metrics(C.Structure):
    _fields_ = [("wall_ms", C.c_double), ("device_ms", C.c_double),
                ("host_buffer_bytes", C.c_uint64), ("gpu_buffer_bytes", C.c_uint64),
                ("gpu_allocated_bytes", C.c_uint64)]


FLOAT_P = C.POINTER(C.c_float)


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
