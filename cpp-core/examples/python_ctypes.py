"""A complete standard-library-only binding to ABI 1 (no pybind11).

Run: python3 python_ctypes.py /absolute/path/libfpm_c.dylib
Replace .dylib with .so on Linux or fpm_c.dll on Windows.
"""
from __future__ import annotations

import ctypes as C
import json
import sys


class Image(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_uint8)), ("width", C.c_int),
                ("height", C.c_int), ("stride", C.c_size_t)]


class Point(C.Structure):
    _fields_ = [("x", C.c_double), ("y", C.c_double)]


class AngleRange(C.Structure):
    _fields_ = [("min", C.c_double), ("max", C.c_double)]


class Config(C.Structure):
    _fields_ = [("max_targets", C.c_int), ("min_score", C.c_double),
                ("max_overlap", C.c_double), ("tolerance_angle", C.c_double),
                ("subpixel", C.c_int), ("invert_source", C.c_int), ("scalar", C.c_int),
                ("stop_layer", C.c_int),
                ("ranges", C.POINTER(AngleRange)), ("range_count", C.c_size_t)]


class Result(C.Structure):
    _fields_ = [("corners", Point * 4), ("center", Point),
                ("angle", C.c_double), ("score", C.c_double)]


class Library:
    def __init__(self, path: str):
        self.dll = C.CDLL(path)
        self.dll.fpm_abi_version.argtypes = []
        self.dll.fpm_abi_version.restype = C.c_uint
        if self.dll.fpm_abi_version() != 1:
            raise RuntimeError("unsupported FPM ABI")
        self.dll.fpm_default_config.argtypes = []
        self.dll.fpm_default_config.restype = Config
        self.dll.fpm_learn.argtypes = [Image, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_char), C.c_size_t]
        self.dll.fpm_learn.restype = C.c_int
        self.dll.fpm_destroy.argtypes = [C.c_void_p]
        self.dll.fpm_destroy.restype = None
        self.dll.fpm_match.argtypes = [Image, C.c_void_p, C.POINTER(Config), C.POINTER(Result),
                                      C.c_size_t, C.POINTER(C.c_size_t), C.POINTER(C.c_char), C.c_size_t]
        self.dll.fpm_match.restype = C.c_int
        if hasattr(self.dll, 'fpm_match_pruned'):
            self.dll.fpm_match_pruned.argtypes = self.dll.fpm_match.argtypes
            self.dll.fpm_match_pruned.restype = C.c_int

    def config(self) -> Config:
        return self.dll.fpm_default_config()

    def learn(self, image: Image, min_area: int = 256) -> "Pattern":
        return Pattern(self, image, min_area)


class Pattern:
    def __init__(self, library: Library, image: Image, min_area: int):
        self.library = library
        self.handle = C.c_void_p()
        error = C.create_string_buffer(1024)
        status = library.dll.fpm_learn(image, min_area, C.byref(self.handle), error, len(error))
        if status:
            raise ValueError(error.value.decode("utf-8", errors="replace"))

    def match(self, image: Image, config: Config | None = None, *, prune: bool = False) -> list[dict]:
        if not self.handle:
            raise ValueError("pattern is closed")
        config = config if config is not None else self.library.config()
        if config.max_targets < 1:
            raise ValueError("max_targets must be positive")
        results = (Result * config.max_targets)()
        count = C.c_size_t()
        error = C.create_string_buffer(1024)
        function = self.library.dll.fpm_match_pruned if prune else self.library.dll.fpm_match
        status = function(image, self.handle, C.byref(config), results,
                                            len(results), C.byref(count), error, len(error))
        if status:
            raise ValueError(error.value.decode("utf-8", errors="replace") or f"FPM status {status}")
        return [{"index": i, "score": r.score, "angle": r.angle,
                 "pos_x": r.center.x, "pos_y": r.center.y,
                 "corners": [[p.x, p.y] for p in r.corners]}
                for i, r in enumerate(results[:count.value])]

    def close(self):
        if self.handle:
            self.library.dll.fpm_destroy(self.handle)
            self.handle = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def demo(path: str):
    library = Library(path)
    # Keep ctypes/NumPy buffers alive until match returns. Learning copies them.
    pixels = (C.c_uint8 * 64)(*((i * 31 + i * i) % 256 for i in range(64)))
    image = Image(pixels, 8, 8, 8)
    config = library.config()
    config.max_targets, config.min_score = 1, 0.99
    with library.learn(image) as pattern:
        results = pattern.match(image, config)
    assert len(results) == 1 and results[0]["pos_x"] == 4
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    demo(sys.argv[1])
