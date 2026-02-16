"""bench_speed.py

C++ vs Python 運算速度對比實驗。

自動執行 C++ MatchTool CLI 與 Python pyramid_match()，
比較不同後端在各種場景下的執行速度，輸出 Markdown 格式表格。

使用方式：
    cd python-port
    source venv/bin/activate
    PYTHONPATH=src python3 benchmarks/bench_speed.py
    PYTHONPATH=src python3 benchmarks/bench_speed.py --repeats 5
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IMG_DIR = REPO_ROOT / "Test Images"
CPP_BIN = REPO_ROOT / "macos-cli" / "build" / "MatchTool"

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES = [
    {
        "label": "Src7+Dst7 旋轉180°",
        "tag": "大圖大模板+旋轉",
        "src": "Src7.bmp",
        "tmpl": "Dst7.bmp",
        "target_num": 5,
        "score": 0.8,
        "angle": 180,
        "overlap": 0.0,
        "min_area": 256,
    },
    {
        "label": "Src3+Dst3 無旋轉",
        "tag": "小模板多目標",
        "src": "Src3.bmp",
        "tmpl": "Dst3.bmp",
        "target_num": 38,
        "score": 0.8,
        "angle": 0,
        "overlap": 0.0,
        "min_area": 256,
    },
    {
        "label": "Src1+Dst1 旋轉180°",
        "tag": "大圖+旋轉",
        "src": "Src1.bmp",
        "tmpl": "Dst1.bmp",
        "target_num": 10,
        "score": 0.8,
        "angle": 180,
        "overlap": 0.1,
        "min_area": 256,
    },
    {
        "label": "Src8+Dst8 旋轉180°",
        "tag": "中等+旋轉",
        "src": "Src8.bmp",
        "tmpl": "Dst8.bmp",
        "target_num": 5,
        "score": 0.8,
        "angle": 180,
        "overlap": 0.8,
        "min_area": 256,
    },
    {
        "label": "Src4+Dst4 旋轉180°",
        "tag": "多目標+旋轉",
        "src": "Src4.bmp",
        "tmpl": "Dst4.bmp",
        "target_num": 70,
        "score": 0.5,
        "angle": 180,
        "overlap": 0.0,
        "min_area": 256,
    },
]

# ---------------------------------------------------------------------------
# C++ runner
# ---------------------------------------------------------------------------

RE_TIME = re.compile(r"Execution Time:\s*(\d+)\s*ms")
RE_FOUND = re.compile(r"Found:\s*(\d+)\s*match")


def run_cpp(case: dict, simd: bool = True, repeats: int = 3) -> dict:
    """執行 C++ MatchTool CLI，回傳 {time_ms, matches, times}。"""
    if not CPP_BIN.exists():
        return {"time_ms": -1, "matches": -1, "times": [], "error": "binary not found"}

    src_path = str(IMG_DIR / case["src"])
    tmpl_path = str(IMG_DIR / case["tmpl"])

    cmd = [
        str(CPP_BIN),
        "--src", src_path,
        "--tmpl", tmpl_path,
        "--target-num", str(case["target_num"]),
        "--score", str(case["score"]),
        "--angle", str(case["angle"]),
        "--overlap", str(case.get("overlap", 0)),
        "--min-area", str(case.get("min_area", 256)),
        "--no-display",
    ]
    if not simd:
        cmd.append("--no-simd")

    times = []
    matches = -1

    for _ in range(repeats):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            stdout = result.stdout

            m_time = RE_TIME.search(stdout)
            m_found = RE_FOUND.search(stdout)

            if m_time:
                times.append(int(m_time.group(1)))
            if m_found:
                matches = int(m_found.group(1))
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return {"time_ms": -1, "matches": -1, "times": [], "error": str(e)}

    if not times:
        return {"time_ms": -1, "matches": matches, "times": []}

    return {
        "time_ms": statistics.median(times),
        "matches": matches,
        "times": times,
    }


# ---------------------------------------------------------------------------
# Python runner
# ---------------------------------------------------------------------------


def run_python(case: dict, backend: str = "opencv", repeats: int = 3) -> dict:
    """執行 Python pyramid_match()，回傳 {time_ms, matches, times}。"""
    from fpm_py.pyramid import pyramid_match

    src_path = str(IMG_DIR / case["src"])
    tmpl_path = str(IMG_DIR / case["tmpl"])

    src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    tmpl = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)

    if src is None or tmpl is None:
        return {"time_ms": -1, "matches": -1, "times": [], "error": "image not found"}

    times = []
    matches = 0

    for _ in range(repeats):
        t0 = time.perf_counter()
        results = pyramid_match(
            src,
            tmpl,
            target_num=case["target_num"],
            score=case["score"],
            max_overlap=case.get("overlap", 0.5),
            tolerance_angle=case["angle"],
            min_reduced_area=case.get("min_area", 256),
            backend=backend,
        )
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
        matches = len(results)

    return {
        "time_ms": statistics.median(times),
        "matches": matches,
        "times": times,
    }


# ---------------------------------------------------------------------------
# Image size helper
# ---------------------------------------------------------------------------


def get_image_size(name: str) -> str:
    """回傳 'WxH' 格式的影像尺寸。"""
    path = str(IMG_DIR / name)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "N/A"
    h, w = img.shape[:2]
    return f"{w}x{h}"


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_time(ms: float) -> str:
    if ms < 0:
        return "N/A"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s"


def format_ratio(a: float, b: float) -> str:
    if a <= 0 or b <= 0:
        return "N/A"
    return f"{b / a:.1f}x"


def print_results(all_results: list, repeats: int) -> None:
    """輸出 Markdown 格式的對比表格。"""
    print()
    print("=" * 100)
    print("  C++ vs Python 運算速度對比實驗")
    print(f"  每個配置執行 {repeats} 次取中位數")
    print("=" * 100)
    print()

    # ── 速度對比表 ──
    print("## 速度對比")
    print()
    header = (
        "| 測試案例 | 圖像尺寸 | 模板尺寸 | "
        "C++ SIMD | C++ noSIMD | Py OpenCV | Py FFT | "
        "SIMD加速 | C++/Py比 |"
    )
    sep = "|" + "|".join(["---"] * 9) + "|"
    print(header)
    print(sep)

    for row in all_results:
        c = row["case"]
        cpp_s = row["cpp_simd"]["time_ms"]
        cpp_n = row["cpp_nosimd"]["time_ms"]
        py_cv = row["py_opencv"]["time_ms"]
        py_fft = row["py_fft"]["time_ms"]

        simd_speedup = format_ratio(cpp_s, cpp_n)
        # C++/Py 比 = Py_OpenCV / C++_SIMD
        cpp_py_ratio = format_ratio(cpp_s, py_cv)

        print(
            f"| {c['label']} "
            f"| {row['src_size']} "
            f"| {row['tmpl_size']} "
            f"| {format_time(cpp_s)} "
            f"| {format_time(cpp_n)} "
            f"| {format_time(py_cv)} "
            f"| {format_time(py_fft)} "
            f"| {simd_speedup} "
            f"| {cpp_py_ratio} |"
        )

    # ── 匹配數量驗證表 ──
    print()
    print("## 匹配數量驗證")
    print()
    header2 = "| 測試案例 | C++ SIMD | C++ noSIMD | Py OpenCV | Py FFT | 一致? |"
    sep2 = "|" + "|".join(["---"] * 6) + "|"
    print(header2)
    print(sep2)

    for row in all_results:
        c = row["case"]
        m_cs = row["cpp_simd"]["matches"]
        m_cn = row["cpp_nosimd"]["matches"]
        m_po = row["py_opencv"]["matches"]
        m_pf = row["py_fft"]["matches"]

        vals = [v for v in [m_cs, m_cn, m_po, m_pf] if v >= 0]
        consistent = len(set(vals)) <= 1 if vals else False

        print(
            f"| {c['label']} "
            f"| {m_cs} "
            f"| {m_cn} "
            f"| {m_po} "
            f"| {m_pf} "
            f"| {'Yes' if consistent else 'No'} |"
        )

    # ── 分析摘要 ──
    print()
    print("## 分析摘要")
    print()

    # 計算平均加速比
    simd_ratios = []
    cpp_py_ratios = []
    fft_cv_ratios = []

    for row in all_results:
        cpp_s = row["cpp_simd"]["time_ms"]
        cpp_n = row["cpp_nosimd"]["time_ms"]
        py_cv = row["py_opencv"]["time_ms"]
        py_fft = row["py_fft"]["time_ms"]

        if cpp_s > 0 and cpp_n > 0:
            simd_ratios.append(cpp_n / cpp_s)
        if cpp_s > 0 and py_cv > 0:
            cpp_py_ratios.append(py_cv / cpp_s)
        if py_cv > 0 and py_fft > 0:
            fft_cv_ratios.append(py_fft / py_cv)

    if simd_ratios:
        avg_simd = statistics.mean(simd_ratios)
        print(f"- **SIMD 加速比**（C++ noSIMD / C++ SIMD）: 平均 {avg_simd:.1f}x")
    if cpp_py_ratios:
        avg_cpp_py = statistics.mean(cpp_py_ratios)
        print(f"- **C++ vs Python** (Py OpenCV / C++ SIMD): 平均 {avg_cpp_py:.1f}x 慢")
    if fft_cv_ratios:
        avg_fft_cv = statistics.mean(fft_cv_ratios)
        if avg_fft_cv > 1:
            print(f"- **FFT vs OpenCV** (Py FFT / Py OpenCV): 平均 {avg_fft_cv:.1f}x 慢")
        else:
            print(f"- **FFT vs OpenCV** (Py FFT / Py OpenCV): 平均 {1/avg_fft_cv:.1f}x 快")

    # 每個案例的詳細時間
    print()
    print("## 詳細計時（所有迭代）")
    print()
    for row in all_results:
        c = row["case"]
        print(f"### {c['label']}")
        for label, key in [
            ("C++ SIMD", "cpp_simd"),
            ("C++ noSIMD", "cpp_nosimd"),
            ("Py OpenCV", "py_opencv"),
            ("Py FFT", "py_fft"),
        ]:
            times = row[key].get("times", [])
            if times:
                ts = ", ".join(f"{t:.0f}" for t in times)
                print(f"  - {label}: [{ts}] ms")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="C++ vs Python 運算速度對比實驗"
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="每個配置的執行次數（取中位數）",
    )
    parser.add_argument(
        "--cases", type=str, default=None,
        help="指定測試案例索引（如 '0,1,3'），預設全部",
    )
    parser.add_argument(
        "--skip-cpp", action="store_true",
        help="跳過 C++ 測試（僅跑 Python）",
    )
    args = parser.parse_args()

    # 選擇測試案例
    if args.cases:
        indices = [int(x.strip()) for x in args.cases.split(",")]
        cases = [CASES[i] for i in indices]
    else:
        cases = CASES

    # 檢查前置條件
    if not IMG_DIR.exists():
        print(f"Error: Test Images 目錄不存在: {IMG_DIR}", file=sys.stderr)
        sys.exit(1)

    if not args.skip_cpp and not CPP_BIN.exists():
        print(f"Warning: C++ binary 不存在: {CPP_BIN}", file=sys.stderr)
        print("  將跳過 C++ 測試。用 --skip-cpp 隱藏此警告。", file=sys.stderr)
        args.skip_cpp = True

    repeats = args.repeats

    all_results = []

    for i, case in enumerate(cases):
        print(
            f"\n[{i+1}/{len(cases)}] {case['label']} ({case['tag']})",
            file=sys.stderr,
        )

        row = {
            "case": case,
            "src_size": get_image_size(case["src"]),
            "tmpl_size": get_image_size(case["tmpl"]),
        }

        # C++ SIMD
        if not args.skip_cpp:
            print("  C++ SIMD ...", end="", flush=True, file=sys.stderr)
            row["cpp_simd"] = run_cpp(case, simd=True, repeats=repeats)
            print(
                f" {format_time(row['cpp_simd']['time_ms'])}",
                file=sys.stderr,
            )

            print("  C++ noSIMD ...", end="", flush=True, file=sys.stderr)
            row["cpp_nosimd"] = run_cpp(case, simd=False, repeats=repeats)
            print(
                f" {format_time(row['cpp_nosimd']['time_ms'])}",
                file=sys.stderr,
            )
        else:
            row["cpp_simd"] = {"time_ms": -1, "matches": -1, "times": []}
            row["cpp_nosimd"] = {"time_ms": -1, "matches": -1, "times": []}

        # Python OpenCV
        print("  Py OpenCV ...", end="", flush=True, file=sys.stderr)
        row["py_opencv"] = run_python(case, backend="opencv", repeats=repeats)
        print(
            f" {format_time(row['py_opencv']['time_ms'])}",
            file=sys.stderr,
        )

        # Python FFT
        print("  Py FFT ...", end="", flush=True, file=sys.stderr)
        row["py_fft"] = run_python(case, backend="fft", repeats=repeats)
        print(
            f" {format_time(row['py_fft']['time_ms'])}",
            file=sys.stderr,
        )

        all_results.append(row)

    print_results(all_results, repeats)


if __name__ == "__main__":
    main()
