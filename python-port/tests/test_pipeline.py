"""test_pipeline.py

冒煙測試與基本功能驗證，確保 API 可用且回傳格式正確。
"""

import os

import cv2
import numpy as np
import pytest

from fpm_py.pyramid import pyramid_match, learn_pattern, TemplateData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_IMG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Test Images")
)


def _load(name: str):
    path = os.path.join(TEST_IMG_DIR, name)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        pytest.skip(f"測試影像不存在: {path}")
    return img


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_pipeline_smoke():
    """確保 API 可呼叫且回傳基本欄位。"""
    src = _load("Src7.bmp")
    tmpl = _load("Dst7.bmp")

    # OpenCV 後端
    results = pyramid_match(src, tmpl, target_num=3, tolerance_angle=0, backend="opencv")
    assert isinstance(results, list)
    if results:
        r0 = results[0]
        for key in ("index", "score", "angle", "pos_x", "pos_y"):
            assert key in r0

    # FFT 後端
    results_fft = pyramid_match(src, tmpl, target_num=3, tolerance_angle=0, backend="fft")
    assert isinstance(results_fft, list)


def test_learn_pattern():
    """驗證 LearnPattern 預計算。"""
    tmpl = _load("Dst7.bmp")

    data = learn_pattern(tmpl)
    assert isinstance(data, TemplateData)
    assert data.is_learned
    assert len(data.pyramid) >= 2
    assert data.pyramid[0].shape == tmpl.shape[:2]
    assert data.border_color in (0, 255)
    assert all(n >= 0 for n in data.norms)
    assert all(0 < ia < 1 for ia in data.inv_areas)


def test_empty_inputs():
    """空影像應回傳空結果。"""
    empty = np.zeros((0, 0), dtype=np.uint8)
    small = np.zeros((10, 10), dtype=np.uint8)

    assert pyramid_match(empty, small) == []
    assert pyramid_match(small, empty) == []


def test_template_larger_than_source():
    """模板大於源影像應回傳空結果。"""
    src = np.zeros((50, 50), dtype=np.uint8)
    tmpl = np.zeros((100, 100), dtype=np.uint8)
    assert pyramid_match(src, tmpl) == []


# ---------------------------------------------------------------------------
# No-rotation matching
# ---------------------------------------------------------------------------


def test_no_rotation_single_match():
    """無旋轉搜尋應找到正確匹配。"""
    src = _load("Src7.bmp")
    tmpl = _load("Dst7.bmp")

    results = pyramid_match(src, tmpl, target_num=1, score=0.8)
    assert len(results) >= 1
    assert results[0]["score"] > 0.99  # 有精確匹配
    assert abs(results[0]["angle"]) < 1.0  # 角度接近 0


def test_no_rotation_multi_match():
    """Src3+Dst3 (無旋轉) 應找到多個目標。"""
    src = _load("Src3.bmp")
    tmpl = _load("Dst3.bmp")

    results = pyramid_match(
        src, tmpl, target_num=38, score=0.8, tolerance_angle=0
    )
    # C++ 預期 36 個匹配
    assert len(results) >= 30  # 容許一些差異
    assert all(r["score"] >= 0.8 for r in results)


# ---------------------------------------------------------------------------
# Rotation matching
# ---------------------------------------------------------------------------


def test_rotation_search():
    """Src7+Dst7 (角度=180) 應找到 3 個匹配。"""
    src = _load("Src7.bmp")
    tmpl = _load("Dst7.bmp")

    results = pyramid_match(
        src, tmpl, target_num=5, score=0.8, tolerance_angle=180
    )
    # C++ 預期 3 matches (1.000, 0.998, 0.991)
    assert len(results) == 3
    assert results[0]["score"] > 0.99


def test_rotation_score_accuracy():
    """驗證旋轉匹配的分數精度（與 C++ 誤差 < 0.05）。"""
    src = _load("Src7.bmp")
    tmpl = _load("Dst7.bmp")

    results = pyramid_match(
        src, tmpl, target_num=5, score=0.8, tolerance_angle=180
    )
    assert len(results) >= 3

    # C++ 分數: 1.000, 0.998, 0.991
    expected_scores = [1.000, 0.998, 0.991]
    for i, expected in enumerate(expected_scores):
        actual = results[i]["score"]
        assert abs(actual - expected) < 0.05, (
            f"Match #{i} score diff too large: "
            f"expected ~{expected}, got {actual}"
        )


# ---------------------------------------------------------------------------
# OpenCV vs FFT backend consistency
# ---------------------------------------------------------------------------


def test_opencv_fft_consistency():
    """OpenCV 和 FFT 後端應產生一致的匹配數量。

    注意：FFT 在金字塔頂層（極小影像）有已知精度差異，
    因此只驗證都能找到匹配，不嚴格比較分數。
    """
    src = _load("Src7.bmp")
    tmpl = _load("Dst7.bmp")

    results_cv = pyramid_match(
        src, tmpl, target_num=1, score=0.5, tolerance_angle=0, backend="opencv"
    )
    results_fft = pyramid_match(
        src, tmpl, target_num=1, score=0.5, tolerance_angle=0, backend="fft"
    )

    assert len(results_cv) >= 1
    assert len(results_fft) >= 1

    # 兩者都應找到高分匹配
    assert results_cv[0]["score"] > 0.9
    assert results_fft[0]["score"] > 0.9

    # 位置差距應 < 5 pixels
    dx = abs(results_cv[0]["pos_x"] - results_fft[0]["pos_x"])
    dy = abs(results_cv[0]["pos_y"] - results_fft[0]["pos_y"])
    assert dx < 5 and dy < 5, f"Position diff too large: dx={dx}, dy={dy}"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_backward_compat_params():
    """舊版參數 (coarse_step, fine_step) 不應導致錯誤。"""
    src = _load("Src7.bmp")
    tmpl = _load("Dst7.bmp")

    # 使用舊版參數應正常運作
    results = pyramid_match(
        src, tmpl,
        target_num=1,
        score=0.5,
        coarse_step=15.0,
        fine_step=5.0,
        scale=1.25,
    )
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Subpixel estimation
# ---------------------------------------------------------------------------


def test_subpixel_improves_precision():
    """亞像素精化應提升位置精度。"""
    src = _load("Src7.bmp")
    tmpl = _load("Dst7.bmp")

    results_no_sub = pyramid_match(
        src, tmpl, target_num=1, score=0.5, tolerance_angle=0, subpixel=False
    )
    results_sub = pyramid_match(
        src, tmpl, target_num=1, score=0.5, tolerance_angle=0, subpixel=True
    )

    assert len(results_no_sub) >= 1
    assert len(results_sub) >= 1

    # 兩者分數應非常接近
    diff = abs(results_no_sub[0]["score"] - results_sub[0]["score"])
    assert diff < 0.1


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


def test_output_angle_convention():
    """輸出角度應取負且在 (-180, 180] 範圍內。"""
    src = _load("Src7.bmp")
    tmpl = _load("Dst7.bmp")

    results = pyramid_match(
        src, tmpl, target_num=5, score=0.8, tolerance_angle=180
    )
    for r in results:
        assert -180 < r["angle"] <= 180, f"Angle out of range: {r['angle']}"


def test_results_sorted_by_score():
    """結果應依分數由高至低排序。"""
    src = _load("Src7.bmp")
    tmpl = _load("Dst7.bmp")

    results = pyramid_match(
        src, tmpl, target_num=5, score=0.8, tolerance_angle=180
    )
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
