"""nms.py

非極大值抑制（NMS）/ 去重疊：用於過濾多重偵測結果。
此處以矩形重疊度（IoU/Overlap）為準。

提供兩組 API：
  - rect_iou / nms：適用於軸對齊矩形（angle=0）。
  - rotated_rect_overlap / nms_rotated：適用於帶旋轉角度的矩形，
    與 C++ FilterWithRotatedRect 邏輯一致。
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
#: Axis-aligned rect: (x, y, w, h)
AARect = Tuple[int, int, int, int]

#: OpenCV RotatedRect tuple: ((cx, cy), (w, h), angle_degrees)
RotatedRectTuple = Tuple[Tuple[float, float], Tuple[float, float], float]


# ---------------------------------------------------------------------------
# Axis-aligned helpers (backward compatible)
# ---------------------------------------------------------------------------

def rect_iou(a: AARect, b: AARect) -> float:
    """計算兩矩形 IoU（x, y, w, h）。"""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(
    boxes: List[AARect],
    scores: List[float],
    max_overlap: float = 0.5,
) -> List[int]:
    """根據重疊度過濾框，回傳保留之索引（由高分到低分）。"""
    order = np.argsort(scores)[::-1]
    keep: List[int] = []
    for idx in order:
        ok = True
        for kept_idx in keep:
            if rect_iou(boxes[idx], boxes[kept_idx]) > max_overlap:
                ok = False
                break
        if ok:
            keep.append(int(idx))
    return keep


# ---------------------------------------------------------------------------
# Rotated-rectangle helpers (matches C++ FilterWithRotatedRect)
# ---------------------------------------------------------------------------

def rotated_rect_overlap(
    rect1: RotatedRectTuple,
    rect2: RotatedRectTuple,
) -> float:
    """計算兩個旋轉矩形的重疊比例（intersection_area / rect1_area）。

    此函式對應 C++ ``FilterWithRotatedRect`` 中的重疊判定邏輯——
    注意分母是 **rect1** 的面積，而非 IoU。

    Parameters
    ----------
    rect1 : RotatedRectTuple
        ``((cx, cy), (w, h), angle_degrees)``，標準 OpenCV RotatedRect 格式。
    rect2 : RotatedRectTuple
        同上。

    Returns
    -------
    float
        重疊比例，介於 0.0 ~ 1.0 之間。
        ``INTERSECT_NONE``  -> 0.0
        ``INTERSECT_FULL``  -> ``min(area1, area2) / area1``
        部分交集           -> ``intersection_area / area1``
    """
    area1 = rect1[1][0] * rect1[1][1]  # w * h
    if area1 <= 0:
        return 0.0

    ret_code, intersection_pts = cv2.rotatedRectangleIntersection(rect1, rect2)

    if ret_code == cv2.INTERSECT_NONE or intersection_pts is None:
        return 0.0

    if ret_code == cv2.INTERSECT_FULL:
        area2 = rect2[1][0] * rect2[1][1]
        return float(min(area1, area2) / area1)

    # INTERSECT_PARTIAL
    pts = intersection_pts.reshape(-1, 2)
    if len(pts) < 3:
        return 0.0

    hull = cv2.convexHull(pts.astype(np.float32))
    inter_area = cv2.contourArea(hull)
    return float(inter_area / area1)


def nms_rotated(
    rects: List[RotatedRectTuple],
    scores: List[float],
    max_overlap: float = 0.5,
) -> List[int]:
    """對旋轉矩形執行 Greedy NMS，回傳保留之索引。

    與 C++ ``FilterWithRotatedRect`` 邏輯一致：
      1. 依分數由高至低排序。
      2. 逐一檢查：若該矩形與所有已保留矩形的重疊度均 <= max_overlap，則保留。

    Parameters
    ----------
    rects : list of RotatedRectTuple
        ``((cx, cy), (w, h), angle_degrees)`` 格式之旋轉矩形清單。
    scores : list of float
        對應每個矩形的匹配分數。
    max_overlap : float, optional
        允許的最大重疊比例，預設 0.5。

    Returns
    -------
    list of int
        保留的索引（依分數由高至低排列）。
    """
    if len(rects) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep: List[int] = []

    for idx in order:
        idx = int(idx)
        ok = True
        for kept_idx in keep:
            overlap = rotated_rect_overlap(rects[idx], rects[kept_idx])
            if overlap > max_overlap:
                ok = False
                break
        if ok:
            keep.append(idx)

    return keep


# ---------------------------------------------------------------------------
# Score filtering
# ---------------------------------------------------------------------------

def filter_with_score(
    scores: List[float],
    threshold: float,
) -> List[int]:
    """回傳分數 >= threshold 的索引，依分數由高至低排列。

    Parameters
    ----------
    scores : list of float
        匹配分數清單。
    threshold : float
        最低分數門檻。

    Returns
    -------
    list of int
        滿足條件之索引（由高分到低分）。
    """
    arr = np.asarray(scores, dtype=np.float64)
    mask = arr >= threshold
    indices = np.where(mask)[0]
    # Sort by score descending
    sorted_indices = indices[np.argsort(arr[indices])[::-1]]
    return [int(i) for i in sorted_indices]
