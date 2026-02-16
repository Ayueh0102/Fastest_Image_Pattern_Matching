"""rotation.py

角度處理：旋轉樣板或來源影像並回傳旋轉後影像與適配之邊界尺寸。
採用 OpenCV 仿射轉換，並對邊界做自動擴張以避免裁切。
"""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


def rotate_template(tmpl: np.ndarray, angle_deg: float, border_value: int = 0) -> np.ndarray:
    """旋轉樣板：以中心為軸，回傳完整保留內容之影像。

    使用 BORDER_CONSTANT 填充模式，與 C++ 實作一致
    （對應 C++ 的 BORDER_CONSTANT + Scalar(pTemplData->iBorderColor)）。

    參數
    -----
    tmpl : np.ndarray
        樣板影像（灰階）。
    angle_deg : float
        旋轉角度（度）。
    border_value : int
        邊界填充值，預設為 0（黑色）。
    """
    h, w = tmpl.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # 先計算旋轉後的外接矩形尺寸，避免裁切
    rad = math.radians(angle_deg)
    cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
    new_w = int(w * cos_a + h * sin_a)
    new_h = int(h * cos_a + w * sin_a)

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    # 位移使得旋轉後影像居中於新畫布
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    rotated = cv2.warpAffine(
        tmpl, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated


def rotate_source(src: np.ndarray, angle_deg: float, tmpl_shape: Tuple[int, ...], border_value: int = 0) -> np.ndarray:
    """旋轉來源影像以進行頂層搜尋。

    對應 C++ 的 GetBestRotationSize + warpAffine
    （match_core.cpp:196-209, 519-563）。

    計算旋轉後的最佳畫布尺寸，回傳旋轉後的來源影像。

    參數
    -----
    src : np.ndarray
        來源影像。
    angle_deg : float
        旋轉角度（度）。
    tmpl_shape : tuple
        樣板影像的形狀 (h, w, ...)。
    border_value : int
        邊界填充值，預設為 0（黑色）。

    回傳
    -----
    np.ndarray
        旋轉後的來源影像，畫布大小為完整外接矩形。
    """
    h, w = src.shape[:2]
    th, tw = tmpl_shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    rad = math.radians(angle_deg)
    cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))

    # Full rotated bounding box
    new_w = int(math.ceil(w * cos_a + h * sin_a))
    new_h = int(math.ceil(h * cos_a + w * sin_a))

    # Build rotation matrix with translation to center in new canvas
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M[0, 2] += (new_w - 1) / 2.0 - cx
    M[1, 2] += (new_h - 1) / 2.0 - cy

    rotated = cv2.warpAffine(
        src, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated
