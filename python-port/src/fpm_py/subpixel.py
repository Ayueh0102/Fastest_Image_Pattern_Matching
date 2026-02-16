"""subpixel.py

次像素峰值估計模組，提供二維與三維（含角度）二次曲面擬合方法。

方法一 — 2D 二次曲面擬合 (refine_peak_subpixel)：
  使用 3x3 鄰域（9 點）擬合六參數二次曲面
    z = a*x² + b*y² + c*x*y + d*x + e*y + f
  以 SVD-based least squares 求解後，由解析公式求峰值偏移 (dx, dy)。

方法二 — 3D 二次曲面擬合 (refine_peak_3d)：
  使用 3x3x3 鄰域（27 點，含三個角度層）擬合十參數二次曲面
    z = a*x² + b*y² + c*θ² + d*x*y + e*x*θ + f*y*θ + g*x + h*y + i*θ + j
  對應 C++ ``CMatchCore::SubPixEstimation`` (match_core.cpp:712-764)。
  透過 Hessian 矩陣與梯度向量求解三維峰值偏移 (dx, dy, dθ)。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def refine_peak_subpixel(
    response: np.ndarray, x: int, y: int
) -> Tuple[float, float]:
    """在 *response* 上的整數峰值 ``(x, y)`` 周圍做 2D 次像素估計。

    使用 3x3 鄰域共 9 點擬合二次曲面：

        z = a*x² + b*y² + c*x*y + d*x + e*y + f

    以 ``np.linalg.lstsq`` (SVD) 求解，再由解析公式算出峰值偏移量。

    Parameters
    ----------
    response : np.ndarray
        二維相關分數圖，shape ``(H, W)``。
    x : int
        峰值所在的整數 x 座標（列索引）。
    y : int
        峰值所在的整數 y 座標（行索引）。

    Returns
    -------
    Tuple[float, float]
        ``(dx, dy)``，次像素偏移量，範圍限制在 ``[-1, 1]``。
        若鄰域不足則回傳 ``(0.0, 0.0)``。
    """
    h, w = response.shape[:2]
    if not (1 <= x < w - 1 and 1 <= y < h - 1):
        return 0.0, 0.0

    # 擷取 3x3 鄰域
    patch = response[y - 1 : y + 2, x - 1 : x + 2].astype(np.float64)

    # 建立設計矩陣 A (9x6)，相對座標 (-1,0,+1)
    # 行排列順序：dy=-1..+1 (外), dx=-1..+1 (內)
    # 各列：[x², y², x*y, x, y, 1]
    A = np.empty((9, 6), dtype=np.float64)
    S = np.empty(9, dtype=np.float64)
    row = 0
    for oy in range(-1, 2):
        for ox in range(-1, 2):
            A[row] = [ox * ox, oy * oy, ox * oy, ox, oy, 1.0]
            S[row] = patch[oy + 1, ox + 1]
            row += 1

    # SVD-based least squares 求解：z = [a, b, c, d, e, f]
    z, _, _, _ = np.linalg.lstsq(A, S, rcond=None)
    a, b, c, d, e, f = z

    # 峰值位置解析解：∂z/∂x = 2a*x + c*y + d = 0, ∂z/∂y = c*x + 2b*y + e = 0
    # dx = (c*e - 2*b*d) / (4*a*b - c²)
    # dy = (c*d - 2*a*e) / (4*a*b - c²)
    denom = 4.0 * a * b - c * c
    if abs(denom) < 1e-12:
        return 0.0, 0.0

    dx = (c * e - 2.0 * b * d) / denom
    dy = (c * d - 2.0 * a * e) / denom

    # 限制偏移量，避免不穩定情況
    dx = float(np.clip(dx, -1.0, 1.0))
    dy = float(np.clip(dy, -1.0, 1.0))
    return dx, dy


def refine_peak_3d(
    scores_3x3x3: np.ndarray,
    center_x: float,
    center_y: float,
    center_angle: float,
    angle_step: float,
) -> Tuple[float, float, float]:
    """三維次像素估計：同時精修 x, y 與角度。

    對應 C++ ``CMatchCore::SubPixEstimation`` (match_core.cpp:712-764)。

    使用 27 點（3 個角度 x 3x3 空間鄰域）擬合十參數二次曲面：

        z = a*x² + b*y² + c*θ² + d*x*y + e*x*θ + f*y*θ + g*x + h*y + i*θ + j

    其中 x, y 使用絕對座標（center ± 1 pixel），θ 使用弧度值
    （center_angle ± angle_step 轉換為弧度）。

    透過 Hessian 矩陣與梯度向量求解最佳峰值位置後，計算相對於中心的偏移量。

    Parameters
    ----------
    scores_3x3x3 : np.ndarray
        匹配分數陣列，shape ``(3, 3, 3)``。
        索引 ``[theta_idx, y_offset+1, x_offset+1]``，其中：
        - ``theta_idx`` 0/1/2 對應 center_angle - angle_step / center_angle / + angle_step
        - ``y_offset`` -1/0/+1（行偏移）
        - ``x_offset`` -1/0/+1（列偏移）
    center_x : float
        中心峰值的 x 座標。
    center_y : float
        中心峰值的 y 座標。
    center_angle : float
        中心峰值的角度（度）。
    angle_step : float
        角度搜尋步長（度）。

    Returns
    -------
    Tuple[float, float, float]
        ``(dx, dy, dtheta_degrees)``，分別為 x, y 偏移量（像素）
        與角度偏移量（度）。若矩陣奇異無法求解則回傳 ``(0.0, 0.0, 0.0)``。
    """
    try:
        # 建立設計矩陣 A (27x10) 與響應向量 S (27,)
        # 採用絕對座標，與 C++ 實作一致
        A = np.empty((27, 10), dtype=np.float64)
        S = np.empty(27, dtype=np.float64)

        row = 0
        for theta_idx in range(3):
            # θ 值（弧度）：center_angle + (theta_idx - 1) * angle_step
            theta_rad = np.deg2rad(center_angle + (theta_idx - 1) * angle_step)
            for oy in range(-1, 2):
                for ox in range(-1, 2):
                    xi = center_x + ox
                    yi = center_y + oy
                    ti = theta_rad

                    # [x², y², θ², x*y, x*θ, y*θ, x, y, θ, 1]
                    A[row] = [
                        xi * xi,
                        yi * yi,
                        ti * ti,
                        xi * yi,
                        xi * ti,
                        yi * ti,
                        xi,
                        yi,
                        ti,
                        1.0,
                    ]
                    S[row] = scores_3x3x3[theta_idx, oy + 1, ox + 1]
                    row += 1

        # SVD-based least squares: z = [a, b, c, d, e, f, g, h, i, j]
        z, _, _, _ = np.linalg.lstsq(A, S, rcond=None)

        # 建立 Hessian 矩陣 K1 (3x3) 與梯度向量 K2 (3,)
        # ∂²z/∂x² = 2a, ∂²z/∂y² = 2b, ∂²z/∂θ² = 2c
        # ∂²z/∂x∂y = d, ∂²z/∂x∂θ = e, ∂²z/∂y∂θ = f
        K1 = np.array(
            [
                [2.0 * z[0], z[3], z[4]],
                [z[3], 2.0 * z[1], z[5]],
                [z[4], z[5], 2.0 * z[2]],
            ],
            dtype=np.float64,
        )
        K2 = np.array([-z[6], -z[7], -z[8]], dtype=np.float64)

        # 求解 K1 * delta = K2 → delta 為絕對最佳座標 (x*, y*, θ*)
        delta = np.linalg.solve(K1, K2)

        # 轉換為相對偏移量
        dx = delta[0] - center_x
        dy = delta[1] - center_y
        dtheta_degrees = np.rad2deg(delta[2]) - center_angle

        return float(dx), float(dy), float(dtheta_degrees)

    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0
