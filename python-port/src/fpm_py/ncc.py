"""ncc.py

正規化互相關（NCC）樣板比對的通用實作封裝。
為符合跨平台移植需求，主要以 OpenCV `matchTemplate` 為核心，
並提供較直觀的介面與輸出格式，以利與金字塔/角度搜尋整合。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    """將輸入影像轉為單通道灰階。

    參數
    -----
    img : np.ndarray
        輸入影像（可為灰階或 BGR）。
    """
    if img is None:
        raise ValueError("輸入影像為 None")
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"不支援的影像維度: {img.shape}")


def ncc_match(
    src: np.ndarray,
    tmpl: np.ndarray,
    method: str = "ccoeff_normed",
) -> np.ndarray:
    """以 NCC 進行樣板比對，回傳相似度地圖。

    使用 OpenCV 的 `matchTemplate`，方法預設為 `TM_CCOEFF_NORMED`，
    與零均值 NCC 定義相容且輸出範圍落在 [-1, 1]。

    參數
    -----
    src : np.ndarray
        來源影像（灰階或 BGR）。
    tmpl : np.ndarray
        樣板影像（灰階或 BGR）。
    method : str
        `ccorr_normed` 或 `ccoeff_normed`，對應 OpenCV 之 NCC 變種。

    回傳
    -----
    np.ndarray
        相似度地圖（越大越相似）。
    """
    src_g = _to_gray(src)
    tmpl_g = _to_gray(tmpl)

    if src_g.shape[0] < tmpl_g.shape[0] or src_g.shape[1] < tmpl_g.shape[1]:
        raise ValueError("樣板尺寸不能大於來源影像")

    if method == "ccorr_normed":
        cv_method = cv2.TM_CCORR_NORMED
    elif method == "ccoeff_normed":
        cv_method = cv2.TM_CCOEFF_NORMED
    else:
        raise ValueError("method 僅支援 'ccorr_normed' 或 'ccoeff_normed'")

    resp = cv2.matchTemplate(src_g, tmpl_g, cv_method)
    return resp


def topk_from_response(
    response: np.ndarray,
    k: int = 10,
    min_distance: int = 5,
    score_thresh: float = 0.0,
) -> List[Tuple[int, int, float]]:
    """由相似度地圖擷取前 k 大峰值，並以最小距離抑制相近峰。

    回傳為 (x, y, score) 列表，座標為左上角位置。
    """
    resp = response.copy()
    h, w = resp.shape[:2]
    picks: List[Tuple[int, int, float]] = []

    for _ in range(k):
        _, max_val, _, max_loc = cv2.minMaxLoc(resp)
        if max_val < score_thresh:
            break
        x, y = max_loc
        picks.append((x, y, float(max_val)))
        # 抑制鄰域
        x0 = max(0, x - min_distance)
        y0 = max(0, y - min_distance)
        x1 = min(w, x + min_distance + 1)
        y1 = min(h, y + min_distance + 1)
        resp[y0:y1, x0:x1] = -1  # 設為低分避免再次挑中

    return picks


def topk_blocked(
    response: np.ndarray,
    block_size: tuple[int, int],
    k: int = 50,
    score_thresh: float = 0.0,
) -> List[Tuple[int, int, float]]:
    """以分塊（Block Max）策略擷取候選峰值（向量化版本）。

    會將相似度地圖切成不重疊小區塊，於每塊取最大值作為候選，
    再依分數排序取前 k 名。適合「大圖 + 小樣板」情境的快速擷取。
    """
    h, w = response.shape[:2]
    bw, bh = max(4, int(block_size[0])), max(4, int(block_size[1]))

    # Pad to make divisible
    pad_h = (bh - h % bh) % bh
    pad_w = (bw - w % bw) % bw
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(response, ((0, pad_h), (0, pad_w)), constant_values=-np.inf)
    else:
        padded = response

    nrows, ncols = padded.shape[0] // bh, padded.shape[1] // bw
    blocks = padded.reshape(nrows, bh, ncols, bw).transpose(0, 2, 1, 3)
    blocks_flat = blocks.reshape(nrows, ncols, -1)

    max_vals = blocks_flat.max(axis=2)
    max_idxs = blocks_flat.argmax(axis=2)
    local_y, local_x = np.divmod(max_idxs, bw)

    global_y = np.arange(nrows)[:, None] * bh + local_y
    global_x = np.arange(ncols)[None, :] * bw + local_x

    mask = (max_vals >= score_thresh) & (global_y < h) & (global_x < w)
    scores_flat = max_vals[mask]
    xs_flat = global_x[mask]
    ys_flat = global_y[mask]

    order = np.argsort(-scores_flat)[:k]
    return [(int(xs_flat[i]), int(ys_flat[i]), float(scores_flat[i])) for i in order]


# ---------------------------
# 擴充四：FFT 版 NCC（獨立函式）
# ---------------------------

def _integral_sums(img: np.ndarray, ksize: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """以積分影像同時計算所有滑窗的像素和與平方和。

    回傳
    -----
    sums : np.ndarray
        各滑窗的像素總和，尺寸 (H-h+1, W-w+1)
    sums2 : np.ndarray
        各滑窗的平方總和，尺寸同上
    """
    h, w = img.shape[:2]
    kh, kw = ksize
    img32 = img.astype(np.float32, copy=False)
    sqr = (img32 * img32).astype(np.float32)
    I = cv2.integral(img32)  # (H+1, W+1)
    I2 = cv2.integral(sqr)
    # 利用積分影像的矩形求和
    sums = I[kh:, kw:] - I[:-kh, kw:] - I[kh:, :-kw] + I[:-kh, :-kw]
    sums2 = I2[kh:, kw:] - I2[:-kh, kw:] - I2[kh:, :-kw] + I2[:-kh, :-kw]
    return sums.astype(np.float32), sums2.astype(np.float32)


def ncc_match_fft(
    src: np.ndarray,
    tmpl: np.ndarray,
    method: str = "ccoeff_normed",
) -> np.ndarray:
    """以 FFT 實作的 NCC，相容 `matchTemplate` 之輸出大小。

    支援：
    - `ccoeff_normed`：零均值正規化互相關（區間約 [-1, 1]）
    - `ccorr_normed`：相關係數正規化（區間 [0, 1]）

    設計說明：
    - 以 `ifft2(fft(src)*conj(fft(tmpl)))` 求交互相關之「全域」地圖，再切出 valid 區域。
    - 分母以滑窗的方差（或能量）配合樣板能量計算，避免逐點重複計算。
    """
    src_g = _to_gray(src).astype(np.float32, copy=False)
    tmpl_g = _to_gray(tmpl).astype(np.float32, copy=False)

    H, W = src_g.shape
    h, w = tmpl_g.shape
    if H < h or W < w:
        raise ValueError("樣板尺寸不能大於來源影像")

    # 交互相關（Full），使用零均值樣板（ccoeff）或原樣板（ccorr）
    if method == "ccoeff_normed":
        mu_t = float(tmpl_g.mean())
        T = tmpl_g - mu_t
        sum_t2 = float((T * T).sum())
    elif method == "ccorr_normed":
        T = tmpl_g
        sum_t2 = float((T * T).sum())
    else:
        raise ValueError("method 僅支援 'ccoeff_normed' 或 'ccorr_normed'")

    P = H + h - 1
    Q = W + w - 1
    FI = np.fft.fft2(src_g, s=(P, Q))
    FT = np.fft.fft2(T, s=(P, Q))
    CC_full = np.fft.ifft2(FI * np.conj(FT)).real  # 交互相關
    # 擷取 valid 區域：位置 (0,0) = 模板在左上角
    CC = CC_full[0 : H - h + 1, 0 : W - w + 1]

    # 計算滑窗和/平方和以估計分母
    sums, sums2 = _integral_sums(src_g, (h, w))
    N = float(h * w)

    if method == "ccoeff_normed":
        # 零均值 NCC：分母 = std_I * std_T
        mean_I = sums / N
        var_I = np.maximum(sums2 - N * (mean_I * mean_I), 0.0)
        std_I = np.sqrt(var_I)
        std_T = np.sqrt(max(sum_t2, 1e-12))
        denom = std_I * std_T
        resp = np.zeros_like(CC, dtype=np.float32)
        np.divide(CC, np.maximum(denom, 1e-12), out=resp, where=denom > 0)
        # 範圍理論上在 [-1, 1]；不強行裁切
        return resp
    else:
        # CCORR_NORMED：分母 = sqrt(sum_I2 * sum_T2)
        denom = np.sqrt(np.maximum(sums2, 0.0) * max(sum_t2, 1e-12))
        resp = np.zeros_like(CC, dtype=np.float32)
        np.divide(CC, np.maximum(denom, 1e-12), out=resp, where=denom > 0)
        # 範圍約 [-1, 1]；交叉相關可為負
        return resp
