"""pyramid.py

金字塔多尺度 NCC 樣板匹配 + 旋轉搜尋核心流程。

本模組對齊 C++ CMatchCore::Match() (match_core.cpp:135-401) 的完整流程：
1. LearnPattern — 預計算模板統計量
2. buildPyramid — 建立影像金字塔 (÷2 per layer)
3. Top Layer 全域搜尋 — 旋轉 Source 圖，對未旋轉模板做 matchTemplate
4. Coarse-to-Fine 逐層精化 — GetRotatedROI + 3 角度測試
5. SubPixEstimation — 27 點 3D 曲面擬合亞像素精化
6. NMS — 旋轉矩形非極大值抑制
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .ncc import ncc_match, ncc_match_fft
from .nms import filter_with_score, nms, nms_rotated
from .rotation import rotate_source
from .subpixel import refine_peak_3d

# 常數（對應 C++ match_core.h）
R2D = 180.0 / math.pi
D2R = math.pi / 180.0
VISION_TOLERANCE = 0.0001
MATCH_CANDIDATE_NUM = 5


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TemplateData:
    """模板預計算資料，對應 C++ s_TemplData (match_core.h:15-43)。"""

    pyramid: List[np.ndarray] = field(default_factory=list)
    means: List[float] = field(default_factory=list)
    norms: List[float] = field(default_factory=list)
    inv_areas: List[float] = field(default_factory=list)
    result_equal1: List[bool] = field(default_factory=list)
    border_color: int = 0
    is_learned: bool = False


@dataclass
class MatchResult:
    """單一匹配結果。"""

    x: float
    y: float
    score: float
    angle: float


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _build_pyramid(img: np.ndarray, max_level: int) -> List[np.ndarray]:
    """建立高斯金字塔（÷2 per level），等價於 C++ buildPyramid。

    回傳 [level_0(原圖), level_1(÷2), ..., level_N(最小)]。
    """
    pyr = [img]
    for _ in range(max_level):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _get_top_layer(tmpl: np.ndarray, min_reduced_area: int = 256) -> int:
    """計算金字塔層數，對應 C++ GetTopLayer (match_core.cpp:122-133)。

    每層面積 ÷4，直到面積 <= min_reduced_area。
    """
    min_length = int(math.sqrt(min_reduced_area))
    min_area = min_length * min_length
    area = tmpl.shape[0] * tmpl.shape[1]
    top_layer = 0
    while area > min_area:
        area //= 4
        top_layer += 1
    return top_layer


def _pt_rotate(
    pt: Tuple[float, float],
    center: Tuple[float, float],
    angle_rad: float,
) -> Tuple[float, float]:
    """2D 點旋轉，對應 C++ ptRotatePt2f (match_core.cpp:565-576)。

    等價於 OpenCV getRotationMatrix2D(center, angle, 1) 作用於點。
    """
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (
        dx * cos_a + dy * sin_a + center[0],
        -dx * sin_a + dy * cos_a + center[1],
    )


def _get_rotated_roi(
    src: np.ndarray,
    tmpl_size: Tuple[int, int],
    pt_lt: Tuple[float, float],
    angle_deg: float,
) -> np.ndarray:
    """提取旋轉 ROI（帶 ±3 pixel padding），對應 C++ GetRotatedROI
    (match_core.cpp:506-517)。

    Parameters
    ----------
    src : 來源影像（某層金字塔）
    tmpl_size : 模板尺寸 (height, width)
    pt_lt : 候選者左上角位置（在原始座標系中）
    angle_deg : 旋轉角度（度）

    Returns
    -------
    裁切出的旋轉 ROI，尺寸為 (th+6, tw+6)
    """
    h, w = src.shape[:2]
    th, tw = tmpl_size
    padding = 3
    roi_w = tw + 2 * padding
    roi_h = th + 2 * padding

    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    # 將 ptLT 旋轉到旋轉座標系
    pt_rotated = _pt_rotate(pt_lt, (cx, cy), angle_deg * D2R)

    # 建立旋轉矩陣並調整平移量以裁切 ROI
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M[0, 2] -= pt_rotated[0] - padding
    M[1, 2] -= pt_rotated[1] - padding

    return cv2.warpAffine(
        src,
        M,
        (roi_w, roi_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _layer_thresholds(base_score: float, num_levels: int) -> List[float]:
    """逐層分數衰減，對應 C++ vecLayerScore (match_core.cpp:183-185)。

    layer 0 = base_score（最嚴格），layer i = layer[i-1] * 0.9
    """
    thresholds = [base_score]
    for _ in range(1, num_levels):
        thresholds.append(thresholds[-1] * 0.9)
    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# LearnPattern
# ─────────────────────────────────────────────────────────────────────────────


def learn_pattern(
    tmpl_gray: np.ndarray, min_reduced_area: int = 256
) -> TemplateData:
    """預計算模板統計量，對應 C++ LearnPattern (match_core.cpp:85-120)。"""
    data = TemplateData()
    top_layer = _get_top_layer(tmpl_gray, min_reduced_area)

    # buildPyramid: level 0 = 原圖, level N = 最小
    data.pyramid = _build_pyramid(tmpl_gray, top_layer)

    # borderColor: 暗模板用白邊界，亮模板用黑邊界
    data.border_color = 255 if float(np.mean(tmpl_gray)) < 128 else 0

    n = len(data.pyramid)
    data.means = [0.0] * n
    data.norms = [0.0] * n
    data.inv_areas = [0.0] * n
    data.result_equal1 = [False] * n

    for i in range(n):
        rows, cols = data.pyramid[i].shape[:2]
        inv_area = 1.0 / (rows * cols)
        mean_val, stddev_val = cv2.meanStdDev(data.pyramid[i])
        tmpl_mean = float(mean_val[0][0])
        tmpl_stddev = float(stddev_val[0][0])
        tmpl_norm_sq = tmpl_stddev**2  # variance (single channel)

        if tmpl_norm_sq < 1e-15:
            data.result_equal1[i] = True

        # C++: templNorm = sqrt(variance) / sqrt(invArea)
        tmpl_norm = math.sqrt(tmpl_norm_sq) / math.sqrt(inv_area)

        data.inv_areas[i] = inv_area
        data.means[i] = tmpl_mean
        data.norms[i] = tmpl_norm

    data.is_learned = True
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Top layer helpers
# ─────────────────────────────────────────────────────────────────────────────


def _do_match(
    src: np.ndarray, tmpl: np.ndarray, method: str, backend: str
) -> np.ndarray:
    """執行模板匹配（OpenCV 或 FFT 後端）。"""
    if backend == "fft":
        return ncc_match_fft(src, tmpl, method=method)
    return ncc_match(src, tmpl, method=method)


def _suppress_and_find(
    resp: np.ndarray,
    prev_loc: Tuple[int, int],
    tmpl_w: int,
    tmpl_h: int,
    max_overlap: float,
) -> Tuple[float, Tuple[int, int]]:
    """抑制峰值附近區域，找下一個最大值。
    對應 C++ GetNextMaxLoc (match_core.cpp:650-677)。
    """
    suppress_w = int(tmpl_w * (1 - max_overlap))
    suppress_h = int(tmpl_h * (1 - max_overlap))
    x0 = max(0, prev_loc[0] - suppress_w)
    y0 = max(0, prev_loc[1] - suppress_h)
    x1 = min(resp.shape[1], prev_loc[0] + suppress_w)
    y1 = min(resp.shape[0], prev_loc[1] + suppress_h)
    resp[y0:y1, x0:x1] = -1

    _, max_val, _, max_loc = cv2.minMaxLoc(resp)
    return float(max_val), max_loc


# ─────────────────────────────────────────────────────────────────────────────
# Main matching function
# ─────────────────────────────────────────────────────────────────────────────


def pyramid_match(
    src: np.ndarray,
    tmpl: np.ndarray,
    target_num: int = 1,
    score: float = 0.5,
    max_overlap: float = 0.5,
    tolerance_angle: float = 0.0,
    min_reduced_area: int = 256,
    method: str = "ccoeff_normed",
    backend: str = "opencv",
    subpixel: bool = True,
    # 向後相容參數（不再使用，改用自適應角度步進）
    coarse_step: float = 15.0,
    fine_step: float = 5.0,
    scale: float = 1.25,
    topk_per_level: int = 50,
) -> List[Dict]:
    """金字塔多尺度 NCC 匹配 + 旋轉搜尋，對齊 C++ Match()。

    Parameters
    ----------
    src : 來源影像
    tmpl : 模板影像
    target_num : 最大目標數
    score : 最低分數門檻 (layer 0)
    max_overlap : NMS 最大重疊比
    tolerance_angle : 角度容忍範圍 (±度)，0 表示不搜尋旋轉
    min_reduced_area : 金字塔頂層最小面積
    method : NCC 方法 ('ccoeff_normed' 或 'ccorr_normed')
    backend : 計算後端 ('opencv' 或 'fft')
    subpixel : 是否啟用 27 點亞像素精化

    Returns
    -------
    匹配結果列表，每筆包含 index, score, angle, pos_x, pos_y
    """
    src_g = _ensure_gray(src)
    tmpl_g = _ensure_gray(tmpl)

    # 基本驗證（對應 C++ match_core.cpp:140-148）
    if src_g.size == 0 or tmpl_g.size == 0:
        return []
    th, tw = tmpl_g.shape[:2]
    sh, sw = src_g.shape[:2]
    if tw > sw or th > sh:
        return []
    if (tw < sw and th > sh) or (tw > sw and th < sh):
        return []

    # ── ① LearnPattern ──────────────────────────────────────────────────
    tmpl_data = learn_pattern(tmpl_g, min_reduced_area)
    if not tmpl_data.is_learned:
        return []
    top_layer = len(tmpl_data.pyramid) - 1
    if top_layer < 0:
        return []

    # ── ② 建立來源影像金字塔 ────────────────────────────────────────────
    src_pyr = _build_pyramid(src_g, top_layer)

    # ── ③ 自適應角度步進 ────────────────────────────────────────────────
    top_tmpl = tmpl_data.pyramid[top_layer]
    top_tmpl_h, top_tmpl_w = top_tmpl.shape[:2]
    angle_step = math.atan(2.0 / max(top_tmpl_w, top_tmpl_h)) * R2D

    # 角度列舉（C++ 順序：0, +step, +2step, ..., -step, -2step, ...）
    if tolerance_angle < VISION_TOLERANCE:
        angles = [0.0]
    else:
        angles = []
        a = 0.0
        while a < tolerance_angle + angle_step:
            angles.append(a)
            a += angle_step
        a = -angle_step
        while a > -tolerance_angle - angle_step:
            angles.append(a)
            a -= angle_step

    # ── ④ 層級分數門檻 ─────────────────────────────────────────────────
    layer_scores = _layer_thresholds(score, top_layer + 1)

    # ── ⑤ Top Layer 全域搜尋 ───────────────────────────────────────────
    top_src = src_pyr[top_layer]
    top_src_h, top_src_w = top_src.shape[:2]
    src_center_top = ((top_src_w - 1) / 2.0, (top_src_h - 1) / 2.0)

    # candidates: (pt_x, pt_y, score, angle) — pt 在「旋轉後扣除平移」座標系中
    candidates: List[Tuple[float, float, float, float]] = []

    for angle in angles:
        # 旋轉 source 圖（非 template）
        if abs(angle) < 1e-6:
            rotated_src = top_src
            trans_x, trans_y = 0.0, 0.0
        else:
            rotated_src = rotate_source(
                top_src, angle, top_tmpl.shape, tmpl_data.border_color
            )
            rot_h, rot_w = rotated_src.shape[:2]
            trans_x = (rot_w - 1) / 2.0 - src_center_top[0]
            trans_y = (rot_h - 1) / 2.0 - src_center_top[1]

        # 尺寸檢查
        if (
            rotated_src.shape[0] < top_tmpl_h
            or rotated_src.shape[1] < top_tmpl_w
        ):
            continue

        # 模板匹配
        resp = _do_match(rotated_src, top_tmpl, method, backend)

        # 找第一個峰值
        _, max_val, _, max_loc = cv2.minMaxLoc(resp)
        if max_val < layer_scores[top_layer]:
            continue

        candidates.append((
            max_loc[0] - trans_x,
            max_loc[1] - trans_y,
            float(max_val),
            angle,
        ))

        # 搜索更多候選（對應 C++ GetNextMaxLoc 迴圈）
        resp_work = resp.copy()
        prev_loc = max_loc
        for _ in range(target_num + MATCH_CANDIDATE_NUM - 1):
            next_val, next_loc = _suppress_and_find(
                resp_work, prev_loc, top_tmpl_w, top_tmpl_h, max_overlap
            )
            if next_val < layer_scores[top_layer]:
                break
            candidates.append((
                next_loc[0] - trans_x,
                next_loc[1] - trans_y,
                next_val,
                angle,
            ))
            prev_loc = next_loc

    # 依分數排序
    candidates.sort(key=lambda c: c[2], reverse=True)

    # ── ⑥ Coarse-to-Fine 逐層精化 ─────────────────────────────────────
    all_results: List[Dict] = []

    for cand_x, cand_y, cand_score, cand_angle in candidates:
        # 候選點從「旋轉後」座標系轉回原始座標系
        # 對應 C++: ptLT = ptRotatePt2f(pt, ptCenter, -matchAngle*D2R)
        pt_lt = _pt_rotate(
            (cand_x, cand_y), src_center_top, -cand_angle * D2R
        )
        matched_angle = cand_angle

        # top_layer == 0：無需精化
        if top_layer == 0:
            all_results.append({
                "pt": pt_lt,
                "score": cand_score,
                "angle": matched_angle,
            })
            continue

        # 逐層精化 (top_layer-1 → 0)
        for layer in range(top_layer - 1, -1, -1):
            cur_tmpl = tmpl_data.pyramid[layer]
            cur_src = src_pyr[layer]
            cur_th, cur_tw = cur_tmpl.shape[:2]
            cur_sh, cur_sw = cur_src.shape[:2]

            # 此層的角度步進
            local_step = math.atan(2.0 / max(cur_tw, cur_th)) * R2D
            cur_center = ((cur_sw - 1) / 2.0, (cur_sh - 1) / 2.0)

            # 測試角度：matched ± step（或固定 0°）
            if tolerance_angle < VISION_TOLERANCE:
                test_angles = [0.0]
            else:
                test_angles = [
                    matched_angle - local_step,
                    matched_angle,
                    matched_angle + local_step,
                ]

            best_score = -1.0
            best_idx = 0
            angle_results: List[Dict] = []

            for j, test_angle in enumerate(test_angles):
                # 提取旋轉 ROI（ptLT × 2 放大到此層）
                pt_lt_scaled = (pt_lt[0] * 2, pt_lt[1] * 2)
                roi = _get_rotated_roi(
                    cur_src, (cur_th, cur_tw), pt_lt_scaled, test_angle
                )

                if roi.shape[0] < cur_th or roi.shape[1] < cur_tw:
                    angle_results.append({
                        "pt": (0, 0),
                        "score": -1.0,
                        "angle": test_angle,
                        "on_border": True,
                        "scores_3x3": None,
                    })
                    continue

                resp = _do_match(roi, cur_tmpl, method, backend)
                _, max_val, _, max_loc = cv2.minMaxLoc(resp)

                # 邊界檢查
                on_border = (
                    max_loc[0] == 0
                    or max_loc[1] == 0
                    or max_loc[0] == resp.shape[1] - 1
                    or max_loc[1] == resp.shape[0] - 1
                )

                # 收集 3×3 分數（用於 SubPixEstimation 的 27 點擬合）
                scores_3x3 = None
                if not on_border:
                    mx, my = max_loc
                    if (
                        1 <= mx < resp.shape[1] - 1
                        and 1 <= my < resp.shape[0] - 1
                    ):
                        scores_3x3 = resp[
                            my - 1 : my + 2, mx - 1 : mx + 2
                        ].astype(np.float64)

                angle_results.append({
                    "pt": max_loc,
                    "score": float(max_val),
                    "angle": test_angle,
                    "on_border": on_border,
                    "scores_3x3": scores_3x3,
                })

                if float(max_val) > best_score:
                    best_score = float(max_val)
                    best_idx = j

            # 分數門檻（對應 C++ vecLayerScore 檢查）
            if best_score < layer_scores[layer]:
                break

            best = angle_results[best_idx]

            # ── Layer 0 亞像素精化 ──
            # 條件：中間角度最佳 (best_idx==1)，且非邊界
            # 對應 C++ match_core.cpp:321-330
            if (
                subpixel
                and layer == 0
                and not best["on_border"]
                and len(test_angles) == 3
                and best_idx == 1
            ):
                all_have = all(
                    r["scores_3x3"] is not None for r in angle_results
                )
                if all_have:
                    scores_3x3x3 = np.stack(
                        [r["scores_3x3"] for r in angle_results], axis=0
                    )
                    bx, by = best["pt"]
                    dx, dy, dtheta = refine_peak_3d(
                        scores_3x3x3,
                        float(bx),
                        float(by),
                        best["angle"],
                        local_step,
                    )
                    best = dict(best)  # copy
                    best["pt"] = (bx + dx, by + dy)
                    best["angle"] = best["angle"] + dtheta

            # ── 座標轉換：ROI → 原始源座標 ──
            # 對應 C++ match_core.cpp:332-337
            new_angle = best["angle"]
            pt_lt_scaled = (pt_lt[0] * 2, pt_lt[1] * 2)

            # ptPaddingLT = rotate(ptLT*2, srcCenter, angle) - (3, 3)
            pt_padding_lt = _pt_rotate(
                pt_lt_scaled, cur_center, new_angle * D2R
            )
            pt_padding_lt = (pt_padding_lt[0] - 3, pt_padding_lt[1] - 3)

            # pt = maxLoc + ptPaddingLT → 旋轉空間中的位置
            px, py = best["pt"]
            pt_abs = (px + pt_padding_lt[0], py + pt_padding_lt[1])

            # 轉回原始座標
            pt_orig = _pt_rotate(pt_abs, cur_center, -new_angle * D2R)

            if layer == 0:
                all_results.append({
                    "pt": pt_orig,
                    "score": best_score,
                    "angle": new_angle,
                })
            else:
                matched_angle = new_angle
                pt_lt = pt_orig

    # ── ⑦ 分數過濾 ────────────────────────────────────────────────────
    if all_results:
        scores_list = [r["score"] for r in all_results]
        keep = filter_with_score(scores_list, score)
        all_results = [all_results[i] for i in keep]

    # ── ⑧ NMS — 旋轉矩形 ─────────────────────────────────────────────
    tmpl_h0, tmpl_w0 = tmpl_data.pyramid[0].shape[:2]

    nms_scores: List[float] = []
    for r in all_results:
        ptLT = r["pt"]
        dRAngle = -r["angle"] * D2R

        # 從 ptLT 計算 4 角（對應 C++ match_core.cpp:361-367）
        ptRT = (
            ptLT[0] + tmpl_w0 * math.cos(dRAngle),
            ptLT[1] - tmpl_w0 * math.sin(dRAngle),
        )
        ptLB = (
            ptLT[0] + tmpl_h0 * math.sin(dRAngle),
            ptLT[1] + tmpl_h0 * math.cos(dRAngle),
        )
        ptRB = (
            ptRT[0] + tmpl_h0 * math.sin(dRAngle),
            ptRT[1] + tmpl_h0 * math.cos(dRAngle),
        )

        cx = (ptLT[0] + ptRT[0] + ptRB[0] + ptLB[0]) / 4.0
        cy = (ptLT[1] + ptRT[1] + ptRB[1] + ptLB[1]) / 4.0

        r["center"] = (cx, cy)
        r["corners"] = (ptLT, ptRT, ptRB, ptLB)
        nms_scores.append(r["score"])

    if len(all_results) > 0:
        if tolerance_angle >= VISION_TOLERANCE:
            # 有旋轉：用旋轉矩形 NMS
            rects = []
            for r in all_results:
                corners = r["corners"]
                pts = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
                rects.append(cv2.minAreaRect(pts))
            keep = nms_rotated(rects, nms_scores, max_overlap=max_overlap)
        else:
            # 無旋轉：AABB NMS（更快）
            boxes = []
            for r in all_results:
                boxes.append((int(r["pt"][0]), int(r["pt"][1]), tmpl_w0, tmpl_h0))
            keep = nms(boxes, nms_scores, max_overlap=max_overlap)
    else:
        keep = []

    # 依分數排序，取前 target_num 個
    keep.sort(key=lambda i: nms_scores[i], reverse=True)
    keep = keep[:target_num]

    # ── ⑨ 建立輸出 ────────────────────────────────────────────────────
    # 輸出角度 = -匹配角度（對應 C++ match_core.cpp:389）
    results: List[Dict] = []
    for idx in keep:
        r = all_results[idx]
        cx, cy = r["center"]

        output_angle = -r["angle"]
        if output_angle < -180:
            output_angle += 360
        if output_angle > 180:
            output_angle -= 360

        results.append({
            "index": len(results),
            "score": float(r["score"]),
            "angle": float(output_angle),
            "pos_x": float(cx),
            "pos_y": float(cy),
        })

    return results
