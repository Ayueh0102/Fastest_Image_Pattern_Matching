# Python Port 完整架構文件

本文件詳細描述 `python-port/src/fpm_py/` 的完整架構，涵蓋每個模組的職責、資料結構、函式簽名、呼叫關係、以及與 C++ 原版的對應關係。適合作為學習與維護的參考。

> **互動式視覺化**：建議搭配 [algorithm-visual.html](algorithm-visual.html) 一起閱讀，有動畫演示核心演算法的運作方式。
> **演算法原理**：數學推導與公式詳見 [ALGORITHM_zh-TW.md](ALGORITHM_zh-TW.md)。

---

## 目錄

1. [整體架構](#1-整體架構)
2. [模組總覽](#2-模組總覽)
3. [pyramid.py — 核心流程](#3-pyramidpy--核心流程)
4. [ncc.py — NCC 匹配引擎](#4-nccpy--ncc-匹配引擎)
5. [rotation.py — 旋轉處理](#5-rotationpy--旋轉處理)
6. [subpixel.py — 次像素估計](#6-subpixelpy--次像素估計)
7. [nms.py — 非極大值抑制](#7-nmspy--非極大值抑制)
8. [cli.py — 命令列介面](#8-clipy--命令列介面)
9. [gui.py — 圖形介面](#9-guipy--圖形介面)
10. [資料流全貌](#10-資料流全貌)
11. [C++ ↔ Python 對照表](#11-c--python-對照表)
12. [依賴與環境](#12-依賴與環境)

---

## 1. 整體架構

```
python-port/src/fpm_py/
├── pyramid.py      ← 核心：金字塔匹配流程 (Match + LearnPattern)
├── ncc.py          ← NCC 計算引擎 (OpenCV / FFT 雙後端)
├── rotation.py     ← 影像旋轉 (source / template)
├── subpixel.py     ← 次像素精化 (2D / 3D 二次曲面擬合)
├── nms.py          ← NMS (軸對齊 / 旋轉矩形)
├── cli.py          ← 命令列入口 fpm-py
├── gui.py          ← Tkinter GUI 入口 fpm-py-gui
└── gpu.py          ← (實驗性) GPU 加速介面
```

### 呼叫層次

```
cli.py / gui.py
    └── pyramid.py :: pyramid_match()          ← 唯一公開 API
            ├── learn_pattern()                 ← 預計算模板統計量
            ├── _build_pyramid()                ← 建立影像金字塔
            ├── rotation.py :: rotate_source()  ← 旋轉來源影像
            ├── ncc.py :: ncc_match()           ← OpenCV NCC
            │   └── ncc.py :: ncc_match_fft()   ← FFT NCC (可選)
            ├── _find_topk_suppressed()         ← 向量化 Top-K 峰值搜尋
            ├── _get_rotated_roi()              ← 旋轉 ROI 提取
            ├── subpixel.py :: refine_peak_3d() ← 27 點亞像素精化
            ├── nms.py :: filter_with_score()   ← 分數過濾
            └── nms.py :: nms_rotated() / nms() ← NMS 去重疊
```

---

## 2. 模組總覽

| 模組 | 行數 | 職責 | C++ 對應 |
|------|------|------|---------|
| `pyramid.py` | ~692 | 金字塔搜尋主流程、候選精化、結果組裝 | `Match()` + `LearnPattern()` |
| `ncc.py` | ~327 | NCC 相似度計算（OpenCV / FFT）、TopK 搜尋 | `MatchTemplate()` + `GetNextMaxLoc()` |
| `rotation.py` | ~101 | 影像旋轉（source / template）| `GetBestRotationSize()` + `warpAffine` |
| `subpixel.py` | ~192 | 2D/3D 次像素精化 | `SubPixEstimation()` |
| `nms.py` | ~219 | 軸對齊 / 旋轉矩形 NMS | `FilterWithRotatedRect()` |
| `cli.py` | ~122 | 命令列解析與執行 | `main()` |
| `gui.py` | ~969 | Tkinter GUI + 結果繪製 | MFC GUI |

---

## 3. pyramid.py — 核心流程

這是整個匹配管線的指揮中心，對齊 C++ `CMatchCore::Match()` (match_core.cpp:135-401)。

### 3.1 資料結構

#### `TemplateData`

```python
@dataclass
class TemplateData:
    """模板預計算資料，對應 C++ s_TemplData。"""
    pyramid: List[np.ndarray]    # 模板金字塔 [level_0(原圖), ..., level_N(最小)]
    means: List[float]           # 各層平均值
    norms: List[float]           # 各層 norm = sqrt(variance) / sqrt(invArea)
    inv_areas: List[float]       # 各層 1/(rows*cols)
    result_equal1: List[bool]    # 各層是否為均勻影像（variance ≈ 0）
    border_color: int            # 旋轉邊界填充色（亮模板→黑，暗模板→白）
    is_learned: bool             # 是否已完成預計算
```

**設計要點**：
- `border_color` 根據模板平均亮度決定：`mean < 128 → 255（白）, 否則 → 0（黑）`
- `norms[i]` 的計算方式與 C++ 一致：`sqrt(variance) / sqrt(1/area)` = `stddev * sqrt(area)`
- `result_equal1[i]` 標記均勻影像，此時 NCC 分母為 0，結果直接為 1.0

#### `MatchResult` （輸出格式）

```python
# pyramid_match() 回傳 List[Dict]，每筆：
{
    "index": int,       # 結果編號（0-based）
    "score": float,     # NCC 分數 (0~1)
    "angle": float,     # 偵測角度（度），= -match_angle
    "pos_x": float,     # 中心 X 座標
    "pos_y": float,     # 中心 Y 座標
}
```

**重要**：
- `pos_x`, `pos_y` 是匹配矩形的**中心點**（非左上角）
- `angle` 是**輸出角度** = `-match_angle`，意義為「模板需旋轉此角度才能對上影像中的目標」

### 3.2 核心函式

#### `pyramid_match()` — 主入口

```python
def pyramid_match(
    src, tmpl,
    target_num=1, score=0.5, max_overlap=0.5,
    tolerance_angle=0.0, min_reduced_area=256,
    method="ccoeff_normed", backend="opencv",
    subpixel=True,
) -> List[Dict]
```

**九階段流程**：

| 階段 | 說明 | C++ 對應行 |
|------|------|-----------|
| ① LearnPattern | 預計算模板金字塔與統計量 | cpp:85-120 |
| ② 建立來源影像金字塔 | `cv2.pyrDown` 逐層 ÷2 | cpp:164-165 |
| ③ 自適應角度步進 | `atan(2/max(w,h))` 度 | cpp:167-180 |
| ④ 層級分數門檻 | 每層衰減 ×0.9 | cpp:183-185 |
| ⑤ Top Layer 全域搜尋 | 旋轉 source → matchTemplate | cpp:188-250 |
| ⑥ Coarse-to-Fine 逐層精化 | ROI 提取 + 3 角度測試 | cpp:255-340 |
| ⑦ 分數過濾 | 過濾 < score 的結果 | cpp:343-345 |
| ⑧ NMS | 旋轉矩形 NMS | cpp:347-385 |
| ⑨ 建立輸出 | 中心座標 + 角度轉換 | cpp:387-401 |

#### `learn_pattern()` — 模板預計算

```python
def learn_pattern(tmpl_gray, min_reduced_area=256) -> TemplateData
```

- 計算金字塔層數：`while area > min_area: area //= 4; top_layer += 1`
- 建立模板金字塔：`cv2.pyrDown` 逐層
- 各層統計量：`cv2.meanStdDev` → mean, stddev, norm

#### `_build_pyramid()` — 金字塔建構

```python
def _build_pyramid(img, max_level) -> List[np.ndarray]
```

- 使用 `cv2.pyrDown`（÷2 高斯下採樣），**非** cv2.resize
- 回傳 `[level_0(原圖), level_1(÷2), ..., level_N(最小)]`

#### `_get_top_layer()` — 金字塔層數計算

```python
def _get_top_layer(tmpl, min_reduced_area=256) -> int
```

- 每層面積 ÷4，直到 `area <= min_reduced_area`
- 例：模板 200×200 = 40000 → 10000 → 2500 → 625 → 156 → 4 層

#### `_get_rotated_roi()` — 旋轉 ROI 提取

```python
def _get_rotated_roi(src, tmpl_size, pt_lt, angle_deg) -> np.ndarray
```

- 帶 ±3 pixel padding
- 先將 `ptLT` 旋轉到旋轉座標系，再用 `warpAffine` 裁切
- 輸出尺寸 `(th+6, tw+6)`

#### `_find_topk_suppressed()` — 向量化 Top-K

```python
def _find_topk_suppressed(resp, k, suppress_w, suppress_h, score_thresh) -> List[Tuple]
```

- 使用 `np.argpartition` 一次性提取前 k×50 大候選
- 再以距離抑制篩選，複雜度 O(H×W + n_cand × k)
- 取代 C++ 的反覆 `minMaxLoc` 迴圈

### 3.3 角度搜尋策略

**自適應角度步進**（取代舊版固定 coarse/fine step）：

```
angle_step = atan(2.0 / max(top_tmpl_w, top_tmpl_h)) × (180/π)
```

- 模板越大 → 步進越小 → 角度越精確
- 例：模板 100px → step ≈ 1.15°；模板 30px → step ≈ 3.81°

**角度列舉順序**（C++ 一致）：
```
0, +step, +2×step, ..., +tolerance, -step, -2×step, ..., -tolerance
```

**Top Layer 旋轉策略**：
- **旋轉 Source 圖**（非 Template）
- 原因：模板已預計算統計量，旋轉模板需重新計算；旋轉 source 可直接匹配原始模板

**逐層精化角度**：
- 每層測試 3 個角度：`matched_angle - step, matched_angle, matched_angle + step`
- 最佳角度向下傳遞

### 3.4 座標轉換

整個流程涉及多次座標系轉換：

```
旋轉後 source 座標 (top layer)
  ↓ ptRotatePt2f(pt, center, -angle)
原始 source 座標 (top layer)
  ↓ ×2 放大
原始 source 座標 (layer i)
  ↓ ptRotatePt2f → 旋轉座標系中的 ptPaddingLT
  ↓ + maxLoc（ROI 內的最大值位置）
旋轉後 source 座標 (layer i)
  ↓ ptRotatePt2f(pt, center, -angle)
原始 source 座標 (layer i)
```

最終 `ptLT`（左上角）→ 透過方向向量計算 4 個角 → 求中心：
```python
cx = (ptLT + ptRT + ptRB + ptLB).x / 4
cy = (ptLT + ptRT + ptRB + ptLB).y / 4
```

---

## 4. ncc.py — NCC 匹配引擎

提供兩個後端：OpenCV 和 FFT。

### 4.1 `ncc_match()` — OpenCV 後端

```python
def ncc_match(src, tmpl, method="ccoeff_normed") -> np.ndarray
```

- 直接呼叫 `cv2.matchTemplate`
- 支援 `TM_CCOEFF_NORMED`（零均值）和 `TM_CCORR_NORMED`（非零均值）
- 輸出尺寸：`(H-h+1, W-w+1)`

### 4.2 `ncc_match_fft()` — FFT 後端

```python
def ncc_match_fft(src, tmpl, method="ccoeff_normed") -> np.ndarray
```

**計算流程**：

1. **零均值模板**（ccoeff_normed）：`T = tmpl - mean(tmpl)`
2. **FFT 交互相關**：
   - 填充尺寸：`P = H+h-1, Q = W+w-1`
   - `CC_full = IFFT(FFT(src) × conj(FFT(T)))`
3. **擷取 valid 區域**：`CC = CC_full[0:H-h+1, 0:W-w+1]`
4. **滑窗統計量**（積分影像）：
   - `sums, sums2 = _integral_sums(src, (h, w))`
5. **正規化分母**：
   - ccoeff：`denom = std_I × std_T`
   - ccorr：`denom = sqrt(sum_I² × sum_T²)`

**FFT 後端選擇**：
- `scipy.fft` 可用 → 多執行緒 `fft2(workers=-1)`
- 否則 fallback 到 `numpy.fft`

### 4.3 `_integral_sums()` — 積分影像滑窗統計

```python
def _integral_sums(img, ksize) -> (sums, sums2)
```

- 使用 `cv2.integral` 建立積分影像
- 矩形求和公式：`sums = integ[kh:, kw:] - integ[:-kh, kw:] - integ[kh:, :-kw] + integ[:-kh, :-kw]`
- 一次計算所有滑窗的像素和與平方和，O(1) per window

### 4.4 `topk_from_response()` — 向量化 Top-K

```python
def topk_from_response(response, k=10, min_distance=5, score_thresh=0.0) -> List[Tuple]
```

- `np.argpartition` 提取前 k×50 候選
- 距離抑制：`abs(x-px) <= min_distance && abs(y-py) <= min_distance`

### 4.5 `topk_blocked()` — 分塊最大值（Block Max）

```python
def topk_blocked(response, block_size, k=50, score_thresh=0.0) -> List[Tuple]
```

- 將響應圖切成不重疊區塊
- 向量化 reshape + transpose 一次完成所有塊的 max
- 適合「大圖 + 小模板」場景（對應 C++ `s_BlockMax`）

### 4.6 Numba 加速

```python
@njit(cache=True)
def _numba_suppress_and_argmax(resp, cx, cy, sw, sh) -> (max_val, max_x, max_y)
```

- 當 `numba` 可用時自動啟用
- 抑制 + 全圖最大值搜尋的 JIT 編譯版本

---

## 5. rotation.py — 旋轉處理

### 5.1 `rotate_template()`

```python
def rotate_template(tmpl, angle_deg, border_value=0) -> np.ndarray
```

- 以模板中心為軸旋轉
- 自動擴張畫布避免裁切：`new_w = w×|cos| + h×|sin|`
- 邊界填充：`BORDER_CONSTANT`

### 5.2 `rotate_source()`

```python
def rotate_source(src, angle_deg, tmpl_shape, border_value=0) -> np.ndarray
```

- 以 source 影像中心為軸旋轉
- 計算完整外接矩形尺寸
- 中心公式使用 `(w-1)/2.0`（與 C++ 一致，OpenCV 的像素中心定義）
- 回傳旋轉後影像，用於 Top Layer 全域搜尋

**重要差異**：`rotate_template` 的中心用 `w/2.0`，`rotate_source` 的中心用 `(w-1)/2.0`。兩者在 C++ 中的定義不同。

---

## 6. subpixel.py — 次像素估計

### 6.1 `refine_peak_subpixel()` — 2D 二次曲面（9 點）

```python
def refine_peak_subpixel(response, x, y) -> (dx, dy)
```

- 3×3 鄰域擬合六參數曲面：`z = ax² + by² + cxy + dx + ey + f`
- SVD 最小平方法求解
- 解析公式求極值：`dx = (ce-2bd)/(4ab-c²), dy = (cd-2ae)/(4ab-c²)`
- 偏移量限制在 `[-1, 1]`

### 6.2 `refine_peak_3d()` — 3D 二次曲面（27 點）

```python
def refine_peak_3d(scores_3x3x3, center_x, center_y, center_angle, angle_step) -> (dx, dy, dtheta)
```

**核心演算法**（對應 C++ `SubPixEstimation`）：

1. **27 點設計矩陣** A (27×10)：3 角度 × 3×3 空間
   - 使用**絕對座標**（非相對偏移），與 C++ 一致
   - 十參數曲面：`z = ax² + by² + cθ² + dxy + exθ + fyθ + gx + hy + iθ + j`

2. **SVD 求解**：`z = A\S`

3. **Hessian + 梯度求解**：
   ```
   K1 = [[2a, d, e],    K2 = [-g, -h, -i]
         [d, 2b, f],
         [e, f, 2c]]

   delta = K1⁻¹ × K2    # 絕對最佳座標 (x*, y*, θ*)
   ```

4. **轉換為偏移量**：
   - `dx = delta[0] - center_x`
   - `dy = delta[1] - center_y`
   - `dθ = rad2deg(delta[2]) - center_angle`

**觸發條件**：
- `subpixel=True`
- Layer 0（最精細層）
- 中間角度最佳（`best_idx == 1`）
- 非邊界點
- 3 個角度層都有有效的 3×3 分數

---

## 7. nms.py — 非極大值抑制

### 7.1 軸對齊 NMS

```python
def nms(boxes, scores, max_overlap=0.5) -> List[int]
```

- `boxes`: `(x, y, w, h)` 格式
- Greedy NMS：高分優先，IoU > threshold 則抑制

### 7.2 旋轉矩形 NMS

```python
def nms_rotated(rects, scores, max_overlap=0.5) -> List[int]
```

- `rects`: OpenCV `RotatedRect` 格式 `((cx,cy), (w,h), angle)`
- **AABB 預篩**：先算軸對齊包圍框，不重疊直接跳過
- 旋轉矩形交集：`cv2.rotatedRectangleIntersection` + `cv2.convexHull` + `cv2.contourArea`

### 7.3 `rotated_rect_overlap()`

```python
def rotated_rect_overlap(rect1, rect2) -> float
```

- 分母是 **rect1 的面積**（非 IoU），與 C++ `FilterWithRotatedRect` 一致
- 三種情況：`INTERSECT_NONE` → 0, `INTERSECT_FULL` → min(a1,a2)/a1, `INTERSECT_PARTIAL` → hull_area/a1

### 7.4 `filter_with_score()`

```python
def filter_with_score(scores, threshold) -> List[int]
```

- 回傳 score >= threshold 的索引，按分數降序

---

## 8. cli.py — 命令列介面

```bash
fpm-py --src image.bmp --tmpl template.bmp \
       --target-num 5 --score 0.8 --tolerance-angle 180 \
       --backend opencv --method ccoeff_normed
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--src` | (必填) | 來源影像路徑 |
| `--tmpl` | (必填) | 模板影像路徑 |
| `--target-num` | 1 | 最大匹配數 |
| `--score` | 0.5 | 最低分數門檻 |
| `--overlap` | 0.5 | NMS 最大重疊比 |
| `--tolerance-angle` | 0.0 | 角度容忍度 (±度) |
| `--backend` | opencv | `opencv` 或 `fft` |
| `--method` | ccoeff_normed | NCC 方法 |
| `--no-subpixel` | False | 停用亞像素精化 |
| `--min-area` | 256 | 金字塔頂層最小面積 |
| `--angle` | None | `--tolerance-angle` 的別名 |

---

## 9. gui.py — 圖形介面

### 9.1 視窗結構

```
┌─────────────────────────────────────────────────────────┐
│ Menu Bar (File | Help)                                   │
├────────────────┬────────────────────────────────────────┤
│ Left Panel     │ Right Canvas                           │
│ (280px, 可捲動) │ (結果影像，支援 Zoom/Pan)               │
│                │                                        │
│ [Load Src]     │                                        │
│ [Load Tmpl]    │     結果影像 + 匹配框繪製               │
│ ─────────────  │                                        │
│ Parameters:    │                                        │
│  Target Num    │                                        │
│  Score         │                                        │
│  Overlap       │                                        │
│  Angle         │                                        │
│  Backend       │                                        │
│  Method        │                                        │
│ ─────────────  │                                        │
│ [Match!]       │                                        │
│ ─────────────  │                                        │
│ Results Tree   │                                        │
│  #0 Score=0.99 │                                        │
│  #1 Score=0.85 │                                        │
│ ─────────────  │                                        │
│ Template       │                                        │
│ Preview        │                                        │
│ [Export JSON]  │                                        │
└────────────────┴────────────────────────────────────────┘
```

### 9.2 關鍵繪製邏輯

**旋轉矩形角點計算**（與 pyramid.py 一致）：

```python
dRAngle = angle_deg * D2R       # angle_deg 是 output_angle
cos_a = cos(dRAngle)
sin_a = sin(dRAngle)

# 方向向量 u=(cos,-sin), v=(sin,cos)
ptLT = center - (w/2)×u - (h/2)×v
ptRT = ptLT + w×u
ptLB = ptLT + h×v
ptRB = ptRT + h×v
```

**繪製功能**：
- 旋轉矩形框（4 邊虛線）
- 中心十字標記
- 編號文字（FONT_HERSHEY_SIMPLEX, scale=1.2）

### 9.3 平台特殊處理

**Tk 9.0 macOS 觸控板捲動**：
- 使用 `<TouchpadScroll>` 事件（非 `<MouseWheel>`）
- Delta 解碼：`tk::PreciseScrollDeltas` 優先，手動 fallback
- 像素級捲動 `yscrollincrement=1` + delta÷3 節流

---

## 10. 資料流全貌

```mermaid
flowchart TD
    subgraph 輸入
        SRC[Source Image]
        TMPL[Template Image]
    end

    subgraph "① LearnPattern"
        TMPL --> LP[learn_pattern]
        LP --> TD[TemplateData<br/>pyramid + stats]
    end

    subgraph "② 建立金字塔"
        SRC --> BP[_build_pyramid]
        BP --> SP[src_pyr<br/>level 0~N]
    end

    subgraph "③ 角度步進"
        TD --> AS[angle_step =<br/>atan(2/max_dim)]
        AS --> ANGLES[角度列表<br/>0, ±step, ±2step...]
    end

    subgraph "⑤ Top Layer 全域搜尋"
        SP --> |"level N"| TL
        TD --> |"tmpl level N"| TL
        ANGLES --> TL[對每個角度:<br/>rotate_source → ncc_match]
        TL --> CANDS[candidates<br/>pt, score, angle]
    end

    subgraph "⑥ 逐層精化"
        CANDS --> REF[layer N-1 → 0:<br/>_get_rotated_roi → ncc_match<br/>3 角度測試]
        SP --> REF
        TD --> REF
        REF --> |"layer 0"| SUB[refine_peak_3d<br/>27 點亞像素]
        SUB --> RAW[raw results<br/>pt, score, angle]
    end

    subgraph "⑦⑧⑨ 後處理"
        RAW --> FS[filter_with_score]
        FS --> NMS[nms_rotated / nms]
        NMS --> OUT[最終結果<br/>index, score, angle,<br/>pos_x, pos_y]
    end
```

---

## 11. C++ ↔ Python 對照表

### 函式對應

| C++ 函式 (match_core.cpp) | Python 函式 | 位置 |
|---------------------------|-------------|------|
| `LearnPattern()` :85-120 | `learn_pattern()` | pyramid.py:178 |
| `Match()` :135-401 | `pyramid_match()` | pyramid.py:314 |
| `GetTopLayer()` :122-133 | `_get_top_layer()` | pyramid.py:82 |
| `buildPyramid()` | `_build_pyramid()` | pyramid.py:71 |
| `MatchTemplate()` :453+ | `ncc_match()` / `ncc_match_fft()` | ncc.py:48 / :224 |
| `GetNextMaxLoc()` :650-677 | `_suppress_and_find()` / `_find_topk_suppressed()` | pyramid.py:233 / :264 |
| `GetRotatedROI()` :506-517 | `_get_rotated_roi()` | pyramid.py:116 |
| `ptRotatePt2f()` :565-576 | `_pt_rotate()` | pyramid.py:97 |
| `GetBestRotationSize()` :519-563 | `rotate_source()` | rotation.py:54 |
| `SubPixEstimation()` :712-764 | `refine_peak_3d()` | subpixel.py:89 |
| `FilterWithRotatedRect()` | `nms_rotated()` | nms.py:138 |

### 資料結構對應

| C++ 結構 | Python 結構 | 說明 |
|----------|-------------|------|
| `s_TemplData` | `TemplateData` | 模板預計算資料 |
| `s_MatchParameter` | `Dict` (results item) | 單一匹配結果 |
| `s_BlockMax` | `topk_blocked()` | 分塊最大值搜尋 |
| `vecLayerScore` | `_layer_thresholds()` | 逐層分數衰減 |

### 常數對應

| C++ | Python | 值 |
|-----|--------|-----|
| `R2D` | `R2D` | `180/π ≈ 57.2958` |
| `D2R` | `D2R` | `π/180 ≈ 0.01745` |
| `VISION_TOLERANCE` | `VISION_TOLERANCE` | `0.0001` |
| `MATCH_CANDIDATE_NUM` | `MATCH_CANDIDATE_NUM` | `5` |

### Python 特有優化

| 優化 | 說明 | 對應 C++ |
|------|------|---------|
| FFT NCC 後端 | `scipy.fft` 多執行緒 + 積分影像 | 無（C++ 用 SIMD） |
| 向量化 Top-K | `np.argpartition` O(n) | 反覆 `minMaxLoc` O(kn) |
| AABB 預篩 NMS | 跳過不可能重疊的矩形對 | 無 |
| Numba JIT | 抑制 + argmax 加速 | SIMD intrinsics |
| Block Max 向量化 | reshape+transpose 一次完成 | 逐塊迴圈 |

---

## 12. 依賴與環境

### 必要依賴

| 套件 | 用途 |
|------|------|
| `numpy` | 陣列運算、線性代數 |
| `opencv-python` | 影像 I/O、matchTemplate、warpAffine、積分影像 |
| `tqdm` | 進度條（CLI） |
| `Pillow` | GUI 影像顯示 |

### 可選加速

| 套件 | 用途 | 加速幅度 |
|------|------|---------|
| `scipy` | `scipy.fft` 多執行緒 FFT | FFT 後端 2-4× |
| `numba` | JIT 編譯 suppress+argmax | Top-K 搜尋 ~3× |

### 安裝與執行

```bash
cd python-port
pip install -e .[dev]       # 開發安裝

# CLI
fpm-py --src image.bmp --tmpl template.bmp --target-num 5

# GUI
fpm-py-gui

# 測試
pytest tests/ -v
```
