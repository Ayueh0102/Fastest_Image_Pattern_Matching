# 演算法原理詳解

本文件深入解說 Python Port 的核心演算法，包含完整數學推導、與 C++ 原版的逐步對應、以及各項優化技巧。適合想深入理解「為什麼這樣做」的讀者。

> **互動式視覺化**：建議搭配 [algorithm-visual.html](algorithm-visual.html) 一起閱讀，有動畫演示每個階段的運作方式。

---

## 目錄

0. [直觀總覽 — 為什麼這個演算法有效？](#0-直觀總覽--為什麼這個演算法有效)
1. [NCC — 正規化互相關](#1-ncc--正規化互相關)
2. [影像金字塔](#2-影像金字塔)
3. [旋轉搜尋策略](#3-旋轉搜尋策略)
4. [Top Layer 全域搜尋](#4-top-layer-全域搜尋)
5. [Coarse-to-Fine 逐層精化](#5-coarse-to-fine-逐層精化)
6. [次像素估計](#6-次像素估計)
7. [非極大值抑制 (NMS)](#7-非極大值抑制-nms)
8. [FFT NCC 加速](#8-fft-ncc-加速)
9. [Top-K 峰值搜尋](#9-top-k-峰值搜尋)
10. [完整流程圖解](#10-完整流程圖解)
11. [複雜度分析](#11-複雜度分析)
12. [數值穩定性](#12-數值穩定性)

---

## 0. 直觀總覽 — 為什麼這個演算法有效？

### 問題場景

> 在一張大照片中找到所有與小模板「長得一樣」的位置，即使目標旋轉了任意角度。

這在工業視覺檢測中非常常見：PCB 上的螺絲、藥錠在輸送帶上的定位、晶圓上的對位記號…。

### 為什麼不直接暴力搜尋？

假設影像 4000×3000，模板 100×100，搜尋 360°：
- 位置候選：~1200 萬
- 每個位置 × 360 個角度 ≈ **43 億次比對**
- 每次比對需計算 10000 像素的 NCC → **天文數字**

### 核心策略：由粗到細

本演算法用四個技巧把計算量降低數千倍：

| 技巧 | 類比 | 效果 |
|------|------|------|
| **影像金字塔** | 先看縮圖找大致位置，再看原圖精修 | 搜尋空間 ÷ 64~256 |
| **角度粗→細** | 先每 10° 掃一遍，再在最佳角度 ±10° 內每 1° 精修 | 角度候選 ÷ 3~10 |
| **ROI 裁切** | 精化時只看候選周圍的小區域，不看整張圖 | 每層計算量 ÷ 100+ |
| **提前終止** | 低分候選直接淘汰，不往下精化 | 節省無用計算 |

### 一句話總結

> **在最小的圖上窮舉所有角度 → 找到候選 → 逐層放大精修位置和角度 → 次像素精化 → NMS 去重疊**

這就是為什麼一個 4000×3000 的影像 + 360° 搜尋，在 Python 上也能在幾百毫秒內完成。

---

## 1. NCC — 正規化互相關

### 1.1 問題定義

給定來源影像 **I** (H×W) 與模板 **T** (h×w)，需要在 I 中找到與 T 最相似的位置。

NCC 是衡量「兩個信號形狀是否一致」的度量，具有光照不變性——即便整體亮度改變（平移+縮放），NCC 分數不受影響。

### 1.2 數學定義

#### TM_CCOEFF_NORMED（零均值正規化互相關）

在 source 影像的位置 (x, y) 計算：

```
                  Σᵢⱼ [I(x+i, y+j) - μᵢ] × [T(i,j) - μₜ]
NCC(x,y) = ──────────────────────────────────────────────────────────
            √{ Σᵢⱼ [I(x+i, y+j) - μᵢ]² } × √{ Σᵢⱼ [T(i,j) - μₜ]² }
```

其中：
- `μᵢ = (1/N) Σᵢⱼ I(x+i, y+j)` — 滑窗內 source 區域的平均值
- `μₜ = (1/N) Σᵢⱼ T(i,j)` — 模板平均值
- `N = h × w` — 模板像素數
- 求和範圍：`i ∈ [0, w-1], j ∈ [0, h-1]`
- 結果範圍：**[-1, 1]**，1 表示完全正相關

**光照不變性證明**：若 I' = αI + β（亮度線性變換），則 `I' - μᵢ' = α(I - μᵢ)`，代入後 α 在分子分母抵消。

#### TM_CCORR_NORMED（相關係數正規化）

```
                    Σᵢⱼ I(x+i, y+j) × T(i,j)
CCORR(x,y) = ────────────────────────────────────────────
              √{ Σᵢⱼ I(x+i, y+j)² } × √{ Σᵢⱼ T(i,j)² }
```

- 不去均值，對絕對亮度敏感
- 結果範圍：[0, 1]

### 1.3 Python 實作

```python
# ncc.py:48-86 — OpenCV 後端
def ncc_match(src, tmpl, method="ccoeff_normed"):
    cv_method = cv2.TM_CCOEFF_NORMED  # 或 TM_CCORR_NORMED
    resp = cv2.matchTemplate(src_gray, tmpl_gray, cv_method)
    return resp  # shape: (H-h+1, W-w+1)
```

OpenCV `matchTemplate` 內部使用 DFT 加速（大模板時自動切換），但 Python 版另外提供純 FFT 後端（見[第 8 節](#8-fft-ncc-加速)）。

### 1.4 模板預計算（LearnPattern）

C++ 的 `LearnPattern` 在 Python 中對應 `learn_pattern()`，預先計算各金字塔層的統計量：

```python
# pyramid.py:178-216
mean_val, stddev_val = cv2.meanStdDev(pyramid[i])
tmpl_mean = mean_val[0][0]
tmpl_norm_sq = stddev_val[0][0] ** 2   # variance
tmpl_norm = sqrt(variance) / sqrt(inv_area)  # = stddev × sqrt(area)
```

**為什麼 norm = stddev × sqrt(area)?**

NCC 分母中模板部分：
```
√{ Σᵢⱼ (T - μₜ)² } = √{ N × variance } = stddev × √N
```

其中 `inv_area = 1/N`，所以 `stddev × √N = stddev / √(inv_area) = √(variance) / √(inv_area)`。

---

## 2. 影像金字塔

### 2.1 原理

金字塔是一種**多解析度表示法**，用來將搜尋空間從 O(W×H×A) 降低到 O((W/2ⁿ)×(H/2ⁿ)×A)，其中 n 為層數，A 為角度數。

策略：先在小圖（低解析度）找到大致位置，再逐層放大精修——避免在原始大圖上窮舉搜尋。

### 2.2 建構方式

使用 `cv2.pyrDown`（高斯 ÷2 下採樣）：

```python
# pyramid.py:71-79
def _build_pyramid(img, max_level):
    pyr = [img]                    # level 0 = 原圖
    for _ in range(max_level):
        pyr.append(cv2.pyrDown(pyr[-1]))  # 每層寬高各 ÷2
    return pyr  # [原圖, ÷2, ÷4, ..., ÷2^N]
```

**為什麼用 pyrDown 而非 resize?**
- `pyrDown` 先做高斯模糊再下採樣，避免混疊（aliasing）
- 與 C++ 的 `buildPyramid` 行為完全一致

### 2.3 層數計算

```python
# pyramid.py:82-94
def _get_top_layer(tmpl, min_reduced_area=256):
    area = h × w
    top_layer = 0
    while area > min_reduced_area:
        area //= 4        # 每層面積 ÷4
        top_layer += 1
    return top_layer
```

例子：

| 模板尺寸 | 面積 | min_area=256 | 層數 |
|----------|------|-------------|------|
| 200×200 | 40000 | 40000→10000→2500→625→156 | 4 |
| 50×50 | 2500 | 2500→625→156 | 2 |
| 30×30 | 900 | 900→225 | 1 |
| 15×15 | 225 | 225 ≤ 256 | 0 |

### 2.4 層級分數門檻

```python
# pyramid.py:162-170
def _layer_thresholds(base_score, num_levels):
    thresholds = [base_score]
    for _ in range(1, num_levels):
        thresholds.append(thresholds[-1] * 0.9)
    return thresholds
```

- Layer 0（最精細）：`score`（使用者指定，如 0.8）
- Layer 1：`score × 0.9`
- Layer 2：`score × 0.81`
- ...
- Top Layer：最寬鬆

**為什麼逐層衰減？** 低解析度時影像模糊，NCC 分數自然偏低。0.9 的衰減係數是經驗值。

---

## 3. 旋轉搜尋策略

### 3.1 自適應角度步進

```python
# pyramid.py:377
angle_step = atan(2.0 / max(top_tmpl_w, top_tmpl_h)) × R2D
```

**幾何意義**：模板上最遠的點距中心 `r = max(w,h)/2`，旋轉 `angle_step` 度後位移量 = `r × sin(angle_step) ≈ r × angle_step(rad) ≈ 1 pixel`。也就是說角度步進恰好讓模板邊緣移動約 1 pixel，不遺漏也不冗餘。

| 模板最大維度 | angle_step |
|-------------|------------|
| 10 px | 11.3° |
| 30 px | 3.81° |
| 50 px | 2.29° |
| 100 px | 1.15° |
| 200 px | 0.57° |

### 3.2 角度列舉順序

```python
# pyramid.py:380-391
# tolerance_angle = 180 → 搜尋 ±180°
angles = [0, +step, +2×step, ..., +180,
          -step, -2×step, ..., -180]
```

與 C++ 一致：先正角度再負角度。0° 最先測試，讓最常見的無旋轉情況最快被處理。

### 3.3 旋轉策略：旋轉 Source 而非 Template

**Top Layer**：旋轉 source 影像，用**原始（未旋轉）模板**做 matchTemplate。

**為什麼？**
1. 模板已預計算統計量（mean, norm），旋轉後需重新計算
2. 旋轉 source 後可直接用 matchTemplate，不需改動分母
3. C++ 原版也是這樣做的（match_core.cpp:196-209）

```python
# pyramid.py:404-441
for angle in angles:
    if abs(angle) < 1e-6:
        rotated_src = top_src   # 0° 不旋轉
    else:
        rotated_src = rotate_source(top_src, angle, tmpl_shape, border_color)
        # 記錄平移量：旋轉後畫布變大，中心偏移
        trans_x = (rot_w-1)/2 - src_center[0]
        trans_y = (rot_h-1)/2 - src_center[1]

    resp = ncc_match(rotated_src, top_tmpl)  # 未旋轉模板
    peaks = _find_topk_suppressed(resp, ...)
    # 候選點扣除平移量，轉回原始座標系
    for px, py, sc in peaks:
        candidates.append((px - trans_x, py - trans_y, sc, angle))
```

---

## 4. Top Layer 全域搜尋

### 4.1 搜尋空間

在金字塔頂層（最小層），影像尺寸已縮小到約 32×32 ~ 128×128。在這個小空間中做**全域**搜尋，對每個角度都遍歷整張圖。

### 4.2 峰值搜尋

每次 matchTemplate 產生一張 response map，需從中提取前 K 個峰值：

```python
topk_count = target_num + MATCH_CANDIDATE_NUM  # 比目標數多取 5 個
peaks = _find_topk_suppressed(resp, topk_count, suppress_w, suppress_h, threshold)
```

抑制窗口大小 = `tmpl_size × (1 - max_overlap)`，避免同一目標重複取到。

### 4.3 候選排序

所有角度的候選匯總後按分數降序排列：

```python
candidates.sort(key=lambda c: c[2], reverse=True)
```

後續精化從最高分開始，並在收集到足夠結果後提前終止。

---

## 5. Coarse-to-Fine 逐層精化

### 5.1 精化流程

對 Top Layer 的每個候選，從 `layer N-1` 到 `layer 0` 逐層精修：

```
Top Layer (N) → 候選 (ptLT, angle)
    ↓ ×2 放大座標
Layer N-1 → ROI 提取 → matchTemplate → 3角度最佳 → 更新 ptLT, angle
    ↓ ×2 放大座標
Layer N-2 → ROI 提取 → matchTemplate → 3角度最佳 → 更新 ptLT, angle
    ↓ ...
Layer 0 → ROI 提取 → matchTemplate → 亞像素精化 → 最終結果
```

### 5.2 ROI 提取（GetRotatedROI）

不是在整張大圖搜尋，而是只裁切候選點附近的小區域：

```python
# pyramid.py:116-159
def _get_rotated_roi(src, tmpl_size, pt_lt, angle_deg):
    padding = 3
    roi_w = tw + 2 × padding    # 多留 ±3 pixel
    roi_h = th + 2 × padding

    # 1. 將 ptLT 旋轉到旋轉座標系
    pt_rotated = _pt_rotate(pt_lt, (cx, cy), angle × D2R)

    # 2. 建立旋轉+平移矩陣，裁切 ROI
    M = getRotationMatrix2D(center, angle, 1.0)
    M[0,2] -= pt_rotated[0] - padding
    M[1,2] -= pt_rotated[1] - padding

    return warpAffine(src, M, (roi_w, roi_h))
```

**為什麼需要 padding?**
- matchTemplate 輸出尺寸 = `(roi_h - th + 1) × (roi_w - tw + 1)` = `(2×3+1) × (2×3+1)` = `7×7`
- 最大值位置 (3,3) 表示候選恰好在預期位置；偏移 ±3 表示有微調空間

### 5.3 三角度測試

每層測試 3 個角度：

```python
test_angles = [matched_angle - local_step, matched_angle, matched_angle + local_step]
```

- `local_step` 用該層模板尺寸重新計算（層越大、模板越大、步進越小）
- 取三者中分數最高者作為本層最佳角度

### 5.4 座標逆轉換

ROI 內的 `maxLoc` → 加上 `ptPaddingLT`（旋轉空間中的偏移）→ 旋轉回原始座標系：

```python
# pyramid.py:585-611
pt_padding_lt = _pt_rotate(pt_lt_scaled, cur_center, angle × D2R)
pt_padding_lt = (pt_padding_lt[0] - 3, pt_padding_lt[1] - 3)

pt_abs = (maxLoc[0] + pt_padding_lt[0], maxLoc[1] + pt_padding_lt[1])
pt_orig = _pt_rotate(pt_abs, cur_center, -angle × D2R)
```

### 5.5 分數門檻篩除

每層精化後檢查分數：

```python
if best_score < layer_scores[layer]:
    break  # 放棄此候選
```

低分候選在早期就被淘汰，節省後續計算。

---

## 6. 次像素估計

### 6.1 動機

matchTemplate 的峰值位於整數像素格點上。實際的真正峰值可能在像素之間。次像素估計將定位精度從 1 pixel 提升到 ~0.1 pixel。

### 6.2 方法一：2D 二次曲面擬合（9 點）

在峰值 (x, y) 的 3×3 鄰域擬合二次曲面：

```
z(x, y) = ax² + by² + cxy + dx + ey + f
```

6 個未知參數，9 個觀測值 → 過定系統，用 SVD 最小平方法求解。

**極值位置**（對 x, y 偏微分 = 0）：

```
∂z/∂x = 2ax + cy + d = 0
∂z/∂y = cx + 2by + e = 0

解得：
    dx = (ce - 2bd) / (4ab - c²)
    dy = (cd - 2ae) / (4ab - c²)
```

偏移量限制在 [-1, 1]，超出表示擬合不穩定。

### 6.3 方法二：3D 二次曲面擬合（27 點）

同時精修 x, y 與角度 θ，使用 3×3×3 = 27 個觀測點（3 個角度層 × 3×3 空間）。

**十參數曲面模型**：

```
z(x, y, θ) = ax² + by² + cθ² + dxy + exθ + fyθ + gx + hy + iθ + j
```

**建立方式**（使用**絕對座標**，與 C++ 一致）：

```python
# subpixel.py:132-162
for theta_idx in range(3):       # 3 個角度
    theta_rad = deg2rad(center_angle + (theta_idx-1) × angle_step)
    for oy in range(-1, 2):       # 3 行
        for ox in range(-1, 2):   # 3 列
            xi = center_x + ox
            yi = center_y + oy
            ti = theta_rad
            A[row] = [xi², yi², ti², xi×yi, xi×ti, yi×ti, xi, yi, ti, 1]
            S[row] = scores[theta_idx, oy+1, ox+1]
```

**求解**：

1. SVD 最小平方：`z = lstsq(A, S)`
2. 建構 Hessian 矩陣和梯度向量：

```
        ⎡ 2a  d  e ⎤              ⎡ -g ⎤
K1  =   ⎢  d  2b f ⎥      K2  =  ⎢ -h ⎥
        ⎣  e  f  2c⎦              ⎣ -i ⎦
```

3. 解線性方程：`K1 × δ = K2`，`δ = (x*, y*, θ*)` 為最佳座標
4. 偏移量 = 最佳座標 - 中心座標

**觸發條件**（pyramid.py:559-584）：
- Layer 0（最精細層）
- 中間角度是三者最佳（`best_idx == 1`）
- 非邊界點
- 三個角度層都有有效的 3×3 鄰域

### 6.4 精度分析

| 方法 | 點數 | 參數數 | 精度 | 同時精修角度 |
|------|------|--------|------|-------------|
| 2D (9點) | 9 | 6 | ~0.1-0.5 px | 否 |
| 3D (27點) | 27 | 10 | ~0.05-0.3 px + ~0.1° | 是 |

---

## 7. 非極大值抑制 (NMS)

### 7.1 問題

由於多角度搜尋和逐層精化，同一目標可能產生多個高分候選（不同角度或相鄰位置）。NMS 去除重複偵測，只保留最佳的。

### 7.2 軸對齊 NMS（angle=0 時使用）

```python
# nms.py:52-68
def nms(boxes, scores, max_overlap=0.5):
    order = argsort(scores)[::-1]  # 高分優先
    for idx in order:
        for kept in keep:
            if rect_iou(boxes[idx], boxes[kept]) > max_overlap:
                suppress  # 抑制此候選
```

IoU 計算：`intersection_area / union_area`

### 7.3 旋轉矩形 NMS（有旋轉時使用）

```python
# nms.py:138-188
def nms_rotated(rects, scores, max_overlap=0.5):
    aabbs = [_aabb_from_rotated_rect(r) for r in rects]  # 預算 AABB

    for idx in order:
        for kept in keep:
            if not _aabb_overlap(aabbs[idx], aabbs[kept]):
                continue  # AABB 不重疊 → 跳過（省時）
            overlap = rotated_rect_overlap(rects[idx], rects[kept])
            if overlap > max_overlap:
                suppress
```

**三層加速**：
1. **分數排序**：低分候選直接被高分候選抑制
2. **AABB 預篩**：軸對齊包圍框不重疊 → 旋轉矩形必然不重疊
3. **精確交集**：`cv2.rotatedRectangleIntersection` + convexHull

### 7.4 重疊度定義

```python
# nms.py:75-119
def rotated_rect_overlap(rect1, rect2):
    return intersection_area / rect1_area  # 注意：分母是 rect1 面積，非 IoU
```

與 C++ 一致。這意味著小矩形完全包含在大矩形內時，overlap ≈ 1.0（會被抑制）。

---

## 8. FFT NCC 加速

### 8.1 卷積定理

互相關（Cross-correlation）可用 FFT 加速：

```
CC(I, T) = IFFT( FFT(I) × conj(FFT(T)) )
```

- 時域互相關 O(WHwh) → 頻域乘法 O(WH log WH)
- 模板越大，加速比越高

### 8.2 實作步驟

```python
# ncc.py:224-289
def ncc_match_fft(src, tmpl, method="ccoeff_normed"):
    # ① 零均值模板（ccoeff_normed）
    mu_t = tmpl.mean()
    T = tmpl - mu_t
    sum_t2 = (T * T).sum()

    # ② FFT 交互相關
    P, Q = H+h-1, W+w-1           # Full convolution size
    FI = fft2(src, s=(P, Q))
    FT = fft2(T, s=(P, Q))
    CC_full = ifft2(FI × conj(FT)).real

    # ③ 擷取 Valid 區域
    CC = CC_full[0 : H-h+1, 0 : W-w+1]
    #                ↑ 重要：是 [0:H-h+1]，非 [h-1:H]

    # ④ 滑窗統計（積分影像加速）
    sums, sums2 = _integral_sums(src, (h, w))
    mean_I = sums / N
    var_I = max(sums2 - N × mean_I², 0)
    std_I = sqrt(var_I)

    # ⑤ 正規化
    resp = CC / (std_I × sqrt(sum_t2))
```

### 8.3 Valid Region 的正確提取

FFT 交互相關的 `CC_full` 尺寸為 `(P, Q) = (H+h-1, W+w-1)`。

**Valid region 在左上角** `[0:H-h+1, 0:W-w+1]`，而非 `[h-1:H, w-1:W]`。

原因：`CC_full[dy, dx]` 表示模板在位置 `(dx, dy)` 的互相關值，`(0,0)` = 模板左上角對齊 source 左上角。

### 8.4 積分影像加速分母

NCC 分母需要每個滑窗位置的 `sum(I)` 和 `sum(I²)`。

**積分影像**（Summed Area Table, SAT）讓任意矩形區域求和為 O(1)：

```python
# ncc.py:202-221
integ = cv2.integral(img)           # (H+1, W+1)
integ2 = cv2.integral(img * img)
sums  = integ[kh:,kw:]  - integ[:-kh,kw:]  - integ[kh:,:-kw]  + integ[:-kh,:-kw]
sums2 = integ2[kh:,kw:] - integ2[:-kh,kw:] - integ2[kh:,:-kw] + integ2[:-kh,:-kw]
```

一次建立 O(HW)，之後每個滑窗 O(1)，比逐窗計算 O(hw) 快得多。

### 8.5 FFT 後端選擇

```python
if scipy 可用:
    fft2(x, s=s, workers=-1)   # 多執行緒 FFTW
else:
    np.fft.fft2(x, s=s)        # 單執行緒 NumPy
```

scipy.fft 在 multi-core 上快 2-4 倍。

---

## 9. Top-K 峰值搜尋

### 9.1 問題

matchTemplate 輸出 response map，需要從中找到前 K 個不重疊的峰值。

### 9.2 C++ 做法（反覆 minMaxLoc）

```
for i in range(K):
    找最大值 → 記錄 → 抑制周圍區域 → 再找最大值
```

每次 `minMaxLoc` 遍歷整張圖 O(HW)，總計 O(K × HW)。

### 9.3 Python 向量化做法

```python
# pyramid.py:264-306
def _find_topk_suppressed(resp, k, suppress_w, suppress_h, threshold):
    flat = resp.ravel()
    n_cand = min(k × 50, len(flat))

    # O(n) argpartition 找前 n_cand 大
    part_idx = np.argpartition(flat, -n_cand)[-n_cand:]
    # O(n_cand log n_cand) 排序
    sort_order = np.argsort(flat[part_idx])[::-1]

    # 距離抑制：O(n_cand × k)
    for each sorted candidate:
        if not too close to any picked:
            pick it
```

**總複雜度**：O(HW + n_cand × k)，比 C++ 的 O(K × HW) 快得多。

### 9.4 Block Max（分塊最大值）

```python
# ncc.py:142-181
def topk_blocked(response, block_size, k=50, score_thresh=0.0):
    # 向量化 reshape：將響應圖重塑為 (nrows, ncols, block_area)
    blocks = padded.reshape(nrows, bh, ncols, bw).transpose(0, 2, 1, 3)
    blocks_flat = blocks.reshape(nrows, ncols, -1)

    max_vals = blocks_flat.max(axis=2)       # 每塊最大值
    max_idxs = blocks_flat.argmax(axis=2)    # 每塊最大值位置
```

**適用場景**：大圖 + 小模板（如 3648×3648 + 54×54），response map 巨大但大部分區域分數很低。

---

## 10. 完整流程圖解

### 10.1 無旋轉匹配 (angle=0)

```
Source (1024×768)  +  Template (64×48)
    ↓
LearnPattern → 4 層金字塔
    layer 0: 64×48 (原圖)
    layer 1: 32×24
    layer 2: 16×12
    layer 3: 8×6

Source 金字塔：
    layer 0: 1024×768
    layer 1: 512×384
    layer 2: 256×192
    layer 3: 128×96

Top Layer (3) 搜尋：
    angles = [0°]（無旋轉）
    matchTemplate(src_128×96, tmpl_8×6) → resp 121×91
    top-K → 6 個候選

Layer 2 精化：
    每個候選 → ROI 22×18 → matchTemplate → 最佳位置

Layer 1 精化：
    每個候選 → ROI 38×30 → matchTemplate → 最佳位置

Layer 0 精化：
    每個候選 → ROI 70×54 → matchTemplate → 次像素 → 最終位置

NMS → 輸出
```

### 10.2 旋轉匹配 (angle=180°)

```
Top Layer (3)：
    模板 8×6 → angle_step = atan(2/8) × R2D ≈ 14.0°
    angles = [0°, 14°, 28°, ..., 180°, -14°, -28°, ..., -180°]
    共 ~26 個角度

    對每個角度：
        rotate_source(src_128×96, angle) → 旋轉後 source
        matchTemplate(rotated_src, tmpl_8×6) → resp
        top-K → 候選

    匯總所有角度候選 → 按分數排序

逐層精化：每層測試 3 個角度 (angle ± local_step)
    Layer 2: local_step = atan(2/16) × R2D ≈ 7.1°
    Layer 1: local_step = atan(2/32) × R2D ≈ 3.6°
    Layer 0: local_step = atan(2/64) × R2D ≈ 1.8° → 27 點亞像素

NMS (旋轉矩形) → 輸出
```

---

## 11. 複雜度分析

### 11.1 時間複雜度

設 source 尺寸 W×H，模板 w×h，層數 N，角度數 A。

| 階段 | 複雜度 | 說明 |
|------|--------|------|
| LearnPattern | O(wh) | 金字塔建構 + 統計量 |
| Source 金字塔 | O(WH) | pyrDown 每層 O(HW/4^i) |
| Top Layer 搜尋 | O(A × (W/2ᴺ)(H/2ᴺ) × wh/4ᴺ) | A 角度 × matchTemplate |
| 逐層精化 | O(K × N × wh) | K 候選 × N 層 × 小 ROI |
| 次像素 | O(K × 27 × 10) | 27 點 × 10 參數最小平方 |
| NMS | O(K² × 旋轉交集) | Greedy + AABB 預篩 |

**瓶頸**：通常是 Top Layer 搜尋（角度數 × matchTemplate），但因為在最小層執行，實際計算量很小。

### 11.2 空間複雜度

| 項目 | 記憶體 |
|------|--------|
| Source 金字塔 | O(WH × 4/3) ≈ 1.33 × 原圖 |
| 模板金字塔 | O(wh × 4/3) |
| Response map | O(WH/4ᴺ)（Top Layer）或 O(ROI)（精化層）|
| FFT 緩衝 | O(WH)（FFT 後端額外開銷）|

### 11.3 實測效能（Apple M1 Pro）

| 案例 | Source | Template | 角度 | OpenCV | FFT |
|------|--------|----------|------|--------|-----|
| 無旋轉 | 4024×3036 | 762×521 | 0° | 32 ms | ~80 ms |
| 旋轉 180° | 4024×3036 | 762×521 | 180° | ~200 ms | ~3000 ms |
| 多目標 | 3648×3648 | 54×54 | 0° | 45 ms | ~30 ms |

**觀察**：
- 大模板：OpenCV 後端較快（內部 DFT 已優化）
- 小模板 + 大圖：FFT 後端較快（避免 matchTemplate 的 O(wh) overhead per pixel）
- 旋轉搜尋：計算量正比角度數

---

## 12. 數值穩定性

### 12.1 分母保護

NCC 分母可能為 0（均勻區域）：

```python
# ncc.py:280
np.divide(CC, np.maximum(denom, 1e-12), out=resp, where=denom > 0)
```

- `denom > 0` 的位置才做除法
- 小分母用 `1e-12` 保護
- 均勻模板（`result_equal1 = True`）直接回傳 1.0

### 12.2 偏移量限制

次像素偏移超過 [-1, 1] 表示擬合不穩定：

```python
dx = float(np.clip(dx, -1.0, 1.0))
dy = float(np.clip(dy, -1.0, 1.0))
```

### 12.3 Hessian 奇異性

3D 次像素的 `K1 × δ = K2` 可能無解：

```python
try:
    delta = np.linalg.solve(K1, K2)
except np.linalg.LinAlgError:
    return 0.0, 0.0, 0.0  # 放棄精化
```

### 12.4 邊界檢查

ROI 精化中的邊界檢查：

```python
on_border = (maxLoc[0] == 0 or maxLoc[1] == 0 or
             maxLoc[0] == resp.shape[1]-1 or maxLoc[1] == resp.shape[0]-1)
```

峰值在邊界 → 真正峰值可能在 ROI 外 → 不做次像素精化。

### 12.5 OpenCV 版本差異

`cv2.pyrDown` / `cv2.warpAffine` 在不同 OpenCV 版本（如 4.5 vs 4.13）有微小數值差異（~0.001），可能導致邊界 case 的分數些許不同。這是已知限制。
