# 10. 深度技術解析

## 🧠 演算法深度剖析

本章將深入探討 Python 移植版本的核心技術實作，包括數學原理、優化策略和實作細節。

---

## 🔬 NCC 演算法深度解析

### 數學基礎與變種比較

#### 標準 NCC 公式推導

正規化互相關的數學定義：

```
NCC(x,y) = Σ[I(x+i,y+j) - μᵢ][T(i,j) - μₜ] / √[Σ(I(x+i,y+j) - μᵢ)² × Σ(T(i,j) - μₜ)²]
```

其中：
- I(x+i,y+j) = 影像在位置 (x+i,y+j) 的像素值
- T(i,j) = 樣板在位置 (i,j) 的像素值  
- μᵢ = 影像區域平均值
- μₜ = 樣板平均值

#### OpenCV 變種的數值分析

**1. TM_CCOEFF_NORMED (零均值版本)**

```python
def manual_ccoeff_normed(src_region: np.ndarray, tmpl: np.ndarray) -> float:
    """手動實作 CCOEFF_NORMED 以理解演算法"""
    
    # 轉換為浮點數避免溢出
    src_f = src_region.astype(np.float64)
    tmpl_f = tmpl.astype(np.float64)
    
    # 計算均值
    src_mean = np.mean(src_f)
    tmpl_mean = np.mean(tmpl_f)
    
    # 零均值化
    src_centered = src_f - src_mean
    tmpl_centered = tmpl_f - tmpl_mean
    
    # 計算分子（互相關）
    numerator = np.sum(src_centered * tmpl_centered)
    
    # 計算分母（標準差乘積）
    src_var = np.sum(src_centered ** 2)
    tmpl_var = np.sum(tmpl_centered ** 2)
    denominator = np.sqrt(src_var * tmpl_var)
    
    # 避免除零
    if denominator < 1e-10:
        return 0.0
    
    return numerator / denominator

# 數值驗證
src_patch = src[100:150, 200:250]  # 50x50 區域
manual_result = manual_ccoeff_normed(src_patch, tmpl)
opencv_result = cv2.matchTemplate(src_patch, tmpl, cv2.TM_CCOEFF_NORMED)[0,0]

print(f"手動計算: {manual_result:.6f}")
print(f"OpenCV: {opencv_result:.6f}")
print(f"差異: {abs(manual_result - opencv_result):.8f}")
```

**2. TM_CCORR_NORMED (相關係數版本)**

```python
def manual_ccorr_normed(src_region: np.ndarray, tmpl: np.ndarray) -> float:
    """手動實作 CCORR_NORMED"""
    
    src_f = src_region.astype(np.float64)
    tmpl_f = tmpl.astype(np.float64)
    
    # 直接計算相關（不去均值）
    numerator = np.sum(src_f * tmpl_f)
    
    # 計算能量
    src_energy = np.sum(src_f ** 2)
    tmpl_energy = np.sum(tmpl_f ** 2)
    denominator = np.sqrt(src_energy * tmpl_energy)
    
    if denominator < 1e-10:
        return 0.0
    
    return numerator / denominator
```

### FFT 加速實作深度解析

#### 卷積定理應用

```python
def fft_ncc_detailed_analysis(src: np.ndarray, tmpl: np.ndarray) -> Dict:
    """詳細分析 FFT NCC 的每個步驟"""
    
    H, W = src.shape
    h, w = tmpl.shape
    
    analysis = {"steps": [], "timing": {}}
    
    # 步驟 1: 預處理
    start_time = time.perf_counter()
    
    src_f = src.astype(np.float64)
    tmpl_f = tmpl.astype(np.float64)
    
    # 零填充到 2^n 大小以最佳化 FFT
    P = 2 ** int(np.ceil(np.log2(H + h - 1)))
    Q = 2 ** int(np.ceil(np.log2(W + w - 1)))
    
    analysis["steps"].append(f"零填充: {H+h-1}x{W+w-1} -> {P}x{Q}")
    analysis["timing"]["preprocessing"] = time.perf_counter() - start_time
    
    # 步驟 2: FFT 轉換
    start_time = time.perf_counter()
    
    # 來源影像 FFT
    src_padded = np.zeros((P, Q), dtype=np.float64)
    src_padded[:H, :W] = src_f
    F_src = np.fft.fft2(src_padded)
    
    # 樣板 FFT（需要翻轉以實現相關而非卷積）
    tmpl_padded = np.zeros((P, Q), dtype=np.float64)
    tmpl_padded[:h, :w] = np.flip(tmpl_f)  # 翻轉實現相關
    F_tmpl = np.fft.fft2(tmpl_padded)
    
    analysis["timing"]["fft"] = time.perf_counter() - start_time
    
    # 步驟 3: 頻域乘法
    start_time = time.perf_counter()
    
    # 頻域相關 = F(src) * conj(F(tmpl))
    F_corr = F_src * np.conj(F_tmpl)
    
    analysis["timing"]["multiplication"] = time.perf_counter() - start_time
    
    # 步驟 4: 逆 FFT
    start_time = time.perf_counter()
    
    correlation_full = np.fft.ifft2(F_corr).real
    
    # 擷取有效區域
    correlation = correlation_full[:H-h+1, :W-w+1]
    
    analysis["timing"]["ifft"] = time.perf_counter() - start_time
    
    # 步驟 5: 正規化分母計算
    start_time = time.perf_counter()
    
    # 使用積分影像快速計算滑窗統計
    src_integral = cv2.integral(src_f.astype(np.float32))
    src_sq_integral = cv2.integral((src_f ** 2).astype(np.float32))
    
    # 每個滑窗的和與平方和
    sums = (src_integral[h:, w:] - src_integral[:-h, w:] - 
            src_integral[h:, :-w] + src_integral[:-h, :-w])
    sums_sq = (src_sq_integral[h:, w:] - src_sq_integral[:-h, w:] - 
               src_sq_integral[h:, :-w] + src_sq_integral[:-h, :-w])
    
    # 計算方差
    N = h * w
    means = sums / N
    variances = sums_sq / N - means ** 2
    
    # 樣板統計
    tmpl_mean = np.mean(tmpl_f)
    tmpl_var = np.var(tmpl_f)
    
    # 分母
    denominators = np.sqrt(variances * tmpl_var * N * N)
    
    analysis["timing"]["normalization"] = time.perf_counter() - start_time
    
    # 最終正規化
    normalized_corr = np.divide(correlation, denominators, 
                               out=np.zeros_like(correlation), 
                               where=denominators > 1e-10)
    
    analysis["final_shape"] = normalized_corr.shape
    analysis["max_correlation"] = np.max(normalized_corr)
    
    return analysis, normalized_corr

# 深度分析範例
analysis, result = fft_ncc_detailed_analysis(src, tmpl)

print("FFT NCC 深度分析:")
for step in analysis["steps"]:
    print(f"  {step}")

print("\n時間分析:")
for step, duration in analysis["timing"].items():
    print(f"  {step}: {duration:.4f}s")
```

---

## 🏔️ 金字塔演算法深入分析

### 縮放策略的數學原理

#### 最佳縮放比例推導

```python
def analyze_pyramid_scaling_effects(image_sizes: List[tuple], template_size: tuple, 
                                   scales: List[float]) -> Dict:
    """分析不同縮放比例對金字塔效果的影響"""
    
    results = {}
    
    for scale in scales:
        analysis = {
            "scale": scale,
            "pyramid_levels": [],
            "total_search_space": 0,
            "reduction_factor": 0
        }
        
        for img_h, img_w in image_sizes:
            # 建立金字塔
            current_h, current_w = img_h, img_w
            level = 0
            search_spaces = []
            
            while current_w >= template_size[1] and current_h >= template_size[0]:
                # 計算搜尋空間
                search_h = current_h - template_size[0] + 1
                search_w = current_w - template_size[1] + 1
                search_space = search_h * search_w
                
                search_spaces.append({
                    "level": level,
                    "image_size": (current_h, current_w),
                    "search_space": search_space
                })
                
                # 縮放到下一層
                current_w = int(current_w / scale)
                current_h = int(current_h / scale)
                level += 1
                
                if current_w * current_h < 256:  # 最小面積限制
                    break
            
            # 計算總搜尋空間（加權）
            total_space = sum(space["search_space"] * (scale ** (2 * space["level"])) 
                            for space in search_spaces)
            
            # 原始搜尋空間
            original_space = (img_h - template_size[0] + 1) * (img_w - template_size[1] + 1)
            
            analysis["pyramid_levels"].append({
                "image_size": (img_h, img_w),
                "levels": search_spaces,
                "total_weighted_space": total_space,
                "original_space": original_space,
                "reduction": original_space / total_space if total_space > 0 else 0
            })
        
        results[scale] = analysis
    
    return results

# 分析範例
image_sizes = [(512, 512), (1024, 1024), (2048, 2048)]
template_size = (64, 64)
scales = [1.1, 1.25, 1.5, 2.0]

pyramid_analysis = analyze_pyramid_scaling_effects(image_sizes, template_size, scales)

print("金字塔縮放分析:")
for scale, data in pyramid_analysis.items():
    avg_reduction = np.mean([level["reduction"] for level in data["pyramid_levels"]])
    print(f"縮放比例 {scale}: 平均效率提升 {avg_reduction:.1f}x")
```

### ROI 擷取演算法優化

```python
def adaptive_roi_extraction(img: np.ndarray, candidates: List[tuple], 
                           template_size: tuple, level: int) -> Dict:
    """自適應 ROI 擷取演算法"""
    
    tmpl_h, tmpl_w = template_size
    img_h, img_w = img.shape
    
    # 根據金字塔層級調整 ROI 大小
    base_margin = max(tmpl_h, tmpl_w) // 4
    level_factor = 1.5 ** level  # 上層需要更大的搜尋範圍
    adaptive_margin = int(base_margin * level_factor)
    
    roi_data = {
        "level": level,
        "adaptive_margin": adaptive_margin,
        "rois": [],
        "statistics": {"valid_rois": 0, "boundary_clips": 0}
    }
    
    for cx, cy in candidates:
        # 計算 ROI 邊界
        x1 = max(0, cx - adaptive_margin)
        y1 = max(0, cy - adaptive_margin)
        x2 = min(img_w, cx + adaptive_margin)
        y2 = min(img_h, cy + adaptive_margin)
        
        # 確保 ROI 足夠大
        roi_w, roi_h = x2 - x1, y2 - y1
        if roi_w >= tmpl_w and roi_h >= tmpl_h:
            roi = img[y1:y2, x1:x2]
            
            # 計算 ROI 品質指標
            roi_variance = np.var(roi)
            edge_strength = np.sum(cv2.Laplacian(roi, cv2.CV_64F) ** 2)
            
            roi_info = {
                "center": (cx, cy),
                "bounds": (x1, y1, x2, y2),
                "roi": roi,
                "quality_metrics": {
                    "variance": roi_variance,
                    "edge_strength": edge_strength
                },
                "clipped": (x1 == 0 or y1 == 0 or x2 == img_w or y2 == img_h)
            }
            
            roi_data["rois"].append(roi_info)
            roi_data["statistics"]["valid_rois"] += 1
            
            if roi_info["clipped"]:
                roi_data["statistics"]["boundary_clips"] += 1
    
    return roi_data
```

---

## 🎯 次像素估計深度分析

### 曲面擬合理論

#### 二維二次曲面模型

```python
def advanced_subpixel_estimation(response: np.ndarray, peak_x: int, peak_y: int, 
                                method: str = "2d_quadratic") -> Dict:
    """進階次像素估計，支援多種擬合方法"""
    
    h, w = response.shape
    
    if not (1 <= peak_x < w-1 and 1 <= peak_y < h-1):
        return {"dx": 0.0, "dy": 0.0, "method": method, "valid": False}
    
    # 擷取 3x3 鄰域
    neighborhood = response[peak_y-1:peak_y+2, peak_x-1:peak_x+2]
    
    if method == "1d_parabolic":
        # 原始一維二次擬合
        dx = _1d_parabolic_fit(neighborhood[1, :])  # X 方向
        dy = _1d_parabolic_fit(neighborhood[:, 1])  # Y 方向
        
    elif method == "2d_quadratic":
        # 二維二次曲面擬合
        dx, dy = _2d_quadratic_fit(neighborhood)
        
    elif method == "centroid":
        # 重心法
        dx, dy = _centroid_method(neighborhood)
        
    else:
        raise ValueError(f"未知的次像素方法: {method}")
    
    return {
        "dx": float(dx), 
        "dy": float(dy), 
        "method": method, 
        "valid": True,
        "neighborhood": neighborhood
    }

def _2d_quadratic_fit(patch: np.ndarray) -> tuple:
    """二維二次曲面擬合
    
    模型: f(x,y) = ax² + by² + cxy + dx + ey + f
    """
    
    # 建立座標網格 (-1, 0, 1)
    y_coords, x_coords = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing='ij')
    
    # 建立係數矩陣
    A = np.column_stack([
        x_coords.flatten() ** 2,    # x²
        y_coords.flatten() ** 2,    # y²  
        x_coords.flatten() * y_coords.flatten(),  # xy
        x_coords.flatten(),         # x
        y_coords.flatten(),         # y
        np.ones(9)                  # 常數項
    ])
    
    # 觀測值
    b = patch.flatten()
    
    # 最小平方法求解
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        a, b_coef, c, d, e, f = coeffs
        
        # 計算極值點（偏微分 = 0）
        # ∂f/∂x = 2ax + cy + d = 0
        # ∂f/∂y = 2by + cx + e = 0
        
        det = 4 * a * b_coef - c ** 2
        if abs(det) > 1e-10:
            dx = (c * e - 2 * b_coef * d) / det
            dy = (c * d - 2 * a * e) / det
            
            # 限制偏移量
            dx = np.clip(dx, -1.5, 1.5)
            dy = np.clip(dy, -1.5, 1.5)
        else:
            dx, dy = 0.0, 0.0
            
    except np.linalg.LinAlgError:
        dx, dy = 0.0, 0.0
    
    return dx, dy

def _centroid_method(patch: np.ndarray) -> tuple:
    """重心法次像素估計"""
    
    # 只考慮正值
    positive_patch = np.maximum(patch - np.min(patch), 0)
    
    if np.sum(positive_patch) == 0:
        return 0.0, 0.0
    
    # 計算重心
    y_coords, x_coords = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing='ij')
    
    total_weight = np.sum(positive_patch)
    centroid_x = np.sum(x_coords * positive_patch) / total_weight
    centroid_y = np.sum(y_coords * positive_patch) / total_weight
    
    return float(centroid_x), float(centroid_y)

# 次像素方法比較
def compare_subpixel_methods(response: np.ndarray, true_peak: tuple) -> Dict:
    """比較不同次像素方法的精度"""
    
    true_x, true_y = true_peak
    peak_x, peak_y = int(true_x), int(true_y)
    true_dx, true_dy = true_x - peak_x, true_y - peak_y
    
    methods = ["1d_parabolic", "2d_quadratic", "centroid"]
    results = {}
    
    for method in methods:
        estimation = advanced_subpixel_estimation(response, peak_x, peak_y, method)
        
        if estimation["valid"]:
            error_x = abs(estimation["dx"] - true_dx)
            error_y = abs(estimation["dy"] - true_dy)
            error_magnitude = np.sqrt(error_x ** 2 + error_y ** 2)
            
            results[method] = {
                "estimated": (estimation["dx"], estimation["dy"]),
                "true": (true_dx, true_dy),
                "error": (error_x, error_y),
                "magnitude_error": error_magnitude
            }
    
    return results
```

---

## 🚀 效能分析與瓶頸識別

### 微觀效能分析

```python
import cProfile
import pstats
from functools import wraps

def profile_detailed(func):
    """詳細效能分析裝飾器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        
        print(f"\n=== {func.__name__} 詳細效能分析 ===")
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # 顯示前 20 個函數
        
        return result
    return wrapper

@profile_detailed
def profiled_pyramid_match(src, tmpl, **kwargs):
    return pyramid_match(src, tmpl, **kwargs)

# 記憶體分析
def memory_analysis(func, *args, **kwargs):
    """記憶體使用分析"""
    import tracemalloc
    import gc
    
    # 垃圾回收
    gc.collect()
    
    # 開始記憶體追蹤
    tracemalloc.start()
    
    # 記錄初始記憶體
    snapshot1 = tracemalloc.take_snapshot()
    
    # 執行函數
    result = func(*args, **kwargs)
    
    # 記錄結束記憶體
    snapshot2 = tracemalloc.take_snapshot()
    
    # 計算記憶體差異
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print(f"\n=== {func.__name__} 記憶體分析 ===")
    print("前 10 個記憶體增長最多的行:")
    for stat in top_stats[:10]:
        print(stat)
    
    tracemalloc.stop()
    return result

# 使用範例
result = memory_analysis(pyramid_match, src, tmpl, target_num=5)
```

### 演算法複雜度實測

```python
def complexity_analysis(algorithm_func, test_sizes: List[tuple], 
                       template_size: tuple = (64, 64)) -> Dict:
    """實測演算法複雜度"""
    
    import matplotlib.pyplot as plt
    
    results = {
        "sizes": [],
        "times": [],
        "memory_usage": [],
        "theoretical_complexity": []
    }
    
    for img_h, img_w in test_sizes:
        # 產生測試影像
        test_src = np.random.randint(0, 256, (img_h, img_w), dtype=np.uint8)
        test_tmpl = np.random.randint(0, 256, template_size, dtype=np.uint8)
        
        # 記錄記憶體使用前
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss
        
        # 時間測量
        start_time = time.perf_counter()
        _ = algorithm_func(test_src, test_tmpl, target_num=1, tolerance_angle=0)
        end_time = time.perf_counter()
        
        # 記錄記憶體使用後
        mem_after = process.memory_info().rss
        
        elapsed_time = end_time - start_time
        memory_used = mem_after - mem_before
        
        # 理論複雜度（金字塔演算法）
        total_pixels = img_h * img_w
        theoretical = total_pixels * np.log(max(img_h, img_w))
        
        results["sizes"].append((img_h, img_w))
        results["times"].append(elapsed_time)
        results["memory_usage"].append(memory_used)
        results["theoretical_complexity"].append(theoretical)
        
        print(f"尺寸 {img_h}x{img_w}: {elapsed_time:.3f}s, "
              f"記憶體 {memory_used/1024/1024:.1f}MB")
    
    # 擬合複雜度曲線
    pixel_counts = [h * w for h, w in results["sizes"]]
    
    # 線性擬合 log(time) vs log(pixels)
    log_pixels = np.log(pixel_counts)
    log_times = np.log(results["times"])
    
    coeffs = np.polyfit(log_pixels, log_times, 1)
    complexity_exponent = coeffs[0]
    
    results["fitted_complexity"] = complexity_exponent
    
    print(f"\n實測複雜度指數: O(n^{complexity_exponent:.2f})")
    print("理論預期: O(n log n) ≈ O(n^1.1-1.2)")
    
    return results

# 複雜度測試
test_sizes = [(256, 256), (512, 512), (768, 768), (1024, 1024), (1280, 1280)]
complexity_results = complexity_analysis(pyramid_match, test_sizes)
```

---

## 🔧 演算法改進與擴展

### 進階最佳化技術

#### 1. 多尺度金字塔

```python
def multi_scale_pyramid_match(src: np.ndarray, tmpl: np.ndarray, 
                             scale_range: tuple = (0.5, 2.0), 
                             scale_steps: int = 5, **kwargs) -> List[Dict]:
    """多尺度金字塔匹配，處理目標尺寸變化"""
    
    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    all_results = []
    
    for scale in scales:
        # 縮放樣板
        tmpl_h, tmpl_w = tmpl.shape
        new_h, new_w = int(tmpl_h * scale), int(tmpl_w * scale)
        
        if new_h < 8 or new_w < 8 or new_h > src.shape[0] or new_w > src.shape[1]:
            continue
        
        scaled_tmpl = cv2.resize(tmpl, (new_w, new_h))
        
        # 執行匹配
        results = pyramid_match(src, scaled_tmpl, **kwargs)
        
        # 記錄尺度資訊
        for result in results:
            result["template_scale"] = scale
            result["template_size"] = (new_w, new_h)
            all_results.append(result)
    
    # 跨尺度 NMS
    return cross_scale_nms(all_results, max_overlap=0.3)

def cross_scale_nms(results: List[Dict], max_overlap: float = 0.3) -> List[Dict]:
    """跨尺度非極大值抑制"""
    
    if not results:
        return []
    
    # 按分數排序
    results.sort(key=lambda r: r['score'], reverse=True)
    
    keep = []
    for result in results:
        x, y = result['pos_x'], result['pos_y']
        w, h = result['template_size']
        
        # 檢查與已保留結果的重疊
        overlapped = False
        for kept in keep:
            kx, ky = kept['pos_x'], kept['pos_y']
            kw, kh = kept['template_size']
            
            # 計算 IoU
            iou = calculate_iou_with_size((x, y, w, h), (kx, ky, kw, kh))
            
            if iou > max_overlap:
                overlapped = True
                break
        
        if not overlapped:
            keep.append(result)
    
    return keep

def calculate_iou_with_size(box1: tuple, box2: tuple) -> float:
    """計算考慮中心點和尺寸的 IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 轉換為邊界框格式
    left1, top1 = x1 - w1/2, y1 - h1/2
    right1, bottom1 = x1 + w1/2, y1 + h1/2
    
    left2, top2 = x2 - w2/2, y2 - h2/2
    right2, bottom2 = x2 + w2/2, y2 + h2/2
    
    # 計算交集
    inter_left = max(left1, left2)
    inter_top = max(top1, top2)
    inter_right = min(right1, right2)
    inter_bottom = min(bottom1, bottom2)
    
    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0
    
    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    
    # 計算聯集
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0
```

#### 2. 特徵增強預處理

```python
def feature_enhanced_preprocessing(img: np.ndarray, enhancement_type: str = "edges") -> np.ndarray:
    """特徵增強預處理"""
    
    if enhancement_type == "edges":
        # 邊緣增強
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        enhanced = cv2.convertScaleAbs(magnitude)
        
    elif enhancement_type == "texture":
        # 紋理增強（局部二值模式）
        enhanced = local_binary_pattern(img)
        
    elif enhancement_type == "contrast":
        # 對比度增強
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        
    else:
        enhanced = img
    
    return enhanced

def local_binary_pattern(img: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """局部二值模式特徵提取"""
    
    # 簡化版 LBP 實作
    h, w = img.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = img[i, j]
            binary_code = 0
            
            # 8 鄰域比較
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                x = int(i + radius * np.cos(angle))
                y = int(j + radius * np.sin(angle))
                
                if img[x, y] > center:
                    binary_code |= (1 << k)
            
            lbp[i, j] = binary_code
    
    return lbp
```

### 演算法創新擴展

#### 1. 自適應金字塔

```python
def adaptive_pyramid_match(src: np.ndarray, tmpl: np.ndarray, **kwargs) -> List[Dict]:
    """自適應金字塔匹配，根據影像內容調整策略"""
    
    # 分析影像特徵
    src_analysis = analyze_image_characteristics(src)
    tmpl_analysis = analyze_image_characteristics(tmpl)
    
    # 根據特徵調整參數
    adaptive_params = determine_optimal_params(src_analysis, tmpl_analysis)
    
    # 合併使用者參數
    final_params = {**adaptive_params, **kwargs}
    
    print(f"自適應參數: {adaptive_params}")
    
    return pyramid_match(src, tmpl, **final_params)

def analyze_image_characteristics(img: np.ndarray) -> Dict:
    """分析影像特徵"""
    
    # 計算統計特徵
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    
    # 邊緣密度
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
    
    # 紋理複雜度（局部變異）
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    texture_complexity = np.var(laplacian)
    
    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "edge_density": edge_density,
        "texture_complexity": texture_complexity,
        "size": img.shape
    }

def determine_optimal_params(src_analysis: Dict, tmpl_analysis: Dict) -> Dict:
    """根據影像分析結果確定最佳參數"""
    
    params = {}
    
    # 根據紋理複雜度選擇後端
    if src_analysis["texture_complexity"] > 1000:
        params["backend"] = "fft"  # 複雜紋理用 FFT
    else:
        params["backend"] = "opencv"  # 簡單紋理用 OpenCV
    
    # 根據邊緣密度調整搜尋策略
    if src_analysis["edge_density"] > 0.1:
        params["method"] = "ccoeff_normed"  # 豐富邊緣用零均值
    else:
        params["method"] = "ccorr_normed"   # 簡單結構用相關係數
    
    # 根據大小調整金字塔參數
    src_area = src_analysis["size"][0] * src_analysis["size"][1]
    if src_area > 1000000:  # 大圖像
        params["scale"] = 1.5
        params["min_reduced_area"] = 512
    else:
        params["scale"] = 1.25
        params["min_reduced_area"] = 256
    
    return params
```

---

這個深度技術解析涵蓋了演算法的數學原理、實作細節、效能分析和創新擴展。透過這些深入分析，你可以更好地理解演算法的工作原理，並根據具體需求進行優化和擴展。

**下一步：** [擴充與客製化 →](11-customization.md)