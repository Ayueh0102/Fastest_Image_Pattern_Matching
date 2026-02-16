# 9. 常見問題解答

## 🎯 安裝與設定問題

### Q1: pip install 失敗怎麼辦？

**A1:** 常見解決方案：

```bash
# 1. 更新 pip
pip install --upgrade pip

# 2. 使用虛擬環境
python -m venv fpm_env
source fpm_env/bin/activate  # Linux/Mac
# 或 fpm_env\Scripts\activate  # Windows

# 3. 安裝依賴
pip install numpy opencv-python tqdm

# 4. 安裝本專案
cd python-port
pip install -e .[dev]
```

**常見錯誤與解決：**
- `No module named cv2`: 執行 `pip install opencv-python`
- `Microsoft Visual C++ 14.0 is required`: 安裝 Visual Studio Build Tools
- `Permission denied`: 使用 `pip install --user` 或虛擬環境

### Q2: GUI 無法啟動？

**A2:** 檢查 Tkinter 安裝：

```bash
# 測試 Tkinter
python -c "import tkinter; print('Tkinter 可用')"

# Ubuntu/Debian 系統可能需要：
sudo apt-get install python3-tk

# macOS 可能需要：
brew install python-tk
```

---

## 🔍 使用相關問題

### Q3: 為什麼找不到目標物件？

**A3:** 逐步除錯：

```python
import cv2
import numpy as np
from fpm_py import pyramid_match

# 1. 檢查影像是否正確載入
src = cv2.imread("your_image.jpg", cv2.IMREAD_GRAYSCALE)
tmpl = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)

if src is None:
    print("❌ 來源影像載入失敗")
if tmpl is None:
    print("❌ 樣板影像載入失敗")

# 2. 檢查影像尺寸
print(f"來源影像尺寸: {src.shape}")
print(f"樣板影像尺寸: {tmpl.shape}")

if src.shape[0] < tmpl.shape[0] or src.shape[1] < tmpl.shape[1]:
    print("❌ 樣板太大")

# 3. 視覺化檢查
cv2.imshow('Source', src)
cv2.imshow('Template', tmpl)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. 降低門檻重試
results = pyramid_match(src, tmpl, score=0.3, target_num=10)
print(f"找到 {len(results)} 個候選結果")
```

**常見原因：**
- 相似度門檻太高（試試 `score=0.3`）
- 角度變化太大（增加 `tolerance_angle`）
- 尺度變化（樣板與目標大小差異太大）
- 光照變化（考慮使用 `method="ccoeff_normed"`）

### Q4: 如何處理旋轉的目標？

**A4:** 角度搜尋配置：

```python
# 小角度變化（±15°）
results = pyramid_match(
    src, tmpl,
    tolerance_angle=15,
    coarse_step=5,     # 精細步進
    fine_step=1
)

# 任意角度（0-360°）
results = pyramid_match(
    src, tmpl,
    tolerance_angle=180,    # ±180° = 360°
    coarse_step=30,        # 粗糙步進加速
    fine_step=10
)

# 已知大致角度範圍
# 注意：目前版本僅支援對稱角度搜尋
# 如需 30°-60° 範圍，可用 tolerance_angle=45 (45±15°)
```

### Q5: 記憶體使用量太大？

**A5:** 記憶體最佳化策略：

```python
import gc

# 1. 分批處理大量影像
def process_image_batch(image_paths, template_path, batch_size=10):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        
        for path in batch:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            result = pyramid_match(img, template)
            results.extend(result)
            del img  # 立即釋放
        
        gc.collect()  # 強制垃圾回收
    
    return results

# 2. 降低影像解析度
def resize_for_memory(img, max_size=1024):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

# 3. 使用較小的參數
results = pyramid_match(
    src, tmpl,
    target_num=3,          # 減少候選數量
    topk_per_level=20      # 減少每層候選數
)
```

---

## ⚡ 效能問題

### Q6: 執行速度太慢？

**A6:** 效能最佳化：

```python
# 1. 選擇適當後端
# 小角度搜尋：使用 OpenCV
results = pyramid_match(src, tmpl, backend="opencv", tolerance_angle=0)

# 大角度搜尋：使用 FFT  
results = pyramid_match(src, tmpl, backend="fft", tolerance_angle=180)

# 2. 調整搜尋參數
results = pyramid_match(
    src, tmpl,
    target_num=1,          # 只要最佳結果
    coarse_step=30,        # 粗糙角度步進
    scale=1.5,            # 更大的金字塔縮放比例
    min_reduced_area=512  # 更大的最小搜尋面積
)

# 3. 預處理影像
def preprocess_for_speed(img):
    # 適度模糊減少雜訊
    return cv2.GaussianBlur(img, (3, 3), 0)

src_processed = preprocess_for_speed(src)
tmpl_processed = preprocess_for_speed(tmpl)
```

### Q7: 如何比較不同配置的效能？

**A7:** 效能測試工具：

```python
import time
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResult:
    config_name: str
    elapsed_time: float
    num_results: int
    best_score: float

def benchmark_configs(src, tmpl, configs):
    """比較多種配置的效能"""
    results = []
    
    for name, config in configs.items():
        print(f"測試 {name}...")
        
        start_time = time.perf_counter()
        matches = pyramid_match(src, tmpl, **config)
        elapsed = time.perf_counter() - start_time
        
        best_score = matches[0]['score'] if matches else 0.0
        
        result = BenchmarkResult(
            config_name=name,
            elapsed_time=elapsed,
            num_results=len(matches),
            best_score=best_score
        )
        results.append(result)
        
        print(f"  時間: {elapsed:.3f}s, 結果: {len(matches)} 個, 最佳分數: {best_score:.3f}")
    
    return results

# 使用範例
test_configs = {
    "快速": {"tolerance_angle": 0, "backend": "opencv"},
    "平衡": {"tolerance_angle": 45, "backend": "fft"},
    "精確": {"tolerance_angle": 180, "backend": "fft", "coarse_step": 10}
}

benchmark_results = benchmark_configs(src, tmpl, test_configs)
```

---

## 🔧 技術問題

### Q8: OpenCV 和 FFT 後端有什麼差別？

**A8:** 後端選擇指南：

| 特性 | OpenCV 後端 | FFT 後端 |
|------|------------|---------|
| **穩定性** | 非常穩定 | 良好 |
| **小角度搜尋** | 更快 | 較慢 |
| **大角度搜尋** | 較慢 | 更快 |
| **記憶體使用** | 較低 | 較高 |
| **精確度** | 標準 | 略微不同（通常可接受） |

```python
# 選擇建議
if tolerance_angle <= 15:
    backend = "opencv"    # 小角度用 OpenCV
elif tolerance_angle >= 45:
    backend = "fft"      # 大角度用 FFT
else:
    backend = "opencv"   # 預設用 OpenCV
```

### Q9: 相似度分數怎麼解讀？

**A9:** 分數解讀指南：

```python
def interpret_score(score, method="ccorr_normed"):
    """解讀相似度分數"""
    if method == "ccorr_normed":
        # 範圍 [0, 1]，1 表示完全相同
        if score >= 0.95:
            return "幾乎完美匹配"
        elif score >= 0.8:
            return "非常好的匹配"
        elif score >= 0.6:
            return "良好匹配"
        elif score >= 0.4:
            return "可能匹配"
        else:
            return "差匹配"
    
    elif method == "ccoeff_normed":
        # 範圍 [-1, 1]，1 表示完美正相關
        if score >= 0.8:
            return "優秀匹配"
        elif score >= 0.5:
            return "良好匹配"
        elif score >= 0.2:
            return "可能匹配"
        else:
            return "差匹配"

# 使用範例
for result in results:
    interpretation = interpret_score(result['score'])
    print(f"位置 ({result['pos_x']:.0f}, {result['pos_y']:.0f}): "
          f"{result['score']:.3f} - {interpretation}")
```

### Q10: 次像素精度是什麼意思？

**A10:** 次像素精度說明：

```python
# 一般整數像素定位
_, _, _, max_loc = cv2.minMaxLoc(response)
x, y = max_loc  # 例如：(120, 85)

# 次像素精度定位  
from fpm_py.subpixel import refine_peak_subpixel
dx, dy = refine_peak_subpixel(response, x, y)
precise_x = x + dx  # 例如：120.3
precise_y = y + dy  # 例如：85.7

print(f"整數像素: ({x}, {y})")
print(f"次像素精度: ({precise_x:.3f}, {precise_y:.3f})")
print(f"精度提升: {abs(dx) + abs(dy):.3f} 像素")
```

**意義：**
- 提供小於 1 像素的定位精度
- 特別適合高精度測量應用
- 基於 3x3 鄰域的二次曲線擬合
- 通常可提升 0.1-0.5 像素的精度

---

## 🐛 除錯技巧

### Q11: 如何除錯比對失敗？

**A11:** 系統化除錯流程：

```python
def debug_matching(src_path, tmpl_path):
    """完整的除錯流程"""
    
    # 1. 載入檢查
    src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    tmpl = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
    
    if src is None:
        print("❌ 來源影像載入失敗")
        return
    if tmpl is None:
        print("❌ 樣板影像載入失敗")
        return
    
    print(f"✅ 影像載入成功")
    print(f"   來源: {src.shape}")
    print(f"   樣板: {tmpl.shape}")
    
    # 2. 尺寸檢查
    if src.shape[0] < tmpl.shape[0] or src.shape[1] < tmpl.shape[1]:
        print("❌ 樣板尺寸大於來源影像")
        return
    
    # 3. 基礎 NCC 測試
    from fpm_py.ncc import ncc_match
    basic_result = ncc_match(src, tmpl)
    max_score = np.max(basic_result)
    print(f"✅ 基礎 NCC 最高分數: {max_score:.3f}")
    
    if max_score < 0.3:
        print("⚠️  分數很低，可能影像內容差異很大")
    
    # 4. 逐步測試不同參數
    test_configs = [
        ("預設", {}),
        ("低門檻", {"score": 0.3}),
        ("大角度", {"tolerance_angle": 90}),
        ("FFT後端", {"backend": "fft"}),
    ]
    
    for name, config in test_configs:
        try:
            results = pyramid_match(src, tmpl, target_num=5, **config)
            print(f"✅ {name}: 找到 {len(results)} 個結果")
            if results:
                print(f"   最佳: 分數={results[0]['score']:.3f}, "
                      f"角度={results[0]['angle']:.1f}°")
        except Exception as e:
            print(f"❌ {name}: 錯誤 {e}")
    
    # 5. 視覺化建議
    print("\n🔍 建議進行視覺化檢查：")
    print("   cv2.imshow('Source', src)")
    print("   cv2.imshow('Template', tmpl)")

# 使用範例
debug_matching("problem_src.jpg", "problem_tmpl.jpg")
```

### Q12: 結果不穩定怎麼辦？

**A12:** 提升穩定性的方法：

```python
def stable_matching(src, tmpl, num_runs=5):
    """多次執行取穩定結果"""
    all_results = []
    
    for i in range(num_runs):
        results = pyramid_match(src, tmpl, target_num=10)
        all_results.extend(results)
    
    # 依位置聚類
    stable_results = []
    tolerance = 10  # 像素容忍度
    
    for result in all_results:
        x, y = result['pos_x'], result['pos_y']
        
        # 檢查是否與現有結果接近
        found_group = False
        for stable in stable_results:
            sx, sy = stable['pos_x'], stable['pos_y']
            if abs(x - sx) < tolerance and abs(y - sy) < tolerance:
                # 更新為更高分數的結果
                if result['score'] > stable['score']:
                    stable.update(result)
                found_group = True
                break
        
        if not found_group:
            stable_results.append(result.copy())
    
    # 按分數排序
    stable_results.sort(key=lambda r: r['score'], reverse=True)
    return stable_results

# 使用穩定匹配
results = stable_matching(src, tmpl)
```

---

## 🚀 進階應用

### Q13: 如何處理多尺度目標？

**A13:** 多尺度檢測策略：

```python
def multi_scale_detection(src, tmpl, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
    """多尺度目標檢測"""
    all_results = []
    
    for scale in scales:
        # 縮放樣板
        h, w = tmpl.shape
        scaled_h, scaled_w = int(h * scale), int(w * scale)
        if scaled_h < 5 or scaled_w < 5:  # 避免太小
            continue
            
        scaled_tmpl = cv2.resize(tmpl, (scaled_w, scaled_h))
        
        # 執行檢測
        results = pyramid_match(src, scaled_tmpl, target_num=5)
        
        # 記錄尺度資訊
        for result in results:
            result['scale'] = scale
            result['template_size'] = (scaled_w, scaled_h)
            all_results.append(result)
    
    # 合併相近結果
    return merge_multi_scale_results(all_results)

def merge_multi_scale_results(results, distance_threshold=20):
    """合併多尺度檢測結果"""
    if not results:
        return []
    
    # 按分數排序
    results.sort(key=lambda r: r['score'], reverse=True)
    
    merged = []
    for result in results:
        x, y = result['pos_x'], result['pos_y']
        
        # 檢查是否與已有結果太接近
        too_close = False
        for existing in merged:
            ex, ey = existing['pos_x'], existing['pos_y']
            distance = np.sqrt((x - ex)**2 + (y - ey)**2)
            if distance < distance_threshold:
                too_close = True
                break
        
        if not too_close:
            merged.append(result)
    
    return merged
```

### Q14: 如何批量處理影像？

**A14:** 批量處理範例：

```python
import os
from pathlib import Path
import json

def batch_process(image_dir, template_path, output_path):
    """批量處理目錄中的所有影像"""
    
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"無法載入樣板: {template_path}")
    
    # 支援的影像格式
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    results = {}
    
    for img_path in image_files:
        print(f"處理: {img_path.name}")
        
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  ⚠️  跳過（無法載入）")
                continue
            
            matches = pyramid_match(img, template, target_num=5)
            results[str(img_path)] = {
                'num_matches': len(matches),
                'matches': matches,
                'processed_time': time.time()
            }
            
            print(f"  ✅ 找到 {len(matches)} 個匹配")
            
        except Exception as e:
            print(f"  ❌ 錯誤: {e}")
            results[str(img_path)] = {'error': str(e)}
    
    # 儲存結果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n結果已儲存至: {output_path}")
    return results

# 使用範例
results = batch_process("./images/", "template.jpg", "results.json")
```

---

## 📞 獲得幫助

### 如何報告問題？

1. **提供完整資訊**：
   - Python 版本
   - 系統版本（Windows/Mac/Linux）
   - 錯誤訊息截圖
   - 最小重現範例

2. **檢查已知問題**：
   - 查看 GitHub Issues
   - 搜尋相關討論

3. **尋求協助**：
   - GitHub Issues: 技術問題
   - 討論區: 使用疑問
   - 文檔: 基礎學習

### 持續學習資源

- **[深度技術解析](10-deep-dive.md)** - 進階演算法分析
- **[擴充與客製化](11-customization.md)** - 客製化開發
- **原專案 README** - 了解 C++ 版本詳情
- **OpenCV 文檔** - 深入理解底層函數

---

**記住：遇到問題時，系統化除錯比盲目嘗試更有效！**