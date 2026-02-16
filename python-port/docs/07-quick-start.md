# 7. 快速上手指南

## 🚀 5 分鐘快速開始

### 步驟 1：安裝套件

```bash
cd python-port
pip install -e .[dev]
```

### 步驟 2：測試安裝

```bash
# 測試 CLI
fpm-py --help

# 測試 GUI  
fpm-py-gui

# 測試 API
python -c "from fpm_py import pyramid_match; print('安裝成功！')"
```

### 步驟 3：第一次執行

```bash
# 使用預設測試影像
fpm-py --src "../Test Images/Src7.bmp" --tmpl "../Test Images/Dst7.bmp" --target-num 3
```

---

## 💻 三種使用方式

```mermaid
flowchart LR
    U[使用者] --> API[API 程式介面]
    U --> CLI[命令列介面]
    U --> GUI[圖形介面]
    API --> CORE[pyramid_match()<br/>金字塔 + NCC]
    CLI --> CORE
    GUI --> CORE
    CORE --> OUT[排序結果/視覺化]
    style CORE fill:#e5f0ff
```

### 🎯 方式 1：API 程式介面（推薦）

```python
import cv2
from fpm_py import pyramid_match

# 載入影像
src = cv2.imread("source_image.jpg", cv2.IMREAD_GRAYSCALE)
tmpl = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)

# 執行比對
results = pyramid_match(
    src=src,
    tmpl=tmpl,
    target_num=5,          # 找出前 5 個最相似的位置
    score=0.7,            # 相似度門檻 
    tolerance_angle=30,   # 允許 ±30 度旋轉
    backend="fft"         # 使用 FFT 加速
)

# 查看結果
for i, result in enumerate(results):
    print(f"結果 {i+1}:")
    print(f"  位置: ({result['pos_x']:.1f}, {result['pos_y']:.1f})")
    print(f"  相似度: {result['score']:.3f}")
    print(f"  角度: {result['angle']:.1f}°")
```

### 🖥️ 方式 2：命令列介面

```bash
# 基礎使用
fpm-py --src image.jpg --tmpl template.jpg

# 進階參數
fpm-py \
  --src "large_image.jpg" \
  --tmpl "small_template.jpg" \
  --target-num 10 \
  --score 0.8 \
  --overlap 0.3 \
  --tolerance-angle 45 \
  --backend fft \
  --method ccoeff_normed
```

**參數說明：**
- `--target-num`: 最多找出幾個結果
- `--score`: 相似度門檻（0-1）
- `--overlap`: 結果重疊容忍度（0-1）
- `--tolerance-angle`: 旋轉角度容忍度（度）
- `--backend`: `opencv` 或 `fft`
- `--method`: `ccorr_normed` 或 `ccoeff_normed`

### 🖼️ 方式 3：圖形介面

```bash
fpm-py-gui
```

圖形介面功能：
- 📁 拖放載入影像
- ⚙️ 圖形化參數調整
- 📊 即時結果顯示
- 💾 結果匯出功能

---

## 📝 實用範例

### 範例 1：工業檢測應用

```python
import cv2
import numpy as np
from fpm_py import pyramid_match

def quality_inspection(product_image_path, standard_template_path):
    """產品品質檢測範例"""
    
    # 載入影像
    product = cv2.imread(product_image_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(standard_template_path, cv2.IMREAD_GRAYSCALE) 
    
    # 執行檢測
    results = pyramid_match(
        src=product,
        tmpl=template,
        target_num=1,           # 只需要最佳結果
        score=0.85,            # 高相似度要求
        tolerance_angle=5,     # 小角度容忍
        max_overlap=0.0,       # 不允許重疊
        backend="opencv"       # 穩定的後端
    )
    
    # 判定結果
    if results and results[0]['score'] >= 0.85:
        print(f"✅ 品質檢測通過！相似度: {results[0]['score']:.3f}")
        return True, results[0]
    else:
        print(f"❌ 品質檢測失敗！")
        return False, None

# 使用範例
passed, result = quality_inspection("product.jpg", "standard.jpg")
```

### 範例 2：多目標檢測

```python
def find_multiple_objects(scene_path, object_template_path):
    """在場景中尋找多個相同物件"""
    
    scene = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
    obj_tmpl = cv2.imread(object_template_path, cv2.IMREAD_GRAYSCALE)
    
    # 尋找多個目標
    results = pyramid_match(
        src=scene,
        tmpl=obj_tmpl,
        target_num=20,         # 最多 20 個結果
        score=0.6,            # 較寬鬆的門檻
        tolerance_angle=180,   # 任意角度
        max_overlap=0.3,      # 30% 重疊容忍
        backend="fft"         # 大範圍搜尋用 FFT
    )
    
    print(f"找到 {len(results)} 個目標物件：")
    for i, r in enumerate(results):
        print(f"  目標 {i+1}: 位置({r['pos_x']:.0f},{r['pos_y']:.0f}), "
              f"相似度 {r['score']:.3f}, 角度 {r['angle']:.1f}°")
    
    return results

# 使用範例
objects = find_multiple_objects("scene.jpg", "object_template.jpg")
```

### 範例 3：結果視覺化

```python
def visualize_results(src_path, tmpl_path, results):
    """將比對結果視覺化"""
    
    # 載入彩色影像用於顯示
    src_color = cv2.imread(src_path)
    tmpl_gray = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
    h, w = tmpl_gray.shape
    
    # 在每個結果位置畫框和標註
    for i, result in enumerate(results):
        x, y = int(result['pos_x'] - w//2), int(result['pos_y'] - h//2)
        
        # 畫矩形框
        cv2.rectangle(src_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 標註資訊
        label = f"#{i+1}: {result['score']:.3f}, {result['angle']:.1f}°"
        cv2.putText(src_color, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 0), 2)
    
    # 顯示結果
    cv2.imshow('Match Results', src_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 儲存結果
    cv2.imwrite('match_results.jpg', src_color)
    print("結果已儲存至 match_results.jpg")

# 整合使用
results = pyramid_match(src, tmpl, target_num=5, tolerance_angle=30)
visualize_results("source.jpg", "template.jpg", results)
```

---

## 🔧 參數調校指南

### 效能 vs 精度平衡

```python
# 🏃‍♂️ 追求速度（適合即時應用）
fast_config = {
    "target_num": 3,
    "score": 0.7,
    "tolerance_angle": 0,     # 不旋轉
    "coarse_step": 30,        # 粗糙角度步進
    "backend": "opencv"       # OpenCV 後端
}

# 🎯 追求精度（適合離線分析）
accurate_config = {
    "target_num": 10,
    "score": 0.5,
    "tolerance_angle": 180,   # 全角度搜尋
    "coarse_step": 10,        # 精細角度步進
    "fine_step": 2,           # 精細精修
    "backend": "fft"          # FFT 後端（大角度搜尋快）
}

# 🔄 平衡配置（推薦）
balanced_config = {
    "target_num": 5,
    "score": 0.6,
    "tolerance_angle": 45,    # 適中角度容忍
    "coarse_step": 15,
    "fine_step": 5,
    "backend": "fft"
}
```

### 不同場景的最佳配置

```python
# 🏭 工業檢測：高精度要求
industrial_config = {
    "score": 0.85,           # 嚴格相似度
    "tolerance_angle": 10,    # 小角度變化
    "max_overlap": 0.1,      # 低重疊容忍
    "backend": "opencv"      # 穩定可靠
}

# 🕵️ 監控應用：多目標檢測
surveillance_config = {
    "target_num": 20,        # 多個目標
    "score": 0.6,           # 適中門檻
    "tolerance_angle": 15,   # 適度角度容忍
    "max_overlap": 0.4,     # 允許部分重疊
    "backend": "fft"        # 大範圍搜尋
}

# 🎮 遊戲自動化：即時性要求
gaming_config = {
    "target_num": 1,         # 只要最佳結果
    "score": 0.75,          # 平衡精度與速度
    "tolerance_angle": 0,    # 無旋轉，最快
    "backend": "opencv"     # 最穩定
}
```

---

## ❗ 常見問題快速解決

### Q1: 找不到目標怎麼辦？

```python
# 降低相似度門檻
results = pyramid_match(src, tmpl, score=0.4)  # 從 0.5 降至 0.4

# 增加角度搜尋範圍
results = pyramid_match(src, tmpl, tolerance_angle=90)  # 增加至 ±90°

# 檢查影像品質
cv2.imshow('Source', src)
cv2.imshow('Template', tmpl)
cv2.waitKey(0)
```

### Q2: 速度太慢怎麼辦？

```python
# 使用快速設定
results = pyramid_match(
    src, tmpl, 
    target_num=1,           # 只要 1 個結果
    tolerance_angle=0,      # 不搜尋角度
    backend="opencv"        # 使用 OpenCV 後端
)

# 或者先縮小影像
small_src = cv2.resize(src, None, fx=0.5, fy=0.5)
small_tmpl = cv2.resize(tmpl, None, fx=0.5, fy=0.5)
```

### Q3: 記憶體使用過高？

```python
# 處理大型影像時
import gc

# 分批處理
for img_path in large_image_list:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    result = pyramid_match(img, template)
    # 處理結果...
    del img  # 明確釋放
    gc.collect()  # 強制垃圾回收
```

---

## 📊 效能基準測試

### 測試你的系統效能

```python
import time
import cv2
from fpm_py import pyramid_match

def benchmark_performance():
    """效能基準測試"""
    
    # 載入標準測試影像
    src = cv2.imread("../assets/Src7.bmp", cv2.IMREAD_GRAYSCALE)
    tmpl = cv2.imread("../assets/Dst7.bmp", cv2.IMREAD_GRAYSCALE)
    
    configs = [
        ("快速模式", {"tolerance_angle": 0, "backend": "opencv"}),
        ("平衡模式", {"tolerance_angle": 45, "backend": "fft"}),
        ("精確模式", {"tolerance_angle": 180, "backend": "fft"})
    ]
    
    for name, config in configs:
        start_time = time.time()
        results = pyramid_match(src, tmpl, target_num=3, **config)
        elapsed = time.time() - start_time
        
        print(f"{name}: {elapsed:.2f}s, 找到 {len(results)} 個結果")
        if results:
            print(f"  最佳相似度: {results[0]['score']:.3f}")

# 執行基準測試
benchmark_performance()
```

---

## 🎓 學習建議

### 初學者步驟
1. **從 API 開始** - 最直觀的使用方式
2. **使用測試影像** - assets 目錄有現成範例
3. **調整參數觀察** - 了解各參數的影響
4. **查看視覺化結果** - 理解演算法行為

### 進階使用者
1. **比較不同後端** - 了解 OpenCV vs FFT 差異
2. **效能分析** - 測試不同參數組合的速度
3. **客製化擴展** - 基於現有模組開發新功能
4. **貢獻改進** - 參與開源專案開發

---

**恭喜！你已經掌握了基本使用方法。**

**下一步建議：**
- 深入了解 **[演算法原理](03-algorithm-principles.md)**
- 學習進階的 **[效能優化技巧](06-performance-optimization.md)**  
- 查看完整的 **[API 使用範例](08-api-examples.md)**
