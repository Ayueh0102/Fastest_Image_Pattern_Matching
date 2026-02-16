# 11. 擴充與客製化

## 🎯 客製化開發指南

本章將說明如何基於現有的 Python 移植版本進行擴充和客製化，包括新功能開發、介面設計和效能優化。

---

## 🧩 模組擴展架構

### 建立自定義模組

#### 1. 創建新的匹配策略

```python
# custom_matchers.py
from typing import List, Dict, Tuple
import numpy as np
import cv2

from fpm_py.ncc import ncc_match
from fpm_py.pyramid import MatchResult

class CustomMatcher:
    """自定義匹配器基類"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.history = []
    
    def match(self, src: np.ndarray, tmpl: np.ndarray, **kwargs) -> List[Dict]:
        """子類必須實作的匹配方法"""
        raise NotImplementedError
    
    def get_statistics(self) -> Dict:
        """獲取匹配統計"""
        return {
            "total_matches": len(self.history),
            "avg_score": np.mean([h["best_score"] for h in self.history]) if self.history else 0,
            "success_rate": sum(1 for h in self.history if h["found"]) / len(self.history) if self.history else 0
        }

class EdgeBasedMatcher(CustomMatcher):
    """基於邊緣的匹配器"""
    
    def __init__(self):
        super().__init__("EdgeBased", "基於邊緣特徵的樣板匹配")
        self.edge_params = {
            "low_threshold": 50,
            "high_threshold": 150,
            "aperture_size": 3
        }
    
    def match(self, src: np.ndarray, tmpl: np.ndarray, **kwargs) -> List[Dict]:
        """基於邊緣的匹配實作"""
        
        # 邊緣檢測
        src_edges = cv2.Canny(src, 
                             self.edge_params["low_threshold"], 
                             self.edge_params["high_threshold"],
                             apertureSize=self.edge_params["aperture_size"])
        
        tmpl_edges = cv2.Canny(tmpl, 
                              self.edge_params["low_threshold"], 
                              self.edge_params["high_threshold"],
                              apertureSize=self.edge_params["aperture_size"])
        
        # 使用邊緣影像進行 NCC 匹配
        response = ncc_match(src_edges, tmpl_edges, method="ccoeff_normed")
        
        # 尋找峰值
        results = self._find_peaks(response, tmpl.shape, **kwargs)
        
        # 記錄歷史
        match_record = {
            "found": len(results) > 0,
            "num_results": len(results),
            "best_score": results[0]["score"] if results else 0,
            "method": "edge_based"
        }
        self.history.append(match_record)
        
        return results
    
    def _find_peaks(self, response: np.ndarray, tmpl_shape: tuple, **kwargs) -> List[Dict]:
        """從響應圖中尋找峰值"""
        
        target_num = kwargs.get("target_num", 1)
        score_thresh = kwargs.get("score", 0.5)
        
        results = []
        response_copy = response.copy()
        
        for i in range(target_num):
            _, max_val, _, max_loc = cv2.minMaxLoc(response_copy)
            
            if max_val < score_thresh:
                break
            
            x, y = max_loc
            tmpl_h, tmpl_w = tmpl_shape[:2]
            
            result = {
                "index": i,
                "score": float(max_val),
                "angle": 0.0,  # 邊緣匹配不支援角度
                "pos_x": float(x + tmpl_w / 2),
                "pos_y": float(y + tmpl_h / 2)
            }
            results.append(result)
            
            # 抑制鄰域
            suppress_size = min(tmpl_h, tmpl_w) // 2
            x1 = max(0, x - suppress_size)
            y1 = max(0, y - suppress_size)
            x2 = min(response_copy.shape[1], x + suppress_size)
            y2 = min(response_copy.shape[0], y + suppress_size)
            
            response_copy[y1:y2, x1:x2] = -1
        
        return results
    
    def configure_edges(self, low_threshold: int, high_threshold: int, aperture_size: int = 3):
        """配置邊緣檢測參數"""
        self.edge_params = {
            "low_threshold": low_threshold,
            "high_threshold": high_threshold, 
            "aperture_size": aperture_size
        }

# 使用範例
edge_matcher = EdgeBasedMatcher()
edge_matcher.configure_edges(low_threshold=30, high_threshold=100)

results = edge_matcher.match(src, tmpl, target_num=3, score=0.6)
stats = edge_matcher.get_statistics()
```

#### 2. 特徵描述子匹配器

```python
class FeatureBasedMatcher(CustomMatcher):
    """基於特徵描述子的匹配器"""
    
    def __init__(self, feature_type: str = "orb"):
        super().__init__(f"Feature_{feature_type}", f"基於 {feature_type} 特徵的匹配")
        self.feature_type = feature_type
        self.detector = self._create_detector()
        self.matcher = cv2.BFMatcher()
        
    def _create_detector(self):
        """創建特徵檢測器"""
        if self.feature_type == "orb":
            return cv2.ORB_create()
        elif self.feature_type == "sift":
            return cv2.SIFT_create()
        elif self.feature_type == "surf":
            return cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError(f"不支援的特徵類型: {self.feature_type}")
    
    def match(self, src: np.ndarray, tmpl: np.ndarray, **kwargs) -> List[Dict]:
        """基於特徵描述子的匹配"""
        
        # 檢測特徵點
        kp1, des1 = self.detector.detectAndCompute(tmpl, None)
        kp2, des2 = self.detector.detectAndCompute(src, None)
        
        if des1 is None or des2 is None:
            return []
        
        # 特徵匹配
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:  # 至少需要 4 個點進行 Homography
            return []
        
        # 計算 Homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return []
        
        # 計算樣板四個角在目標圖中的位置
        h, w = tmpl.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        # 計算中心點
        center = np.mean(transformed_corners.reshape(-1, 2), axis=0)
        
        # 計算匹配品質（基於內點數量）
        inliers = np.sum(mask)
        quality_score = inliers / len(good_matches)
        
        result = {
            "index": 0,
            "score": float(quality_score),
            "angle": self._calculate_rotation_angle(H),
            "pos_x": float(center[0]),
            "pos_y": float(center[1]),
            "homography": H.tolist(),
            "corners": transformed_corners.reshape(-1, 2).tolist(),
            "inliers": int(inliers),
            "total_matches": len(good_matches)
        }
        
        return [result]
    
    def _calculate_rotation_angle(self, H: np.ndarray) -> float:
        """從 Homography 矩陣計算旋轉角度"""
        # 簡化版本：假設主要是旋轉+平移變換
        angle_rad = np.arctan2(H[1, 0], H[0, 0])
        angle_deg = np.degrees(angle_rad)
        return float(angle_deg)

# 使用範例
feature_matcher = FeatureBasedMatcher("orb")
results = feature_matcher.match(src, tmpl)

if results:
    print(f"特徵匹配成功！內點數: {results[0]['inliers']}")
    print(f"旋轉角度: {results[0]['angle']:.1f}°")
```

---

## 🔧 自定義預處理器

### 影像增強預處理

```python
class ImagePreprocessor:
    """影像預處理管道"""
    
    def __init__(self):
        self.processors = []
        self.history = []
    
    def add_processor(self, processor_func, name: str, **params):
        """添加預處理步驟"""
        self.processors.append({
            "func": processor_func,
            "name": name,
            "params": params
        })
    
    def process(self, img: np.ndarray) -> np.ndarray:
        """執行預處理管道"""
        processed = img.copy()
        processing_log = []
        
        for processor in self.processors:
            start_time = time.perf_counter()
            
            try:
                processed = processor["func"](processed, **processor["params"])
                elapsed = time.perf_counter() - start_time
                
                processing_log.append({
                    "step": processor["name"],
                    "success": True,
                    "time": elapsed
                })
                
            except Exception as e:
                processing_log.append({
                    "step": processor["name"],
                    "success": False,
                    "error": str(e)
                })
                print(f"預處理步驟 {processor['name']} 失敗: {e}")
        
        self.history.append(processing_log)
        return processed
    
    def clear_processors(self):
        """清空預處理步驟"""
        self.processors.clear()

# 預處理函數庫
def adaptive_histogram_equalization(img: np.ndarray, clip_limit: float = 2.0, 
                                  tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """自適應直方圖均衡化"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def bilateral_filter_denoise(img: np.ndarray, d: int = 9, 
                           sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """雙邊濾波去噪"""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def unsharp_mask_sharpen(img: np.ndarray, kernel_size: int = 5, 
                        sigma: float = 1.0, amount: float = 1.0, threshold: int = 0) -> np.ndarray:
    """銳化濾鏡"""
    # 高斯模糊
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    # 計算遮罩
    mask = img.astype(np.float32) - blurred.astype(np.float32)
    
    # 應用遮罩
    sharpened = img.astype(np.float32) + amount * mask
    
    # 應用門檻
    if threshold > 0:
        low_contrast_mask = np.abs(mask) < threshold
        sharpened[low_contrast_mask] = img[low_contrast_mask]
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def morphological_processing(img: np.ndarray, operation: str = "opening", 
                           kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """形態學處理"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == "opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == "gradient":
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    else:
        raise ValueError(f"不支援的形態學操作: {operation}")

# 使用範例
preprocessor = ImagePreprocessor()

# 建立預處理管道
preprocessor.add_processor(bilateral_filter_denoise, "denoise", d=5, sigma_color=50, sigma_space=50)
preprocessor.add_processor(adaptive_histogram_equalization, "contrast_enhance", clip_limit=3.0)
preprocessor.add_processor(unsharp_mask_sharpen, "sharpen", amount=0.5)

# 處理影像
processed_src = preprocessor.process(src)
processed_tmpl = preprocessor.process(tmpl)

# 使用處理後的影像進行匹配
results = pyramid_match(processed_src, processed_tmpl)
```

---

## 🎨 自定義使用者介面

### 創建 Web 介面

```python
# web_interface.py
from flask import Flask, render_template, request, jsonify, send_file
import base64
import io
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

class WebMatchingInterface:
    """Web 介面匹配服務"""
    
    def __init__(self):
        self.matcher_registry = {}
        self.register_default_matchers()
    
    def register_matcher(self, name: str, matcher_class):
        """註冊匹配器"""
        self.matcher_registry[name] = matcher_class
    
    def register_default_matchers(self):
        """註冊預設匹配器"""
        from fpm_py import pyramid_match
        
        self.register_matcher("pyramid", lambda: pyramid_match)
        self.register_matcher("edge_based", EdgeBasedMatcher)
        self.register_matcher("feature_based", lambda: FeatureBasedMatcher("orb"))
    
    def process_matching_request(self, src_b64: str, tmpl_b64: str, 
                               matcher_type: str, parameters: dict) -> dict:
        """處理匹配請求"""
        
        try:
            # 解碼影像
            src_img = self._decode_base64_image(src_b64)
            tmpl_img = self._decode_base64_image(tmpl_b64)
            
            # 獲取匹配器
            if matcher_type not in self.matcher_registry:
                raise ValueError(f"不支援的匹配器: {matcher_type}")
            
            matcher = self.matcher_registry[matcher_type]()
            
            # 執行匹配
            if hasattr(matcher, 'match'):  # 自定義匹配器
                results = matcher.match(src_img, tmpl_img, **parameters)
            else:  # 函數匹配器
                results = matcher(src_img, tmpl_img, **parameters)
            
            # 生成視覺化結果
            vis_image = self._create_visualization(src_img, tmpl_img, results)
            vis_b64 = self._encode_image_to_base64(vis_image)
            
            return {
                "success": True,
                "results": results,
                "visualization": vis_b64,
                "num_matches": len(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    def _decode_base64_image(self, b64_string: str) -> np.ndarray:
        """解碼 base64 影像"""
        image_data = base64.b64decode(b64_string.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    def _encode_image_to_base64(self, img: np.ndarray) -> str:
        """編碼影像為 base64"""
        if len(img.shape) == 2:  # 灰階轉 RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(img_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def _create_visualization(self, src: np.ndarray, tmpl: np.ndarray, 
                            results: List[Dict]) -> np.ndarray:
        """創建視覺化結果"""
        
        if len(src.shape) == 2:
            vis = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        else:
            vis = src.copy()
        
        tmpl_h, tmpl_w = tmpl.shape[:2]
        
        for i, result in enumerate(results):
            x, y = int(result['pos_x']), int(result['pos_y'])
            
            # 畫邊界框
            x1 = x - tmpl_w // 2
            y1 = y - tmpl_h // 2
            x2 = x + tmpl_w // 2
            y2 = y + tmpl_h // 2
            
            color = (0, 255, 0) if i == 0 else (0, 255, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # 標註
            label = f"#{i+1}: {result['score']:.3f}"
            cv2.putText(vis, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis

# 創建 Web 介面實例
web_interface = WebMatchingInterface()

@app.route('/')
def index():
    """主頁面"""
    return render_template('matching_interface.html')

@app.route('/api/match', methods=['POST'])
def api_match():
    """匹配 API 端點"""
    data = request.json
    
    result = web_interface.process_matching_request(
        src_b64=data['source_image'],
        tmpl_b64=data['template_image'],
        matcher_type=data['matcher_type'],
        parameters=data['parameters']
    )
    
    return jsonify(result)

@app.route('/api/matchers')
def api_matchers():
    """獲取可用匹配器列表"""
    matchers = list(web_interface.matcher_registry.keys())
    return jsonify(matchers)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### HTML 模板

```html
<!-- templates/matching_interface.html -->
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>影像樣板比對 Web 介面</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 10px 0; }
        .upload-area:hover { border-color: #007bff; }
        .image-preview { max-width: 300px; max-height: 200px; margin: 10px; }
        .controls { background: #f8f9fa; padding: 20px; margin: 10px 0; }
        .results { margin: 20px 0; }
        .result-item { background: #e9ecef; padding: 10px; margin: 5px 0; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        button:disabled { background: #6c757d; cursor: not-allowed; }
        select, input { padding: 5px; margin: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 影像樣板比對 Web 介面</h1>
        
        <!-- 影像上傳區域 -->
        <div class="row">
            <div class="col">
                <h3>來源影像</h3>
                <div class="upload-area" id="sourceUploadArea">
                    <p>點擊或拖拽上傳來源影像</p>
                    <input type="file" id="sourceImageInput" accept="image/*" style="display: none;">
                    <img id="sourcePreview" class="image-preview" style="display: none;">
                </div>
            </div>
            <div class="col">
                <h3>樣板影像</h3>
                <div class="upload-area" id="templateUploadArea">
                    <p>點擊或拖拽上傳樣板影像</p>
                    <input type="file" id="templateImageInput" accept="image/*" style="display: none;">
                    <img id="templatePreview" class="image-preview" style="display: none;">
                </div>
            </div>
        </div>
        
        <!-- 控制參數 -->
        <div class="controls">
            <h3>匹配參數</h3>
            <label>匹配器類型:</label>
            <select id="matcherType">
                <option value="pyramid">金字塔匹配</option>
                <option value="edge_based">邊緣匹配</option>
                <option value="feature_based">特徵匹配</option>
            </select>
            
            <label>目標數量:</label>
            <input type="number" id="targetNum" value="3" min="1" max="20">
            
            <label>相似度門檻:</label>
            <input type="range" id="scoreThreshold" min="0" max="1" step="0.01" value="0.6">
            <span id="scoreValue">0.6</span>
            
            <label>角度容忍度:</label>
            <input type="range" id="toleranceAngle" min="0" max="180" step="5" value="0">
            <span id="angleValue">0°</span>
            
            <button id="matchButton" disabled>開始匹配</button>
        </div>
        
        <!-- 結果顯示 -->
        <div class="results">
            <h3>匹配結果</h3>
            <div id="resultsContainer">
                <p>請上傳影像並調整參數後開始匹配</p>
            </div>
            <div id="visualizationContainer">
                <!-- 視覺化結果將顯示在這裡 -->
            </div>
        </div>
    </div>

    <script>
        // JavaScript 程式碼
        let sourceImage = null;
        let templateImage = null;
        
        // 初始化事件監聽器
        document.addEventListener('DOMContentLoaded', function() {
            initializeUploadAreas();
            initializeControls();
        });
        
        function initializeUploadAreas() {
            // 來源影像上傳
            const sourceArea = document.getElementById('sourceUploadArea');
            const sourceInput = document.getElementById('sourceImageInput');
            const sourcePreview = document.getElementById('sourcePreview');
            
            sourceArea.addEventListener('click', () => sourceInput.click());
            sourceInput.addEventListener('change', (e) => handleImageUpload(e, 'source'));
            
            // 樣板影像上傳
            const templateArea = document.getElementById('templateUploadArea');
            const templateInput = document.getElementById('templateImageInput');
            const templatePreview = document.getElementById('templatePreview');
            
            templateArea.addEventListener('click', () => templateInput.click());
            templateInput.addEventListener('change', (e) => handleImageUpload(e, 'template'));
        }
        
        function initializeControls() {
            // 滑桿即時更新
            document.getElementById('scoreThreshold').addEventListener('input', function() {
                document.getElementById('scoreValue').textContent = this.value;
            });
            
            document.getElementById('toleranceAngle').addEventListener('input', function() {
                document.getElementById('angleValue').textContent = this.value + '°';
            });
            
            // 匹配按鈕
            document.getElementById('matchButton').addEventListener('click', performMatching);
        }
        
        function handleImageUpload(event, type) {
            const file = event.target.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const preview = document.getElementById(type + 'Preview');
                preview.src = e.target.result;
                preview.style.display = 'block';
                
                if (type === 'source') {
                    sourceImage = e.target.result;
                } else {
                    templateImage = e.target.result;
                }
                
                updateMatchButton();
            };
            reader.readAsDataURL(file);
        }
        
        function updateMatchButton() {
            const button = document.getElementById('matchButton');
            button.disabled = !(sourceImage && templateImage);
        }
        
        async function performMatching() {
            const button = document.getElementById('matchButton');
            const originalText = button.textContent;
            button.textContent = '處理中...';
            button.disabled = true;
            
            try {
                const parameters = {
                    target_num: parseInt(document.getElementById('targetNum').value),
                    score: parseFloat(document.getElementById('scoreThreshold').value),
                    tolerance_angle: parseFloat(document.getElementById('toleranceAngle').value)
                };
                
                const response = await fetch('/api/match', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        source_image: sourceImage,
                        template_image: templateImage,
                        matcher_type: document.getElementById('matcherType').value,
                        parameters: parameters
                    })
                });
                
                const result = await response.json();
                displayResults(result);
                
            } catch (error) {
                console.error('匹配失敗:', error);
                document.getElementById('resultsContainer').innerHTML = 
                    `<p style="color: red;">匹配失敗: ${error.message}</p>`;
            } finally {
                button.textContent = originalText;
                button.disabled = false;
            }
        }
        
        function displayResults(result) {
            const container = document.getElementById('resultsContainer');
            const visualizationContainer = document.getElementById('visualizationContainer');
            
            if (result.success) {
                // 顯示結果統計
                let html = `<h4>找到 ${result.num_matches} 個匹配</h4>`;
                
                // 顯示每個結果
                result.results.forEach((match, index) => {
                    html += `
                        <div class="result-item">
                            <strong>結果 ${index + 1}:</strong>
                            位置 (${match.pos_x.toFixed(1)}, ${match.pos_y.toFixed(1)}),
                            相似度 ${match.score.toFixed(3)},
                            角度 ${match.angle.toFixed(1)}°
                        </div>
                    `;
                });
                
                container.innerHTML = html;
                
                // 顯示視覺化結果
                if (result.visualization) {
                    visualizationContainer.innerHTML = 
                        `<h4>視覺化結果</h4><img src="${result.visualization}" style="max-width: 100%; height: auto;">`;
                }
                
            } else {
                container.innerHTML = `<p style="color: red;">匹配失敗: ${result.error}</p>`;
                visualizationContainer.innerHTML = '';
            }
        }
    </script>
</body>
</html>
```

---

## 📊 效能監控與分析工具

### 建立效能監控儀表板

```python
# performance_dashboard.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
from typing import List, Dict
import numpy as np

class PerformanceDashboard:
    """效能監控儀表板"""
    
    def __init__(self, log_file: str = "performance_log.json"):
        self.log_file = log_file
        self.metrics_history = self.load_history()
    
    def load_history(self) -> List[Dict]:
        """載入歷史記錄"""
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_history(self):
        """儲存歷史記錄"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics_history, f, default=str, indent=2)
    
    def log_performance(self, operation_name: str, execution_time: float, 
                       image_size: tuple, template_size: tuple, **kwargs):
        """記錄效能數據"""
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation_name,
            "execution_time": execution_time,
            "image_size": image_size,
            "template_size": template_size,
            "image_area": image_size[0] * image_size[1],
            "template_area": template_size[0] * template_size[1],
            **kwargs
        }
        
        self.metrics_history.append(metrics)
        self.save_history()
    
    def create_performance_report(self, days: int = 7) -> Dict:
        """創建效能報告"""
        
        # 過濾最近 N 天的數據
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m["timestamp"]) >= cutoff_date
        ]
        
        if not recent_metrics:
            return {"error": "無近期數據"}
        
        # 統計分析
        execution_times = [m["execution_time"] for m in recent_metrics]
        image_areas = [m["image_area"] for m in recent_metrics]
        
        report = {
            "period": f"最近 {days} 天",
            "total_operations": len(recent_metrics),
            "avg_execution_time": np.mean(execution_times),
            "median_execution_time": np.median(execution_times),
            "min_execution_time": np.min(execution_times),
            "max_execution_time": np.max(execution_times),
            "std_execution_time": np.std(execution_times),
            "avg_image_area": np.mean(image_areas),
            "throughput_ops_per_hour": len(recent_metrics) / (days * 24),
        }
        
        # 操作類型分析
        operations = {}
        for metric in recent_metrics:
            op = metric["operation"]
            if op not in operations:
                operations[op] = []
            operations[op].append(metric["execution_time"])
        
        report["by_operation"] = {
            op: {
                "count": len(times),
                "avg_time": np.mean(times),
                "min_time": np.min(times),
                "max_time": np.max(times)
            }
            for op, times in operations.items()
        }
        
        return report
    
    def plot_performance_trends(self, save_path: str = None):
        """繪製效能趨勢圖"""
        
        if not self.metrics_history:
            print("無數據可繪圖")
            return
        
        # 準備數據
        timestamps = [datetime.fromisoformat(m["timestamp"]) for m in self.metrics_history]
        execution_times = [m["execution_time"] for m in self.metrics_history]
        image_areas = [m["image_area"] for m in self.metrics_history]
        
        # 創建子圖
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('效能監控儀表板', fontsize=16)
        
        # 1. 執行時間趨勢
        ax1.plot(timestamps, execution_times, 'b-', alpha=0.7)
        ax1.set_title('執行時間趨勢')
        ax1.set_ylabel('執行時間 (秒)')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # 2. 執行時間分佈
        ax2.hist(execution_times, bins=20, alpha=0.7, color='green')
        ax2.set_title('執行時間分佈')
        ax2.set_xlabel('執行時間 (秒)')
        ax2.set_ylabel('頻次')
        ax2.grid(True, alpha=0.3)
        
        # 3. 影像面積 vs 執行時間
        ax3.scatter(image_areas, execution_times, alpha=0.6, color='red')
        ax3.set_title('影像面積 vs 執行時間')
        ax3.set_xlabel('影像面積 (像素)')
        ax3.set_ylabel('執行時間 (秒)')
        ax3.grid(True, alpha=0.3)
        
        # 擴合趨勢線
        if len(image_areas) > 1:
            z = np.polyfit(image_areas, execution_times, 1)
            p = np.poly1d(z)
            ax3.plot(sorted(image_areas), p(sorted(image_areas)), "r--", alpha=0.8)
        
        # 4. 每小時吞吐量
        hourly_counts = self._calculate_hourly_throughput()
        if hourly_counts:
            hours, counts = zip(*hourly_counts)
            ax4.bar(hours, counts, alpha=0.7, color='orange')
            ax4.set_title('每小時處理量')
            ax4.set_xlabel('小時')
            ax4.set_ylabel('處理次數')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已儲存至: {save_path}")
        
        plt.show()
    
    def _calculate_hourly_throughput(self) -> List[tuple]:
        """計算每小時處理量"""
        
        # 按小時分組
        hourly_counts = {}
        for metric in self.metrics_history:
            dt = datetime.fromisoformat(metric["timestamp"])
            hour = dt.hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        
        return sorted(hourly_counts.items())

# 使用範例：集成到現有程式碼中
dashboard = PerformanceDashboard()

def monitored_pyramid_match(src, tmpl, **kwargs):
    """帶監控的金字塔匹配"""
    
    start_time = time.perf_counter()
    results = pyramid_match(src, tmpl, **kwargs)
    execution_time = time.perf_counter() - start_time
    
    # 記錄效能數據
    dashboard.log_performance(
        operation_name="pyramid_match",
        execution_time=execution_time,
        image_size=src.shape,
        template_size=tmpl.shape,
        num_results=len(results),
        best_score=results[0]["score"] if results else 0,
        **kwargs
    )
    
    return results

# 生成報告和圖表
report = dashboard.create_performance_report(days=30)
print("效能報告:")
for key, value in report.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for subkey, subvalue in value.items():
            print(f"  {subkey}: {subvalue}")
    else:
        print(f"{key}: {value}")

dashboard.plot_performance_trends("performance_trends.png")
```

---

這個擴充與客製化指南提供了完整的框架，讓你可以根據具體需求開發新功能、建立自定義介面和效能監控工具。你可以基於這些範例進一步開發符合特定應用場景的解決方案！

**恭喜！你已經完成了整個教學文檔系統的學習！** 🎉
