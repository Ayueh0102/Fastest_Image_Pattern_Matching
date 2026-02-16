# 8. API 使用範例

## 🎯 核心 API 概覽

Python 移植版提供了簡潔而強大的 API 介面，適合各種應用場景。

```python
from fpm_py import pyramid_match
import cv2

# 基本使用
results = pyramid_match(source_image, template_image)
```

---

## 📖 完整 API 參考

### `pyramid_match()` 函數簽名

```python
def pyramid_match(
    src: np.ndarray,                    # 來源影像（灰階或彩色）
    tmpl: np.ndarray,                   # 樣板影像（灰階或彩色）
    target_num: int = 1,                # 目標數量
    score: float = 0.5,                 # 相似度門檻
    max_overlap: float = 0.5,           # 最大重疊度
    tolerance_angle: float = 0.0,       # 角度容忍度（度）
    coarse_step: float = 15.0,          # 角度粗搜尋步進
    fine_step: float = 5.0,             # 角度細搜尋步進
    scale: float = 1.25,                # 金字塔縮放比例
    min_reduced_area: int = 256,        # 最小搜尋面積
    topk_per_level: int = 50,           # 每層候選數量
    method: str = "ccorr_normed",       # NCC 方法
    backend: str = "opencv"             # 計算後端
) -> List[Dict[str, Union[int, float]]]  # 回傳結果列表
```

### 回傳格式

```python
[
    {
        "index": 0,           # 結果索引
        "score": 0.95,        # 相似度分數
        "angle": 15.5,        # 旋轉角度
        "pos_x": 120.3,       # 中心 X 座標
        "pos_y": 85.7         # 中心 Y 座標
    },
    # ... 更多結果
]
```

---

## 🚀 基礎使用範例

### 1. 最簡單的使用

```python
import cv2
from fpm_py import pyramid_match

def basic_example():
    """基礎樣板比對範例"""
    
    # 載入影像
    src = cv2.imread("source.jpg", cv2.IMREAD_GRAYSCALE)
    tmpl = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 執行比對
    results = pyramid_match(src, tmpl)
    
    # 顯示結果
    if results:
        best = results[0]
        print(f"找到目標！")
        print(f"  位置: ({best['pos_x']:.1f}, {best['pos_y']:.1f})")
        print(f"  相似度: {best['score']:.3f}")
        print(f"  角度: {best['angle']:.1f}°")
    else:
        print("未找到匹配的目標")

# 執行範例
basic_example()
```

### 2. 多目標檢測

```python
def multi_target_example():
    """多目標檢測範例"""
    
    src = cv2.imread("scene.jpg", cv2.IMREAD_GRAYSCALE)
    tmpl = cv2.imread("object.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 尋找多個目標
    results = pyramid_match(
        src=src,
        tmpl=tmpl,
        target_num=10,         # 最多找 10 個目標
        score=0.6,            # 較寬鬆的相似度門檻
        max_overlap=0.3       # 允許 30% 重疊
    )
    
    print(f"找到 {len(results)} 個目標：")
    for i, result in enumerate(results):
        print(f"目標 {i+1}: "
              f"位置({result['pos_x']:.0f},{result['pos_y']:.0f}), "
              f"相似度 {result['score']:.3f}")
    
    return results

# 執行多目標檢測
targets = multi_target_example()
```

### 3. 角度搜尋範例

```python
def rotation_example():
    """旋轉目標檢測範例"""
    
    src = cv2.imread("rotated_scene.jpg", cv2.IMREAD_GRAYSCALE)
    tmpl = cv2.imread("upright_template.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 搜尋任意角度的目標
    results = pyramid_match(
        src=src,
        tmpl=tmpl,
        tolerance_angle=180,   # 搜尋 ±180° (全角度)
        coarse_step=30,        # 粗搜尋：每 30° 一次
        fine_step=10           # 精搜尋：每 10° 一次
    )
    
    if results:
        best = results[0]
        print(f"找到旋轉目標！")
        print(f"  檢測角度: {best['angle']:.1f}°")
        print(f"  相似度: {best['score']:.3f}")
        
        # 判斷旋轉程度
        angle = abs(best['angle'])
        if angle < 15:
            print("  → 幾乎無旋轉")
        elif angle < 45:
            print("  → 輕微旋轉")
        elif angle < 135:
            print("  → 明顯旋轉")
        else:
            print("  → 接近倒轉")
    
    return results

# 執行角度搜尋
rotation_results = rotation_example()
```

---

## 🏭 實際應用場景

### 1. 工業品質檢測

```python
class QualityInspector:
    """工業品質檢測器"""
    
    def __init__(self, standard_template_path: str):
        self.standard = cv2.imread(standard_template_path, cv2.IMREAD_GRAYSCALE)
        self.inspection_history = []
    
    def inspect_product(self, product_image_path: str) -> Dict:
        """檢測產品品質"""
        
        product = cv2.imread(product_image_path, cv2.IMREAD_GRAYSCALE)
        
        # 高精度匹配
        results = pyramid_match(
            src=product,
            tmpl=self.standard,
            target_num=1,          # 只要最佳結果
            score=0.8,            # 嚴格的品質要求
            tolerance_angle=10,    # 允許小幅旋轉
            backend="opencv"       # 使用穩定的後端
        )
        
        # 品質判定
        if results:
            score = results[0]['score']
            if score >= 0.95:
                quality = "A級 - 優秀"
            elif score >= 0.9:
                quality = "B級 - 良好"
            elif score >= 0.85:
                quality = "C級 - 合格"
            else:
                quality = "D級 - 不合格"
        else:
            quality = "F級 - 嚴重缺陷"
            score = 0.0
        
        # 記錄檢測結果
        inspection_result = {
            "product_path": product_image_path,
            "quality_grade": quality,
            "similarity_score": score,
            "pass": score >= 0.85,
            "timestamp": time.time()
        }
        
        self.inspection_history.append(inspection_result)
        return inspection_result
    
    def batch_inspect(self, product_dir: str) -> List[Dict]:
        """批量檢測"""
        
        import os
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        results = []
        for filename in os.listdir(product_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                filepath = os.path.join(product_dir, filename)
                result = self.inspect_product(filepath)
                results.append(result)
                
                # 實時反饋
                status = "✅通過" if result['pass'] else "❌不通過"
                print(f"{filename}: {result['quality_grade']} {status}")
        
        return results
    
    def generate_report(self) -> str:
        """生成檢測報告"""
        
        if not self.inspection_history:
            return "無檢測記錄"
        
        total = len(self.inspection_history)
        passed = sum(1 for r in self.inspection_history if r['pass'])
        pass_rate = passed / total * 100
        
        avg_score = sum(r['similarity_score'] for r in self.inspection_history) / total
        
        report = f"""
品質檢測報告
={'='*40}
總檢測數量: {total}
通過數量: {passed}
不合格數量: {total - passed}
合格率: {pass_rate:.1f}%
平均相似度: {avg_score:.3f}

品質分佈:
"""
        
        # 統計各等級數量
        grade_count = {}
        for result in self.inspection_history:
            grade = result['quality_grade'].split(' ')[0]
            grade_count[grade] = grade_count.get(grade, 0) + 1
        
        for grade, count in sorted(grade_count.items()):
            percentage = count / total * 100
            report += f"{grade}: {count} 個 ({percentage:.1f}%)\n"
        
        return report

# 使用範例
inspector = QualityInspector("standard_product.jpg")

# 單個產品檢測
result = inspector.inspect_product("product_001.jpg")
print(f"檢測結果: {result['quality_grade']}")

# 批量檢測
batch_results = inspector.batch_inspect("./products/")
print(inspector.generate_report())
```

### 2. 監控系統應用

```python
class SecurityMonitor:
    """安全監控系統"""
    
    def __init__(self):
        self.suspicious_templates = {}
        self.detection_log = []
    
    def add_suspicious_object(self, name: str, template_path: str):
        """添加可疑物件樣板"""
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        self.suspicious_templates[name] = template
        print(f"已添加可疑物件: {name}")
    
    def monitor_frame(self, frame: np.ndarray, timestamp: str = None) -> List[Dict]:
        """監控單一畫面"""
        
        if timestamp is None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        detections = []
        
        # 檢測每種可疑物件
        for obj_name, template in self.suspicious_templates.items():
            results = pyramid_match(
                src=frame,
                tmpl=template,
                target_num=5,          # 每種物件最多檢測 5 個
                score=0.6,            # 監控系統使用較寬鬆門檻
                tolerance_angle=45,   # 允許一定角度變化
                max_overlap=0.4       # 允許部分重疊
            )
            
            for result in results:
                detection = {
                    "object_type": obj_name,
                    "confidence": result['score'],
                    "position": (result['pos_x'], result['pos_y']),
                    "angle": result['angle'],
                    "timestamp": timestamp,
                    "alert_level": self._get_alert_level(result['score'])
                }
                detections.append(detection)
        
        # 記錄檢測結果
        if detections:
            self.detection_log.extend(detections)
            self._trigger_alerts(detections)
        
        return detections
    
    def _get_alert_level(self, confidence: float) -> str:
        """根據信心度確定警報等級"""
        if confidence >= 0.9:
            return "HIGH"
        elif confidence >= 0.75:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _trigger_alerts(self, detections: List[Dict]):
        """觸發警報"""
        high_alerts = [d for d in detections if d['alert_level'] == 'HIGH']
        
        if high_alerts:
            print(f"🚨 高度警報！檢測到 {len(high_alerts)} 個高信心度目標")
            for alert in high_alerts:
                print(f"   {alert['object_type']}: 信心度 {alert['confidence']:.3f}")
    
    def monitor_video(self, video_path: str, frame_interval: int = 30):
        """監控影片檔案"""
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每隔 N 幀檢測一次
            if frame_count % frame_interval == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                timestamp = f"Frame_{frame_count}"
                
                detections = self.monitor_frame(gray_frame, timestamp)
                
                if detections:
                    print(f"Frame {frame_count}: 檢測到 {len(detections)} 個目標")
            
            frame_count += 1
        
        cap.release()
        print(f"影片監控完成，共處理 {frame_count} 幀")
    
    def get_detection_summary(self) -> Dict:
        """獲取檢測摘要"""
        
        if not self.detection_log:
            return {"message": "無檢測記錄"}
        
        # 統計分析
        total_detections = len(self.detection_log)
        object_counts = {}
        alert_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for detection in self.detection_log:
            obj_type = detection['object_type']
            alert_level = detection['alert_level']
            
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
            alert_counts[alert_level] += 1
        
        return {
            "total_detections": total_detections,
            "object_breakdown": object_counts,
            "alert_breakdown": alert_counts,
            "recent_detections": self.detection_log[-10:]  # 最近 10 次
        }

# 使用範例
monitor = SecurityMonitor()

# 添加監控目標
monitor.add_suspicious_object("weapon", "weapon_template.jpg")
monitor.add_suspicious_object("prohibited_item", "prohibited_template.jpg")

# 監控單一影像
test_frame = cv2.imread("security_camera_frame.jpg", cv2.IMREAD_GRAYSCALE)
detections = monitor.monitor_frame(test_frame)

# 監控影片
monitor.monitor_video("security_footage.mp4", frame_interval=60)

# 獲取檢測摘要
summary = monitor.get_detection_summary()
print(f"總檢測次數: {summary['total_detections']}")
```

### 3. 遊戲自動化

```python
class GameAutomation:
    """遊戲自動化工具"""
    
    def __init__(self, game_window_name: str = None):
        self.game_window = game_window_name
        self.ui_templates = {}
        self.action_history = []
    
    def load_ui_template(self, element_name: str, template_path: str):
        """載入 UI 元素樣板"""
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        self.ui_templates[element_name] = template
        print(f"已載入 UI 元素: {element_name}")
    
    def capture_screen(self) -> np.ndarray:
        """擷取遊戲畫面"""
        # 這裡使用 PIL 作為範例，實際可能需要其他截圖工具
        from PIL import ImageGrab
        import numpy as np
        
        screenshot = ImageGrab.grab()
        screenshot_np = np.array(screenshot)
        
        # 轉為灰階
        gray_screenshot = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)
        return gray_screenshot
    
    def find_ui_element(self, element_name: str, confidence: float = 0.8) -> Dict:
        """尋找 UI 元素"""
        
        if element_name not in self.ui_templates:
            raise ValueError(f"未找到 UI 元素樣板: {element_name}")
        
        screenshot = self.capture_screen()
        template = self.ui_templates[element_name]
        
        results = pyramid_match(
            src=screenshot,
            tmpl=template,
            target_num=1,
            score=confidence,
            tolerance_angle=0,     # UI 元素通常不會旋轉
            backend="opencv"       # UI 檢測用較快的後端
        )
        
        if results:
            result = results[0]
            return {
                "found": True,
                "position": (int(result['pos_x']), int(result['pos_y'])),
                "confidence": result['score'],
                "element_name": element_name
            }
        else:
            return {
                "found": False,
                "element_name": element_name
            }
    
    def wait_for_element(self, element_name: str, timeout: float = 10.0, 
                        check_interval: float = 0.5) -> Dict:
        """等待 UI 元素出現"""
        
        import time
        start_time = time.time()
        
        print(f"等待 UI 元素: {element_name}")
        
        while time.time() - start_time < timeout:
            result = self.find_ui_element(element_name)
            if result['found']:
                print(f"✅ 找到 {element_name}")
                return result
            
            time.sleep(check_interval)
        
        print(f"⏰ 等待超時: {element_name}")
        return {"found": False, "element_name": element_name, "timeout": True}
    
    def click_element(self, element_name: str, offset: tuple = (0, 0)) -> bool:
        """點擊 UI 元素"""
        
        element = self.find_ui_element(element_name)
        
        if element['found']:
            x, y = element['position']
            x += offset[0]
            y += offset[1]
            
            # 這裡使用 pyautogui 作為範例
            try:
                import pyautogui
                pyautogui.click(x, y)
                
                action = {
                    "action": "click",
                    "element": element_name,
                    "position": (x, y),
                    "timestamp": time.time(),
                    "success": True
                }
                self.action_history.append(action)
                
                print(f"✅ 點擊 {element_name} at ({x}, {y})")
                return True
                
            except ImportError:
                print("需要安裝 pyautogui: pip install pyautogui")
                return False
        else:
            print(f"❌ 無法找到 {element_name}")
            return False
    
    def auto_play_sequence(self, sequence: List[Dict]) -> bool:
        """執行自動遊戲序列"""
        
        print("開始執行自動遊戲序列")
        
        for step_num, step in enumerate(sequence, 1):
            print(f"\n步驟 {step_num}: {step['description']}")
            
            action_type = step['type']
            
            if action_type == 'wait_for':
                element_name = step['element']
                result = self.wait_for_element(element_name, 
                                             timeout=step.get('timeout', 10))
                if not result['found']:
                    print(f"❌ 序列中斷：無法找到 {element_name}")
                    return False
            
            elif action_type == 'click':
                element_name = step['element']
                offset = step.get('offset', (0, 0))
                if not self.click_element(element_name, offset):
                    print(f"❌ 序列中斷：無法點擊 {element_name}")
                    return False
            
            elif action_type == 'wait':
                wait_time = step['duration']
                print(f"⏳ 等待 {wait_time} 秒")
                time.sleep(wait_time)
            
            # 步驟間隔
            if 'delay' in step:
                time.sleep(step['delay'])
        
        print("\n✅ 自動遊戲序列執行完成")
        return True

# 使用範例
game_bot = GameAutomation()

# 載入 UI 元素
game_bot.load_ui_template("start_button", "start_btn.png")
game_bot.load_ui_template("settings_icon", "settings.png")
game_bot.load_ui_template("confirm_dialog", "confirm.png")

# 定義自動遊戲序列
play_sequence = [
    {
        "type": "wait_for",
        "element": "start_button",
        "description": "等待開始按鈕出現",
        "timeout": 15
    },
    {
        "type": "click", 
        "element": "start_button",
        "description": "點擊開始遊戲",
        "delay": 2
    },
    {
        "type": "wait_for",
        "element": "settings_icon", 
        "description": "等待進入遊戲主畫面"
    },
    {
        "type": "wait",
        "duration": 3,
        "description": "等待載入完成"
    }
]

# 執行自動遊戲
success = game_bot.auto_play_sequence(play_sequence)
```

---

## 🔧 進階 API 技巧

### 1. 自定義結果處理

```python
def custom_result_processor(results: List[Dict], src_shape: tuple, tmpl_shape: tuple) -> List[Dict]:
    """自定義結果處理器"""
    
    processed_results = []
    
    for result in results:
        # 計算邊界框
        center_x, center_y = result['pos_x'], result['pos_y']
        tmpl_h, tmpl_w = tmpl_shape[:2]
        
        bbox = {
            "x1": int(center_x - tmpl_w / 2),
            "y1": int(center_y - tmpl_h / 2),
            "x2": int(center_x + tmpl_w / 2),
            "y2": int(center_y + tmpl_h / 2)
        }
        
        # 添加額外資訊
        enhanced_result = {
            **result,
            "bbox": bbox,
            "center": (center_x, center_y),
            "template_size": (tmpl_w, tmpl_h),
            "confidence_level": "high" if result['score'] > 0.8 else "medium" if result['score'] > 0.6 else "low"
        }
        
        processed_results.append(enhanced_result)
    
    return processed_results

# 使用範例
raw_results = pyramid_match(src, tmpl, target_num=5)
enhanced_results = custom_result_processor(raw_results, src.shape, tmpl.shape)

for result in enhanced_results:
    print(f"位置: {result['center']}")
    print(f"邊界框: {result['bbox']}")
    print(f"信心級別: {result['confidence_level']}")
```

### 2. 結果可視化工具

```python
def visualize_results_advanced(src: np.ndarray, tmpl: np.ndarray, 
                              results: List[Dict], save_path: str = None) -> np.ndarray:
    """進階結果可視化"""
    
    # 轉為彩色影像
    if len(src.shape) == 2:
        vis_img = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = src.copy()
    
    tmpl_h, tmpl_w = tmpl.shape[:2]
    
    # 顏色配置
    colors = [
        (0, 255, 0),    # 綠色 - 最佳結果
        (0, 255, 255),  # 黃色 - 次佳結果
        (255, 0, 255),  # 紫色 - 其他結果
        (255, 0, 0),    # 藍色
        (0, 0, 255)     # 紅色
    ]
    
    for i, result in enumerate(results):
        color = colors[min(i, len(colors) - 1)]
        
        # 計算邊界框
        cx, cy = result['pos_x'], result['pos_y']
        x1 = int(cx - tmpl_w / 2)
        y1 = int(cy - tmpl_h / 2)
        x2 = int(cx + tmpl_w / 2)
        y2 = int(cy + tmpl_h / 2)
        
        # 繪製邊界框
        thickness = 3 if i == 0 else 2  # 最佳結果用較粗的線
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
        
        # 繪製中心點
        cv2.circle(vis_img, (int(cx), int(cy)), 5, color, -1)
        
        # 標註資訊
        label = f"#{i+1}: {result['score']:.3f}"
        if abs(result['angle']) > 1:
            label += f", {result['angle']:.1f}°"
        
        # 文字背景
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_img, (x1, y1-text_h-10), (x1+text_w+5, y1), color, -1)
        
        # 文字標註
        cv2.putText(vis_img, label, (x1+2, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 角度指示線（如果有旋轉）
        if abs(result['angle']) > 5:
            angle_rad = np.radians(result['angle'])
            line_length = min(tmpl_w, tmpl_h) / 2
            end_x = int(cx + line_length * np.cos(angle_rad))
            end_y = int(cy + line_length * np.sin(angle_rad))
            cv2.line(vis_img, (int(cx), int(cy)), (end_x, end_y), color, 2)
    
    # 添加統計資訊
    stats_text = [
        f"總檢測數: {len(results)}",
        f"最佳分數: {results[0]['score']:.3f}" if results else "無檢測結果",
        f"樣板尺寸: {tmpl_w}x{tmpl_h}"
    ]
    
    for i, text in enumerate(stats_text):
        cv2.putText(vis_img, text, (10, 30 + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 儲存結果
    if save_path:
        cv2.imwrite(save_path, vis_img)
        print(f"視覺化結果已儲存: {save_path}")
    
    return vis_img

# 使用範例
results = pyramid_match(src, tmpl, target_num=5, tolerance_angle=45)
vis_result = visualize_results_advanced(src, tmpl, results, "detection_result.jpg")

# 顯示結果
cv2.imshow("Detection Results", vis_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 3. 批量 API 處理

```python
def batch_pyramid_match(src_list: List[np.ndarray], tmpl: np.ndarray, **kwargs) -> List[List[Dict]]:
    """批量樣板匹配"""
    
    results = []
    
    for i, src in enumerate(src_list):
        print(f"處理影像 {i+1}/{len(src_list)}")
        
        try:
            result = pyramid_match(src, tmpl, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"處理影像 {i+1} 時發生錯誤: {e}")
            results.append([])  # 空結果
    
    return results

def multi_template_match(src: np.ndarray, tmpl_list: List[np.ndarray], **kwargs) -> Dict[int, List[Dict]]:
    """多樣板匹配"""
    
    results = {}
    
    for i, tmpl in enumerate(tmpl_list):
        print(f"匹配樣板 {i+1}/{len(tmpl_list)}")
        
        try:
            result = pyramid_match(src, tmpl, **kwargs)
            results[i] = result
        except Exception as e:
            print(f"匹配樣板 {i+1} 時發生錯誤: {e}")
            results[i] = []
    
    return results

# 使用範例
# 批量處理多張影像
src_images = [cv2.imread(f"image_{i}.jpg", cv2.IMREAD_GRAYSCALE) for i in range(1, 6)]
template = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)

batch_results = batch_pyramid_match(src_images, template, target_num=3)

# 多樣板匹配
templates = [cv2.imread(f"template_{i}.jpg", cv2.IMREAD_GRAYSCALE) for i in range(1, 4)]
source = cv2.imread("complex_scene.jpg", cv2.IMREAD_GRAYSCALE)

multi_results = multi_template_match(source, templates, target_num=5)
```

---

這些 API 範例涵蓋了從基礎使用到複雜應用的各種場景。你可以根據具體需求選擇合適的範例作為起點，並進一步客製化開發！

**下一步：** [常見問題解答 →](09-faq.md)

或者深入學習：[深度技術解析 →](10-deep-dive.md)