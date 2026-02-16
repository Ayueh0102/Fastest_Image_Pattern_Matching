# fpm-py（Python 版影像樣板比對）

本專案旨在以 Python 重現原始 C++/MFC 專案之核心原理與優化流程（正規化互相關 NCC、影像金字塔、角度搜尋、去重疊/NMS 等），在無法直接移植 SIMD 的前提下，採用通用、可跨平台的向量化（NumPy）、OpenCV API 與演算法設計，兼顧精度與效能。

## 安裝與環境
- Python 3.9+
- 依賴：`numpy`、`opencv-python`、`tqdm`
- 安裝（開發模式）：
  ```bash
  cd python-port
  pip install -e .[dev]
  ```

## 基本使用（CLI）
```bash
fpm-py --src "../Test Images/Src7.bmp" \
       --tmpl "../Test Images/Dst7.bmp" \
       --target-num 5 --score 0.8 --overlap 0.5 \
       --tolerance-angle 180 --coarse-step 15 --fine-step 5 \
       --backend fft  # 擴充四：使用 FFT NCC
```
輸出包含：Index、Score、Angle、PosX、PosY。

## 圖形介面（GUI, Tk）
安裝完成後可啟動圖形介面進行互動測試：
```bash
fpm-py-gui
# 或者
python -m fpm_py.gui
```
啟動後會自動載入 `python-port/assets/` 內的範例影像（`Src7.bmp`、`Dst7.bmp`），方便快速體驗。

### 介面亮點
- 左側控制面板可設定匹配參數、旋轉搜尋範圍，以及顯示選項（切換框線/標籤/樣板疊圖、自動聚焦）。預設分數門檻為 0.75，可依影像情況調整以降低偽陽性。
- 旋轉容忍預設為 ±180°，若目標無旋轉可降低以縮短運算時間。
- 右側以分割面板呈現：上方列表整理每筆匹配的分數、角度與座標，可點選/雙擊聚焦；下方同步顯示樣板縮圖。
- 主畫布支援滑鼠滾輪縮放、拖曳平移，狀態列即時回報滑鼠所在像素座標。
- 選擇「將樣板疊加到選定結果」即可將彩色熱度圖覆蓋於選定框，快速對照樣板形狀。
- 匹配完成後可匯出 JSON 結果、檢視摘要對話框，流程與原始 C++ 版工具類似但更容易跨平台啟動。

## Python API 範例
```python
from fpm_py.pyramid import pyramid_match
import cv2

src = cv2.imread("../Test Images/Src7.bmp", cv2.IMREAD_GRAYSCALE)
tmpl = cv2.imread("../Test Images/Dst7.bmp", cv2.IMREAD_GRAYSCALE)
results = pyramid_match(src, tmpl, target_num=5, tolerance_angle=180, backend="fft")
```

## 專案目標與分階段重構
- 最小功能：單尺度 NCC、單角度（0°）。
- 擴充一：影像金字塔 + 區域化搜尋，降低搜尋空間。
- 擴充二：角度粗到細搜尋（粗步進→細精修）。
- 擴充三：NMS 去重疊、結果排序與報表化。
- 擴充四：可選加速（Numba/FFT NCC），維持與原理一致。

更多細節請見 `docs/`（繁體中文，含 mermaid 圖）。

## 擴充四（加速）
- 新增函式：`fpm_py.ncc.ncc_match_fft(src, tmpl, method="ccoeff_normed")`，回傳與 `matchTemplate` 相同大小之分數圖。
- 在高階流程可指定 `backend="fft"` 使用 FFT NCC；預設 `backend="opencv"`。
