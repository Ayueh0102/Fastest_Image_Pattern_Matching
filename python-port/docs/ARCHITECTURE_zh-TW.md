# 架構與原理（Architecture）

本 Python 版本延續原專案之核心思想：以正規化互相關（NCC）為基礎，透過影像金字塔縮小搜尋空間，並以角度粗到細的策略達成旋轉不變性。最後以 NMS 去除重疊，輸出高可信度結果。

```mermaid
flowchart LR
  A[輸入影像 Src] --> B[建構金字塔(大→小)]
  T[樣板 Template] --> T1[角度集\ncoarse→fine]
  B --> C[最小層全域 NCC 搜尋]
  T1 --> C
  C --> D[擷取候選點 Top-K]
  D --> E[逐層放大\nROI 精修]
  E --> F[次像素估計\n(3x3 二次擬合)]
  F --> G[NMS 去重疊]
  G --> H[結果排序\nScore/Angle/Pos]
```

## 模組與職責
- `ncc.py`：封裝 NCC（OpenCV matchTemplate），輸出相似度地圖與前 K 大峰值。
- `rotation.py`：角度處理，旋轉樣板並保持完整內容。
- `pyramid.py`：金字塔構建、分層搜尋、候選傳遞與結果整合。
- `subpixel.py`：在相似度地圖做次像素峰值估計（提高定位精度）。
- `nms.py`：重疊度計算與非極大值抑制。
- `cli.py`：命令列介面，便於快速試用與驗證。

## 重要設計取捨
- 不強依賴平台特定 SIMD；以 OpenCV/NumPy 為基礎維持跨平台一致性。
- 角度搜尋採「粗→細」；小層廣搜，大層局部精修，兼顧效能與精度。
- NMS 控制重疊比例（overlap）以提升多目標情境穩定性。
- 分塊候選（Block Max）加速「大圖+小樣板」情境；次像素估計提升座標精度。
