# 移植紀錄與差異（Migration Notes）

本文件紀錄自 C++/MFC 專案移植至 Python 版本之過程、功能對應與差異說明，便於後續驗證與維護。

## 目標對齊（功能對應）
- NCC 原理：以 `TM_CCORR_NORMED`/`TM_CCOEFF_NORMED` 對應原始 NCC。
- 影像金字塔：相同策略（自大到小建立，再由小至大精修）。
- 角度搜尋：以粗步進→細步進方式逼近最佳角度。
- 結果輸出：Index、Score、Angle、PosX、PosY（排序：Score 由高至低）。
- 多目標：以 NMS 控制重疊，對齊原始專案之去重疊設定。

## 差異與取捨
- SIMD：原專案含平台最佳化（如 Neon）；Python 版以通用向量化 + OpenCV/FFT 取代。
- UI：無 MFC；改以 Tk GUI + CLI + API 驗證與展示。
- 效能：金字塔 + 分塊候選（Block Max）+ ROI 精修 + FFT NCC 提供穩健效能；
  如需極致效能可再評估 Numba JIT 版 NCC（未納入）。

## 前/後對比建議驗證
1. 範例影像：沿用 `../Test Images/` 與 `../Result Images/` 做視覺比對。
2. 指標：
   - 準確度：最佳角度、中心座標與 Score 是否一致或相近（允許像素級偏差）。
   - 效能：Release 模式（或等效環境）下之耗時統計。
3. 報告：將每階段的「功能完成度」「差異原因」「輸出範例」追加到此文件底部。

## 待辦（Roadmap）
- [x] 最小功能：單尺度 NCC + CLI
- [x] 金字塔 + 區域化搜尋
- [x] 角度粗到細搜尋
- [x] NMS + 排序輸出
- [x] 次像素估計（`subpixel.refine_peak_subpixel`）
- [x] 分塊候選（`ncc.topk_blocked`）
- [x] 可選加速（FFT NCC，`ncc_match_fft`）
- [ ] 單元測試與基準測試完善化
