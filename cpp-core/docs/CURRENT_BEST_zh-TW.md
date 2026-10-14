# 目前建議使用的方案

更新：2026-10-14，Apple M4 Pro、48 GiB。這頁記錄研究流程的建議設定；不是宣稱所有場景最快，也不是 GUI 現在已自動使用全部實驗功能。

**穩定基準：同圖共用金字塔 → Metal 批次粗／細搜尋 → CPU 精確選峰 → GPU 常駐姿態精修 → CPU 最終驗分與有預算救援。** 完整流程仍是 CPU＋GPU 異構，研究排程在 Python，運算核心有 C++／Metal 原始碼。

| 部分 | 建議 |
|---|---|
| 同張圖跨尺度 | `SourceFrame.prepare` 一次，共用來源金字塔；每張新圖重新準備 |
| 金字塔／角度批次 | `pyramid_mode=1`、`batch_angles=4` |
| 每層局部細搜尋 | `fine_batch_size=512`，Metal 批次旋轉＋NCC |
| 粗選峰 | `peak_mode=1`，精確 CPU tournament tree |
| 粗 NCC | 通用比較基準維持 `ncc_mode=0`；mode2／3 是本輪新增的空間重用選項 |
| CPU／GPU 粗搜尋重疊 | 基準保持關閉；6／12 物件可另評估，512 物件沒有穩定收益且多用記憶體 |
| 最後連續精修 | resident refinement＋CPU 驗分；`refine_rescue(..., resident=True)` 已包含精修，不重複執行 |
| 救援預算 | 使用 `rescue_policy_for_threshold(實際接受門檻)`，保留既有最多 4 個 seeds 與工作預算 |

本機大圖／大量同模板實例可選 **`ncc_mode=2`**：對同build mode0，8K約減時5.2%、128約2.9%、512約3.5%，512物件全部找齊且零誤報，GPU buffer容量不增加。但12物件約慢3.6%，所以尚未自動選型。

粗 NCC 分塊可以在同一數學定義下減少相鄰輸出的重複讀取，**但不能只看 GPU 核心變快就全面替換**。新 mode2 用四個相鄰輸出的暫存器重用；mode3 用 8×8 輸出合作載入 threadgroup tile。正式數字、記憶體、精確性及採用判斷見 [本輪 NCC 分塊報告](TILE_NCC_zh-TW.md)。

不建議直接採用：舊粗搜尋 SIMD mode1（實測變慢）、強制全部局部工作走 GPU、以 Fourier 提案取代現在的全圖搜尋。Fourier 在裁切／背景干擾下仍有失敗；RGB 三通道也不保證小工作比 CPU 快。

目前已驗證的是合成同模板多實例、旋轉／尺度、部分實圖和已知候選的 RGB 精修。高分週期錯峰、遮擋／透視、很多不同模板、完整 RGB 全圖搜尋、跨硬體與長時間串流仍需要另外驗證。

人工 review 請從 [13 階段實驗總索引](EXPERIMENT_REVIEW_zh-TW.md) 開始：每一輪都保留原始資料、比較版本、失敗與採用決定。API 使用例在該索引開頭；完整建置與 GUI 說明見 [README](../README_zh-TW.md)。
