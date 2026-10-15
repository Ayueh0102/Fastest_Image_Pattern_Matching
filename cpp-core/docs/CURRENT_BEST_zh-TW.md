# 目前建議使用的方案

更新：2026-10-15，Apple M4 Pro、48 GiB。這頁記錄研究流程的建議設定；不是宣稱所有場景最快。灰階尺度研究流程已可在原生 C++ GUI 選用，GUI 預設仍保留 CPU 原有流程。

**最新固定尺度稀疏研究路徑，按場景選CPU／Metal前段：** 第19輪已將方向建圖、稀疏評分與局部選峰搬到Metal，候選整理仍在CPU，精修可獨立選後端。同建置比較：4096²全CPU142.08→Metal前段＋CPU精修15.66ms（9.07倍），RSS311.27→255.26MiB；128物件全CPU23.76→Metal前段＋Metal精修11.03ms（2.15倍），RSS43.58→50.21MiB。小圖全角CPU1.81ms仍優於混合CPU2.83ms與混合Metal5.82ms，細線與本輪實圖也先用CPU。`responseMode=1`、`searchBackend=1`選Metal前段，`backend=0/1`獨立選CPU／Metal精修；沒有通用自動分流門檻。

64程序／192次正式量測的有序候選與工作量完全一致；144正常GT姿態皆零FP／FN，原提案127/128與尺度負控制失敗仍保留。Release32/32與相關ASan／UBSan、Metal驗證通過。完整[新前段實測與來源](SPARSE_METAL_FRONTEND_zh-TW.md)、[Liquid NN／固定運算圖研究](SPARSE_GPU_ARCHITECTURE_zh-TW.md)。此路徑尚未接GUI，不能用其結論改寫以下通用金字塔基準；前一輪[CPU前段＋CPU／Metal精修](SPARSE_CPU_METAL_zh-TW.md)的CPU建議保留為歷史比較。新GPU前段也使用預算方向LUT，而CPU尚用逐像素atan2；收益同時含數學化簡與平行化，還須補同LUT的優化CPU對照。

**穩定基準：同圖共用金字塔 → Metal 批次粗／細搜尋 → CPU 精確選峰 → GPU 常駐姿態精修 → CPU 最終驗分與有預算救援。** 完整流程仍是 CPU＋GPU 異構；灰階尺度排程、NMS 與運算核心現在都有 C++／Metal 源碼，原 Python 排程保留對照。Fourier 原型尚未改寫成全 C++。

| 部分 | 建議 |
|---|---|
| 同張圖跨尺度 | 每次 `prepare` 一次，共用來源金字塔；C++ Session 內含此步驟，新圖仍須重算 |
| 金字塔／角度批次 | `pyramid_mode=1`、`batch_angles=4` |
| 每層局部細搜尋 | `fine_batch_size=512`，Metal 批次旋轉＋NCC |
| 粗選峰 | `peak_mode=1`，精確 CPU tournament tree |
| 粗 NCC | 通用比較基準維持 `ncc_mode=0`；mode2／3 是已實作的空間重用選項 |
| CPU／GPU 粗搜尋重疊 | 基準保持關閉；6／12 物件可另評估，512 物件沒有穩定收益且多用記憶體 |
| 最後連續精修 | resident refinement＋CPU 驗分；`refine_rescue(..., resident=True)` 已包含精修，不重複執行 |
| 救援預算 | 使用 `rescue_policy_for_threshold(實際接受門檻)`，保留既有最多 4 個 seeds 與工作預算 |

最新原生 GUI 排程比較，本機大圖／大量同模板實例可選 **`ncc_mode=2`、重疊關閉**：8K 273.46→259.44 ms、128 物件1189.79→1149.74 ms、512 物件3426.37→3350.39 ms，完整結果一致。512 再開重疊為3352.14 ms，幾乎沒有額外收益，peak RSS 卻由789.84增至841.85 MiB。小圖組合模式另有一個 block 超過效能容忍值，因此仍未自動選型。

GUI 原生化已完成本輪驗證，但記憶體並非全面優於 Python：8K原生基準562.77 MiB、Python current365.93 MiB。下一個記憶體重點是讀圖暫存與來源副本的量測；不要把移除 Python 等同整體必然更省。

粗 NCC 分塊可以在同一數學定義下減少相鄰輸出的重複讀取，**但不能只看 GPU 核心變快就全面替換**。新 mode2 用四個相鄰輸出的暫存器重用；mode3 用 8×8 輸出合作載入 threadgroup tile。第13輪獨立 kernel 實驗的數字、記憶體、精確性及採用判斷見 [NCC 分塊報告](TILE_NCC_zh-TW.md)。

不建議直接採用：舊粗搜尋 SIMD mode1（實測變慢）、強制全部局部工作走 GPU、以 Fourier 提案取代現在的全圖搜尋。Fourier 在裁切／背景干擾下仍有失敗；RGB 三通道也不保證小工作比 CPU 快。

目前已驗證的是合成同模板多實例、旋轉／尺度、部分實圖和已知候選的 RGB 精修。高分週期錯峰、遮擋／透視、很多不同模板、完整 RGB 全圖搜尋、跨硬體與長時間串流仍需要另外驗證。

128物件10分鐘＋6物件2分鐘，共2246次持續比對、22次資源重建已通過。另完成本輪約定的 [512物件30分鐘測試](STABILITY_30MIN_zh-TW.md)：534次、5次重建，完整結果一致，RSS起末中位數622.52→634.55 MiB，增加12.03 MiB並通過事前門檻。這些都是固定圖測試，尚未覆蓋新相機影格串流。

目前不只剩穩定性測試；灰階 GUI 主流程已可使用，大圖記憶體、細搜尋、更多真實場景，以及 RGB／Fourier 擴充仍有後續工作。完成範圍與優先順序見 [專案狀態](PROJECT_STATUS_zh-TW.md)。

**實拍裁模板、只旋轉且不變尺度：** 可先用`scales=[1]`保留金字塔，精修與救援會鎖尺度。第15輪比較同一執行期的無金字塔路徑：四個合成GT場景兩路都零FP／FN，但Metal無金字塔慢6.9～17.1倍，中圖RSS36.70→98.70 MiB；本輪不改預設。完整[實測與限制](FIXED_SCALE_SEARCH_zh-TW.md)、[旋轉模板／稀疏提案／FFT研究方向](FIXED_SCALE_SEARCH_DESIGN_zh-TW.md)已保存。

**固定尺度的新增研究選項：** 第16輪已實作完整C++原圖稀疏方向提案，再接CPU／Metal完整NCC精修。32程序、96正式次；本機128物件241.65→22.48ms，peak RSS65.98→64.37MiB，最終128/128、零FP／FN。小圖全角Metal23.87→5.27ms，CPU41.27→1.74ms；但細線慢約31%，128提案捕獲鄰域只127/128，靠後段精修才找齊。因此不改通用預設、不自動接入GUI；完整[結果、來源與失敗gate](SPARSE_ORIENTATION_zh-TW.md)保留。下一個成本集中在來源方向圖準備與粗種子的精修，值得研究分塊／Metal建圖及CPU／GPU精修分界。

**稀疏研究路徑的省記憶體選項：** 第17輪`responseMode=1`以單張mask＋2KiB LUT取代八張回應圖，54程序／162正式次的有序輸出與工作量相同。4096²同建置157.52→144.95ms、RSS510.74→382.73MiB；128物件22.17→21.58ms、63.14→55.85MiB。小圖／細線略慢，仍保留mode0與通用金字塔預設；見[實測](SPARSE_COMPACT_zh-TW.md)及[數學／agent建議](SPARSE_NEXT_STEPS_MATH_zh-TW.md)。

原生 GUI 的使用、組合模式比較與有界持續運行結果見 [第14輪整合報告](GUI_INTEGRATION_AND_STABILITY_zh-TW.md)。

人工 review 請從 [19 階段實驗總索引](EXPERIMENT_REVIEW_zh-TW.md) 開始：每一輪都保留原始資料、比較版本、失敗與採用決定。API 使用例在該索引開頭；完整建置與 GUI 說明見 [README](../README_zh-TW.md)。
