# 專案完成範圍與階段結論

更新：2026-10-19。**可移植灰階C++核心、Qt GUI與CPU／Metal研究流程已實作，仍不能宣稱所有實拍、遮擋與異色檢測都已完成。** 本階段採用與拒絕的方案都有源碼、測試和量測證據。第一次接觸請看[外部閱讀指南](../../docs/PROJECT_GUIDE_zh-TW.md)，實際設定看[目前建議方案](CURRENT_BEST_zh-TW.md)。

第40輪完成[精確NCC優化與本大階段結案](GPU_NCC_CLOSURE_zh-TW.md)：有界累加、統計重用、SIMD32、正式／stop0／七組保留場景、跨版本控制與完整封存已結束。peak1／fine4使原128／512完整GPU時間再少16.3%／15.4%，命名GPU配置不增、結果精確一致；128接近混合，512仍以混合較快。562次直接GT、70項Release、Metal／ASan／UBSan、乾淨重建、五分鐘連續檢查與八份摘要重播通過。失敗變體與全部速度失敗均保留。

> 2026-10-19更正：合併後驗收審查的非阻塞發現已處理。混合正式呼叫重用pose context，加回後fine4在128約快4%、512仍以混合較快；8K是常數背景的速度負載；舊的`intermediate_readbacks`／`cpu_algorithm_steps`是常數，現改為推導並以攔截Metal呼叫的測試檢查；Metal測試只在沒有Metal裝置時才skip；runner拒絕未知job key。Release測試現為73項。見[第40輪報告文末更正](GPU_NCC_CLOSURE_zh-TW.md)。

第41輪[每換一張影格的成本](NEW_FRAME_COST_zh-TW.md)實測了上述估算：每次呼叫都經過`updateSource`時，fine4對混合每影格為128 −6.9%、512 +13.7%、六物件 +13.3%、8K −70.2%，各6個block同向，估算方向成立。混合每影格重建pose context中位19.94／40.19ms，另有候選搜尋的額外時間。96程序、1008次直接GT零FP／FN，保留13筆每影格time_budget失敗；只做量測，未改預設。

2026-10-24穩定核心（非研究輪次）加入[ARM DOTPROD點積](CORE_DOTPROD_zh-TW.md)：GUI／CLI預設的`Backend::Auto`在有`__ARM_FEATURE_DOTPROD`的aarch64編譯改用UDOT（單一累加器、每4096 bytes歸約、遮罩尾端），其他ARM編譯照舊走NEON。輸出與原NEON、scalar逐位元相同（408次執行零差異）；Src7模式基準快13.7%（359.10→309.84ms），17張固定影像幾何平均快7.0%。舊OpenCV＋DOTPROD版仍快1.64倍，剩餘時間主要在旋轉、NCC其他部分與金字塔。第一個雙累加器候選使Src4慢9.9%，未採用但保留。

2026-10-25接著加入[精確旋轉與金字塔加速](CORE_WARP_PYRDOWN_zh-TW.md)：clang＋aarch64以NEON每次兩像素做`warp`，每一步照原scalar運算式的FMA合併方式；`pyrDown`的內部欄直接計算，全程整數。輸出與`51d95b1`逐位元相同（544次執行零差異，包括純scalar建置），Src7模式基準再快21.3%（319.76→251.49ms），17張固定影像每張都較快、幾何平均快22.7%。舊OpenCV＋DOTPROD版仍快1.32倍；`warp`在大模板仍占近一半，Src5／Src10改以`computeNcc`為主。

第37輪補齊[完整Metal灰階搜尋資料路徑](FULL_GPU_zh-TW.md)，包含細搜尋控制、精修／救援及最終NMS；一次提交、最後一次等待。保留研究開關，原混合路徑與GUI預設不變。

第38輪完成[GPU分階段計時與精確分塊選峰](GPU_TILED_PEAKS_zh-TW.md)：128／512完整GPU耗時降低79.8%／87.3%，有序輸出不變；仍較混合慢15.5%／38.7%，細搜尋成為下一個主要瓶頸。

第39輪[分列快取與執行計數](GPU_FINE_TILES_zh-TW.md)證實原128／512完全未走過大ROI路徑，主改善目標未達。補充stop0可少約10%耗時，保留研究源碼；後續有界uint32、重疊統計及SIMD工作分配已由第40輪完成。

第36輪實作Metal粗NCC正規化、選峰與停止常駐：8K少物件有收益，大量候選顯著退步；保留研究選項與全部源碼，不改預設。見[常駐粗搜尋報告](COARSE_RESIDENT_zh-TW.md)。

第35輪加入真實密集RGB照片、49物件旋轉與異色控制，並整理校準、原始量測、失敗與來源。結果與界線見[本輪報告](STAGE35_REAL_RGB_zh-TW.md)。第34輪完成五項缺口的有界實驗，包含Blender、新影格、CPU／Metal公平比較及30分鐘持續運行；「完成實驗」包含證明某項方案無效，不等於所有場景通過。

## 已實作並有可追溯證據

| 範圍 | 狀態 | 證據入口 |
|---|---|---|
| 固定尺度灰階核心 | C++17完整自有實作；旋轉、金字塔、NCC、NMS與subpixel；ARM編譯期UDOT點積、NEON精確旋轉 | [移植契約](PORTING_zh-TW.md)、[核心與GUI建置](../README_zh-TW.md)、[DOTPROD點積](CORE_DOTPROD_zh-TW.md)、[旋轉與金字塔](CORE_WARP_PYRDOWN_zh-TW.md) |
| 灰階多尺度與連續精修 | 原生C++排程、CPU／Metal後端；GUI可選研究流程 | [總索引](EXPERIMENT_REVIEW_zh-TW.md)、`experiments/research_pipeline.hpp` |
| 大量物件與大圖 | 有128／512物件、4K／8K測量；同模板合成為主 | [第34輪](STAGE34_CLOSURE_zh-TW.md)、各輪原始封存 |
| 記憶體與提交改善 | gray8讀圖、u8初始化、共用影像、細搜尋旋轉／NCC融合 | [第31輪](GRAY_LOAD_MEMORY_zh-TW.md)、[第32輪](RUN_VARIABILITY_POSE_MEMORY_zh-TW.md) |
| 粗搜尋GPU常駐選項 | 整數NCC統計、FP32正規化、逐次選峰與停止同批提交；只讀回候選，非全流程常駐 | [第36輪](COARSE_RESIDENT_zh-TW.md)；56項Release與封存重現，保留性能失敗 |
| 完整灰階GPU常駐研究後端 | 金字塔、旋轉、粗細搜尋、精修／救援、NMS及結果幾何均在Metal；FP32契約 | [第37輪](FULL_GPU_zh-TW.md)，CPU僅作靜態規劃與I/O；不含RGB驗色與稀疏路徑 |
| 新影格與穩定性 | updateSource；192次fresh/reuse一致，第34輪512混合設定30分鐘745次通過；第40輪新GPU變體五分鐘108次通過；第41輪96程序每次呼叫換影格，混合與fine4每影格成本實測 | [第34輪](STAGE34_CLOSURE_zh-TW.md)、[第41輪](NEW_FRAME_COST_zh-TW.md)；同一影像重複，不等於實拍串流精度 |
| RGB候選評分與色度 | 完整CPU／Metal RGB NCC、CPU線性色度源碼；同程序搜尋＋驗色 | [runner契約](COLOR_VERIFY_RUNNER.md)、[第35輪](STAGE35_REAL_RGB_zh-TW.md) |
| 重現與外部review | protocol、正式worker、失敗、hash、來源與驗證分開保存 | [Log分類](LOG_CATALOG_zh-TW.md)、[實驗總索引](EXPERIMENT_REVIEW_zh-TW.md) |

「完整源碼」涵蓋專案自有程式及測試，不是把Qt、OpenCV、Metal系統框架的全部來源打包。灰階核心無Qt／OpenCV依賴；GUI解碼及研究工具依賴分開列於建置文件。CUDA只有設計對應，沒有本機實作／效能驗收。

## 本階段以外的研究缺口

| 缺口 | 已知原因或反例 | 下一個有意義的實驗 |
|---|---|---|
| 小模板粗搜尋召回 | 過深金字塔丟失可辨紋理；第35輪可由保留較高解析層改善，但有時間代價 | 依模板資訊量選擇層級；與固定512、原256、公平預算比較 |
| 遮擋與相似物身份 | 全局NCC漏候選或接受近似負例；增加預算無效 | 分區一致性／部分支撐、多峰歧義，保留無目標負控制 |
| 實拍顏色異常 | 前段漏搜；通道偏移可被NCC抵消；白平衡又可能觸發色度誤報 | 多個正常外觀模板、光照校正、小缺陷局部統計；先補實拍實例GT與正常樣本 |
| 更多GPU並行 | 第40輪三個精確NCC變體已完成；512新方案細搜尋約占GPU47.6%、粗選峰33.7% | 後續可另研究選峰／細搜尋分工或允許近似的架構；沒有證據已達極限，不能將階段時間當硬體利用率 |
| 多種模板庫、透視、離平面 | 尚未有完整場景與標註驗收 | 各自建立獨立benchmark，不能由同模板512物件外推 |
| 跨機器部署 | 大部分性能證據來自M4 Pro | 新CPU／Metal機器重現固定protocol，另做依賴與GUI包裝驗收 |

這些是後續研究項目，不代表要在本輪無限增加實驗。現階段可以交付可閱讀、可重建、可核對的專案；應按應用條件選用已通過的組合，對未通過場景保留限制。

## 審查時最容易混淆的三件事

1. 同一後端輸出一致，不等於GT準確；離線封存完整性通過也可以包含精度失敗。
2. RGB檔案輸入灰階搜尋、已知姿態RGB評分、全圖RGB搜尋、顏色缺陷檢測，是不同能力。
3. kernel加速、完整算法加速、初始化與實際相機吞吐量，是不同測量範圍。

[第1–34輪狀態原文](PROJECT_STATUS_HISTORY_zh-TW.md)保留歷史；其中當時的「目前／下一步」不再作現行狀態。
