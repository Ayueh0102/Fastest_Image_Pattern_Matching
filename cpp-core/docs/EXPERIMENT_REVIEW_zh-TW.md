# Template Matching 研究總索引與人工審查指南

整理日期：2026-10-16。前12階段整理 repository 已保存的報告、protocol 與結果；第13階段為 NCC tiling 正式量測，第14階段接入原生尺度 GUI 並驗證組合模式與持續運行，第15階段比較固定尺度的金字塔與原解析度搜尋，第16階段實作稀疏方向全圖提案並比較128物件，第17階段驗證等價回應壓縮，第18階段比較稀疏全CPU與Metal精修，第19階段實作並驗證Metal稀疏前段，第20階段補相同方向LUT的CPU對照以拆開數學化簡與GPU收益，第21階段驗證GPU建圖與首批搜尋合併提交但保留效能失敗，第22階段完成灰階GPU精修融合與四路效能比較，第23階段再驗證SIMD128群組內加總與共享暫存優化，第24階段減少CPU／Metal精修影像副本並擴充8K24物件，第25階段以直接8-bit入口移除初始化float暫存並比較CPU／Metal，第26階段回到原始NCC，完成自適應矩形上界的C++原型與CPU比較但不採用為金字塔粗搜尋預設，第27階段實作Metal fine統計重用並保留未達加速目標的結果，第28階段以SIMD strip共用相鄰點積載入並取得多候選流程加速。第29階段量測旋轉同步及精確重用上一輪角度圖，完整次像素加速但保留RSS／時間失敗與複查。第30階段融合細搜尋旋轉與NCC提交並定位讀圖暫存，第31階段移除明確灰階輸入的RGB中間圖，第32階段追查慢48%樣本並驗證精修共用／直接入口與多尺度512流程的組合。第33階段補原resident／融合／SIMD與最新多尺度的同binary組合，並以新模板姿態、降質與無目標場景揭露共同精度失敗。各數字是各輪的證據快照，不是同一輪連續量測；不要將不同輪次的加速比相乘。快速設定摘要見 [目前建議方案](CURRENT_BEST_zh-TW.md)，整體完成範圍與待辦見 [專案狀態](PROJECT_STATUS_zh-TW.md)。

## 目前推薦方案

**第33輪：多尺度仍保留原resident0，優先處理新增的尺度精度缺口。** [三路組合與新場景比較](RESIDENT_MULTISCALE_zh-TW.md)完成48程序／96次，前段、有序結果及路徑一致，但準確度與整體性能未全過：7個性能配對失敗、RSS全過；512快速原版／SIMD完整時間2342.54／2402.90ms，精修206.18／240.13ms，不升mode2為通用預設。新增小模板12個都定位正確，5個因尺度偏小超過2%而strict失敗；12條固定GT幾何的NCC尺度曲線都偏好較小尺度，另有遮擋錯方向與缺候選。這些共同失敗完整保留，不能用舊512全過推論所有場景可靠。後續先做取樣／抗混疊精度實驗；完整[實驗缺口審查](EXPERIMENT_GAPS_zh-TW.md)另列新影格、真實GT、CPU公平比較與統計限制。

**第32輪：慢48%未重現，精修配置合用可再省約37%記憶體。** [追查與512物件整合報告](RUN_VARIABILITY_POSE_MEMORY_zh-TW.md)：12程序／48次交錯對照未見可重現退化；原慢樣本97.92%的額外時間在前段搜尋，結果／工作量相同，但缺少當時系統紀錄，根因仍未確認，原失敗保留。另16程序／32次比較已有`poseInputMode=1`＋`poseStorageMode=1`與新多尺度流程的組合：快速RSS689.47→431.02MiB、完整次像素697.88→440.09MiB；repeat時間−0.21%／+0.56%，有序結果相同、512個GT零FP／FN，所有事前容忍通過。可手動搭配fine mode2／融合／適用時角度重用，本輪residentMode保持0；沒有NCC新加速、預設或GUI選單變更。

第29輪新增[精確角度圖重用](FINE_ANGLE_REUSE_zh-TW.md)：同候選／層級／pass僅重用上一輪位元相同角度的分數圖，6／128／512完整次像素時間縮短約32%／33%／39%，完整結果相同。原快速stopLayer1設定零命中，不改其建議。主72程序／312次，有1個default速度及3個RSS配對門檻失敗；額外8程序複查未重現，所有證據保留。手動`fineAngleReuse=true`、診斷保持關閉；512仍需部署記憶體驗證。

第28輪新增[Metal fine點積載入重用](FINE_NCC_STRIP_zh-TW.md)：120程序／960次，585,648個CPU窗口比較逐位一致、666既有GT姿態零FP／FN；128／512完整時間約快22%／23%。本機多候選金字塔工作可手動優先試`fineNccMode=2`，顯式buffer與提交數不變，原預設保留。8K少候選總時間沒有收益；以下第27輪結論保留作為歷史對照。

第27輪新增[Metal fine窗口統計重用](FINE_NCC_REUSE_zh-TW.md)：80程序／620次、365,344個CPU窗口逐位一致，666個既有GT姿態零FP／FN。128／512物件1152.31→1181.34ms／3372.58→3424.14ms，高瘦模板另有明顯退步，因此保持原fine模式0；完整數據及來源見第27節。

另見[原始金字塔／旋轉／NCC的下一輪研究](ORIGINAL_ALGORITHM_NEXT_zh-TW.md)：四項演算法實驗設計、自適應投影的推導，以及10模板／1860視窗的探索性探針。提案A後續已完成[第26輪C++實作與量測](ADAPTIVE_NCC_zh-TW.md)；完整搜尋變慢，不更改以下推薦方案。

固定尺度的新稀疏研究路徑，第20輪CPU也採同樣LUT後，4096²的Metal前段＋CPU精修仍比查表CPU快8.36倍、128物件快1.42倍，小圖則CPU較快。查表CPU已保留可選模式，第19輪的Metal精修選項另留比較；尚無通用自動routing門檻。第21輪合併提交雖減少等待，128物件仍有速度回退，因此維持分開提交預設。第22輪融合精修在已測八場景快於原Metal精修；128可選融合，多數少候選場景仍選CPU精修，完整證據見第22節。第23輪新增SIMD128，128物件可優先試mode2：同輪fused256／SIMD128為10.52／8.56ms、複查10.37／8.55ms；多數少候選仍先CPU，完整比較與歸因限制見第23節。第24輪可加`poseStorageMode=1`：4K／8K各省約64／256MiB，數值不變；8K時間49.82／49.78ms幾乎持平，詳第24節。第25輪再加`poseInputMode=1`，8K峰值RSS 946.59→690.53MiB，初始化125.84→81.29ms；暖機無穩定加速，CPU慢樣本亦保留，見第25節。下表屬於原有金字塔流程，其推薦與新稀疏路徑分開判斷。

在本機 Metal 金字塔研究流程，以**共享來源金字塔＋批次細搜尋＋精確選峰樹**作為下一次比較的基準：

| 設定／階段 | 建議值 | 適用界線 |
|---|---|---|
| 來源 frame | 同張新圖 `prepare` 一次，各尺度共用 | 新影像必須重算；每張 prepare 包含在總時間 |
| 金字塔 | `pyramid_mode=1` | 精確整數直接版；mode2 可分離版另外付出較大暫存 |
| 粗搜尋角度批次 | `batch_angles=4` | 上限與尾批仍按容量拆分 |
| 金字塔細搜尋 | `fine_batch_size=512` | Metal guarded warp＋整數 NCC；CPU repair／double normalization／搜尋控制仍存在 |
| 粗選峰 | `peak_mode=1` | CPU 精確 tournament tree，保留原峰值順序 |
| 粗 NCC | 通用基準 `ncc_mode=0`；大場景可選mode2 | mode2大場景有收益；小圖／重疊組合依第14輪結果判斷，mode1不推薦，mode3保留對照 |
| CPU／GPU coarse overlap | 關閉，`enabled=False` | 完整實作已可用，但 512 物件效能混合且記憶體增加 |
| 後段連續姿態 | resident refinement＋既有有預算 rescue | 使用 CPU 最終驗分；低分救援觸發上限配合實際接受門檻 |

這是**已測研究組合的基準設定**，不是所有工作負載都最快的結論，Qt GUI 已能選用此原生尺度流程，但預設仍為 CPU 原有流程。小型局部精修有 CPU 勝過 Metal 的實測；歷史 auto routing 的門檻針對當時混合精修校準，未證實可直接套用至最新整條流程。

GUI 與原生程式現可直接使用以下 C++ 排程；來源／模板為 `fpm::ImageView`，Session 自行持有像素、模型與 GPU 資源，一次只供一個呼叫者使用：

```cpp
#include "research_pipeline.hpp"
fpm::research::Settings settings;
settings.backend = 1; // Metal；CPU 為 0
settings.scales = {.75, 1., 1.25};
fpm::research::Session session(sourceView, templateView, settings);
fpm::research::Options options;
options.match.minScore = .8;
options.match.maxTargets = 70;
options.nccMode = 0;
options.overlap = false;
auto result = session.run(options); // 包含共享 frame、refine/rescue 與跨尺度 NMS
```

CMake 啟用實驗庫並連結 `fpm_research_pipeline`。完整建置及 GUI 操作見 [README](../README_zh-TW.md#原生尺度搜尋與-metal-gui)；舊 Python 排程繼續保留研究對照。

已有 `models`（各尺度 `CoarsePattern`）、`source_view` 與 `config` 時，前段設定如下；跨尺度合併、pose 座標轉換、精修與最後 NMS 由研究排程完成：

```python
from coarse_experiment import SourceFrame

with SourceFrame(build, backend='metal', pyramid_mode=1,
                 batch_angles=4, fine_batch_size=512) as frame:
    frame.set_coarse_optimization(peak_mode=1, ncc_mode=0)
    frame.set_coarse_overlap(enabled=False)
    frame.prepare(source_view, models)
    proposals_by_scale = [frame.match(model, config) for model in models]
```

後段既有接口是 `Scorer.refine_rescue(seeds, resident=True, **policy)`，其中 `policy = rescue_policy_for_threshold(minimum_score=實際最終門檻)`；它包含原 resident 精修與必要的多起點，不必先重複呼叫一次 resident。保留預設每次最多 4 個救援 seeds 與原工作預算，不能為了比較速度暗中減少候選。完整座標／尺度／生命週期範例見 [共享 frame 使用方式](SHARED_FRAME_AND_FINE_METAL_zh-TW.md)、[rescue API](NCC_RESCUE_AND_METAL_COARSE_zh-TW.md)、[研究排程](../tools/real_image_experiments.py)。

## 先辨認比較的是哪一層

| 名稱 | 目前含義 | 不包含的保證 |
|---|---|---|
| 原 MFC C++ | `MatchTool/` 上游歷史版本，Git 源碼稽核 | 本機未執行 Windows MFC binary |
| 舊 macOS C++ | 保留的 OpenCV／DOTPROD 移植，比較工具的執行期歷史參考 | 不等同直接在 Windows 跑最早版 |
| 新 portable CPU | `fpm_core` 的完整自有灰階、固定尺度 NCC／旋轉／金字塔／subpixel | 不是預設原生全圖尺度／RGB 搜尋 |
| pose scorer／refiner | float32 幾何、灰階或 RGB、連續平移／角度／log-scale，C++／Metal 核心 | 只給 seeds 的測試不是全圖偵測召回率 |
| 全圖研究 pipeline | 各尺度模板搜尋＋後段 pose 精修／rescue／NMS；原生 C++ 已接入選配 GUI，Python 保留對照 | 灰階尺度流程已原生化，仍不是全 C++ Fourier 或全 GPU 流程 |
| 金字塔 fine search | 每層局部 ROI 的離散角度與位置搜尋 | 與最後四參數 resident LM 是不同階段 |

目前跨階段的大部分效能證據來自 Apple M4 Pro、12 CPU／16 GPU cores、48 GiB、macOS 26.6.2、Release；早期稽核另記 AppleClang 21、Qt 6.10.2、OpenCV 4.13.0。CPU 實驗多使用 8 threads，但 portable core 的 NEON 搜尋不可因此稱作八核心搜尋。CUDA、Windows／Linux 實機性能並無本頁可引用的完成證據。

## 逐階段目的、證據與決定

### 1. 原版稽核、完整 CPU 核心與 GUI（2026-10-08）

**目的／實作：** 核對最早有程式的 Git `f521243`、上游 MFC `d8edabb`、舊 macOS／Python 移植。補齊自有金字塔、旋轉插值、NCC、SIMD、選峰、矩形交集、次像素求解與 Qt GUI 連接；不冒稱逐位重現 OpenCV。Python 原有 top-k、提前停止與 NMS 定義差異保留稽核證據。

**資料／時間／記憶體／精度：** 17 組 repository 圖片。初輪 Src7 舊 macOS／新 CPU 為 63.75／134.23 ms；Src10 100 個結果為 14.85／58.67 ms，此輪新核心較慢。另一次統一六目標設定的 mode 比較為舊 C++ 177.600 ms、新 NEON 344.348 ms、FastMode 134.198 ms、Subpixel 3188.826 ms，精度模式不同不能混比。此階段沒有可引用的標準化 peak RSS 表。合成旋轉在 minArea256 下改善後最大位置 0.0051 px、角度 0.0217°；Src7 低分三物件約 0.64–0.65，不能透過任意提高分數門檻保留。

**決定：採用作可移植正確性基準；未證實比舊 OpenCV C++ 快。** FastMode 犧牲解析度，subpixel 追加精度有明顯時間代價。

來源：[稽核](AUDIT_zh-TW.md)・[17 組結果](validation.json)・[模式對照](mode-benchmark.json)・[旋轉／subpixel](rotation-search-validation.json)・[旋轉來源或模板診斷](rotation-direction-validation.json)・[移植契約](PORTING_zh-TW.md)。

### 2. Fourier／旋轉尺度／RGB 提案與批次評分（2026-10-08～12）

**目的／實作：** 以 Fourier magnitude、log-polar、phase correlation 提案旋轉／尺度，再由 NCC 驗證；C++ 多執行緒與 Metal 實作固定姿態取樣＋RGB joint NCC。FFT 仍是 CPU NumPy，沒有 GPU FFT、CUDA 或全 C++ Fourier 管線；全圖版本仍先跑既有角度枚舉。

**資料／時間／記憶體／精度：** 111 組早期實驗、解析紋理／裁切／stress、固定姿態與後續實圖。常駐評分 128² RGB／512 poses：CPU8 6.805 ms、Metal 1.182 ms；64² 灰階單 pose：CPU1 0.011 ms、Metal 0.095 ms。ROI 搜尋灰階平均 grid Metal 14.004 ms、Fourier Metal 36.725 ms；八個案例最大 RSS 69.6／145.6 MiB。裁切 Fourier 0/2、guarded fallback 2/2；stress guarded 仍有 2 FP／4 FN。實圖 Src7 加 Fourier 提案由 175.809 增至 341.470 ms；受控六物件幾何通過僅從 3/6 到 4/6。

**決定：採用可獨立重用的批次 scorer；Fourier 提案保留研究，不作目前推薦主路徑。** RGB 三通道不保證 GPU 更快；裁切／背景會破壞整體配準假設。

來源：[數學與原始文獻](RESEARCH_FOURIER_SCALE_RGB_zh-TW.md)・[方法](EXPERIMENTS_zh-TW.md)・[111 組結果](EXPERIMENT_RESULTS_zh-TW.md)・[原始 benchmark](pose-experiment-benchmark.json)・[數值驗證](pose-experiment-validation.json)・[原型驗證](fourier-ncc-research-validation.json)・[實圖比較](REAL_IMAGE_RESULTS_zh-TW.md)・[實圖原始資料](real-image-experiments.json)。

### 3. CPU 連續姿態精修與保守 NCC 剪枝（2026-10-12）

**目的／實作：** 自有解析 Jacobian、四參數 LM 求解修正粗種子；Cauchy–Schwarz／投影上界只排除不可能達門檻的粗 NCC。並將 Metal 評分 workspace 分批。

**資料／時間／記憶體／精度：** 48 組局部 seeds 的嚴格門檻由舊網格 12/48 到 LM 48/48；全圖六／十二物件由 3/6、7/12 到 6/6、12/12，總時間約 221.65→217.28、541.05→537.35 ms。此時 `continuous_metal` 的 LM 仍在 CPU。4K RGB／256²／8192 poses 固定評分 CPU8 500.92 ms、Metal 74.56 ms，Metal workspace 16.00 MiB、RSS 382.66 MiB，CPU RSS 275.27 MiB。NCC 剪枝隨機128²模板的單張 map 約 2.92×；沒有同倍數全圖加速證據。

**決定：採用連續精修作後續基礎；剪枝保留選配。** 精度提升包含擴大可達位置／角度範圍，不能當同搜尋範圍的求解器競速；額外角度／尺度候選庫的消融未顯示必要。

來源：[報告](PERFORMANCE_ACCURACY_zh-TW.md)・[全圖](performance-accuracy/full-image.json)・[候選庫消融](performance-accuracy/ablation.json)・[壓力測試](performance-accuracy/stress.json)・[剪枝](performance-accuracy/ncc-pruning.json)・[驗證](performance-accuracy/validation.json)。

### 4. GPU 導數、CPU 求解的混合精修（2026-10-12）

**目的／實作：** Metal 累加取樣／Jacobian／穩定統計，CPU 解 4×4 阻尼系統及末端驗分；每輪仍往返，可另加 CPU polish。

**資料／時間／記憶體／精度：** 大模板局部測試，768² RGB／32 seeds CPU 373.59 ms、無 polish GPU 144.51 ms；小模板跨案例中位數 CPU 約 1.12 ms、GPU 6.85 ms。完整 Src7 254.26→201.37 ms，但十二物件 513.60→547.55 ms。1024² RGB／32 完整支撐，GPU workspace4.50 MiB、Metal buffers64.50 MiB、RSS225.9 MiB，CPU RSS153.7 MiB，32/32 通過；先前被來源邊界裁切版本只有2/32，仍保留失敗。

**決定：大型局部工作有效，作後續 resident 的對照；不全面強制 GPU，不以更換裁切 fixture 宣稱解決遮擋。**

來源：[報告](GPU_REFINEMENT_zh-TW.md)・[局部資料](gpu-refinement/local-refinement.json)・[大型資料選取視圖](gpu-refinement/large-workloads.json)・[裁切失敗](gpu-refinement/large-workloads-clipped.json)・[無 polish 全圖](gpu-refinement/full-image-no-polish.json)・[驗證](gpu-refinement/validation.json)。

### 5. CPU／Metal routing 與成本診斷（2026-10-13）

**目的／實作：** 24 案例校準後固定面積×通道門檻、12 案例保留測試，使用 lazy Metal 初始化；另以隔離診斷 build 量 submit／wait／資料複製，未更改算法。

**資料／時間／記憶體／精度：** routing 保留集各方法120/120通過；Src7 CPU244.90、強制Metal197.56、Auto198.06 ms。四個小模板全圖選CPU；Src3 RSS Auto92.47／強制Metal112.09 MiB。不是每次都選最快：384²灰階8 seeds 校準反例選Metal8.33 ms而CPU7.20 ms。診斷 Src7 精修 wall61.641 ms、commit/wait56.269 ms、GPU區間45.725 ms；小 pose複製與結果／矩陣讀回合計0.0223 ms，不能把等待全部稱為傳輸。

**決定：routing 已實作但只是當時混合精修的本機暖機政策；診斷支持減少往返，未證明已達算力／頻寬極限。** 512²RGB診斷插入計時造成約20.9%擾動；沒有硬體 counter，不能由這份報告推論 cache miss 或純 transfer 占比。

來源：[routing](METAL_ROUTING_zh-TW.md)・[policy](pose-routing/policy-selection.json)・[held-out](pose-routing/heldout.json)・[全圖](pose-routing/full-image.json)・[成本診斷](METAL_COST_ANALYSIS_zh-TW.md)・[事件與 trace 索引](metal-cost-profile/report.json)・[診斷驗證](metal-cost-profile/validation.json)。

### 6. GPU resident 求解與停止（2026-10-13）

**目的／實作：** LM 矩陣組裝、求解、阻尼、接受與停止留在GPU；每批一次提交／等待；可保留CPU末端驗分，沒有CPU polish。

**資料／時間／記憶體／精度：** 凍結 `4144ffc`，46案例×4方法；五組大型局部純GPU比原混合約減時23–29%，含CPU驗分仍減時5–14%。Src7整圖原混合200.631→resident含驗分178.854 ms，RSS281.67→282.02 MiB，額外retained buffer2560 B。局部真值所有方法均325/326：`heldout_16`共同停在約0.304假峰，而真值約0.999969；因此本輪絕對精度驗收 `acceptance_passed=false`。

**決定：常駐控制實作有效、保留作後續基礎；該輪不能稱全面驗收成功。** GPU-only與CPU驗分有不同保證，且小模板仍可能CPU更快。

來源：[報告](METAL_RESIDENT_REFINEMENT_zh-TW.md)・[完整紀錄](resident-refinement/report.json.gz)・[數值表](resident-refinement/table.json)・[驗收](resident-refinement/validation.json)・[共同錯峰診斷](resident-refinement/basin-diagnosis.json)・[補充稽核](resident-refinement/audit-summary.json)。

### 7. 有預算多起點 rescue 與首版 Metal coarse NCC（2026-10-13）

**目的／實作：** 對低分候選以固定16個Halton起點短跑、最多4個seeds、有限工作額度重新找吸引域；另將頂層NCC整數統計搬到Metal，CPU double正規化保持。

**資料／時間／記憶體／精度：** 凍結 `958ab79`，90案例×4方法；納入局部驗收的6182 seeds由6181到6182。已知錯峰4 seeds整批CPU2.229→3.777 ms、Metal3.635→7.246 ms，3/4→4/4；GPU額外約18.9 KiB。首輪11個比較超速度門檻；修正呼叫端救援觸發上限為 `min(.8,實際接受門檻)` 後，另輪全圖Src7／6／12通過原門檻，未覆寫首輪失敗。8K RGB已知32候選局部精修Metal16.38 ms，但setup192.68 ms、RSS1188.95 MiB；這不是8K全圖搜尋。首版8192²灰階0°全圖CPU101.648／Metal104.128 ms，Metal約多12.13 MiB RSS，找齊6個。

**決定：採用有預算 rescue＋匹配實際分數門檻；首版粗NCC單獨搬移不足以加速大圖。** 高分週期假峰約0.999992但位置錯8 px仍不觸發，未解決所有共同失敗。

來源：[報告](NCC_RESCUE_AND_METAL_COARSE_zh-TW.md)・[原protocol](ncc-rescue/protocol.json)・[首輪摘要與失敗](ncc-rescue/summary.json)・[跟進protocol](ncc-rescue/protocol-v2.json)・[跟進結果](ncc-rescue/dispatch-summary.json)・[粗搜尋](ncc-rescue/coarse-summary.json)・[saved-data稽核](ncc-rescue/saved-audit.json)。

### 8. 分段量測、Metal 精確金字塔、角度 batch4（2026-10-13）

**目的／實作：** 先辨識大圖CPU金字塔成本，再以整數直接濾波保持像素精確；粗角度最多4張一起提交。來源跨模型當時尚未共用。

**資料／時間／記憶體／精度：** 凍結 `91842dd`、六場景、正反兩block各5次；8192²／256²／0°舊Metal105.28→新12.32 ms，RSS400.81→317.03 MiB；十二物件509.45→490.78 ms，RSS116.27→125.89 MiB。全部完整輸出一致；五有GT場景零FP/FN。mode2可分離版tuning比直接版更快，但金字塔buffer85.25→149.25 MiB，因此正式採mode1。

**決定：採用精確直接金字塔＋batch4作研究組合；中型場景仍由fine主導。** 8K零度只有一張粗map，不能將它的收益歸因角度批次；其六個植入物件為正向scale1，不是任意旋轉真值。

來源：[報告](PIPELINE_STAGES_AND_METAL_PYRAMID_zh-TW.md)・[protocol](pipeline-stages/protocol.json)・[選擇前探針](pipeline-stages/decision-probe.json)・[正式摘要](pipeline-stages/formal-summary.json)・[金字塔tuning](pipeline-stages/pyramid-tuning-summary.json)・[驗證](pipeline-stages/validation.json)・[完整封存](pipeline-stages/raw-results.tar.gz)。

### 9. 同圖跨尺度共享金字塔、Metal fine512（2026-10-13）

**目的／實作：** `SourceFrame` 每圖準備一次；同層ROI角度請求批次GPU旋轉與NCC，Q28 guard／CPU double repair保持原採樣。與resident pose精修是兩個獨立階段。

**資料／時間／記憶體／精度：** 凍結 `2c28ffb`，十場景、160個新process、800次正式計時。六物件舊Metal201.59→96.57 ms；十二476.32→243.80 ms。8K三尺度0°28.75→13.47 ms，RSS500.2→331.5 MiB；8K三尺度全角度306.72→330.81 ms確實退步，但各block仍在10%／2ms容忍內。完整有序輸出均與凍結CPU相同；Release19/19、ASan/UBSan17/17。

**決定：採用共享frame＋fine512作後續基準；不是全面加速或全GPU常駐。** warp後CPU repair、NCC後CPU normalization／選峰仍各有等待；新圖金字塔不能預編譯成固定像素。

來源：[報告](SHARED_FRAME_AND_FINE_METAL_zh-TW.md)・[protocol](shared-fine/protocol.json)・[正式摘要](shared-fine/formal-summary.json)・[驗證](shared-fine/validation.json)・[完整封存](shared-fine/raw-results.tar.gz)。

### 10. 精確 peak tree 與失敗的 coarse SIMD 協作（2026-10-13～14）

**目的／實作：** 32×32 tile winner＋tournament tree只重掃失去最大值的tile；獨立實驗將每個coarse視窗交給一整個SIMD group。

**資料／時間／記憶體／精度：** 凍結 `010c898`，十場景、180個新process、900次正式。僅peak：六物件90.54→81.02 ms，十二248.04→156.96 ms，十二RSS121.6→121.2 MiB；所有輸出精確相同。SIMD-only六物件106.34 ms、十二286.29 ms、8K全角度三尺度634.29 ms，皆比原路徑慢。六物件GPUdevice3.62→16.12 ms、copy仍約0.10 ms，退步不是新增傳輸。Release22/22、ASan/UBSan20/20，289張coarse maps逐值相等。

**決定：採用 `peak_mode=1`；不推薦 `ncc_mode=1`，保留失敗實驗。** 8K512模板凍結baseline兩block10.41／24.59 ms存在明顯波動，新build全關10.53與peak10.50 ms幾乎相同，不把pooled差額當選峰收益。

來源：[報告](COARSE_SELECTION_AND_PARALLEL_zh-TW.md)・[protocol](coarse-next/protocol.json)・[正式摘要](coarse-next/formal-summary.json)・[驗證與不採用門檻](coarse-next/validation.json)・[完整封存](coarse-next/raw-results.tar.gz)。

### 11. 超過100物件模擬（2026-10-14）

**目的／實作：** 不改matcher／rescue，增加固定種子場景與事後最大一對一精度配對，檢查候選膨脹。current為 `4092754`，baseline為 `010c898`。

**資料／時間／記憶體／精度：** Dst3真實紋理102×140，同模板的分離副本，±180°、尺度0.72–1.28、光度／噪聲；4096²的64／128／256與8192×4096的512，另128正向控制。新選峰128／256／512分別1237.46／1804.66／3711.11 ms，RSS396.7／406.4／720.1 MiB，TP全數、FP=FN=0；與baseline有序輸出完全相同。512最大位置0.128 px、角度0.100°、相對尺度0.275%。

**決定：目前組合可處理此類128～512實例，未證明廣泛密集／遮擋場景。** 512同時加大畫布與maxTargets，不是純物件數曲線；也不是512個不同模板。512約108220個粗候選，profile fine旋轉804 ms＋fineNCC1576 ms主導，不能照十二物件的36.7%加速外推。

來源：[報告與實際框選](MANY_OBJECT_STUDY_zh-TW.md)・[固定protocol](many-objects/protocol.json)・[正式results](many-objects/results.json)・[完整封存](many-objects/raw-results.tar.gz)・[封存SHA](many-objects/artifact-sha256.json)。

### 12. CPU／GPU coarse 雙槽重疊與512混合結果（2026-10-14）

**目的／實作：** CPU生成下一批旋轉、Metal計算當前批、CPU消費上一批正規化與峰值；兩槽獨立資源與嚴格生命週期，按原角度序收回。

**資料／時間／記憶體／精度：** 凍結 `4092754`＋同新build同步控制，六場景正反兩block。同步→重疊：六物件82.31→75.09 ms、RSS96.2→97.4 MiB；十二161.01→145.86 ms、RSS120.8→124.0 MiB；128物件1260.06→1227.50 ms、RSS395.3→440.9 MiB；512物件3691.49→3823.28 ms、RSS720.3→770.6 MiB。所有有序輸出、工作量與GT通過。

**決定：功能完成，保持預設關閉。** 512首輪整體慢3.6%，第二block慢12.43%，未過固定性能門檻。另一次相同配置複測同步3664.41→重疊3635.46 ms，僅快0.8%而多47.3 MiB；複測沒有取代正式失敗，不足以支持穩定加速。其他五場景通過效能容忍門檻，亦不等於普遍更快。

六物件host wait9.189→0.474 ms、device4.079→4.221 ms、copy約0.10 ms，支持減少主機空等；**未擷取硬體timeline，尚未量出CPU/GPU實際相交毫秒數，pending不等於GPU kernel彼此並行**。Release23/23、ASan/UBSan21/21；未聲稱LeakSanitizer通過。

來源：[報告](COARSE_OVERLAP_zh-TW.md)・[事先protocol](coarse-overlap/protocol.json)・[正式results](coarse-overlap/results.json)・[正式封存](coarse-overlap/raw-results.tar.gz)・[512獨立複測](coarse-overlap/recheck-512/results.json)・[複測封存](coarse-overlap/recheck-512/raw-results.tar.gz)。

### 13. NCC 空間分塊與相鄰輸出重用（2026-10-14）

**目的／實作：** mode2每thread算四個水平輸出並滑動四像素暫存值；mode3由64threads合作載入最多16×16模板＋23×23來源，計算8×8輸出。整數統計／CPU double正規化不變，支援batch、切段、非同步生命週期，完整C++／Metal源碼保留。

**資料／時間／記憶體／精度：** 凍結 `d60ec04`＋同build mode0，六場景84workers、420正式次；另九形狀單map探針54workers、540正式次。全流程mode0→2：8K291.86→276.74 ms、RSS368.1→369.2 MiB；128物件1239.20→1203.43 ms、397.2→401.8 MiB；512物件3767.99→3635.22 ms、721.0→721.0 MiB。12物件155.93→161.51 ms確實退步。所有完整輸出／工作量相同、GT零FP／FN，既有GPU buffer容量未增加。512 coarse device196.97→120.72 ms、copy2.33→2.42 ms；fine仍主導。單map13×17模板mode2 device2.18×、wall1.29×；257×258且少輸出面反而更慢，mode3也非普遍更快。

**決定：mode0保留通用基準；mode2作本機大圖／大量同模板實例的選項；mode3保留研究。** 兩block所有全流程效能容忍門檻通過，不等於全部變快。小fullres profile最大偏移+69.46%，全流程只引用未profile數據。未做自動選型、硬體cache／DRAM counters或tile＋overlap的效能組合，不能宣稱固定放入L1或套用FlashAttention倍數。Release24項最終皆通過、ASan／UBSan22/22、Metal API／Shader Validation確認啟用，795張map逐位相等；Python harness9/9。

來源：[本輪實作與正式結果](TILE_NCC_zh-TW.md)・[數學與原始文獻](NCC_TILING_AND_OVERLAP_RESEARCH_zh-TW.md)・[事前protocol](coarse-tiling/protocol.json)・[review數據](coarse-tiling/results.json)・[全流程封存](coarse-tiling/raw-results.tar.gz)・[單map數據](coarse-tiling/probe/results.json)・[單map封存](coarse-tiling/probe/raw-results.tar.gz)。

### 14. 原生尺度 GUI、分塊＋重疊組合與有界穩定性（2026-10-14）

**目的／實作：** 將灰階尺度 bank、共享 frame、resident/refine/rescue、座標轉換與跨尺度 NMS 整理成原生 C++ Session，接入 Qt GUI。GUI 新增 CPU／Metal 尺度搜尋、mode2／3、重疊開關、尺度結果與完整執行設定匯出；不需要 Python 子程序。預設保留 CPU 原有流程，Metal mode0、重疊關閉。

**資料／時間／記憶體／精度：** 六場景、六路、72workers、360正式次，凍結 `1cba464` 與 current Python 都使用 Metal，再比較 native mode0／mode2／overlap／combined。Native original→mode2：8K273.46→259.44 ms、128物件1189.79→1149.74 ms、512物件3426.37→3350.39 ms。512再加overlap3352.14 ms，peak RSS789.84→841.85 MiB，沒有額外時間收益。全部有序結果及GT通過，native模式逐值一致；512最大位置0.1281 px、角度0.0995°、尺度0.2744%。小圖combined第2block9.233 ms對original6.896 ms，慢33.90%，效能gate失敗；保留正式失敗。8K原生基準RSS562.77 MiB高於Python current365.93 MiB，原生化並非全面省記憶體。

Metal Shader Validation另發現導數kernel插裝後共享記憶體超過32KiB；修正為64共享槽＋下半lanes暫存器，保持原128-lane數值合併順序，修正前後輸出通過。Release全套26/26後重跑受影響8/8；ASan/UBSan全套24/24後重跑4/4；CPU-only GUI3/3；Python工具19/19。失敗診斷與成功重跑皆封存。

**有界持續測試：** 128場景600.40秒、523次／5次重建，RSS起末中位數337.12→338.27 MiB；6場景兩分鐘、1723次／17次重建，53.12→52.14 MiB。2246次輸出、generation、工作量、buffer容量及async drain皆通過。這是固定圖測試，尚未覆蓋持續新相機影格。

**同版本30分鐘追加驗證：** 依本輪約定對512場景執行1802.26秒、534次／5次重建；RSS起末中位數622.52→634.55 MiB，增加12.03 MiB，低於62.25 MiB事前門檻。完整輸出、async drain、generation、工作量、buffer容量與來源／binary／輸入hash均通過。本輪沒有修改演算法或重新做效能比較，沒有再追加24小時門檻。見 [30分鐘報告](STABILITY_30MIN_zh-TW.md)、[事前規則](stability-30min/protocol.json)、[摘要](stability-30min/summary.json)與[完整封存](stability-30min/raw-results.tar.gz)。

**決定：原生灰階 GUI 可使用；大場景可選mode2，重疊仍選配。** 新的C++ NMS在512場景由Python123.835降至0.604 ms，但不是GPU收益。下一個主要時間成本仍是proposal／fine搜尋；大圖來源副本與Qt解碼配置需另拆量。Fourier與完整RGB全圖GUI、CUDA／跨平台實測、新相機影格串流仍未完成。

來源：[整合與完整報告](GUI_INTEGRATION_AND_STABILITY_zh-TW.md)・[事前protocol](gui-integration/protocol.json)・[六路正式結果](gui-integration/results.json)・[粗搜尋等待診斷](gui-integration/diagnostics.json)・[驗證紀錄](gui-integration/validation.json)・[完整封存](gui-integration/raw-results.tar.gz)・[封存SHA](gui-integration/artifact-sha256.json)。

### 15. 固定尺度、全圖旋轉與無金字塔對照（2026-10-14）

**目的／實作：** 使用實拍原生像素裁切模板，在整張圖找固定尺度物件；比較`scales=[1]`的金字塔與只保留level0的原解析度路徑。新增獨立實驗工具與CPU／Metal契約測試，無runtime或GUI改動；runner／兩個動態庫與前輪binary SHA相同。預先固定相同分數、角域、maxTargets、subpixel及常駐精修；不供位置ROI或GT。

**資料／時間／記憶體／精度：** 六場景、28新程序、84正式次。Metal金字塔→原解析度：小圖全角21.61→370.45ms、RSS28.75→37.11MiB；中圖六物件19.54→331.16ms、36.70→98.70MiB；低對比12.22→150.97ms；細線6.20→42.86ms。四個正常GT場景共有10個合成姿態，兩路均零FP／FN，Metal最大位置0.12120／0.12020px、角度0.25096／0.23759°；沒有召回率收益。小圖CPU42.21→1326.42ms；原repo Src4均24個結果，沒有GT不報準確率。尺度+4%負控制雖高分約0.94473，仍尺度錯3.85%，保留失敗。

**決定：固定尺度保留金字塔作實用基準；無金字塔保留研究。** 所有原解析度速度gate失敗，中圖兩個block與低對比一個block另超RSS門檻。小圖粗角度52→202、粗視窗182,416→11,186,978，減少金字塔準備成本不足以抵銷增加的全圖搜尋。後續優先原圖稀疏方向提案＋完整NCC精修，遮罩FFT／張量模板另作數學研究；尚未實作新提案器或GPU FFT。新契約測試2/2通過，來源／輸入hash及離線摘要重播通過；不宣稱已驗證新相機資料或8K全角無金字塔。

來源：[完整報告與框選結果](FIXED_SCALE_SEARCH_zh-TW.md)・[數學與改善方向](FIXED_SCALE_SEARCH_DESIGN_zh-TW.md)・[事前protocol](fixed-scale/protocol.json)・[正式摘要](fixed-scale/results.json)・[原始封存](fixed-scale/raw-results.tar.gz)・[驗證](fixed-scale/validation.json)。

### 16. 原圖稀疏方向提案＋完整 NCC 精修（2026-10-14）

**目的／實作：** 固定尺度1、原圖全範圍旋轉搜尋；C++分散強梯度點、8方向鄰域回應、CPU按角度平行、稀疏分數提前排除、選峰／分區候選，接既有CPU或Metal常駐NCC精修。不建立金字塔、不旋轉來源整圖、不傳真值ROI；完整自有源碼保留，GUI未新增此研究選項。

**資料／時間／記憶體／精度：** 7場景、32新程序、96正式次，金字塔舊binary保持第15輪SHA，新來源／binary／輸入事前凍結。兩路最終上限統一256；128場景保留至正式測試。Metal金字塔→稀疏：小圖23.87→5.27ms，六物件18.78→11.07ms，低對比12.49→10.49ms，128物件241.65→22.48ms、RSS65.98→64.37MiB。五正常GT場景138個不同合成姿態，兩路均零FP／FN；小圖CPU41.27→1.74ms。Src4均24個但無GT，尺度+4%負控制仍失敗。

**保留的失敗：** 細線6.51→8.51ms，兩block慢31.74%／31.48%，效能gate失敗。128提案事前6px／6°鄰域僅127/128；一個角度誤差6.129°的候選靠後段精修才恢復，不能事後放寬提案gate。128候選已達256上限，簡單背景的成功不代表複雜背景不漏檢。

**決定：保留有效研究選項，不改通用預設。** 新128流程來源準備9.45ms、稀疏掃描3.02ms、精修9.58ms；下一步先研究方向圖建置與CPU／GPU精修分界，仍需更多實拍／背景與候選壓力驗證。Release30/30；ASan／UBSan受影響4/4加最後量化修正後2/2（未啟用LeakSanitizer）；Python5/5。Metal API／Shader Validation確認啟用，128首呼叫／重複輸出通過。完整封存含試跑、源碼、原始worker與失敗，SHA及換目錄摘要重播通過。

來源：[完整報告與實際框選](SPARSE_ORIENTATION_zh-TW.md)・[事前protocol](sparse-orientation/protocol.json)・[正式摘要](sparse-orientation/results.json)・[原始資料／源碼封存](sparse-orientation/raw-results.tar.gz)・[驗證](sparse-orientation/validation.json)・[封存SHA](sparse-orientation/artifact-sha256.json)。

### 17. 稀疏方向回應的等價壓縮（2026-10-14）

**目的／實作：** 將八張方向回應圖改成一張8-bit鄰域方向集合＋共用2048-byte LUT。新增`responseMode=1`，保留0作參考；C++17編譯期分支保持特徵順序與剪枝界，沒有新增GPU kernel或改候選策略。源碼完整。

**資料／時間／記憶體／精度：** 凍結70727a8、新建置八圖、新建置compact三路，8場景、54新程序、162正式次。所有有序JSON候選／最終數值、工作量相同；144正常合成姿態最終零FP／FN，128提案127/128與+4%尺度負控制失敗不變。4096²同建置八圖→compact157.52→144.95ms、RSS510.74→382.73MiB；128物件22.17→21.58ms、63.14→55.85MiB。4096²建圖94.14→67.49ms，但掃描45.08→58.89ms，額外依賴查表並非免費。

**決定：省記憶體研究選項，不全面替換。** 全部速度／記憶體容忍gate通過，但小圖5.19→5.64ms、細線8.09→8.78ms反而較慢。來源表示8N→N，不能說速度快8倍；GUI／金字塔預設不變。Release31/31、ASan／UBSan受影響3/3、Python稽核5/5；新測試85組提案／工作量精確一致，Metal插裝128通過。完整source／binary／input與snapshot SHA、封存換目錄重播通過，初次測試fixture失敗日誌保留。

來源：[實測報告](SPARSE_COMPACT_zh-TW.md)・[數學／agent建議](SPARSE_NEXT_STEPS_MATH_zh-TW.md)・[protocol](sparse-compact/protocol.json)・[結果](sparse-compact/results.json)・[完整源碼與原始封存](sparse-compact/raw-results.tar.gz)。

### 18. 稀疏全CPU與Metal精修對照（2026-10-15整理）

**目的／方法：** 回答目前是否全CPU更快。凍結第17輪`ae0489f`執行期，八圖／compact兩種表示各比較CPU與Metal精修；兩路都用相同CPU稀疏提案，沒有新runtime改動或全GPU搜尋。8重用場景、64新程序、192正式次，兩block正反順序，建立與首次呼叫獨立記錄。

**時間／記憶體／精度：** compact全CPU／混合Metal：小圖1.82／5.47ms，六物件5.62／10.93ms，128物件21.30／21.33ms，4096²132.76／148.13ms。128峰值RSS43.11／55.93MiB，4096²311.31／382.80MiB。原始候選與工作量完全相同，144正常合成姿態兩路皆零FP／FN；CPU最大誤差0.12021px／0.23185°，Metal為0.12119px／0.23170°。128提案鄰域127/128、尺度+4%負控制失敗、無GT實圖24結果保留。

**決定：這條稀疏路徑先採CPU，大圖配compact；128只稱打平。** 其兩block Metal／CPU比分別1.009與0.997，不能訂為穩定勝負或256候選交叉點。4096²只有23提案，來源建圖／搜尋才是主要時間。精修總時間差已量到，copy／wait／device未分開，不宣稱是傳輸主導。GUI預設不變，原密集金字塔建議保留。工具負控制5/5通過；64程序路徑、精度與hash檢查通過；418內容檔封存SHA及換目錄摘要重播通過。前輪C++／Metal測試沒有重報成本輪新跑。

來源：[報告](SPARSE_CPU_METAL_zh-TW.md)・[protocol](sparse-backends/protocol.json)・[全部結果](sparse-backends/results.json)・[驗證](sparse-backends/validation.json)・[完整來源與原始封存](sparse-backends/raw-results.tar.gz)・[封存SHA](sparse-backends/artifact-sha256.json)。

**追加局部診斷：** 同binary、小圖5候選與128物件256候選，Metal各7次；兩例皆一次提交／等待、1023個編碼派工。小圖wall3.979ms，encode0.277、wait3.656（內含GPU區間3.529）、CPU驗分0.040；128 wall9.472ms，encode0.281、wait7.880（內含GPU7.767）、CPU驗分1.299。未拆GPU區間的像素／派工／barrier成本，不能說傳輸或CPU編碼主導。全部局部輸出等於同後端救援入口，未改原正式結果；[診斷摘要與來源](sparse-backends/resident-diagnosis.json)、[原始封存](sparse-backends/resident-diagnosis.tar.gz)保留。當時候選內融合kernel與連續GPU稀疏前段仍為待實驗方向；第19輪已實作GPU前段，第22輪完成灰階融合精修。

### 19. Metal稀疏前段與獨立精修後端（2026-10-15）

**目的／實作：** 回答粗略／稀疏搜尋能否放GPU。完整C++／Metal實作來源梯度LUT、OR擴散、角度／位置稀疏整數加總與3×3局部峰值；保留CPU候選排序／空間輪替與原NCC精修。每批至多64角度、4M位置，峰值溢出重試不截斷，僅讀回峰值與統計；沒有神經網路訓練，Liquid／CfC僅作未來時序預測研究。GPU使用預算方向LUT，CPU仍逐像素atan2，不能將收益全歸因GPU硬體。

**資料／時間／記憶體／精度：** 凍結ae0489f CPU、新CPU、Metal前段＋CPU、Metal前段＋Metal四路；8重用場景、64新程序／192次正式。4096²新CPU142.08→混合CPU15.66ms、RSS311.27→255.26MiB；128物件新CPU23.76→混合Metal11.03ms、RSS43.58→50.21MiB。小圖CPU1.81／混合CPU2.83／混合Metal5.82ms，細線與無GT實圖CPU仍最快。候選順序／整數工作量全等，CPU最終結果逐值等於凍結版；144正常GT姿態四路皆零FP／FN。128提案127/128、尺度+4%負控制嚴格TP0／FP1／FN1、無GT實圖24結果保留。

**決定：大圖／大量物件的可用研究選項，不全面替換。** 小圖慢57%但仍在2ms容忍內，不能以gate通過稱加速；混合Metal的5個小場景速度gate失敗，所有RSS gate通過。GUI與金字塔預設不變。下一個對照應讓CPU也用相同方向LUT（第20輪已完成），再研究減少同步與精修融合。Release32/32、相關ASan／UBSan3/3、Metal57組精確性及Python負控制7/7通過；正式source／binary／input與snapshot SHA未變，631內容檔封存與換目錄摘要逐byte重播通過。

來源：[完整報告](SPARSE_METAL_FRONTEND_zh-TW.md)・[Liquid／固定運算圖研究](SPARSE_GPU_ARCHITECTURE_zh-TW.md)・[protocol](sparse-metal/protocol.json)・[結果](sparse-metal/results.json)・[驗證](sparse-metal/validation.json)・[完整來源與原始封存](sparse-metal/raw-results.tar.gz)・[封存SHA](sparse-metal/artifact-sha256.json)。

### 20. 精確CPU方向LUT與GPU收益歸因（2026-10-15）

**目的／實作：** CPU／Metal共用511²灰階梯度LUT生成器，保留逐像素atan2模式；CPU只改來源量化，擴散／掃描／候選規則不變。CPU建圖仍單執行緒、掃描8threads，不是最佳CPU上限。每非空CPU模型多261121bytes，建模增加約1.7ms；Metal不保留重複host表。演算法不必異構，但現有GPU前段＋GPU精修仍有CPU驗分、救援與NMS，不能用它推論完整全GPU設計的勝負。

**資料／時間／記憶體／精度：** 凍結8e0bd61 CPU、同建置CPU、查表CPU、Metal前段四路，後段皆CPU精修；8重用場景、64程序／192次正式量測。4096²原CPU136.47→查表CPU130.53→Metal15.61ms，Metal對查表CPU快8.36倍、RSS311.50→254.90MiB；128物件22.33→19.23→13.55ms，快1.42倍、RSS43.49→43.96MiB。小圖1.80／1.78／2.85ms，細線查表完整時間反而稍慢。全部候選、工作量、最終CPU結果逐值相同，144正常姿態零FP／FN；提案127/128、尺度負控制失敗與Src4無GT結果保留。

**決定：保留查表CPU選項，大圖Metal收益在相同LUT下仍成立。** 4096²兩block相對查表CPU為8.53／8.00倍，128為1.32／1.65倍；小幅CPU差異有block波動，沒有自動分流或改GUI。所有容忍gate通過不表示全面加速，小圖Metal較查表CPU慢約60%。Release33/33、ASan／UBSan4/4、CPU新測試78組、Metal驗證57組與工具14項通過；645內容檔封存與搬移摘要逐byte重播通過。下一步可分別驗證CPU建圖平行、合併GPU建圖與首批提交、候選內精修融合。

來源：[完整報告與架構界線](SPARSE_CPU_LUT_zh-TW.md)・[protocol](sparse-cpu-lut/protocol.json)・[結果](sparse-cpu-lut/results.json)・[驗證](sparse-cpu-lut/validation.json)・[完整來源與原始封存](sparse-cpu-lut/raw-results.tar.gz)・[封存SHA](sparse-cpu-lut/artifact-sha256.json)。

### 21. GPU來源建圖與首批搜尋合併提交（2026-10-15）

**目的／實作：** mode1將梯度、OR擴散與首批搜尋／局部峰值放入同一command buffer，沒有更動kernel、數學、候選預算、後續批次或溢出策略。每個非空搜尋少一次提交與CPU等待，完整來源保留；CPU模式忽略有效Metal設定，非法值拒絕。

**時間／記憶體／精度：** 凍結de02299、同建置分開、合併三路皆Metal前段＋CPU精修；8重用場景、48程序／144次。小圖2.885→2.787ms，128物件12.140→14.999ms，4K16.084→15.727ms；128兩block比分別1.356／1.351，速度gate失敗，4K一快一慢。128／4K另12程序／36次複查，12812.715→13.594ms仍有一個block失敗，4K15.276→15.034ms。主量測RSS12843.96→43.85MiB、4K254.90→254.75MiB，所有記憶體容忍通過。主／複查候選、工作量、最終CPU結果逐值相同；正常144姿態零FP／FN，既有127/128提案與尺度負控制失敗不變。

**決定：可選實驗，維持mode0預設。** 沒有因省一次等待就推定更快；小幅4K收益不足以掩蓋128回退。CPU複製很小，尚未量GPU時脈／occupancy，不斷言回退機制。Release33/33、ASan／UBSan4/4、Metal58組三路精確比較與工具19項通過；主／複查hash一致，原始資料與來源封存。尚未實作候選內精修融合、全GPU或自動分流。

來源：[完整報告](SPARSE_METAL_MERGE_zh-TW.md)・[protocol](sparse-metal-merge/protocol.json)・[主要結果](sparse-metal-merge/results.json)・[獨立複查](sparse-metal-merge/recheck-results.json)・[速度失敗細節](sparse-metal-merge/validation-details.json)・[完整來源與原始資料](sparse-metal-merge/raw-results.tar.gz)。

### 22. 灰階GPU精修融合與群組內統計（2026-10-15）

**目的／實作：** 前段固定分開提交；一個256執行緒群組完成單候選NCC、導數、阻尼GN、試步及停止。1024像素tile、加總順序、迭代／救援預算與CPU驗分保留，中間統計重用不到10KiB群組暫存。灰階模板≤16384像素、每批≤512候選；mode1選配，mode0原實作不變，完整來源保留。

**時間／記憶體／精度：** 凍結60a8936原Metal、同建置CPU／原Metal／融合四路，64程序／192次。八場景融合比原Metal完整時間快約12%～50%；128 12.07→10.31ms、RSS49.63→48.64MiB，Src4 17.69→8.78ms。CPU比較：128 12.41ms／43.95MiB，4K CPU15.37ms／254.80MiB、融合16.55ms／319.51MiB。候選／工作量與最終融合輸出逐值相同，144正常GT姿態零FP／FN；尺度負控制TP0／FP1／FN1與Src4無GT24結果保留。

**獨立複查／決定：** 128與4K另16程序／48次。128 CPU／原Metal／融合11.43／11.65／10.66ms，兩block融合仍較快；4K15.35／18.91／16.46ms，CPU仍較快且省約64.7MiB。融合可改善既有Metal精修，128可手動採用；多數少候選場景仍用CPU。融合相對CPU有8個主量測速度配對失敗，4K主／複查共4個記憶體配對失敗；通過容忍不等於加速，不作自動分流或改GUI。

**驗證／歸因：** Release34/34、ASan／UBSan6/6、Metal驗證59次及工具30項通過。原30次迭代dispatch1023→1，128提交／等待仍1→1，GPU時間8.492→6.962ms；少派工並非少傳輸的證明。128融合對CPU的精修階段差小於完整時間差，保留前段時間變異的限制。初次Session測試預期修正及失敗log保留；本輪無新GUI、RGB全圖或30分鐘持續測試。

來源：[完整報告](POSE_FUSED_zh-TW.md)・[protocol](pose-fused/protocol.json)・[主結果](pose-fused/results.json)・[複查](pose-fused/recheck-results.json)・[失敗與檢查細節](pose-fused/validation-details.json)・[完整來源與原始資料](pose-fused/raw-results.tar.gz)。

### 23. 灰階SIMD128精修與群組內同步優化（2026-10-15）

**目的／實作：** 在第22輪融合kernel上，從256改128執行緒，讓導數取樣使用整個群組；保留原像素累積與合併樹，最後五層用SIMD shuffle。編譯後共享暫存9536→6976bytes，單tile評分／導數barrier皆10→4；每批提交／等待仍一次，來源buffer與輔助workspace容量不變。mode2手動選配，完整來源保留，灰階≤16384像素、32-wide SIMD、每批≤512候選。

**時間／記憶體／精度：** 凍結e57cd8f fused256、同建置CPU／fused256／SIMD128四路，8場景64程序／192次。SIMD128完整時間比同建置fused256縮短5.5%～25.7%，所有block同方向；128 10.52→8.56ms，精修階段8.43→6.40ms，RSS49.35→48.81MiB；4K16.62→15.70ms、RSS319.61→319.80MiB。CPU比較：12813.78ms／43.59MiB，4K15.38ms／254.84MiB。新舊GPU最終結果逐值相同、正常144GT姿態零FP／FN；尺度負控制TP0／FP1／FN1、Src4無GT24結果及提案127/128限制保留。

**複查／決定：** 小圖、Src4、128、4K另32程序／96次；128 fused256／SIMD128為10.37／8.55ms，4K16.57／15.78ms。128可手動優先選mode2，CPU／SIMD精修本身主量測8.72／6.40ms、複查8.89／6.42ms。Src4完整時間僅比CPU少0.21～0.25ms，精修本身仍慢，前段時間差參與收益；小圖及4K少候選仍優先CPU。SIMD相對CPU主量測6個速度配對失敗，4K主／複查共4個記憶體配對失敗。沒有自動分流或GUI改動。

**驗證／歸因：** Release35/35、相關ASan／UBSan7/7、Metal API／Shader Validation119次、工具31項通過。新增10種模板形狀兩驗分模式，mode1／2姿態、分數、迭代、評估次數與停止狀態逐值相同。128 GPU區間7.238→5.188ms，提交／等待仍1→1，支持群組內優化，不是少傳輸證據；完整CPU／GPU差額不能全部歸因精修。主／複查換目錄重播與內容hash皆通過，開發編譯／插樁容量測試失敗及修正log保留。子agent遇服務額度中斷，主agent完成，不聲稱獨立agent最終審查。本輪無新RGB、全GPU或30分鐘驗證。

來源：[完整報告](POSE_SIMD_zh-TW.md)・[protocol](pose-simd/protocol.json)・[主結果](pose-simd/results.json)・[複查](pose-simd/recheck-results.json)・[失敗與檢查細節](pose-simd/validation-details.json)・[來源與原始資料](pose-simd/raw-results.tar.gz)。

### 24. CPU／Metal共用精修影像與8K擴充（2026-10-15）

**目的／實作：** 核心持有page-aligned float32頁面，CPU驗分與Metal no-copy buffer共用；Metal deallocator保留owner，呼叫端可先釋放原圖。新增可選`poseStorageMode=1`與獨立容量統計，原API、模式與GUI預設保留，所有pose Metal kernel hash不變。另修正精修後關閉精修的保留容量診斷，最終版本重新跑全部測試與比較。

**時間／記憶體／精度：** 凍結06eecd6 SIMD128、同建置CPU／複製／共用四路，8場景64程序／192次。4K複製／共用15.63／15.37ms、RSS 319.85→255.80MiB；128 8.67／8.55ms、RSS 49.34→44.41MiB。正常主GT144姿態零FP／FN，GPU最終輸出逐值相同；尺度負控制與提案127/128限制保留。

**複查／決定：** 128、4K及新8K擴充另24程序／72次。8K複製／共用49.82／49.78ms，RSS 1202.60→946.57MiB、24/24零FP／FN；由4K六物件2×2拼接，非新獨立真實場景。共用相對複製所有block性能容忍通過，但速度有快有慢；本輪主要收益是大圖省記憶體。相對CPU仍3個主速度失敗，小圖與少候選仍先CPU。需Metal精修時可手動選共用；不新增自動routing。

**驗證／歸因：** Release37/37、ASan／UBSan10/10、Metal API／Shader Validation共用精修119次、9組灰階RGB生命週期對照、工具34項通過。容量用host+gpu−shared避免重複計數，另量RSS；建立時間與暖機時間分開。保留初次編譯失敗、修正前全套研究及最終重測；主／追加換目錄重播通過。沒有新RGB全圖、GUI、串流或30分鐘soak；服務額度使子agent不可用，主agent完成，不宣稱獨立agent最終審查。

來源：[完整報告](POSE_SHARED_STORAGE_zh-TW.md)・[protocol](pose-shared/protocol.json)・[主結果](pose-shared/results.json)・[複查／8K](pose-shared/recheck-results.json)・[檢查細節](pose-shared/validation-details.json)・[完整來源與原始資料](pose-shared/raw-results.tar.gz)。

### 25. 直接8-bit初始化與CPU／Metal暫存移除（2026-10-15）

**目的／實作：** 新增`fpm_pose_create_u8`，直接從有stride的HWC灰階／RGB轉入最終自有float影像，來源可反色；Session以`poseInputMode=1`選用。沿用相同中心化與求解，8-bit入口不需再掃描有限值；原float驗證保留。呼叫端可釋放原圖，storage與lazy規則不變，所有Metal shader hash不變。

**時間／記憶體／精度：** 凍結54ae7b6共用Metal、同建置CPU float／直接、Metal float／直接五路，80程序／240次；前段均Metal。4K Metal float／直接初始化30.36／19.71ms，RSS 255.77→191.80MiB，暖機15.95／15.93ms。CPU亦省約64MiB。正常144姿態零FP／FN，同後端新舊輸出逐值相同；尺度負控制、Src4無GT與提案127/128限制保留。

**複查／決定：** 六物件、128、4K、8K另40程序／120次，8K是沿用4K拼接、24同模板實例。Metal初始化125.84→81.29ms、RSS 946.59→690.53MiB，暖機51.29／51.40ms。Metal直接對float性能容忍全過；CPU直接主量測六物件block0與128 block1暖機失敗，複查未重現。Metal直接對CPU直接主6個／複查1個速度失敗保留，所有記憶體容忍通過。可手動用於降低大圖初始化成本，沒有通用暖機加速或自動分流結論。

**驗證／歸因：** Release39/39、ASan／UBSan12/12、新入口CPU／Metal各10組精確對照、Metal119次共用精修與工具42項通過。新舊入口常駐容量與求解工作相同，變化來自初始化暫存與掃描；暫存容量不與RSS相加。主／追加／試跑摘要搬移重播逐byte一致。兩協作agent遇服務額度中斷，由主agent補全及驗證；沒有獨立agent最終審查、新GUI、RGB全圖、串流或新soak。

來源：[完整報告](POSE_U8_INPUT_zh-TW.md)・[protocol](pose-u8/protocol.json)・[主結果](pose-u8/results.json)・[複查／8K](pose-u8/recheck-results.json)・[失敗與驗證](pose-u8/validation-details.json)・[來源與原始資料](pose-u8/raw-results.tar.gz)。

## 人工審查時應保留的限制

1. **時間範圍先對齊。** 固定姿態score、局部refine、全圖proposal、含rescue全流程、setup／first各不相同；多數暖機表排除解碼與一次學習，但shared frame每次prepare已計入。原CPU／GPU勝負只能在同輪、同工作量列內比較。
2. **精確一致不是絕對準確。** coarse/fine整數路徑能逐值相同；pose FP32與CPU參考容許誤差並由GT驗證。325/326共同錯峰、高分週期錯峰、裁切2/32必須與成功並存。實圖沒有校準GT時不能把數量相同稱準確率100%。
3. **RSS與buffer不相加。** shared memory有重疊；16MiB通常只指局部輔助workspace，不包含source、template、score map、Python、driver或擴容瞬時副本。某些早期階段沒有標準化RSS比較，本頁不補推估。
4. **效能 gate 與正確性 flag 分開。** `integrity_passed`、`detection_all_passed`均true不代表效能全過；overlap512就是反例。通過 `max(10%,2ms)`容忍也可能實際變慢。保留profile干擾、block波動與慢樣本。
5. **版本必須查hash。** 早期JSON的`git_head`可能只是當時未提交修改的基底，需一起核對`source_sha256`、library SHA與build provenance；不能僅用HEAD聲稱重現同binary。封存中absolute paths是歷史來源，離線replay需明確指定解壓後worker root。
6. **仍缺的覆蓋。** 多物件目前只有單模板、多個分離實例；遮擋、透視、真實相機尺度、跨類別模板庫、長時間串流、功耗與跨硬體交叉點未全面驗證；第14輪補10分鐘／2分鐘固定圖測試，同版本再完成512場景30分鐘驗證，仍不等同新影格串流覆蓋。RGB證據主要是已知候選局部評分／精修，沒有本頁可證明的完整RGB全圖GUI。

閱讀時先看每階段報告的「比較方法／限制」，再查相鄰JSON與protocol。所有實驗源碼入口在 [tools](../tools)、[C++／Metal實驗庫](../experiments)、[核心](../src)與[測試](../tests)；GUI及原核心的建置與授權見 [README](../README_zh-TW.md)。

## 26. 自適應矩形投影 NCC 上界

[完整報告](ADAPTIVE_NCC_zh-TW.md)、[事前協定](adaptive-ncc/protocol.json)、[全部摘要](adaptive-ncc/summary.json)、[驗證](adaptive-ncc/validation.json)、[原始封存](adaptive-ncc/raw-results.tar.gz)。

- C++自有矩形樹、global／block保守上界、原NCC存活驗分；不改fine／subpixel或預設。
- 384新程序、1920正式呼叫，1408診斷map、20,298,904視窗沒有誤刪或存活分數差異；合成138姿態零FP／FN，Src4無GT只報一致。
- 結構256²模板19.27→7.72ms；隨機模板9.29→19.60ms；完整128物件544.62→740.16ms。83個速度容忍失敗、0個RSS容忍失敗，完整保留。
- Release40/40、相關ASan／UBSan3/3；Apple不支援LeakSanitizer的首次啟動失敗留存。凍結3793e9e與一般CLI另72程序，序列化輸出及時間容忍通過。
- 採用決定：保留研究選項，不換金字塔粗搜尋、不直接搬Metal。下一步fine相鄰視窗重用；這輪不宣稱全GPU、RGB或4K／8K新增成果。

## 27. Metal fine相鄰窗口統計重用

[完整報告](FINE_NCC_REUSE_zh-TW.md)、[事前協定](fine-reuse/protocol.json)、[全部摘要](fine-reuse/summary.json)、[驗證](fine-reuse/validation.json)、[原始資料與來源快照](fine-reuse/raw-results.tar.gz)。

- 新mode1將總和／平方和改成行滑動統計再垂直加總，點積、旋轉、修補與搜尋契約不變。同NCC command的兩個encoder，每批沒有額外CPU等待；行workspace有16MiB上限，高瘦模板可能多拆批。
- 11種孤立工作量與6個原生全流程場景，80串行新程序／620正式呼叫；365,344個CPU參考窗口逐位相同，提案／結果與凍結版精確一致，既有666合成姿態零FP／FN。
- 128／512原版→共享1152.31→1181.34ms／3372.58→3424.14ms；8K268.53→267.37ms，但fine GPU統計變慢，不能據總時間聲稱kernel加速。高瘦1×1024×291為1.070→2.441ms、2→4提交；共4個性能容忍失敗、0個RSS容忍失敗。
- Release41/41、相關ASan／UBSan9/9、Metal API／Shader Validation、Python binding2項與工具7項通過。獨立agent審查完成，正式前移除暖機紀錄累積的RSS干擾，正式後摘要失敗處理修正與原始工具均保留。
- 保持fine模式0與GUI預設；下一步B2在單kernel共用相鄰點積載入，尚未實作。此輪不增加獨立真實場景、RGB、全GPU、Windows或長時間穩定性證據。

## 28. Metal fine SIMD strip載入重用

[完整報告](FINE_NCC_STRIP_zh-TW.md)、[事前協定](fine-strip/protocol.json)、[完整摘要](fine-strip/summary.json)、[驗證](fine-strip/validation.json)、[完整來源與原始封存](fine-strip/raw-results.tar.gz)。

- 每map由49個SIMD群改為14群，各群算3／4相鄰輸出，lane負責模板行片段；新增chunk32／16模式，來源及模板載入重用，仍做49個完整整數點積。沒有row pass、額外全域／threadgroup暫存、encoder或CPU等待。
- 12種孤立工作量、6個全流程，120串行新程序／960正式呼叫；585,648個CPU窗口比較逐位相同，提案／結果與凍結版精確一致，既有666合成姿態零FP／FN。
- 128／512原版→strip32為1170.67→915.60ms／3422.74→2619.62ms；fine GPU統計379.96→135.27ms／1221.29→430.74ms。RSS411.23→411.22MiB／797.02→802.37MiB，顯式buffer容量與工作量相同。8K少候選262.73→264.66ms，極小ROI稍慢。
- 新模式相對原版／凍結版時間容忍全通過，chunk16對chunk32另有5個孤立block時間失敗；所有RSS容忍通過。chunk32可手動優先試，不宣稱普遍最快或改GUI預設。
- Release43/43、相關ASan／UBSan11/11、兩種strip Metal API／Shader Validation、Python binding2項、工具9項通過。Agent完成kernel與測試，後續複查及量測委派中斷，由主agent補全；不宣稱獨立agent最終複查完成。無新RGB／CUDA／全GPU／soak證據。

### 29. 原始fine角度圖精確重用與旋轉同步診斷（2026-10-16）

**目的／實作：** 比較完全相同ROI請求的重複率及CPU修補／GPU等待。加入最多65536 key的觀察器；真正加速使用同候選、同層、同pass上一輪角度圖，移動ScoreMap保留原始分數與選峰順序，原Metal kernel逐byte未變。新增C ABI、C++與Python選項，預設關閉。

**證據：** [完整報告](FINE_ANGLE_REUSE_zh-TW.md)、[正式協定](fine-angle-reuse/protocol.json)、[摘要](fine-angle-reuse/summary.json)、[驗證](fine-angle-reuse/validation.json)、[原始來源及所有資料](fine-angle-reuse/raw-results.tar.gz)。九設定四路兩block，72程序／312次；原有666獨立GT姿態零FP／FN（重複設定共1312個設定×姿態）。Release44/44、相關ASan／UBSan8/8、Metal驗證、新舊binding各2項與工具11項通過。

**採用／失敗：** 6／128／512完整次像素約32%／33%／39%加速。原快速模式無命中，仍保留第28輪設定。512主測RSS相對關閉高88.20MiB、三個配對未過門檻；四程序記憶體複查差−1.55／+1.30MiB，未重現，沒有抹除主測失敗。128關閉功能主測一個block慢16.30%，另四程序暖機複查約慢1.4%通過。容量不能取代RSS；512不作無條件推薦，保持手動選項。

**下一步：** 同command暫定warp＋NCC、CPU以像素差修補整數統計，可另行驗證能否少一次等待。本輪只減少重複圖與提交次數，尚未實作此融合或新CPU／GPU重疊。

### 30. 細搜尋旋轉與NCC提交融合（2026-10-16）

**第30輪：多候選金字塔細搜尋可優先試`fineFused=true`，搭配fine mode2與適用時的角度重用。** [旋轉與NCC提交融合](FINE_FUSED_SUBMISSION_zh-TW.md)每批由兩次提交／等待減為一次，保留CPU精確整數統計修補。54程序／162次，九設定三路兩block，全部有序結果一致、所有事前性能容忍通過；128／512快速設定914.20→834.33ms／2616.86→2388.29ms，完整次像素4352.49→4009.85ms／15827.85→14718.34ms。8K少候選250.38→254.78ms稍慢，因此仍是手動選項，不宣稱全面最快。512完整RSS中位數786.66→793.66MiB，所有顯式容量相同；這不撤回第29輪88MiB失敗。Release49/49、相關ASan／UBSan8/8與Metal驗證通過。獨立512診斷以實際位址確認：已釋放的96MiB讀圖RGB暫存仍可能駐留，與pose大配置重疊，解釋新診斷的大部分差距；無法追溯證明舊88MiB每一byte的來源。讀圖後強制回收無效，原紀錄保留。

[事前協定](fine-fused/protocol.json)、[完整摘要](fine-fused/summary.json)、[驗證](fine-fused/validation.json)、[原始資料與完整來源](fine-fused/raw-results.tar.gz)。單command先Q28旋轉＋NCC，CPU對實際變動byte精確修正S/Q/P；密集變動／repairAll退路及原有模式保留。changed不是guard數；combined GPU時間不能與舊單獨NCC時間直接相除。量測後只修正分析工具把GPU缺值-1相加的問題，原工具與摘要保留，所有headline數值／判定相同。512記憶體原因另以隔離診斷追查，不混入正式速度。

### 31. 灰階直接讀取與程序記憶體（2026-10-16）

**第31輪已改善灰階讀圖的暫存記憶體。** [直接讀取gray8](GRAY_LOAD_MEMORY_zh-TW.md)已套用native runner與GUI；彩色／調色盤／16-bit沿用舊轉換。512快速／完整次像素程序RSS中位數786.16→687.02MiB／791.99→697.28MiB，約省99／95MiB，讀圖約快20–23%。18程序／36次有序結果與算法容量一致；新路徑各對照通過，但舊讀法控制的一筆repeat慢48.34%，整體性能門檻為false並保留。完整次像素repeat約慢1.42%，沒有NCC加速主張；GUI記憶體未另量測。算法推薦仍沿用第30輪。

[事前協定](gray-loader/protocol.json)、[正式摘要](gray-loader/summary.json)、[驗證](gray-loader/validation.json)、[全部來源與原始資料](gray-loader/raw-results.tar.gz)。讀圖與算法時間分列；50項Release、最終loader／GUI重測及loader ASan通過，23項harness與17項診斷工具測試通過。

### 32. 慢樣本重現與多尺度精修記憶體組合（2026-10-16）

**第32輪：慢48%未重現，精修配置合用可再省約37%記憶體。** [追查與512物件整合報告](RUN_VARIABILITY_POSE_MEMORY_zh-TW.md)：12程序／48次交錯對照未見可重現退化；原慢樣本97.92%的額外時間在前段搜尋，結果／工作量相同，但缺少當時系統紀錄，根因仍未確認，原失敗保留。另16程序／32次比較已有`poseInputMode=1`＋`poseStorageMode=1`與新多尺度流程的組合：快速RSS689.47→431.02MiB、完整次像素697.88→440.09MiB；repeat時間−0.21%／+0.56%，有序結果相同、512個GT零FP／FN，所有事前容忍通過。可手動搭配fine mode2／融合／適用時角度重用，本輪residentMode保持0；沒有NCC新加速、預設或GUI選單變更。

新增可關閉的C++兩點程序資源診斷；48次重現與32次配置比較採各組兩個新程序、正反順序、無暖機，保留所有first／repeat。原始48%慢樣本未重現，不能追認為已確定OS／溫度原因，也不撤銷第31輪性能失敗。四路2×2配置只整合第24／25輪已實作模式；不改搜尋或Metal kernel。Release51/51、probe ASan／UBSan1/1、harness28與memory工具25項通過。完整[協定與原始封存](run-diagnostics/raw-results.tar.gz)、[驗證紀錄](run-diagnostics/validation.json)保留；相同合成512場景不視為新增實景準確度證據。

### 33. 多尺度精修組合、新場景與缺口審查（2026-10-16）

**第33輪：多尺度仍保留原resident0，優先處理新增的尺度精度缺口。** [三路組合與新場景比較](RESIDENT_MULTISCALE_zh-TW.md)完成48程序／96次，前段、有序結果及路徑一致，但準確度與整體性能未全過：7個性能配對失敗、RSS全過；512快速原版／SIMD完整時間2342.54／2402.90ms，精修206.18／240.13ms，不升mode2為通用預設。新增小模板12個都定位正確，5個因尺度偏小超過2%而strict失敗；12條固定GT幾何的NCC尺度曲線都偏好較小尺度，另有遮擋錯方向與缺候選。這些共同失敗完整保留，不能用舊512全過推論所有場景可靠。後續先做取樣／抗混疊精度實驗；完整[實驗缺口審查](EXPERIMENT_GAPS_zh-TW.md)另列新影格、真實GT、CPU公平比較與統計限制。

同一第32輪binary，不改native算法；新工具、兩組24個不同合成姿態、降質與空場景完整保留。原／融合／SIMD的前段與最終有序結果都exact，不代表準確：小模板strict7/12、降質strict4/12；獨立CPU＋float64尺度掃描1032點支持保存影像的NCC目標偏差。相關C++／Metal5/5、harness22與scene4項通過，沒有新soak／跨平台主張。[protocol](resident-multiscale/protocol.json)、[summary](resident-multiscale/summary.json)、[失敗](resident-multiscale/failures.json)、[原始封存](resident-multiscale/raw-results.tar.gz)與[驗證](resident-multiscale/validation.json)均保留。
