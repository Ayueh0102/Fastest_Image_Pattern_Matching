# 第32輪：48%慢樣本追查與512物件精修記憶體整合

2026-10-16，基準`c3c59ee`，Apple M4 Pro、48 GiB。這輪分開處理歷史慢樣本與下一步配置整合：新增可關閉的程序資源診斷，並將第24／25輪已有的共用影像、直接8-bit入口接到目前多尺度512物件比較。沒有更換搜尋、求解、旋轉、NCC或Metal kernel，也沒有把既有模式當成新算法。

## 那筆慢48%的時間在哪裡

保留第31輪全部原始失敗：`many512_pose/current_legacy/block1/repeat`為3536.796ms，對凍結版同block的2384.252ms，慢48.3399%。同建置另一block的repeat為2373.064ms，沒有同等退步。

原始資料可確認：

- 多出的1152.544ms中，前段proposal多1128.515ms，約97.92%；scale=1模型的fine NCC、粗正規化、粗等待與fine旋轉四個互不重疊區間，合計占總額外時間約76.67%。
- CPU執行的正規化牆鐘時間、GPU device時間及等待同時增加。fine fused device多264.451ms、wait多497.578ms；兩者重疊，不能相加。wait減device也不能稱為傳輸時間。
- 全部12筆舊first／repeat的512個有序匹配與1301個有序提案完全相同；非時間工作計數及同階段容量相同。慢樣本是repeat，pose建立與轉換暫存皆為0，讀圖在計時區間外。
- 該程序peak RSS786.94MiB，相較凍結同block788.41MiB、同建置另一block785.38MiB，沒有明顯配置容量增加。

所以沒有證據顯示多搜了一輪、重建精修影像、結果改變或重複解碼。但當時未保留程序CPU使用量、page fault、排程、功耗或溫度時間線，仍不能在「系統干擾」與「罕見程式／runtime延遲」間作確定歸因。原始計時不是可以任意刪除的錯誤值。

## 重現與診斷方式

第一階段六組：凍結版legacy／direct、現行版legacy／direct、現行診斷版legacy／direct。兩個新程序block採正反順序，每程序首次＋三次repeat、無暖機，共12程序／48次。凍結binary先對第31輪protocol驗hash；所有新舊source、binary與輸入在事前固定，結束再驗。每組只有兩個獨立程序；同程序repeat不是新的獨立場景或程序樣本。

新增`run_resource_diagnostics=true`只用於first／repeat模式，不接受warmup或soak。完整C++標頭`tools/run_resource_probe.hpp`在`Session.run`前後各取一次steady clock、getrusage與Apple主機CPU ticks；JSON保留user／system CPU時間、minor／major faults、voluntary／involuntary switches。程序與host各有availability，缺值為−1。host ticks涵蓋全機，不能當成本程序或GPU使用率；多執行緒CPU時間亦可能大於牆鐘時間。

診斷外層wall含run返回前析構與邊界開銷，稍寬於算法`total_ms`，但不含解碼、Session建構、JSON或結果比對。停用時不取clock或系統計數，仍有固定本地物件與分支，不能宣稱字面零指令開銷。計數倒退、溢位與host32-bit wrap有明確檢查；Mach send right每次釋放。沒有高頻採樣、強制GC、刻意注入背景負載或GPU profiler。

| 組別 | 首次中位數 ms | repeat中位數 ms | repeat範圍 ms | peak RSS中位數 MiB |
|---|---:|---:|---:|---:|
| 凍結legacy | 2563.05 | 2398.14 | 2352.14–2464.36 | 789.82 |
| 凍結direct | 2533.79 | 2357.43 | 2333.24–2392.71 | 696.69 |
| 現行legacy | 2527.23 | 2354.57 | 2324.30–2393.98 | 790.59 |
| 現行direct | 2589.13 | 2390.34 | 2350.12–2402.09 | 698.88 |
| 診斷legacy | 2557.05 | 2399.32 | 2372.95–2410.82 | 791.24 |
| 診斷direct | 2516.71 | 2404.66 | 2383.48–2430.53 | 697.60 |

48次全部有序結果、工作量、容量與路徑檢查通過，512個既有GT零FP／FN；最大位置誤差0.12810px、角度0.099514°、相對尺度0.00274433（約0.2744%）。所有逐block時間與RSS容忍通過，沒有重現48%退步。現行對凍結的repeat中位數差，legacy為−0.96%／−0.65%、direct為+1.90%／−0.20%；沒有可重現的同路徑性能退化。

診斷對現行同讀法的block repeat中位數差為legacy +2.55%／+0.75%、direct +0.48%／+0.72%，都在容忍內，但不能把跨程序差額全部當成兩次資源快照的成本。12次診斷repeat的major faults全為0，minor為0–334，程序CPU user約1012–1046ms、system86–100ms；沒有對應歷史異常的同時紀錄。外層wall比total多約0.0044–0.0105ms，不代表整個診斷零擾動。

**判讀：目前較符合偶發執行時間波動，未發現可重現的程式退化；歷史根因仍未確認。** 不能將這輪未重現改寫成已證明OS／溫度造成，也不能宣稱程式絕無罕見延遲。保留原失敗與新增診斷，若後續再次發生，可再對慢call做排程／Metal時間線檢查。

## 下一步：把已有精修配置接到多尺度流程

第二階段固定直接灰階讀圖、fine mode2、角度重用、旋轉＋NCC融合、Metal residentMode0、搜尋範圍及救援預算，僅比較2×2組合：input0/storage0、input1/storage0、input0/storage1、input1/storage1。兩設定為同一8192×4096合成512場景的快速與完整次像素模式；兩block正反順序，每程序首次＋一次repeat，共16程序／32次。資源診斷關閉；不是增加32個獨立真實場景。

來源＋模板14280pixels的float影像邏輯容量：

```text
F = 4 × (8192 × 4096 + 14280) = 134274848 bytes ≈ 128.054 MiB
input1：直接gray8轉入最終自有float儲存，首次少一份F暫存；repeat本來就是0。
storage1：CPU驗分與Metal讀同一份持有的頁面，各影像單獨page-round。
unique_image_bytes = host_image_bytes + gpu_image_bytes − shared_image_bytes
```

容量預測與實際RSS分開驗證，不能把host、gpu、shared重疊相加。兩種儲存的非影像workspace保持相同；共用頁面與direct入口已在第24／25輪完整實作，本輪只驗證與新多尺度前段組合是否仍正確、省記憶體且沒有明顯速度損失。

下表為各組兩個獨立程序的中位數。初始化屬於首次總時間的一部分，不可再相加。

| 設定 | 組合 | 首次 ms | repeat ms | pose建立 ms | peak RSS MiB |
|---|---|---:|---:|---:|---:|
| 快速 | 原配置 | 2564.50 | 2390.26 | 69.90 | 689.47 |
| 快速 | 直接8-bit | 2543.09 | 2385.99 | 53.56 | 560.86 |
| 快速 | 共用影像 | 2505.48 | 2402.97 | 61.11 | 558.49 |
| 快速 | 兩者合用 | 2488.23 | 2385.20 | 40.60 | 431.02 |
| 完整次像素 | 原配置 | 14755.22 | 14641.72 | 66.90 | 697.88 |
| 完整次像素 | 直接8-bit | 14722.81 | 14684.92 | 50.35 | 569.04 |
| 完整次像素 | 共用影像 | 14715.15 | 14606.15 | 76.55 | 570.79 |
| 完整次像素 | 兩者合用 | 14733.69 | 14723.04 | 42.55 | 440.09 |

合用後快速省258.45MiB（37.49%），完整次像素省257.78MiB（36.94%）。repeat分別−0.21%／+0.56%，接近持平；首次pose建立69.90→40.60ms、66.90→42.55ms。各自單獨模式的容量減少也符合公式；shared單獨的完整次像素初始化有較慢中位數，沒有宣稱所有模式都加速。合用的主要收益是較低峰值記憶體，不能將首次收益稱為穩態NCC加速。

32次所有有序proposals／matches、非時間搜尋工作及frame容量完全相同；pose非影像workspace相同，image與temporary容量依預定公式改變。共用模式的頁對齊image為134283264bytes，原兩份unique為268549696bytes；直接入口的first暫存134274848→0，repeat兩者皆0。這些是不同生命期的容量，實際RSS下降不必恰好等於容量相加。

每次512個GT零FP／FN；快速最大誤差與第一階段相同，完整次像素為0.12733px、0.098995°、相對尺度0.00278198（約0.2782%）。**兩階段共28程序／80次，正確性、路徑、事前時間與RSS容忍均通過，無性能失敗被刪除。** 事前門檻仍是逐block／phase時間max(10%,2ms)、RSS max(10%,16MiB)；通過容忍不等於統計上證明更快。

可在已測多尺度512研究流程優先搭配`poseInputMode=1`與`poseStorageMode=1`，Metal residentMode維持本輪0；原核心和GUI預設不變，沒有新增GUI選單。這些選項早已存在，本輪交付的是可重現的組合比較與診斷。native JSON對應：

```json
{
  "pose_input_mode": 1,
  "pose_storage_mode": 1,
  "pose_resident_mode": 0,
  "fine_ncc_mode": 2,
  "fine_angle_reuse": true,
  "fine_fused": true,
  "gray_load_mode": "direct_gray8",
  "run_resource_diagnostics": false
}
```

這是既有job的選項片段；完整輸入、尺度、門檻、批量及預算見封存jobs，不能省略其餘配置當成完整job。固定尺度稀疏流程的SIMD128建議仍依第23～25輪，不把本輪當成residentMode2再次驗證。

## 驗證與重現

Release51/51（含GUI流程）通過；新增程序probe的ASan／UBSan1/1通過，detect_leaks=0，不是長時間無洩漏證明。分析harness28項、記憶體診斷工具25項通過，包含hash／缺值／工作量／容量／錯誤配置等負控制。小圖確認診斷開關有序輸出exact、關閉不產生resources欄位、開啟拒絕warmup／soak；既有memory工具亦支援新舊runner布局、從選定來源複製probe標頭，build-only通過，沒有將重型記憶體取樣混入速度數據。

另一名agent從raw獨立重驗234份受測來源、binary／輸入／worker hash、有序結果、工作量、容量與GT，兩階段結論一致。封存source全新建置後，probe、灰階讀圖、GUI流程3/3通過；三種小圖入口有序輸出一致，搬移後兩份摘要重算逐byte一致。完整源碼、凍結版源碼與binary、輸入、28組job／raw／stderr、原48%慢樣本與測試log皆封存。這輪沒有新的相機串流、RGB全圖、CUDA、Windows／Linux實機、功耗或長時間soak證據；兩種設定仍共用一張合成同模板512實例圖，不能當成實景泛化保證。

[重現協定](run-diagnostics/reproduction-protocol.json)、[重現摘要](run-diagnostics/reproduction-summary.json)、[配置協定](run-diagnostics/pose-memory-protocol.json)、[配置摘要](run-diagnostics/pose-memory-summary.json)、[歷史慢樣本分析](run-diagnostics/historical-forensics.json)、[驗證與封存hash](run-diagnostics/validation.json)、[完整來源與原始證據](run-diagnostics/raw-results.tar.gz)。

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake -S cpp-core -B build/run-diagnostics \
  -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_BUILD_TESTS=ON \
  -DFPM_BUILD_GUI=ON -DFPM_WITH_OPENCV=ON
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build build/run-diagnostics -j 6
ctest --test-dir build/run-diagnostics --output-on-failure
python-port/venv/bin/python cpp-core/tools/benchmark_run_diagnostics.py \
  --phase reproduction --output build/new-reproduction
# 先review重現結果，再獨立啟動第二階段：
python-port/venv/bin/python cpp-core/tools/benchmark_run_diagnostics.py \
  --phase pose_memory --output build/new-pose-memory
```

保留`build/gray-loader`原凍結binary；兩phase採不同空目錄。離線摘要重算只讀相鄰protocol與jobs，不需要GPU／原輸入絕對路徑，解壓後用封存README指令；重新量測則先依`input-paths.json`重定位輸入並保留舊job不動。
