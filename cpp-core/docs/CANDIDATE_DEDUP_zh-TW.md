# 細搜尋前的粗候選去重（會改變語意的研究選項）

第43輪，2026-10-24，Apple M4 Pro／48 GiB。

第40輪的profile顯示，多物件場景的細搜尋工作大多重複：512物件fine4在最後一層細搜尋仍有83,824個候選，最後只剩512個結果；細搜尋約占GPU時間48%（1326ms），`model_nms`另占148ms。原因是每個粗候選在NMS之前都會先做完細搜尋（`src/matcher.cpp`的註解寫明「coarse scores do not bound fine scores」）。

本輪在粗搜尋與細搜尋之間加入一個預設關閉的去重選項，量測它省下多少工作、會失去什麼。

**這個選項會改變語意。** 被丟掉的候選不再細搜尋，所以matches與proposals不保證和基線相同。本報告不宣稱精確等價，GT準確與「輸出和基線相同」分開驗收。GUI、核心與研究預設都維持`top_k=0`。

**結論：主設定k3（radius 0.5、angle_steps 3）在兩條路徑都GT中立，四個正式案例中有三個時間明顯縮短，但輸出不是逐位相同。**

- GT：全部112個混合案例變體與79個完整GPU案例變體，嚴格TP都沒有減少、FP都沒有增加。
- 速度：原128物件混合−30.4%、完整GPU−26.1%；原512物件−37.0%／−25.7%；六物件控制−25.8%／−30.2%。這三個案例各6個block全部同向。8K幾乎不變。
- 失敗保留：8K混合有1個block慢11.5%，超過time_budget門檻，保留為失敗。
- 輸出：有序matches與基線完全相同的，混合97/112、完整GPU 67/79；不同的案例GT計數都沒變，只是位姿或分數有小差異。proposals完全相同的只有74/112與44/79。
- 敏感度：k2也GT中立。k1在混合6個、完整GPU 4個案例變體出現GT退步。
- 負控制：兩個刻意放寬的消融都如預期失敗。只看位置的k1失去對稱模板的物件；半徑過大的k1失去相鄰物件。

選項保留為明確指定的研究開關，不改任何預設。

## 正式結果（速度）

表中是同一建置、同一工作量的完整算法時間（runner `total_ms`），單位ms。每格是6個新程序×3次正式呼叫共18次的中位數；冷呼叫與至少1秒的暖機不計入。

| 場景 | 混合 off | 混合 k3 | 差 | 各block | 完整GPU off | 完整GPU k3 | 差 | 各block |
|---|---:|---:|---:|---|---:|---:|---:|---|
| 原128物件 | 835.61 | **581.67** | −30.4% | 6/6，−31.0～−29.8% | 802.42 | **593.32** | −26.1% | 6/6，−27.1～−25.8% |
| 原512物件 | 2407.51 | **1516.65** | −37.0% | 6/6，−38.9～−35.3% | 2835.04 | **2105.87** | −25.7% | 6/6，−26.2～−24.9% |
| 六物件控制 | 58.84 | **43.64** | −25.8% | 6/6，−27.1～−22.2% | 70.13 | **48.99** | −30.2% | 6/6，−32.2～−24.4% |
| 8K、256模板、全角三尺度 | 251.75 | 251.51 | −0.1% | 3/6，−3.1～+11.5% | 59.87 | 59.36 | −0.8% | 5/6，−2.1～+0.2% |

依事前規則判定：

- 主要速度目標（128與512合併中位數至少少10%，且6個成對block同向）：兩條路徑都達成。
- time_budget（每個block的k3不得超過off＋max(2ms, 10%)）：六物件兩條路徑都通過，8K完整GPU通過。8K混合的block 0為279.13對250.26ms（+11.5%），超過門檻，保留為失敗；其餘5個block在−3.1～+1.9%。8K每模型只有約530個候選、6個物件，細搜尋工作本來就少，不是這個選項的目標工作量。
- pilot（1 block，只供篩選）方向與正式相同：−29.2／−26.2／−36.2／−25.6／−26.1／−28.8／−0.9／−1.2%。[正式摘要](stage43-candidate-dedup/formal-summary.json)、[pilot摘要](stage43-candidate-dedup/pilot-summary.json)

同一輪、同一建置下也可以比較兩條路徑：k3時混合在128快2.0%（581.67對593.32ms）、在512快28.0%（1516.65對2105.87ms）。這些是靜態正式呼叫，混合會重用首次建立的pose context（第41輪每換一張影格約多20／40ms），所以128的差距小於這個成本，不能說每換影格時混合也比較快；512的差距遠大於它。這組比較不與其他輪的倍率相乘。

### 省下的是哪些工作

第一張表是GT階段第一次呼叫的計數（與正式呼叫相同：正式首呼叫的輸出與GT階段逐項相同）：

| 場景 | 留下的候選 混合 | 留下的候選 完整GPU | 混合細搜尋maps off→k3 |
|---|---|---|---|
| 原128 | 20,188／42,458（47.5%） | 22,863／42,460（53.8%） | 228,667→104,799（45.8%） |
| 原512 | 44,795／108,220（41.4%） | 52,576／108,220（48.6%） | 659,376→269,462（40.9%） |
| 六物件 | 1,382／2,488 | 1,501／2,488 | 13,059→7,188 |
| 8K | 402／531 | 418／530 | 1,698→1,311 |
| 512 stop0 | 44,795／108,220 | 52,576／108,220 | 790,104→303,005 |
| 512 原解析度subpixel | 44,795／108,220 | 52,576／108,220 | 1,850,128→520,260（28.1%） |

混合的正式分段（18次中位數）中，候選搜尋（含粗、細搜尋）從770.52降到516.78ms（128）、從2194.72降到1318.88ms（512）。精修從66.43到63.70、從207.87到198.88ms，NMS不到1ms。去重本身在CPU上的耗時中位為10.18ms（128）與27.72ms（512），已包含在上述時間內。

完整GPU的encoder timestamp profile另外跑，有額外開銷，不與正式時間相比：

| 階段GPU ms | 128 off | 128 k3 | 512 off | 512 k3 |
|---|---:|---:|---:|---:|
| fine_search | 415.58 | 212.23 | 1326.86 | 628.08 |
| coarse_peaks | 249.80 | 252.84 | 1021.81 | 1029.91 |
| initial_refine | 77.02 | 77.76 | 271.93 | 261.24 |
| model_nms | 22.79 | 15.82 | 147.90 | 115.60 |
| candidate_dedup | — | 3.47 | — | 13.62 |
| 全部階段合計 | 829.6 | 625.8 | 2875.4 | 2155.1 |

- 細搜尋角度評估次數：512從659,373降到319,658（−51.5%），128從228,673降到120,606（−47.3%）。
- 最後一層細搜尋仍在的候選：512從83,824降到38,501，128從25,118降到12,357。
- k3之後最大的GPU階段變成粗選峰（512約1030ms），它不受本選項影響。[profile摘要](stage43-candidate-dedup/profile-summary.json)

### 記憶體

| 場景 | 混合 peak RSS MiB off／k3 | 完整GPU peak RSS MiB off／k3 | 完整GPU命名Metal配置 MiB off／k3 |
|---|---|---|---|
| 原128 | 229.74／228.11 | 90.89／90.91 | 116.47／116.68 |
| 原512 | 437.78／438.45 | 164.52／164.29 | 236.30／236.71 |
| 六物件 | 39.16／38.41 | 27.31／27.21 | 6.15／6.19 |
| 8K | 370.55／370.52 | 279.89／279.85 | 161.50／161.51 |

RSS取6個block的中位數。每個block都通過RSS門檻（k3不得超過off＋max(8 MiB, 10%)）。

命名Metal配置只在開啟時多出三樣：`dedup-flags`（每slot 4 bytes）、`dedup-counts`，以及加長的final readback。512多0.41 MiB。off的配置與第40輪完全相同（116.47／236.30 MiB）。

RSS與命名Metal配置是不同範圍，不能相加。

## GT（全部場景）

每個案例變體×路徑×arm都是一個新程序，跑首次呼叫加1次重複。GT在父程序評估，從不傳給搜尋器。判定標準：

- 嚴格：一對一匹配，位置≤2px、角度≤1°、尺度比在2%內。FP與FN都計入，180°別名算失敗。
- 輔助：20px的定位結果只列出，不作判定。

1146個worker全部成功。每個worker的重複呼叫與首次逐位相同，GT計數也相同。

| 家族（案例變體數） | 內容 | k3留下的候選 混合／完整GPU | k3輸出matches相同 混合／完整GPU |
|---|---|---|---|
| 第40輪（13） | 正式、保留、stop0、邊界、空場景、常數來源 | 43.3%／50.2% | 6/13／6/13 |
| 第34輪合成挑戰（6） | 遮擋（已知失敗）、相似干擾物、接觸與12%／30%重疊 | 56.1%／61.8% | 6/6／6/6 |
| 第34輪Blender（4） | 中性、換光、遮擋（已知失敗）、干擾物 | 50.3%／56.5% | 4/4／4/4 |
| 第33輪（5） | new_texture12、new_small12與degraded12（已知失敗）、空場景、512原解析度subpixel | 43.7%／50.7% | 3/5／4/5 |
| 第34輪尺度支撐（66；完整GPU 33） | Dst4小模板24、Dst10／Dst3對照8、retained 1，各有不開／開scale support（後者只混合） | 73.9%／77.8% | 62/66／31/33 |
| 第35輪RGB 49物件（16） | 8張半合成圖，調整後與原始預算（開發集原始預算為已知失敗） | 59.6%／66.7% | 14/16／14/16 |
| 第43輪控制（2） | sym_marker12、adjacent_grid16 | 54.2%／59.9% | 2/2／2/2 |

各arm對同路徑off的合計（第一次呼叫）：

| arm | 混合：退步案例數、TP／FP變化 | 完整GPU：退步案例數、TP／FP變化 | 留下的候選 混合／完整GPU |
|---|---|---|---|
| **k3（主）** | **0**，±0／±0 | **0**，±0／±0 | 46.7%／53.1% |
| k2 | 0，±0／±0 | 0，±0／±0 | 30.2%／30.9% |
| k1 | 6，−6／+5 | 4，−4／+3 | 14.7%／14.8% |
| k1只看位置（消融） | 20，−44／+27 | 18，−42／+25 | 8.4%／8.1% |
| k1半徑2.0（消融） | 17，−53／+8 | 15，−51／+6 | 4.4%／4.4% |

k3與k2沒有任何案例改變嚴格TP／FP／FN，包括已知失敗、遮擋、干擾物與空場景。原本就失敗的案例，失敗計數完全不變。例如第34輪開發集遮擋仍是2/1/4，第35輪開發集原始預算的clean仍是4/0/45。

**召回率在k3與k2沒有變化。** [GT摘要](stage43-candidate-dedup/gt-summary.json)

k1的退步：

- many128_holdout：TP−1、FP+1（同一物件輸出成錯誤位姿）。
- Dst10接近90°對稱的方形模板：兩個渲染器，開或不開scale support，TP−1、FP+1。
- sym_marker12：TP−1。

只看位置的k1失敗得最廣，連原128／512、保留、stop0與512原解析度都出現TP−1～−4並伴隨同樣多的FP。這代表真位姿被同一位置、另一角度的別名候選取代。所以「把角度納入叢集」不只是對稱模板控制的需要，在一般多物件場景也成立。

### 輸出與基線的差異（另列，不是門檻）

- k3混合有15個案例變體的有序matches與off不同，完整GPU有12個。這些案例的matches數量都相同、GT計數也都相同。不同之處在位姿或分數的小差異，因為留下來的另一個候選先完成細搜尋。
- 涉及的案例：原128／512、兩個保留、兩個stop0、512原解析度、heldout12、new_texture12（只有混合）、Dst10接近對稱的方形模板，以及第35輪保留集原始預算的兩張。
- proposals數量普遍變少，例如512從1301降到1249（混合）或1264（完整GPU）：每個尺度模型送進精修的候選較少。

### 負控制

| 場景（真物件數） | off | k3 | k2 | k1 | k1只看位置 | k1半徑2.0 |
|---|---|---|---|---|---|---|
| sym_marker12（12） | 11/0/1 | 11/0/1 | 11/0/1 | 10/0/2 | **4/0/8** | 10/0/2 |
| adjacent_grid16（16） | 16/0/0 | 16/0/0 | 16/0/0 | 16/0/0 | 16/0/0 | **6/1/10** |

表中為TP/FP/FN，兩條路徑結果相同。

- sym_marker12：模板除了一塊每8×8格平均為零的棋盤標記以外完全180°對稱，所以粗層幾乎分不出真位姿與180°別名。只看位置時，別名把真位姿擠掉，12個物件少了7個。分開角度的k3／k2則與基線相同。基線本身漏掉的1個物件沒有任何proposal，與本選項無關。
- adjacent_grid16：16個物件中心相距1.04倍模板寬高，兩兩重疊都不超過0.2，所以全部可同時輸出。半徑2.0（R≈163px，大於102px的鄰距）時，較強的鄰居把較弱的物件丟掉，16個少了10個。主設定的R＝40.8px則全部保留。
- 第34輪的接觸場景（相鄰、12%／30%重疊）在k3與基線相同（5/0/1）。在半徑2.0下兩組各少2個。

單元測試另有獨立控制：

- 格點實作與逐對比較的定義在300組隨機配置下逐一相同。
- 兩個180°模態與兩個相距50px的物件。
- Metal kernel與`dedupKeep`在40組FP32 slot配置下逐一相同。
- Session層面：關閉時與預設逐位相同；CPU與Metal混合frame的決定相同；完整GPU仍是1個command buffer、1次等待。

## 量測方法與界線

- 事前登錄：protocol與凍結源碼在任何量測前提交（`50027de`，凍結`5d1d1ac`的294個源碼檔）。protocol寫明量測前看過的資料：
  - 實作smoke：128的k3各路徑一次。
  - 兩個控制場景的迭代：標記太弱、尺度範圍太寬時，基線本身只有9/12。
  - 消融arm與控制場景一起設計。

  k3、radius 0.5與angle_steps 3在任何執行前固定，之後沒有改。
- 路徑：混合用第40輪`hybrid_best` job。完整GPU用第40輪`full_gpu_simd`（peak1／fine4）。scale support是CPU功能，那些變體只跑混合。
- 階段：
  - GT：1146個worker，最多2個平行，2026-10-24 13:07:54～13:14:44 UTC，不作計時宣稱。其中159個worker執行時，同機有突變測試的`cmake`在跑（本session與另一個session）。
  - pilot：16程序，13:15:13～13:16:33。
  - 正式：96程序，13:16:36～13:24:28。每個（案例、arm、block）一個新程序；偶數block依混合off、混合k3、完整GPU off、完整GPU k3的順序，奇數block反向。
  - profile：4程序，13:24:31～13:25:00。
- 主機：pilot、正式與profile每個程序執行前都記錄競爭程序，全部沒有。正式期間1分鐘load average為1.59～6.89，包含runner本身的8個CPU執行緒。同機其他Claude session事先約定在這段時間暫停建置與量測。
- 暖機：runner以牆鐘至少1秒暖機。有2個正式程序的暖機算法時間合計略少於1000ms，牆鐘仍滿1秒，如實保留。
- 一致性：正式首呼叫的有序輸出與GT階段同案例、同路徑、同arm的輸出逐項相同。所有正式呼叫都是零FP／FN。
- 界線：
  - 只有一台M4 Pro。
  - GT都是合成或半合成；FSC與VisA沒有位姿GT，未納入。
  - GT不允許對稱等價。
  - 第34輪的新影格序列（lifecycle frames）沒有重跑。
  - 混合與完整GPU的粗候選本身不同（FP32旋轉與正規化），所以留下的數量不同，例如128為20,188對22,863。本輪沒有進一步分解兩者粗候選的差異。

## 驗證

| 檢查 | 結果與範圍 |
|---|---|
| 量測前Release | `50027de`全新建置（GUI＋experiments＋OpenCV）：ctest 78／78（原73項加5項本輪測試）、Python harness 584／584、`color_verify_runner_test.py --metal` 5／5 |
| 單元測試 | 格點過濾與逐對比較定義在300組隨機配置下相同，含±180°接縫；兩個對稱模態控制與相鄰物件控制；邊界與非法參數 |
| Metal kernel | `fg_dedup_mark`／`fg_dedup_apply`在40組FP32 slot配置（含同分、角度接縫）與`dedupKeep`逐一相同，也不寫出模型範圍 |
| Session契約 | CPU、Metal混合、完整GPU：關閉時與預設逐位相同；計數一致；CPU與Metal混合決定相同；完整GPU 1個command buffer、1次等待、0次中途讀回；非法參數與稀疏路徑會被拒絕 |
| harness負控制 | 15項：GT退步、只有輸出不同、計數契約、residency、證據雜湊、缺worker、不重複、速度／時間／RSS／host／GT門檻、執行順序、job鍵與top_k回報 |
| 編譯突變 | 9個突變（只計鄰域內留下者、忽略角度、用角點代替中心、GPU同分提前停止、GPU用第0層尺寸、GPU不套用丟棄、忽略過濾結果、Session不傳k、GPU把丟棄計為留下），在含量測後測試的樹上全部被抓到，還原後全部通過。只用事前登錄的測試時，「角點代替中心」與「忽略過濾結果」兩個存活。這兩項之後由量測後新增的中心約定場景與細搜尋maps檢查抓到。這些測試只改測試，量測的程式碼（`5d1d1ac`的源碼，`494f774`未改）也通過它們 |
| 獨立稽核 | 另一個agent只依protocol與檔案格式撰寫，不匯入任何harness：自行實作一對一GT匹配、中位數與門檻。重算1146個GT與96個正式worker，52,248項檢查零不符，GT門檻失敗80筆、正式1筆與發佈摘要相同 |
| 封存重播 | 解壓兩個raw封存與凍結源碼快照（294檔雜湊全符），用快照內工具重算gt／pilot／formal／profile摘要，與發佈檔逐位相同；不重跑GPU |
| 量測後Release | `494f774`、源碼樹乾淨、全新建置85個編譯單元：ctest 78／78、Python harness 585／585、color verify `--metal` 5／5 |
| Metal validation layer | `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`，dedup、完整GPU與runner測試22／22 |
| ASan／UBSan | Debug、`FPM_SANITIZERS`、`FPM_DISABLE_SIMD`、experiments：dedup、portable_core與native_research共8／8，沒有sanitizer報告；LeakSanitizer未啟用 |

[驗證紀錄](stage43-candidate-dedup/validation.json)

## 使用與完整源碼

這是研究選項，GUI沒有開關。runner job範例（在任何既有job上加三個鍵）：

```json
{"candidate_dedup_top_k": 3, "candidate_dedup_radius": 0.5, "candidate_dedup_angle_steps": 3}
```

- C++：`fpm::research::Options::candidateDedupTopK`等。
- frame C ABI：`fpm_frame_set_candidate_dedup`、`fpm_frame_get_candidate_dedup_metrics`（[coarse_ncc.h](../experiments/coarse_ncc.h)）。
- 核心規則：[candidate_dedup.cpp](../src/candidate_dedup.cpp)與`matchWithCoarseNcc`。公開的`fpm::match`不受影響。
- Metal：[full_gpu_dedup.metal](../experiments/full_gpu_dedup.metal)。

量測工具是[benchmark_candidate_dedup.py](../tools/benchmark_candidate_dedup.py)，控制場景由[candidate_dedup_scenes.py](../tools/candidate_dedup_scenes.py)產生，harness負控制在[test_candidate_dedup_harness.py](../tests/test_candidate_dedup_harness.py)。從`cpp-core/tools`、以有NumPy／OpenCV的Python執行：

```sh
python benchmark_candidate_dedup.py --phase gt --jobs 2 --build ../../build/stage43-release \
  --output ../../build/stage43-candidate-dedup/gt/manifest.json
python benchmark_candidate_dedup.py --phase pilot --build ../../build/stage43-release \
  --output ../../build/stage43-candidate-dedup/pilot/manifest.json
python benchmark_candidate_dedup.py --phase formal --build ../../build/stage43-release \
  --output ../../build/stage43-candidate-dedup/formal/manifest.json
python benchmark_candidate_dedup.py --phase profile --build ../../build/stage43-release \
  --output ../../build/stage43-candidate-dedup/profile/manifest.json
python benchmark_candidate_dedup.py --summarize ../../build/stage43-candidate-dedup/formal/manifest.json \
  --gt-manifest ../../build/stage43-candidate-dedup/gt/manifest.json --output formal-summary.json
```

- GT輸入取自主checkout `build/`下各輪凍結的fixture；`--data-root`指定其位置。輸入雜湊列在[input-sha256.json](stage43-candidate-dedup/input-sha256.json)。
- 正式量測需要macOS、Metal與Release研究建置（GUI＋experiments＋OpenCV）。
- 重現應使用凍結的[source-snapshot.tar.gz](stage43-candidate-dedup/source-snapshot.tar.gz)。

## 封存

[封存清單與SHA256](stage43-candidate-dedup/archive.json)：

| 檔案 | 內容 |
|---|---|
| [protocol.json](stage43-candidate-dedup/protocol.json) | 量測前提交的規格、arm、門檻、判定與事前揭露 |
| [source-snapshot.tar.gz](stage43-candidate-dedup/source-snapshot.tar.gz)、[source-sha256.json](stage43-candidate-dedup/source-sha256.json) | 凍結commit `5d1d1ac`的294個源碼檔 |
| [raw-results-gt.tar.gz](stage43-candidate-dedup/raw-results-gt.tar.gz) | GT階段manifest、1146個job、stdout與結果，以及兩個控制場景的輸入 |
| [raw-results-timing.tar.gz](stage43-candidate-dedup/raw-results-timing.tar.gz) | pilot、正式與profile的manifest、job、stdout與結果 |
| [gt-summary.json](stage43-candidate-dedup/gt-summary.json)、[pilot-summary.json](stage43-candidate-dedup/pilot-summary.json)、[formal-summary.json](stage43-candidate-dedup/formal-summary.json)、[profile-summary.json](stage43-candidate-dedup/profile-summary.json) | 可由封存重算的摘要 |
| [decision-and-failures.json](stage43-candidate-dedup/decision-and-failures.json) | 判定與全部失敗（GT 80筆皆在k1與兩個消融arm，正式1筆） |
| [validation.json](stage43-candidate-dedup/validation.json)、[validation-logs.tar.gz](stage43-candidate-dedup/validation-logs.tar.gz) | 建置與測試log、突變、獨立稽核、封存重播 |
| [input-sha256.json](stage43-candidate-dedup/input-sha256.json) | 全部輸入的雜湊 |

## 未涵蓋

- 變化內容的影格序列與實拍串流。
- 其他Metal裝置、CUDA與Windows／Linux。
- 粗選峰本身的加速（第42輪另行處理，本建置不含）。
- 依場景自動選k的規則。

本輪不改GUI或任何預設；k3的成績只代表上述場景。對稱週期小於3個粗角度步的模板（例如細齒輪）可能讓不同模態落在同一鄰域，本輪沒有這類場景。
