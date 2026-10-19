# 每換一張影格的完整成本：混合方案與完整GPU fine4

第41輪，2026-10-19，Apple M4 Pro／48 GiB。第40輪的正式呼叫在同一張來源上重複：混合Session重用首次呼叫建立的source pose context，完整GPU每次都重建全部與來源相關的狀態（見[第40輪報告文末更正](GPU_NCC_CLOSURE_zh-TW.md)）。該更正只用封存數字相加做估算，並把實測標為NOT RUN。本輪讓每次正式呼叫都經過`Session::updateSource`，實測每換一張影格的成本。只做量測，不改算法、GUI或通用預設；pilot、正式量測、失敗門檻與獨立稽核全部保留。

**結論：每次換新影格時，fine4在原128物件比混合快6.9%（808.35對868.05ms），原512物件仍以混合較快（fine4多13.7%，2806.40對2467.33ms）；兩組都是6個block全部同向。** 六物件控制fine4多13.3%，8K少70.2%，同樣6/6同向。第40輪封存估算的方向四組全部成立。128的差距比估算的4.4%大：混合每換一張影格除了重建pose context（中位19.94ms，與首次值相差不到1ms），候選搜尋與精修另多約8ms，估算沒有包含這部分。96個新程序、1008次直接GT全部零FP／FN。

## 正式結果

每影格時間（`frame_wall_ms`）＝`updateSource`（驗證、複製來源、釋放pose context）加上完整執行，不含圖片解碼、輸出比對與JSON。每格為6個新程序×3次正式呼叫共18次的中位數；每個程序的冷呼叫（建構Session）與暖機呼叫另外保存，不進分布。

| 場景 | 混合每影格ms | fine4每影格ms | fine4對混合 | 6個block | 各block差 |
|---|---:|---:|---:|---|---|
| 原128物件 | 868.05 | **808.35** | −6.9% | 6/6 fine4較快 | −11.1～−5.7% |
| 原512物件 | **2467.33** | 2806.40 | +13.7% | 6/6 混合較快 | +11.5～+15.0% |
| 六物件控制 | **62.57** | 70.88 | +13.3% | 6/6 混合較快 | +12.2～+15.0% |
| 8K、256模板、全角三尺度 | 259.88 | **77.49** | −70.2% | 6/6 fine4較快 | −73.3～−68.5% |

依事前規則，合併中位數與6個成對block中位數全部同向時才判定哪個方法較快；四組都符合。[正式摘要](stage41-new-frame-cost/formal-summary.json)

同一批block另跑第40輪的靜態job（同一來源重複；首次、1秒暖機、3次正式）作同建置對照：

| 場景 | 每影格只算執行 混合／fine4 ms | 差 | 靜態對照 混合／fine4 ms | 差 | 第40輪封存的靜態差 |
|---|---:|---:|---:|---:|---:|
| 原128 | 865.71／807.70 | −6.7% | 838.10／802.89 | −4.2% | −2.2% |
| 原512 | 2463.91／2804.59 | +13.8% | 2390.18／2798.14 | +17.1% | +17.3% |
| 六物件 | 61.93／70.82 | +14.4% | 59.68／69.95 | +17.2% | +17.0% |
| 8K | 258.51／74.73 | −71.1% | 255.77／66.66 | −73.9% | −74.8% |

「只算執行」是`total_ms`，即第40輪的計時範圍。靜態對照在本輪同樣是6/6同向；它和第40輪的差異來自不同建置與時間，不能與本輪倍率相乘。

### 每次重建的pose context

| 場景 | 每影格`pose_create_ms`中位（範圍，18次） | 冷呼叫（6程序） | 靜態對照首次（6程序） | 第40輪封存首次中位 |
|---|---:|---|---|---:|
| 原128 | 19.94（18.31～21.01） | 18.69～21.98 | 18.59～20.11 | 19.08 |
| 原512 | 40.19（39.35～44.36） | 39.90～40.64 | 39.58～42.91 | 40.87 |
| 六物件 | 1.35（1.22～2.62） | 1.43～1.48 | 1.41～3.02 | 1.52 |
| 8K | 0 | 0 | 0 | 0 |

每一次精修的混合呼叫都建立了新的pose context（`pose_create_ms`大於0；第1次以後`pose_invalidated`皆為真）；靜態對照的正式呼叫全部為0，確認它重用首次的context。三組精修場景的穩態每影格值與首次值相差不到1ms；第40輪「首次值可能含冷啟動成本」的保留，在pose重建本身並不成立。8K是core-only不精修，混合從未建立pose context。

### 多出的時間在哪裡

以下為正式呼叫本來就輸出的分段欄位，取18次中位；不是另跑的profile。[分段摘要](stage41-new-frame-cost/stage-breakdown.json)

| 場景 | 混合每影格比靜態多 | 其中pose重建 | 精修其餘：每影格／靜態 | 候選搜尋：每影格／靜態 | `updateSource` |
|---|---:|---:|---:|---:|---:|
| 原128 | +27.61ms | 19.94 | 67.76／66.93 | 778.00／771.84 | 2.17 |
| 原512 | +73.73ms | 40.19 | 213.99／206.98 | 2212.39／2182.59 | 3.66 |
| 六物件 | +2.25ms | 1.35 | 16.59／15.85 | 44.00／43.84 | 0.59 |
| 8K | +2.74ms | 0 | — | 258.50／255.76 | 1.38 |

前兩欄是`total_ms`，`updateSource`在它之外、計入每影格時間。混合多出的時間大於pose重建：128約多8ms、512約多34ms，主要在候選搜尋（+6.2／+29.8ms），精修在新建的pose buffer上也略慢（+0.8／+7.0ms）。

fine4的GPU時間每影格只比靜態多0.3～7.0ms（128為806.32對801.55ms，512為2802.45對2795.47ms），`updateSource`為0.64／1.36ms。8K例外：每影格GPU時間72.73ms，前2個block約64.9ms（與靜態相同），後4個block為73.1～75.3ms；靜態對照6個block都在64.5～65.3ms。

runner在每次呼叫之間重新解碼圖片（128約65ms、512與8K約130ms、六物件約4ms，均不計時），這段期間的CPU／GPU工作型態與連續重複同一來源不同。新的來源記憶體、每影格新建的pose buffer，以及解碼空檔中的時脈狀態都可能有關；本輪沒有硬體計數器，不把其中任何一項當成已量出的原因。

### 封存估算是否成立

| 場景 | 封存估算：fine4對加回後混合 | 本輪實測每影格 | 實測－估算 | 方向 |
|---|---:|---:|---:|---|
| 原128 | −4.4% | −6.9% | −2.5點 | 成立 |
| 原512 | +15.3% | +13.7% | −1.6點 | 成立 |
| 六物件 | +14.1% | +13.3% | −0.8點 | 成立 |
| 8K | −74.8% | −70.2% | +4.6點 | 成立 |

事前規則只判方向：四組實測方向都已確立（6/6），而且與估算同號；差距只報告，不設門檻。估算假設每影格成本等於首次`pose_create_ms`。本輪顯示pose重建本身與首次值相差不到1ms，但混合另有上節的額外時間，fine4在8K也略增。以本輪自己的靜態對照套用相同估算法，混合每影格`total_ms`比估算多7.71／33.32／0.78／2.74ms（128／512／六物件／8K），每影格時間多10.05／36.74／1.42／4.11ms。[決定與失敗](stage41-new-frame-cost/decision-and-failures.json)

### 記憶體

| 場景 | peak RSS MiB：混合每影格／fine4每影格／混合靜態／fine4靜態 | fine4命名Metal配置MiB | 混合frame配置／pose context MiB |
|---|---|---:|---|
| 原128 | 268.88／127.36／232.12／94.08 | 116.47 | 42.97／64.82 |
| 原512 | 510.16／234.30／442.56／167.58 | 236.30 | 64.50／131.03 |
| 六物件 | 48.83／36.81／42.47／30.02 | 6.15 | 5.96／3.32 |
| 8K | 502.61／411.77／373.09／282.86 | 161.50 | 104.39／0 |

RSS與命名Metal配置範圍不同，不能相加。混合的pose context使用`pose_storage_mode=1`共用頁面，它的GPU buffer容量和主機影像可能是同一批統一記憶體，也不能與RSS相加。每影格程序的RSS比靜態高，包含每次重新解碼與runner保留全部輸出列。fine4命名配置與第40輪完全相同；RSS門檻沒有失敗。

## 量測方法與界線

- 每影格arm使用runner的`source_frames`序列模式：`sequence_mode`為`reuse`、`warmup_seconds=0`，`source_frames`只列該場景的封存fixture，以`repeats`重複「1＋暖機＋3」個cycle。這等同把同一張影像重複列出，但讓runner把每個cycle和第0次比較。第1次以後每次呼叫先重新解碼（不計），再由`updateSource`複製並釋放pose context（計入每影格時間），然後執行。fine4的`updateSource`在主機端複製兩次來源（Session快照與pipeline內部副本），執行時照常重新上傳與重算。
- 暖機：第0次（建構Session，冷）與其後W次排除在分布外，但仍做GT。W＝max(1, ceil(1500ms／第40輪兩方法中較快的正式中位))，128／512／六物件／8K為2／1／26／23次。事前門檻要求每個程序從第0次結束到最後一次暖機結束至少1秒（對應第40輪的1秒暖機）；實測最短1748／2571／1715／4606ms，全部通過。
- 靜態對照沿用第40輪job，只把正式次數由2改為3。
- 每個場景6個交錯順序block：偶數block依序為混合每影格、fine4每影格、混合靜態、fine4靜態，奇數block反向。每個（場景、arm、block）一個新程序，共96程序、288次正式呼叫，2026-10-19 06:00:21～06:10:52 UTC。量測期間沒有其他建置、測試或實驗；driver在每個程序前紀錄負載與競爭程序，96次都沒有競爭的建置或benchmark程序，1分鐘load average為0.89～3.65。
- driver在每個程序前取樣的GPU使用率是ioreg的瞬時值，緊接在前一個程序結束之後，96次中有77次高於50%（最高100%），反映的是前一個程序的尾段；正式開始前為0～7%。判斷沒有其他工作的依據是競爭程序紀錄，不是這個數值。
- 靜態對照的暖機依runner的牆鐘1秒；8K fine4的block 2暖機算法時間合計999.81ms（牆鐘仍滿1秒），靜態暖機沒有另設算法時間門檻。manifest的源碼指紋涵蓋快照286檔中的272檔（macos-cli與根目錄CMakeLists.txt等不在指紋規則內），runner與dylib另有直接雜湊。
- pilot（1 block、16程序）在建置後最先執行，也吸收首次Metal shader編譯；方向與正式相同，不作決定證據。[pilot摘要](stage41-new-frame-cost/pilot-summary.json)
- GT在父程序獨立評估，從不傳給搜尋器：每影格arm的冷、暖機與正式呼叫，以及靜態arm的首次與正式呼叫。正式1008次加pilot 168次，全部零FP／FN（2px／1°／2%相對尺度），最大誤差0.128px／0.0995°／0.274%（512）。同一arm內全部呼叫的有序matches／proposals完全相同，每影格arm也與同方法的靜態對照完全相同；這與GT分開驗收，全部通過。
- 保留的13筆失敗全是fine4對混合的每影格`time_budget`（block中位不得超過混合＋max(2ms, 10%)）：pilot 512的block 0，以及正式512與六物件各6個block。它們是「fine4不能取代混合」的證據，不因量測完成而刪除。沒有GT、route、一致性、暖機、RSS或host門檻失敗。

界線：同一張影像重複、內容不變，不是變化中的相機串流，也不是實拍精度驗收。每次呼叫之間的解碼空檔屬於這個情境，沒有另外控制。只有一台M4 Pro。第40輪估算來自另一個建置，只做方向對照，不與本輪倍率相乘。

## 驗證

| 檢查 | 結果與範圍 |
|---|---|
| 量測前Release | ctest 73／73、Python harness 567／567（含本輪26項）、`color_verify_runner_test.py --metal` 5／5，皆在任何計時之前 |
| 量測工具 | 26項負控制；14個手動突變（暖機排除、暖機門檻、同方法一致性、GT範圍、估算方向、host、RSS與時間容忍、pose規則、靜態重用、job、暖機取整、方向一致性、drain）全部被測試抓到 |
| 摘要重算 | `--summarize`由manifest重算的正式摘要與量測當下內嵌摘要逐項相同 |
| 獨立稽核 | 另一個agent只依protocol撰寫腳本，不匯入量測harness、不啟動GPU：自行實作一對一GT匹配，重算正式96與pilot 16程序的檔案雜湊、輸入與binary雜湊、執行順序、stdout與結果JSON一致性、1008＋168次GT、各arm與block中位數、暖機、同方法一致性、time／RSS門檻與計數，4059＋1339個數值與發佈摘要零不符（4個最大角度誤差差1.4e−14，在容許範圍內） |
| 封存重播 | 解壓源碼快照（286檔雜湊全符）與raw-results，用凍結工具重算pilot與正式摘要，與發佈檔逐位相同；不重跑GPU |
| 量測後Release | 06:15 UTC、HEAD `8fbb2b3`：ctest 73／73、Python harness 567／567、color verify `--metal` 5／5 |

## 使用與完整源碼

[量測工具](../tools/benchmark_new_frame_cost.py)、[負控制](../tests/test_new_frame_cost_harness.py)；它重用第40輪的job、GT與route驗證，沒有改動第37～40輪工具。從repo根目錄，以有NumPy／OpenCV的Python與第37輪fixture manifest重跑：

```sh
python cpp-core/tools/benchmark_new_frame_cost.py --blocks 6 --measured 3 \
  --build build/stage41-release \
  --fixtures-manifest build/stage37-full-gpu/formal/manifest.json \
  --output build/stage41-new-frame-cost/formal/manifest.json
python cpp-core/tools/benchmark_new_frame_cost.py \
  --summarize build/stage41-new-frame-cost/formal/manifest.json \
  --round40-raw cpp-core/docs/stage40-ncc-closure/raw-results.tar.gz \
  --output build/stage41-new-frame-cost/formal-summary.json
```

量測結束後只在[Log分類器](../tools/build_log_catalog.py)加入stage41目錄對照，並新增報告與索引；量測工具與算法源碼未改。正式量測需要macOS、Metal與Release研究建置（GUI＋experiments＋OpenCV）。輸入檔與第40輪[fixtures.tar.gz](stage40-ncc-closure/fixtures.tar.gz)逐位相同，本輪只保存雜湊。重現應使用凍結的[source-snapshot.tar.gz](stage41-new-frame-cost/source-snapshot.tar.gz)，不把日後HEAD當成同一份源碼。

## 封存

[封存清單與SHA256](stage41-new-frame-cost/archive.json)：

| 檔案 | 內容 |
|---|---|
| [protocol.json](stage41-new-frame-cost/protocol.json) | 量測前提交的規格：場景、arm、暖機、門檻與判定規則 |
| [source-snapshot.tar.gz](stage41-new-frame-cost/source-snapshot.tar.gz) | 凍結commit `7270946`的286個已提交源碼檔與雜湊清單 |
| [raw-results.tar.gz](stage41-new-frame-cost/raw-results.tar.gz) | pilot與正式的manifest、job、stdout／stderr與每個程序的完整結果 |
| [pilot-summary.json](stage41-new-frame-cost/pilot-summary.json)、[formal-summary.json](stage41-new-frame-cost/formal-summary.json) | 可由封存重算的摘要，含第40輪估算對照 |
| [stage-breakdown.json](stage41-new-frame-cost/stage-breakdown.json) | 正式呼叫分段欄位的中位數 |
| [decision-and-failures.json](stage41-new-frame-cost/decision-and-failures.json) | 判定、估算是否成立與全部失敗門檻 |
| [validation.json](stage41-new-frame-cost/validation.json)、[validation-logs.tar.gz](stage41-new-frame-cost/validation-logs.tar.gz) | 建置、測試、突變、獨立稽核、重播與封存腳本 |
| [input-sha256.json](stage41-new-frame-cost/input-sha256.json) | 輸入雜湊，與第40輪清單相同 |

## 未涵蓋

變化內容的影格序列與實拍串流、其他Metal裝置、CUDA與Windows／Linux都沒有執行。混合每影格多出的候選搜尋時間與8K的fine4每影格增量沒有硬體計數器說明。本輪不改GUI或通用預設；完整GPU仍是明確指定的研究選項。
