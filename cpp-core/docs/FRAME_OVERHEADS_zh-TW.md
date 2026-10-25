# 每換一張影格的額外成本從哪裡來

第48輪，2026-10-26，Apple M4 Pro／48 GiB，macOS 27.0.1。[第41輪](NEW_FRAME_COST_zh-TW.md)發現，每次呼叫都換新影格時，混合方案比靜態對照多出的時間大於pose context重建（候選搜尋+6／+30ms、精修+1／+7ms，128／512物件），完整GPU fine4在8K的每影格GPU時間也增加約8ms。但那一輪每次換影格前，runner都先解碼圖片（65–131ms，GPU閒置），靜態對照則是連續執行，兩個因素混在一起。本輪把「換來源」與「呼叫之間的空檔」拆開量測。這是診斷輪，不改GUI、核心或研究預設。

**結論：在本輪的建置與作業系統上，混合方案每換一張影格多出的時間幾乎全是pose context重建，解碼空檔本身沒有可確立的影響；第41輪多出的候選搜尋時間與8K的GPU時間增加都沒有重現。** 真正會讓下一次呼叫變慢的是「閒置」空檔：呼叫前睡65–131ms，混合慢31.6–42.3ms，完整GPU在512與8K慢5.6–6.9ms（128沒有可確立的影響）；混合的慢化隨空檔變長而增加，完整GPU（8K劑量）則在空檔達10–30ms時多約5.5–6ms，之後不再增加（300ms時fine4 +4.5ms、peak2 +5.7ms）。同樣長度但CPU在解碼的空檔，除了8K混合的影像準備外沒有這種效果。8K完整GPU多出的是主機端上傳與命令編碼，GPU執行時間不變或減少；512只有約一半在上傳與編碼，另一半在等待GPU完成，GPU執行時間沒有可確立的方向（見文末更正）。事前登錄的H1（空檔拖慢第一段GPU工作）判為「混合」，H2（新pose buffer讓精修變慢）成立但只有約1ms，H3被否定，H4與補充S1判為「兩者皆非」。四個phase共4760次直接GT全部零FP／FN。

## 設計

每個arm都是runner的`source_frames`序列，重複同一張封存fixture。本輪在runner加入兩個預設關閉的選項：`sequence_mode:"static"`（第0次之後不再`updateSource`，來源與pose context保留）與`sequence_gap`（呼叫之間`decode`解碼〔預設，原行為〕、`none`不做事、`sleep`閒置`sequence_gap_sleep_ms`）。

| 條件 | 來源 | 呼叫之間 | 用途 |
|---|---|---|---|
| static_none | 不換 | 無 | 基準：連續執行 |
| static_sleep | 不換 | 閒置65／130／131ms | 只有閒置空檔（長度等於第41輪的解碼中位數） |
| static_decode | 不換 | 解碼後丟棄 | 只有解碼空檔（CPU忙） |
| reuse_none | 每次`updateSource` | 無（事先解碼） | 只有換來源 |
| reuse_decode | 每次`updateSource` | 解碼 | 第41輪每影格job原樣 |

方法為第41輪的`hybrid_best`與`full_gpu_simd`（peak1／fine4），8K另加現行建議`full_gpu_parallel`（peak2、batch 8、fine4）作補充。正式量測三個場景（原128、原512、8K），6個交錯順序block，每個程序第0次（建構Session）與第41輪相同的暖機次數（2／1／23）排除在分布外，之後3次正式呼叫。效果＝兩個條件的合併中位數差；只有6個成對block全部同號時才標示方向。判定規則在[protocol](stage48-frame-overheads/protocol.json)中事先登錄；H1是看過第41輪封存的每模型欄位後才具體化（512粗搜尋GPU時間+14ms），protocol已揭露，因此不是盲測預測。

## 正式結果

全部為正式呼叫的中位數（ms，每格6程序×3次）。

| 場景／方法 | static_none | static_sleep | static_decode | reuse_none | reuse_decode |
|---|---:|---:|---:|---:|---:|
| 128 混合 total | 836.68 | 869.71 | 837.81 | 856.76 | 859.58 |
| 128 fine4 total | 833.73 | 833.25 | 832.64 | 832.74 | 832.40 |
| 512 混合 total | 2380.33 | 2411.91 | 2381.52 | 2418.42 | 2419.25 |
| 512 fine4 total | 2983.66 | 2990.58 | 2987.14 | 2987.40 | 2982.78 |
| 8K 混合 total | 242.13 | 284.39 | 251.33 | 243.01 | 251.20 |
| 8K fine4 total | 62.03 | 67.64 | 62.34 | 62.11 | 62.41 |
| 8K peak2 total | 47.29 | 53.11 | 48.02 | 47.87 | 47.78 |

`total_ms`只算執行；reuse條件的每影格時間另加`updateSource`（中位數混合1.31–2.44ms、完整GPU 0.66–2.88ms）。[正式摘要](stage48-frame-overheads/formal-summary.json)

### 換來源與空檔各自的效果

| 場景／方法 | 第41輪式（reuse_decode−static_none） | 只換來源（reuse_none−static_none） | 只有解碼空檔 | 只有閒置空檔 |
|---|---:|---:|---:|---:|
| 128 混合 | +22.90 ↑ | +20.08 ↑ | +1.13 | +33.03 ↑ |
| 512 混合 | +38.92 ↑ | +38.09 ↑ | +1.19 | +31.58 ↑ |
| 8K 混合 | +9.07 ↑ | +0.87 | +9.19 ↑ | +42.26 ↑ |
| 128 fine4 | −1.33 ↓ | −0.99 | −1.09 | −0.48 |
| 512 fine4 | −0.88 | +3.74 | +3.48 | +6.92 ↑ |
| 8K fine4 | +0.37 | +0.07 | +0.31 | +5.61 ↑ |
| 8K peak2 | +0.48 | +0.57 ↑ | +0.72 | +5.82 ↑ |

↑／↓表示6個block全部同號；沒有箭頭表示方向不確立。

**換來源**：混合每影格多的時間＝pose重建（128 20.22ms、512 39.81ms）加上新pose buffer上的精修（精修扣pose重建+1.24／+0.85ms，皆↑；GPU精修時間128 +0.24ms無方向、512 +0.80ms↑）。候選搜尋（proposal）是−1.34／−4.43ms，沒有增加；512的粗搜尋GPU時間−0.10ms。完整GPU換來源沒有可確立的影響，只有8K peak2的+0.57ms是6/6同號。

**解碼空檔**：只有8K混合成立（+9.19ms），幾乎都在`prepare_ms`（8.07→17.02ms），即64MB解碼之後的影像準備；其他場景與方法都在±3.5ms內且方向不一。8K混合的第41輪式效果（+9.07ms）因此來自解碼空檔，不是換來源（+0.87ms）。

**閒置空檔**：混合多出的時間主要在CPU部分——CPU正規化+10.81／+8.67／+9.09ms，`prepare_ms`+0.41／+9.19／+12.45ms，8K粗搜尋GPU時間+6.93ms（皆↑）。完整GPU的`device_ms`在8K反而少0.89ms（↓），多出的是主機端上傳：8K fine4 1.04→6.47ms、peak2 1.06→5.66ms、512 fine4 0.52→3.21ms（皆↑）。512 fine4的total +6.92ms中，上傳+2.69ms、命令編碼+0.86ms、等待GPU完成（`wait_ms`）+3.46ms（皆↑），`device_ms` +2.95ms沒有方向（6個block中有2個是+7.3／+8.7ms），所以512不能說只有上傳、GPU時間不變。本輪沒有CPU／GPU頻率讀數（powermetrics需root），所以「閒置後CPU降頻」只是與資料一致的解釋，不是量出的原因。

### 空檔長度（劑量）

靜態、不換來源，3個block，在呼叫之間閒置0／10／30／100／300ms。[劑量摘要](stage48-frame-overheads/dose-summary.json)

| 方法與指標 | 0 | 10 | 30 | 100 | 300 ms |
|---|---:|---:|---:|---:|---:|
| 512 混合 total | 2378.85 | 2380.62 | 2392.69 | 2412.17 | 2435.56 |
| 512 混合 粗搜尋GPU時間（事前指標） | 78.75 | 80.54 | 82.97 | 80.70 | 101.19 |
| 512 混合 prepare | 4.15 | 4.41 | 4.44 | 6.83 | 11.64 |
| 8K fine4 total | 62.54 | 62.37 | 68.53 | 67.65 | 67.04 |
| 8K fine4 GPU時間（事前指標） | 60.81 | 59.96 | 59.73 | 59.23 | 59.39 |
| 8K fine4 上傳 | 1.03 | 1.31 | 6.97 | 6.62 | 6.30 |
| 8K peak2 total | 47.31 | 52.83 | 52.96 | 53.42 | 52.96 |
| 8K peak2 上傳 | 1.03 | 6.12 | 5.46 | 6.14 | 6.15 |

混合的total隨空檔長度單調增加（300ms時+56.7ms）；事前登錄的粗搜尋GPU時間不單調，300ms對0ms+22.43ms（↑）。完整GPU的GPU時間不隨空檔增加（300ms對0ms：fine4 −1.42ms、peak2 −0.40ms，皆↓）；上傳時間在fine4於10–30ms之間、peak2在10ms以內就跳到約6ms，之後持平；total對0ms在fine4為+6.0／+5.1／+4.5ms（30／100／300ms）、peak2為+5.5–6.1ms，沒有再增加。表中非事前指標（total、prepare、上傳）是本輪事後的說明，事前判定只看GPU時間與粗搜尋GPU時間。

### GPU分段（profile，診斷用）

8K fine4開啟timestamp，2個block，每格6次正式呼叫。五個條件的各階段GPU時間幾乎相同（粗NCC 34.6–35.0ms、粗選峰34.0–34.4ms、來源金字塔4.9–5.3ms、細搜尋3.9ms），GPU整段57.7–58.3ms；閒置條件的total多約9ms，其中上傳多約6ms（1.04→6.99ms）、命令編碼多約3ms（1.50→4.42ms），都在主機端。閒置條件下沒有任何一段GPU工作的中位數比static_none多0.2ms以上。模板金字塔的單次值在同一條件內就相差1.8–3.5ms，中位數1.5–4.1ms、以static_none最高；profile只有2個block，不據此判斷方向。[profile摘要](stage48-frame-overheads/profile-summary.json)

分段中位數原本把每個程序的冷呼叫與23次暖機也算進去（每格54次），與protocol「只算正式呼叫」不符；獨立稽核發現後，量測後修正driver並重算，變動最大的是reuse_none的模板金字塔（4.11→2.29ms）。total、GPU時間、上傳與編碼本來就只算正式呼叫，不受影響；pilot、正式與劑量的摘要逐位不變，判定不受影響（profile不是判定輸入）。原摘要保留為[profile-summary-frozen-driver.json](stage48-frame-overheads/profile-summary-frozen-driver.json)，修訂與每個分段的前後數值見[analysis-revision.json](stage48-frame-overheads/analysis-revision.json)。

## 假說判定

| 事前假說 | 判定 | 依據（正式、6/6 block規則） |
|---|---|---|
| 重現第41輪 | 否 | 混合128／512的第41輪式效果↑，但8K fine4的GPU時間沒有增加 |
| H1 空檔拖慢第一段GPU工作 | 混合（mixed） | 閒置空檔使512混合粗搜尋GPU時間+2.65ms↑，但8K fine4 GPU時間沒有增加；解碼空檔兩者都沒有 |
| H2 新pose buffer讓精修變慢 | 成立 | 512混合GPU精修+0.80ms↑，兩種空檔都沒有 |
| H3 新來源讓影像準備變慢 | 否定 | 512混合`prepare_ms`換來源+0.13ms無方向；閒置空檔才+9.19ms↑ |
| H4 8K fine4增加的來源 | 兩者皆非 | 換來源與空檔都沒有提高GPU時間 |
| S1 8K peak2（補充） | 兩者皆非 | 同上 |

[判定與失敗](stage48-frame-overheads/decision-and-failures.json)

## 對第41輪與現行建議的意義

- 第41輪「混合每換一張影格另有候選搜尋與精修約8ms」的現象，本輪沒有重現；換來源的成本是pose重建加不到1.3ms的精修。第41輪與本輪的作業系統不同（macOS 26→27），本輪無法說明當時多出時間的原因，也不改寫第41輪的量測。
- 第41輪的每影格設計（呼叫之間CPU在解碼）並沒有額外懲罰；但如果影格之間CPU與GPU都閒著（例如相機幀間隔大於處理時間），閒置65–131ms時在這台機器上下一張會多付約31–42ms（混合）或約6ms（完整GPU；8K幾乎都是主機端上傳，512約一半）。本輪沒有量「換影格＋閒置」的組合，不據此推算兩種方法的每影格排名。
- 不改任何預設；runner的新選項預設關閉，原有job的輸出只多了`sequence_gap`與`gap_ms`欄位。

## 量測方法與界線

- 正式窗口2026-10-25 16:27–16:56 UTC（劑量16:56–17:01、profile 17:01–17:02）。driver在每個程序前記錄負載與競爭程序；本輪把偵測補強為ucomm名稱加完整參數行、不分大小寫（含Blender）、repo實驗腳本與任何佔滿一核的程序。正式210個程序只有一次系統Spotlight（corespotlightd 187%）與一次1分鐘負載4.09，各記為一筆`host_busy`／`host_load_high`；劑量與profile各有一次Spotlight。protocol只規定記錄主機gate、沒有規定後果，正式摘要保留這兩個程序（128 fine4 static_sleep block 3、8K混合static_decode block 4）。事後診斷：把這兩個block從同場景、同方法的每個arm移除後重算110個效果，六項判定不變，報告表中的方向也不變；只有128 fine4的gap_on_update（不在判定與表中）有3個指標變成↓。
- 第一次開始（16:17 UTC）因另一個session的`fpm_color_verify`而被driver拒絕，沒有產生資料。pilot（16:22–16:27）期間另一個session執行cmake／clang／ld與含Metal測試的ctest，pilot的512與8K有25筆主機gate；pilot只作篩選。該session（第47輪）事後告知：16:23:54–16:26:57 UTC依序執行全新Release建置、含Metal測試的ctest、Python測試，以及中途停止的ASan建置與測試；最後一個程序早於正式manifest的開始時間（16:27:29）。
- GT在父程序獨立評估：每個arm的冷、暖機與正式呼叫，pilot 515、正式3090、劑量885、profile 270次，全部零FP／FN（2px／1°／2%）。同arm、同方法各arm之間的有序matches／proposals完全相同，8K peak2與peak1相同；與GT分開驗收，全部通過。
- 靜態與換來源的路徑檢查：static之後`source_generation`為0、不複製來源、pose只在第0次建立；reuse每次遞增並重建pose；解碼空檔確實解碼，閒置空檔在[ms, ms+50]內，無空檔≤1ms。全部通過。
- 記憶體：峰值RSS因解碼而增加（512混合static_none 444.1MiB、static_decode 504.1MiB、reuse_decode 508.3MiB）；命名Metal配置不變（fine4 128／512／8K為116.47／236.30／161.50MiB，8K peak2 184.83MiB）。RSS與Metal配置不相加。

界線：只有一台M4 Pro；作業系統與第41輪不同；同一張影像重複；只有時間推論，沒有頻率或硬體計數器；睡眠長度取第41輪的解碼中位數，不是本作業系統的解碼長度。

## 驗證

| 檢查 | 結果與範圍 |
|---|---|
| runner選項 | 先寫的`native_runner_sequence_gaps`與「fixture須等於runner完整鍵清單」的`native_runner_job_keys`在實作前失敗、實作後通過；隔離副本中8個runner編譯突變全部被抓到（第一次因GNU Make秒級時間戳沒有重編而中止，保留紀錄後重跑） |
| 量測工具 | 18項負控制；22個手動突變（凍結driver）全部被抓到 |
| 量測前Release | ctest 83／83、Python harness 670／670、color verify `--metal`通過 |
| 摘要重算 | 四個phase由manifest重算的摘要與量測當下內嵌摘要逐項相同 |
| 獨立稽核 | 不匯入harness、只用stdlib、不啟動GPU或runner：四個phase 300個程序、4760列，比對32,966個數值零不符，重新推導的判定與發佈相同；GT 4760列TP 294850、FP 0、FN 0（最大誤差0.128px、0.0995°、尺度0.27%）。指出profile分段的範圍問題（已修正，見下列）與正式主機gate沒有規定後果（已揭露） |
| 量測後分析修訂 | profile分段改為只算正式呼叫：先改負控制（暖機慢2ms），凍結driver失敗（32.5≠31.5）、修正後通過；修正後的分段中位數與稽核的「只算正式呼叫」重算逐項相同，原數值與稽核的「全部呼叫」重算逐項相同 |
| 封存重播 | 從封存解出的源碼快照310個檔雜湊相符；凍結driver重算的pilot、正式、劑量摘要與原profile摘要，以及修正後driver重算的四份發佈摘要，全部逐位相同 |
| 量測後Release | 量測後、分析修訂前（`1da20c6`）：ctest 83／83、Python harness 671／671、color verify `--metal` 5／5。修訂後在`176b661`（乾淨工作樹，前後都檢查）全新Release研究建置86個編譯單元：ctest 83／83、Python harness 671／671（含本輪18項）、color verify `--metal` 5／5、其他Python 49／49，沒有skip。沒有跑ASan／UBSan：本輪唯一的C++改動在runner，sanitized組態（不含GUI）不建置它 |

## 封存

[封存清單與SHA256](stage48-frame-overheads/archive.json)：[protocol.json](stage48-frame-overheads/protocol.json)（含計時前的修訂1）、[source-snapshot.tar.gz](stage48-frame-overheads/source-snapshot.tar.gz)（凍結commit `06467f2`的310個已提交源碼檔）、raw-results-formal-hybrid-best.tar.gz與raw-results-formal-full-gpu.tar.gz（正式依方法分兩包，manifest在後者）、raw-results-other.tar.gz（pilot、劑量、profile），內含manifest、job、stdout／stderr與每個程序的完整結果；四個phase的摘要與凍結driver的原profile摘要、[analysis-revision.json](stage48-frame-overheads/analysis-revision.json)（量測後分析修訂1）、[decision-and-failures.json](stage48-frame-overheads/decision-and-failures.json)、[validation.json](stage48-frame-overheads/validation.json)與validation-logs.tar.gz（含獨立稽核、主機gate敏感度與重播）、[input-sha256.json](stage48-frame-overheads/input-sha256.json)（與第40輪fixtures逐位相同，只存雜湊）。

[量測工具](../tools/benchmark_frame_overheads.py)、[負控制](../tests/test_frame_overheads_harness.py)、[runner](../tools/native_pipeline_runner.cpp)、[runner契約測試](../tests/native_runner_contract_test.cpp)。重現量測應使用凍結的源碼快照；目前的driver只在profile分段的統計範圍與快照不同。

## 未涵蓋

「換影格＋閒置」的組合、其他空檔長度以外的作業系統電源設定、其他Metal裝置、CUDA與Windows／Linux都沒有執行；沒有CPU／GPU頻率讀數。第41輪多出時間的原因仍未知。

## 2026-10-26 合併驗收更正

合併到 main 前的獨立重算（只用 stdlib、不匯入本輪工具，從封存重算 300 個程序、4760 次 GT 呼叫：TP 294850、FP／FN 0，所有中位數與發佈相同）沒有改變任何事前判定，但指出兩處文字說得比資料強；作者用 `formal-summary.json` 與 `dose-summary.json` 核對屬實，本頁、`EXPERIMENT_REVIEW`、`PROJECT_STATUS`、`CURRENT_BEST` 與 `CLAUDE.md` 已改寫。

- **「完整GPU多出的是主機端上傳，GPU執行時間不變」只在8K成立**：8K閒置空檔的 total +5.61ms（fine4）＝上傳 +5.43、編碼 +0.63ms（皆↑），`device_ms` −0.89ms、`wait_ms` −0.42ms（皆↓）；peak2 +5.82ms 中上傳 +4.60ms、`device_ms` 無方向。512 fine4 的 total +6.92ms 只有 +2.69（上傳）＋0.86ms（編碼）在主機端，等待 GPU 完成 +3.46ms（6/6 ↑），`device_ms` +2.95ms 無方向（block 0／3 為 +8.7／+7.3ms，其餘 −0.9 至 +1.5ms）。512 的另一半時間花在哪裡，本輪資料無法再細分。
- **「約6ms後持平」**：8K fine4 total 對 0ms 在 10／30／100／300ms 為 −0.2／+6.0／+5.1／+4.5ms，peak2 為 +5.5／+5.6／+6.1／+5.7ms，所以是「出現後不再增加」，不是固定約 6ms；上傳時間本身才接近持平（fine4 6.97／6.62／6.30ms）。
- 其他稽核紀錄（不影響數字）：各 phase 的 manifest 記錄了源碼、binary、輸入與 fixtures 的 SHA256，但沒有記錄 `protocol.json` 的雜湊，protocol 的雜湊只在 `archive.json`；正式的兩筆主機 gate 沒有規定後果、pilot 與第47輪的建置重疊、profile 分段在量測後改了 driver，這三點上文已揭露。
- 合併驗收在本輪 head `465191a` 的乾淨 worktree 重跑：全新 Release 研究建置、ctest 83／83、Python harness 671、color verify `--metal` 5、其他 Python 49，以及含 experiments 的 scalar ASan／UBSan 組態 ctest 77／77，全部通過（前後都是乾淨工作樹）。
