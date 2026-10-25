# 候選去重疊在平行選峰上：合併後的實際速度

第44輪，2026-10-25，Apple M4 Pro／48 GiB。這是**量測輪**，不改任何演算法、kernel或預設；所有arm都是main（847e785）上已有的選項。

第43輪的候選去重（`candidate_dedup_top_k=3`）在完整GPU路徑上是用peak mode 1量的，當時第42輪的平行選峰（`full_gpu_peak_mode=2`）還沒合併；合併後第42輪把完整GPU的研究建議改成peak2／fine4。兩輪的加速不能相乘，所以本輪在同一個建置上把兩者一起量，並重新排出「每換一張影格」時四種組合的順序。

k3仍是**會改變語意**的選項：輸出不保證與去重關閉時相同，本報告不宣稱等價；peak2則與peak1逐位相同（第42輪）。GUI、核心與研究預設都不變。

**結論：peak2上開k3仍然大幅省時，而且GT中立、輸出和第43輪封存逐位相同。每換一張影格時，「完整GPU peak2＋k3」在128、512與六物件場景都是四種組合中最快的；8K上k3沒有差別。**

- Q1（主門檻）：完整GPU peak2靜態時間，k3對off在128物件−35.5%（580.66→374.57ms）、512物件−38.9%（1870.81→1143.71ms），各6/6 block同向，達成事前10%目標。比第43輪在peak1上量到的−26.1%／−25.7%更多，因為peak2已把粗選峰從約1秒降到33ms，細搜尋占的比例變大。
- Q2（每換一張影格）：完整GPU＋k3在128（384.13ms）、512（1149.05ms）、六物件（41.91ms）都比其他三組快，且6個block全部同向；8K的完整GPU k3與off相差−0.3%、只有3/6 block同向，不點名最快者，但兩組完整GPU都比混合快約80%。
- GT：79個完整GPU案例變體零GT退步；158次輸出全部與第43輪peak1封存逐位相同。正式的192個程序、每次呼叫都零FP／FN，且輸出都與第43輪封存相同。
- 失敗保留：GT階段4筆`job_equals_round43`失敗，原因是兩個控制場景重新產生在本輪目錄，job裡的路徑不同；影像SHA-256與第43輪相同、輸出也相同。依事前規則照列為失敗，不放寬門檻。正式、pilot、profile沒有失敗門檻。

## 正式結果

同一建置、同一工作量。每格是6個新程序×3次正式呼叫共18次的中位數；冷呼叫與暖機不計入。block順序在奇數block反轉。

### 靜態（第40輪範圍：同一張影像重複呼叫，runner `total_ms`）

| 場景 | 混合 off | 混合 k3 | 差 | 各block | 完整GPU off | 完整GPU k3 | 差 | 各block |
|---|---:|---:|---:|---|---:|---:|---:|---|
| 原128物件 | 839.21 | 588.55 | −29.9% | 6/6，−36.5～−28.8% | 580.66 | **374.57** | −35.5% | 6/6，−36.0～−35.3% |
| 原512物件 | 2374.22 | 1528.50 | −35.6% | 6/6，−37.6～−30.3% | 1870.81 | **1143.71** | −38.9% | 6/6，−39.0～−38.7% |
| 六物件控制 | 58.77 | 43.30 | −26.3% | 6/6，−30.2～−21.6% | 62.99 | **41.32** | −34.4% | 6/6，−35.4～−33.8% |
| 8K、256模板、全角三尺度 | 238.93 | 240.79 | +0.8% | 3/6，−1.1～+1.4% | 45.79 | 45.86 | +0.1% | 2/6，−0.2～+0.6% |

- 主要速度目標（完整GPU，128與512合併中位數至少少10%，6個成對block同向）：達成。
- time_budget（六物件與8K，每個block的k3不得超過off＋max(2ms, 10%)）：兩條路徑全部通過。
- 混合k3對off在本輪重現第43輪的方向（第43輪：−30.4／−37.0／−25.8／−0.1%）；這是同建置重測，不是新的加速。
- 混合的靜態呼叫重用首次建立的pose context，所以靜態不做跨路徑比較；跨路徑看下一節。

### 每換一張影格（第41輪範圍：每次呼叫都經過`Session::updateSource`，`frame_wall_ms`）

| 場景 | 混合 off | 混合 k3 | 完整GPU off | 完整GPU k3 | 最快 |
|---|---:|---:|---:|---:|---|
| 原128物件 | 879.46 | 630.13 | 590.94 | **384.13** | 完整GPU k3 |
| 原512物件 | 2581.20 | 1579.18 | 1869.81 | **1149.05** | 完整GPU k3 |
| 六物件控制 | 63.85 | 47.28 | 63.52 | **41.91** | 完整GPU k3 |
| 8K | 249.93 | 250.39 | 49.42 | 49.25 | 不點名（兩組完整GPU相同） |

「最快」只在它對其他三組都是6個block全部較快、且合併中位數也較快時才點名。主要的成對比較：

| 場景 | 完整GPU k3對完整GPU off | 完整GPU k3對混合 k3 | 完整GPU k3對混合 off | 完整GPU off對混合 off |
|---|---|---|---|---|
| 原128 | −35.0%（6/6） | −39.0%（6/6） | −56.3%（6/6） | −32.8%（6/6） |
| 原512 | −38.5%（6/6） | −27.2%（6/6） | −55.5%（6/6） | −27.6%（6/6） |
| 六物件 | −34.0%（6/6） | −11.4%（6/6） | −34.4%（6/6） | −0.5%（3/6，無方向） |
| 8K | −0.3%（3/6，無方向） | −80.3%（6/6） | −80.3%（6/6） | −80.2%（6/6） |

- 六物件：關閉去重時完整GPU與混合相差−0.5%、只有3/6同向，沒有方向（第42輪曾報告混合快1.5%）；開了k3以後完整GPU對兩組混合都6/6較快。
- 8K：完整GPU k3對off的各block差距在−19.8～+18.5%之間，合併中位數只差−0.3%。8K每模型只有約530個候選、6個物件，去重能省的細搜尋本來就少；pilot（1個block，只供篩選）看到的k3較慢在正式量測中沒有重現。
- 每影格的`source_update_ms`中位數：混合1.80／2.88ms（128／512），完整GPU 0.63／1.32ms；混合每影格另外重建pose context，這些都包含在`frame_wall_ms`內。

## 工作量與階段（profile，只作診斷）

encoder timestamp profile另外跑（128與512、完整GPU靜態off／k3、各1程序2次呼叫），有額外開銷，不與正式時間相比；encoder重疊時階段和會大於GPU區間。

| 階段GPU ms | 128 off | 128 k3 | 512 off | 512 k3 |
|---|---:|---:|---:|---:|
| fine_search | 415.35 | 212.71 | 1331.45 | 629.24 |
| initial_refine | 76.85 | 76.17 | 272.53 | 263.66 |
| model_nms | 22.92 | 15.85 | 148.36 | 115.80 |
| coarse_ncc | 41.47 | 41.47 | 93.57 | 93.92 |
| coarse_peaks | 12.17 | 12.46 | 33.55 | 33.49 |
| candidate_dedup | — | 3.37 | — | 13.91 |
| rescue | 19.25 | 18.95 | 7.33 | 7.10 |
| 呼叫總時間 | 583.18 | 375.83 | 1870.07 | 1140.13 |

- k3把細搜尋減半（−48.8%／−52.7%），去重kernel本身3.37／13.91ms。
- 細搜尋仍是最大階段（約54%），其次是initial_refine（20–22%）與model_nms；粗選峰只剩約3%。
- GT階段79個案例變體合計，k3留下374,675／705,697個粗候選（53.1%）。

## 記憶體

程序peak RSS、命名Metal配置與混合frame緩衝是不同範圍，不相加。

| 場景 | 完整GPU命名Metal配置 off→k3 | 完整GPU peak RSS（靜態） | 混合peak RSS（靜態） |
|---|---|---|---|
| 原128 | 139.73→139.94 MiB | 91.1→91.0 MiB | 229.9→230.2 MiB |
| 原512 | 282.04→282.45 MiB | 164.7→164.6 MiB | 440.8→441.7 MiB |
| 六物件 | 12.85→12.89 MiB | 27.3→26.9 MiB | 39.2→39.2 MiB |
| 8K | 184.83→184.84 MiB | 279.9→279.9 MiB | 370.6→370.5 MiB |

k3只多出`dedup-flags`與`dedup-counts`（512為0.41 MiB）。RSS門檻（k3不得超過off＋max(8 MiB, 10%)）在兩種模式、兩條路徑都通過。每影格模式的RSS較高（例如128完整GPU 126.9 MiB），因為多了來源複本與序列狀態，與靜態分開報告。

## GT階段

第43輪全部有完整GPU路徑的79個案例變體（第40輪正式／保留／stop0 13個、第34輪挑戰6個與Blender 4個、第33輪5個、第34輪尺度支撐33個、第35輪49物件RGB 16個、第43輪對稱與相鄰控制2個），在peak2上各跑off與k3，各一個新程序、第一次呼叫加一次重複。

- GT中立：k3相對off沒有任何案例變體的嚴格TP減少或FP增加。79個變體合計off與k3都是TP 3201／FP 31／FN 204（既知失敗案例照列在內）。
- 與第43輪封存比對：158個程序的每次呼叫，有序matches、proposals（全部欄位）與去重計數（去掉時間欄位）都與第43輪peak1結果相同。
- 輸出與off相同（這不是門檻，k3也不預期相同）：matches 67/79、proposals 44/79，與第43輪完整GPU相同。
- 兩個控制場景由`candidate_dedup_scenes.py`重新產生，4張影像的SHA-256與第43輪相同，所以本輪不依賴第43輪session的worktree。

## 驗證

- 量測工具`tools/benchmark_dedup_peak2.py`有19項負控制測試（`tests/test_dedup_peak2_harness.py`）：多一個誤判、peak mode 1、去重計數契約、第43輪封存差1e-7、第43輪job差一個鍵、只快5%、一個block反向、六物件time_budget、RSS、最快者需每個block都贏、每影格輸出需等於靜態、證據被改、少一個worker、GT少一個物件、GT封存去重計數差1、GT的peak mode 1、GT worker失敗。10個工具突變全部被抓到（其中「每影格與靜態一致性檢查被略過」第一次存活，補上測試後被抓到）。
- 不匯入harness的獨立稽核（自寫的最大二分配對嚴格GT、自讀第43輪封存）從本輪封存重算全部中位數、k3對off與跨路徑差、block方向、速度目標、最快者規則與GT計數：17,117項檢查零不符。把摘要中一個中位數改0.01ms的負控制會被抓到。
- 用封存重新產生四份摘要，與提交的摘要逐位相同。
- 測試與建置結果見[validation.json](stage44-dedup-peak2/validation.json)。

## 量測方法與界線

- 正式計時：2026-10-25 02:08:46～02:34:00 UTC，192個新程序；開始前沒有其他建置或benchmark程序，每個worker前記錄主機狀態，沒有host_busy紀錄。本輪計時期間母session沒有執行其他工作。
- 每影格暖機次數是第41輪的3倍（128／512／六物件／8K為6／3／78／69次），因為實作smoke中完整GPU k3每影格只要387ms，第41輪的2次暖機不足1秒；每個worker的暖機實際時間都≥1秒（門檻）。這是protocol事前規定的，所有arm相同。
- 靜態job除了peak鍵與重複次數之外，與第43輪正式job逐鍵相同（自動檢查）。
- protocol（[protocol.json](stage44-dedup-peak2/protocol.json)）與凍結源碼在任何量測前提交；事前揭露包含：先前整合重播看過的1146＋474個GT輸出、以及128物件每影格smoke的4個數字（856／610／589／387ms）。

## 使用方式

完整GPU、大量相同物件、可接受輸出不與基線逐位相同時（研究runner JSON；尺度、角度、分數等仍依任務設定）：

```json
{
  "full_gpu": true,
  "coarse_peak_mode": 1,
  "full_gpu_peak_mode": 2,
  "full_gpu_peak_batch": 8,
  "full_gpu_fine_mode": 4,
  "candidate_dedup_top_k": 3,
  "candidate_dedup_radius": 0.5,
  "candidate_dedup_angle_steps": 3
}
```

要求輸出與基線逐位相同時，去掉三個`candidate_dedup_*`鍵（或設`top_k=0`）。

## 封存

| 檔案 | 內容 |
|---|---|
| [protocol.json](stage44-dedup-peak2/protocol.json) | 事前登錄的問題、arm、門檻、判定規則與揭露 |
| [source-snapshot.tar.gz](stage44-dedup-peak2/source-snapshot.tar.gz)、[source-sha256.json](stage44-dedup-peak2/source-sha256.json) | 5c569f1的凍結源碼 |
| [gt-summary.json](stage44-dedup-peak2/gt-summary.json) | GT階段 |
| [formal-summary.json](stage44-dedup-peak2/formal-summary.json)、[pilot-summary.json](stage44-dedup-peak2/pilot-summary.json)、[profile-summary.json](stage44-dedup-peak2/profile-summary.json) | 計時與診斷摘要 |
| [decision-and-failures.json](stage44-dedup-peak2/decision-and-failures.json) | 判定與全部失敗 |
| raw-results-gt／pilot-profile／formal-static／formal-frames-hybrid／formal-frames-gpu.tar.gz | 全部manifest、job、stdout與結果；正式分三包，各在100 MB以內 |
| [input-sha256.json](stage44-dedup-peak2/input-sha256.json) | 全部輸入的雜湊 |
| [validation.json](stage44-dedup-peak2/validation.json)、validation-logs.tar.gz | 建置與測試log、工具突變、獨立稽核、摘要重播 |

## 未涵蓋

- 內容會變的影格序列與實拍串流（每影格模式重複同一張影像）。
- 其他Metal裝置、CUDA與Windows／Linux。
- 對稱週期小於3個粗角度步的模板（第43輪的限制仍在）。
- 下一個瓶頸：細搜尋仍占約54%，其次是initial_refine與model_nms；本輪沒有處理。

本輪不改GUI或任何預設；完整GPU與去重都仍是需要明確指定的研究選項。
