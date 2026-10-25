# 實驗與實作 Log 分類索引

本索引讓外部審查者從一個結論追到量測條件、正式執行、失敗、驗證與當時源碼。歷史檔案保留原名、原路徑及原內容，避免移動後破壞報告連結與來源雜湊；分類透過新增索引完成。

先看[專案閱讀指南](../../docs/PROJECT_GUIDE_zh-TW.md)與[實驗總索引](EXPERIMENT_REVIEW_zh-TW.md)，再用下表找證據。最新建議以[目前建議方案](CURRENT_BEST_zh-TW.md)為準；資料夾裡有 `validation.json` 並不表示該輪速度、記憶體與精度全部達標。

## 檔案依用途分類

| 類別 | 典型名稱與位置 | 用途 |
|---|---|---|
| `protocol` 事前規格 | `protocol.json`、`baseline-protocol.json` | 輸入、參數、工作預算、量測順序及門檻；先讀這類再解讀結果 |
| `pilot` 先導探索 | `pilot/`、`pilot-diagnosis/` | 供設定選擇與問題診斷，不當正式性能結果 |
| `calibration` 閾值校準 | `calibration.json` | 保存正常校準值、無候選與暫定狀態，先於heldout |
| `formal` 正式量測 | 封存內 `formal/`、`formal-workers/`、`formal-jobs/` | 可重算的逐程序／逐次結果；不同 worker 不能只挑較快的 |
| `raw` 原始紀錄 | `raw-results.tar.gz`、stdout、stderr、trace、`.log` | 保留當時完整輸出；檔名為 raw 的封存也可能包含來源與輸入 |
| `summary` 彙整 | `summary.json`、`results.json`、分析與複查結果 | 供閱讀及比對；不能取代正式 worker |
| `failure_review` 失敗與修正歷史 | `performance-failures.json`、clipped、interrupted、attempt 等 | 需讀上下文的反例或早期執行，不因後來複查成功而刪除 |
| `false_gate` 明示未通過門檻 | JSON 中 `*_pass: false`、`*_passed: false` 等 | 索引保存精確 JSON pointer，方便定位；可能是負控制或重播歷史失敗，不能直接當作目前程式故障 |
| `validation` 驗證 | Release／ASan／UBSan／Metal logs、`validation.json`、archive replay | 區分演算法測試、資源生命週期、封存完整性與統計重播 |
| `source` 源碼與執行腳本 | 封存的 `source/`、`baseline-source.tar`、C++／Metal／Python 檔 | 對應當時版本；今日 HEAD 與封存版本可能不同 |
| `provenance` 來源與雜湊 | `artifact-sha256.json`、manifest、build provenance | 核對檔案、建置條件及環境；雜湊一致不等於方法正確 |
| `input` 與 `visual` 輸入及圖表 | fixtures、inputs、GT、`.npz`、PNG、Blender | 圖片不一定是輸入或 ground truth；回到報告確認標註語意 |

同一檔案可有多種角色。例如原始封存含正式結果、失敗與來源；圖表可能只是視覺摘要。分類由路徑規則產生，不自動判定一輪實驗採用或不採用。

## 歷次資料入口

| 階段 | 主題 | 資料入口 |
|---|---|---|
| 1 | 原版稽核、旋轉與模式 | [validation](validation.json)、[模式比較](mode-benchmark.json)、[旋轉方向](rotation-direction-validation.json) |
| 2 | Fourier、RGB scorer 與實圖 | [pose benchmark](pose-experiment-benchmark.json)、[實圖](real-image-experiments.json)、[實圖素材](real-image-experiments/) |
| 3–4 | CPU 連續精修、混合 GPU 精修 | [performance-accuracy](performance-accuracy/)、[gpu-refinement](gpu-refinement/) |
| 5–7 | 分流、成本、GPU 常駐、錯峰救援 | [pose-routing](pose-routing/)、[metal-cost-profile](metal-cost-profile/)、[resident-refinement](resident-refinement/)、[ncc-rescue](ncc-rescue/) |
| 8–10 | 分段量測、共享金字塔、精確選峰 | [pipeline-stages](pipeline-stages/)、[shared-fine](shared-fine/)、[coarse-next](coarse-next/) |
| 11–14 | 大量物件、重疊、分塊、GUI 與持續運行 | [many-objects](many-objects/)、[coarse-overlap](coarse-overlap/)、[coarse-tiling](coarse-tiling/)、[gui-integration](gui-integration/)、[stability-30min](stability-30min/) |
| 15–18 | 固定尺度、稀疏前段、回應壓縮與後端比較 | [fixed-scale](fixed-scale/)、[sparse-orientation](sparse-orientation/)、[sparse-compact](sparse-compact/)、[sparse-backends](sparse-backends/) |
| 19–21 | Metal 前段、CPU LUT 與提交合併 | [sparse-metal](sparse-metal/)、[sparse-cpu-lut](sparse-cpu-lut/)、[sparse-metal-merge](sparse-metal-merge/) |
| 22–25 | 融合精修、SIMD、影像共用與 u8 初始化 | [pose-fused](pose-fused/)、[pose-simd](pose-simd/)、[pose-shared](pose-shared/)、[pose-u8](pose-u8/) |
| 26–28 | 原 NCC 上界、細搜尋統計與點積重用 | [adaptive-ncc](adaptive-ncc/)、[fine-reuse](fine-reuse/)、[fine-strip](fine-strip/) |
| 29–31 | 角度圖重用、旋轉與 NCC 合併、讀圖記憶體 | [fine-angle-reuse](fine-angle-reuse/)、[fine-fused](fine-fused/)、[gray-loader](gray-loader/) |
| 32–34 | 慢樣本、精修組合、新場景與階段驗收 | [run-diagnostics](run-diagnostics/)、[resident-multiscale](resident-multiscale/)、[stage34-closure](stage34-closure/) |
| 35 | 密集實拍、RGB驗色與小模板召回 | [stage35-real-rgb](stage35-real-rgb/)、[報告](STAGE35_REAL_RGB_zh-TW.md) |
| 36 | Metal常駐粗選峰、少／多候選的性能反例 | [stage36-coarse-resident](stage36-coarse-resident/)、[報告](COARSE_RESIDENT_zh-TW.md) |
| 37 | 完整灰階Metal常駐、獨立GT、9場景與同步／門檻修正 | [stage37-full-gpu](stage37-full-gpu/)、[完整報告](FULL_GPU_zh-TW.md) |
| 38 | 128／512 GPU分段計時、精確分塊排序／合併 | [stage38-gpu-peaks](stage38-gpu-peaks/)、[報告](GPU_TILED_PEAKS_zh-TW.md) |
| 39 | 細搜尋分列快取、實際路徑計數、主假說失敗與stop0補充 | [stage39-fine-tiles](stage39-fine-tiles/)、[報告](GPU_FINE_TILES_zh-TW.md) |
| 40 | 精確u32／SIMD NCC、負面統計重用、七組保留場景、跨版本與大階段結案 | [stage40-ncc-closure](stage40-ncc-closure/)、[報告](GPU_NCC_CLOSURE_zh-TW.md) |
| 41 | 每換一張影格的成本：updateSource序列模式、同建置靜態對照與封存估算驗證 | [stage41-new-frame-cost](stage41-new-frame-cost/)、[報告](NEW_FRAME_COST_zh-TW.md) |
| 42 | 完整GPU粗選峰精確平行回合：批次分數ring、批次1消融、pilot／formal／profile／保留集 | [stage42-parallel-peaks](stage42-parallel-peaks/)、[報告](GPU_PARALLEL_PEAKS_zh-TW.md) |
| 43 | 細搜尋前的粗候選去重（改變語意）：全部GT家族、敏感度與消融負控制、兩路徑計時 | [stage43-candidate-dedup](stage43-candidate-dedup/)、[報告](CANDIDATE_DEDUP_zh-TW.md) |
| 44 | 去重疊在平行選峰上的實際速度（量測輪）：靜態與每換新影格8個arm、GT階段與第43輪封存逐位比對 | [stage44-dedup-peak2](stage44-dedup-peak2/)、[報告](DEDUP_PEAK2_zh-TW.md) |
| 46 | 物件內白參考的照明正規化同色驗證（`chroma_vk`）：測試樣本集 v1 的19場景×9方法、校準／保留照明切分、既有方法與改動前runner逐位比對 | [stage46-illuminant-chroma](stage46-illuminant-chroma/)、[報告](ILLUMINANT_CHROMA_zh-TW.md) |

新一輪正式結果由[實驗總索引](EXPERIMENT_REVIEW_zh-TW.md)連接，重新產生機器索引即可列出新增資料。沒有階段對照的目錄保留 `stage: null`，不猜測輪次或把它算成通過項目。

## 使用機器索引查詢

- [log-catalog.json](log-catalog.json)：逐檔路徑、bytes、SHA256、角色、資料夾與階段；獨立 JSON 的明示 false gate 以 JSON pointer 保留。
- [log-archive-members.json.gz](log-archive-members.json.gz)：壓縮的 JSON，列出每個 `.tar.gz` 內一般檔案的名稱、大小與角色。完整成員路徑可用來查特定實作或 log，無須先把所有封存解壓。
- [產生工具](../tools/build_log_catalog.py)：只使用 Python 標準函式庫，不修改實驗資料、不執行 benchmark、不解壓封存。

從 repository 根目錄更新或核對：

```sh
python3 cpp-core/tools/build_log_catalog.py
python3 cpp-core/tools/build_log_catalog.py --check
```

查出所有獨立 JSON 的明示失敗門檻：

```python
import json
from pathlib import Path

catalog = json.loads(Path("cpp-core/docs/log-catalog.json").read_text())
for artifact in catalog["artifacts"]:
    for pointer in artifact.get("explicit_false_gates", []):
        print(artifact["path"], pointer)
```

查出封存內與驗證相關的紀錄：

```python
import gzip
import json

with gzip.open("cpp-core/docs/log-archive-members.json.gz", "rt") as handle:
    index = json.load(handle)
for member in index["members"]:
    if "validation" in member["roles"]:
        print(member["archive"], member["path"])
```

索引雜湊涵蓋 `cpp-core/docs` 的一般非隱藏檔案，排除兩份生成索引自身；沒有追蹤 build 暫存或 Python 虛擬環境。archive member 索引只讀 tar header，不深入巢狀封存、不判讀每個內部 JSON 的 gate；因此「沒有列出 false gate」不能當作全通過證據。需要數值重算時使用該輪報告的 replay 工具。

## 審查一筆結論的最短順序

1. 找到報告中的場景、方法、採用決定與限制；確認是本輪還是歷史比較。
2. 打開 protocol，核對精度門檻與速度／記憶體容忍是否事先定義。
3. 用 summary 找到對應 worker，再查看原始 JSON、stderr 與建置來源。
4. 查看同場景失敗、負控制、早期修正及獨立複查，避免只讀成功摘要。
5. 核對測試／封存／統計重播的範圍。需要在新硬體採用時，另跑相同輸入與設定。

第34輪可作完整範例：[報告](STAGE34_CLOSURE_zh-TW.md) → [封存清單](stage34-closure/archive.json) → [重播與重建驗證](stage34-closure/validation.json)。資料與來源分成兩份封存，須依報告解壓到同一目錄；驗證紀錄成功保留了某些精度與速度 gate 為 false 的原結果，沒有把失敗改寫為成功。

第36輪另保留pilot與正式結果、舊非法enum測試失敗與更新後56項通過、ASan初次未重設建置的失敗，以及360張map中FP32有序峰與double的差異。正式性能gate失敗不因封存、測試或重建通過而改寫；完整來源、原作者fixtures與生成輸入分成三份封存，原始量測與驗證log另存。

## 後續紀錄規則

每輪保留 protocol、逐程序正式結果、彙整、來源與輸入雜湊、驗證 log，以及採用／不採用的理由。先導資料、正式量測、修正前失敗與重跑各用不同檔名，避免覆蓋。外部資料集另附來源、版本、標註語意與使用條件；真實、半合成、純合成分開列示。

新增資料後先產生索引，再執行 `--check`。索引檔的時間與記憶體分類只提供查詢入口；不將不同輪次結果直接合併成一個總速度，也不把歷史失敗從目錄隱藏。

第37輪將探索pilot、凍結來源的formal、單元測試早期失敗、共同金字塔失敗、Metal儀器化容量限制與修正、ASan／UBSan、審查紀錄分開保存。正式數據的源碼／binary／輸入hash與source snapshot對應；pilot保留當時hash與結果，不聲稱pilot具有獨立可重建的完整來源快照。詳細分類見本輪archive與validation索引。

第38輪將profile-original、profile-tiled與未插樁formal分開；診斷不混入速度gate。早期GPU counter resolve零值、修正前host來源、獨立stage稽核、64組選峰差分、JSON負控制、乾淨重建與sanitizer分別保留。九個相對混合的速度gate失敗不因分塊較舊GPU顯著改善而移除；下一輪細搜尋建議明確標示為尚未實測的源碼分析。

第39輪formal／profile保留原設定，stop0-formal／stop0-profile是診斷後新增的單一參數補充，8k-profile說明原8K的分列適用性。主假說的兩項改善失敗與共14個相對混合的速度block失敗均保留；不能把stop0收益算成原128／512加速。完整來源、96組細搜尋差分、計數審查、乾淨重建、離線重播與圖表程式分開封存。

第41輪的protocol與凍結源碼在任何計時前提交；pilot與formal分開保存，pilot只供篩選。每影格arm的冷呼叫、暖機與正式呼叫都在同一份runner輸出內，摘要依protocol的暖機次數切分，沒有另存刪減後的資料。13筆每影格time_budget失敗（含pilot 1筆）保留在摘要與判定檔。輸入檔與第40輪fixtures逐位相同，因此只保存雜湊；獨立稽核腳本、封存重播與最終測試log在validation-logs.tar.gz。

第42輪的protocol（含事前開發篩選數字的揭露）與凍結源碼在任何計時前提交；pilot、formal、profile、保留集依序執行並分開保存，profile只作階段診斷，保留集只驗精確性與GT。本輪沒有失敗門檻；母session一次與formal重疊的`find`掃描及round 43的短暫壓縮檢查記錄在判定檔與`validation/host-notes.md`，相關worker沒有刪除或重跑。開發篩選輸出、kernel與工具mutants的build／test log、Metal validation與ASan log都在validation-logs.tar.gz。

第43輪的protocol與凍結源碼在任何量測前提交，量測前看過的smoke與控制場景迭代寫在protocol的事前揭露中。GT階段（不作計時宣稱）與pilot／formal／profile分成兩個raw封存：raw-results-gt.tar.gz、raw-results-timing.tar.gz，各在100 MB以內。80筆GT退步全部來自k1敏感度與兩個消融負控制arm，正式只有1筆8K混合time_budget失敗，都保留在decision-and-failures.json。突變log、獨立稽核腳本與結果、封存重播與最終測試log在validation-logs.tar.gz。

第44輪的protocol與凍結源碼在任何量測前提交，事前揭露包含先前的整合重播與128物件每影格smoke。GT階段（不作計時宣稱）、pilot與profile、正式靜態、正式每影格混合、正式每影格完整GPU分成五個raw封存，各在100 MB以內。4筆`job_equals_round43`失敗（控制場景路徑不同、內容相同）保留在decision-and-failures.json；工具突變、獨立稽核、摘要重播與最終測試log在validation-logs.tar.gz。

第46輪的protocol與凍結源碼在任何評估前提交，事前揭露包含測試樣本集 v1 探針、產生器照明參數與只看管線的換種子smoke。241個worker（含既有四方法在改動前runner的對照）、遮罩與run manifest放在單一raw-results.tar.gz，每個成員的SHA-256在raw-files-sha256.json；不作時間宣稱。H1a、H2a、H3共8項未通過的檢查、6項預測的對照與事後診斷保留在decision-and-failures.json；獨立重算、摘要與job重播、事後診斷、編譯與工具突變及最終測試log在validation-logs.tar.gz。
