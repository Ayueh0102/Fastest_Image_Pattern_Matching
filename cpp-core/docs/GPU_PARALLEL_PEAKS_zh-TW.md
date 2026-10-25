# 完整GPU粗選峰改為精確平行回合

第42輪，2026-10-24，Apple M4 Pro／48 GiB。第40輪profile顯示粗選峰占完整GPU時間的28–34%，而且是逐次依賴的單一工作群組計算。本輪新增`full_gpu_peak_mode=2`：以精確的平行回合取代逐次「選最大→抑制」，並讓多張分數圖同批選峰。protocol與凍結源碼在任何計時之前提交；pilot、正式、profile、保留集與全部門檻都保留。mode 0／1與所有預設不變。

**結論：mode 2的完整有序輸出與mode 1逐位相同，原128／512完整GPU時間各少27.3%／34.4%（807.81→587.22ms、2868.42→1881.34ms），6/6 block同向，達成事前登錄的10%目標。** 每次換新影格時，fine4＋mode 2在原128比混合快31.4%、512快22.7%（各6/6），這是完整GPU第一次在512物件勝過混合；六物件仍以混合快1.5%，8K快80.1%。依protocol規則，完整GPU研究建議改為`full_gpu_peak_mode=2`；GUI、核心與混合預設不變。

## 正式結果

靜態範圍（第40輪job：首次呼叫、1秒暖機、3次正式呼叫，同一張來源）比較兩種peak mode；兩者每次呼叫都重建全部來源相關狀態，範圍對稱。每格為6個新程序×3次正式呼叫共18次的中位數，各block交錯順序。

| 場景 | mode 1 ms | mode 2、批次1 ms | mode 2、批次8 ms | 批次8對mode 1 | 6個block | 批次1對mode 1 | 批次8對批次1 |
|---|---:|---:|---:|---:|---|---:|---:|
| 原128物件 | 807.81 | 643.64 | **587.22** | −27.3% | 6/6，−28.0～−26.9% | −20.3% | −8.8% |
| 原512物件 | 2868.42 | 2046.07 | **1881.34** | −34.4% | 6/6，−35.5～−32.5% | −28.7% | −8.0% |
| 六物件控制 | 70.62 | 71.42 | **63.19** | −10.5% | 6/6，−11.5～−8.5% | +1.1%（6/6較慢） | −11.5% |
| 8K、256模板、全角三尺度 | 61.20 | 59.66 | **47.19** | −22.9% | 6/6，−25.4～−19.0% | −2.5%（5/6，方向未定） | −20.9% |

批次1只用一張分數圖（分數圖配置與mode 1相同，總命名配置還少約2～6MiB）；它顯示多數收益來自平行回合本身（128／512為−20.3%／−28.7%），多張圖同批選峰再少約8%。小圖批次1沒有收益（六物件慢1.1%，在時間容忍內），批次8才轉為較快。[正式摘要](stage42-parallel-peaks/formal-summary.json)

每換一張影格（第41輪序列模式，每次呼叫都經過`updateSource`；混合每次重建pose context）：

| 場景 | 混合每影格ms | fine4＋mode 2每影格ms | 差 | 6個block |
|---|---:|---:|---:|---|
| 原128 | 857.72 | **588.32** | −31.4% | 6/6，−33.0～−30.4% |
| 原512 | 2433.47 | **1880.45** | −22.7% | 6/6，−26.8～−22.1% |
| 六物件 | **62.84** | 63.78 | +1.5% | 6/6混合較快，+0.6～+2.6% |
| 8K | 254.87 | **50.67** | −80.1% | 6/6，−80.4～−79.7% |

同建置靜態範圍下，mode 2對混合為−29.0%／−20.9%／+6.4%／−80.7%；mode 1對混合為−2.3%／+20.6%／+18.9%／−75.0%，與第40／41輪的方向相同。跨輪數字來自不同建置，只作方向對照，不相乘。

依事前規則，pooled中位數與6個成對block全部同向才判定較快；上述每組都符合，只有8K批次1對mode 1為5/6，不判定方向。正式144程序、432次正式呼叫、1200次直接GT全部零FP／FN；沒有任何門檻失敗，包括對mode 1與對混合的時間／RSS容忍。[判定與失敗](stage42-parallel-peaks/decision-and-failures.json)

### 記憶體

| 場景 | peak RSS MiB：mode 1／批次1／批次8 | 命名Metal MiB：mode 1／批次1／批次8 | 其中分數圖／選峰暫存 MiB（批次8） | 混合RSS 靜態／每影格 |
|---|---|---|---|---|
| 原128 | 91.12／90.98／91.14 | 116.47／114.48／139.73 | 14.91／13.95 | 229.84／263.72 |
| 原512 | 164.88／164.86／164.69 | 236.30／230.73／282.04 | 34.34／24.29 | 441.86／506.09 |
| 六物件 | 27.12／27.05／26.80 | 6.15／6.79／12.85 | 0.60／6.32 | 38.66／42.27 |
| 8K | 280.14／279.95／280.03 | 161.50／159.49／184.83 | 15.00／13.97 | 372.81／501.44 |

批次8多保留7張分數圖、雙狀態陣列、removal bitmask與候選清單，命名Metal配置在128／512多23.26／45.74MiB；批次1反而少1.99／5.57MiB（不需mode 1的排序暫存）。程序peak RSS三者差在0.5MiB內，私有Metal buffer容量與RSS是不同範圍，不能相加。每影格arm的RSS較高，包含每次重新解碼與runner保留全部輸出列。

## 階段診斷

下列為另跑的profile phase（encoder邊界timestamp，1 block、2次正式呼叫），是診斷，不參與時間判定。

| 階段ms | 128 mode 1 | 128 批次1 | 128 批次8 | 512 mode 1 | 512 批次1 | 512 批次8 |
|---|---:|---:|---:|---:|---:|---:|
| 粗選峰 | 262.40 | 73.17 | **12.44** | 989.82 | 202.52 | **33.50** |
| 粗NCC | 37.77 | 70.32 | 43.81 | 76.49 | 168.14 | 93.43 |
| 細搜尋 | 431.61 | 431.44 | 427.88 | 1327.33 | 1341.64 | 1326.52 |
| 初次LM精修 | 80.72 | 78.73 | 80.88 | 271.32 | 274.52 | 277.63 |
| Command buffer GPU區間 | 835.54 | 651.92 | 599.35 | 2783.81 | 2023.88 | 1867.79 |
| 階段和－GPU區間 | +29.08 | +52.99 | +17.25 | +60.26 | +574.71 | +38.91 |

粗選峰由占GPU區間31.4%／35.6%降到2.1%／1.8%（批次8）；細搜尋現在占約71%，是下一個主要瓶頸。六物件與8K的粗選峰為9.34→1.78、33.44→3.61ms。

階段和大於GPU區間，表示彼此沒有資料相依的encoder在GPU上重疊執行，各自的timestamp區間重複計入。批次1的512最明顯：`pose_prepare`量到438.52ms（其他方法約2.5ms），階段和多出574.71ms；粗NCC的增加也可能部分來自等待前一張圖的選峰。因此階段值只能看相對大小，整段時間以command buffer GPU區間與正式`total_ms`為準。[profile摘要](stage42-parallel-peaks/profile-summary.json)

## 保留場景

第40輪的七個保留場景（本輪開發時沒有使用）各跑2個block、2次正式呼叫：42程序、126次直接GT全部零FP／FN，mode 2兩種批次的完整有序matches／proposals都與mode 1相同。

| 場景 | 物件 | mode 1 ms | 批次1 ms | 批次8 ms |
|---|---:|---:|---:|---:|
| heldout12 | 12 | 98.62 | 96.17 | 83.87 |
| small_fullres | 2 | 8.16 | 8.11 | 7.79 |
| boundary_one | 1 | 0.39 | 0.40 | 0.41 |
| empty_scene | 0 | 0.24 | 0.24 | 0.26 |
| constant_source | 0 | 0.23 | 0.24 | 0.26 |
| many128_holdout | 128 | 793.13 | 631.88 | 573.63 |
| many512_holdout | 512 | 2788.26 | 2035.57 | 1872.86 |

依protocol，保留集的時間只報告、不設門檻。三個不到1ms的極小場景，批次8多0.02ms左右（+4%～+13%），是多一次批次dispatch與初始化的固定成本。[保留集摘要](stage42-parallel-peaks/holdout-summary.json)

## 方法

### 原本為何慢

第40輪profile（`full_gpu_simd`）中，粗選峰為229.8／934.3ms，粗NCC只有33.7／72.4ms。原因有二：

1. 所有角度／尺度的分數圖共用一個`coarse-scores` buffer，下一張圖必須等上一張選完才能寫入，many512的143張圖完全串行。
2. `fg_coarse_peaks_tiled`每張圖只用一個256-lane工作群組，做最多`limit = maxTargets+5`次互相依賴的「選最大→抑制矩形」，many512每張圖773次。

### 精確平行回合

候選是分數有限且不低於門檻的點；任何`+INFINITY`都會讓mode 0／1第一次就停止且沒有輸出，mode 2照做。優先順序與原本相同：FP32分數由大到小（−0等於+0），同分時原始索引由小到大（`fg_better`）。

原本逐次貪婪選出的集合G滿足：候選p屬於G，若且唯若沒有任何優先更高的G成員，其抑制矩形涵蓋p。每一回合：

- 接受所有「沒有任何更高優先且仍存活的候選涵蓋它」的存活候選；
- 再移除被「本回合接受、且優先更高的點」涵蓋的存活候選。

矩形由`trunc(x−rw)`與`int(2·rw)`決定，左／上邊界會往右／下偏一格，因此不對稱：低優先的點可能涵蓋高優先的點。被接受的點只能移除比自己低優先的點；若把矩形內較高優先的點也刪掉，結果就錯。發起本輪的session曾以Python probe（用OpenCV近似重建many128／many512的最上層分數圖，只看結構、不計時）比較：天真版本25張圖全部與逐次結果不一致，正確規則50張圖0不一致，每張3–8回合，逐次則為300–773次。本輪重跑該probe的many512部分得到相同結論（抽樣25張、0不一致、平均4.0回合、最多8回合），輸出存於validation封存的`dev-screens/`。

正確性：被接受的點，所有可能涵蓋它的更高優先點都已不存活——不是被移除（不在G），就是已在G卻沒有涵蓋它（否則它早被移除）——所以它屬於G。被移除的點被G中更高優先的點涵蓋，不屬於G。每回合最高優先的存活點必被接受，所以一定前進。

停止：已接受至少`limit`個，且沒有存活候選的優先高於第`limit`個已接受點時，G的前`limit`名就已確定，因為之後接受的點都來自存活集合。輸出依優先順序寫入與mode 1相同的slot（`first+k`），座標、分數與`order`使用相同的FP32算式。

### 實作

- **分數ring**：`full_gpu_peak_batch`張圖（預設8，可設1–16）各寫入ring的一格；滿一批或到一個尺度的最後一張圖時，一次dispatch選完整批，每張圖一個1024-lane工作群組。批次1只用一格，作為消融對照。
- **雙緩衝狀態**：每個phase只從一個狀態陣列讀鄰點、只寫自己負責項目在另一陣列的值，phase之間以threadgroup barrier分隔；分數本身不修改。被接受點的抑制以atomic OR寫入removal bitmask，再由各點自己套用。
- **工作量**：選取時先在lane內檢查8鄰點；否則由整個SIMD group分攤鄰域，中心列優先、上下交替。候選不超過65536個時，各回合只走存活清單與已接受清單；超過則每個phase掃整張圖。兩條路徑判斷完全相同，測試兩者都覆蓋。
- **輸出排序**：已接受數超過`limit`時，以64-bit鍵（分數有序位元＋索引）做基數選擇找出第`limit`名，再以兩兩比較算出每個輸出的名次。
- 仍是一次提交、最後一次等待；中途零讀回、零CPU算法步驟。`full_gpu_peak_batch`只接受於mode 2，否則明確拒絕。

### 開發過程與事前篩選

下列為事前登錄前的開發篩選：開發建置、單程序、開啟profiling，且當時有其他session在同一台機器工作。只用來決定kernel設計、預設批次8與批次1消融，不作任何判定證據；各版本對mode 1都完全一致。

| 版本 | 128 總ms／粗選峰ms | 512 總ms／粗選峰ms |
|---|---|---|
| mode 1 | 797.9／248.0 | 2778.6／987.6 |
| v1：每lane自行掃鄰域，批次8 | 853.2／282.7 | 2663.5／837.0 |
| v2：SIMD group分攤鄰域 | 642.7／71.7 | 2020.0／195.1 |
| v3：加8鄰點捷徑與抑制散射 | 606.3／35.8 | 1943.4／114.7 |
| v4：候選清單（正式版本） | 583.3／12.3 | 1858.5／33.4 |

v1比mode 1還慢：每個lane各自序列掃描約945格鄰域，SIMD分歧讓多數group都跑完整掃描。批次1／2／4／8／16的128總時間為630.6／600.1／592.1／583.3／578.8ms，512為2002.5／1913.7／1878.2／1858.5／1886.4ms，8之後沒有再改善，因此預設8。數字保存在validation封存的`dev-screens/`。

## 量測方法與界線

- 階段順序pilot→formal→profile→保留集，每個都在前一個結束後才開始；每個（場景、arm、block）一個新程序，偶數block依宣告順序、奇數反向。2026-10-24 13:28:24～13:46:40 UTC。
- 靜態時間為`total_ms`（第40輪範圍：只算執行，排除解碼、Session建構、JSON與GT）；每影格為`frame_wall_ms`（`updateSource`＋執行）。每影格arm的暖機次數沿用第41輪事前規劃（2／1／26／23），每個程序都通過至少1秒的暖機門檻。
- GT在父程序獨立評估，從不傳給搜尋器：靜態arm的首次與正式呼叫、每影格arm的冷、暖機與正式呼叫。四個phase共1538次直接GT（formal 1200、pilot 176、profile 36、保留集126）全部零FP／FN；最大誤差0.128px／0.0995°／0.274%（512），mode 2與mode 1完全相同。
- 與GT分開：mode 2三個arm每一次呼叫的完整有序matches與proposals，都與同場景mode 1靜態block 0首次呼叫逐值相同，四個phase全部通過。
- **主機干擾揭露**：本輪開始前以共享筆記的lock檔與其他session協調。母session回報，13:30:27～約13:33 UTC曾執行一次`find /Users/Ayueh -maxdepth 6 -name 'timing-lock-*'`（單一程序、磁碟I/O與少量CPU），與formal的many128全部36個worker及many512前2個重疊。driver的程序檢查不會比對到`find`，所以沒有`host_busy`記錄。這些worker沒有刪除或重跑：每個block內各arm交錯共享同一時段，many128各block中位數範圍很窄（mode 2為583.5～590.5ms、mode 1為801.7～812.7ms）；在干擾之前的pilot（−26.7%）、之後的profile（−28.3%）與保留集many128_holdout（−27.7%）方向與幅度一致。round 43回報其最後一次約20秒的壓縮檢查約在13:28結束，最多與pilot開頭重疊。受影響worker清單在[判定檔](stage42-parallel-peaks/decision-and-failures.json)。
- driver在每個worker前記錄負載與競爭程序：所有phase都沒有競爭的建置或benchmark程序；1分鐘load average在干擾時段最高6.28，其餘多在2～4。
- 稽核另指出三點，均不影響判定：manifest的源碼指紋涵蓋快照289檔中的275檔（根目錄CMakeLists.txt、LICENSE、vcpkg.json、macos-cli等不在指紋規則內，與第41輪相同），runner與dylib另有直接雜湊；每影格程序的peak RSS取`final_memory`，比釋放後的紀錄最多少80KiB；靜態暖機依runner牆鐘1秒，小場景的暖機算法時間合計最低約978ms。

界線：只有一台M4 Pro；合成場景與第40輪保留集，不是實拍標註序列；每影格arm重複同一張影像，不是變化中的相機串流；粗選峰以外的階段（細搜尋占約71%）沒有改動。

## 驗證

| 檢查 | 結果與範圍 |
|---|---|
| 差分測試（新） | 170組fixture（含第38輪64組）× lane 256／512／1024 × 清單容量0／7／65536，再加3輪每批最多16張圖的混合批次；有序候選bytes與mode 1相同、與獨立純量貪婪參考相同、ring分數未被修改 |
| 全流程與常駐 | pipeline測試加入mode 2、批次1／3／16、全圈多批次與第2批的90°物件，結果與mode 1逐值相同；攔截Metal呼叫的常駐測試72次呼叫（含mode 2）皆一次提交、一次等待 |
| kernel／host mutants | 16個編譯突變殺掉15個；存活的M05（8鄰點捷徑忽略矩形）與輸出等價：選取階段的「誤判被覆蓋」只會延後接受、不會移除任何點，而最高優先的存活點永遠不會被誤判，所以只改變回合數 |
| 量測工具 | 20項負控制；17個手動突變（精確性、GT範圍、容忍、目標、方向、批次界限、job、host、暖機、parity、時間範圍、保留集門檻、phase設定）全部被抓到 |
| 量測前Release | 全新建置：ctest 77／77、Python harness 590／590、color verify `--metal` 5／5；Metal API＋Shader Validation 21／21 |
| ASan／UBSan | 另建Debug＋`FPM_SANITIZERS`＋`FPM_DISABLE_SIMD`＋experiments（無GUI），72／72，含新差分與mode 2全流程測試；未啟用LeakSanitizer，不當洩漏證明 |
| 摘要重算 | 四個phase由manifest以`--summarize`重算，與量測當下內嵌摘要逐項相同 |
| 獨立稽核 | 另一個agent只依protocol撰寫腳本，不匯入或閱讀量測harness、不啟動GPU：重算222程序444檔雜湊、truth隔離、執行順序、自行實作的一對一GT（1538次）、與mode 1的完全一致、常駐route、各arm與block中位數、7組比較、時間／RSS預算與主要目標，6296個數值與發佈摘要零不符；另以13種記憶體內竄改自我測試，全部被抓到 |
| 封存重播 | 解壓源碼快照（289檔雜湊全符）與兩個raw封存，用凍結工具重算四份摘要，與發佈檔逐位相同；不重跑GPU |
| 量測後Release | 14:02 UTC、HEAD `1e23ec5`（工作樹只有文件、Log分類器與新證據不同）：ctest 77／77、Python harness 590／590、color verify `--metal` 5／5 |

## 使用與完整源碼

完整GPU研究設定：`full_gpu=true, full_gpu_peak_mode=2, full_gpu_fine_mode=4`；`full_gpu_peak_batch`可省略（8）或設1–16，1不增加記憶體。其他設定沿用[第40輪](GPU_NCC_CLOSURE_zh-TW.md)。mode 0／1保留為可重現對照；混合路徑與GUI預設不變。

| 職責 | 來源 |
|---|---|
| 平行回合kernel、批次ABI | [Metal kernels](../experiments/full_gpu_peaks.metal)、[ABI](../experiments/full_gpu_peaks_types.h) |
| ring配置、批次dispatch與量測欄位 | [Metal host](../experiments/full_gpu_metal.mm)、[研究API](../experiments/research_pipeline.hpp)、[runner](../tools/native_pipeline_runner.cpp) |
| 差分、全流程與常駐測試 | [平行選峰差分](../tests/full_gpu_parallel_peaks_test.mm)、[pipeline](../tests/full_gpu_pipeline_test.cpp)、[常駐攔截](../tests/full_gpu_residency_test.mm) |
| 量測與負控制 | [量測工具](../tools/benchmark_parallel_peaks.py)、[負控制](../tests/test_parallel_peaks_harness.py) |

從repo根目錄，以有NumPy／OpenCV的Python與Release研究建置（GUI＋experiments＋OpenCV）重跑：

```sh
cd cpp-core/tools
python benchmark_parallel_peaks.py --phase formal --build ../../build/stage42-release \
  --output ../../build/stage42-parallel-peaks/formal/manifest.json
python benchmark_parallel_peaks.py --summarize ../../build/stage42-parallel-peaks/formal/manifest.json \
  --output ../../build/stage42-parallel-peaks/formal-summary.json
```

`--phase`另有`pilot`、`profile`、`holdout`；保留集預設讀第40輪的`holdout-fixtures.json`。離線重算摘要（不需GPU）：

```sh
mkdir -p build/stage42-repro/source build/stage42-repro/raw
tar -xzf cpp-core/docs/stage42-parallel-peaks/source-snapshot.tar.gz -C build/stage42-repro/source
tar -xzf cpp-core/docs/stage42-parallel-peaks/raw-results-formal.tar.gz -C build/stage42-repro/raw
tar -xzf cpp-core/docs/stage42-parallel-peaks/raw-results-other.tar.gz -C build/stage42-repro/raw
cd build/stage42-repro/source/cpp-core/tools
python benchmark_parallel_peaks.py --summarize ../../../raw/formal/manifest.json --output ../../../formal-summary.json
cmp ../../../formal-summary.json ../../../../../cpp-core/docs/stage42-parallel-peaks/formal-summary.json
```

validation封存的`replay_check.py`對四個phase做同樣的重算與逐位比較。輸入與第40輪[fixtures.tar.gz](stage40-ncc-closure/fixtures.tar.gz)逐位相同，本輪只保存雜湊。重現請用凍結的[source-snapshot.tar.gz](stage42-parallel-peaks/source-snapshot.tar.gz)，不要把日後HEAD當成同一份源碼。

## 封存

[封存清單與SHA256](stage42-parallel-peaks/archive.json)：

| 檔案 | 內容 |
|---|---|
| [protocol.json](stage42-parallel-peaks/protocol.json) | 計時前提交的規格：方法、arm、phase、暖機、門檻、判定規則與開發篩選揭露 |
| [source-snapshot.tar.gz](stage42-parallel-peaks/source-snapshot.tar.gz) | 凍結commit `4b0ad8f`的289個已提交源碼檔與雜湊清單 |
| [raw-results-formal.tar.gz](stage42-parallel-peaks/raw-results-formal.tar.gz)、[raw-results-other.tar.gz](stage42-parallel-peaks/raw-results-other.tar.gz) | formal與pilot／profile／保留集的manifest、job、stdout／stderr與每個程序的完整結果；分成兩個檔以保持在GitHub單檔上限之下 |
| pilot／formal／profile／holdout `-summary.json` | 可由封存重算的摘要 |
| [decision-and-failures.json](stage42-parallel-peaks/decision-and-failures.json) | 判定、各phase全部失敗門檻（本輪0筆）與主機干擾揭露 |
| [validation.json](stage42-parallel-peaks/validation.json)、[validation-logs.tar.gz](stage42-parallel-peaks/validation-logs.tar.gz) | 建置、測試、mutants、Metal validation、ASan、獨立稽核、重播與封存腳本 |
| [input-sha256.json](stage42-parallel-peaks/input-sha256.json) | 輸入雜湊，與第40輪清單相同 |

## 未涵蓋

細搜尋（現占GPU約71%）與粗NCC沒有改動；CUDA、其他Metal裝置、非32寬SIMD與Windows／Linux沒有執行。RGB驗色與稀疏路徑不在完整GPU範圍。實拍串流與變化內容的影格序列沒有驗收。本輪不改GUI或通用預設，完整GPU仍是明確指定的研究選項。
