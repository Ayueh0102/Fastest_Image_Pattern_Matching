# 自適應矩形 NCC 上界：C++ 原型與採用界線

2026-10-16，第26輪。Apple M4 Pro、48 GiB，Release、NEON。基準 `3793e9e`，承接[原算法研究提案](ORIGINAL_ALGORITHM_NEXT_zh-TW.md)的實驗 A。本輪是灰階 CPU 原始金字塔搜尋，沒有使用稀疏方向前段或 GPU；不能和第25輪 Metal 稀疏搜尋時間直接相除。

## 結論

**數值與功能驗證通過，但不適合直接取代目前金字塔粗搜尋。** 大而容易以矩形表示的模板有收益；小模板或隨機紋理的上界成本反而較高。保留完整 C++ 原型、量測工具、正反結果，不改 GUI／API 預設，也不將這版直接搬到 Metal。

- 256×256結構模板、384×384來源：現有剪枝19.27 ms → 新global16 7.72 ms，約縮短60%，兩個順序區塊皆超過10%改善。峰值RSS 19.42 → 19.48 MiB。
- 同尺寸隨機模板：現有9.29 → 新19.60 ms；新上界沒有額外排除任何位置，增加的工作全部成為負擔。
- 完整128物件搜尋：現有544.62 → 新740.16 ms；block16亦為733.74 ms。所有方法仍找到128／128、零FP／FN，速度未達標。
- 完整6個場景的global16／block16均比現有慢，不能把局部勝利當成全流程加速。

本輪正式384個獨立程序、1920次暖機後量測；1408張診斷map、20,298,904個視窗比較，沒有誤刪達標分數、存活分數位元差異或選峰順序差異。視窗數包含不同方法與重複區塊，不是兩千萬個獨立真實物件。另有72個獨立CLI程序對照凍結基準。完整原始資料與失敗門檻見[摘要](adaptive-ncc/summary.json)、[驗證](adaptive-ncc/validation.json)、[封存](adaptive-ncc/raw-results.tar.gz)。

## 實作與數學

核心：[adaptive_ncc.hpp](../experiments/adaptive_ncc.hpp)、[adaptive_ncc.cpp](../experiments/adaptive_ncc.cpp)。`AdaptiveNccModel`持有自己的模板像素、均值與norm；不依賴呼叫者的生命週期。4→8→16→32→64個葉節點，每次用模板平方誤差下降最大的水平／垂直切分。候選計算由4塊逐步細分；移除父節點、加入子節點，不重算所有葉子。

矩形基底與相關近似的研究背景見[Segmented NCC](https://arxiv.org/abs/2502.01286)、[Briechle／Hanebeck](https://isas.iar.kit.edu/pdf/SPIE01_BriechleHanebeck_CrossCorr.pdf)。本次不是論文完整重現：使用greedy SSE切分、保守上界及原NCC驗分，沒有實作論文的merge流程，也不宣稱新的數學定理。

對矩形分區 `b` 定義 `n_b`、來源／模板總和 `S_xb,S_tb`、平方和 `Q_xb,Q_tb`。使用未去均值的非負8-bit數據，以減少區間乘法的符號分支：

```text
P   = Σ_b S_xb S_tb / n_b
R_x = Q_x − Σ_b S_xb² / n_b
R_t = Q_t − Σ_b S_tb² / n_b

raw_global_upper = P + sqrt(R_x R_t)
raw_block_upper  = P + Σ_b sqrt((Q_xb−S_xb²/n_b)(Q_tb−S_tb²/n_b))

<x,t> ≤ raw_block_upper ≤ raw_global_upper
```

這是正交投影分解加Cauchy–Schwarz；分區涵蓋常數向量，因此與提案的整體去均值推導等價。再減去原程式的`sum(source) × template.mean`，除以**原NCC函式實際算出的分母**。只有上界嚴格小於該層門檻才排除；存活者仍呼叫原整數dot product及原正規化。

有限精度處理不是固定`1e-9`裕量：

- 新上界的乘、加、減、除和平方根使用向外捨入區間；父節點移除使用完整區間相減，不能直接減一個未知量的上界。
- 殘差總和中保存的是「已算出的上界浮點值」之點區間，故移除該點值有效。原程式分子可能使用融合／分離乘減，中心項下界同時涵蓋兩者。
- 整數積分圖要求總面積不超過`2^53/65025`，IEEE binary64、round-to-nearest、非fast-math。異常分母、不支援門檻或形狀走原路徑；1維模板不建樹。非正門檻與大於1、非有限門檻停用新上界。
- 原常數模板／來源、近零能量和正負飽和規則保留。可用時仍接既有partial-dot剪枝；**新區間推導沒有把舊partial-dot的誤差裕量變成新證明**，另有直接只測新screen的測試。

`src/ncc.cpp`只新增內部可選screen掛點；一般`ncc()`仍傳空screen。新模型不進公開`MatchConfig`、C ABI、Session或GUI。完整搜尋的provider只處理coarse；fine的所有分數與subpixel鄰域保持原計算。不能將帶`−∞`的稀疏map直接交給subpixel擬合。

核心無Qt／OpenCV依賴；量測worker用Qt讀圖與JSON。runner的RSS介面目前限Unix。C++庫及測試保留一般CMake建置，Windows／SSE2本輪沒有執行；MSVC設定`/fp:strict`，其他編譯器停用上界檔案的FMA contraction。

## 事前協定與完整搜尋結果

[protocol](adaptive-ncc/protocol.json)在測量前保存輸入、源碼快照與binary SHA256。20個場景：10個實圖原解析度7×7 response map、2個頂層map、2個大模板dense map、6個完整旋轉搜尋。map比較full、legacy、global／block各4／8／16／32／64；完整搜尋固定比較full、legacy、global16、block16，沒有看到結果後再挑最佳cap。

每組兩個新程序，方法先正序後反序；每程序先記首次呼叫，再至少0.2秒暖機、量5次。計時不含解碼、建模、map來源金字塔、hash、統計及oracle，完整搜尋則包含自己的來源金字塔、旋轉、fine和NMS。建模時間另外保存。每個程序在oracle／profile之前擷取RSS；沒有計時時收集逐視窗統計。以下為10次正式時間中位數，ms：

| 完整搜尋 | 完整NCC | 現有剪枝 | global16 | block16 |
|---|---:|---:|---:|---:|
| 小圖、360°、1物件 | 9.13 | 9.00 | 75.72 | 55.69 |
| 六物件、±20° | 27.28 | 26.86 | 94.17 | 79.69 |
| 低對比、2物件 | 9.37 | 9.38 | 41.43 | 39.81 |
| 細線、1物件 | 2.39 | 2.43 | 9.11 | 5.33 |
| Src4實圖、無GT | 58.05 | 58.18 | 339.03 | 291.59 |
| 128物件、±20° | 549.69 | 544.62 | 740.16 | 733.74 |

所有完整最終輸出的分數、位置、角度、角點及順序一致。5個有合成GT場景共138個姿態，四路全部零FP／FN；128場景最大位置誤差0.2652px、角度誤差0.7778°，尺度固定1。GT匹配為獨立一對一最大匹配，門檻位置≤2px、角度≤1°、尺度誤差≤2%。沿用已測的原始素材與合成標註，不是新的盲測；名稱含`heldout`僅為歷史檔名。Src4有24個一致輸出，沒有獨立GT，不能宣稱24個皆正確。

峰值RSS是每路兩個程序峰值的中位數，MiB：

| 完整搜尋 | 完整NCC | 現有剪枝 | global16 | block16 |
|---|---:|---:|---:|---:|
| 小圖 | 17.35 | 17.36 | 17.48 | 17.20 |
| 六物件 | 19.59 | 19.59 | 19.34 | 18.80 |
| 低對比 | 17.81 | 18.19 | 17.83 | 17.81 |
| 細線 | 17.29 | 17.44 | 17.37 | 17.38 |
| Src4 | 17.75 | 18.10 | 18.55 | 18.52 |
| 128物件 | 30.67 | 31.42 | 29.23 | 31.46 |

記憶體沒有系統性下降；小差異含程序配置波動。128場景頂層16×12，新global16模型常駐名義容量3544bytes、建模暫存4112bytes，新增約0.0036ms建模。這次沒有完整4K／8K搜尋測試；最大完整場景1536×768，前輪大圖結果不能當作本輪驗收。

## 為什麼「乘積少很多」仍變慢

128場景的獨立分段診斷（在正式計時之後，單次profile）顯示：

| 項目 | 現有剪枝 | global16 | block16 |
|---|---:|---:|---:|
| coarse NCC ms | 8.48 | 201.96 | 195.65 |
| coarse dot像素數 | 104,489,280 | 9,455,616 | 4,588,032 |
| 新矩形查詢數 | 0 | 8,022,660 | 5,778,484 |
| fine旋轉 ms | 287.83 | 288.73 | 288.62 |
| fine NCC ms | 207.02 | 217.35 | 209.90 |
| fine maps | 29,143 | 29,143 | 29,143 |

細搜尋工作量沒有減少。頂層模板只有192像素，SIMD整數乘積已很便宜；為避免這些乘積而新增矩形查詢、區間更新及平方根並不划算。block界雖更緊，也沒有扭轉全流程結論。尚未以硬體profile分別歸因每個`nextafter`或cache成本，不能聲稱已量出個別指令占比。

**這是CPU計算成本問題，這輪沒有CPU↔GPU傳輸或等待。** 結果不能用來判定GPU不適合此任務，也不表示所有保守篩選法都失敗。

## 大模板局部結果與限制

下表固定global16，非逐例選最快cap；完整12方法結果均保留：

| map工作 | 完整NCC ms | 現有剪枝 ms | global16 ms | 現有／新RSS MiB |
|---|---:|---:|---:|---:|
| Dst1、466×135、49位置 | 0.1605 | 0.2269 | 0.1603 | 41.47／41.05 |
| Dst6、848×446、49位置 | 1.1608 | 1.3683 | 0.8994 | 71.97／77.86 |
| Dst7、762×521、49位置 | 1.2256 | 1.4225 | 0.9767 | 82.23／88.87 |
| Dst4、33×17、49位置 | 0.0020 | 0.0023 | 0.0075 | 17.23／16.84 |
| Dst3頂層18×13 | 0.6203 | 0.6170 | 12.7320 | 26.02／26.20 |
| 結構256²、16641位置 | 31.8191 | 19.2678 | 7.7165 | 19.42／19.48 |
| 隨機256²、16641位置 | 32.2981 | 9.2928 | 19.5999 | 19.48／20.66 |

實圖49位置取固定左上角裁切，**不是完整fine實際遇到的高分候選分布，也沒有將這方法接入fine**。積分圖仍在每次NCC呼叫建立；RSS包含先解碼整張原圖的峰值，不能當作49位置本身的記憶體需求。

Dst6模型常駐381560bytes、建模暫存6072624bytes、建模約0.625ms；此成本不含在0.899ms暖機時間。一個模型只用一次時，總成本可能抵消收益。256²模型常駐68888bytes、建模暫存1057360bytes，global16建模約0.085ms。這些是容器容量與像素大小，不含allocator metadata。

結構案例global16把15999／16641個位置提前排除，dot像素從668M降到42M。隨機案例新上界零排除，最後仍靠舊partial-dot排除16640個位置，所以新方法只增加成本。更多矩形也不是越快：結構例global4無改善，global8／16兩block有改善，global32／64比16慢。

性能容忍為每個配對block不慢於`baseline + max(10%,2ms)`、RSS不超過`baseline + max(10%,32MiB)`；採用還要求兩block皆至少快10%。本輪**83個速度容忍失敗、0個RSS容忍失敗**全部保存；通過容忍不等於有加速。未因局部勝利調整正式全流程設定。

## 測試、凍結比較與重現

- Release：40／40，包含原有CPU、Metal、GUI；新測試2259組map、36組完整搜尋，涵蓋pyramid、旋轉、反相、stopLayer、subpixel、重疊與排序。
- 新screen獨立測試10800次門檻檢查、2793次安全排除；含門檻相等與相鄰double。另覆蓋常數、低對比、padded stride、大於32-bit乘積、模型自有像素、變動來源、非有限門檻及非nearest模式。
- ASan／UBSan：針對變動核心、C ABI及新實驗共3／3。首次設定`detect_leaks=1`三項均在啟動時因Apple runtime不支援而中止；改成`detect_leaks=0`後通過。**沒有LeakSanitizer通過聲明**，原失敗log保留。
- 獨立驗證重建summary，核對48個輸入／來源hash與worker清單；7個蓄意破壞控制均拒絕，包括偽造成功旗標下的錯誤刪除、分數不一致、計數錯誤、錯route、非有限時間等。
- 額外由Git `3793e9e`原碼在獨立目錄重建CLI，6場景×兩順序×兩版×3新程序＝72。一般未啟用screen的CLI輸出皆相同、12個時間容忍全過。此CLI輸出只序列化12位有效數，不稱為位元比較；計的是每個程序首次搜尋，亦不稱為暖機速度。

完整source與CMake留在repo。協作agent分工提供測試與runner初稿、數學設計檢查；服務額度中止後由主agent接手完成驗證，沒有把中止稱為完成獨立最終審查。

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake -S cpp-core -B build/adaptive-ncc \
  -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_BUILD_GUI=ON -DFPM_WITH_OPENCV=ON
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build build/adaptive-ncc -j 6
ctest --test-dir build/adaptive-ncc --output-on-failure -j 1
python-port/venv/bin/python cpp-core/tools/benchmark_adaptive_ncc.py prepare --out build/adaptive-ncc-rerun
python-port/venv/bin/python cpp-core/tools/benchmark_adaptive_ncc.py run --out build/adaptive-ncc-rerun
python-port/venv/bin/python cpp-core/tools/validate_adaptive_ncc_study.py build/adaptive-ncc-rerun
```

prepare使用第25輪protocol指定的既有合成輸入；換電腦可先解開本輪封存的`inputs`並修正前輪素材位置，或以本輪封存protocol和匹配hash的binary指定`run --binary`。worker的job位置由study目錄重新解析；GT不送入worker。封存含相關source快照，完整建置請使用此repo，快照不是所有專案依賴的獨立發行包。原始絕對路徑屬量測時的定位記錄。

## 下一個算法實驗

優先進入提案B：**fine的49個相鄰位置共用正規化統計與來源讀取**。先固定旋轉、候選、精度和排程，只改等價計算，分別量fine kernel與完整流程。128場景現有fine旋轉＋NCC約495ms，比coarse NCC約8ms更值得優先處理。

A保留兩條後續研究：模板可壓縮性／成本門檻，與低成本但有完整誤差推導的上界。尚未實作或驗證；不能為了消除慢結果直接跳過測試或以近似分數砍候選。Metal分歧／壓縮候選、IC／ECC、旋轉基底與FFT依舊是各自獨立的研究題目。
