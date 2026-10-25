# 穩定核心窮舉NCC：滑動整數和與8視窗點積

2026-10-25，Apple M4 Pro／48 GiB，Apple clang 21.0.0，Release。接續[精確旋轉與金字塔](CORE_WARP_PYRDOWN_zh-TW.md)：該次profiling後`computeNcc`在Src5／Src10占82～86%。這次只改`cpp-core/src/ncc.cpp`的預設窮舉路徑，演算法、API、預設參數與所有分數都不變。這不是研究輪次，比較基準是同一台機器上的`65b7dad`。

**結論：輸出與`65b7dad`逐位元相同（binary64，544次執行零差異，包括純scalar建置）；Src7模式基準快37.6%（245.72→153.22ms，三個交錯session各快37.0～38.4%）；17張固定影像每張每輪都較快，幾何平均快56.1%。** 同一個session裡，舊OpenCV＋DOTPROD版是186.22ms，穩定核心現在比它快17.7%。在稽核設定下，17張影像也全部比舊版快。

## 先量到哪裡慢

用插樁的scratch副本（只供profiling）分段計時：

- **Src5、Src10幾乎都在頂層視窗迴圈。** 頂層模板寬只有10和14，小於16，所以點積走純量迴圈（Src10：37ms裡約33ms）。
- **Src7、Src6、Src1的細層：** 約一半時間在binary64積分圖。積分圖每次要為整個ROI每像素寫16 bytes，實際卻只用到7×7＝49個視窗。另一半是點積。L0模板（Src6為848×446，約378KB）超過L1，49個視窗每個都從L2重讀整個模板。

第一個pilot（4列交錯的整窗UDOT＋uint64積分圖）對窄模板很有效，但Src7／Src6只快11%：L0的點積仍受頻寬限制。這個結果決定了最後的設計。所有pilot輸出逐字記在[pilot紀錄](core-exhaustive-ncc/pilot/pilot-console-transcript.txt)。其中兩次早期量測是無效的：一次是重複的點積呼叫被編譯器合併，另一次是插樁修補沒有套用。兩次都已註明，並由後一次量測取代。

## 改了什麼

`computeNcc`遇到預設窮舉情況時改呼叫新的`exhaustiveNcc`。條件是：

- 來源仍在binary64的整數範圍內；
- 沒有界限剪枝，也沒有screen；
- 模板高度不超過65536。

有界限或screen的路徑是原封不動的舊程式碼。

- **視窗和與平方和：** 用uint32欄和逐列滑動，再用uint64橫向滑動，取代binary64積分圖。原本積分圖每一格、以及四格相減的結果都是小於2⁵³的精確整數，所以轉成double後的值相同。之後的變異數、分母、正規化都是同一段double運算式。
- **點積（UDOT建置、`Backend::Auto`、模板寬不超過4096）：** `windowDot8`一次算同一列中8個相鄰視窗，每個模板16-byte區塊只載入一次。
  - 模板每列補零到16的倍數；來源複製成緊密排列、尾端補零的私有緩衝區，所以每列多讀最多22 bytes也不會越界，多讀的部分乘到的是0。
  - 8個視窗的分母都是0的那一組直接跳過，這和原碼「分母為0不算點積」一致。
  - 每個累加器在兩次flush之間少於256＋blocks次UDOT，也就是少於8192個乘積，四個lane加總也小於2³²。
  - 原碼把每列的整數點積逐列以double相加；每個部分和都是小於2⁵³的整數，加法是精確的，所以和整數總和轉一次double的值相同。
- **其他建置：** `Backend::Scalar`、無DOTPROD的NEON、SSE2也走滑動和，點積仍是原本逐列的寫法。

## 驗證

| 檢查 | 結果 |
|---|---|
| Release ctest | 3/3；`portable_core`含新測試`NCC/exhaustive/exact-reference` |
| `-march=armv8-a`（滑動和＋逐列NEON點積） | 3/3 |
| Release＋`FPM_DISABLE_SIMD`（滑動和＋scalar點積） | 3/3 |
| ASan／UBSan scalar與開SIMD（Debug） | 各2/2 |
| 完整research建置ctest | 73/73（含adaptive NCC、coarse各實驗對`detail::ncc`的比較） |
| Python harness | 570項OK |
| [`check_backend_identity.py`](../tools/check_backend_identity.py)：17組×4模式×4個建置（`65b7dad`、新版、新版無DOTPROD、新版純scalar）×SIMD／scalar | 544次執行、724筆結果，與`65b7dad`的scalar binary64全部相同 |
| `compare_versions.py`（17張影像） | 新舊都是17/17在容許值內；新舊portable輸出17/17相同 |
| 編譯突變（7個） | 7/7被抓到 |

新測試拿逐字複製的原窮舉NCC（binary64積分圖＋逐列double相加）當參考，以`memcmp`比較`Backend::Scalar`與`Backend::Auto`的整張分數圖：

- 1500次隨機試驗：模板寬1～160，涵蓋小於16、16、17、32、33與多個block；高1～75。
- 來源stride有padding，也有緊貼的情況。
- 資料有接近飽和、低對比與隨機三種。
- 另有接近飽和的4096×20與4097×20寬列：前者走8視窗kernel，後者不走。乘積總數超過單一uint32能裝的量，所以flush必須正確。

7個突變與抓到它們的測試：

| 突變 | 抓到的測試 |
|---|---|
| lane位移錯一個、跳過錯誤的組、欄和沒扣掉舊列、橫向滑動用錯欄 | 新測試與既有的NCC／比對測試 |
| 拿掉flush | 只有新的飽和測試 |
| 有界限的路徑誤走窮舉 | 既有的剪枝測試 |
| 私有副本少了尾端零 | ASan回報heap-buffer-overflow |

## 效能

計時範圍是`fpm_cli`的`match_ms`：不含解碼和模板學習，每次都是新程序。計時前先登記[protocol](core-exhaustive-ncc/protocol.json)，沒有丟掉或重跑任何一次。當時機器上有其他Claude／Codex程序在跑，load average的1分鐘值是3.7～4.4。

### `benchmark_modes.py`（Src7，score .5、70目標、overlap .2、180°）

一次暖機後量9次；舊、新交錯跑三個session。下表是每邊27次的合併中位數。

| 模式 | `65b7dad` ms | 新版 ms | 變化 | 三個session |
|---|---:|---:|---:|---|
| portable_simd（預設） | 245.72 | **153.22** | −37.6% | −37.0／−38.4／−37.7% |
| portable_fast（FastMode） | 91.57 | **58.64** | −36.0% | −36.7／−35.2／−35.6% |
| portable_subpixel | 2255.19 | **1392.68** | −38.2% | −38.1／−38.3／−38.1% |
| portable_scalar | 714.65 | **631.46** | −11.6% | −11.8／−12.0／−12.2% |
| legacy_cpp（兩邊是同一個二進位檔，對照） | 185.50 | 186.22 | +0.4% | +1.6／−0.4／−0.9% |

六個session的輸出都相同。這次的舊版數字和前兩份報告是不同時間、不同負載下量的，倍率不能相乘。[摘要](core-exhaustive-ncc/formal/summary.json)

### 17張固定影像A/B（[`benchmark_backend_ab.py`](../tools/benchmark_backend_ab.py)）

每組8輪，順序每輪交換，第0輪是暖機。下表是7次的中位數。

| 影像 | `65b7dad` ms | 新版 ms | 變化 | 新較快 |
|---|---:|---:|---:|---|
| Src1 | 115.57 | 68.43 | −40.8% | 7/7 |
| Src2 | 47.28 | 30.94 | −34.6% | 7/7 |
| Src3 | 4.74 | 3.23 | −31.8% | 7/7 |
| Src4 | 24.56 | 13.37 | −45.6% | 7/7 |
| Src6 | 313.02 | 194.22 | −38.0% | 7/7 |
| Src7 | 91.70 | 58.03 | −36.7% | 7/7 |
| Src8 | 17.36 | 10.67 | −38.5% | 7/7 |
| Src9 | 0.63 | 0.45 | −27.8% | 7/7 |
| Src10 | 51.02 | 13.47 | −73.6% | 7/7 |
| Src5，8種旋轉 | 30.4～32.2 | 9.5～10.8 | −66.3～−68.6% | 各7/7 |

幾何平均比值0.439。每一輪兩邊結果都相同，否則工具會中止。

### 與舊macOS版（OpenCV＋DOTPROD）

稽核設定（`compare_versions.py`，1次暖機＋3次，屬次要數字）下，新版17張全部比舊版快：

| 影像 | 舊macOS版 ms | 新版 ms |
|---|---:|---:|
| Src1 | 105.50 | 70.96 |
| Src4 | 21.13 | 13.31 |
| Src6 | 222.80 | 192.11 |
| Src7 | 67.77 | 56.08 |
| Src10 | 15.35 | 12.66 |
| Src5，8種旋轉 | 15.4～16.5 | 9.7～10.5 |

改之前只有Src9比舊版快。[2026-10-08稽核](AUDIT_zh-TW.md)當時Src7是63.75對134.23ms。這些數字分屬不同建置與時間，只用來說明方向，不能相乘。

### 剩下的時間

profiling用`sample`，與正式計時分開；樣本都在validation-logs封存裡。

| 影像 | `warp` 改前→改後 | `computeNcc` 改前→改後 | 其他（改後） |
|---|---|---|---|
| Src7 | 45%→72% | 48%→19% | `pyrDown` 4.5% |
| Src6 | 47%→78% | 50%→18% | — |
| Src5-0 | 11%→35% | 82%→43% | `selectPeaks` 9.5%、`pyrDown` 8.2% |
| Src10 | 6%→28% | 86%→37% | `pyrDown` 21%、`selectPeaks` 8% |

大模板上現在主要的成本是binary64旋轉取樣；小圖則分散在NCC、旋轉、金字塔與選峰之間。

## 界線

- 只在一台Apple M4 Pro上驗證；Linux、Jetson、x86都沒跑。滑動和是平台無關的整數程式，在x86上也會生效，但沒有在硬體上執行過。
- 8視窗kernel只用在UDOT建置、模板寬不超過4096的情況，其他寬度與建置照舊逐列計算。
- 每次呼叫都會複製來源影像與補零的模板，記憶體多出約一份來源大小，例如Src10頂層約3.3MB。這次沒有另外量測RSS。
- 有界限剪枝（`coarseNccPruning`）與實驗用screen的路徑沒有加速。研究pipeline與Metal核心沒改；沒有重新產生LOG_CATALOG。

## 重現

```bash
cmake -S cpp-core -B build/ncc-new -DCMAKE_BUILD_TYPE=Release -DFPM_WITH_OPENCV=ON
cmake --build build/ncc-new -j
ctest --test-dir build/ncc-new --output-on-failure
python-port/venv/bin/python cpp-core/tools/check_backend_identity.py \
  --lib before=<65b7dad建置>/libfpm_c.dylib --lib after=build/ncc-new/libfpm_c.dylib --output identity.json
python-port/venv/bin/python cpp-core/tools/benchmark_backend_ab.py \
  --build before=<65b7dad建置> --build after=build/ncc-new --output ab.json
```

完整的gate腳本與log、身分比對、突變log、插樁原始碼與改前改後profiling都在[validation-logs.tar.gz](core-exhaustive-ncc/validation-logs.tar.gz)。摘要在[validation.json](core-exhaustive-ncc/validation.json)，採用與限制在[decision-and-failures.json](core-exhaustive-ncc/decision-and-failures.json)，SHA-256清單在[archive.json](core-exhaustive-ncc/archive.json)。
