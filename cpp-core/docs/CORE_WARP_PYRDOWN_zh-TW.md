# 穩定核心的精確旋轉與金字塔加速

2026-10-25，Apple M4 Pro／48 GiB，Apple clang 21.0.0，Release。接續[DOTPROD點積](CORE_DOTPROD_zh-TW.md)：該次profiling顯示Src7約47%的樣本在旋轉取樣`warp`、15%在`pyrDown`。這次只改`cpp-core/src/image.cpp`的這兩個函式，演算法、API、預設參數與每個像素的浮點運算都不變。這不是研究輪次，比較基準是同一台機器上的`51d95b1`（已含UDOT）。

**結論：輸出與`51d95b1`逐位元相同（binary64，544次執行零差異，包括不含任何SIMD的建置）；Src7模式基準快21.3%（319.76→251.49ms，三個交錯session各快19.4～24.4%）；17張固定影像每張每輪都較快，幾何平均快22.7%。** 與舊OpenCV＋DOTPROD版的差距從1.67倍降到1.32倍。

## 改了什麼

### `warp`：每次兩個像素的NEON版本

原本每個像素的運算式整段搬進`warpPixel`，內容一字未改。scalar迴圈（x86、`FPM_DISABLE_SIMD`、非clang編譯器）就是原本的迴圈。

在clang＋aarch64＋NEON、沒設`FPM_DISABLE_SIMD`時，每一列先走`warpRowNeon`：

- 每次用`float64x2`處理兩個像素，每一步都照著scalar版的運算做。clang預設`-ffp-contract=on`是逐個運算式決定要不要合併成FMA，從組語確認的合併方式是：
  - `sx=(ax+c*x)−s*y`、`sy=(ay+s*x)+c*y`各做兩次fused multiply-add；
  - `top=p00*(1−dx)+round(p10*dx)`，`bottom`同理；
  - 最後的值算法相同，再加0.5取floor、clamp。

  NEON版用`vfmaq_f64`／`vfmsq_f64`在同樣的位置做同樣的合併。
- 只有兩個像素都滿足`0≤sx<w−1`、`0≤sy<h−1`時才走向量路徑，此時四個鄰點都在影像內。整數位移`fy*stride+fx`遠小於2⁵³，換成整數是精確的。每列從上下兩列各讀一次相鄰的兩個byte，再拆成四個鄰點。
- 只要有一個像素碰到邊界，這一對以及奇數寬的最後一個像素都改用`warpPixel`。

為什麼限定clang：其他編譯器的FMA合併規則可能不同（例如GCC預設`-ffp-contract=fast`），所以那些編譯器保留scalar迴圈。若有人用不同的contract旗標編譯clang，新增的`rotation/pyramid/exact-reference`測試會失敗。這個測試的參考實作是逐字複製的原迴圈，和函式庫用同一組旗標編譯。

### `pyrDown`：內部欄直接計算

濾波器仍是`[1,4,6,4,1]`可分離、REFLECT_101、兩個方向做完後才捨入一次，全程整數運算。改動：

- 5個tap都在列內的欄直接相加；只有最左一欄和右邊幾欄呼叫`reflect101`。
- 垂直方向每個輸出列只算一次5個來源列的指標，之後是連續記憶體的迴圈，clang會自動向量化。

整數加法沒有捨入，所以任何平台的結果都和原本相同。

## 驗證

| 檢查 | 結果 |
|---|---|
| Release ctest | 3/3；`portable_core`含新測試`rotation/pyramid/exact-reference` |
| `-march=armv8-a`（關掉UDOT，NEON旋轉照開） | 3/3 |
| Release＋`FPM_DISABLE_SIMD`（純scalar旋轉） | 3/3 |
| ASan／UBSan scalar（Debug、`FPM_DISABLE_SIMD=ON`） | 2/2 |
| ASan／UBSan開SIMD（Debug `-O0`，NEON旋轉） | 2/2；`-O0`下clang的合併方式也和NEON版一致 |
| 完整research建置ctest | 73/73 |
| Python harness | 570項OK |
| [`check_backend_identity.py`](../tools/check_backend_identity.py)：17組×4模式（窮舉、pruned、FastMode、subpixel）×4個建置（`51d95b1`、新版、新版無DOTPROD、新版純scalar）×SIMD／scalar | 544次執行、724筆結果，與`51d95b1`的scalar binary64全部相同 |
| `compare_versions.py`（17張影像） | 新舊都是17/17在容許值內；新舊portable輸出17/17相同 |
| 編譯突變（7個） | 7/7被`rotation/pyramid/exact-reference`抓到；未突變版通過 |

新測試做3000次隨機試驗，與逐字複製的原`warp`、`pyrDown`迴圈比較每個byte：

- 影像1～40像素見方，最後100次是約200×180；stride有額外padding。
- 角度包含0、±90、180、45、1e-9、−37.25與隨機角度。
- anchor有的落在影像外很遠；邊界色有0和255兩種。
- 同時測`rotatedRoi`、`rotateCanvas`和`pyrDown`。

7個突變分別是：

- NEON內部判斷多放一欄、少了列的上限；
- `top`少做一次FMA合併（只差一次捨入）；
- 下方兩個鄰點對調、位移的兩軸對調；
- pyrDown內部欄多算一欄、垂直反射位移錯一列。

## 效能

計時範圍是`fpm_cli`的`match_ms`：不含解碼和模板學習，每次都是新程序。計時前先登記[protocol](core-warp-pyrdown/protocol.json)，沒有丟掉或重跑任何一次。當時機器上有其他Claude／Codex程序在跑，每個session都記錄了load average（1分鐘值3.0～3.9）。

### `benchmark_modes.py`（Src7，score .5、70目標、overlap .2、180°）

一次暖機後量9次；舊、新交錯跑三個session。下表是每邊27次的合併中位數。

| 模式 | `51d95b1` ms | 新版 ms | 變化 | 三個session |
|---|---:|---:|---:|---|
| portable_simd（預設） | 319.76 | **251.49** | −21.3% | −19.4／−21.1／−24.4% |
| portable_fast（FastMode） | 127.93 | **93.76** | −26.7% | −25.0／−25.1／−29.9% |
| portable_subpixel | 2854.78 | **2316.59** | −18.9% | −19.2／−15.4／−22.8% |
| portable_scalar | 796.43 | **717.62** | −9.9% | −7.3／−10.8／−11.4% |
| legacy_cpp（兩邊是同一個二進位檔，對照） | 191.97 | 190.30 | −0.9% | +5.9／−1.6／−3.1% |

`portable_scalar`也變快，是因為`Backend::Scalar`只決定點積怎麼算：旋轉的SIMD是編譯期選擇，而`pyrDown`的改動是一般C++。六個session的輸出都相同。[摘要](core-warp-pyrdown/formal/summary.json)

這次的舊版數字（319.76ms）和上一份報告的新版（309.84ms）是不同時間、不同機器負載下量的，兩次的倍率不能相乘。

### 17張固定影像A/B（[`benchmark_backend_ab.py`](../tools/benchmark_backend_ab.py)）

每組8輪，順序每輪交換，第0輪是暖機。下表是7次的中位數。

| 影像 | `51d95b1` ms | 新版 ms | 變化 | 新較快 |
|---|---:|---:|---:|---|
| Src1 | 143.31 | 118.76 | −17.1% | 7/7 |
| Src2 | 63.76 | 47.81 | −25.0% | 7/7 |
| Src3 | 8.08 | 4.72 | −41.6% | 7/7 |
| Src4 | 25.68 | 24.65 | −4.0% | 7/7 |
| Src6 | 388.46 | 309.70 | −20.3% | 7/7 |
| Src7 | 119.20 | 87.12 | −26.9% | 7/7 |
| Src8 | 20.14 | 17.17 | −14.8% | 7/7 |
| Src9 | 1.07 | 0.61 | −43.0% | 7/7 |
| Src10 | 64.78 | 47.44 | −26.8% | 7/7 |
| Src5，8種旋轉 | 37.7～39.1 | 30.3～31.3 | −18.9～−20.8% | 各7/7 |

幾何平均比值0.773。每一輪兩邊結果都相同，否則工具會中止。

### 與舊macOS版及剩下的時間

- 同session：舊版190.30ms、新版251.49ms，是1.32倍（改之前1.67倍）。
- 稽核設定（`compare_versions.py`，1次暖機＋3次，屬次要數字）：Src7舊版64.45ms、新版85.76ms（改之前119.07ms）。

另做profiling（`sample`，1ms取樣4秒，與正式計時分開；改前改後的樣本都在validation-logs封存裡）：

| 影像 | `warp` 改前→改後 | `pyrDown` 改前→改後 | `computeNcc` 改後 |
|---|---|---|---:|
| Src7 | 47%→45% | 15%→2.8% | 48% |
| Src6 | 53%→47% | 4.7%→<2% | 50% |
| Src5-0 | 12%→11% | 19%→2.5% | 82% |
| Src10 | 7%→6% | 26%→4.5% | 86% |

`warp`在大模板影像上仍占近一半，因為每個像素都必須以binary64運算；`computeNcc`（積分圖、正規化與已內嵌的點積）成為Src5、Src10的主要成本。

## 試驗過的變體

微基準只用來選擇方案（[pilot](core-warp-pyrdown/pilot/)）。768×527的ROI上原版約1.09ms，各變體都與原版逐位元相同：

- **A：** scalar迴圈加內部快速路徑。`warp`只快約8～20%；為了讓scalar參考迴圈保持原樣，沒有採用。
- **B：** NEON版，每個像素各自轉成double、經過堆疊組成向量。比C慢。
- **C：** NEON版，相鄰兩byte用整數一次讀入。**採用**，`warp`快約30～38%。
- **D：** C展開成每次4像素。比C再快約3～12%，但要多一份展開的程式碼，沒有採用。

微基準在選定方案之後又跑過一次，才把輸出存檔；當初做選擇時那幾次的輸出沒有存下來。

## 界線

- 只在一台Apple M4 Pro上驗證。Linux、Jetson、Windows on ARM、x86都沒跑；`pyrDown`的一般C++改動在x86上也會生效，只是沒有在硬體上執行過。
- NEON旋轉只用在clang。GCC建置保留原scalar迴圈，所以拿不到這部分加速。
- 旋轉和金字塔的SIMD沒有執行期開關；scalar參考版要用`FPM_DISABLE_SIMD`建置取得。
- 研究pipeline自己的旋轉與Metal核心都沒改。為了避免和其他進行中的輪次衝突，沒有重新產生`log-catalog.json`與LOG_CATALOG。

## 重現

```bash
cmake -S cpp-core -B build/warp-new -DCMAKE_BUILD_TYPE=Release -DFPM_WITH_OPENCV=ON
cmake --build build/warp-new -j
ctest --test-dir build/warp-new --output-on-failure
python-port/venv/bin/python cpp-core/tools/check_backend_identity.py \
  --lib before=<51d95b1建置>/libfpm_c.dylib --lib after=build/warp-new/libfpm_c.dylib --output identity.json
python-port/venv/bin/python cpp-core/tools/benchmark_backend_ab.py \
  --build before=<51d95b1建置> --build after=build/warp-new --output ab.json
```

完整的gate腳本與log、身分比對、突變log、改前改後的profiling都在[validation-logs.tar.gz](core-warp-pyrdown/validation-logs.tar.gz)。摘要在[validation.json](core-warp-pyrdown/validation.json)，採用與限制在[decision-and-failures.json](core-warp-pyrdown/decision-and-failures.json)，SHA-256清單在[archive.json](core-warp-pyrdown/archive.json)。
