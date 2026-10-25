# 穩定核心的ARM DOTPROD點積

2026-10-24，Apple M4 Pro／48 GiB，Apple clang 21.0.0，Release。穩定核心（GUI與CLI預設的固定尺度路徑）的`Backend::Auto`在ARM上改用ARMv8.2 `UDOT`（`vdotq_u32`）計算每列u8×u8點積。這不是研究輪次：演算法、API、預設參數與運算順序都沒變，只換整數點積的實作；scalar參考路徑、原NEON與x86 SSE2保留。

**結論：輸出與原NEON、scalar逐位元相同（binary64，408次執行零差異）；Src7的模式基準快13.7%（359.10→309.84ms，三個交錯session各快12.9～14.2%）；17張固定影像的幾何平均快7.0%，Src5以外的九張每一輪都較快，Src5八種旋轉在±2%內沒有一致方向。** 舊OpenCV＋DOTPROD版仍較快（同session 188.82ms），差距從1.93倍降到1.64倍。剩下的時間主要在旋轉、NCC的其他部分與金字塔，不在點積本身。

## 改了什麼

`cpp-core/src/ncc.cpp`的`dotProduct`，增29行、刪2行：

- **編譯期選擇。** 有`__ARM_FEATURE_DOTPROD`且是`__aarch64__`、沒設`FPM_DISABLE_SIMD`時定義`FPM_DOTPROD`。Apple clang在arm64 macOS的預設CPU是`apple-m1`，已定義此巨集，所以不需改CMake。其他ARM編譯（例如GCC預設`-march=armv8-a`）照舊走原NEON路徑。`compiledBackend()`回報`NEON+DOTPROD`，CLI JSON的`backend`欄位可看出實際用了哪個核心。
- **主迴圈。** 每16 bytes一個UDOT、單一累加器；沿用原本每4096 bytes歸約一次的分段。
- **尾端。** 列寬至少16、還剩1～15 bytes時，重新讀這列最後16 bytes（不越界），用遮罩把已算過的位元組設為0，再做一次UDOT。列寬小於16仍用純量迴圈。
- **沒動的部分。** `Backend::Scalar`、`FPM_DISABLE_SIMD`、x86 SSE2、原NEON核心（現在是fallback），以及`computeNcc`的其餘程式。

沒有做執行期HWCAP／sysctl分派：本機無法驗證Linux分支。在不支援dotprod的CPU上強制開啟該旗標會SIGILL，這屬於建置者的選擇。Jetson Orin（Cortex-A78AE）等支援dotprod的Linux機器可用`-DCMAKE_CXX_FLAGS=-march=armv8.2-a+dotprod`開啟，但沒有在硬體上跑過。

## 為什麼不會溢位

推理與第40輪有界uint32模式（[GPU_NCC_CLOSURE](GPU_NCC_CLOSURE_zh-TW.md)）相同。每個u8乘積最多255²＝65025，uint32可以精確容納`floor((2³²−1)/65025) = 66051`個這樣的乘積。

- UDOT每次在每個lane加4個乘積。一個4096-byte分段總共最多4096個乘積：`4096 × 65025 = 266,342,400 < 2³²−1`。所以不只各lane，連四個lane的總和（`vaddvq_u32`）都精確，之後才加進uint64。
- 尾端最多15個乘積（975,375）。
- 每列總和小於`2³¹ × 65025 < 2⁵³`，轉成double時精確；列與列之間的double累加順序與舊程式相同。

測試的全255長度包含66052：這是第一個「不分段的單一uint32總和」會超過2³²的長度（`66052 × 65025 = 4,295,031,300`）。拿掉4096分段的突變就是被這個長度抓到的。

## 驗證

| 檢查 | 結果 |
|---|---|
| Release ctest（`portable_core`、`c_abi`、`opencv_stage_comparison`） | 3/3 |
| `-march=armv8-a` fallback建置 | 3/3；回報`NEON`，`libfpm_core.a`裡沒有UDOT |
| ASan／UBSan scalar（Debug、`FPM_DISABLE_SIMD=ON`） | 2/2 |
| ASan／UBSan，開SIMD（UDOT） | 2/2；隨機測試的每個運算元都剛好結束在配置尾端，越界讀取會被抓到 |
| 完整research建置（GUI＋experiments＋OpenCV）ctest | 73/73 |
| Python harness（`test_*.py`） | 570項OK |
| [`check_backend_identity.py`](../tools/check_backend_identity.py)：17組影像×4模式（窮舉、pruned、FastMode、subpixel）×3個建置（舊NEON、新UDOT、新fallback）×SIMD／scalar | 408次執行、724筆結果，與舊建置scalar的binary64全部相同 |
| `compare_versions.py`（17張repository影像） | 新舊建置都是17/17在對舊macOS版的容許值內；新舊portable輸出17/17相同 |
| 編譯突變：遮罩差一、只取一個lane、不加遮罩、拿掉4096分段、尾端重複計算 | 5/5被`SIMD/overflow/tails`抓到；未突變版通過 |

`core_test.cpp`新增：全255長度在每個16／4096分段邊界附近及66052，以及長度0～300×4種對齊的隨機資料（長度為7的倍數時其中一個運算元全為255），與獨立的uint64參考值比較。

## 效能

計時範圍是`fpm_cli`的`match_ms`：不含圖片解碼和模板學習，每次都是新程序。舊建置是HEAD `4a78498`的凍結副本，兩邊編譯器與旗標相同。計時前先登記[protocol](core-dotprod/protocol.json)；沒有丟掉或重跑任何一次；換核心前第一個候選的正式結果另外保存（見下文）。當時機器上有其他Claude session，每個session都記錄了load average（1分鐘值1.8～3.0）。

### `benchmark_modes.py`（Src7，score .5、70目標、overlap .2、180°）

一次暖機後量9次；舊、新交錯跑三個session。下表是每邊27次的合併中位數。

| 模式 | 舊NEON ms | 新UDOT ms | 變化 | 三個session |
|---|---:|---:|---:|---|
| portable_simd（預設） | 359.10 | **309.84** | −13.7% | −12.9／−14.2／−13.9% |
| portable_fast（FastMode） | 137.65 | **122.70** | −10.9% | −11.0／−11.1／−10.8% |
| portable_subpixel | 3296.23 | **2764.81** | −16.1% | −15.5／−16.1／−16.2% |
| portable_scalar（程式沒變，對照） | 783.37 | 775.44 | −1.0% | +0.3／−1.7／−1.2% |
| legacy_cpp（兩邊是同一個二進位檔，對照） | 186.30 | 188.82 | +1.4% | +0.7／+2.3／−0.3% |

六個session的輸出都相同。[原始資料與摘要](core-dotprod/formal/summary.json)

### 17張固定影像A/B（[`benchmark_backend_ab.py`](../tools/benchmark_backend_ab.py)）

每組8輪，舊、新的順序每輪交換，第0輪是暖機。下表是7次的中位數；「新較快」是同一輪內新版較快的次數。

| 影像 | 舊 ms | 新 ms | 變化 | 新較快 |
|---|---:|---:|---:|---|
| Src1 | 199.94 | 136.68 | −31.6% | 7/7 |
| Src2 | 69.77 | 62.27 | −10.8% | 7/7 |
| Src3 | 8.55 | 7.57 | −11.4% | 7/7 |
| Src4 | 25.75 | 24.99 | −3.0% | 7/7 |
| Src6 | 447.34 | 377.27 | −15.7% | 7/7 |
| Src7 | 135.86 | 118.20 | −13.0% | 7/7 |
| Src8 | 21.91 | 19.98 | −8.8% | 7/7 |
| Src9 | 1.20 | 1.05 | −12.1% | 7/7 |
| Src10 | 64.98 | 63.79 | −1.8% | 7/7 |
| Src5，8種旋轉 | 37.0～38.3 | 37.2～38.4 | −1.9～+0.6% | 3～6/7 |

幾何平均比值0.930。Src5的八組有升有降，不算加速也不算退步。每一輪兩邊的結果都相同，否則工具會中止。

### 與舊macOS版（OpenCV＋DOTPROD）的差距

- 上表同session：舊版188.82ms，新版309.84ms，是1.64倍（改之前1.93倍）。
- 稽核設定（`compare_versions.py`，1次暖機＋3次，屬次要數字）：Src7舊版65.38ms、新版122.43ms；改之前是66.31／138.10ms，[2026-10-08稽核](AUDIT_zh-TW.md)當時是63.75／134.23ms。

另做一次profiling（`sample`，1ms取樣4秒，只跑新建置，與正式計時分開；檔案在validation-logs封存裡）。Src7的樣本中，旋轉`warp`約占47%，`computeNcc`約36%（點積已內嵌在這裡，無法單獨拆開），`pyrDown`約15%。Src10則是`computeNcc` 64%、`pyrDown` 26%、`warp` 7%。

## 第一個候選與選擇過程

第一個候選用兩個UDOT累加器加遮罩尾端，也跑完上面整套驗證（408次零差異、突變5/5）和同一份protocol的正式計時：Src7預設模式快16.6%、subpixel快21.2%，比最後採用的版本更快。但17張影像A/B裡**Src4慢了9.9%，7輪全慢**。Src4只有一層，模板寬33：兩個16-byte區塊加1 byte尾端。這組結果保留在[candidate1](core-dotprod/candidate1/)。

之後在9張影像上試了6種變體（[pilot](core-dotprod/pilot/)，只用來選擇，不是正式數字）：

- 收益主要來自遮罩尾端。頂層模板很窄（例如Src1寬30，尾端14 bytes），尾端的純量乘加才是熱點。單一累加器但尾端用純量（v2）時，Src1只差+2.0%。
- 雙累加器在短列會變慢：尾端用純量的雙累加器版本Src4慢19.5%～27.6%。
- 單一累加器加遮罩尾端（v4）在9張影像上都沒有變慢，程式也最短，所以採用它。正式計時與全部驗證都在換成v4之後重跑。

雙累加器在長列仍可再快約2～5個百分點（第一候選的Src7 subpixel −21.2%對−16.1%），但短列會變慢；要兼顧就得依列寬分支，這次沒有做。

## 界線與沒做的事

- 只在一台Apple M4 Pro上驗證。Linux、Jetson、Windows on ARM沒跑；x86沒改也沒跑。
- 沒有執行期分派；GCC等預設旗標的ARM建置仍用原NEON。
- 這次只量了CLI的`match_ms`，GUI與C ABI用的是同一個核心函式，但沒另外計時。
- 研究pipeline裡自己實作的點積與Metal核心都沒改。
- 為了避免和進行中的第42、43輪衝突，沒有重新產生`log-catalog.json`與LOG_CATALOG。

## 重現

```bash
cmake -S cpp-core -B build/dotprod-new -DCMAKE_BUILD_TYPE=Release -DFPM_WITH_OPENCV=ON
cmake --build build/dotprod-new -j
ctest --test-dir build/dotprod-new --output-on-failure
python-port/venv/bin/python cpp-core/tools/check_backend_identity.py \
  --lib before=<舊建置>/libfpm_c.dylib --lib after=build/dotprod-new/libfpm_c.dylib --output identity.json
python-port/venv/bin/python cpp-core/tools/benchmark_backend_ab.py \
  --build before=<舊建置> --build after=build/dotprod-new --output ab.json
```

舊建置可用`git archive 4a78498`凍結後建置。`benchmark_modes.py`的舊建置session要從凍結副本執行，source hash才會對應舊源碼。完整的gate腳本、各建置log、身分比對與突變log都在[validation-logs.tar.gz](core-dotprod/validation-logs.tar.gz)，摘要在[validation.json](core-dotprod/validation.json)，採用與失敗紀錄在[decision-and-failures.json](core-dotprod/decision-and-failures.json)，SHA-256清單在[archive.json](core-dotprod/archive.json)。
