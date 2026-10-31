# 穩定核心的x86 SIMD分級：8視窗點積、精確旋轉與SSE2金字塔

2026-10-31，實驗室PTRI-PRO-4000（Intel Core Ultra 9 285K，8個P核＋16個E核，Windows 11），MSVC 19.44，Release。起因是同日在這台機器上第一次實機跑Windows：17張固定影像的CLI輸出（12位有效數字）和Mac完全相同，但x86只有SSE2逐視窗點積和scalar旋轉，SIMD只比scalar快1.5倍（Mac是3.9倍）。這次只改穩定核心的x86-64路徑，演算法、API、預設參數與每個像素的浮點運算都不變；ARM路徑只多一次函式指標呼叫。這不是研究輪次，比較基準是同一台機器上的`fe60159`。

**結論：PC上5個建置（改前、新版自動選擇、上限AVX2、上限SSE2、純scalar）的binary64輸出全部相同（680次執行零差異）；預設模式17張影像總和1924→839ms，幾何平均快57.2%，每張影像的7輪量測全部較快；FastMode與subpixel分別快58.0%、56.4%。Mac的arm64輸出不變（272次零差異），速度也不變（幾何平均比值0.9975）。** 預先登記的跨平台gate沒有通過：PC和Mac的binary64輸出只有9/68組相同。這個差異在改動前就存在，原因是兩個平台的浮點運算方式不同，見[驗證](#驗證)。

## 先量到哪裡慢

用計數器插樁的副本（只做profiling，不是正式計時）在17張影像上加總，預設模式：

| 項目 | x86 `fe60159` | Mac M4 Pro（同一份程式碼） |
|---|---:|---:|
| 低層ROI旋轉`rotatedRoi` | 1041 ms（54%） | 242 ms（57%） |
| 粗搜尋NCC與低層NCC `ncc` | 658 ms（34%） | 115 ms（27%） |
| `pyrDown`（含模板學習） | 88 ms（5%） | 24 ms（6%） |
| 頂層旋轉畫布`rotateCanvas` | 78 ms（4%） | 22 ms（5%） |
| 總計`match_ms` | 1915 ms | 422 ms |

除了Src10（3.0倍），每張影像的`rotatedRoi`都是Mac的4.2～4.5倍。Mac的NEON旋轉對scalar只快1.5倍（Src6：130 vs 193 ms），所以差距主要不在Mac有SIMD，而是x86上binary64的scalar旋轉本身就比M4慢（Src6：x86 scalar 562 ms，M4 scalar 193 ms，約2.9倍）。NCC的差距則來自點積：Mac用UDOT一次算8個視窗，x86每個視窗、每一列各呼叫一次SSE2點積。

## 改了什麼

### 執行期分級

x86-64（`_M_X64`或`__x86_64__`，沒有`FPM_DISABLE_SIMD`）啟動後第一次用到時，以cpuid和XCR0判斷一次：

- **SSE2**：x86-64的基本指令集，一定有；
- **AVX2**：需要OSXSAVE、作業系統開啟YMM狀態（XCR0 bit 1、2）與AVX2；
- **AVX-VNNI-INT8**：再加上cpuid(7,1).EDX bit 4（`vpdpbuud`，u8×u8累加到u32，等同ARM的UDOT）。

新的CMake快取選項`FPM_X86_MAX_ISA`（`SSE2`、`AVX2`、`VNNI_INT8`，預設`VNNI_INT8`）設定可選的上限；編譯器做不出AVX-VNNI-INT8 intrinsic時自動降到AVX2。核心只用baseline旗標編譯，AVX2／VNNI函式在GCC／clang以`__attribute__((target(...)))`標記，MSVC直接使用intrinsic，所以設計上同一個執行檔可以在任何x86-64上跑（實際只在一顆CPU上驗證過，見[界線](#界線)）。`compiledBackend()`在x86回報實際選到的分級：`SSE2`、`AVX2`或`AVX2+VNNI-INT8`。

### 窮舉NCC：8視窗點積

原本只有UDOT建置走的8視窗分組點積，現在x86也走。資料配置、讀取範圍與flush規則都和UDOT核心相同：模板列補零到16×blocks bytes，來源複製成緊密列並在尾端補零，每累積256個區塊歸約一次到64位元。三個核心：

- **SSE2**：模板區塊展開成u16一次，每個視窗做2次unpack、2次`pmaddwd`；
- **AVX2**：每個視窗用`vpmovzxbw`把16 bytes展開成一個ymm，做1次`vpmaddwd`；
- **AVX-VNNI-INT8**：每個視窗1次128位元`vpdpbuud`。

每個累加lane在兩次flush之間最多收到511個區塊，SSE2與VNNI每區塊4個乘積、AVX2每區塊2個，所以lane的和小於511×4×65025 < 2³¹。所有乘積都是精確整數，結果和原本逐列、逐視窗的點積相同。

### `warp`：SSE2兩像素、AVX2四像素

x86-64在沒有FMA的編譯下不會合併任何乘加，所以scalar的`warpPixel`每個乘法、加法都各自捨入一次。SSE2版本每次兩像素、AVX2版本每次四像素，照`warpPixel`的順序各做一次：

- `sx=(ax+c*x)−s*y`、`sy=(ay+s*x)+c*y`；
- `top=p00*(1−dx)+p10*dx`，`bottom`同理；
- `value=(top*(1−dy)+bottom*dy)+0.5`。

只有整組像素都滿足`0≤sx<w−1`、`0≤sy<h−1`才走向量路徑。這時`sx`、`sy`與`value`都不是負數，所以floor等於截斷，可以直接用`cvttpd2dq`。鄰點照NEON版的做法，每列一次讀兩個相鄰byte。其他像素與尾端一律走原本的`warpPixel`。

為什麼要求「沒有FMA」：GCC用`-mfma`或`-march=native`編譯時，預設會把scalar迴圈的乘加合併成FMA，捨入就會不同。所以定義了`__FMA__`的建置保留scalar迴圈。MSVC 2022只有在`/fp:contract`或`/fp:fast`下才會合併；若有人這樣編譯，`rotation/pyramid/exact-reference`測試會失敗，因為它的參考實作是逐字複製的原迴圈，和函式庫用同一組旗標編譯。

### `pyrDown`：SSE2整數

MSVC沒有向量化`pyrDown`的兩個迴圈（Mac的clang有）。x86-64現在水平方向一次算8個輸出：從偶數、奇數byte lane取出e[k]、o[k]，再算`e[k]+4o[k]+6e[k+1]+4o[k+1]+e[k+2]`，16位元內最大4080。垂直方向一次處理8欄，`(sum+128)/256`的被除數不是負數，所以等於右移。兩個方向算出的整數都和scalar迴圈相同，邊緣欄與尾端仍走scalar。

### 工具與Windows測試

- 17組fixture清單移到只用標準函式庫的`tools/fixture_cases.py`；`compare_versions.py`改從那裡import。
- 新增`tools/export_fixture_pgm.py`：用OpenCV與`grayFromRgb`把27張fixture存成8-bit PGM，並寫出SHA-256 manifest。
- `check_backend_identity.py`與`benchmark_backend_ab.py`加上`--pgm-dir`，不需要OpenCV或NumPy；A/B工具也認得`fpm_cli.exe`，沒有load average的系統記為null。
- `7506003`修了兩個只在Windows出現的測試問題：`pyramid_rule_test`用了MSVC沒有的`M_PI`；`experimental_pose_shared`在非Apple平台改為正確跳過（77），不再失敗。
- 飽和寬列測試多了160列的版本。原本的4096×20只會讓「所有lane加總」超過uint32，單一lane不flush也不會溢位。160列下每個分組核心（包括UDOT）的單一lane都會超過2³²，所以拿掉flush的突變現在會被抓到。修改前在Mac上確認過：UDOT不flush的突變在20列版本存活，在160列版本失敗。

## 驗證

計時前先登記[protocol](core-x86-simd/protocol.json)（`e8d1919`）。兩台機器都從`git archive fe60159`與`git archive 8695027`解出的原始碼重新建置。輸入是`export_fixture_pgm.py`產生的27張PGM，PC上的副本先和manifest的SHA-256比對過（27個檔案全部相符）。

| 檢查 | 結果 |
|---|---|
| PC ctest：改前、新版自動、上限AVX2、上限SSE2、純scalar | 各3/3 |
| PC ctest：MSVC `/fsanitize=address`，自動（AVX2旋轉＋VNNI-INT8點積）與上限SSE2 | 各3/3 |
| PC實驗模組建置（`FPM_BUILD_EXPERIMENTS=ON`，新核心） | 33/33；Metal相關6項跳過 |
| PC [`check_backend_identity.py`](../tools/check_backend_identity.py) `--pgm-dir`：17組×4模式（窮舉、pruned、FastMode、subpixel）×5個建置×SIMD／scalar | 680次執行，與改前的scalar輸出全部相同 |
| PC 編譯突變（20個） | 20/20被`fpm_core_test`抓到；每個突變在會執行它的分級上限下建置；未突變與還原後的原始碼都通過 |
| Mac ctest：改前、新版純核心 | 各3/3 |
| Mac研究建置ctest（`e8d1919`，程式碼同`8695027`） | 91/91 |
| Mac身分比對：改前與新版×SIMD／scalar | 272次執行，全部相同 |
| Python harness（工具改動後） | 865項；10個error都是`color_verify_runner_test`在discover下沒有`--runner`的已知情況，帶runner直接執行11/11 OK |
| 跨平台：PC的參考輸出等於Mac的（binary64） | **沒有通過：9/68** |

20個突變的分布：

- SSE2旋轉4個：內部判斷多放一欄、`sx`的加法順序改變、`p10`取錯byte、少了+0.5。
- SSE2點積3個：視窗5讀成視窗4、來源高半部用錯unpack、lane不flush。
- SSE2 `pyrDown`4個：o[k+1]換成o[k]、e[k+2]多移一個lane、捨入偏差127、中間列權重4。
- AVX2旋轉3個：內部判斷多放一列、下方兩個鄰點對調、`value`的加法順序改變。
- AVX2點積3個：來源晚讀一個byte、lane不flush、lane加總少了上半部。
- VNNI-INT8點積3個：視窗7讀成視窗6、lane不flush、來源乘以自己。

旋轉與`pyrDown`的突變都被`rotation/pyramid/exact-reference`抓到。點積的突變都被`NCC/exhaustive/exact-reference`抓到；3個不flush的突變只在新加的160列飽和測試失敗，依lane上界計算，原本的20列版本抓不到它們（UDOT的同類突變在Mac上實際確認過）。每個突變讓哪些測試失敗，都記在[mutants.json](core-x86-simd/formal/mutants.json)。

### 跨平台gate為什麼沒過

登記時我以為PC和Mac的binary64輸出會逐位元相同，根據是CLI輸出的12位有效數字在68組裡全部一致。實際比對binary64，只有9組相同。差異在這次改動之前就存在：兩邊的參考都是`fe60159`的scalar輸出。

- 匹配數量和順序都一樣。分數最多差17 ulp（相對誤差2×10⁻¹⁵），位置與四角最多差27 ulp；角度接近0，所以ulp數較大，但相對誤差只有9.4×10⁻¹³。
- 主因是Apple clang預設的`-ffp-contract=on`：它在`warp`以外的運算式（例如分數正規化與幾何計算）也會合併乘加，MSVC x64則不會。用`FPM_DISABLE_SIMD`加`-ffp-contract=off`在Mac重建`fe60159`後，和PC相同的組數從9增加到63（[mac-before-nocontract-identity.json](core-x86-simd/formal/mac-before-nocontract-identity.json)）。
- 剩下5組各只有一個匹配的位置或四角不同，分數相同，這和兩邊libm的`cos`／`sin`最後一位不同的情況相符，但沒有進一步確認。

所以「x86新版與x86舊版相同」和「arm64新版與arm64舊版相同」都成立；「x86與arm64在binary64相同」本來就不成立，這次也沒有改變它。登記的決策規則要求所有gate都通過，因此依規則，這次改動是在這個gate失敗並揭露的情況下提出，是否合併由使用者決定。

## 效能

計時範圍是`fpm_cli`的`match_ms`：不含解碼和模板學習，每次都是新程序。工具是[`benchmark_backend_ab.py`](../tools/benchmark_backend_ab.py)：每組8輪，順序每輪交換，第0輪是暖機，下表是7次的中位數。PC的計時程序用`start /affinity C03C02`固定在P核（CPU 0和全部E核都排除），子程序會繼承這個設定。計時期間（22:12:38～22:16:21）PC上沒有其他工作；Mac每240秒用SSH讀一次log。沒有丟掉或重跑任何一次。

### PC：`fe60159` vs 新版（自動選到AVX2+VNNI-INT8）

| 影像 | 改前 ms | 新版 ms | 預設 | 新版較快 | FastMode | subpixel |
|---|---:|---:|---:|---|---:|---:|
| Src1 | 313.73 | **125.41** | −60.0% | 7/7 | −66.8% | −58.7% |
| Src2 | 117.41 | **59.01** | −49.7% | 7/7 | −48.5% | −53.2% |
| Src3 | 12.18 | **6.08** | −50.1% | 7/7 | −41.4% | −56.0% |
| Src4 | 48.80 | **20.94** | −57.1% | 7/7 | −57.3% | −57.2% |
| Src6 | 727.09 | **329.71** | −54.7% | 7/7 | −52.8% | −57.4% |
| Src7 | 211.88 | **99.30** | −53.1% | 7/7 | −51.0% | −55.7% |
| Src8 | 40.80 | **18.08** | −55.7% | 7/7 | −60.8% | −53.3% |
| Src9 | 1.78 | **0.92** | −48.4% | 7/7 | −47.7% | −47.9% |
| Src10 | 72.09 | **25.77** | −64.3% | 7/7 | −64.4% | −64.1% |
| Src5-0 | 47.83 | **19.38** | −59.5% | 7/7 | −60.9% | −57.4% |
| Src5-45 | 49.23 | **20.06** | −59.3% | 7/7 | −60.3% | −56.5% |
| Src5-90 | 46.81 | **19.37** | −58.6% | 7/7 | −61.5% | −56.6% |
| Src5-135 | 46.32 | **18.16** | −60.8% | 7/7 | −61.0% | −57.1% |
| Src5-180 | 47.51 | **19.45** | −59.1% | 7/7 | −59.6% | −56.8% |
| Src5-225 | 47.25 | **19.24** | −59.3% | 7/7 | −60.2% | −55.4% |
| Src5-270 | 46.94 | **18.87** | −59.8% | 7/7 | −60.8% | −57.5% |
| Src5-315 | 46.69 | **19.11** | −59.1% | 7/7 | −62.8% | −56.3% |

| 模式 | 改前總和 | 新版總和 | 幾何平均比值 |
|---|---:|---:|---:|
| 預設 | 1924.3 ms | **838.8 ms** | 0.428 |
| FastMode（`--fast`） | 1014.6 ms | **420.2 ms** | 0.420 |
| subpixel（`--subpixel`） | 12883.7 ms | **5547.8 ms** | 0.436 |

每組資料都要求兩個建置每一輪的輸出相同，否則工具會中止；三組都跑完了。[default](core-x86-simd/formal/ab-default.json)、[fast](core-x86-simd/formal/ab-fast.json)、[subpixel](core-x86-simd/formal/ab-subpixel.json)

### 各分級的貢獻（PC，預設模式）

| 比較 | 幾何平均比值 | 總和 | 每輪都較快 |
|---|---:|---|---|
| 上限SSE2 → 自動 | 0.935 | 893.9 → 836.5 ms | 12/17 |
| 上限AVX2 → 自動 | 0.946 | 871.4 → 835.2 ms | 11/17 |

大部分收益在SSE2這一級就拿到了：x86-64一定有SSE2，所以任何x86-64都能拿到這部分。VNNI-INT8點積再快約5%，AVX2旋轉與AVX2點積合起來約1%。

### Mac：`fe60159` vs `8695027`（arm64，回歸檢查）

17張影像總和443.2 → 442.6 ms，幾何平均比值0.9975，每張介於0.987和1.011之間。ARM路徑只多了一次函式指標呼叫，速度沒有可見的變化。計時期間load average約3（其他程序在跑），已記錄在[JSON](core-x86-simd/formal/mac-ab-default.json)。

兩台機器只各自和同一台機器的改前版本比較，不同硬體之間的時間只並列，不算倍率。

## 剩下的時間

`8695027`的插樁副本（只做profiling，不是正式計時；同一組17張，預設模式）：

| 項目 | `fe60159` | `8695027` | Mac `fe60159` |
|---|---:|---:|---:|
| `rotatedRoi` | 1041 ms | **444 ms**（53%） | 242 ms |
| `ncc` | 658 ms | **228 ms**（27%） | 115 ms |
| `pyrDown` | 88 ms | **73 ms**（9%） | 24 ms |
| `rotateCanvas` | 78 ms | **41 ms**（5%） | 22 ms |
| 其他（選峰、NMS、配置等） | 49 ms | 52 ms | 19 ms |
| 總計 | 1915 ms | **838 ms** | 422 ms |

- 旋轉仍占一半以上，因為每個像素都要做binary64運算；AVX2版本在pilot裡的單位像素時間已和Mac的NEON相近。
- SSE2 `pyrDown`只省了15 ms（`c74913c`的profiling是88.7 ms）。Mac只要24 ms，可見剩下的成本不在這兩個整數迴圈，這次沒有追下去。

## 試驗過的變體

微基準只用來選擇方案（[pilot](core-x86-simd/pilot/)）。每個變體都先和逐字複製的scalar參考逐byte（旋轉）或逐值（點積）比對，全部零差異，之後才計時。表中是固定在P核上的結果（`pilot-p1.json`）；第二次P核（`pilot-p2.json`）差異在1%內。

旋轉，輸出768×527（41個角度，含0、±90、180、45、1e−9、−37.25，邊界色0與255，以及anchor遠在影像外的情況）：

| 變體 | ms | 結果 |
|---|---:|---|
| scalar（原迴圈） | 1.287 | 參考 |
| SSE2，每次2像素 | 0.871 | **採用為SSE2分級** |
| AVX2，每次4像素，byte pair讀取 | 0.652 | **採用為AVX2分級** |
| AVX2，每次4像素，32位元gather | 0.602 | 快8%，未採用：要把內部判斷收緊到`sx<w−3`，而且gather在較舊的Intel（Downfall微碼修補後）和AMD上很慢，這台機器測不出來 |
| 每次配置並清零768×527輸出（`Image`建構子的成本） | 0.106 | 只量配置；不是旋轉的主要成本 |

8視窗點積（所有視窗；`current_per_window_sse2`是`fe60159`的做法）：

| 形狀 | 目前SSE2逐視窗 | SSE2分組 | AVX2分組 | VNNI-INT8 128位元 | VNNI-INT8 256位元 |
|---|---:|---:|---:|---:|---:|
| 低層ROI 762×521模板、7×7視窗 | 0.660 | 0.405 | 0.236 | **0.161** | 0.502 |
| 頂層48×33模板、1000×800畫布 | 42.75 | 21.80 | 15.57 | **8.76** | 19.52 |
| 中層190×130模板、7×7視窗 | 0.049 | 0.027 | 0.015 | **0.011** | 0.034 |

256位元的VNNI-INT8（每次兩個區塊）反而比128位元慢，所以沒有採用。同一個pilot固定在E核上（`pilot-e1.json`）時，AVX2分組比SSE2分組慢（0.655 vs 0.614 ms），VNNI-INT8 128位元仍最快（0.165 ms）；正式計時都固定在P核。第一次沒有固定核心的執行（`pilot1.json`，比後來的版本少了配置清零那一項）數字和P核相同。

## 界線

- 只在一顆x86 CPU上執行過（Core Ultra 9 285K，Arrow Lake，P核＋E核混合）。沒有跑過AMD、沒有AVX2的CPU，或有AVX2但沒有VNNI-INT8的CPU；SSE2與AVX2分級只在這顆CPU上以限制上限的建置執行。
- 正式計時都固定在P核。pilot顯示E核上AVX2分組點積比SSE2慢；沒有固定核心時Windows可能把程式排到E核。
- 沒有AVX-VNNI（u8×s8的`vpdpbusd`，Alder／Raptor Lake與Zen 5）的分級，這些CPU會用AVX2。
- 有界限／pruned路徑的逐視窗`dotProduct`在x86上仍只有SSE2。
- GCC與clang在x86-64只做過編譯（Mac上的Apple clang `-arch x86_64`，目的檔裡沒有FMA指令），沒有在Linux x86或用GCC執行過。
- MSVC的`/fp:contract`或`/fp:fast`會讓scalar迴圈合併乘加；`rotation/pyramid/exact-reference`會失敗，但沒有編譯期的防護。
- 研究pipeline、Metal程式碼與GUI都沒改；Windows的實驗模組建置（33/33）用的是新核心。
- pilot、插樁profiling與`run_cli_cases.py`的初步執行都在登記之前、在未提交的原始碼上進行，只用來選擇方案，不是正式數字。

## 重現

```bash
# 在任何有OpenCV的機器上匯出PGM（一次）
python-port/venv/bin/python cpp-core/tools/export_fixture_pgm.py fixtures-pgm
# x86-64：三個分級上限與純scalar
cmake -S cpp-core -B build/x86-auto -DCMAKE_BUILD_TYPE=Release
cmake -S cpp-core -B build/x86-sse2 -DCMAKE_BUILD_TYPE=Release -DFPM_X86_MAX_ISA=SSE2
cmake -S cpp-core -B build/x86-scalar -DCMAKE_BUILD_TYPE=Release -DFPM_DISABLE_SIMD=ON
python cpp-core/tools/check_backend_identity.py --pgm-dir fixtures-pgm \
  --lib before=<fe60159建置>/fpm_c.dll --lib auto=build/x86-auto/fpm_c.dll --output identity.json
python cpp-core/tools/benchmark_backend_ab.py --pgm-dir fixtures-pgm \
  --build before=<fe60159建置> --build after=build/x86-auto --output ab.json
```

Windows上的完整gate是`formal.bat`，Mac是`formal-mac.sh`；兩者、突變腳本、profiling插樁、pilot之外的所有log都在[validation-logs.tar.gz](core-x86-simd/validation-logs.tar.gz)。摘要在[validation.json](core-x86-simd/validation.json)，採用與限制在[decision-and-failures.json](core-x86-simd/decision-and-failures.json)，SHA-256清單在[archive.json](core-x86-simd/archive.json)。
