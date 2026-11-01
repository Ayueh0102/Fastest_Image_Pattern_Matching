# CUDA kernel：粗搜尋視窗區塊（dp4a與int8 Tensor core）與細搜尋無分支篩選

2026-11-01，實驗室PTRI-PRO-4000（NVIDIA RTX PRO 4000 Blackwell，compute capability 12.0，24 GB，PCIe 5.0 x16；驅動596.71，WDDM，沒有接顯示器；Core Ultra 9 285K；Windows 11），CUDA 13.2 Update 2、MSVC 19.44，Release。這是[CUDA後端](CUDA_BACKEND_zh-TW.md)、[transfer mode](CUDA_TRANSFER_zh-TW.md)與[kernel mode](CUDA_KERNELS_zh-TW.md)之後的第四步，不是研究輪次。新的mode都是選配，預設不變（粗搜尋ncc_mode 0、細搜尋NCC mode 0）；核心（`src/`、`include/`）一行都沒改，GUI與所有預設不變。

**結論：粗搜尋ncc_mode 6（一個warp一次算16×8個視窗，乘積用int8 Tensor core）配上細搜尋NCC mode 4（無分支的32位元取樣篩選），輸出仍和同一個建置的CPU後端逐位元相同。在17張影像上，每幀時間的幾何平均只是上一步最好設定（下稱b2）的0.949（預設）、0.947（FastMode）、0.961（subpixel），合計18.9→17.7 ms、15.2→14.4 ms、64.1→58.4 ms；因為這些影像的頂層模板很小，粗搜尋已經不是主要成本，Src3、Src8、Src9還慢了0.6～2.1%。把同樣17組的頂層面積下限放大16倍（金字塔變淺、粗搜尋成為主要成本）時，差距才明顯：幾何平均0.219（預設）與0.289（subpixel），合計1126→189 ms、1210→265 ms，17/17組變快，Src1從446.7 ms變成62.0 ms。** int8 Tensor core和`__dp4a`的差別也只在大模板才看得到：17張影像上幾何平均0.998／0.999／0.997，等於沒有差；金字塔變淺時0.800／0.842，17/17組比較快。Src6 subpixel的細搜尋單靠mode 4就從18.75 ms變成15.29 ms（0.815）。細搜尋的Tensor core這次沒有做，見[細搜尋為什麼還沒用Tensor core](#細搜尋為什麼還沒用tensor-core)。正式計時與對照共4828次執行，0失敗，每一組（輪、影像、模式）所有arm都相同，並用另一支Python腳本逐筆重新比對過。預先登記的五個假設都成立。

## 問題

上一步留下兩個瓶頸：

- **粗搜尋：** `coarseQ28<3>`每個thread逐byte從shared memory讀區塊與模板（`windowStats()`），Nsight Compute顯示L1／shared memory用到96%。頂層模板越大，每個視窗讀的byte越多，時間大約和「視窗數×模板面積」成正比。
- **細搜尋：** Src6 subpixel的`fineStripes<2>`每次launch約1.5 ms，occupancy 33%，以等待為主。

使用者也要求試int8 Tensor core（Src1粗搜尋、Src6 subpixel細搜尋）。上一篇定下的順序是先`__dp4a`再Tensor core，兩者都要保持逐位元相同。

## 結果

8輪（第0輪是暖機，不算），每輪每組每個模式各開一個新行程；行程內先記錄冷啟動的第一次呼叫，再不計時地連續呼叫500 ms，之後取5次呼叫的中位數。數字是7輪的中位數，計時期間driver固定在P核。arm的設定是`BACKEND,PYRAMID,FINE_BATCH,粗搜尋ncc_mode,TRANSFER,細搜尋mode`。

### 17張影像（標準設定）

| arm | 設定 | 預設 | FastMode | subpixel | 對b2幾何平均 | 變快的組數 |
|---|---|---:|---:|---:|---|---|
| CPU | 0,0,0,0 | 762.8 ms | 378.6 ms | 4991.8 ms | | |
| b2（基準，上一步最好） | 2,1,256,3,3,2 | 18.90 ms | 15.25 ms | 64.11 ms | 1 | |
| c5（粗搜尋dp4a視窗） | 2,1,256,5,3,2 | 18.05 ms | 14.47 ms | 63.36 ms | 0.957／0.948／0.985 | 14／13／13 |
| c6（粗搜尋Tensor core視窗） | 2,1,256,6,3,2 | 18.08 ms | 14.43 ms | 63.32 ms | 0.958／0.946／0.986 | 13／12／12 |
| f4（細搜尋篩選） | 2,1,256,3,3,4 | 18.57 ms | 15.31 ms | 59.36 ms | 0.994／1.005／0.980 | 9／6／6 |
| k5（c5＋f4） | 2,1,256,5,3,4 | 17.73 ms | 14.42 ms | 58.52 ms | 0.951／0.948／0.964 | 14／13／14 |
| **k6（c6＋f4）** | 2,1,256,6,3,4 | **17.69 ms** | **14.43 ms** | **58.41 ms** | **0.949／0.947／0.961** | **14／13／14** |

時間是17組合計；幾何平均與組數依序是預設／FastMode／subpixel。k6對CPU路徑的幾何平均是0.0434／0.0526／0.0367（b2是0.0457／0.0556／0.0382）。冷啟動（行程的第一次呼叫）k6對b2是0.977／0.974／0.971。

預設模式各組（ms）與subpixel的主要幾組：

| 影像 | CPU | b2 | k5 | k6 | k6/b2 | 模式 |
|---|---:|---:|---:|---:|---:|---|
| Src1 | 100.77 | 2.214 | 1.856 | 1.819 | 0.822 | 預設 |
| Src2 | 48.98 | 1.236 | 1.212 | 1.220 | 0.987 | 預設 |
| Src3 | 5.32 | 0.482 | 0.482 | 0.486 | 1.009 | 預設 |
| Src4 | 18.27 | 0.660 | 0.582 | 0.577 | 0.874 | 預設 |
| Src6 | 321.24 | 3.340 | 3.059 | 3.067 | 0.918 | 預設 |
| Src7 | 91.63 | 1.971 | 1.940 | 1.939 | 0.984 | 預設 |
| Src8 | 15.75 | 0.625 | 0.636 | 0.631 | 1.009 | 預設 |
| Src9 | 0.524 | 0.295 | 0.301 | 0.301 | 1.021 | 預設 |
| Src10 | 25.40 | 1.546 | 1.457 | 1.447 | 0.937 | 預設 |
| Src5（8組） | 16.2–17.7 | 0.79–0.85 | 0.75–0.80 | 0.75–0.80 | 0.940–0.954 | 預設 |
| Src1 | 879.09 | 10.317 | 9.271 | 9.254 | 0.897 | subpixel |
| Src6 | 2588.29 | 18.750 | 15.289 | 15.290 | 0.816 | subpixel |
| Src7 | 601.74 | 6.539 | 6.204 | 6.195 | 0.947 | subpixel |

不快的組是Src3、Src8、Src9（三個模式都是，慢0.6～2.1%），FastMode另有Src2（慢1.4%）。比較可能的原因是這幾組的粗搜尋kernel本身很短，新kernel多一道`__syncthreads()`與欄統計賺不回來；這幾組沒有做Nsight，原因沒有驗證。

### 金字塔變淺（頂層面積下限×16）

同樣17組，`--min-area-scale 16`，頂層模板比較大、視窗數從1161萬變成6.25億，粗搜尋成為每幀時間的主要部分。

| arm | 預設 | subpixel | 對b2幾何平均 | 變快的組數 |
|---|---:|---:|---|---|
| b2（基準） | 1126.4 ms | 1210.5 ms | 1 | |
| c5 | 252.2 ms | 335.6 ms | 0.276／0.351 | 17／17 |
| k5 | 251.2 ms | 325.9 ms | 0.274／0.343 | 17／17 |
| **k6** | **189.2 ms** | **265.1 ms** | **0.219／0.289** | **17／17** |

k6對k5是0.800／0.842（合計0.753／0.813），17/17組比較快；模板越大差越多。各組預設模式（ms）：

| 影像 | b2 | c5 | k5 | k6 | k6/b2 | k6/k5 |
|---|---:|---:|---:|---:|---:|---:|
| Src1 | 446.65 | 88.92 | 88.64 | 62.02 | 0.139 | 0.700 |
| Src2 | 20.17 | 5.85 | 5.78 | 5.12 | 0.254 | 0.886 |
| Src3 | 1.68 | 0.656 | 0.643 | 0.576 | 0.343 | 0.896 |
| Src4 | 4.85 | 1.577 | 1.579 | 1.422 | 0.293 | 0.901 |
| Src6 | 14.23 | 6.06 | 5.66 | 4.94 | 0.347 | 0.872 |
| Src7 | 12.09 | 4.16 | 4.05 | 3.49 | 0.289 | 0.862 |
| Src8 | 4.16 | 1.26 | 1.27 | 1.015 | 0.244 | 0.799 |
| Src9 | 0.502 | 0.279 | 0.279 | 0.255 | 0.508 | 0.914 |
| Src10 | 31.73 | 6.44 | 6.44 | 3.92 | 0.124 | 0.608 |
| Src5（8組） | 73.6–73.9 | 17.1 | 17.1 | 13.3 | 0.180–0.181 | 0.773–0.780 |

粗搜尋kernel是用等待時間看的：Src1的粗搜尋等待440.96→55.25 ms（k6），Src5-0是73.28→12.78 ms。subpixel的比值比較高，因為細搜尋的時間不變（Src6 subpixel 35.0→21.6 ms，其中細搜尋還占很大一塊）。

CPU路徑只在對照執行裡跑過（每輪一次呼叫，不是正式計時），同一組設定Src1約5.6 s、Src5-0約1.1 s，所以k6約是CPU的1%。這次只量時間與輸出是否相同，沒有量金字塔變淺時召回率有沒有變好；那是第51輪的問題。

### 記憶體與視窗統計

- **行程峰值RSS：** 標準設定所有CUDA arm中位數153.8–153.9 MiB（最大174.3 MiB），arm之間差不到0.1 MiB。金字塔變淺時Src1是2152 MiB，保留的device bytes 2.1 GB，所有arm都一樣：粗搜尋的結果緩衝是依「每個視窗一個16 byte entry」的最壞情況配置的（1.3億個視窗），這是之前就有的設計，不是這次造成的；對大模板是值得改的地方。
- **視窗統計：** 標準設定11,613,972個視窗裡只有12,407個（0.11%）算了binary64分數，所有arm相同（篩選規則相同）；金字塔變淺時624,669,882個視窗裡138,901個（0.022%）。這些數字每輪都一樣。
- **走FP64的區塊像素：** 標準設定mode 3是11,147個，mode 5／6是9,405個；金字塔變淺時15,533,294與10,624,153個。視窗區塊一次涵蓋64欄（mode 3是32欄），相鄰區塊重複取樣的像素比較少，所以同一個像素被算到的次數少。

## 時間去了哪裡（Nsight，不是正式計時）

正式計時之後，用同一個建置剖析b2與k6（Nsight Systems，`--trace=cuda,nvtx,wddm`，每幀平均）：

| 影像 | arm | 粗搜尋kernel | packEntries | 細搜尋kernel | kernel合計 |
|---|---|---:|---:|---:|---:|
| Src1 預設 | b2 | coarseQ28 524 µs | 22 µs | 5×125 µs | 1.240 ms |
| Src1 預設 | k6 | coarseWarpWindows 159 µs | 23 µs | 365 µs＋4×56 µs | 0.838 ms |
| Src5-0 預設 | b2 | 105 µs | 2 µs | 4×27 µs | 0.274 ms |
| Src5-0 預設 | k6 | 63 µs | 2 µs | 4×27 µs | 0.233 ms |
| Src6 subpixel | b2 | 24 µs | 2 µs | 27×607 µs | 16.63 ms |
| Src6 subpixel | k6 | 19 µs | 2 µs | 22×579 µs＋5×45 µs | 13.20 ms |
| Src1 變淺 | b2 | 427.8 ms | 11.8 ms | 6×245 µs | 441.2 ms |
| Src1 變淺 | k6 | 41.2 ms | 11.8 ms | 3×326 µs＋3×114 µs | 54.4 ms |
| Src6 變淺 | b2 | 10.08 ms | 2 µs | 6×392 µs | 12.55 ms |
| Src6 變淺 | k6 | 1.43 ms | 3 µs | 2×817 µs＋4×75 µs | 3.49 ms |

- **標準設定的粗搜尋kernel其實快很多，只是占比小：** Src1的粗搜尋kernel少了七成（524→159 µs），但每幀kernel只少0.40 ms，每幀時間也是少了約0.40 ms（2.21→1.82 ms），剩下約1 ms不在kernel（prepare、傳輸、主機挑峰值與送出命令）。Src5-0也一樣，kernel少41 µs，每幀少41 µs。
- **金字塔變淺時粗搜尋kernel快7～10倍：** Src1的427.8→41.2 ms、Src6的10.08→1.43 ms。之後Src1的`packEntries`（11.8 ms，沒有變）已占kernel時間的兩成，這是下一個可以看的地方。

Nsight Compute（每個kernel一次launch）：

| kernel | 時間 | 指令 | IPC | SM／L1 throughput | occupancy |
|---|---:|---:|---:|---|---:|
| coarseWarpWindows<5>，Src1變淺 | 67.44 ms | 22.12G | 2.28 | 96%／96% | 56% |
| coarseWarpWindows<6>，Src1變淺 | 43.18 ms | 13.47G | 2.18 | 79%／79% | 57% |
| coarseWarpWindows<6>，Src1標準 | 0.200 ms | 73.3M | 2.85 | 67%／70% | 51% |
| fineStripes<2>，Src6 subpixel | 1.49 ms | 298M | 1.41 | 41%／26% | 33% |
| fineStripes<4>，Src6 subpixel | 1.21 ms | 230M | 1.36 | 33%／32% | 33% |

mode 6比mode 5少了39%的指令、36%的時間，SM與L1都不再滿載，所以大模板上Tensor core確實有效。上一篇粗搜尋「逐byte讀shared memory，L1用到96%」的瓶頸，mode 5是靠讀word減少次數，mode 6再把乘加交給Tensor core。細搜尋的指令少了23%、時間少了19%，IPC和occupancy都沒有變：仍然被每個block 49.7 KB的shared memory限制在33%。

## 細搜尋為什麼還沒用Tensor core

這次沒有做，也沒有實測。使用者上一步問的是Src1粗搜尋與Src6 subpixel細搜尋，這次只把Tensor core用在粗搜尋；細搜尋改的是取樣。理由是下面這組數字，但它只是估計：

- Src6 subpixel一次大launch的`fineStripes<4>`（`pilot/pilot-raw.tar.gz`的SASS統計）：`__dp4a`（IDP）占指令11.9%、stall樣本3.5%；funnel shift（SHF）占11.8%、3.3%；shared memory讀取（LDS）占4.1%、1.9%。加起來約28%的指令、9%的stall樣本。
- 粗搜尋在大模板上，Tensor core比dp4a少了39%的指令。如果細搜尋的那28%也少這個比例，指令約少10%；時間通常少得更少，因為這個kernel以等待為主（issue slot只忙33%）。
- 標準設定的粗搜尋顯示，模板小的時候Tensor core和dp4a沒有差別；細搜尋的ROI通常也很窄。`m16n8k32`一次固定算32欄，窄的launch賺不到。

相較之下，occupancy還卡在33%（`fineStripes<4>`每個block 49.7 KB shared memory），Nsight估計提高occupancy的上限是66%。如果要繼續壓細搜尋，應該先減少每個block的shared memory，再看Tensor core；兩者都可以保持逐位元相同。

## 設計

### 粗搜尋視窗區塊（ncc_mode 5、6）

mode 3的`coarseQ28`一個block是32×8個thread，每個thread算一個視窗，逐byte讀shared memory。新的`coarseWarpWindows`一個block是32×4個thread，涵蓋64×8個視窗，每個warp負責16×8個，每個thread負責4個視窗（對應`m16n8k32`累加器的格式：lane = g·4 + t時，m ∈ {g, g+8}、n ∈ {2t, 2t+1}）。

1. **取樣：** 每個warp輪流取一列區塊，經`screenRow()`（細搜尋的無分支篩選，見下）寫進shared memory，列對齊到word。
2. **視窗和與平方和：** 先算區塊前th列每欄的總和（放在shared memory），lane對dx做掃描得到第一列視窗；之後每往下一列，只加進一列、減掉一列（`slideWindows()`，`__dp4a`算該列的和與平方和，再做一次lane掃描）。結果在uint32的模數下精確。
3. **乘積：** 兩種做法算同一個值。mode 6用Tensor core：對區塊第ρ列，A[m][k]是該列第dx+m+k欄，B[k][n]是模板第ρ−n列，A·B就把這一列對每個視窗(m, n)的貢獻加進去。mode 5用`__dp4a`：每個lane替它的4個視窗走一遍模板。uint32在n·65025 < 2^32時不會溢位（n ≤ 66051）。
4. **分數：** 和mode 3相同，精確整數算binary32先篩，可能達到門檻的才用`normalizeScore()`算binary64。

自動退回mode 3的情況：常數模板、模板超過66051像素、區塊放不進shared memory、來源寬或高小於2 px。mode 6要求compute capability ≥ 8.0，否則`fpm_frame_set_coarse_optimization`直接拒絕。Metal仍然拒絕大於3的mode。

**Tensor core的資料排列是量出來的。** `m16n8k32`的u8格式PTX手冊沒有逐lane寫清楚。我原本假設B的第n欄在lane & 7，kernel測試馬上失敗；之後寫了一支探針（`pilot/probe/`），把A、B、累加器分別放單一非0值，由輸出反推每個byte的位置，896個點全部吻合：A的暫存器放第group、group+8列、第4·inGroup..+3欄，再放+16的部分；B放第n = group欄、第4·inGroup..+3列，再放+16；累加器放(group, 2·inGroup)、(group, 2·inGroup+1)、(group+8, 2·inGroup)、(group+8, 2·inGroup+1)。

### 細搜尋無分支篩選（fine NCC mode 4）

Nsight Compute看fine mode 2時，約40%的stall樣本落在BSSY、BRA、NOP，也就是每個像素的控制流程；同一個kernel的fp32版本幾乎沒有。`q28Sample()`要對每個像素判斷邊界、用64位元算分子、再判斷guard。mode 4的`q28Screen()`把這些換成32位元、無分支：

- x的權重切到Q23，y用一次`__mulhi`，得到的Q23值離精確值不超過257個單位（每個角點差最多動255，y的高位乘法再2）。
- 值離捨入點超過`258 + (guard >> 33)`個單位時，捨入結果一定和`q28Sample()`相同，不需要它的分子和退回。
- 四個鄰點都在來源外的像素直接給border；其他情況（離捨入點太近、影像邊緣帶）設旗標。
- `screenRow()`每32個像素收集一次旗標，用`__ffs`逐個交給`q28Sample()`重做。正式執行裡走FP64的像素約占0.1%（Src6 subpixel 0.27%，Src4、Src8約0.01%）。

細搜尋只在ROI寬度 ≥ 256（`FINE_SCREEN_MIN_WIDTH`）時用它，比較窄的launch仍走mode 2：不設門檻時窄ROI為主的影像反而變慢（pilot裡Src4 subpixel 1.079、Src8 1.058），原因沒有再追，門檻是由pilot選的。粗搜尋mode 5、6的區塊取樣也一律用`screenRow()`。

### 計數器的資料競爭

`coarseQ28`原本在tid 0把block的計數變數歸零，別的warp可能已經開始加，所以走FP64區塊像素的診斷統計可能少算幾個。現在改成直接`atomicAdd`到全域計數器，拿掉了static shared變數與最後一道barrier。這只影響診斷統計，不影響輸出；kernel測試裡mode 3的計數穩定在23,125。[上一篇](CUDA_KERNELS_zh-TW.md#失敗與揭露)揭露的「開發時差1」就是這個競爭，當時的FP64區塊像素統計可能略為少算。

因為這個修正也改到`coarseQ28`，基準b2和上一篇的best不是同一份程式碼；本文所有比值都用同一個建置的b2。

## 選擇設計的過程（不是正式計時）

- **warp數：** 每個block 8、4、2、1個warp（開發中的樹，尚未有fine篩選的最終版本），10組影像預設模式對c3的幾何平均，k5／k6：W8 0.964／0.958、W4 0.964／0.951、W2 0.973／0.958、W1 0.972／0.970。選W=4。
- **細搜尋的欄區塊（否決）：** 想把細搜尋也改成一次算多個視窗（T256、T128、T64個ROI欄一個區塊）。c6f4對best的幾何平均（預設／subpixel）：T0（不分區）0.944／0.968、T256 0.957／0.978、T128 1.003／1.078、T64 1.114／1.316。只有Src1 subpixel在T256賺一點，其餘變慢；原因沒有再追。已移除，保留在`pilot/dev-tree-with-tiles.diff`。
- **細搜尋篩選的三個版本：** Src6 subpixel一次大launch（Nsight Compute）：

  | 版本 | 時間 | 指令 | 說明 |
  |---|---:|---:|---|
  | fine mode 2 | 1.50 ms | 298M | 基準 |
  | fine mode 3（fp32，對照） | 1.16 ms | 281M | |
  | 第一版（每個像素分支） | 1.40 ms | 258M | 只省約7% |
  | 第二版（整列重做） | 1.95 ms | 384M | 0.1%的像素有旗標，幾乎每一列都重做，反而變慢 |
  | 最終版（每32像素逐個重做） | 1.21 ms | 230M | 正式版本 |

- **ROI寬度門檻：** 同上10組subpixel，f4對best的幾何平均與合計：T0 0.978（48.56 ms）、T64 0.963（47.58）、T128 0.965（47.56）、**T256 0.957（47.31）**、T512 0.966（48.14）。T256時Src6 0.816、Src4 0.992、Src8 0.998。
- **大模板pilot：** 6個案例（`pilot/pilotbig.py`）輸出相同，c6／c3是0.129–0.402，c6／c5約0.69–0.91，定出登記的H3、H4。
- **pilot的環境變數：** 上面的warp數、區塊大小與寬度門檻當時用環境變數切換（`FPM_PILOT_*`），正式版本全部移除，只留下選定的常數。
- **突變pilot：** 在開發中的樹上先跑新的突變（`pilot/mutpilot*.log`），L22存活，因為沒有測試的分數剛好等於門檻；補上「threshold at a score」測試後抓到。U7–U9存活（只回報，見下）。L23、L25由構造可知等價，登記前移除。

## 驗證

| 檢查 | 結果 |
|---|---|
| PC ctest（1d0e951，git archive） | 35/35；`experimental_cuda_backend`與`experimental_cuda_kernels`有執行，6個Metal測試跳過 |
| Mac研究版ctest（1d0e951，git archive含`macos-cli/`與`Test Images/`） | 91/91 |
| 輸入 | 27個PGM的SHA256與manifest相同 |
| 逐位元相同 | 計時2856＋1088次、對照612＋272次，共4828次（其中CUDA 4250次），每一組（輪、影像、模式）所有arm相同，跨輪、跨檔案也相同；0失敗。`formal/identity_check.py`不用driver的比對，直接比`matches`，結果相同 |
| 對照arm | k6 batch 8、k6搭配transfer mode 0（沒有lean calls）、k6搭配CPU細搜尋、k5搭配transfer mode 0、k6搭配4個複製執行緒；金字塔變淺的cpu、b2、k5、k6 |
| 核心 | `git diff 5f2c6f7 1d0e951 -- cpp-core/src cpp-core/include`是空的 |
| H1：標準設定k6/b2幾何平均 < 0.98 | 成立（0.949／0.947／0.961；pilot 0.957／0.949／0.963） |
| H2：標準設定Src6 subpixel f4/b2 < 0.90 | 成立（0.815；pilot 0.816–0.832） |
| H3：變淺的預設模式k6/b2 < 0.5 | 成立（0.219；pilot 0.129–0.402，只有6個案例） |
| H4：變淺的預設模式k6/k5 < 1 | 成立（0.800；pilot 0.69–0.91） |
| H5：標準設定k6/k5在1的3%以內 | 成立（0.998／0.999／0.997） |

`cuda_backend_test`新增的部分：14個場景各用ncc_mode 5、6與fine mode 4的8種組合（含batch 64、batch 3、4個複製執行緒、CPU細搜尋與transfer mode 1、3）各prepare兩次，全部和CPU相同；metrics測試另外確認mode 5、6真的有走到；不接受ncc_mode 7與fine mode 5。`cuda_kernel_test`新增：粗搜尋mode 5、6的形狀案例（33×3、64×9、65×8、97×2、1×1、2×30、3×9、4×1、31×33，以及會退回mode 3的300×260）、分數剛好等於門檻的案例；細搜尋mode 4的262×41寬模板（不一致的pose、隨機、邊界、影像外、特殊角度、repairAll）。

### 突變

`cuda_mutants.py`在1d0e951的另一份副本上逐一編譯79個突變、跑會受影響的測試。

| 組別 | 結果 |
|---|---|
| L1–L22、L24（視窗區塊與無分支篩選，列入門檻） | 23/23被抓到 |
| K1–K13（kernel mode與複製執行緒，列入門檻） | 13/13被抓到 |
| T1–T10（transfer mode，列入門檻） | 10/10被抓到 |
| 之前的24個 | 22被抓到；N2、F2存活，和前幾步相同，[CUDA後端](CUDA_BACKEND_zh-TW.md#驗證)已論證兩者在可達輸入上等價 |
| U1–U9（只回報） | U3、U4、U6被抓到；U1、U2、U5（和上一步相同）與U7、U8、U9存活 |
| 未突變與還原後的樹 | 都通過 |

L系列涵蓋Tensor core的欄排列與提早結束、用到視窗外的模板列、乘積沒有位移、dp4a差一個word、滑動減錯列、lane掃描只到8、欄統計少一列、尾端沒遮罩、模板補零、常數模板、地圖外的視窗被計分、binary64視窗沒計數，以及篩選的margin、捨入、邊緣帶、chunk重做、x權重、y乘法與門檻相等的比較。新存活的三個都是預期的：U7拿掉66051像素的上限，只有模板超過這個大小、而且Σxt真的超過2^32的亮影像才會錯，測試的300×260是隨機亮度，達不到；U8不把區塊裁到地圖，只是多算地圖外不會被計分的視窗；U9拿掉寬或高小於2 px的保護，這種影像的像素都交給`q28Sample()`，輸出不變，保護的是多讀幾個byte。

### 失敗與揭露

- **計時期間的SSH：** 等待腳本在計時開始前每60秒讀一次formal.log，所以計時開始後60秒內（標準設定的第0輪，暖機，不算）讀了一次；之後每20分鐘讀一次，計時結束後的第一次讀到FORMALDONE。每次讀整個formal.log（不到400 bytes），符合protocol。其他時間沒有連線到PC。
- **Src3、Src8、Src9與FastMode的Src2沒有變快：** 最多慢2.1%，不在假設裡（H1量幾何平均），照實記錄。
- **f4在預設與FastMode幾乎沒有好處：** 幾何平均0.994與1.005，好處集中在細搜尋重的subpixel（Src6 0.815、Src1 0.931、Src2 0.932、Src7 0.952）。不在假設裡，照實記錄。
- **第一次的Tensor core排列是錯的：** 開發時kernel測試抓到，改用探針量出來之後才對。
- **一次沒有重新編譯的開發建置：** 開發中查計數器抖動時，突變還原讓PC上的`coarse_cuda.cu`比從Mac解開的新，ninja因此沒有重新編譯，看到的仍是舊kernel。重新建置後穩定在23,125。正式執行用全新的`git archive`與全新的建置目錄，不受影響。
- **細搜尋的Tensor core沒有做：** 使用者問到Src6 subpixel的細搜尋，這次只做粗搜尋與細搜尋的取樣篩選；原因與估計見上面一節，沒有實測。
- **一次Nsight Systems失敗，補跑一次：** Src1變淺的k6在`after.bat`裡回報「Connection to Agent lost」，沒有資料（`profile/Src1-exhaustive-area4096-k6.*`保留原樣）；看到後用同樣的參數單獨重跑一次，放在`profile/retry/`。這只是剖析，不是正式計時。
- **pilot的輪次：** 報告裡的pilot數字是`pilot/pilot_summary.py`重算的；protocol裡的預測當時用全部輪次算，差在第三位。
- **事前登記的pilot預測和正式結果：** H1預測0.957／0.949／0.963，正式0.949／0.947／0.961；H2預測0.816–0.832，正式0.815，都在預期內。

## 界線

- 一張GPU（RTX PRO 4000 Blackwell）、一個驅動（596.71，WDDM）、只有Windows。
- 只有研究coarse／frame API的frame路徑；handle API（`fpm_coarse_match`）不受影響。
- 金字塔變淺只量時間與輸出一致性，沒有量召回；標準設定仍是預設建議，變淺要等第51輪的證據。
- 粗搜尋的結果緩衝是最壞情況配置，金字塔變淺時Src1的RSS與device bytes都超過2 GB；有界的緩衝（溢位時重跑）還沒做。
- 細搜尋的Tensor core與occupancy還沒做；小影像（Src3、Src8、Src9）大多是固定成本，CUDA Graphs還沒試。

## 重現

```bash
# Windows，x64 Native Tools環境，CUDA 13.2的bin在PATH裡
cmake -S cpp-core -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
# 標準設定（start /affinity C03C02 固定在P核）
python cpp-core/tools/benchmark_frame_backend.py --bench build-cuda/experiments/fpm_frame_bench.exe --pgm-dir PGM \
  --arm cpu=0,0,0,0 --arm b2=2,1,256,3,3,2 --arm c5=2,1,256,5,3,2 --arm c6=2,1,256,6,3,2 --arm f4=2,1,256,3,3,4 \
  --arm k5=2,1,256,5,3,4 --arm k6=2,1,256,6,3,4 --rounds 8 --repeat 5 --warmup-ms 500 --output timing-standard.json
# 金字塔變淺
python cpp-core/tools/benchmark_frame_backend.py --bench build-cuda/experiments/fpm_frame_bench.exe --pgm-dir PGM \
  --min-area-scale 16 --modes exhaustive,subpixel --arm b2=2,1,256,3,3,2 --arm c5=2,1,256,5,3,2 --arm k5=2,1,256,5,3,4 \
  --arm k6=2,1,256,6,3,4 --rounds 8 --repeat 5 --warmup-ms 500 --output timing-shallow.json
python cpp-core/tools/frame_backend_summary.py timing-standard.json --reference b2 --arm c5 --arm c6 --arm f4 --arm k5 --arm k6 --h1 0.98 --h2 0 --output summary.json
# 突變
python cpp-core/tools/cuda_mutants.py build-cuda . mutants.json
```

證據在[`cuda-coarse-windows/`](cuda-coarse-windows/)：`protocol.json`（ac0a538，計時前登記）、`formal/`（摘要、獨立的一致性重算、突變結果與`derived_tables.py`）、`pilot/`、`profile/`、`raw-results.tar.gz`、`validation-logs.tar.gz`、`validation.json`、`decision-and-failures.json`、`archive.json`。
