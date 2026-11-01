# CUDA kernel：Q28粗搜尋區塊、binary32篩選與細搜尋取樣

2026-11-01，實驗室PTRI-PRO-4000（NVIDIA RTX PRO 4000 Blackwell，compute capability 12.0，24 GB，PCIe 5.0 x16；驅動596.71，WDDM，沒有接顯示器；Core Ultra 9 285K；Windows 11），CUDA 13.2 Update 2、MSVC 19.44，Release。這是[CUDA後端](CUDA_BACKEND_zh-TW.md)與[transfer mode](CUDA_TRANSFER_zh-TW.md)之後的第三步，不是研究輪次。新的kernel mode都是選配，預設不變（粗搜尋ncc_mode 0、細搜尋NCC mode 0、transfer mode 0、1個複製執行緒）；核心（`src/`、`include/`）一行都沒改，GUI與所有預設不變。

**結論：粗搜尋的Q28區塊取樣加binary32篩選（ncc_mode 3）配上細搜尋逐列取樣（fine mode 2），輸出仍和同一個建置的CPU後端逐位元相同，17張影像每幀時間的幾何平均是上一步最好設定（transfer mode 3，下稱m3）的0.765（預設）、0.745（FastMode）、0.832（subpixel），合計24.8→18.9 ms、20.4→15.3 ms、80.6→64.2 ms；三個模式17/17組都變快。** 對CPU路徑的幾何平均從0.060／0.075／0.046變成0.046／0.056／0.038。另外試的兩個做法都比較慢：細搜尋請求改回先上傳（幾何平均1.22／1.19／1.30倍時間），4個執行緒複製來源（1.16／1.09／1.08倍）。事後的Nsight診斷顯示，上一步細搜尋kernel慢的約60%確實來自從mapped記憶體讀請求，但每次呼叫多一次傳輸更貴；多執行緒複製則讓GPU從主機讀來源的DMA慢了2～12倍。全部用binary32取樣與計分（不精確的對照）再快4～11%，但408次執行只有8次和CPU逐位元相同；所有4344個CPU匹配都找得到、沒有多出的匹配，最大差異是位置0.0012 px、角度0.0015°、分數1.35e-5。正式計時與對照共4182次執行，3264次精確模式的CUDA執行全部和同輪CPU相同，0失敗。預先登記的五個假設都成立。

後續：[粗搜尋視窗區塊與細搜尋篩選](CUDA_COARSE_WINDOWS_zh-TW.md)做了本文留下的下一步：粗搜尋改用word讀取加`__dp4a`（ncc_mode 5）與int8 Tensor core（ncc_mode 6），細搜尋取樣加上無分支篩選（fine mode 4），輸出不變，17張影像再降到本文best的約0.95倍，頂層模板變大時約0.22倍。

## 問題

上一步之後，Nsight Compute（`pilot/ncu1`，8aa1630的建置）看兩個最重的kernel：

- **粗搜尋（Src1，coarseFused）：** 一次launch 1.85 ms，FP64 pipe用到85%。每個畫布像素都用binary64的`warpPixel()`旋轉取樣，這張卡FP32與FP64的峰值比是64:1。
- **細搜尋（Src6 subpixel，fineStripes）：** 一次launch 1.98 ms，SM throughput 57%，occupancy 33%（每個block 49.7 KB shared memory，每個SM只能放2個block）。SASS指令裡IMAD、IADD、ISETP、MOV等整數指令占六成，是取樣的位址與座標計算；`__dp4a`（IDP）只占5.1%。

使用者也問了四件事：用FP32會不會更好、精準度會不會差；複製來源能不能用多執行緒；上一步lean calls下細搜尋kernel慢了約7%，是不是從mapped記憶體讀請求造成的，能不能試先上傳請求的變體；Src1粗搜尋與Src6 subpixel細搜尋能不能用int8 Tensor core。

## 結果

8輪（第0輪是暖機，不算），每輪每組每個模式各開一個新行程；行程內先記錄冷啟動的第一次呼叫，再不計時地連續呼叫500 ms，之後取5次呼叫的中位數。數字是7輪的中位數，計時期間driver固定在P核。arm的設定是`BACKEND,PYRAMID,FINE_BATCH,粗搜尋ncc_mode,TRANSFER,細搜尋mode,執行緒`。

| arm | 設定 | 預設 | FastMode | subpixel | 對m3幾何平均 |
|---|---|---:|---:|---:|---|
| CPU | 0,0,0,0 | 760.1 ms | 378.5 ms | 4924.7 ms | |
| m3（基準） | 2,1,256,0,3 | 24.8 ms | 20.4 ms | 80.6 ms | 1 |
| c2（Q28區塊） | 2,1,256,2,3 | 20.7 ms | 16.3 ms | 76.6 ms | 0.827／0.800／0.912 |
| c3（＋binary32篩選） | 2,1,256,3,3 | 19.9 ms | 15.5 ms | 75.8 ms | 0.785／0.754／0.893 |
| f2（細搜尋逐列取樣） | 2,1,256,0,3,2 | 23.8 ms | 20.1 ms | 69.2 ms | 0.977／0.986／0.937 |
| **best（c3＋f2）** | 2,1,256,3,3,2 | **18.9 ms** | **15.3 ms** | **64.2 ms** | **0.765／0.745／0.832** |
| u7（上傳請求） | 2,1,256,0,7 | 28.8 ms | 23.5 ms | 93.4 ms | 1.220／1.185／1.298 |
| t4（4個複製執行緒） | 2,1,256,0,3,0,4 | 27.7 ms | 21.7 ms | 83.1 ms | 1.155／1.088／1.084 |
| fp32（不精確對照） | 2,1,256,4,3,3 | 17.7 ms | 14.7 ms | 55.2 ms | 0.721／0.714／0.745 |

時間是17組合計；幾何平均依序是預設／FastMode／subpixel。best在三個模式都是17/17組比m3快；c2、c3也是17/17。f2在預設模式14/17、FastMode 12/17組變快（Src10與幾組Src5不快，這幾組的細搜尋很輕），subpixel 17/17。

冷啟動（行程的第一次呼叫）：best對m3的幾何平均是0.946／0.950／0.928，預設模式合計84.7→79.3 ms。

預設模式各組（ms）：

| 影像 | CPU | m3 | c2 | c3 | f2 | best | best/m3 | best/CPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Src1 | 100.83 | 3.36 | 2.41 | 2.38 | 3.22 | 2.22 | 0.660 | 0.022 |
| Src2 | 49.58 | 1.46 | 1.36 | 1.32 | 1.36 | 1.23 | 0.845 | 0.025 |
| Src3 | 5.29 | 0.51 | 0.50 | 0.49 | 0.50 | 0.49 | 0.960 | 0.092 |
| Src4 | 18.31 | 1.13 | 0.74 | 0.67 | 1.12 | 0.66 | 0.590 | 0.036 |
| Src6 | 313.35 | 3.96 | 3.88 | 3.88 | 3.43 | 3.34 | 0.844 | 0.011 |
| Src7 | 95.06 | 2.21 | 2.15 | 2.14 | 2.03 | 1.96 | 0.885 | 0.021 |
| Src8 | 15.82 | 0.66 | 0.64 | 0.64 | 0.65 | 0.63 | 0.958 | 0.040 |
| Src9 | 0.53 | 0.31 | 0.30 | 0.30 | 0.31 | 0.30 | 0.953 | 0.564 |
| Src10 | 25.34 | 1.95 | 1.55 | 1.54 | 1.95 | 1.53 | 0.786 | 0.060 |
| Src5（8組） | 16.2–17.7 | 1.13–1.17 | 0.88–0.92 | 0.80–0.84 | 1.13–1.17 | 0.79–0.84 | 0.700–0.715 | 0.047–0.049 |

粗搜尋重的影像（Src1、Src4、Src5、Src10）主要靠c2，c3再多省一點（Src4 FastMode：m3 0.80、c2 0.42、c3 0.35 ms）。細搜尋重的subpixel主要靠f2：Src6 subpixel 25.07→18.84 ms（f2）、18.80 ms（best），Src1 subpixel 13.34→10.36 ms，Src7 subpixel 8.26→6.55 ms。小影像（Src3、Src8、Src9）每幀只有0.3–0.7 ms，大多是launch與WDDM送出命令的延遲，kernel變快只省4～5%。

c3的篩選在17組裡共有11,613,972個粗搜尋視窗，只有12,407個（0.11%）算了binary64分數；比例最高的是Src9（2378個裡80個，3.4%）。Q28取樣碰到guard、改走FP64的畫布像素共11,147個。這些數字每輪、每個模式都一樣。

記憶體：所有CUDA arm的行程峰值RSS中位數都是151.7–151.9 MiB（最大172 MiB），差不到0.3 MiB；保留的device bytes，c2／c3／best多了4.6 KB（每個角度的job多帶Q28係數），u7多了22.5 KB（細搜尋請求的device buffer）。

## FP32會不會更好、精準度差多少

fp32 arm把粗搜尋與細搜尋的取樣（`warpPixel32()`）和分數（binary32）都改成binary32，整數視窗統計仍然精確。這是對照，不是精確模式：

| 項目 | fp32對CPU |
|---|---|
| 逐位元相同的執行 | 408次中8次（都是Src5-0預設模式） |
| CPU的匹配找得到（中心1 px、角度1°內） | 4344／4344，沒有多出的匹配 |
| 最大差異：中心／角點 | 0.00123 px／0.00293 px（subpixel） |
| 最大差異：角度 | 0.00145°（subpixel） |
| 最大差異：分數 | 1.35e-5（Src3 subpixel） |
| 順序改變 | Src10 subpixel的8次，分數接近的匹配交換順序 |
| 速度對best | 0.942／0.958／0.895（再快6%／4%／11%） |

預設與FastMode的位置與角度完全相同（整數像素、角度格點），只有分數差到4.3e-6以內；位置與角度的差異只出現在subpixel，因為二次擬合會放大分數的小差異。差異在8輪之間完全一樣，是確定的，不是雜訊。

所以：對這17組影像，FP32在精度上沒有實際影響，速度再快4～11%。代價是失去「和CPU逐位元相同」的保證：分數在門檻附近1e-5以內的候選可能被收或被丟（這17組沒有發生），分數接近的匹配順序會換（Src10）。c3用binary32先篩、只在可能達到門檻時用binary64重算，已經拿到FP32大部分的好處，輸出又不變，所以研究建議用c3／best，不用fp32。

## 兩個替代方案為什麼比較慢

### 細搜尋請求改回先上傳（u7，transfer bit 4）

上一步發現lean calls下細搜尋kernel慢了約7%，懷疑是kernel從mapped主機記憶體讀請求。u7在每次細搜尋launch前改用`cudaMemcpyAsync`把請求傳到device。結果每次細搜尋呼叫反而變慢：Src1預設模式每幀5次細搜尋合計0.93→1.21 ms（每次約多55 µs），粗搜尋的等待不變。WDDM下每多一次傳輸就多一次命令送出與DMA的延遲，比kernel少讀幾次PCIe貴得多。H3（u7比m3慢）成立，三個模式17組裡只有Src4 FastMode（沒有細搜尋）不慢。

kernel本身的7%有一大半確實是mapped讀取造成的，見下一節的事後診斷；但省下的kernel時間不到多一次傳輸的代價。

### 4個執行緒複製來源（t4）

t4讓lean prepare的來源複製分給4個執行緒（呼叫端加3個常駐的worker），每段都等4份做完才排入上傳。prepare本身確實變快：Src1 0.315→0.155 ms、Src10 0.750→0.388 ms、Src5 0.30→0.16 ms。但接著的粗搜尋等待變長更多：Src1 1.67→2.13 ms、Src5 0.58→1.00 ms，合計反而慢0.25–0.30 ms；小影像只有一段，叫醒worker比複製本身還貴（Src9 prepare 0.027→0.034 ms）。H4（t4比m3慢）成立；17組裡不慢的只有Src7（三個模式）、Src6（FastMode與subpixel）與Src10（subpixel）。

正式arm分不出等待為什麼變長。事後的Nsight診斷（下一節）顯示，變慢的是GPU從主機讀資料的DMA本身，不是叫醒worker。

## 時間去了哪裡（Nsight，不是正式計時）

正式計時之後，用同一個建置剖析m3與best（Nsight Systems，`--trace=cuda,nvtx,wddm`，每幀平均）：

| 影像 | arm | 粗搜尋kernel | 細搜尋kernel | kernel合計 |
|---|---|---:|---:|---:|
| Src1 預設 | m3 | coarseFused 1516 µs | 5×153 µs | 2372 µs |
| Src1 預設 | best | coarseQ28 526 µs | 5×126 µs | 1244 µs |
| Src9 預設 | m3 | 24.8 µs | 3×27.6 µs | 127 µs |
| Src9 預設 | best | 13.5 µs | 3×26.2 µs | 112 µs |
| Src6 subpixel | m3 | 88.6 µs | 27×836 µs | 22.88 ms |
| Src6 subpixel | best | 23.9 µs | 27×611 µs | 16.74 ms |

Src1每幀的kernel時間少了一半（2.37→1.24 ms），正式的每幀時間是3.36→2.22 ms，剩下約1 ms不在kernel（prepare、傳輸、主機挑峰值與送出命令）。Src9的kernel只有約0.11 ms，正式每幀0.30 ms，大部分是固定成本。

Nsight Compute（每個kernel一次launch）：

| kernel | 時間 | 瓶頸 |
|---|---:|---|
| coarseFused，Src1 | 1.85 ms | FP64 pipe 85%，occupancy 82% |
| coarseQ28<3>，Src1 | 0.64 ms | L1／shared memory 96%，FP64 1%，occupancy 65% |
| fineStripes<0>，Src6 subpixel | 1.98 ms | 整數指令（SM 58%），occupancy 33% |
| fineStripes<2>，Src6 subpixel | 1.49 ms | 延遲（SM 41%，每道指令11個週期），occupancy 33% |

粗搜尋的瓶頸從FP64換成shared memory：`windowStats()`每個thread逐byte從shared memory讀區塊與模板。同一次launch的指令只少了19%（251M→203M），時間卻少了65%，因為FP64指令的吞吐量只有FP32的1/64。細搜尋取樣的指令少了，剩下的時間主要是等待；occupancy仍被每個block 49.7 KB的shared memory限制在33%，Nsight估計提高occupancy的上限是59%。另外細搜尋旋轉取樣的global load有一半的sector是多讀的（旋轉後相鄰lane讀的像素不在同一列），這是旋轉本身的性質。

### 事後診斷：上傳請求與複製執行緒

看完正式結果後另外加的Nsight Systems剖析（`profile/posthoc/`，沒有事前登記，不是正式計時，driver也沒有固定在P核），回答上面兩個arm為什麼慢：

| 影像 | arm | fineStripes每次launch | fineNormalize | 每次細搜尋呼叫（NVTX） |
|---|---|---:|---:|---:|
| Src6 subpixel | m0（transfer 0） | 780.3 µs | 3.40 µs | 921 µs |
| Src6 subpixel | m3（請求從mapped讀） | 830.9 µs | 4.72 µs | 873 µs |
| Src6 subpixel | u7（請求先上傳） | 800.1 µs | 4.69 µs | 912 µs |

- **mapped讀取確實讓kernel變慢：** m3比m0每次launch多50.6 µs（＋6.5%），u7只多19.8 µs（＋2.5%），所以約60%的差距來自從mapped記憶體讀請求。這次launch有2520個block，每個SM同時只放得下2個，約分成18波，每波多一次PCIe讀取（約1.7 µs）就差不多是這個數字。剩下的2.5%與fineNormalize的變慢（3.40→4.7 µs，u7也一樣）不是請求造成的，原因沒有再追。
- **但上傳更貴：** u7每次細搜尋呼叫在主機上比m3多39 µs，kernel快的31 µs被多一次`cudaMemcpyAsync`的送出與等待吃掉，還倒賠。

| 影像 | arm | 2 MiB的來源段（GPU時間） | 最後一段0.85 MB | 每幀H2D合計 |
|---|---|---:|---:|---:|
| Src5-0 預設 | m3 | 50.5 µs（41.5 GB/s） | 24.6 µs（34.3 GB/s） | 126 µs |
| Src5-0 預設 | t4 | 116.7 µs（18.0 GB/s） | 231.8 µs（3.6 GB/s） | 525 µs |
| Src1 預設 | m3 | 50.3 µs（41.7 GB/s） | 26.3 µs（32.2 GB/s） | 131 µs |
| Src1 預設 | t4 | 139.6 µs（15.0 GB/s） | 309.9 µs（2.7 GB/s） | 613 µs |

- **4個執行緒複製時，GPU讀主機記憶體的DMA慢了2～12倍：** 每段上傳都是在那一段的4份都複製完之後才排入，最後一段排入時已經沒有執行緒在複製，卻是最慢的一段。所以不是和還在複製的執行緒搶頻寬，也不是叫醒worker的延遲。比較可能的原因是剛寫入的資料還在4個核心各自的快取裡，DMA讀取時要逐條從那些快取取回；這點沒有驗證。單執行緒時同樣的資料只在一個核心，DMA維持32～42 GB/s。
- **結果：** 正式計時裡prepare省下的0.14–0.16 ms，被後面多出的0.4–0.5 ms上傳等待抵掉還倒賠。要讓多執行緒複製有用，可能要讓複製用non-temporal store直接寫回記憶體，或讓呼叫端直接提供page-locked的來源；這次都沒做。

## 為什麼這次沒試int8 Tensor core

Tensor core只加速乘加的部分，這次的profile顯示兩個kernel的時間都不在那裡：

- **細搜尋：** 改動前`__dp4a`只占指令的5.1%。取樣變便宜之後，同一次launch的指令數少了45%（544M→298M），乘加相關的部分（`__dp4a`、funnel shift與word讀取，數量沒變）約占剩下的兩成。但這個kernel現在以等待為主（issue slot只忙35%），occupancy被shared memory限在33%。乘加就算完全交給Tensor core，指令最多少兩成，時間省得更少；提高occupancy（Nsight估計的上限是59%）比較有效。
- **粗搜尋：** 改動前是FP64取樣占85%，乘加不是瓶頸，所以先改取樣。改完之後瓶頸變成`windowStats()`逐byte讀shared memory，這時乘加相關的讀取才是主要成本。

所以Tensor core在粗搜尋這邊現在才有意義。做法可以是：對每一列畫布，算出這一列和模板每一列的相關（A是這一列像素排成的Toeplitz矩陣，B是轉置的模板，N=模板列數），再沿對角線加起來，用`mma.m16n8k32`的u8×u8→s32，整數完全精確。比較簡單的一步是先像細搜尋那樣用word讀取加`__dp4a`，一次算4個相鄰視窗，shared memory的讀取次數可以少到約四分之一。這兩個都留給下一步，順序是先dp4a、再Tensor core，兩者都可以保持逐位元相同。

## 設計

### 粗搜尋Q28區塊（ncc_mode 2）

mode 0的`coarseFused`對區塊裡每個畫布像素算binary64的`warpPixel()`。mode 2的`coarseQ28`改用細搜尋已經在用的Q28精確規則（Metal路徑的規則）：

1. 主機把每個角度的畫布warp換成Q28定點係數與guard（`describeFineWarp()`），連同binary64的係數一起傳給kernel。
2. 每個warp取一列區塊，32個lane各取一個像素，Q28座標每次往右32欄只做整數加法。
3. 雙線性插值的分子用wraparound的uint64算成`top*2^28+(bottom-top)*dy`；真值落在[0, 255·2^56]，所以wraparound的結果是精確的。
4. 分子離捨入點的距離不超過guard的像素，改走binary64的`warpPixel()`，和mode 0一樣。
5. 畫布超出來源的部分是常數border。Q28規則延伸到「來源外補border」的影像上，這個影像仍然連續、Lipschitz常數仍是255，所以guard的論證照樣成立。

區塊大小超過kernel可用的shared memory（opt-in上限減掉static的計數器）時，退回mode 0。

### binary32篩選（ncc_mode 3）

視窗的整數統計Σx、Σx²、Σxt本來就是精確的。mode 3先用精確的中心化整數V = nΣx² − (Σx)²與C = nΣxt − ΣxΣt（n ≤ 10^6時在uint64裡精確）算binary32的分數C / (√V·√Vt)：

- binary32分數低於「門檻 − 1e-4」（往下取到binary32）的視窗直接丟掉；
- 其他視窗用和mode 0相同的`normalizeScore()`算binary64分數，再和門檻比；
- V為0的視窗（例如畫布邊緣的常數區）直接給0，和`normalizeScore()`的結果相同；
- 變異數不到`normalizeScore()`零變異數界限4倍的視窗，一律走binary64。

只有模板不是常數、norm ≥ 0.5（整數像素的非常數模板必定成立）、n ≤ 10^6時才篩選。在篩選的視窗裡，binary64分數離精確NCC不到1.9e-5，binary32分數離它不到6·2^-24，所以binary32低於門檻−1e-4的視窗，binary64必定低於門檻。留下的集合與每個留下的分數都和mode 0相同。

### 細搜尋逐列取樣（fine mode 2）

mode 0的取樣迴圈（`finePixel()`）對ROI每個像素算一次除法與餘數（列與欄），用64位元乘法算Q28座標，四個鄰點都逐一檢查邊界，雙線性分子用三組64位元乘積。mode 2讓一個warp取一整列，lane x取第x欄，Q28座標每次往右32欄只做整數加法；經`q28Sample()`取樣，內部像素直接讀四個鄰點，分子用上面的wraparound形式，產生的byte和mode 0相同（細搜尋的border是0）。模板的複製也改成逐列。

### 不精確的對照（ncc_mode 4、fine mode 3）

`warpPixel32()`把`warpPixel()`的每個量都換成binary32，分數用上面的binary32公式但不再回頭算binary64。API註解標明是INEXACT，輸出和CPU不同。

### 上傳請求（transfer bit 4）與複製執行緒

- **bit 4**（要和bit 2一起用）：lean calls下細搜尋請求先用`cudaMemcpyAsync`傳到device，kernel不讀mapped記憶體。
- **`fpm_frame_set_transfer_threads`**（1..16，預設1）：lean prepare的來源複製分給多個執行緒。frame保留worker；第k個worker複製每一段的第k份，呼叫端複製第0份，每一段的份數都完成後，呼叫端排入這一段的上傳。
- **`fpm_frame_get_cuda_kernel_metrics`**：回報粗搜尋與細搜尋的mode、粗搜尋視窗數、走FP64的區塊像素數、binary64計分的視窗數與複製執行緒數。

## 選擇設計的過程（不是正式計時）

- **ncu1／ncu2：** 8aa1630的Nsight Compute，選出改動目標（見[問題](#問題)）。`pilot/ncu2`另有Src6 subpixel的40次fineStripes launch與一次大launch的SASS指令統計。
- **pilot1：** 開發中的樹上跑兩輪十個arm（多了all3＝c3＋f2＋t4、all7＝c3＋u7＋f2＋t4），510次CUDA執行0不一致，定出登記的預測：c3 0.78／0.76／0.90、f2 0.98／0.99／0.94、u7 1.22／1.19／1.31、t4 1.16／1.11／1.10；fp32 1086／1086找到、最大分數差1.35e-5。all3、all7比best慢，所以正式arm沒有放。
- **篩選的零變異數修正：** 第一版的篩選讓42%的視窗走binary64，因為畫布邊緣的常數視窗變異數為0、不符合篩選條件。改成V為0直接給0之後降到0.1%，並加了突變K13（給零變異數視窗非0分數）。
- **shared memory上限：** 第一版coarseQ28用完整的opt-in上限，加上static的計數器後launch失敗（invalid argument）；改成減掉static部分。細搜尋的三個取樣變體static shared memory不同，預算改用最大的那個。
- **突變pilot：** 在開發中的樹上先跑新的突變（`pilot/mutpilot.log`），K1–K13都被抓到，U5存活。

## 驗證

| 檢查 | 結果 |
|---|---|
| PC ctest（c67ef78，git archive） | 35/35；`experimental_cuda_backend`與`experimental_cuda_kernels`有執行，6個Metal測試跳過 |
| Mac研究版ctest（c67ef78，git archive含`macos-cli/`與`Test Images/`） | 91/91 |
| 輸入 | 27個PGM的SHA256與manifest相同 |
| 逐位元相同 | 計時3672次＋對照510次；精確模式的3264次CUDA執行全部和同輪CPU相同，0失敗 |
| 對照arm | c3 batch 8、c3搭配transfer mode 0（沒有lean calls）、c3搭配CPU細搜尋、f2搭配4個複製執行緒 |
| 核心 | `git diff 8aa1630 c67ef78 -- cpp-core/src cpp-core/include`是空的 |
| H1：best/m3幾何平均 < 0.9 | 成立（0.765／0.745／0.832；預測約0.76／0.76／0.84） |
| H2：至少13/17組變快 | 成立（每個模式17/17） |
| H3：u7/m3 > 1 | 成立（1.220／1.185／1.298） |
| H4：t4/m3 > 1 | 成立（1.155／1.088／1.084） |
| H5：fp32找到每個CPU匹配、沒有多的，分數差 < 1e-4、中心 < 0.01 px、角度 < 0.01° | 成立（4344／4344、0多出；1.35e-5、0.0012 px、0.0015°） |

`cuda_backend_test`新增的部分：14個場景各用c2、c3（batch 64與3）、c3＋f2＋4個執行緒（batch 3）、上傳請求（transfer 7）、c3＋f2＋transfer 7＋4個執行緒、c3＋f2＋transfer 1＋4個執行緒＋CPU細搜尋各prepare兩次，全部和CPU相同；fp32對照只計數（14個場景中2個相同）。metrics測試另外確認4個執行緒真的有用到、c3的binary64計分視窗少於四分之一。`cuda_kernel_test`新增fine mode 2的不一致pose、隨機trial與常數／寬模板，以及粗搜尋mode 0..3在常數與隨機trial上的比對，並印出mode 3的篩選統計（23,121個FP64區塊像素，6,946,200個視窗中2,673,529個走binary64；這裡刻意用分數接近門檻的隨機場景）。

### 突變

`cuda_mutants.py`在c67ef78的另一份副本上逐一編譯53個突變、跑會受影響的測試。

| 組別 | 結果 |
|---|---|
| K1–K13（kernel mode與複製執行緒，列入門檻） | 13/13被抓到 |
| T1–T10（transfer mode，列入門檻） | 10/10被抓到 |
| 之前的24個 | 22被抓到；N2、F2存活，和前兩步相同，[CUDA後端](CUDA_BACKEND_zh-TW.md#驗證)已論證兩者在可達輸入上等價 |
| U1–U6（只回報） | U3、U4、U6被抓到；U1、U2存活（和上一步相同），U5存活 |
| 未突變與還原後的樹 | 都通過 |

K系列涵蓋Q28內部路徑讀過列尾、雙線性權重、guard、border、區塊步長、篩選的邊界（丟掉門檻＋margin以下）、零變異數分數、細搜尋逐列步進、模板補零、複製執行緒漏列、計數器沒複製回主機，以及常數模板與binary64分支的門檻比較。U5（篩選忽略零變異數規則）存活是預期的：零變異數視窗會改走binary64，`normalizeScore()`同樣給0，輸出不變，只是變慢；K13（給它非0分數）則被抓到。U6（每段的份數還沒做完就排入上傳）被抓到。

### 失敗與揭露

- **計時期間的SSH：** 等待腳本在計時開始前每60秒讀一次formal.log，所以計時開始後60秒內（第0輪，暖機，不算）讀了一次；之後每20分鐘讀一次（約第3輪與第7輪），計時結束後的第一次讀到FORMALDONE。每次讀的是整個formal.log（不到200 bytes），不是protocol寫的「一行」。其他時間沒有連線到PC。
- **fp32只有8次逐位元相同：** 這是預期的對照結果，不算失敗；H5量的是位置、角度與分數的差。
- **f2在預設與FastMode不是每組都變快：** 只是組成best的一部分，不在假設裡，照實記錄。
- **cuda_kernel_test的統計和開發時差1：** 開發時印出23,122個FP64區塊像素，c67ef78是23,121；兩者都是同一個測試的診斷輸出，不是gate。（2026-11-01補記：原因在[下一步](CUDA_COARSE_WINDOWS_zh-TW.md#計數器的資料競爭)找到了，tid 0把計數變數歸零時別的warp可能已經開始加，所以本文的FP64區塊像素統計可能略為少算；不影響輸出。）
- **pilot的all3、all7：** 比best慢，沒有放進正式arm；pilot的結果保留在`pilot/`。
- **事後診斷沒有事前登記：** `profile/posthoc/`是看完正式結果才決定跑的Nsight剖析，用來解釋H3、H4的原因，在突變跑完之後執行，driver沒有固定在P核；它的時間不是正式數字。

## 界線

- 一張GPU（RTX PRO 4000 Blackwell）、一個驅動（596.71，WDDM）、只有Windows。
- 只有研究coarse／frame API的frame路徑（粗搜尋mode由frame的coarse optimization設定）；handle API（`fpm_coarse_match`）不受影響。
- binary32篩選的誤差界要求模板不超過10^6像素；更大的模板自動不篩選（mode 3等於mode 2）。區塊放不進shared memory時mode 2、3退回mode 0的路徑，仍然精確；fp32對照在這兩種情況都直接拒絕。
- 小影像（Src3、Src8、Src9）每幀0.3–0.7 ms，大多是固定成本，kernel再快也省不多；CUDA Graphs還沒試。
- 多執行緒複製只試了一般的`memcpy`；non-temporal store、讓呼叫端提供page-locked來源都還沒做。DMA變慢的原因（快取裡的資料）只是推測。
- 粗搜尋的shared memory讀取（dp4a word或int8 Tensor core）與細搜尋的occupancy是下一步，見[為什麼這次沒試int8 Tensor core](#為什麼這次沒試int8-tensor-core)。

## 重現

```bash
# Windows，x64 Native Tools環境，CUDA 13.2的bin在PATH裡
cmake -S cpp-core -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
# 每幀時間、逐位元比對與fp32的差異（start /affinity C03C02 固定在P核）
python cpp-core/tools/benchmark_frame_backend.py --bench build-cuda/experiments/fpm_frame_bench.exe --pgm-dir PGM \
  --arm cpu=0,0,0,0 --arm m3=2,1,256,0,3 --arm c2=2,1,256,2,3 --arm c3=2,1,256,3,3 --arm f2=2,1,256,0,3,2 \
  --arm best=2,1,256,3,3,2 --arm u7=2,1,256,0,7 --arm t4=2,1,256,0,3,0,4 --arm fp32=2,1,256,4,3,3 --inexact fp32 \
  --rounds 8 --repeat 5 --warmup-ms 500 --output timing.json
python cpp-core/tools/frame_backend_summary.py timing.json --reference m3 --arm best --h1 0.9 --h2 13 --output summary.json
python cpp-core/tools/frame_output_deviation.py timing.json --reference cpu --arm fp32 --output deviation.json
# 突變
python cpp-core/tools/cuda_mutants.py build-cuda . mutants.json
```

證據在[`cuda-kernels/`](cuda-kernels/)：`protocol.json`（ea7b229，計時前登記）、`formal/`（摘要、fp32差異、突變結果與`derived_tables.py`）、`pilot/`、`profile/`（`posthoc/`是事後診斷）、`raw-results.tar.gz`、`validation-logs.tar.gz`、`validation.json`、`decision-and-failures.json`、`archive.json`。
