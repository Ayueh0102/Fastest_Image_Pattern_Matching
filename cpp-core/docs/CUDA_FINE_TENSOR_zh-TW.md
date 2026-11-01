# CUDA細搜尋：int8 Tensor core、更多warp，以及全部32位元對照

2026-11-01～02，實驗室PTRI-PRO-4000（NVIDIA RTX PRO 4000 Blackwell，compute capability 12.0，24 GB，PCIe 5.0 x16；驅動596.71，WDDM，沒有接顯示器；Core Ultra 9 285K；Windows 11），CUDA 13.2 Update 2、MSVC 19.44，Release。這是[CUDA後端](CUDA_BACKEND_zh-TW.md)、[transfer mode](CUDA_TRANSFER_zh-TW.md)、[kernel mode](CUDA_KERNELS_zh-TW.md)與[粗搜尋視窗區塊](CUDA_COARSE_WINDOWS_zh-TW.md)之後的第五步，不是研究輪次。新的mode都是選配，預設不變；核心（`src/`、`include/`）一行都沒改，GUI與所有預設不變。

**結論：細搜尋NCC mode 5（Tensor core算乘積、統計分給所有warp、寬ROI每個block用14個warp）的輸出仍和同一個建置的CPU後端逐位元相同。17張影像每幀時間的幾何平均是上一步最好設定（下稱k6）的0.938（預設）、0.960（FastMode）、0.880（subpixel），合計17.68→16.57 ms、14.38→13.83 ms、58.46→50.89 ms；17／16／17組變快，不快的只有FastMode的Src4（慢0.7%）。** 對CPU路徑的幾何平均是0.0398／0.0494／0.0319。Src1 subpixel從9.23降到7.38 ms（0.800），Src6 subpixel從15.24降到13.48 ms（0.885）。金字塔變淺時粗搜尋占大多數，只有0.985／0.952。

**全部改成32位元（粗搜尋mode 7＋細搜尋mode 6）在同樣的kernel上只再快4.5%／4.1%／10.6%（幾何平均0.955／0.959／0.894），代價是408次執行只有8次和CPU逐位元相同。** 精度上：4344個CPU匹配全部找得到，沒有多出的，最大差異0.0012 px、0.0015°、分數1.35e-5。正式計時與對照共4352次執行，其中精確模式3264次全部相同，0失敗。預先登記的七個假設都成立。

Tensor core本身不是單一功臣：單獨換成Tensor core反而比較慢，要和「統計分給所有warp」「block放更多warp」一起才變快，見[選擇設計的過程](#選擇設計的過程不是正式計時)。

## 問題

上一步之後，Src6 subpixel的`fineStripes<4>`一次launch 1.20 ms，占用率（occupancy）33%。使用者要求試Src6 subpixel細搜尋的int8 Tensor core，也問到如果轉成32位元效果如何、為什麼。

## 結果

8輪（第0輪是暖機，不算），每輪每組每個模式各開一個新行程；行程內先記錄冷啟動的第一次呼叫，再不計時地連續呼叫500 ms，之後取5次呼叫的中位數。數字是7輪的中位數，計時期間driver固定在P核。arm的設定是`BACKEND,PYRAMID,FINE_BATCH,粗搜尋ncc_mode,TRANSFER,細搜尋mode`。

### 17張影像（標準設定）

| arm | 設定 | 預設 | FastMode | subpixel | 對k6幾何平均 | 比k6快的組數 |
|---|---|---:|---:|---:|---|---|
| CPU | 0,0,0,0 | 778.9 ms | 388.9 ms | 5044.8 ms | | |
| b2（粗搜尋mode 3＋細搜尋mode 2） | 2,1,256,3,3,2 | 18.94 ms | 15.34 ms | 64.01 ms | 1.056／1.066／1.042 | 3／3／2 |
| k6（基準，上一步最好） | 2,1,256,6,3,4 | 17.68 ms | 14.38 ms | 58.46 ms | 1 | |
| **t5（細搜尋mode 5）** | 2,1,256,6,3,5 | **16.57 ms** | **13.83 ms** | **50.89 ms** | **0.938／0.960／0.880** | **17／16／17** |
| f32（全部32位元，不精確） | 2,1,256,7,3,6 | 15.79 ms | 13.31 ms | 45.66 ms | 0.895／0.920／0.786 | 17／17／17 |
| old32（舊kernel的全部32位元，不精確） | 2,1,256,4,3,3 | 17.74 ms | 14.72 ms | 55.13 ms | 0.997／1.022／0.932 | 10／6／16 |

時間是17組合計；幾何平均與組數依序是預設／FastMode／subpixel。同一個建置裡t5對b2是0.889／0.901／0.844，17/17組變快。冷啟動t5對k6是0.978／0.996／0.946。

各組（ms）：

| 影像 | 預設CPU | 預設k6 | 預設t5 | t5/k6 | subpixel CPU | subpixel k6 | subpixel t5 | t5/k6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Src1 | 100.94 | 1.838 | 1.683 | 0.916 | 878.83 | 9.225 | 7.377 | 0.800 |
| Src2 | 51.80 | 1.204 | 1.116 | 0.927 | 407.31 | 4.995 | 4.119 | 0.825 |
| Src3 | 5.46 | 0.485 | 0.454 | 0.936 | 9.93 | 0.645 | 0.594 | 0.920 |
| Src4 | 18.76 | 0.578 | 0.558 | 0.965 | 49.19 | 4.231 | 3.966 | 0.938 |
| Src6 | 323.20 | 3.057 | 2.810 | 0.919 | 2549.95 | 15.235 | 13.480 | 0.885 |
| Src7 | 98.24 | 1.946 | 1.798 | 0.924 | 690.66 | 6.267 | 5.523 | 0.881 |
| Src8 | 15.69 | 0.628 | 0.566 | 0.902 | 131.15 | 2.746 | 2.412 | 0.878 |
| Src9 | 0.521 | 0.300 | 0.261 | 0.869 | 0.748 | 0.386 | 0.322 | 0.835 |
| Src10 | 26.26 | 1.429 | 1.410 | 0.987 | 29.48 | 1.682 | 1.630 | 0.969 |
| Src5（8組） | 16.4–17.9 | 0.75–0.80 | 0.72–0.76 | 0.937–0.961 | 35.3–41.4 | 1.43–1.94 | 1.27–1.66 | 0.858–0.889 |

上一步Src3、Src8、Src9比b2慢，這一步這三組都變快了：統計分給所有warp之後，它們的細搜尋launch短了很多。

### 金字塔變淺（頂層面積下限×16）

| arm | 預設 | subpixel | 對b2幾何平均 |
|---|---:|---:|---|
| b2 | 1106.8 ms | 1189.0 ms | 1 |
| k6 | 185.4 ms | 260.2 ms | 0.220／0.290 |
| **t5** | **184.5 ms** | **247.7 ms** | **0.216／0.276** |
| f32（不精確） | 175.3 ms | 231.4 ms | 0.207／0.259 |

t5對k6是0.985／0.952（14／17組變快；預設模式不快的是Src4、Src5-0、Src5-180，都在1.001以內）。這裡粗搜尋占大多數，細搜尋變快看不太出來。f32對t5是0.955／0.939。

### 記憶體

- **行程峰值RSS：** 標準設定所有CUDA arm中位數155.5 MiB（最大175.9 MiB），差不到0.1 MiB。金字塔變淺時Src1 2154 MiB，所有arm一樣（粗搜尋結果緩衝是最壞情況配置，見上一篇）。
- **保留的device bytes：** k6與t5相同。新模式多用的是block內的shared memory（每個block 49.7→50.0 KB），不是device bytes。

## 如果全部轉成32位元

使用者問的「32位元」有兩種，先分清楚：

| 種類 | 現況 | 輸出 |
|---|---|---|
| 粗搜尋的binary32分數篩選、細搜尋的32位元整數取樣篩選 | **已經在用**（k6、t5都有），只有不確定的才退回64位元 | 和CPU逐位元相同 |
| 全部改成binary32（f32） | 只作不精確對照 | 不同 |

**為什麼只再快這麼多：** 精確模式需要64位元的地方已經很少。粗搜尋1161萬個視窗只有0.11%算binary64，細搜尋只有約0.1%的像素走FP64（Src6 subpixel 0.27%），所以把這些也改成32位元省不到多少。Nsight的數字（不是正式計時，每幀kernel時間）：

| 影像 | k6 | t5 | f32 | f32/t5 |
|---|---:|---:|---:|---:|
| Src1 subpixel | 6.97 ms | 5.08 ms | 4.13 ms | 0.814 |
| Src6 subpixel | 13.13 ms | 11.33 ms | 10.37 ms | 0.915 |
| Src5-0 預設 | 0.232 ms | 0.194 ms | 0.157 ms | 0.809 |
| Src9 subpixel | 0.182 ms | 0.121 ms | 0.103 ms | 0.851 |

f32省下的幾乎都在細搜尋取樣：Src6 subpixel每幀kernel少0.96 ms，其中`fineNormalize`改binary32只省約0.05 ms。粗搜尋在大模板上完全沒差（Src6 ×16：1469 vs 1454 µs），因為那裡的時間在乘積，不在取樣。

**f32的指令其實比較多。** Src6 subpixel一次launch：t5 215M道指令、IPC 1.54、1.00 ms；f32 261M道、IPC 2.05、0.91 ms。取樣階段的指令從178M變成224M，等待的樣本卻從44,677降到39,072。比較可能的原因是32位元整數篩選用很多`IMAD`（2700萬道），而FP32的`FMUL`／`FADD`吞吐量比較好；這點沒有驗證。

**精度差多少（f32對CPU）：**

| 項目 | 標準設定 | 金字塔變淺（對t5） |
|---|---:|---:|
| 逐位元相同的執行 | 408次中8次 | 272次中8次 |
| 匹配找得到（中心1 px、角度1°內） | 4344／4344 | 2896／2896 |
| 多出的匹配 | 0 | 0 |
| 最大差異：中心／角點 | 0.00123 px／0.00293 px | 0.00123 px／0.00372 px |
| 最大差異：角度 | 0.00145° | 0.00104° |
| 最大差異：分數 | 1.35e-5 | 1.35e-5 |
| 順序改變 | 8次（都是Src10 subpixel） | 8次（都是Src10 subpixel） |

預設與FastMode的位置與角度完全相同，只有分數差（4.3e-6以內）；位置與角度的差異只出現在subpixel，因為二次擬合會放大分數的小差異。差異在8輪之間完全一樣，是確定的，不是雜訊。f32和old32在51個（影像、模式）的輸出也逐位元相同，因為兩者的32位元算術一樣，差別只在kernel；這也是Tensor core算對乘積的另一個證據。

**怎麼選：**

- **要和CPU逐位元相同、要能驗證：** 用t5。
- **subpixel很重、可以接受1e-5等級的差異：** f32再快約10.6%，17張影像上沒有漏掉或多出匹配。但分數離門檻1e-5以內的候選可能被收或被丟（這17組沒有發生），分數接近的匹配順序會換。
- **old32不用再考慮：** 它在所有模式17/17組都比精確的t5慢（1.063／1.065／1.059）。上一篇還能說「FP32再快4～11%」，現在舊kernel的FP32已經輸給精確模式。

## 時間去了哪裡（Nsight，不是正式計時）

Nsight Compute，Src6 subpixel的一次大launch：

| 細搜尋mode | 時間 | 指令 | IPC | occupancy | 取樣階段指令 | 乘積階段指令 | 等待最後barrier的樣本 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4（k6） | 1.20 ms | 236M | 1.39 | 33% | 170M | 64.5M | 14,117 |
| 5（t5） | 1.00 ms | 215M | 1.54 | 57% | 178M | 34.6M | 3,270 |
| 6（f32） | 0.91 ms | 261M | 2.05 | 57% | 224M | 34.2M | 3,144 |

乘積階段的指令少了46%。取樣階段的指令反而多4%，因為stripe變短、block從2520個變成2800個。t5的block是50.0 KB，每個SM仍放2個，但warp從8個變14個，所以occupancy從33%變57%。Nsight估計再提高occupancy的上限是42%，但現在的限制是暫存器（每個thread 72個）與shared memory同時卡住。

窄ROI也變快：Src4 subpixel的一次launch 29.1→19.7 µs，指令3.42M→3.05M，這裡用的是8個warp的`fineStripes<6>`。

Nsight Systems，每幀kernel：

| 影像 | arm | 細搜尋 | 粗搜尋 | kernel合計 |
|---|---|---:|---:|---:|
| Src1 subpixel | k6 | 29×220 µs＋4×56 µs | 157 µs | 6.97 ms |
| Src1 subpixel | t5 | 29×156 µs＋4×44 µs | 158 µs | 5.08 ms |
| Src6 subpixel | k6 | 22×576 µs＋5×46 µs | 19 µs | 13.13 ms |
| Src6 subpixel | t5 | 22×497 µs＋5×32 µs | 19 µs | 11.33 ms |
| Src9 subpixel | k6 | 5×27 µs | 19 µs | 0.182 ms |
| Src9 subpixel | t5 | 5×15 µs | 19 µs | 0.121 ms |

## 設計

### 細搜尋mode 5

1. **一次`m16n8k32`算兩列ROI。** 細搜尋每個請求要49個視窗（dx、dy各7種）。對ROI的一對列(ρ, ρ+1)，A的第m列（m = dx）放第ρ列從dx開始的像素，第dx+8列放第ρ+1列，B的第n欄放模板第ρ+1−n列。於是累加器(dx, n)屬於視窗(dx, n−1)，(dx+8, n)屬於視窗(dx, n)。這樣16×8個累加器裡用得到98個。
2. **不做邊界檢查。** 模板列補零到32 bytes的倍數，迴圈直接走滿；模板之後多放一列全0，代表超出這個stripe的列，不必逐次遮罩。
3. **統計分給所有warp。** 原本warp 7專做每列的和與平方和，乘積warp變快之後它就是最慢的。現在第w個warp負責編號除以warp數餘w的列對與統計列。
4. **寬ROI用14個warp。** ROI寬度≥256時每個block 448個thread，比較窄的維持256個。每個SM仍放2個block，所以warp多了occupancy就上去；16個warp時每個SM只放得下1個block，反而慢。
5. **取樣照舊。** 寬ROI用上一篇的`screenRow()`，窄ROI用`q28Sample()`。

每個warp的累加器只累加幾列，用uint32是精確的；主機確認`2 × 每個warp的列對數 × tw ≤ 66051`，否則退回mode 4／2（這張卡的shared memory放不下會觸發的模板，所以實際到不了）。mode 5要求compute capability ≥ 8.0。

### 不精確的對照（粗搜尋mode 7、細搜尋mode 6）

粗搜尋mode 7是mode 6的kernel把取樣換成`warpPixel32()`、分數用binary32不再回頭算binary64；放不進視窗區塊時退回mode 4。細搜尋mode 6是mode 5的乘積配`warpPixel32()`，分數用`fineNormalize<true>`。API註解標明是INEXACT。

## 選擇設計的過程（不是正式計時）

第一版和第二版都比mode 4慢，全部保留在`pilot/`。同一次Src6 subpixel launch：

| 版本 | 時間 | 指令 | 問題 |
|---|---:|---:|---|
| mode 4（基準） | 1.20 ms | 230M | |
| 第一版：一次`mma`一列，統計在warp 7 | 1.26 ms | 226M | 每次`mma`帶約27.5道指令（邊界檢查、補0的暫存器），乘積階段只少10%；stripe還變短 |
| 第二版：一次`mma`兩列 | 1.30 ms | 213M | 乘積指令減半，但等待barrier的時間沒變：warp 7的統計變成最慢的 |
| 第三版：統計分給所有warp | 1.12 ms | 211M | barrier的等待幾乎消失 |
| 最終版：寬ROI 14個warp | 1.00 ms | 215M | occupancy 57% |

其他pilot：

- **ROI寬度門檻（第三版，8個warp）：** Tensor core只用在寬度≥T的ROI，10組影像對k6的幾何平均（預設／subpixel）：T0 0.941／0.895、T64 0.956／0.917、T128 0.979／0.929、T256 0.996／0.953、T512 0.986／0.970。所以不設門檻，所有寬度都用。
- **縮小shared memory（否決）：** 想讓每個SM多放block。目標改成三分之一時，每個block 36.4 KB，但Nsight顯示每個SM仍只放2個block，occupancy還是33%，時間1.25 ms（比1.12 ms慢）；要低於約33 KB才放得下3個。目標四分之一時總時間也比較慢（預設／subpixel 11.88／45.39 ms，對上42.75 ms的subpixel）。stripe變短、重複取樣的列比例變高，代價比較大。D=4的occupancy沒有用Nsight確認。
- **每個block的warp數（Src6 subpixel一次launch，Nsight）：**

  | warp | 時間 | occupancy | 每個SM的block |
  |---:|---:|---:|---:|
  | 8 | 1.12 ms | 33% | 2 |
  | 10 | 1.00 ms | 42% | 2 |
  | 12 | 1.01 ms | 50% | 2 |
  | 14 | 0.99 ms | 58% | 2 |
  | 16 | 1.45 ms | 33% | 1 |
  | 20 | 1.46 ms | 42% | 1 |
  | 24 | 1.52 ms | 50% | 1 |

  10組影像對k6的幾何平均（預設／FastMode／subpixel）：8個warp 0.936／0.943／0.895，10個0.936／0.955／0.889，12個0.938／0.968／0.888，14個0.931／0.977／0.879。寬ROI受益，窄ROI（Src4）多warp反而慢，所以只有寬度≥256才用14個。
- **最後一輪pilot（定案的樹，10組）：** t5/k6 0.929／0.947／0.881，f32/t5 0.955／0.960／0.913，old32/t5 1.074／1.080／1.074，定出登記的預測。
- **突變pilot：** 新的M系列都抓到，U10、U12存活；U11（shared memory少算全0列）抓到，kernel測試回報illegal memory access。

## 驗證

| 檢查 | 結果 |
|---|---|
| PC ctest（a6aa630，git archive） | 35/35；`experimental_cuda_backend`與`experimental_cuda_kernels`有執行，6個Metal測試跳過 |
| Mac研究版ctest（a6aa630，git archive含`macos-cli/`與`Test Images/`） | 91/91 |
| 輸入 | 27個PGM的SHA256與manifest相同 |
| 逐位元相同 | 計時2448＋1088次、對照612＋204次，共4352次；精確模式3264次（其中CUDA 2686次）每一組（輪、影像、模式）相同，跨輪、跨檔案也相同；0失敗。`formal/identity_check.py`不用driver的比對，直接比`matches` |
| 對照arm | t5 batch 8、t5搭配transfer mode 0（沒有lean calls）、搭配粗搜尋mode 5、搭配粗搜尋mode 3＋transfer 0、搭配4個複製執行緒；金字塔變淺的cpu、b2、t5 |
| 核心 | `git diff 806fb0d a6aa630 -- cpp-core/src cpp-core/include`是空的 |
| H1：標準設定t5/k6幾何平均 < 0.97 | 成立（0.938／0.960／0.880；pilot 0.929／0.947／0.881） |
| H2：標準設定Src6 subpixel t5/k6 < 0.93 | 成立（0.885；pilot 0.882–0.890） |
| H3：每個模式至少14/17組比k6快 | 成立（17／16／17） |
| H4：f32/t5在0.85到1之間 | 成立（0.955／0.959／0.894） |
| H5：f32找到每個CPU匹配、沒有多的，分數差 < 1e-4、中心 < 0.01 px、角度 < 0.01° | 成立（4344／4344、0多出；1.35e-5、0.0012 px、0.0015°） |
| H6：金字塔變淺t5/k6 < 1 | 成立（0.985／0.952） |
| H7：old32/t5 > 1 | 成立（1.063／1.065／1.059） |

`cuda_backend_test`新增：14個場景各用t5的4種組合（粗搜尋6＋transfer 3、粗搜尋5＋4個複製執行緒＋batch 3、粗搜尋3＋transfer 0、粗搜尋6＋transfer 1＋batch 3）各prepare兩次，全部和CPU相同；f32對照要能執行（14個場景中2個相同）；不接受ncc_mode 8與fine mode 7。`cuda_kernel_test`新增：mode 5的不一致pose、隨機、邊界、影像外、特殊角度與repairAll（262×41），以及20種模板形狀，圍繞32欄步長、warp數與stripe高度（1×2到300×33，含全白來源）。

### 突變

`cuda_mutants.py`在a6aa630的另一份副本上逐一編譯99個突變、跑會受影響的測試。

| 組別 | 結果 |
|---|---|
| M1–M17（細搜尋mode 5，列入門檻） | 17/17被抓到 |
| L1–L22、L24（視窗區塊與無分支篩選，列入門檻） | 23/23被抓到 |
| K1–K13、T1–T10（kernel mode與transfer mode，列入門檻） | 23/23被抓到 |
| 之前的24個 | 22被抓到；N2、F2存活，和前幾步相同，[CUDA後端](CUDA_BACKEND_zh-TW.md#驗證)已論證兩者在可達輸入上等價 |
| U1–U12（只回報） | U3、U4、U6、U11被抓到；U1、U2、U5、U7、U8、U9、U10、U12存活 |
| 未突變與還原後的樹 | 都通過 |

新存活的U10、U12是預期的：U10是mode 5不用篩選，只是變慢；U12拿掉uint32上限，shared memory放得下的模板都到不了。

### 失敗與揭露

- **我先前說錯occupancy：** [上一篇](CUDA_COARSE_WINDOWS_zh-TW.md#細搜尋為什麼還沒用tensor-core)建議「先減少每個block的shared memory」，也估計Tensor core「最多少約10%的指令」。兩個都不準：縮小到36 KB時occupancy根本沒變，真正有效的是增加每個block的warp；指令實際少9%，但時間少17%（1.20→1.00 ms）。pilot過程中我也一度說「拉高occupancy沒有變快」，那時occupancy其實沒有被拉高，Nsight看到才更正。
- **Tensor core的貢獻沒有單獨分開：** 正式arm只比了三件事合在一起的結果。第一版只換Tensor core比較慢，所以不能說「Tensor core快了12%」。乘積階段用`__dp4a`配上統計分散與14個warp會多快，**沒有量**，它可能接近t5。
- **第一次突變pilot失敗：** `pilot6.bat`沒有設好PATH，`cmake`找不到；修好後重跑。失敗的log保留在`pilot/mutpilot-no-path-failure.log`。
- **第一輪寬度pilot中止：** 跑到第二個寬度時發現設計有問題，就停掉改設計；留下的`w0`、`w64`在`pilot/pilot-raw.tar.gz`的`designs/first/`。
- **pilot的環境變數：** 寬度、shared memory與warp數當時用`FPM_PILOT_*`切換，正式版本全部移除。
- **FastMode的Src4沒有變快（1.007）：** H3允許最多3組不快，照實記錄。
- **mode 4的二進位也有小改動：** 取樣迴圈共用`warps = blockDim.x / 32`，同一次launch的指令從230M變236M，時間同為1.20 ms。比值都用同一個建置的k6。
- **計時期間的連線：** 等待腳本在計時開始前每60秒讀一次formal.log，計時開始後60秒內讀了一次，之後每20分鐘讀一次（每次不到400 bytes），符合protocol。計時期間使用者問進度時，我只讀本機的等待紀錄。Mac上同時做了ctest與磁碟清理，沒有連到PC。
- **每輪之間的變動：** Src9等小影像（每幀0.2–0.3 ms）最大有19%的輪間變動，所以小影像的單組比值參考就好，幾何平均比較可靠。

## 界線

- 一張GPU（RTX PRO 4000 Blackwell）、一個驅動（596.71，WDDM）、只有Windows。
- 只有研究coarse／frame API的frame路徑；handle API不受影響。
- 只量時間與輸出是否相同，沒有量召回；f32對CPU的差異不等於對ground truth的差異。
- 14個warp是這張卡上選出來的（72個暫存器、102.4 KB shared memory設定）；別張卡可能不同。
- `__dp4a`搭配同樣的warp安排沒有量；粗搜尋大模板的`packEntries`與最壞情況配置的結果緩衝沒有改。

## 重現

```bash
# Windows，x64 Native Tools環境，CUDA 13.2的bin在PATH裡
cmake -S cpp-core -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
# 標準設定（start /affinity C03C02 固定在P核）
python cpp-core/tools/benchmark_frame_backend.py --bench build-cuda/experiments/fpm_frame_bench.exe --pgm-dir PGM \
  --arm cpu=0,0,0,0 --arm b2=2,1,256,3,3,2 --arm k6=2,1,256,6,3,4 --arm t5=2,1,256,6,3,5 --arm f32=2,1,256,7,3,6 \
  --arm old32=2,1,256,4,3,3 --inexact f32 --inexact old32 --rounds 8 --repeat 5 --warmup-ms 500 --output timing-standard.json
# 金字塔變淺
python cpp-core/tools/benchmark_frame_backend.py --bench build-cuda/experiments/fpm_frame_bench.exe --pgm-dir PGM \
  --min-area-scale 16 --modes exhaustive,subpixel --arm b2=2,1,256,3,3,2 --arm k6=2,1,256,6,3,4 --arm t5=2,1,256,6,3,5 \
  --arm f32=2,1,256,7,3,6 --inexact f32 --rounds 8 --repeat 5 --warmup-ms 500 --output timing-shallow.json
python cpp-core/tools/frame_backend_summary.py timing-standard.json --reference k6 --arm t5 --arm f32 --h1 0.97 --h2 14 --output summary.json
python cpp-core/tools/frame_output_deviation.py timing-standard.json --reference cpu --arm f32 --output deviation.json
# 突變
python cpp-core/tools/cuda_mutants.py build-cuda . mutants.json
```

證據在[`cuda-fine-tensor/`](cuda-fine-tensor/)：`protocol.json`（6e7cd03，計時前登記）、`formal/`（摘要、32位元差異、獨立的一致性重算、突變結果與`derived_tables.py`）、`pilot/`（三個否決版本與各pilot）、`profile/`、`raw-results.tar.gz`、`validation-logs.tar.gz`、`validation.json`、`decision-and-failures.json`、`archive.json`。
