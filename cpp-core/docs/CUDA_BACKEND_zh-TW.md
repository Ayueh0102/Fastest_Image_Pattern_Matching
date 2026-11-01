# CUDA後端：研究coarse／frame API的精確GPU路徑

2026-11-01，實驗室PTRI-PRO-4000（NVIDIA RTX PRO 4000 Blackwell，compute capability 12.0，70個SM，24 GB；驅動596.71，WDDM，沒有接顯示器；Core Ultra 9 285K；Windows 11），CUDA 13.2 Update 2（nvcc V13.2.86）、MSVC 19.44，Release。這是底層優化路線（CPU→NVIDIA GPU→Jetson）的GPU第一步，不是研究輪次。新的程式只在`FPM_WITH_CUDA=ON`時編譯，是研究coarse／frame API的backend 2；核心、GUI與所有預設都不變。

**結論：CUDA後端的輸出和同一個建置的CPU後端逐位元相同（正式計時與對照共1632次執行，其中1122次CUDA執行全部和同一輪的CPU結果相同，0失敗）；17張影像每幀時間（prepare＋match）的幾何平均是CPU的0.142（預設）、0.187（FastMode）、0.097（subpixel），合計768.6→59.3 ms、377.3→53.8 ms、4966→135 ms。** 每個模式都是16/17組變快；最小的Src9在GPU上比較慢（0.52→0.64 ms），事前就預測到。大影像的時間現在大多在prepare：Src7的6.38 ms裡4.62 ms是複製來源與傳輸金字塔，match只要1.76 ms。CUDA的行程峰值RSS是126–171 MiB（CPU 5–60 MiB）。預先登記的突變gate沒有通過：24個中19個被抓到；補強測試後重跑是22/24，存活的2個是可論證的等價突變，見[驗證](#驗證)。

後續：prepare與同步的成本由選配的[transfer mode](CUDA_TRANSFER_zh-TW.md)處理（mode 3每幀時間約為本文mode 0的0.42倍），kernel本身由選配的[kernel mode](CUDA_KERNELS_zh-TW.md)處理（再降到transfer mode 3的約0.77倍，輸出不變），之後的[粗搜尋視窗區塊與細搜尋篩選](CUDA_COARSE_WINDOWS_zh-TW.md)再降到約0.95倍（頂層模板變大時約0.22倍），[細搜尋Tensor core](CUDA_FINE_TENSOR_zh-TW.md)再降到0.94／0.96／0.88倍。

## 結果

8輪（第0輪是暖機，不算），每輪每組每個模式各開一個新行程；行程內先記錄冷啟動的第一次呼叫，再不計時地連續呼叫500 ms，之後取5次呼叫的中位數。數字是7輪的中位數，計時期間CPU固定在P核。`cuda_coarse`只開GPU粗搜尋（CPU金字塔與細搜尋），是粗搜尋貢獻的對照，不在gate裡。

| 模式 | CPU | CUDA | CUDA/CPU 合計 | 幾何平均 | 變快 | 只開GPU粗搜尋的幾何平均 |
|---|---:|---:|---:|---:|---:|---:|
| 預設 | 768.6 ms | 59.3 ms | 0.077 | **0.142** | 16/17 | 0.677 |
| FastMode | 377.3 ms | 53.8 ms | 0.143 | **0.187** | 16/17 | 0.584 |
| subpixel | 4966.3 ms | 134.9 ms | 0.027 | **0.097** | 16/17 | 0.880 |

冷啟動（行程的第一次呼叫，含GPU buffer配置與尚未升頻）：預設模式合計846.6→118.0 ms，幾何平均0.263；FastMode 0.369；subpixel 0.133。

預設模式各組（ms）：

| 影像 | CPU | CUDA | 其中prepare | 其中match | 細搜尋GPU時間 | 細搜尋FP64像素 | RSS MiB CPU/CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Src1 | 100.47 | 5.46 | 1.96 | 3.49 | 1.289 | 21,797／1960萬 | 26／171 |
| Src2 | 50.41 | 3.49 | 1.99 | 1.51 | 0.925 | 11,668／1078萬 | 26／142 |
| Src3 | 5.29 | 1.38 | 0.80 | 0.58 | 0.393 | 0／77萬 | 15／129 |
| Src4 | 18.38 | 1.44 | 0.14 | 1.31 | 0.328 | 90／67萬 | 5／144 |
| Src6 | 321.21 | 8.17 | 4.61 | 3.58 | 2.854 | 168,412／7455萬 | 56／157 |
| Src7 | 96.13 | 6.38 | 4.62 | 1.76 | 1.279 | 13,158／1975萬 | 56／153 |
| Src8 | 16.26 | 1.11 | 0.19 | 0.92 | 0.618 | 366／387萬 | 6／126 |
| Src9 | 0.52 | **0.64** | 0.18 | 0.46 | 0.341 | 0／11萬 | 7／126 |
| Src10 | 25.34 | 6.06 | 4.89 | 1.17 | 0.274 | 0／49萬 | 60／168 |
| Src5（8個角度） | 16.13–17.47 | 3.08–3.19 | 1.95–1.99 | 1.11–1.21 | 0.494–0.544 | 413–540／75–93萬 | 26／150 |

subpixel差最多：Src6 2570→29.5 ms、Src1 878→17.6 ms、Src7 601→14.0 ms。完整數字在[timing-summary.json](cuda-backend/formal/timing-summary.json)。

怎麼讀：

- **粗搜尋已經很小**：每次match只同步一次；Src7的GPU時間0.14 ms。只開GPU粗搜尋時合計只少19%（預設），因為CPU的時間大多在細搜尋。
- **細搜尋**：每次match有2–7批（subpixel 5–41批），Src6預設模式2.85 ms GPU時間。FP64像素占ROI像素的0–0.27%，沒有整個ROI改走FP64。
- **prepare**：frame API每次prepare都把來源複製到新配置的記憶體（Src7約2.6 ms，CPU也一樣），GPU還要經PCIe上傳level 0、拷回各層（約1.5 ms）。這是獨立顯示卡的固定成本；Jetson的統一記憶體可以省掉傳輸。
- **小影像**：Src9每次呼叫有1次粗搜尋同步、3批細搜尋與金字塔，每次來回約60 µs，比CPU的0.52 ms還多。
- **記憶體**：device配置（`gpu_bytes`）中位數19.8 MiB、最大39.3 MiB（Src1）；細搜尋的模板副本另外最多0.79 MiB（正式計時時`gpu_bytes`還沒算這一項，見[失敗與修正](#失敗與修正)）。RSS多出的約100 MiB是CUDA context與page-locked buffer，和device配置是不同的範圍，不相加。

## 剩下的時間（Nsight Systems，不是正式計時）

正式計時結束後，用Nsight Systems（`--trace=cuda,nvtx,wddm`）跑`9623880`的正式建置，每個行程先暖機1秒再連續呼叫20次，摘要在[profile/](cuda-backend/profile/)：

| 案例 | 每次呼叫的NVTX範圍 | 每次呼叫的kernel時間 | 主要成本 |
|---|---|---|---|
| Src7預設 | pyramid 1.94 ms、6批fine共1.49 ms、coarse 0.18 ms | fine 0.72 ms、pyrDown 0.09 ms、coarse 0.09 ms | pageable記憶體的上傳與拷回（memcpy平均58 µs，level 0約1 ms）與同步 |
| Src6預設 | pyramid 2.16 ms、7批fine共3.36 ms、coarse 0.25 ms | fine 2.20 ms | fine kernel與傳輸各半 |
| Src6 subpixel | 27批fine共25.0 ms | fine 21.1 ms（GPU時間的98.7%） | fine kernel |
| Src9預設 | pyramid 0.23 ms、3批fine共0.46 ms、coarse 0.11 ms | fine 0.08 ms、coarse 0.025 ms | 約5次同步（平均69 µs）與11次memcpy（平均22 µs） |

所以接下來能改善的方向是：

- **大影像**：prepare的傳輸。level 0直接放page-locked記憶體、分段上傳並和金字塔重疊，拷回各層和粗搜尋重疊。Jetson的統一記憶體可以整個省掉。
- **小影像與每批細搜尋**：同步與memcpy的次數。小請求可以改用mapped memory，或把一層的多批合成一次送出；細搜尋每一層要等主機決定下一層，CUDA Graphs只能省下啟動成本。
- **subpixel與大模板**：細搜尋kernel本身。點積可以改用int8 Tensor core（u8的`mma`，int32累加是精確的；ROI列對模板列是Toeplitz形式），或一個block同時處理多個請求、共用模板列。

## 設計

### 接在哪裡

研究coarse／frame API本來就有後端抽象（`fpm_coarse_detail::Backend`，CPU是0、Metal是1），核心的`matchWithCoarseNcc()`透過`CoarseNccProvider`呼叫它。CUDA做成第2個後端（`experiments/coarse_cuda.cu`），用法和Metal一樣：

- `fpm_coarse_create_backend(..., 2, ...)`：GPU粗搜尋，CPU金字塔與細搜尋；
- `fpm_frame_create`的`backend=2`：`pyramid_mode=1`用GPU建金字塔，`fine_batch_size>0`用GPU細搜尋；`fpm_frame_set_coarse_optimization`的`ncc_mode=1`是不融合的對照。

核心多了兩個不改變CPU運算的部分（`500d1cd`）：

- `canvasWarp()`／`roiWarp()`：把`rotateCanvas()`、`rotatedRoi()`在主機上算的反向映射（anchor、cos、sin、大小）獨立出來。CPU與CUDA用同一份主機運算，取樣座標一定相同。
- `deviceCoarse()`／`computeCoarseDevice()`：一次交給provider所有角度，provider回傳每個角度的畫布大小、平移量，以及`selectPeaks()`會挑出的峰值。`selectPeaksSparse()`只用分數達門檻的項目就能挑出和`selectPeaks()`相同的峰值，`core_test`在300張隨機map上比對兩者。

### 為什麼能和CPU逐位元相同

x86-64的CPU路徑不合併乘加（見[x86 SIMD](CORE_X86_SIMD_zh-TW.md)），所以GPU只要照同樣的順序、每個運算各自捨入：

- **旋轉**：主機算好`WarpMap`；GPU用`__dmul_rn`／`__dadd_rn`／`__dsub_rn`照`warpPixel()`的順序做FP64。這些intrinsic不受`-fmad`影響；整個目標另外用`-fmad=false`編譯。
- **NCC統計**：視窗的和、平方和與模板乘積都是精確整數。
- **正規化**：照`nccDenominator()`與`nccNormalizeProduct()`的binary64運算，用`__ddiv_rn`、`__dsqrt_rn`（IEEE正確捨入）。
- **峰值**：GPU只留下分數≥門檻的項目，主機用`selectPeaksSparse()`挑峰值。

### 粗搜尋：一次launch、不寫score map（FlashAttention的作法）

所有角度在一次launch裡完成。每個block負責一個角度的32×8個視窗：把需要的畫布區塊直接從頂層來源旋轉到shared memory，算完這些視窗的統計與分數，只把達門檻的項目用warp聚合的atomic寫出。畫布和score map都不會寫進device memory；這和FlashAttention把中間的大矩陣留在晶片上是同一個想法。另一個kernel把各角度的計數與項目寫到mapped pinned host memory，所以整個粗搜尋只同步一次、沒有`cudaMemcpy`。`ncc_mode=1`則把畫布寫進device memory再算（對照用，也是模板太大放不進shared memory時的路徑）。

### 細搜尋：Q28快速路徑＋FP64後備

Nsight Compute顯示第一版細搜尋kernel卡在FP64管線（68.8%；這張卡FP32:FP64是64:1）。現在的版本：

- **取樣**：沿用Metal路徑已證明的精確Q28整數雙線性取樣（`fine_sampling.hpp`）。主機替每個請求算保守的誤差上限（rounding guard）；結果離四捨五入門檻比上限近的像素，或上限不成立的請求（`repairAll`），GPU直接改用FP64 `warpPixel()`重算。Metal沒有FP64，要把這些像素送回CPU修正；CUDA在GPU上就能修正，所以整條路徑留在GPU。
- **點積**：每個block處理一個請求的最多32列模板（stripe）。warp dy（0..6）的lane r負責模板第r列，用`__dp4a`一次算4個像素、同時保留7個dx的和（漏斗位移取出錯位的4 bytes）。
- **視窗和與平方和**：第8個warp算每一列ROI的和，再滑動到7個dx，不放進49個視窗的迴圈。
- **共用記憶體**：列寬取奇數個word，32個lane落在不同bank；stripe大小優先讓一個SM放得下兩個block。
- **模板**：同一次match裡上傳過的模板層直接重用。

### 其他

- **金字塔**：GPU上做精確的`pyrDown()`（整數、REFLECT_101），再拷回主機，因為frame API的主機金字塔語意不變。
- **NVTX**：pyramid、coarse、fine batch、source upload各有NVTX範圍，Nsight Systems時間軸上可以直接看各階段。
- **記憶體**：device與page-locked buffer只增不減；`gpu_bytes`只算device配置，page-locked host buffer算在行程RSS裡。

## 選擇設計的過程（不是正式計時）

以下都在protocol登記前、未commit的樹上做，只用來挑設計，資料放在[pilot/](cuda-backend/pilot/)。

1. **Pilot**（[`cuda_pilot.cu`](cuda-backend/pilot/cuda_pilot.cu)，連結真正的`fpm_core`當參考）：在17組fixture上分別檢查金字塔、粗搜尋畫布與score map、64個level-0 ROI與細搜尋map。第一次執行在11組出現畫布不一致，原因是pilot自己的參考用邊界值0、GPU用模板的`borderColor()`（[記錄](cuda-backend/pilot/pilot1-border-bug.jsonl)）；修正後所有階段都是0差異（[結果](cuda-backend/pilot/pilot2.jsonl)）。延遲（WDDM）：空kernel加同步7.2 µs，pinned上傳＋kernel＋下載＋同步63 µs，同樣的來回用CUDA Graph 59.5 µs。所以設計上要減少來回次數。
2. **第一版後端**（細搜尋是FP64取樣＋逐像素乘加，2輪、無暖機）：每幀時間的幾何平均cuda/cpu是0.184（預設）、0.188（fast）、0.156（subpixel），只開GPU粗搜尋是0.62／0.53／0.83，輸出全部相同。同時發現每個行程只呼叫5次時GPU還沒升頻：Src8五次呼叫是15.3、14.7、14.7、12.2、1.8 ms。所以量測工具改成先記錄冷啟動的第一次呼叫，再不計時地連續呼叫500 ms後才計時。
3. **Nsight Systems**（暖機後）：Src7每次呼叫的GPU時間有95%是細搜尋kernel（6個批次共4.05 ms），粗搜尋0.08 ms，金字塔kernel 0.09 ms；金字塔的NVTX範圍1.9 ms，大部分是pageable記憶體的上傳與拷回。**Nsight Compute**（同一個launch）：FP64是最忙的管線（68.8%），897萬個FP64指令、2.86億個指令，佔用率被shared memory限制在33%。
4. **第二版細搜尋**（Q28＋guard、`__dp4a`、逐列和）：同一個launch從2.64 ms降到0.727 ms，FP64指令降到18萬個，佔用率變成被shared memory限制在16.7%（一個SM只放一個block）。改成優先讓一個SM放兩個block後，Src7 subpixel的細搜尋GPU時間11.9→8.3 ms、Src6 subpixel 33.7→22.3 ms。
5. **第二版的探索量測**（尚未調整block大小）：17組預設模式合計735.6→59.9 ms（幾何平均0.142），subpixel 4877→151 ms（0.100），輸出全部相同；Src9仍比CPU慢（0.61 vs 0.52 ms）。同一次match重用模板的效果很小（Src7 subpixel的match 12.8→12.6 ms）。
6. **guard命中率**：Src7預設模式的1974萬個ROI像素中12,819個走FP64（0.065%），subpixel是1.58億個中的124,062個（0.078%），都沒有整個ROI改走FP64的請求。

## 驗證

計時前先登記[protocol](cuda-backend/protocol.json)（`078a109`）。PC與Mac都從`git archive`解出的原始碼重新建置；輸入是17組fixture的PGM，PC上的副本先和manifest（SHA-256 `3a539fec…`）比對過。

| 檢查 | 結果 |
|---|---|
| PC ctest（`9623880`，`FPM_WITH_CUDA=ON`） | 35/35；`experimental_cuda_backend`與`experimental_cuda_kernels`都有執行，Metal相關6項跳過 |
| `cuda_backend_test`：14個合成場景×（handle API、frame批次64與3、不融合粗搜尋＋CPU金字塔），加dense map、兩個模型的frame、參數驗證 | 全部逐位元相同（93個CPU比對結果） |
| `cuda_kernel_test`：細搜尋map、粗搜尋峰值、dense map、金字塔直接和`fpm_core`比 | 通過；在2047萬個像素中找到24個「Q28與binary64捨入不同」的姿勢，每個不同的像素都在guard內，GPU結果仍和CPU相同；55,852個像素與16張map走了FP64後備 |
| 身分比對：正式計時（cpu、cuda、cuda_coarse，8輪×17組×3模式）與對照（cuda_unfused、cuda_batch8、cuda_cpu_pyramid，2輪） | 1632次執行，CUDA的1122次全部和同一輪的CPU相同，0失敗 |
| 核心不變：`check_backend_identity.py`，`c67ca33`對`9623880`的`fpm_c`，17組×4模式×SIMD／scalar | PC 272次、Mac 272次，全部相同 |
| Mac ctest：`c67ca33`與`9623880`純核心；`9623880`研究建置（GUI、OpenCV、實驗，無CUDA） | 純核心各3/3；研究建置90/91，`gui_workflow`因archive裡沒有`Test Images/`失敗，補上`9623880`的`Test Images/`後重跑通過（見下） |
| H1：三個模式的幾何平均都低於0.5 | 成立（0.142／0.187／0.097） |
| H2：每個模式至少15/17組變快 | 成立（都是16/17，例外是事前預測的Src9） |
| 編譯突變（24個，`cuda_mutants.py`） | **未通過**：19個被抓到，5個存活 |

### 突變

`9623880`上存活的5個：

- **W1**（`sx`重新結合成`ax+(c*x−s*y)`）、**W2**（`sy`合併成FMA）：座標只差1 ulp，只會改變剛好落在捨入平手點的像素；測試沒有這種像素。
- **C1**（只保留分數「大於」門檻的項目）：只有分數剛好等於門檻時不同；測試沒有這種情況。
- **N2**（飽和區間1.125改成1.25）：等價。精確整數統計下|分子|≤分母（Cauchy–Schwarz），binary64的誤差遠小於12.5%，[1.125, 1.25)這個區間到不了；CPU核心有同一個分支。
- **F2**（忽略`repairAll`）：在目前的尺寸限制下等價。`repairAll`只會在座標超出int範圍時出現（誤差上限≥0.25需要510×(1+寬高和)/2²⁹≥0.25，也就是ROI寬高和超過約26萬像素；模板上限讓它最多約13萬），這時正確的ROI全是邊界0；突變版用全0的描述子取樣，得到常數ROI。兩者的NCC都是0。

`1714c97`補強了測試：新增細搜尋NCC mode 1（每個像素都用FP64取樣，預設不變），在結構化的姿勢（特殊角度、半整數與1/3、1/7格點的anchor）中找出「重新結合或合併成FMA就會改變像素」的24個姿勢（只看了83,327個像素）；粗搜尋加上均勻模板配門檻1、平坦來源配門檻0。先在補強後的樹上只跑W1、W2、C1、N2、F2：W1、W2、C1被抓到，N2、F2仍存活。

接著在`1714c97`重跑全部24個：21個被抓到，但這次**N1**（拿掉變異數門檻的0.5上限）存活了。原因是新加的平坦案例和後面的隨機粗搜尋試驗共用同一個亂數產生器，改變了所有隨機試驗；原本抓到N1的那組（coarse mode 1 trial 29）消失了。`7f5737b`讓平坦案例用自己的產生器，隨機試驗恢復原樣（N1在單獨的紅燈檢查裡又被trial 29抓到），另外在dense map測試加了一張亮而近乎平坦的影像，它的視窗落在有上限與沒上限的門檻之間；這張影像是否能單獨抓到N1沒有分開驗證。

最後在`7f5737b`重跑全部24個：**22個被抓到，存活的只剩N2與F2**，也就是上面論證為等價的兩個。登記的gate以`9623880`的19/24為準，仍算未通過；22/24是事後補強測試後的結果。各次結果在[formal/](cuda-backend/formal/)。

### 失敗與修正

- **pilot參考錯誤**：第一次pilot的CPU參考畫布用邊界值0，GPU用`borderColor()`，11組不一致；是pilot自己的錯，修正後0差異。
- **沒有暖機的量測**：第一版的探索量測每個行程只呼叫5次，GPU還在低時脈（Src8從15 ms降到1.8 ms）。之後的工具都先暖機500 ms，冷啟動另外記錄。
- **PC第1次正式嘗試**：`formal.bat`在同一行裡先展開了`%PATH%`才執行vcvars，找不到cmake，CUDA沒有建置；step 6、7因此失敗，沒有產生計時資料（step 5的核心比對有完成）。改用獨立的`build-cuda.bat`後第2次全部通過。第1次的log保留在validation-logs的`attempt1/`。
- **Mac第1、2次正式嘗試**：第1次的archive少了`macos-cli/`，研究建置configure失敗；第2次少了`Test Images/`，`gui_workflow`失敗（90/91）。從`9623880`匯出`Test Images/`補進同一個原始碼樹後，`gui_workflow`單獨重跑通過。兩次的log都保留。
- **計時期間的SSH**：等待step 6結束的輪詢在計時step剛開始的2分鐘內讀過一次log（第0輪，暖機輪，不算在摘要裡）；之後到計時結束都沒有連線。
- **`gpu_bytes`**：正式計時時`gpu_bytes`沒有算進細搜尋的模板副本（最多0.79 MiB，含在`fine.retained_gpu_bytes`裡）；`1714c97`已改正，正式數字照當時量到的報告。
- **突變gate**：見上。
- **正式建置的Nsight Compute**：`--launch-skip 6`超過了3次呼叫的行程裡`coarseFused`的launch次數，沒有抓到任何kernel（[記錄](cuda-backend/profile/ncu-coarseFused-nomatch.txt)）；`fineStripes`抓到的是一個很小的層（18 µs）。kernel分析改用選擇設計階段的兩份報告（[第一版](cuda-backend/pilot/ncu-fineFused-v1-src7.txt)、[第二版](cuda-backend/pilot/ncu-fineStripes-v2-src7.txt)）。

## 界線

- 只在一張GPU（RTX PRO 4000 Blackwell，sm_120）、一個驅動（596.71，WDDM）與Windows上執行過；沒有跑過Linux、TCC模式、其他架構或Jetson。
- 逐位元相同的前提是主機CPU路徑不合併乘加（x86-64沒有FMA的編譯）。Jetson等aarch64主機若用clang預設的合併，CPU路徑的`warpPixel`會用FMA；這時GPU要改用`__fma_rn`對應，目前沒有做，測試會在那種平台上失敗。
- 只接在研究coarse／frame API；穩定的`fpm_match()`、GUI、研究pipeline的`Session`（多尺度、pose refinement）與第37–44輪的完整Metal GPU管線都沒有CUDA版本。
- 量測的是17組固定fixture；沒有量多物件（128／512個）或8K場景，也沒有和Metal比較（不同硬體只並列，不算倍率）。
- prepare的來源複製與PCIe傳輸沒有最佳化（沒有pinned的level 0、沒有和運算重疊），Tensor core、CUDA Graphs也還沒用。
- pilot、探索量測與Nsight數字都不是正式計時。

## 重現

```bash
# Windows，x64 Native Tools環境，CUDA 13.2的bin在PATH裡
cmake -S cpp-core -B build/cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFPM_BUILD_EXPERIMENTS=ON -DFPM_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/cuda
ctest --test-dir build/cuda --output-on-failure
# 每幀時間與逐位元比對（start /affinity C03C02 固定在P核）
python cpp-core/tools/benchmark_frame_backend.py --bench build/cuda/experiments/fpm_frame_bench.exe \
  --pgm-dir fixtures-pgm --arm cpu=0,0,0,0 --arm cuda=2,1,256,0 --rounds 8 --repeat 5 --warmup-ms 500 --output timing.json
python cpp-core/tools/frame_backend_summary.py timing.json --reference cpu --arm cuda --output summary.json
# 突變
python cpp-core/tools/cuda_mutants.py build/cuda . mutants.json
```

PC的完整流程是`formal.bat`＋`build-cuda.bat`，Mac是`formal-mac.sh`，都和所有log放在[validation-logs.tar.gz](cuda-backend/validation-logs.tar.gz)；原始記錄在[raw-results.tar.gz](cuda-backend/raw-results.tar.gz)，摘要在[validation.json](cuda-backend/validation.json)，失敗與決定在[decision-and-failures.json](cuda-backend/decision-and-failures.json)，SHA-256清單在[archive.json](cuda-backend/archive.json)。
