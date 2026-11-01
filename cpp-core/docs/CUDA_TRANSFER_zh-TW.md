# CUDA傳輸與同步：transfer mode

2026-11-01，實驗室PTRI-PRO-4000（NVIDIA RTX PRO 4000 Blackwell，compute capability 12.0，24 GB，PCIe 5.0 x16；驅動596.71，WDDM，沒有接顯示器；Core Ultra 9 285K；Windows 11），CUDA 13.2 Update 2、MSVC 19.44，Release。這是[CUDA後端](CUDA_BACKEND_zh-TW.md)的第二步，不是研究輪次。新增的行為都在`fpm_frame_set_transfer_mode`後面，預設是mode 0（上一步的行為）；核心（`src/`、`include/`）一行都沒改，GUI與所有預設不變。

**結論：開啟transfer mode 3（lean prepare＋lean calls）後，17張影像每幀時間（prepare＋match）的幾何平均是同一個建置mode 0的0.422（預設）、0.404（FastMode）、0.478（subpixel），合計59.4→24.9 ms、53.7→20.5 ms、134.3→79.6 ms；三個模式17/17組都變快。** 對CPU路徑的幾何平均因此從0.143／0.186／0.097變成0.060／0.075／0.046，上一步唯一比CPU慢的Src9現在每個模式都比CPU快（預設0.52→0.31 ms）。正式計時與對照共2652次執行，其中2142次CUDA執行全部和同一輪的CPU結果逐位元相同，0失敗。預先登記的三個假設都成立。

後續：[kernel mode](CUDA_KERNELS_zh-TW.md)試了本文留下的兩個做法，先上傳細搜尋請求、多執行緒複製來源，兩者都比mode 3慢；改粗搜尋與細搜尋的取樣之後，每幀時間再降到mode 3的約0.77倍，輸出不變。

## 問題

上一步的Nsight剖析指出，CUDA後端剩下的時間大多不在kernel：

- **大圖：** prepare占大部分。Src7（4024×3036）每幀6.37 ms裡prepare 4.62 ms，其中來源複製2.74 ms、金字塔1.55 ms。
- **小圖：** 每次呼叫的同步與API次數。Src9每幀5次同步、11次memcpy。

## 結果

8輪（第0輪是暖機，不算），每輪每組每個模式各開一個新行程；行程內先記錄冷啟動的第一次呼叫，再不計時地連續呼叫500 ms，之後取5次呼叫的中位數。數字是7輪的中位數，計時期間driver固定在P核。m1、m2是兩項改動各自單獨開啟的對照，不在假設裡。

| 模式 | CPU | m0（mode 0） | m1（lean prepare） | m2（lean calls） | m3（兩者） | m3/m0 幾何平均 | m3 變快 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 預設 | 751.7 ms | 59.4 ms | 30.4 ms | 52.8 ms | 24.9 ms | **0.422** | 17/17 |
| FastMode | 382.9 ms | 53.7 ms | 24.9 ms | 48.4 ms | 20.5 ms | **0.404** | 17/17 |
| subpixel | 4998.2 ms | 134.3 ms | 104.4 ms | 106.7 ms | 79.6 ms | **0.478** | 17/17 |

單獨開啟時，m1對m0的幾何平均是0.544／0.507／0.722，m2是0.847／0.870／0.729。lean prepare主要省在大圖，lean calls主要省在小圖與呼叫次數多的subpixel；兩者合起來比相乘的估計還好一點（預設0.422 vs 0.461）。m2在Src4 FastMode沒有變快（0.89 vs 0.89 ms），這組只有一次粗搜尋同步，細搜尋不跑。

冷啟動（行程的第一次呼叫）：m3對m0的幾何平均是0.840／0.872／0.772，預設模式合計101.6→86.4 ms。第一次lean prepare要配置並page-lock主機的level 0（cudaHostRegister在12 MB上約0.07 ms），之後的prepare才重用。

預設模式各組（ms）：

| 影像 | CPU | m0 | 其中prepare＋match | m3 | 其中prepare＋match | m3/m0 | m3/CPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Src1 | 100.50 | 5.44 | 1.97＋3.49 | 3.35 | 0.31＋3.04 | 0.616 | 0.033 |
| Src2 | 48.70 | 3.51 | 2.00＋1.50 | 1.45 | 0.31＋1.14 | 0.413 | 0.030 |
| Src3 | 5.35 | 1.37 | 0.79＋0.57 | 0.50 | 0.10＋0.40 | 0.368 | 0.094 |
| Src4 | 18.42 | 1.44 | 0.14＋1.31 | 1.15 | 0.01＋1.14 | 0.796 | 0.062 |
| Src6 | 309.37 | 8.21 | 4.65＋3.57 | 3.89 | 0.71＋3.18 | 0.474 | 0.013 |
| Src7 | 93.57 | 6.37 | 4.62＋1.74 | 2.21 | 0.70＋1.51 | 0.347 | 0.024 |
| Src8 | 16.04 | 1.09 | 0.19＋0.91 | 0.66 | 0.03＋0.64 | 0.606 | 0.041 |
| Src9 | 0.52 | 0.63 | 0.19＋0.44 | **0.31** | 0.03＋0.28 | 0.494 | **0.594** |
| Src10 | 25.28 | 6.08 | 4.90＋1.16 | 1.96 | 0.75＋1.20 | 0.322 | 0.077 |
| Src5（8組） | 16.0–17.4 | 3.12–3.17 | 約1.98＋1.19 | 1.14–1.19 | 約0.30＋0.87 | 0.365–0.377 | 0.068–0.072 |

lean prepare不等金字塔建完就回傳，所以只能比較prepare＋match，不能單看prepare：m3的match比m0的match多了原本在prepare裡等待的GPU時間（例如Src10的match 1.16→1.20 ms）。

Src9（640×480）在其他兩個模式：FastMode CPU 0.30、m0 0.51、m3 0.26 ms；subpixel CPU 0.74、m0 0.89、m3 0.40 ms。FastMode的差距最小（m3/CPU 0.873），是事前登記時預測最接近的一組。

記憶體：行程峰值RSS m0與m3相同（126–171 MiB，差異不到1 MiB）；保留的device bytes也幾乎一樣（m3少了細搜尋請求的device buffer，多了兩個計數器buffer）。主機的level 0本來就是frame的副本，lean prepare只是把它留著重用並page-lock，沒有多配置主機記憶體。

## 時間去了哪裡（Nsight Systems，不是正式計時）

正式計時之後，用同一個建置以`--trace=cuda,nvtx,wddm`剖析m0與m3（每幀平均）：

| 影像 | 模式 | cudaStreamSynchronize | cudaMemcpyAsync | timing event | memset | D2H | kernel合計 |
|---|---|---:|---:|---:|---:|---:|---:|
| Src7 預設 | m0 | 8 | 20 | 16 | 7 | 6次、210 µs | 909 µs |
| Src7 預設 | m3 | 0 | 11 | 0 | 0 | 0 | 998 µs |
| Src5-0 預設 | m0 | 6 | 14 | 12 | 5 | 4次、46 µs | 590 µs |
| Src5-0 預設 | m3 | 0 | 8 | 0 | 0 | 0 | 622 µs |
| Src9 預設 | m0 | 5 | 11 | 10 | 4 | 3次、4 µs | 116 µs |
| Src9 預設 | m3 | 0 | 5 | 0 | 0 | 0 | 127 µs |
| Src6 subpixel | m0 | 29 | 41 | 58 | 28 | 6次、217 µs | 20.93 ms |
| Src6 subpixel | m3 | 0 | 11 | 0 | 0 | 0 | 22.36 ms |

m3每幀另有1次不計時的event與1次`cudaEventSynchronize`（下一次prepare等上一次的上傳），不算在timing event裡。

- **重疊有發生：** Src7的m3短trace裡，4段各3 MB的上傳在排入後約27 µs開始、約80 µs做完，CPU這時在複製下一段（約210 µs一段）。最後一段排入後，pyrDown kernel在GPU上跑，主機已經在算粗搜尋的工作。m0則是一次12 MB的pageable上傳讓API呼叫卡住669 µs，之後6次複製回主機又卡住約610 µs。
- **上傳變快：** H2D的GPU時間Src7從每幀496 µs變成328 µs（從page-lock的緩衝區上傳，細搜尋請求也不再上傳），D2H完全不見。
- **代價：kernel變慢。** fineStripes在Src7每幀709→794 µs、Src6 subpixel 20.65→22.05 ms（＋6.8%），fineNormalize 91→126 µs。比較可能的原因是每個block開始時從mapped主機記憶體讀請求（經過PCIe），以及fineNormalize每個thread的system fence；這次的arm沒有把這兩件事和其他改動分開，所以沒有證實。subpixel的Src6因此只快了17%（m3/m0 0.832），是大圖裡改善最少的一組。把請求改回先上傳到device，可能在細搜尋多的情況更好，留待之後量測。
- **GPU仍有空檔：** 同一個trace裡，pyrDown在最後一段上傳完成後約80 µs才開始；粗搜尋的命令在pyrDown還沒跑完時就已排入並送出，GPU卻在pyrDown做完後約100 µs才開始執行它們。這兩段空檔不是主機在算，比較像WDDM送出命令的延遲。

## 設計

### lean prepare（bit 1）

只在frame的device金字塔（`pyramid_mode=1`）下生效：

1. **frame保留主機的金字塔：** 尺寸與層數相同時不重新配置，免掉每幀的page fault。尺寸或層數不同時才重建，並先解除舊的page-lock。
2. **level 0 page-lock：** 重建時對level 0呼叫`cudaHostRegister`；失敗（例如和別人已register的範圍共用一頁）就用一般的上傳，結果一樣，只是比較慢。
3. **分段複製與上傳重疊：** 來源分成至多4段、每段至少2 MiB，每段複製（需要時同時反相）進level 0後立刻排入非同步上傳，下一段的複製和這段的DMA同時進行。
4. **不等金字塔：** pyrDown kernel排進同一個stream後就回傳；之後的粗搜尋排在它後面，順序由stream保證。下一次prepare開始前會等上一次的上傳讀完level 0（event），frame要釋放或換掉主機金字塔前也一樣。
5. **level 1..top不複製回主機：** 細搜尋在GPU上跑時，主機從不讀這些層的像素；它們的主機`Image`只保留尺寸，`residentLevel()`拒絕上傳這些沒寫入的主機層。細搜尋在CPU（`fine_batch_size=0`）時照舊複製回來並等待。

### lean calls（bit 2）

粗搜尋與細搜尋每次呼叫：

- **不記錄計時event、不跑memset：** 粗搜尋的計數改放在獨立buffer，`packEntries`讀完就清零；細搜尋的統計量由`fineNormalize`讀完清零，FP64像素計數也一樣。buffer第一次配置或擴大時清零一次；切換transfer mode或呼叫失敗後也重新清零。
- **細搜尋請求不上傳：** kernel直接從mapped的主機staging buffer讀，每個block只讀一次進shared memory。
- **不呼叫`cudaStreamSynchronize`：** 每次呼叫最後一個kernel的最後一個block在system-scope fence之後，把遞增的序號寫進mapped記憶體；主機先`cudaStreamQuery`把WDDM排著的命令送出，再輪詢這個值，每4096次輪詢查一次stream狀態以回報非同步錯誤。
- **代價：** 沒有event就沒有GPU時間，`device_ms`回報-1；GPU時間改由Nsight量。

### 為什麼輸出不變

所有kernel的算術都沒有改：取樣、整數統計、正規化與峰值選擇和mode 0走同一段程式。改的只有資料什麼時候、從哪裡到GPU，以及主機怎麼知道GPU做完了。正確性因此取決於三件事，各有測試或突變：

- **計數器在每次呼叫開始時是零：** 由上一次呼叫清零，或配置時清零（T1–T3、T10、U3、U4）。
- **主機讀結果時GPU已寫完：** 完成值由最後一個block在fence後寫入（T4、U1）。
- **主機的level 0在DMA讀完前不被改寫：** 下一次prepare先等event（U2）。

## 選擇設計的過程（不是正式計時）

pilot（`pilot/xfer_pilot.cu`，同一台PC）先量傳輸與等待本身：

| 項目 | 中位數 |
|---|---:|
| 新配置12 MB的vector再memcpy | 2.57 ms |
| 同一塊緩衝區重複memcpy（頁面已在） | 0.28 ms |
| cudaHostRegister 12 MB | 0.07 ms |
| 暖緩衝區複製＋pageable上傳（mode 0的做法） | 1.20 ms |
| 複製進已register的緩衝區、分4段、每段複製完就上傳 | 0.47 ms |
| 同上分8、16、32段 | 0.74–0.97 ms |
| 小上傳＋kernel＋`cudaStreamSynchronize`往返 | 72.7 µs |
| 同上，kernel寫mapped旗標、主機先`cudaStreamQuery`再輪詢 | 45.1 µs |
| 細搜尋呼叫的形狀：event、上傳、memset、2個kernel、event、同步 | 113.6 µs |
| 同上去掉event與memset | 72.7 µs |

mode 0的來源複製2.7 ms幾乎都是新配置記憶體的page fault：它每次prepare都釋放再配置frame的金字塔。分段太細反而變慢，所以上限是4段、每段至少2 MiB。`cudaDeviceScheduleSpin`沒有幫助（輪詢往返反而是71 µs），沒有採用。

接著在開發中的樹上跑了兩輪五個arm（pilot1，510次執行，0不一致），定出登記的門檻：m3/m0幾何平均0.42／0.41／0.48，17/17變快，最接近的是Src4 FastMode（0.93）；m3在17組都比CPU快，最接近的是Src9 FastMode。

## 驗證

| 檢查 | 結果 |
|---|---|
| PC ctest（c1068b2，git archive） | 35/35；`experimental_cuda_backend`與`experimental_cuda_kernels`有執行，6個Metal測試跳過 |
| Mac研究版ctest（c1068b2，git archive含`macos-cli/`與`Test Images/`） | 91/91 |
| 輸入 | 27個PGM的SHA256與第一步的manifest相同 |
| 逐位元相同 | 計時2040次＋對照612次，2142次CUDA執行全部和同輪CPU相同，0失敗 |
| 對照arm | m3 batch 8、m3 unfused、m3 CPU金字塔（只有lean calls生效）、m1與m3搭配CPU細搜尋 |
| 核心 | `git diff 31820ad c1068b2 -- cpp-core/src cpp-core/include`是空的 |
| H1：m3/m0幾何平均 < 0.6 | 成立（0.422／0.404／0.478） |
| H2：至少16/17組變快 | 成立（每個模式17/17） |
| H3：m3在17組都比CPU快 | 成立（每個模式17/17；最接近Src9 FastMode 0.873） |

`cuda_backend_test`新增的部分：14個場景各用mode 1、2、3與batch 64、3（mode 1另加batch 0）各prepare兩次；一個frame依序換尺寸（含只換高度）、反相、只prepare不match、換成層數不同的模型、切到mode 0再切回；以及各mode的metrics（lean prepare真的有跑、level 0有page-lock、2400×1800分成3段、lean calls的等待次數與`device_ms`為-1，且FP64像素數與細搜尋次數和mode 0相同）。

### 突變

`cuda_mutants.py`在c1068b2的另一份副本上逐一編譯38個突變、跑會受影響的測試。

| 組別 | 結果 |
|---|---|
| T1–T10（transfer mode，列入門檻） | 10/10被抓到 |
| 上一步的24個 | 22被抓到；N2、F2存活，和上一步相同，上一步已論證兩者在可達輸入上等價 |
| U1–U4（只回報） | U3、U4被抓到；U1（signal前沒有system fence）、U2（prepare不等上一次的上傳）存活 |
| 未突變與還原後的樹 | 都通過 |

U1、U2是記憶體順序與競爭：在這台x86主機上，PCIe寫入照順序到達，而且每次prepare的CPU複製比上一次的DMA慢，所以測試看不到差別。程式碼保留fence與等待，因為正確性不應該依賴這兩個時間條件。

### 失敗與揭露

- **T7在登記前就換掉：** 突變pilot裡原本的T7（`same = !levels.empty()`，不比層數）存活。層數變少時多留一層不影響輸出，是等價的；層數變多時會讀到vector之外（未定義行為），測試未必看得到。登記前把它換成「只比寬度、不比高度」的突變，並在測試序列加一個只改高度的步驟，pilot確認會被抓到。原本的結果保留在`pilot/mutpilot.log`。
- **計時期間的SSH：** 計時開始後約20秒，我讀了一次formal.log、看了一次log目錄（第0輪，暖機，不算）；約第6輪時等待腳本讀了一次formal.log與timing.log的最後幾行。其他時間沒有連線。
- **lean calls沒有GPU時間：** 正式結果裡m2、m3的`device_ms`是-1，GPU時間只看Nsight。
- **m2在Src4 FastMode沒有變快：** 只是對照，不在假設裡，照實記錄。

## 界線

- 一張GPU（RTX PRO 4000 Blackwell）、一個驅動（596.71，WDDM）、只有Windows。WDDM會批次送出命令，`cudaStreamQuery`的送出效果與輪詢的節省在TCC或Linux上可能不同。
- 只有研究coarse／frame API的frame路徑；handle API（`fpm_coarse_match`）不受影響。
- 輪詢會讓一個CPU核在等待時保持忙碌。
- 單執行緒的來源複製（Src7約0.64 ms，19 GB/s）現在是大圖prepare的主要成本；多執行緒複製、讓呼叫端直接給page-locked的來源都還沒做。
- kernel本身（Src1粗搜尋、Src6 subpixel的細搜尋）沒有改動，Tensor core、CUDA Graphs都還沒試。

## 重現

```bash
# Windows，x64 Native Tools環境，CUDA 13.2的bin在PATH裡
cmake -S cpp-core -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
# 每幀時間與逐位元比對（start /affinity C03C02 固定在P核）
python cpp-core/tools/benchmark_frame_backend.py --bench build-cuda/experiments/fpm_frame_bench.exe --pgm-dir PGM \
  --arm cpu=0,0,0,0 --arm m0=2,1,256,0,0 --arm m1=2,1,256,0,1 --arm m2=2,1,256,0,2 --arm m3=2,1,256,0,3 \
  --rounds 8 --repeat 5 --warmup-ms 500 --output timing.json
python cpp-core/tools/frame_backend_summary.py timing.json --reference m0 --arm m1 --arm m2 --arm m3 --h1 0.6 --h2 16 --output summary.json
# 突變
python cpp-core/tools/cuda_mutants.py build-cuda . mutants.json
```

證據在[`cuda-transfer/`](cuda-transfer/)：`protocol.json`（0cbf3f6，計時前登記）、`formal/`（摘要與突變結果）、`pilot/`、`profile/`、`raw-results.tar.gz`、`validation-logs.tar.gz`、`validation.json`、`decision-and-failures.json`、`archive.json`。
