# 第29輪：細搜尋的精確角度圖重用與旋轉同步量測

2026-10-16，基準`b31c143`，Apple M4 Pro、48 GiB、macOS 26.6.2。沿用第28輪`fineNccMode=2`，此次改搜尋排程，不改旋轉插值、Metal NCC kernel、選峰或最終驗分的數學定義。

## 實測結論與採用界線

**完整解析度且開啟細搜尋次像素時，精確重用可明顯加速；既有快速設定沒有重用收益。** 六／128／512完整次像素時間分別縮短31.6%／32.7%／39.1%，全程有序提案與結果逐值相同。但主測有一個關閉新功能對凍結版的速度門檻失敗，以及三個512物件RSS配對門檻失敗，不能寫成所有性能門檻通過。保持手動選項與現有GUI預設。

證據：[事前協定](fine-angle-reuse/protocol.json)、[完整摘要](fine-angle-reuse/summary.json)、[驗證與封存hash](fine-angle-reuse/validation.json)、[完整來源、輸入、原始結果與複查](fine-angle-reuse/raw-results.tar.gz)。

下表時間為兩block正式呼叫中位數（原六設定各10次、新增完整次像素各6次），RSS為兩個程序峰值的中位數。工作量內含source prepare，不含讀圖／一次建模／結果序列化／GT評估。

| 設定 | 凍結版 ms | 同建置關閉 ms | 只開診斷 ms | 重用 ms | 關閉→重用 RSS MiB |
|---|---:|---:|---:|---:|---:|
| 6物件／既有快速設定 | 61.59 | 63.22 | 64.37 | 63.55 | 50.64→50.36 |
| 12物件／既有快速設定 | 118.29 | 117.57 | 119.58 | 117.14 | 80.13→80.53 |
| 8K、6物件／既有設定 | 259.38 | 261.63 | 259.74 | 266.32 | 563.48→565.52 |
| 小圖2物件／原有完整次像素 | 6.38 | 6.89 | 7.30 | 6.44 | 27.42→27.68 |
| 128物件／既有快速設定 | 915.49 | 916.18 | 937.26 | 912.83 | 411.36→410.80 |
| 512物件／既有快速設定 | 2600.96 | 2605.71 | 2687.68 | 2615.06 | 796.69→800.44 |
| 6物件／完整次像素 | 300.82 | 308.87 | 312.20 | 211.35 | 54.20→54.57 |
| 128物件／完整次像素 | 6797.29 | 6892.45 | 7021.28 | 4641.93 | 414.27→417.03 |
| 512物件／完整次像素 | 25086.28 | 25656.56 | 25626.77 | 15624.73 | 702.03→790.23 |

128完整次像素兩block的關閉／重用為7904.84／4337.68與6803.86／4933.63ms；512為25786.67／15626.95與25567.86／15622.51ms。兩種順序皆加速，但128關閉版第一block相對凍結版慢16.30%，已超過事前速度容忍，原始6975.95／7959.71／7904.84ms三次數據保留。

五個既有快速設定皆零命中，128約916.18→912.83ms、512約2605.71→2615.06ms，沒有可靠收益。因此即使完整次像素加速，也不代表它超過原本更快的「stopLayer1＋末端精修」設定；這是不同工作量，不能以25秒→15秒取代現有約2.6秒方案的建議。

所有主測輸出與精度契約通過：仍是既有666個獨立GT姿態，新增設定重複其中646個姿態，合計1312個「設定×GT」而非1312個獨立樣本。位置2px、角度1°、尺度±2%的一對一嚴格門檻下零FP／FN。精確重用不提升同設定的分數或準確度，省下的是相同結果的重算。

## 記憶體失敗與複查

512完整次像素主測，關閉／重用峰值702.03／790.23MiB，相差88.20MiB；相對關閉版兩block與相對凍結版第二block，三個RSS配對超門檻。GPU容量完全相同（fine為14,352,128bytes），cache專用容量最大26,728bytes，不能用這兩個容量去否定實際RSS失敗，也不能直接把88MiB全歸因於cache。

同一凍結執行檔兩block峰值791.03與700.55MiB已有明顯差異。針對512另做四個新程序、current/reuse/reuse/current、每程序首次＋再一次、無暖機的記憶體複查：current為795.84／795.83MiB，reuse為794.30／797.12MiB；兩配對差−1.55／+1.30MiB，記憶體容忍通過，輸出相同。主測差距沒有重現，但這是較短的記憶體複查，不替換主測的暖機速度／RSS失敗。完整vm_stat前後快照保留；尚未定位到特定配置器、壓縮或系統因素，亦未證明額外RSS永遠很小。

另針對128完整次像素關閉新功能的速度失敗，四個新程序以frozen/current/current/frozen、各暖機1秒及3次正式量測複查：凍結／關閉為6744.25／6841.53及6757.51／6852.94ms，兩配對慢1.44%／1.41%，皆在容忍內。主測16.30%的單block退步未重現，但原紀錄仍保留；不宣稱新增診斷掛鉤完全零開銷。額外兩項複查合計8程序，12次暖機後正式呼叫＋8次首次／重複記憶體呼叫，與主72程序／312次分開計。

因此6／128完整次像素可在已測條件下優先試重用；512是有速度收益的研究選項，需依部署機器記憶體預算驗證，不直接升為全域預設。觀察器則保持關閉：128／512快速設定單是observe區間約27.14／63.91ms，程序峰值多約5.94／7.24MiB。

## 省下的工作與同步成本

| 完整次像素 | 128物件：關閉→重用 | 512物件：關閉→重用 |
|---|---:|---:|
| 實際細搜尋圖 | 798,291→536,425 | 2,866,495→1,850,128 |
| 重用圖 | 261,866 | 1,016,367 |
| fine command總數 | 5,972→5,104 | 19,010→15,672 |
| fine NCC GPU ms | 2179.45→1270.60 | 8229.38→4676.53 |
| fine旋轉host ms | 3650.33→2299.39 | 14204.65→8163.45 |
| fine修補host ms | 312.77→198.90 | 1492.83→759.49 |
| fine總wait ms | 5906.91→3761.98 | 22150.49→13120.34 |

每個實際批次仍是warp等待→CPU修補→NCC等待；本輪藉較少實體工作降低總提交／等待，沒有實作兩槽重疊或將兩command融合。

為下一步分析，既有快速128／512的診斷組warp GPU約147.64／504.26ms、warp commit+wait約229.27／738.44ms，guard掃描＋修補14.59／54.42ms，NCC wait211.78／651.00ms。大量時間不在CPU像素修補本身；GPU運算、提交與等待均值得研究，仍不能把wait與GPU時間差全說成傳輸。完整次像素的修補密度約0.047%，但像素絕對量大，修補仍達數百毫秒到一秒以上。

有界觀察器在快速128／512記錄228,667／659,376請求，未追蹤32,059／462,768；完整次像素128／512記錄798,291／2,866,495，觀察到跨呼叫重複124,839／111,302，未追蹤476,844／2,558,585。每模型限制65536，三尺度合計最多196608，不能把此合計誤讀成單模型超限；大型跨呼叫重複率僅為下界。

## 重用的範圍與數學契約

令`M(I_l,T_l,a,θ)`為既有程式生成旋轉ROI、精確整數NCC及binary64正規化後的7×7分數圖。同一候選、同一層、同一次重新定心pass內，`I_l`、`T_l`與anchor `a`固定。角度括峰與步距減半可能讓相鄰兩次評估使用相同`θ`；此時直接搬移前次的`M`，無須再旋轉與累加49個NCC。

使用`memcmp(double)`比較角度位元，**不做角度量化、epsilon合併或360°折疊**。目前角度列表沒有重複元素；按照新列表原順序放回圖，因此同分選峰、3×3×3次像素擬合與後續控制流程不變。只保留上一輪評估，跨候選／層級／pass／模板／呼叫都不共用。每次pass可能移動anchor，即使某些角度重複也必須重算。

這是省去相同問題的重算，不是近似NCC或減少邏輯搜尋範圍。每模型驗證`邏輯請求 = 實際計算圖 + 重用圖`；原`fine_maps`、旋轉像素與提交等計數仍只計實體工作。原portable CPU非批次流程與所有Metal kernel保持原樣。

新增選項`Options.fineAngleReuse`，預設false。C介面為`fpm_frame_set_fine_search_options(frame, profiling, angle_reuse, ...)`，Python為`SourceFrame.set_fine_search_options(...)`，native JSON為`fine_angle_reuse`；原有C struct大小不變，診斷另增`FpmFineSearchMetricsV1`。CPU frame只接受兩個選項都關閉；固定尺度稀疏路徑不接受這些選項。

## 為何不能只用量化旋轉參數當key

Metal的Q28 descriptor可能讓兩個不同binary64幾何量量化成同值，但CPU修補仍使用原binary64 anchor／sin／cos。只看descriptor會將原本應有不同四捨五入結果的請求混在一起。本輪快取由候選狀態與pass生命週期保證來源／模板／anchor相同，再精確比較原角度位元。

另設`fineProfiling`觀察實際送到backend的ROI請求。key包含來源pointer、寬高stride、模板寬高與anchor x/y／angle的double位元；僅表示旋轉ROI相同，**不等於任意不同模板的NCC分數可以共用**。每模型／新frame清空歷史，最多保留65536個key。每呼叫先去重，跨呼叫只數已在歷史中的不同key，兩種重複計數互斥：

```text
requests = within_call_duplicates + cross_call_duplicates
         + tracked_unique_keys + untracked_call_keys
```

超出容量的新key計為未追蹤，不能稱它們全是不同請求；飽和後跨呼叫重複數只是下界。觀察器的排序、hash及配置都有成本，預設關閉，正式比較把它獨立列出，不把診斷模式當部署速度。

大型設定的歷史容量可能飽和：觀察到零跨呼叫重複不能證明所有未追蹤請求都不同。部署的rolling cache沒有這個全域history限制，它按候選生命週期確定key；因此它實際命中數可能大於有界觀察器報出的跨呼叫重複下界，兩者並不矛盾。不同模型與pass不建跨域分數快取，避免為少量額外重用引入大型cache與失效風險。

## 時間與記憶體如何解讀

新增host區間：plan、descriptor寫入與guard清零、warp encode／commit+wait、全圖修補、guard遍歷與被標記像素實際修補、NCC encode／commit+wait；另記warp GPU timestamp。既有fine旋轉、NCC、正規化、總wait仍保留。`warp_wait_ms+ncc_wait_ms`應等於既有`fine_wait_ms`。

GPU timestamp與host wait包含關係不能相加。wait包含提交、排隊與GPU執行，**不是傳輸時間**；Apple共享記憶體仍可能有同步及cache成本，此量測沒有DRAM頻寬、cache miss或硬體stall計數器。`guard_scan_ms`包含真正的CPU像素重採樣，不是純mask掃描。

fine來源上傳counter只涵蓋細搜尋自己上傳的來源。它可能為零，因為讀取的是frame金字塔已準備的Metal buffer；不能據此推論整條pipeline完全沒有CPU寫入或資料搬移。來源prepare仍包含在總時間，descriptor／guard的主機寫入另有計時。

GPU緩衝區容量、程序峰值RSS與CPU快取容量分開報。`cache_peak_bytes`為同時存活的cache專用容器容量：候選的角度列表、暫存前輪ScoreMap結構與尚由它持有的值；排除當前分數圖與配置器metadata，不能當成精確新增RSS。不同模型依序執行，容量應看各模型最大值，不能把和當峰值。觀察器另報key payload、bucket數與單呼叫vector容量，仍非整個hash allocator的實際RSS。

## 正式比較設計

四路：凍結第28輪strip32、此次建置但關閉新功能、只開診斷、只開角度圖重用。其餘皆同一`ncc_mode=2`、fine mode2、粗搜尋overlap關閉。每個worker新程序、串行，兩block反轉方法順序，至少暖機1秒；第一次與建模時間分開。

原六場景各5次正式呼叫；其中五個使用`stop_layer=1, subpixel=0`，不觸發這次重用，必須零命中。小圖原有`stop_layer=0, subpixel=1`為正向案例。再以6／128／512物件同一圖片，新增完整解析度與subpixel設定，各3次正式呼叫，要求重用命中大於零。合計72程序、312次正式呼叫。新增設定不是新增獨立圖像；不把不同設定的速度當等價效能比較，也不將不同輪的倍率相乘。

GT只由父程序評估，worker只能得到來源／模板與搜尋參數。完整有序提案與結果須與同設定凍結版相同；觀察模式工作量／容量亦相同，重用模式則核對邏輯請求、粗搜尋量、減少的細圖與提交。時間容忍`max(10%,2ms)`，RSS容忍`max(10%,16MiB)`；失敗與原始樣本全保留。

## 後續獨立實驗候選

下一個可以推導的方向是：同一command先產生暫定Q28旋轉ROI與NCC統計，一次等待後CPU只校正少數guard像素所影響的統計。某像素從`a`改成`b`，對涵蓋它的每個窗口：

```text
S += b − a
Q += b² − a²
P += (b − a) T(u)
```

整數恆等式可保留原NCC；需要signed64暫存、完整修補／密集guard退路及逐窗位元驗證，最多49個窗口受到單一像素影響。**本輪尚未實作此融合方式**，也沒有消除每批兩次CPU等待。先依本輪量出的修補密度和成本決定是否值得；兩槽pipeline可另外研究，但增加資源與生命週期複雜度。

## 已完成的功能驗證與重現入口

Release 44/44、相關ASan／UBSan 8/8、Metal API／GPU Shader Validation原生整合均通過；新Python binding 2項、舊binding 2項及量測工具11項通過。Scheduler測試包含batch1／7／32、三種模板尺寸、多層金字塔、重複呼叫、角度邊界、平分排序、同地址換圖，以及subpixel關閉／stopLayer1必須零命中。Native Session另驗多尺度、模式切換、觀察與重用組合、冷暖計數及相同輸出。

完整來源：[scheduler](../src/matcher.cpp)、[provider契約](../src/internal.hpp)、[有界觀察器](../experiments/fine_request_profile.hpp)、[Metal主機計時](../experiments/coarse_metal.mm)、[C ABI](../experiments/coarse_ncc.h)、[Python介面](../tools/coarse_experiment.py)、[正式量測工具](../tools/benchmark_fine_search.py)。兩個實作agent分別完成scheduler／測試與觀察器／測試；量測agent後續因工具權限中斷而未交付工具，主agent完成量測工具與整合。另一次獨立agent複查發現原五場景無法觸發重用、量測資料防護與重複次數硬編碼問題，均在正式協定固定前處理。

開發中的失敗也保留：macOS混合auto型別測試編譯失敗；正向整合fixture原門檻讓所有候選在layer0前淘汰，調整fixture後確實產生重用；量測fixture共用Python物件導致測試資料被重複改動，改為逐筆深複製。首次未最佳化ASan Debug測試7項通過，最後一項中止後改用與前輪相同RelWithDebInfo，重新8項全通過；不把中止當通過。正式量測不與這些測試或其他GPU工作並跑。

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake -S cpp-core -B build/fine-search \
  -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_BUILD_TESTS=ON \
  -DFPM_BUILD_GUI=ON -DFPM_WITH_OPENCV=ON
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build build/fine-search -j 6
ctest --test-dir build/fine-search --output-on-failure
FPM_FINE_SEARCH_BUILD=build/fine-search python-port/venv/bin/python cpp-core/tests/test_fine_search_binding.py
python-port/venv/bin/python cpp-core/tests/test_fine_search_harness.py
python-port/venv/bin/python cpp-core/tools/benchmark_fine_search.py --output build/fine-search-replay
```

保留`build/fine-strip`的第28輪凍結執行檔，不以新源碼覆蓋基準。新C++選項為`options.fineNccMode=2; options.fineAngleReuse=true;`；`fineProfiling=true`僅供觀察，日常使用維持false。GUI預設與現有快速設定不變，本輪沒有新增GUI勾選。

本輪沒有新RGB全圖、CUDA、跨硬體或串流穩定性結果；所報加速限於本機既有合成圖片與指定設定。

封存開啟完整測試的初次configure另發現事前hash清單未收錄既有`tests/c_api_test.c`；已補入完整C測試源碼、核對與凍結commit逐byte相同，並以`source-extras.json`記錄補充hash。舊instrumentation patch亦保留。正式受測算法及執行檔未更動；失敗記錄保留。

封存來源已以tests ON重新建置native runner、scheduler與觀察器測試，兩項測試通過；小圖完整搜尋重用36張圖，提案與結果和正式紀錄相同。移動資料目錄後摘要逐byte相同，重複worker、GT流入worker、破壞hash、改動邏輯計數及刪除全部匹配的負控制均被拒絕。
