# 第30輪：合併細搜尋旋轉與NCC提交，追查512物件記憶體

2026-10-16，基準`66d362e`，Apple M4 Pro、48 GiB。此次不改Metal kernel的數學：使用同一command buffer依序編碼warp與NCC，完成後CPU修正精確整數統計，再沿用原binary64正規化。所有選項預設保持關閉。

## 整數推導與執行流程

舊路徑是GPU旋轉→等待→CPU修補像素→GPU NCC→等待→CPU正規化。新路徑是GPU旋轉＋NCC→一次等待→CPU修正受影響統計→原正規化。這是合併提交，沒有將兩個shader合為一個kernel，也沒有讓CPU與GPU非同步重疊。

對被修補像素`a→b`，在每個涵蓋該像素的窗口，以相對模板座標`u`更新：

```text
S' = S + (b-a)
Q' = Q + (b-a)(b+a)
P' = P + (b-a)T(u)
NCC = 原有normalize(S',Q',P')
```

每次更新都對應一張合法gray8圖；正確的before值取自尚未改寫的ROI，每個guard像素處理一次。中間差值與加法使用signed64，避免unsigned負差環繞；原面積限制保證`65025×area≤2^53`，可無損轉binary64。模板與來源的插值、零補邊界、四捨五入、模板點積、選峰與最後驗分均未近似。

Metal共用buffer使用tracked hazards，不同compute encoder依序寫ROI再讀ROI／寫statistics；CPU在整個command完成後才讀寫。source／template生命週期、frame失效及原角度重用契約不變。

## 保守退路與實作界線

- 任一request的Q28 plan要求`repairAll`，整個batch使用原兩次提交，計`fallback_batches`。
- guard只代表「可能有誤差」，實際舊新byte相同不做統計更新。若某map實際變動像素超過ROI的1/8，停止增量更新，全部byte修好後CPU完整重算49個窗口，計`dense_maps`；這個罕見退路不增加GPU提交，但可能較慢。
- 模式0／1／2／3皆支援。模式1的row工作區仍供GPU暫定統計用，CPU後修後不再讀取那些row統計。
- 新helper無heap配置，host每批最多1024個uint32計數（4KiB stack），只初始化實際batch範圍；關閉時不構造helper或初始化表。未新增常駐GPU配置。

新增`Options.fineFused`、C `fpm_frame_set_fine_fused`、Python `set_fine_fused`與native JSON `fine_fused`；另加80byte的`FpmFineFusionMetricsV1`，舊C struct不改尺寸。CPU只接受false，稀疏固定尺度路徑拒絕此選項；GUI預設及選單本輪不變。

## 計數與時間定義

融合開啟時，`statistics_dispatches = fused_batches + fallback_batches`、`command_buffers = fused_batches + 2×fallback_batches`。物理map、ROI、修補造訪次數（含guard與repairAll整張圖）與同設定舊路徑相同；只少提交／等待。`changed_pixels`只計融合批次確實不同的byte，不含legacy退路；`corrected_windows`計實際執行的增量更新，包含密集map在切換完整重算前已做的更新，不能稱為最終只由增量修好的不同窗口數。

`correction_ms`包含guard掃描、legacy pixel重採樣、增量統計更新及密集完整重算，並非只有差值乘加。`fused_device_ms`是整個旋轉＋NCC command的GPU timestamp，不是NCC單獨時間；融合選項開啟時舊warp／stats獨立GPU時間一律回-1，避免混淆。`fused_wait_ms`含提交、排隊及執行，不能當成CPU/GPU傳輸時間。

融合的`fine_ncc_ms`包含NCC encoder建構、融合command等待及正規化；warp encoder仍記於rotation；`fine_rotation_ms`扣掉同段時間，保留plan、descriptor、warp encode與CPU修補。這兩欄跨新舊模式不具有同一階段定義，性能比較使用完整總時間、總wait、融合GPU時間與修補時間。啟用profiling時總wait應等`fused_wait_ms+warp_wait_ms+ncc_wait_ms`。各GPU／host欄位有包含關係，不相加推估總時間。

## 正式量測設計

沿用九種設定、六張既有場景；凍結第29輪、同建置關閉融合、開啟融合三路都開啟精確角度重用及fine mode2。兩block反轉順序，串行新程序，每程序第一次另列、至少暖機1秒後三次正式呼叫，共54程序／162次。沒有把上輪不同次數的數字與本輪直接相除。總時間為原生算法的total_ms，包含來源prepare，排除讀圖、一次建模與結果序列化；程序RSS峰值則會包含這些外圍階段，因此兩種指標的涵蓋範圍不同。

六個既有場景中的666個GT物件姿態，新增完整解析度設定重複其中646個，合計1312個「設定×姿態」，不是1312獨立新樣本。GT只在父程序評估；worker只有像素／搜尋參數。所有有序proposals／matches需逐值一致，位置2px、角度1°、尺度±2%一對一匹配要求零FP／FN。時間容忍max(10%,2ms)，RSS容忍max(10%,16MiB)，失敗完整保留，不以平均值掩蓋。

## 結果與記憶體調查

正式54程序／162次完成，全部有序提案與最終匹配逐值一致，既有GT門檻零FP／FN。各欄為六次暖機後呼叫中位數；RSS為兩個新程序峰值中位數。

| 設定 | 第29輪凍結 ms | 同建置關閉融合 ms | 融合 ms | 時間減少 | 關閉→融合 RSS MiB |
|---|---:|---:|---:|---:|---:|
| 6物件快速 | 63.72 | 63.16 | 58.71 | 7.04% | 49.23→49.44 |
| 12物件快速 | 115.99 | 115.27 | 108.23 | 6.11% | 78.99→79.67 |
| 8K、6物件 | 252.59 | 250.38 | 254.78 | -1.76% | 562.59→562.41 |
| 小圖完整次像素 | 6.81 | 8.12 | 5.70 | 29.87% | 27.20→27.18 |
| 128物件快速 | 909.16 | 914.20 | 834.33 | 8.74% | 407.41→404.90 |
| 512物件快速 | 2601.70 | 2616.86 | 2388.29 | 8.73% | 788.99→791.41 |
| 6物件完整次像素 | 215.97 | 217.45 | 193.46 | 11.03% | 54.24→54.68 |
| 128物件完整次像素 | 4332.45 | 4352.49 | 4009.85 | 7.87% | 416.25→417.70 |
| 512物件完整次像素 | 15663.39 | 15827.85 | 14718.34 | 7.01% | 786.66→793.66 |

全部性能容忍通過：True。失敗配對數：0。每個block數值與分佈均保留於摘要；即使門檻通過，少數場景仍可能略慢，不能稱全面加速。

小圖29.87%是相對同建置8.12ms，對凍結版6.81ms則為16.34%。同建置相對凍結第二block慢22.48%，因差值仍在2ms絕對容忍內而通過；所有門檻通過不代表關閉新功能完全零退步。8K總時間慢1.76%，兩block方向不同，沒有可靠總加速。

### 工作量與等待

| 設定 | fine提交：關閉→融合 | fine總wait ms：關閉→融合 | 融合GPU ms | CPU修補 ms：關閉→融合 | 實際修補byte |
|---|---:|---:|---:|---:|---:|
| many128_pose | 1,192→596 | 449.35→374.47 | 278.30 | 17.12→20.64 | 68 |
| many512_pose | 3,282→1,641 | 1399.14→1189.80 | 923.58 | 63.46→72.73 | 298 |
| many128_pose_fullres_subpixel | 5,104→2,552 | 3560.73→3225.57 | 2850.50 | 183.02→202.10 | 2,045 |
| many512_pose_fullres_subpixel | 15,672→7,836 | 13120.99→12066.50 | 10902.17 | 942.76→979.29 | 8,079 |

此處combined GPU與舊單獨NCC時間不同，不直接相除。GPU計算仍存在，主要省的是一次command提交及等待；沒有證明傳輸是主要瓶頸。正式九設定的dense／repairAll退路計數皆零，邊界測試則明確強制兩種退路。

### 512物件RSS增加的來源

本輪融合相對同建置，512快速RSS峰值中位數增加2.42MiB、完整次像素增加7.00MiB，已報告的host金字塔、粗搜尋與細搜尋buffer容量完全相同。這與上一輪「角度重用開／關」約88.20MiB的差異是兩個比較，不能混為一談。

另外凍結第29輪演算法，做四個串行程序（off／reuse／reuse／off），每個首次＋一次repeat，逐階段記錄RSS、physical footprint、compressed、malloc live／reserved與已知buffer。所有有序結果、工作計數與容量均通過exact gate。RSS峰值off為686.77／686.70MiB，reuse為787.14／786.08MiB，重新觀察到約99–100MiB差距；physical footprint也同方向，compressed全程為0。差距在pose建立前已形成，malloc live此時只差約0.14MiB，26,728byte的角度快取無法解釋這個量級。

最後兩個分類程序記錄了讀圖暫存的實際位址，並在兩個階段暫停取得完整vmmap。8192×4096來源經`QImage::Format_RGB888`產生100,663,296byte（96MiB）暫存，即使輸入是灰階PNG也經過這一步。讀圖完成後，兩路的同一96MiB位址區間均標為`MALLOC_LARGE (empty)`，仍有96MiB resident／dirty：暫存已釋放，但頁面尚未離開程序記憶體。

| 分類診斷 | 原RGB暫存區仍映射大小：讀圖後→pose前 | pose前RSS MiB | 程序RSS峰值 MiB |
|---|---:|---:|---:|
| 角度重用關閉 | 96→2.047MiB | 296.91 | 686.63 |
| 角度重用開啟 | 96→96MiB | 393.02 | 786.89 |

關閉路徑的原區間其餘部分已不在完整vmmap；剩餘區域可能被重用，不能假定仍是原allocation。開啟路徑則整個原96MiB區域仍為空閒但駐留。這直接解釋了**新診斷中大部分記憶體差距：已釋放的讀圖RGB暫存頁，與後續pose大配置重疊**。兩路pose首次建立都增加約384MiB（float轉換暫存、CPU留存影像、Metal影像），不是重用路徑多建一份pose。不能將重疊的malloc、RSS、footprint與GPU容量直接相加。

兩個讀圖後回收控制程序均呼叫`malloc_zone_pressure_relief`，回傳0且RSS／footprint未下降；末端回收也無效，因此沒有將它當成修補。診斷包含取樣、hook及vmmap暫停，會改變執行與回收時機，時間不能拿來當效能結果。共八個有效大型診斷程序／16次算法呼叫；首次旗標不一致而被exact gate拒絕的嘗試另存，不計入。

原本88.20MiB那批程序沒有即時VM分類，無法追溯證明每一byte都出於相同機制；原失敗紀錄保留。本結論限於native benchmark讀圖流程，不推廣到所有GUI／相機路徑，也不是長時間無洩漏證明。下一個獨立改善可針對已驗證的灰階輸入，避免RGB888中間圖，先確保來源每個像素完全相同，再重新測量。本輪未改讀圖流程，避免把提交融合與另一項記憶體改善混在一起。

## 功能驗證、來源與重現

Release 49/49、追加focused 5/5、相關ASan／UBSan 8/8、Metal API／GPU Shader Validation通過。融合四種NCC模式均對CPU參考逐窗口相同，測到32／64位邊界、薄模板、零補邊界、半像素、dense但zero-change、forced sparse delta、forced dense重算、混合repairAll batch、模式切換與同地址換來源。C++原生Session交叉profiling／reuse／fusion並驗全結果；Python新binding2項、融合harness26項、修正後memory tool11項通過。ASan使用RelWithDebInfo、detect_leaks=0，不將此寫成Metal driver洩漏檢查。

原始Metal shader逐byte未變。新增[可移植整數修正](../experiments/fine_statistics_repair.hpp)、[Metal提交排程](../experiments/coarse_metal.mm)、[C ABI](../experiments/coarse_ncc.h)、[C++ Session](../experiments/research_pipeline.hpp)、[Python binding](../tools/coarse_experiment.py)、[正式比較工具](../tools/benchmark_fine_fused.py)、[獨立記憶體診斷](../tools/profile_fine_memory.py)均保留完整來源。

一名agent實作helper與CPU測試並獨立複查host；另一名實作54-worker量測與負控制；第三名實作並追查記憶體診斷。主agent完成host/API/bindings/整合、執行驗證、數據整合。獨立複查另重讀54份raw、驗hash及重算表格，648個已報告容量比較皆相同。

正式量測後，分析工具修正兩類問題，未改受測算法或原生binary：GPU時間缺值-1不再跨模型錯加成-3；記憶體診斷重建分別沿用runner與research target各自編譯flags，保留`-ffp-contract=off`。原工具、原摘要、失敗診斷及修正diff/hash均保留；正式速度、RSS、精度與容忍判定完全不變。不要以末端容忍通過掩蓋先前診斷失敗。

另修正未來重跑的協定標籤：凍結baseline commit固定為已驗hash的`66d362e`，目前checkout commit另記，避免本輪提交後把舊binary誤標為新commit。既有正式協定不改寫。

封存原始碼已在另一個全新build目錄編譯；整數修補與融合Metal兩項測試通過，小圖首次及repeat的有序結果與三項frame容量對正式資料逐值相同。從搬移的封存資料重算摘要逐byte相同，226份受測來源hash均核對；工具後修版本與原量測版本分開保留。

[事前協定](fine-fused/protocol.json)、[正式摘要](fine-fused/summary.json)、[驗證紀錄](fine-fused/validation.json)、[記憶體歸因](fine-fused/memory-analysis.json)、[全部來源與原始證據](fine-fused/raw-results.tar.gz)。封存包含原／新source、binary、輸入、54份worker、失敗嘗試、8份有效記憶體診斷及完整位址分類；`content-sha256.json`逐檔保護，驗證紀錄另列壓縮檔hash。

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake -S cpp-core -B build/fine-fused \
  -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_BUILD_TESTS=ON \
  -DFPM_BUILD_GUI=ON -DFPM_WITH_OPENCV=ON
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build build/fine-fused -j 6
ctest --test-dir build/fine-fused --output-on-failure
FPM_FINE_FUSED_BUILD=build/fine-fused python-port/venv/bin/python cpp-core/tests/test_fine_fused_binding.py
python-port/venv/bin/python cpp-core/tools/benchmark_fine_fused.py --output build/fine-fused-new-run
```

保留`build/fine-search`第29輪凍結binary，不以新源碼覆蓋基準。使用時設定`fineNccMode=2`與`fineFused=true`；需完整解析度細搜尋subpixel時可加`fineAngleReuse=true`。profiling保持false。正式比較皆開角度重用，未新增全面no-reuse速度比較；無重用的功能精度由交叉測試覆蓋。

仍屬CPU＋Metal架構：GPU算旋轉和整數NCC，CPU保留稀疏誤差修補、binary64正規化、選峰與搜尋控制。本輪沒有新RGB全圖、CUDA、Windows、真實相機泛化或長時間穩定性證據；不是底層已達極限的證明。
