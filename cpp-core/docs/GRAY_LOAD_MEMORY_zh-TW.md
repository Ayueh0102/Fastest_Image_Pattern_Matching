# 第31輪：避免灰階讀圖的RGB中間圖

2026-10-16，基準`e392a01`，Apple M4 Pro、48 GiB。本輪處理第30輪定位的讀圖暫存，沒有改NCC、旋轉、金字塔、Metal kernel或搜尋控制。

## 改動與數值契約

原native runner將所有解碼結果轉成RGB888，再以`(77R+150G+29B+128)>>8`產生gray8。對8192×4096來源，這會配置96MiB RGB中間圖；上一輪以實際位址證明它已free後仍可能駐留，與後續pose配置重疊。

新路徑只接受解碼後明確的`QImage::Format_Grayscale8`，逐列copy寬度內的byte、跳過padding。因`77+150+29=256`，對全部256種灰階值，原轉換`grayFromRgb(v,v,v)`都等於v。其餘格式，包括彩色、alpha／預乘、Indexed8調色盤、Mono與Grayscale16，完全沿用原RGB888轉換。沒有改用Qt的不同權重灰階公式，也沒有以副檔名或allGray猜測格式。

helper以QImage按值（隱式共享）接收；檔案載入直接傳temporary，使fallback與舊讀法相同，在配置輸出gray前可釋放原decoded圖。借用呼叫者影像時不修改它；輸出fpm::Image自行持有像素。完整header保留在`tools/qt_gray_loader.hpp`，核心仍不依賴Qt。

native JSON `gray_load_mode`預設`direct_gray8`，可用`legacy_rgb`逐次對照。新增`load.decode_ms`為來源＋模板的解碼和轉換時間；`rgb_temporary_bytes`是RGB888邏輯容量（含stride padding），不是RSS，也不代表整個decoder無其他配置。`after_load_memory`在建模前取樣；原`total_ms`仍只涵蓋算法，排除讀圖。

GUI同樣在明確gray8時直接保留decoder影像與stride，其餘格式沿用舊轉換（保留原rvalue overload的原地轉換機會）。灰階快速路徑將DPR設回1、清除color-space標籤，維持原新建gray圖的顯示語義；`@2x`預覽大小另有回歸測試。原有`setAutoTransform(false)`未變，維持儲存座標。GUI流程測試覆蓋載入、旋轉與尺度、換圖及關閉；下方記憶體數字只來自native runner，不宣稱已量測GUI整體RSS。

## 預先固定的比較方式

三臂為上一版凍結binary、同建置legacy_rgb、同建置direct_gray8；全部沿用fusion、角度重用及fine mode2。選三個既有設定：小圖完整次像素、512快速、512完整次像素。兩block反轉順序、串行新程序，每程序首次＋一次repeat、無暖機，共18程序／36次算法呼叫。這是讀圖與程序記憶體的重點實驗，不替代上一輪162次暖機後的速度比較，也不是新相機資料的泛化評估。

所有有序proposals／matches、非時間工作計數及frame／pose容量需exact；GT依原位置2px、角度1°、尺度±2%一對一匹配要求零FP／FN。逐block保留首次／repeat的算法時間與程序RSS門檻，容忍分別max(10%,2ms)、max(10%,16MiB)，失敗不刪。只有current兩臂有讀圖計時，凍結版缺值不得補成0。

三臂18程序／36次呼叫的有序提案、最終匹配、非時間工作計數與frame／pose容量逐值一致；原GT門檻零FP／FN。下面為兩個獨立程序的中位數，並非大量重複的穩態估計。

| 設定 | 同建置舊讀法RSS MiB | 直接灰階RSS MiB | 減少MiB | 舊→新讀圖 ms |
|---|---:|---:|---:|---:|
| 小圖完整次像素 | 26.80 | 26.48 | 0.31 | 2.54→2.26 |
| 512物件快速 | 786.16 | 687.02 | 99.14 | 166.90→132.67 |
| 512物件完整次像素 | 791.99 | 697.28 | 94.71 | 176.09→135.30 |

512快速／完整次像素RSS分別減少約12.6%／12.0%，讀圖時間約減少20.5%／23.2%。兩者來源的邏輯RGB中間容量都是96MiB→0，模板也省去42,840byte；實際RSS下降不必剛好等於這些容量，還受配置器和回收時機影響。這支持並改善上一輪定位的機制，並不追溯重寫原88.20MiB那批沒有VM分類的歷史紀錄。

| 設定 | 凍結版首次／repeat ms | 同建置舊讀法首次／repeat ms | 直接灰階首次／repeat ms |
|---|---:|---:|---:|
| 小圖完整次像素 | 13.88／10.26 | 12.62／9.13 | 12.84／9.94 |
| 512物件快速 | 2684.35／2388.94 | 2553.10／2954.93 | 2551.00／2376.52 |
| 512物件完整次像素 | 15106.52／14959.47 | 15063.33／14935.76 | 15183.46／15148.36 |

**整體性能容忍結果為false，失敗保留。** 唯一超門檻的是512快速設定block1的同建置舊讀法repeat：3536.80ms，相對凍結版約慢48.34%。未定位其原因，也未刪除或事後重跑取代；因此上表舊讀法repeat中位數被抬高，不能將2954.93→2376.52ms稱為NCC優化成果。

新直接灰階路徑相對同建置舊讀法與凍結版的12個配對，其首次、repeat與RSS均在預定容忍內。完整次像素repeat相對同建置舊讀法約慢1.42%（相對凍結版約慢1.26%）；本輪只有兩個程序樣本，不對這個小差異作因果或統計顯著性主張。這項改動的已證實收益是省掉讀圖暫存與降低程序記憶體，沒有更改匹配算法，不能解讀成NCC又快了一成。

所有RSS門檻18/18、首次時間門檻18/18、repeat時間門檻17/18通過。完整block值、來源hash與失敗配對可由原始資料重算。

## 驗證與重現

Release50/50（含GUI workflow）；測試最後補上輸入不變檢查後另重建loader目標並驗證；loader ASan／UBSan通過，detect_leaks=0，不是長時間無洩漏證明。獨立測試以舊RGB888轉換當oracle，涵蓋256灰值、非4倍寬、自訂stride／const外部buffer、輸出所有權、彩色／透明／預乘、調色盤、Mono與16-bit，以及PNG／BMP／JPEG解碼後逐像素一致與錯誤輸入。JPEG比較解碼後的相同像素，不錯誤要求有損檔與輸入原圖相同。

分析harness的23項負控制與記憶體工具17項測試通過。診斷工具維持對舊凍結來源及新runner的支援，新增header也從指定source-root複製並記hash；新舊兩種來源皆完成build-only驗證，未追加GPU診斷。所有量測與工具原碼保留。比較前固定來源、binary與輸入hash，正式結束再次驗hash。GUI新測試首次誤將RGB ICC profile寫入灰階PNG，Qt拒絕該profile而使fixture斷言失敗；修正fixture後DPR回歸及完整GUI流程通過，原失敗log保留，未將其混入性能結果。

另外從封存source全新編譯runner及loader／GUI測試，兩項皆通過；封存小圖首次與repeat的有序結果、所有非時間工作與容量和正式資料相同，搬移後重算摘要逐byte一致。第一次封存漏帶GUI轉子測試所需的Src7／Dst7圖片，補齊後重測通過；初次失敗log亦保留。230份受測來源、6個binary與4個輸入hash由另一名agent獨立重驗；36次呼叫的72組有序輸出、432個容量欄位及252組工作計數皆一致。

[事前協定](gray-loader/protocol.json)、[完整摘要](gray-loader/summary.json)、[失敗配對](gray-loader/performance-failures.json)、[驗證紀錄](gray-loader/validation.json)、[完整來源與原始證據](gray-loader/raw-results.tar.gz)。封存含新舊source、binary、輸入、18份worker、測試log與工具；逐檔manifest和壓縮檔hash供review。

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake -S cpp-core -B build/gray-loader \
  -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_BUILD_TESTS=ON \
  -DFPM_BUILD_GUI=ON -DFPM_WITH_OPENCV=ON
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build build/gray-loader -j 6
ctest --test-dir build/gray-loader --output-on-failure
python-port/venv/bin/python cpp-core/tools/benchmark_gray_loader.py --output build/gray-loader-new-run
```

保留`build/fine-fused`為第30輪凍結binary，不覆蓋它。RGB內容與16-bit暫時仍需舊轉換；本輪沒有改浮點pose影像的配置／生命期，沒有新增全GPU、CUDA或長時間穩定性證據。
