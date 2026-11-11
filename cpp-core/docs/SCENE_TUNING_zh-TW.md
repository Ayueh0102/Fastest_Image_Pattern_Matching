# 桌面工具、CUDA 與場景調參

2026-11-11。在 Jetson Orin 上，三個專案固定尺度案例的 CUDA 每張耗時 0.26–5.01 ms，包含新影格準備及比對。Mac 上的密集多角度、小模板、遮擋與驗色實驗回答不同問題。現有資料可以選擇搜尋方式，現場是否達到節拍與允許的漏檢率，仍需要現場影格、標註及允收條件。

## 現場操作

1. **換料號。** 按「開啟料號配方」或把 `.fpmrecipe` 拖到視窗，確認右側的料號名稱與模板預覽。配方檔內含模板、選用的驗色遮罩、運算方式、門檻與預期件數，搬到另一台具備該後端的工作站即可開啟。右側「編輯料號設定」預設收起。
2. **檢查單張或一批。** 按「選圖並檢查」直接執行。按「批次檢查」或拖入多張影像，選擇結果儲存位置。工具會建立本批專用子目錄，並依清單順序逐張比對、存檔。同名來源以清單序號區分。
3. **處理需複核影像。** 零匹配、件數不符、驗色超限、驗色證據不足、讀檔或運算失敗均列入複核。中途遇到需複核的影像，畫面停在該張；最後一張同樣保留結果。核對畫面後按「繼續下一張」，原複核記錄仍保留。寫入失敗會保留當張供重試。「結束本批」會讓正在比對的影像完成存檔。
4. **交班或追查。** 「開啟本批結果資料夾」可查看排程清單 `batch.json`、逐張 `00001.json` 等結果及 `summary.json`。彙總檔記錄已存張數、複核序號、失敗序號與是否中止。單張結果保存當次配方的 SHA-256 及設定修改狀態。畫面上的「件數符合」僅表示數量檢查結果；產品的允收條件應由該料號定義。

### 從工廠影像製作模板

1. 按「從影像製作模板」。若已載入照片，編輯器會開啟當前影像；從即時畫面進入時會先停止取像，保留正在檢查的那一張。
2. 拖曳框住工件，也可以輸入選框左上角 X、Y 和寬高。轉動「選框順時針角度」使框沿著工件，右側顯示拉正後的模板。若角點超出照片，移動或縮小選框後即可套用。
3. 要拆分時設定列數與欄數，格子從左至右、由上而下編號。選「使用第幾格」後按「使用這個模板」。其餘格子可用「匯出全部分格 PNG」存進新的資料夾，分別建立料號。每次檢查使用一格模板。
4. 設定最低分、搜尋角度、件數與搜尋上限，需要時勾選「依模板顏色定位」。先按執行比對，再用另一張實拍核對位置並儲存料號配方。編輯器中的角度會拉正模板，搜尋角度控制待測工件容許的旋轉。

### 本機即時與單張檢查

- 有啟用 Qt 6.6 以上 Multimedia 的建置提供「開啟即時來源」。選本機影片或相機，設定取樣間隔（20–60000 ms，初始為 200 ms）。使用相機時可在 macOS 的權限提示中授權並選擇裝置。
- 尚無模板時可預覽取像，取得合適畫面時按「從影像製作模板」。已有模板時，每次從最新影格取樣並執行檢查。比對忙碌期間會略過影格，輸出的框與顯示影格保持一致。
- 按「停止即時取像」即可保留當張畫面，待當次比對完成後可匯出結果或「儲存取樣影格 PNG」，讓同一張解碼影格可再做單張比對。即時功能用於抽樣觀察；要求每一件都有存檔時，使用「批次檢查」處理已取像的照片。單張與批次均可在未安裝 Multimedia 的 Qt 建置上使用。

### 第一次建立料號配方

- 展開「編輯料號設定」，載入模板，選擇場景或匯入標註調參報告。輸入「每張預期件數」可檢查少件和多件。搜尋數量上限必須大於預期件數，才有空間回報多件。預期件數設為 0 可單純定位。
- 工件與干擾物同形、顏色不同時，可勾「依模板顏色定位」。框入工件上顏色穩定的一段邊緣及其背景。若要檢查上膠或變色，定位模板可取另一段未上膠的紙板。Mac 可選「完整 Metal 色彩特徵搜尋」，料號配方會保存搜尋方式與彩色模板。
- 需要驗色時，載入與模板等大的 PNG。純紅標產品色區、純綠標正常中性參考、純藍可標另一塊較暗的中性參考，其餘填黑。預設只量測色度距離。在獨立良品與缺陷上校準後，可設定「區域色度距離上限」。表格會標出超限或需複核的物件。
- 在代表性影像試跑後，按「儲存料號配方」，以料號及版次命名。更改已開啟配方的參數會顯示「已修改」。重新開啟另一個有效配方會清除上一張來源與結果；無法讀取的配方會保留目前工作。

### 依模板顏色定位

程式從原始 RGB 取 YCoCg 的兩個色度分量 Co 與 Cg，依模板的色度分布估計兩個代表色。對每個像素，取它到較飽和代表色的二維距離，以兩代表色之距離為半徑，換成一張 8 位元相似度圖。模板與整張待測圖使用同一張查表，再由既有金字塔搜尋位置與旋轉。在 Mac 選用完整 Metal 時，CPU 產生色彩特徵圖，Metal 搜尋這張圖。結果的 `feature_ms`、`setup_ms`、`match_ms` 可分開查閱。OpenCV 編解碼版本的命令列提供 `--chroma` 搭配既有 CPU 搜尋。[YCoCg 色彩轉換](https://www.microsoft.com/en-us/research/publication/ycocg-r-a-color-space-with-rgb-reversibility-and-low-dynamic-range/)

同亮度控制圖使用相同紋理的紅標和綠標。門檻 0.90 下，灰階搜尋兩個都命中，色彩搜尋只找到紅標。只有綠標時，色彩搜尋的匹配數為 0。本機 Glue-Detection 三張照片以同一張無膠參考照取樣，門檻 0.75、±15°。模板取上端時，膠層改色使兩張有膠照的框中心偏離原灰階位置 158 與 173 px。看到錯位後改取下緣，在相同三張照片上目視確認全部框住下緣。開啟儲存過的下緣料號配方，以完整 Metal 循環跑四輪，第一輪後的逐圖中位數為 20.2–21.7 ms（特徵、影格準備與搜尋）。這三張用於選模板，新的工件、色溫及負樣本仍須獨立驗證。

在已安裝 Qt 6、OpenCV 的 Mac，從專案根目錄建置、開啟現有的轉子範例。

```sh
cmake -S macos-cli -B build/scene-desktop -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON
cmake --build build/scene-desktop --target fpm_gui -j
build/scene-desktop/MatchTool.app/Contents/MacOS/MatchTool --gui --src "Test Images/Src7.bmp" --tmpl "Test Images/Dst7.bmp"
```

CUDA 版 Qt 工具使用同一個 C++ Session。選「CUDA 搜尋＋CPU 姿態精修」會啟用 SM80+ 的精確粗搜尋 6、細搜尋 5、傳輸 3，連續姿態精修使用 CPU。Windows 在有 Qt 6、CUDA Toolkit 及對應 C++ 編譯器的終端機執行下列命令。Mac 已驗證 Qt 操作與完整 Metal，J1 已驗證 CUDA Session 及核心。J1 沒有 Qt 6，Windows RTX 工作站本輪離線。

```sh
cmake -S cpp-core -B build/desktop-cuda -DFPM_BUILD_GUI=ON -DFPM_BUILD_EXPERIMENTS=ON -DFPM_WITH_CUDA=ON
cmake --build build/desktop-cuda --config Release --target fpm_gui fpm_native_pipeline_runner
```

## 哪些搜尋參數影響結果

| 場景 | 先調整 | 用什麼判斷 |
|---|---|---|
| 工裝固定方向、倍率穩定 | 角度從 0°、尺度從 1 開始。用實際偏轉與倍率範圍擴大搜尋。 | 角度範圍決定粗搜尋方向數，尺度庫每多一個值就多一個模板模型。固定場景先保留單尺度，量測換幀時間。 |
| 小、細長、方向不固定 | 在縮圖之後仍要看得見短邊紋理。先比較頂層短邊下限 0 與 12 px，再比較候選門檻。 | 匯出候選數，查看真實姿態附近是否有候選。沒候選時先處理金字塔深度。有候選卻沒匹配，再檢查最低分與精修。 |
| 數十至數百個相同物件 | 設定足夠的目標數量，再試粗候選去重 k=3。用接觸、重疊及對稱物件核對每個物件是否仍只配到一次。 | Mac 先前的新影格研究，完整 Metal 加去重在 128／512 物件分別為 384／1149 ms。物件數與角度會改變瓶頸。[多物件資料](DEDUP_PEAK2_zh-TW.md) |
| 邊界、空白背景、極小物件 | 加入空白及純干擾圖，開啟「檢查邊界與小模板分數」。預設組合要求邊界內至少一半有效支撐，並依模板面積加嚴小模板分數。 | 同時計算每張誤報及邊界真件召回。與本次短邊實驗配用時，驗證組和單用 12 px 的命中數相同。[邊界與小模板驗證](SMALL_TEMPLATE_VERIFY_zh-TW.md) |
| 互相遮擋、形狀相同但局部圖案不同 | 指定能區分料號的區域；用位置、角度、尺度及該區的紋理共同核對。只要身份區也被遮住，保留複核。 | 本次四分區投票提高遮擋召回，也大幅增加假姿態。研究腳本可重跑，桌面預設使用整模板。 |
| 色差、局部過曝 | 先確認幾何定位，再分別量測色區、亮與暗中性參考。以特定料號、光源、相機曝光校準色度距離。 | 分開計算良品誤判、應抓缺陷漏判及需複核數；零定位也要列入端到端漏檢。色度距離為線性 RGB 正規化色度距離，單位與 ΔE00 不同。 |

### 小模板與實拍

12 個不同模板分成開發 4 個、保留 8 個，每個模板各有乾淨、加噪與純干擾三種場景。正樣本每圖 49 件，同模板的乾淨與加噪圖共用姿態。保留集共有 392 個不同配置、784 次正樣本觀測。嚴格命中採一對一配對，中心誤差 ≤2 px、角度 ≤1°、相對尺度誤差 ≤2%。五組設定均使用 Mac Metal 混合流程、原尺寸、±180°、最低接受分數 0.8。時間涵蓋 Session 搜尋與精修，解碼與建立 Session 分列於原始紀錄。

| 保留集設定 | 正確／784 | 誤報 | 場景時間中位數 | 重複呼叫時間 P95 |
|---|---:|---:|---:|---:|
| 基線，候選 0.5、上限 100 | 728 | 0 | 33.1 ms | 59.1 ms |
| 候選 0.4、上限 200 | 728 | 0 | 46.1 ms | 85.1 ms |
| 候選 0.25、上限 300 | 728 | 0 | 68.9 ms | 100.2 ms |
| 頂層短邊 12 px | 784 | 0 | 75.2 ms | 191.4 ms |
| 12 px 加邊界、小模板驗證 | 784 | 0 | 75.1 ms | 192.0 ms |

開發集五組皆為 392/392、零誤報，以總時間為基線兩倍的探索預算選到最快的基線。它在保留集漏了 56 次。展寬候選組與短邊組的實際搜尋時間仍有差距。P95 彙整每個保留場景的三次重複，涵蓋正負樣本，樣本量與取樣間隔均不代表生產線的尾延遲。[逐圖時間與設定](scene-tuning/metal-results.json)

四張 FSC147 實拍使用類別中心點作參考。下表為中心 10 px 內的一對一覆蓋，照片中的同類物件可能長得和所取模板不同。結果不能當作料件精確姿態的召回率。

| 照片 | 基線覆蓋、時間 | 短邊 12 px 覆蓋、時間 |
|---|---:|---:|
| 郵票 2933 | 0/45，60.6 ms | 1/45，304.2 ms |
| 郵票 7417 | 41/43，44.8 ms | 41/43，97.5 ms |
| 瓶蓋 1913 | 48/50，67.6 ms | 49/50，229.0 ms |
| 瓶蓋 1924 | 11/48，43.8 ms | 11/48，82.5 ms |

公開實拍亦有已知的前段失敗。過去 VisA 膠囊保留集中，三張異常照片都沒有搜尋候選。驗色只能判讀已定位的物件。[實拍與標註條件](STAGE35_REAL_RGB_zh-TW.md)

### 遮擋與身份辨識

用兩個不同的實拍模板合成六張場景，每個模板各含遮擋、相似料件與接觸。分別對四個象限搜尋，將姿態換回整模板中心；至少兩個不同象限在 4 px、2°、約 3% 尺度內同意才新增姿態。身份驗證另對模板指定區做 NCC，採 0.5 下限。六張影像沿用第 34 輪的既有開發／保留切分。

| 保留場景 | 整模板 TP／FP | 加象限投票 TP／FP | 再驗身份區 TP／FP |
|---|---:|---:|---:|
| 遮擋 0–50%，6 件 | 1／1 | 6／13 | 4／12 |
| 相似干擾物，4 件真品 | 4／0 | 4／12 | 4／8 |
| 接觸或重疊，6 件 | 5／0 | 6／7 | 6／7 |

在開發集的相似料件上，身份區把 4 個誤報降到 0。換成保留模板，單一身份區無法排除重複圖案產生的錯姿態。四次象限搜尋也增加計算量。這組結果適合作為失敗案例及後續比對基準。[六個場景逐案結果](scene-tuning/partial-support.json)

### 驗色參考區過曝

在第 47 輪的兩款合成標籤、各八種光照／雜訊條件上，重播 512 個由搜尋器取得的姿態。亮參考失效才使用較暗的中性印刷區。沿用原標籤的閾值，紅色 0.00816、藍綠色 0.01470。

| 指標 | 只有亮參考 | 允許第二參考 |
|---|---:|---:|
| 需複核／512 | 99 | 7 |
| 正常判為正常／160 | 129 | 159 |
| 正常判為異常／160 | 0 | 1 |
| 應抓色差判為異常／272 | 220 | 265 |
| 應抓色差判為正常／272 | 0 | 0 |

「應抓」為非同色且目標 ΔE00 ≥4 的標註。其餘 80 件屬較小色差，未納入「應抓」分母。剩餘 7 件應抓色差均需複核，額外的 1 件良品誤判來自過曝藍綠標籤。這是已有幾何定位的合成回放，工廠實拍色差、當地光源與印墨仍需獨立校準。[驗色原始統計](scene-tuning/secondary-reference.json)

## CUDA 實機與記憶體

Jetson Orin SM87、CUDA 12.6、Release。以下為每次新影格 prepare 加 match 的穩態時間，不含取像、影像解碼、建立模板、Qt 繪圖、驗色與連續姿態精修。每組跑 3 輪，第一輪不列入彙總，其餘各暖機至少 65 ms，再各取 3 次重複的中位數。三個案例的 CPU 與兩組精確 CUDA 結果逐位元一致，14 個生成案例的 185 筆 CPU 匹配也通過比較。

| 固定尺度案例 | 搜尋設定 | 同機 CPU | CUDA | 加速 |
|---|---|---:|---:|---:|
| Src4 | ±180°，上限 5 | 155.97 ms | 1.49 ms | 104.6× |
| Src7 轉子 | ±180°，上限 5 | 413.36 ms | 5.01 ms | 82.5× |
| Src9 | 固定方向，上限 100 | 3.42 ms | 0.26 ms | 13.1× |

同機三張合計 572.75 → 6.76 ms，84.7×；逐張等權的加速幾何平均為 48.3×。CUDA 的第一次 prepare 加 match 分別為 3.43、12.04、0.98 ms。RTX PRO 4000 舊版在其自身 17 張固定尺度預設案例上，CPU 與精確 CUDA 的總時間為 778.9 → 16.57 ms。兩台機器的樣本數不同。[J1 實測](scene-tuning/orin/standard.json) · [RTX 既有實測](CUDA_FINE_TENSOR_zh-TW.md)

- **淺金字塔與顯存。** Src1 的縮減面積下限放大 16 倍後，粗搜尋遍歷約 1.32 億個視窗。粗搜尋 mode 8 每角度先保留最多 1024 筆，超量角度依實際數量重算。回報的粗搜尋 GPU 配置由 2018.0 降到 12.5 MiB，時間由 239.8 增至 258.1 ms。此計數包含 CUDA 後端保留的裝置配置，與整機顯存、程序 RSS 各自獨立。模式 8 現可透過 frame API 或 benchmark 選用。[容量比較](scene-tuning/orin/capacity.json)
- **次像素與指令選擇。** 相同工作分配下，細搜尋維持 Tensor core 時 Src1／Src6 為 27.82／61.58 ms，改用 DP4A 為 37.03／87.88 ms。兩者輸出相同，工具選用 Tensor mode 5。[細搜尋對照](scene-tuning/orin/fine.json)
- **CUDA 的驗證範圍。** 桌面共用的 CUDA Session 用固定尺度植入目標比對、旋轉搜尋、連續 CPU 精修、重複執行與換幀測試驗證。公開案例時間來自 frame 路徑。Jetson 使用 GCC 並以 `-DCMAKE_CXX_FLAGS=-ffp-contract=off` 建置，使 CPU 浮點參照符合逐位元比較條件。測試結束已恢復四路取像並移除該機的暫存來源及執行檔。[驗證紀錄](scene-tuning/orin/validation.json)

## 用自己的資料校準

桌面設定可先試跑。需要可重現的選型時，提供一份 JSON 陣列，每列有唯一的 `name`、`split`（`dev` 或 `test`）、`source`、`template`、`truth`。路徑相對於清單。每筆真值含 `pos_x`、`pos_y`（原圖中心像素座標）、`angle`（逆時針度數）、`scale`。純干擾圖用空陣列。固定料號可在兩組共用同一張模板，各自提供獨立來源。可在每列加上同批次或連續片段的 `group`，程式會阻止同組或相同來源分散到開發與保留集。本次研究使用 `--split-by template`，同一模板也只能出現在其中一組。

下例可先用隨附生成器跑通流程。Python 需有 NumPy、SciPy、OpenCV。以 `--backend 0` 跑 CPU、`1` 跑 Metal、`2` 跑具備 CUDA 的建置。可用 `--angle 10 --scale-range 0.9 1.1 0.1` 將比較限定於工裝可達的角度與尺寸；匯入報告時會一併套用。時間預算比例 2 是本次研究選擇。現場有明確時間要求時可再給 `--max-p95-ms`，它約束開發資料的搜尋呼叫 P95。合格設定之間先比嚴格姿態 F1，同分取總時間較短者；若都超時，報告會保留空選擇。

```sh
python3 cpp-core/tools/scene_tuning.py generate build/tuning-examples
python3 cpp-core/tools/scene_tuning.py measure build/tuning-examples/cases.json \
  --runner build/scene-workbench/experiments/fpm_native_pipeline_runner \
  --backend 1 --split-by template --time-ratio 2 --output build/tuning-result.json
```

以上 runner 路徑對應以 `cmake -S cpp-core -B build/scene-workbench -DFPM_BUILD_GUI=ON -DFPM_BUILD_EXPERIMENTS=ON -DCMAKE_BUILD_TYPE=Release` 建置的產物；目標名稱是 `fpm_native_pipeline_runner`。重跑本次四張實拍，可另給 `--photo-manifest build/stage35-real-rgb/inputs/cases.json`。照片只供評估，不參與參數選擇。既有機器可直接在工具中匯入本次 Metal JSON。

新的相機資料若要支持上線判斷，最有價值的是同料號跨批次與光照的連續影格、邊界及空白反例，以及允許的每幀延遲、漏檢和誤報上限。以獨立批次檢查漏檢、每圖誤報、需複核率與從取像到輸出的 P95，再決定適用料號。

## 重跑對照實驗

Mac 的 Release 建置通過 91/91 項 CTest。Qt 測試透過實際按鈕執行三種場景、校準檔匯入、換幀、完整 Metal、第二驗色參考與 JSON 匯出。從 macOS 相容入口另行建置的 MatchTool.app 已帶入 Src7／Dst7 啟動驗證。

遮擋實驗可重新產生第 34 輪的六個場景。驗色回放使用第 47 輪既有的搜尋輸出與模板遮罩，原始輸入的產生方式及保存檔列在[第 47 輪證據目錄](stage47-chroma-margin/)。

```sh
python3 cpp-core/tools/challenge_coverage_scenes.py --output build/partial-cases
python3 cpp-core/tools/partial_support_study.py build/partial-cases/cases.json \
  --runner build/scene-workbench/experiments/fpm_native_pipeline_runner \
  --backend 1 --output build/partial-results.json
python3 cpp-core/tools/secondary_reference_study.py --scenes build/stage47-holdout/synthetic \
  --masks build/stage47-margin/masks --search build/stage47-margin/run/results \
  --thresholds cpp-core/docs/stage47-chroma-margin/decision-and-failures.json \
  --runner build/scene-workbench/experiments/fpm_color_verify_runner --output build/secondary-results.json
```

在具備 CUDA 的建置上，可匯出專案 PGM 固定案例並重跑同機 CPU／GPU。要重現 J1 的兩個額外對照，淺金字塔採 Src1、exhaustive、`--min-area-scale 16`，比較粗搜尋 6 與 8。細搜尋採 Src1、Src6、subpixel，固定粗搜尋 6、傳輸 3，切換細搜尋 5 與 7。

```sh
python3 cpp-core/tools/export_fixture_pgm.py build/pgm
python3 cpp-core/tools/benchmark_frame_backend.py --bench build/desktop-cuda/experiments/fpm_frame_bench \
  --pgm-dir build/pgm --cases Src4,Src7,Src9 --modes exhaustive --rounds 3 --repeat 3 --warmup-ms 65 \
  --arm cpu=0,0,0,0 --arm cuda=2,1,256,6,3,5 --arm compact=2,1,256,8,3,5 --output build/standard.json
```
