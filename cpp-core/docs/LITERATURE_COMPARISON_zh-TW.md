# 與 2024–2026 新方法的實測比較（第50輪）

第50輪，2026-10-26～27，Apple M4 Pro／48 GiB，macOS 27。這是量測與比較輪，不改任何演算法、預設、GUI 或 CURRENT_BEST 建議。比較對象：
- 本專案：穩定核心（`fpm_cli`），以及研究路徑的混合與完整 GPU（peak2／fine4），去重 k0／k3。
- [PoseMatch-TDCM](https://github.com/ZhouJ6610/PoseMatch-TDCM)（Expert Systems with Applications 2026）。
- [GeCo2](https://github.com/jerpelhan/GECO2)（AAAI 2026）。
- OpenCV `matchTemplate`＋暴力旋轉。

所有方法拿同一張影像、同一個模板檔（GeCo2 改拿影像內的範例框），由同一支獨立評分程式評分。

**GeCo2 的精度來自實驗室 CUDA（2026-10-27 補跑）。** 它在 M4 Pro MPS 上沒有通過事前登錄的真偽關卡。補跑改在實驗室 PRO 4000（RTX PRO 4000 Blackwell、Windows 原生）上執行，用官方編譯的 kernel，在 FSCD-147 test 重現了論文：AP 46.84、AP50 76.08，論文為 46.81、75.89。H1(c)、H2 與 B2 的 GeCo2 精度都以 CUDA 為準；MPS 與 CPU 的輸出只作描述。見[GeCo2 CUDA 補跑](#2026-10-27-geco2-cuda-補跑)。

## 結論：各自在哪裡贏

| 情境 | 勝出 | 依據 |
|---|---|---|
| 外觀相同、要精確位姿（A 組） | **本專案** | 第37輪 7 個場景嚴格（2 px／1°／2%）667/667；TDCM 改編 0/667、OpenCV 9/667。A 組正場景合併的匹配誤差中位數：我們 0.025 px／0.013°，TDCM 改編 1.53 px／1.27° |
| 大圖、大量物件（4096²～8192²，128～512 個） | **本專案** | 完整 GPU k3 每張 373／1,134／1,476 ms。TDCM 改編在 CPU 上 6.4／14.7／73.0 s（MPS 5.5／13.4／70.2 s），而且只找到 23／35／95 個在 20 px 內；OpenCV 全搜尋外推 19～74 分鐘／張 |
| 照明變化（曝光、白平衡、局部光、JPEG） | **本專案** | 17 個場景嚴格 484/484，同色物件 140/140；OpenCV 112/484，TDCM 改編 6/484；只比外接框時 GeCo2（CUDA）482/484、多報 60，我們 484/484、多報 0 |
| 負控制（空場景、全黑、錯誤模板） | **本專案（及各方法）** | 我們全部 arm 0 誤判；OpenCV、TDCM 改編也是 0 |
| 第35輪 43×30 小郵票模板（開發集） | **OpenCV、TDCM 改編** | 我們在預設深度只找到 4／21／8 個（共 49 個）；OpenCV 嚴格 35/49，TDCM 改編在 20 px 內 49／46／31。這是第35、49輪已知的弱點 |
| 遮擋與相似干擾（第34輪產生器，新種子） | 平手，各有長處 | 嚴格：我們 21/32，OpenCV 3/32，TDCM 0/32；20 px 定位：TDCM 改編 23/32 最多，我們 22/32；只比外接框時 GeCo2（CUDA，拿 3 個真值範例框）30/32、多報 10，我們 22/32、多報 4 |
| 微小模板負場景（空白、干擾紋理） | **TDCM 改編** | 48 個負場景的誤判：TDCM 改編 93、OpenCV 198、我們研究路徑 337～338、穩定核心 349（淺金字塔 470） |
| TDCM 自己的 COCO 協定，36×36 模板＋大尺度變化（B1） | **TDCM** | S2 成功率 99.6% 對我們 82.8%，S2.5 為 98.0% 對 75.5%；S1 則我們較好（mIoU 0.957 對 0.936，角度 0.41° 對 1.31°） |
| B1 給原尺寸模板 | **本專案** | 四種設定嚴格 98.9～99.6%，位置 0.03～0.05 px、角度 0.08～0.24°；TDCM 16～24%、0.55～1.0 px、1.2～2.0° |
| 自然影像少樣本計數與偵測（FSCD-147，B2） | **GeCo2** | GeCo2（CUDA）AP50 0.752、計數 MAE 7.7；我們 0.110、約 55 |
| A 組外接框（給 GeCo2 三個真值框當範例） | **GeCo2** | 外接框召回 1015/1015，我們 901/1015。差距全在第35輪郵票模板（294 對 180）。GeCo2 只輸出水平框，不能比位姿 |

一句話：**同一外觀、要位姿、要速度、要抗照明，選本專案；尺度或外觀變化大、自然影像計數，學習式方法較強；低紋理小模板仍是我們的弱點。**

## 假說判定

| 事前假說 | 判定 | 依據 |
|---|---|---|
| H1 A 組：(a) 每個正場景嚴格召回不低於 TDCM 改編與完成全搜尋的 OpenCV；(b) 誤差中位數低於 TDCM 改編；(c) 外接框召回不低於 GeCo2 | **否** | (a) 在第35輪開發集三張失敗（我們 4／21／8 個，OpenCV 各 35 個），其餘 13 張通過；(b) 通過；(c) 也失敗：16 個正場景合併外接框召回，GeCo2（CUDA）1015/1015，我們 901/1015 |
| H2 B2：GeCo2 AP50 > 我們 | **是** | GeCo2（CUDA，真偽關卡通過）0.752 對 0.110（本輪評分程式，pooled 101 點、IoU 0.5） |
| H3 微小模板（官方 36×36 權重，依使用者改寫）：TDCM 改編嚴格召回 ≥ 我們 | **否（我們較好）** | 嚴格 76/2352（3.2%）對 2286/2352（97.2%）；20 px 定位 68.4% 對 100% |
| H4 ≥4096² 多物件，同一台 M4 Pro：我們每張時間 < TDCM 改編 CPU | **是** | 128 物件 373 對 6,386 ms，512 物件 1,134 對 14,663 ms，8192²／512 物件 1,476 對 72,997 ms |
| H5 A 組負控制我們 0 誤判 | **是** | 空場景、全黑、7 個錯誤模板場景，每個 arm 都是 0 |

## 參賽方法與設定

**本專案。** 建置 `build/stage50-ours`。~~C++ 與 main `af7c97a` 相同~~（合併驗收更正：實際從 `8f0e6a9` 建置，少了第49輪預設關閉的 `pyramid_rule`；在 C++ 等於 `af7c97a` 的新建置上重跑 787 個 job，輸出逐位相同，見[驗收補充](#2026-10-27-合併驗收補充)）。
- **arm 設定**：各 arm 用 CURRENT_BEST 的研究設定。第44輪有封存的 4 個 A 案例直接沿用第44輪的 job 檔，只改重複次數、暖機與 profiling。
- **與第44輪封存比對**：48 筆首次呼叫輸出全部逐位相同。
- **不採用的選項**：第49輪的 `pyramid_rule` 沒有在任何 arm 開啟。

**PoseMatch-TDCM**（commit `a6adc3a`，MIT）。
- **權重**：官方權重在 repo 的 `dict/model.pth`，SHA256 `4bbd7178…`。載入後 794,333 個參數，論文寫 3.07M，對不上。
- **主要流程**：用 200 組 S1 開發樣本事前選定。README 範例 `test.py` 比評估腳本 `test_single.py` 更接近論文 S1，距離 0.448 對 1.079，所以 `test_py` 為主要流程，`test_single` 為次要。
- **官方範圍**：B1 單目標。
- **A／C 組的「TDCM 改編」**：
  - 影像與模板先縮放，讓模板最長邊為 36 px。
  - 取分數圖中 3×3 局部最大且 ≥0.3 的點；0.3 是在開發場景上以嚴格 F1 選出。
  - 每個目標照官方流程精修，最後做旋轉 NMS。

**GeCo2**（commit `67d27e8`，MIT＋Apache 子目錄）。
- **權重**：`GECO2FSCD.pth`＋SAM2 hiera-B+，門檻 0.11、NMS 0.5（`inference.py`）。README 的 Drive 版與 HF 版權重的 475 個張量完全相同。
- **在 Mac 上的三處偏離**：
  - 可變形注意力改用同檔案的 PyTorch 參考實作。
  - 純 CPU 時要先設好參考點，因為上游只在換裝置時才計算。
  - scipy 改用 1.16.3。
- **精度用的 CUDA 執行**（實驗室 PRO 4000，Windows 原生）：
  - 官方 `MultiScaleDeformableAttention` 用 MSVC 14.44＋CUDA 12.8 編成 sm_120。
  - torch 2.7.1+cu128，其他套件版本與 Mac 相同。
  - GeCo2 原始碼沒有任何修改，上面三處偏離都不適用。scipy 仍是 1.16.3，與 Mac 相同。

**OpenCV。** `TM_CCOEFF_NORMED`，加上旋轉矩形遮罩，1° 角度網格 × 案例的尺度清單，NMS 與 fpm 相同。超過 2.5 Mpx 的場景只跑 6 張分數圖再外推時間，這些場景的精度標為 N/A。

## 資料與事前登錄

**凍結與快照。** 凍結的 [protocol.json](stage50-literature-comparison/protocol.json) 與源碼快照（`73b926e`，338 個檔）在產生 C 組場景與任何正式輸出之前提交。三條開發選擇規則更早一步提交（`ebcbecc`），開發選擇本身在兩者之間執行。

**各組資料：**
- **A 組**：
  - 正場景：第37輪 7 個、第35輪半合成控制 6 個（只在本機，FSC 衍生）、測試樣本集 v1 的 Blender 灰階正面場景 3 個。
  - 負控制：空場景、全黑，以及 7 個錯誤模板場景（Dst9 灰階縮到各案例模板尺寸）。
- **B1**：
  - 樣本：作者自己的 `rotate_crop`（種子 42），各設定前 1,000 組，36×36 為主要變體。
  - S2：用作者公布的清單。
  - S1.5／S2.5：用作者的 `filterData` 規則產生。
  - S1：論文只寫「僅旋轉、尺度固定為 1」，所以採我們的操作定義「寬高都在 34–38 px」，共 129 列，依影像切成開發與評估兩部分。
  - 重現性：產生器與作者紀錄 `result-single.txt` 的前 26 筆逐筆相同，第 26 筆起分岔（推測是平台數值差異）。因此這批樣本是同清單、同程式、同種子的抽樣，不是作者那一批。
- **B2**：
  - 影像：FSC-147 test 1,190 張（官方 Google Drive）。
  - 框：FSCD-147 標註來自**非官方 Hugging Face 鏡像** `ChipmunkG4/FSCD-147_FSCD-LVIS_temp@48b4cb3d`，經使用者同意並在此揭露。
  - 核對結果：影像與官方逐位元組相同；類別數與切分和論文相同；但框數 77,236 比點數 78,836 少 2.0%，有 180 張的框比點多，1,539 個框超出影像 1–4 px；論文沒有公布框總數（[核對紀錄](stage50-literature-comparison/fscd147-mirror-verification.json)）。
  - 我們的模板：每張影像的範例框 0 裁切。
- **C 組**：
  - 微小模板 24 個（12–24 px，第49輪產生器配新種子）。
  - 遮擋、干擾、接觸 6 個場景（第34輪產生器配新種子）。
  - 8192²／512 物件 1 個。
  - 照明：測試樣本集 v1 的 17 個場景。
  - 全部在凍結後產生，產生器會核對 protocol 記錄的雜湊。

**評分（[literature_score.py](../tools/literature_score.py)）：**
- **嚴格位姿**：2 px／1°／寬高比 0.98–1.02，最大基數二分配對，計數與 `many_object_accuracy.evaluate` 一致。
- **20 px 關聯**：只作描述，不取代主要指標。
- **框**：外接框 IoU≥0.5，COCO 101 點 AP50。
- **計數**：MAE／RMSE。

## A 組：本專案主場

合併各家族的嚴格召回。時間是每張呼叫的中位數，不含解碼與模型建立；OpenCV 只跑 1 次呼叫，時間只作描述。

| 家族（GT 數） | 完整GPU k3 | 混合 k3 | 穩定核心 | OpenCV | TDCM 改編 CPU |
|---|---|---|---|---|---|
| 第37輪 7 場景（667） | 667（45.8 ms） | 667（79.0 ms） | 9/9 固定尺度 | 9（4/7 張完成全搜尋） | 0；20 px 內 71 |
| 第35輪控制 6 場景（294） | 180 | 180 | 152 | **246**；20 px 內 294 | 9；20 px 內 265 |
| Blender 灰階 3 場景（54） | 53 | 53 | 17/18 | 22；20 px 內 54 | 0；20 px 內 16 |
| 負控制 9 場景（誤判） | 0 | 0 | 0 | 0 | 0 |

第35輪的差異全部來自開發集的小郵票模板（每張 49 個）：
- 我們在 clean／exposure／white balance 分別 4／21／8 個。
- OpenCV 各 35 個。
- 保留集模板（58×71）我們三張都是 49/49。

## B1：TDCM 的 COCO 合成協定

每列 1,000 組；OpenCV 只跑前 200 組。成功率與 mIoU 照 `test_single.py` 的定義（分數 >0.01 且 IoU>0.1），表中數字是合併驗收修正 IoU 角度方向後的重算值（原值與差異見[驗收補充](#2026-10-27-合併驗收補充)），嚴格指標是逐軸 2%。尺度是相對 36 px 的非等向值；我們搜尋等向尺度，所以在 36×36 變體上幾乎不可能通過嚴格標準。

| 設定／變體 | TDCM `test_py` | 我們完整GPU k0 | OpenCV |
|---|---|---|---|
| S1 36×36 | 成功 100%、mIoU 0.936、0.55 px、1.31°、81 ms | 99.8%、**0.957**、**0.12 px**、**0.41°**、9 ms | 100%、0.927、0.74 px、0.94° |
| S1.5 36×36 | **99.8%、0.935**、0.60 px、1.20° | 97.5%、0.838、1.08 px、2.15°、58 ms | 94.5%、0.816 |
| S2 36×36 | **99.6%、0.926**、0.70 px、1.37° | 82.8%、0.747、2.22 px、4.47°、120 ms | 81.5%、0.716 |
| S2.5 36×36 | **98.0%、0.890**、1.11 px、2.42° | 75.5%、0.727、2.69 px、5.13°、167 ms | 65.5%、0.696 |
| S2 原尺寸 | 99.7%、0.925、0.70 px、1.38°；嚴格 20.7% | **99.9%、0.997、0.03 px、0.08°；嚴格 99.4%**、6 ms | — |

- **次要流程 `test_single`**：mIoU 0.869–0.927，角度誤差 1.98–3.30°，都比 `test_py` 差。
- **穩定核心（原尺寸）**：嚴格 98.0–99.2%。
- **TDCM 每對約 80 ms**（CPU 8 執行緒，模型已載入）。
- **論文數字**：在 i9-14900KF 上 S1 為 11.3 ms、S2 為 14.2 ms，只並列、不換算倍數。

## B2：FSCD-147（自然影像計數與偵測）

| 方法 | AP50（框） | 精確率／召回率 | 計數 MAE（對點）／RMSE | 時間中位數 |
|---|---|---|---|---|
| 我們 完整GPU k3（分數 0.8，未調） | 0.110 | 0.380／0.206 | 54.9／114.9 | 85 ms |
| 我們 混合 k3 | 0.110 | 0.381／0.207 | 54.9／114.8 | 47 ms |
| 我們 完整GPU k3（分數 0.9，在 val 上選定） | 0.059 | 0.545／0.097 | 56.5／87.3 | 85 ms |
| **GeCo2 CUDA**（實驗室 PRO 4000，真偽關卡通過） | **0.752** | 0.783／0.817 | **7.7**／40.0 | 288 ms（另一台硬體） |
| GeCo2 MPS（**描述性**，未過真偽關卡） | 0.693 | 0.735／0.750 | 9.8／41.2 | 4.49 s |

**真偽關卡（GeCo2 論文 FSCD147 test：AP 46.81、AP50 75.89、MAE 7.64）：**
- **CUDA 結果（精度以此為準）**：AP 46.84、AP50 76.08，差距 +0.03／+0.19，關卡通過；計數 MAE 7.69（對點）。兩種 AP50 算法不同：COCOeval 照作者把框截成整數，本輪評分程式是 pooled 101 點。CUDA 是 76.08 對 75.17，差 0.91 點；MPS 是 69.41 對 69.26，差 0.15 點。
- **MPS 結果**：AP 40.45、AP50 69.41，差距超過事前定的 2 個百分點。
- **權重**：已排除。
- **CPU 對照**（protocol 原本排定的前 100 張描述性執行，同一個參考 kernel）：
  - CPU：AP50 0.861、MAE 3.49。
  - MPS：AP50 0.788、MAE 6.29。
  - 我們：AP50 0.057。
  - 100 張中沒有一張的 CPU 與 MPS 輸出相同。
- **使用者決定**：等實驗室 CUDA 用官方編譯 kernel 重跑；CUDA 也差超過 2 點，才停用 AP50。CUDA 通過，所以 AP50 保留，H2 照原文以 AP50 判定。
- **是 MPS 的數值問題**：逐張比較，CUDA 與 CPU（前 100 張）有 97 張框數相同，框數差平均 0.03；CUDA 與 MPS 在 1,190 張中只有 345 張相同，平均差 6.7。前 100 張上 CUDA 的 AP50 0.861、MAE 3.46，和 CPU 的 0.861、3.49 幾乎一樣。三種裝置之間沒有一張的輸出逐框相同。

## C 組：難題

| 家族（GT 或場景數） | 完整GPU k3 | 混合 k3 | 穩定核心／淺金字塔 | OpenCV | TDCM 改編 CPU |
|---|---|---|---|---|---|
| 微小模板正場景（2352） | 2286 | 2286 | 1643／2291 | 1353；20 px 內 2352 | 76；20 px 內 1609 |
| 微小模板負場景 48 張（誤判） | 338 | 338 | 349／470 | 198 | **93** |
| 遮擋／干擾／接觸（32） | 21；20 px 內 22 | 21 | — | 3；20 px 內 18 | 0；20 px 內 **23** |
| 照明 17 場景（484；同色 140） | **484；140** | 484；140 | — | 112；35 | 6；0 |
| 8192²／512 物件（512） | 512（1,476 ms） | 512（2,311 ms） | — | N/A（只跑 6 張分數圖，外推 74 分鐘） | 0；20 px 內 95（73.0 s） |

## 時間與記憶體（同一台 M4 Pro，H4 場景）

| 場景 | 我們 完整GPU k3 | 我們 混合 k3 | TDCM 改編 CPU | TDCM 改編 MPS | OpenCV（外推全搜尋） |
|---|---|---|---|---|---|
| 128 物件 4096² | 373 ms；RSS 91 MiB＋Metal 140 MiB | 561 ms；230 MiB | 6.39 s；RSS 2.80 GiB | 5.49 s；RSS 0.69 GiB＋MPS 1.03 GiB | 19 分鐘 |
| 512 物件 4096×8192 | 1,134 ms；165＋282 MiB | 1,455 ms；439 MiB | 14.7 s；5.06 GiB | 13.4 s；0.81＋2.03 GiB | 38 分鐘 |
| 512 物件 8192² | 1,476 ms；292＋538 MiB | 2,311 ms；782 MiB | 73.0 s；9.78 GiB | 70.2 s；1.19＋3.03 GiB | 74 分鐘 |

說明：
- RSS 與 GPU 配置分開記，不相加。
- 三個區塊的時間都很一致，例如完整 GPU 在 128 物件是 378／373／372 ms。
- TDCM 改編的時間大多花在逐峰精修（官方流程每個目標都精修）。
- GeCo2 在 B2 的 MPS 驅動配置峰值 40.0 GB、RSS 2.6 GB；CPU（前 100 張）RSS 11.4 GB。
- GeCo2 CUDA（實驗室 PRO 4000，另一台硬體，只並列）：
  - B2 整批的 peak working set 2.62 GB；CUDA 最大配置 5.79 GB，最大保留 9.53 GB。
  - 每張時間中位數 288 ms，整批 7.1 分鐘。
  - A／C 每案中位數 292 ms，最長是 8192² 場景的 1,638 ms。

## 失敗與揭露（全部保留）

**操作失誤與修正：**
- **門檻選擇第1次**：驅動程式太晚建立輸出目錄，每個門檻的第一個案例都失敗。已修正、補測試，整個選擇重跑（[attempt1](stage50-literature-comparison/dev-tdcm-threshold-attempt1.json)）。
- **A／C 正式量測第1次**：zsh 不會對 `$M` 斷詞，所有方法名稱被當成一個字，435 個 worker 瞬間失敗，沒有任何量測執行。已保留，並加上方法名稱檢查。
- **真偽關卡第1次**：pycocotools 的 `summarize()` 把 AP@[.5:.95] 寫死成 maxDets=100，所以 AP 顯示 −1；AP50 不受影響。已照作者的 `COCOevalMaxDets` 修正。

**開發選擇的結果與敏感度：**
- **TDCM 改編的門檻**：嚴格 F1 規則選出 0.3。若看 20 px 定位，多物件開發場景在 0.01 時是 36/36，到 0.3 只剩 11/36。規則事前提交，照規則採用。
- **B2 的分數門檻**：選出 0.9，落在格點邊界。

**我們方法自己的限制：** FSC val 849 號影像在候選上限 4000 時超出完整 GPU 的候選容量（2M 槽）；只發生在開發集，test 沒有遇到。

**主機與並行：**
- B1 的 TDCM 批次和 B2 的 GeCo2 MPS 批次同時執行，兩者都只計精度、時間只作描述。
- A／C 正式計時期間持有本輪鎖，並確認沒有任何重工作（包括我們自己的）在跑。
- 主機負載逐 worker 記在 manifest。

**數字不一致與重現性：**
- TDCM 參數量 794,333，與論文的 3.07M 不符。
- B1 產生器與作者紀錄只有前 26 筆相同。
- FSCD 鏡像的框數與點數不一致（見上）。

**凍結後的修訂：** 凍結時 6 條；GeCo2 CUDA 補跑在量測前又加 4 條，共 10 條，都記在 protocol 的 `revisions`。

## 量測方法與界線

**時間的量法：**
- 每個 worker 都是新程序，時間不含解碼與模型或 Session 建立。
- 我們的 arm：暖機 1 秒、量 3 次，3 個區塊，奇數區塊反向順序。
- TDCM 改編：在 H4 場景跑 3 個區塊（各 1 次暖機＋3 次量測），其他案例只量 1 次，時間只作描述。
- OpenCV、GeCo2：只量 1 次，時間只作描述。GeCo2 CUDA 的 A／C 每案暖機 1 次、量 1 次；B2 整批在第一張暖機 1 次。
- 不同硬體的時間只並列，不換算倍數。

**證據的界線：**
- 本專案、TDCM、OpenCV 都只在一台 M4 Pro 上量；GeCo2 的精度來自實驗室 PRO 4000 的 CUDA。
- A／C 場景都是合成或半合成；FSC 照片只在 B2 與第35輪控制場景出現，沒有完整位姿 GT。
- TDCM 改編是我們設計的多目標延伸，不代表作者沒公開的多目標程式。
- GeCo2 只輸出水平框，所以 A／C 的 GeCo2 只能比外接框（IoU ≥ 0.5），比我們的嚴格位姿判準寬鬆得多。
- GeCo2 的範例框是每個場景前 3 個真值物件的外接框，等於拿到同一張影像裡的正確樣本；我們只拿模板檔。這些物件沒有從計分排除，所以 GeCo2 的外接框召回裡，A 組有 48 個（16 場景 × 3）、C 組有 216 個（72 場景 × 3）本身就是範例框。
- GeCo2 只跑正場景，因為它需要影像內的範例框，負控制無法給。

## 驗證

| 檢查 | 結果 |
|---|---|
| 單元測試 | `test_literature_score.py` 18 項＋`test_literature_compare.py` 14 項通過；評分計數在 20 組隨機資料上與 `many_object_accuracy.evaluate` 相同 |
| 與第44輪封存比對 | 48 筆首次呼叫輸出逐位相同（4 個案例 × 4 個 arm × 3 個區塊） |
| 封存重播 | 從解開的壓縮檔重算 A、C、B1、B2 與假說，和原始輸出算出的結果相同 |
| B1 樣本 | 凍結後在新目錄重新產生，S1 評估、S1 開發、S2 評估的 GT 雜湊都和凍結前相同 |
| FSCD 鏡像 | 影像、FSC 標註、切分與論文 Table 1 都相符；框數不一致照實記錄 |
| GeCo2 AP50 交叉核對 | MPS：COCOeval 69.41，本輪評分程式 69.26；CUDA：76.08 對 75.17 |
| 正式量測的 worker 失敗 | A、C、B1、B2 都是 0；GeCo2 CUDA 89 筆（B2 1 批＋88 案）也是 0，沒有重試 |
| GeCo2 CUDA 前置檢查 | 4 項全過（CUDA／sm_120、編譯 kernel 與 float64 參考差 1.8e-7、單張 6 個有限值框、149 個 sam_mask 鍵全由 SAM2 載入） |
| GeCo2 CUDA 補跑不影響其他結果 | 新摘要中 A、C 非 GeCo2 的 1,080 列、失敗清單、第44輪比對、H3–H5 都與 formal-summary.json 相同；B1 只差驗收已更正的 paper 欄位（66 處） |

## 封存

[stage50-literature-comparison/](stage50-literature-comparison/)：

- **事前登錄與快照**：`protocol.json`（含修訂紀錄）、`source-snapshot.tar.gz`、`source-sha256.json`。
- **開發選擇**：`dev-*.json`。
- **第1階段**：`feasibility.{md,json}`、`phase1-*`。
- **資料核對**：`fscd147-mirror-verification.json`、`b1-prefreeze-fixtures.json`。
- **真偽關卡**：`gate-geco2-mps*.json`、`gate-geco2-cuda.json`。
- **GeCo2 CUDA 補跑**：
  - `geco2-cuda-environment.json`：主機、工具鏈、安裝檔雜湊與簽章、重開機旗標、kernel 建置、GeCo2 原始碼狀態、前置檢查。
  - `formal-summary-geco2-cuda.json`：GeCo2 精度改用 CUDA 的完整摘要。
  - `raw-results-geco2-cuda.tar.gz`：實驗室的 manifest、啟動紀錄、輸出與 log，加上匯入本機 run 目錄的 manifest 列與傳輸包清單。
  - `source-snapshot-geco2-cuda.tar.gz`、`source-sha256-geco2-cuda.json`：補跑所用的程式快照。
- **結果與判定**：`formal-summary.json`（凍結版，保留原值）、`decision-and-failures.json`、`validation.json`、`archive.json`。
- **原始結果壓縮檔**（每個都低於 100 MB）：
  - A 組與 C 組（C 分 4 個檔）是完整原始輸出。
  - B1、B2 與開發集是壓縮版：每個 runner 輸出只留 matches、時間、記憶體與完整原檔的 SHA256；完整原檔留在本機 `build/stage50-external`，用 `full-output-sha256.json` 對應。
- **場景與樣本 metadata**：`scene-metadata.tar.gz`，含 cases、GT、種子、清單與產生紀錄。
- **不放入封存的資料**：FSC、COCO 衍生影像與 FSCD 標註都沒有再散布權，只留雜湊與重建方式。

工具：[literature_compare.py](../tools/literature_compare.py)（驅動、開發選擇、評分彙整）、[literature_score.py](../tools/literature_score.py)、[literature_tdcm_worker.py](../tools/literature_tdcm_worker.py)、[literature_geco2_worker.py](../tools/literature_geco2_worker.py)、[literature_opencv_worker.py](../tools/literature_opencv_worker.py)、[literature_b1_fixtures.py](../tools/literature_b1_fixtures.py)、[literature_scenes.py](../tools/literature_scenes.py)、[literature_fscd_verify.py](../tools/literature_fscd_verify.py)、[literature_geco2_fscd_eval.py](../tools/literature_geco2_fscd_eval.py)、[literature_archive.py](../tools/literature_archive.py)。

## 未完成

- ~~**GeCo2 的 CUDA 精度**~~：已於 2026-10-27 在實驗室 PRO 4000 完成，見下節。
- **未測**：本專案在 Windows／Linux、其他 GPU 上的表現；真實相機的連續影格。GeCo2 CUDA 只用實驗室的一張 GPU。

## 2026-10-27 合併驗收補充

合併到 main 前，主 session 以 reviewer 身分在本輪 head `98e9cae` 的乾淨 worktree 重新驗收（前後工作樹都乾淨）。H1、H3、H4、H5 的判定都沒有變。B1 的描述性論文指標因為一個 IoU 角度方向錯誤而重算；另補測試與幾處揭露。

- **重跑 gate：**
  - 單元測試 34 項（本輪 32＋索引 2）與索引檢查（Python 3.12、3.14）通過。
  - archive.json 的 30 個檔，大小與 SHA-256 都相符。源碼快照 338 個檔與凍結 commit `73b926e` 逐位相同。
  - C 組場景 provenance 記錄的 protocol 雜湊等於凍結版本（`5c45b4d5…`）。正式量測都在凍結（13:07Z）之後才開始：B2 13:08Z、B1 13:10Z、A 15:06Z、C 15:36Z。
  - 從提交的壓縮檔，用本輪工具重算 formal-summary.json，除 `created` 外逐位相同。報告各表的數字也和摘要逐一核對過。
  - B1、B2 的完整原檔在本機 `build/stage50-external`，和 full-output-sha256.json 的 57,633／10,719 個雜湊全部相符。
  - C++ 等於 `af7c97a` 的全新 Release 研究建置，ctest 87/87。
- **建置來源更正：**
  - `build/stage50-ours` 實際是在 `8f0e6a9`（第49輪合併前）建置的。protocol 的 `ours_build` 與本頁原文寫「C++ 與 main af7c97a 相同」，這不正確；第1階段的 feasibility.json 寫的才是對的。
  - 兩者的 C++ 只差第49輪預設關閉的 `pyramid_rule`，沒有任何 arm 開啟它。
  - 用上面的新建置，重跑 A、C 組我們全部 arm 的 block 0，共 787 個 job（含穩定核心與淺金字塔）。首次呼叫的 matches 與 proposals 都和封存逐位相同；改一個分數的負控制有抓到。B1、B2 沒有重跑。
- **TDCM 論文 IoU 的角度方向錯誤（第三方發現，已修正）：**
  - **錯在哪裡**：`literature_score.tdcm_paper_iou` 用 `pose_corners(-angle)`，把兩個矩形都畫成鏡像。TDCM 的 `getIOU` 是呼叫 `rotated_rect(-r)`，等於我們的 `pose_corners(+angle)`。原測試把錯誤的方向寫成了預期值。
  - **修正**：紅燈 `8cf39a8` 照 TDCM 公式獨立重建角點，原程式算出 0.6534，正確值是 0.7027；修正在 `ac9f3e0`。
  - **影響範圍**：只影響 B1 的描述性論文指標（成功率、mIoU、成功樣本的誤差），以及開發選擇的論文距離。嚴格計數、假說、框、計數都不用這個 IoU。
  - **重算結果**：存在 [acceptance-b1-paper-iou-correction.json](stage50-literature-comparison/acceptance-b1-paper-iou-correction.json)，B1 表已換成重算值。勝負方向都不變。主要變化：
    - 我們：S2 成功率 82.6% → 82.8%，mIoU 0.735 → 0.747；S2.5 成功率 74.7% → 75.5%。
    - TDCM：mIoU S2 0.923 → 0.926，S2.5 0.886 → 0.890。
    - OpenCV：S2.5 成功率 63.5% → 65.5%。
  - **開發選擇不變**：修正後的論文距離 `test_py` 0.449、`test_single` 1.079（原為 0.448／1.079），仍選 `test_py`。
  - formal-summary.json 保留原值，不改寫；以本節和更正檔為準。
- **突變與補測試：**
  - 另做 28 個評分與判定規則的突變，每個都實際跑兩個測試檔。涵蓋：嚴格門檻、角度週期、配對、框的排序與 IoU、AP、MAE、中位數、H1(a)(b)、H3、H4、H5、B1 成功門檻。
  - 原測試下有 14 個存活：
    - 框匹配：沒有依分數排序、同一個 GT 框可被配兩次、IoU 0.5 門檻。
    - 數值：MAE 沒取絕對值、角度非有限值與零尺寸、偶數個值的中位數。
    - 判定規則：H1(a) 和 OpenCV 比較、只比完成全搜尋的 OpenCV、H1(b) 的角度條件、H3 的方向、H4 的平手與場景數、H5 的範圍。
    - B1 的成功門檻 IoU>0.1。
  - 新增 8 項測試（`c2d4d4b`）後，28 個全部被抓到。加上 IoU 測試，共 40 項通過。
- **第三方獨立重算：**
  - **方法**：subagent 只用 stdlib，照 protocol 文字自己實作評分（Hopcroft-Karp 最大配對、多邊形裁切的旋轉 IoU、COCO 101 點 AP），算完才看摘要。
  - **相符（15 項中的 14 項）**：A 組各家族、負控制、誤差中位數、與第44輪 48/48 相同、C 組各家族與負場景誤判、H4 時間、B1 S2 原尺寸嚴格率、B2 的 AP50 與 MAE。
  - **不符的那一項**：就是上面的 IoU 方向。
- **揭露補充（封存檔保留原文，以本節為準）：**
  1. **H1 判定程式在凍結後改過。** 原本只要沒有 GeCo2 輸出就回傳「待定」，即使 (a) 已經失敗。現在照 protocol 文字「(a)、(b)、(c) 都通過才通過」，所以 (a) 失敗就判否。這次修改有測試，但沒有列進 protocol 的 revisions。H1 判否依據的是凍結的文字，不是這次修改。
  2. **部分負場景的誤判數是下限。** 微小模板負場景有幾張的輸出數剛好等於上限 100：我們研究路徑 1 張（proc50114_distractor），穩定核心 3 張，OpenCV 1 張。TDCM 改編誤判較少的結論不受影響。
  3. **嚴格失敗多半是 180° 翻轉。** 微小模板正場景中，完整 GPU k3 的 66 個嚴格失敗都是角度不符，其中 31 個是 180° 翻轉：dst5_24 有 28 個，dst5_16_noise 有 3 個。這個模板在這個尺寸接近 180° 對稱，但 period 設為 360。照 protocol 仍算失敗，H3 不受影響。
  4. **GeCo2 有 149 個鍵缺失。** CPU 與 MPS 載入權重時都缺 149 個 `sam_mask` 鍵，CUDA 重跑時要一併確認。
  5. **兩處表格已更正。** 結論表的微小模板負場景列，原文漏了穩定核心的 349。C 組 OpenCV 在 8192² 原寫 2，和「超過 2.5 Mpx 的精度標 N/A」矛盾，已改為 N/A。
- **NOT RUN：** GeCo2 CUDA；TDCM、GeCo2、OpenCV worker 的重跑；B1、B2 我們 arm 的重跑；時間重測；ASan 與 Metal validation（本輪沒有 C++ 變更）。

## 2026-10-27 GeCo2 CUDA 補跑

在分支 `claude/round50-geco2-cuda`（從 main `3c4582f`）上補跑；尚未合併。事前登錄是 protocol 的第 7～10 條修訂，都在任何遠端量測之前提交（`15b4807`、`64e28d2`）。

**裝置的決定過程：**
- 使用者原本指定 WSL2 Ubuntu＋CUDA 12.8 toolkit。
- 遠端第一步 `wsl --install` 被工具的安全分類器擋下，沒有執行。
- 使用者隨後改成「直接用windows原生環境」。`wsl --list` 前後都顯示沒有安裝任何發佈版，也沒改任何 Windows 功能；WSL 腳本已移除，事前登錄另立一條修訂。

**環境**（全部紀錄在 [geco2-cuda-environment.json](stage50-literature-comparison/geco2-cuda-environment.json)）：
- **主機**：PTRI-PRO-4000，Windows 10.0.26300，Intel Core Ultra 9 285K（24 核），63.5 GiB，RTX PRO 4000 Blackwell 24 GB（sm_120）。
- **Driver**：596.71，安裝前後相同。CUDA 安裝檔只裝 toolkit 子套件，沒有 Display.Driver。
- **工具鏈**：VS Build Tools 2022 17.14.43（MSVC 19.44.35229），nvcc 12.8 V12.8.93，Python 3.12.10 venv，torch 2.7.1+cu128。其他套件版本與 Mac 相同。
- **安裝檔**：數位簽章都有效（Microsoft、NVIDIA）。CUDA network installer 的 MD5 與 NVIDIA 公布值相同。兩個安裝程式都回傳 0。
- **重開機旗標**：安裝前就已有 68 筆待改名的檔案（PendingFileRenameOperations），不是本次造成的；安裝後是 78 筆。沒有新增 CBS 或 Windows Update 的重開機旗標，也沒有安裝程式要求重開機。nvcc 與 kernel 編譯不重開機就能正常運作，所以沒有重開機。原本的停止條件把既有的待改名檔也算進去，會誤判；在執行 tools 段之前已改成只看安裝程式的 3010 與新增的旗標（`b0ccd17`）。
- **Kernel**：官方 Deformable-DETR ops 用 `-gencode=arch=compute_120,code=sm_120` 編譯，產出 `MultiScaleDeformableAttention.cp312-win_amd64.pyd`。0 個錯誤；245 個警告，多數是 MSVC 的 C4996 棄用警告。
- **GeCo2 原始碼**：`git diff` 是空的；未追蹤的檔案只有 install.sh 那一步複製到 `models/ops` 的檔案。Windows 編譯不需要改任何原始碼，所以沒有 protocol 偏離。
- **注意力 kernel**：Windows 版 torch 沒有編入 flash attention，SAM2 的注意力退回 math kernel。輸入是 float32，flash attention 只收 fp16／bf16，所以 Linux 上一樣用不到它。

**四項前置檢查**（全過才執行任何正式量測）：
1. **CUDA**：可用，capability 12.0，torch 的 arch list 含 sm_120。
2. **編譯的 kernel**：載入的是上面的 `.pyd`。隨機輸入下，CUDA forward 與 float64 參考實作的最大差 1.8e-7，門檻是 1e-4。
3. **單張影像**：controlled6 經 worker 得到 6 個框，數值全部有限。
4. **權重鍵**：缺失的鍵正好是 149 個 `sam_mask`，換兩個初始化 seed 時這 149 個都相同，代表是 counter_infer 的 MaskProcessor 從 SAM2 載入的，不是隨機初始化。多出的 8 個鍵是訓練用輔助頭（`class_embed_aux` 2、`bbox_embed_aux` 6），和 Mac 相同。合併驗收時留下的「149 個鍵缺失」疑問到此確認是預期行為。

**B2 真偽關卡：** AP 46.84、AP50 76.08，論文 46.81、75.89，關卡通過。計數 MAE 7.69，論文 7.64。

**A／C 組的外接框**（GeCo2 每個正場景拿前 3 個真值物件的外接框當範例；IoU ≥ 0.5）：

| 家族（GT） | GeCo2 CUDA 對中（多報） | 我們 完整GPU k3 對中（多報） |
|---|---|---|
| A 第35輪郵票（294） | **294**（0） | 180（0） |
| A 第37輪（667） | 667（1） | 667（0） |
| A test-bed Blender（54） | 54（0） | 54（0） |
| C 遮擋／干擾／接觸（32） | **30**（10） | 22（4） |
| C 照明 17 場景（484） | 482（60） | **484**（0） |
| C 8192²／512 物件（512） | 512（0） | 512（0） |
| C 微小模板正場景（2352） | 2352（1） | 2352（203） |
| **A 合計（1015）** | **1015**（1） | 901（0） |
| **C 合計（3380）** | 3376（71） | 3370（207） |

- **事後診斷**（未事前登錄，只作描述）：照明組多報的 60 個框中，28 個落在場景的色卡上，每張合成場景 2 個；32 個和真值物件的 IoU ≥ 0.1，屬於重複框或位置偏移。（合併驗收更正：原寫「32 個 IoU 0.1–0.5」。其中 30 個在 0.1–0.5；另 2 個在 syn_red_spot_warm 與 syn_red_wb_warm，和已被分數更高的框配走的同一物件 IoU 約 0.82，是重複框。）難題組的 10 個裡，8 個在干擾物上，2 個是部分重疊。
- **這只比外接框。** 我們的位姿判準（2 px／1°／2%）結果見上面的 A、C 組；GeCo2 不輸出角度，無法比較。
- **A 組的差距全在第35輪郵票模板**，是本專案已知的弱點；其他 A 場景兩邊都全中。

**判定的變化：**
- H1(c) 失敗（GeCo2 1015 對 901）。H1 原本就因為 (a) 判否，結論不變。
- H2 從「待定」改為「是」（AP50 0.752 對 0.110）。
- H3、H4、H5 與本專案、TDCM、OpenCV 的所有數字都沒變（見驗證表）。

**時間**（實驗室硬體，只並列，不與 M4 Pro 換算）：
- B2 每張中位數 288 ms，整批 7.1 分鐘。
- A／C 每案單次呼叫的中位數 292 ms，範圍 198–1,638 ms。
- 每個 worker 開始前，GPU 使用率都是 0%；CPU 負載 0–61%，只記錄、不作門檻。

**操作紀錄與揭露：**
- **長程序的啟動方式**：一律用 Win32_Process.Create 啟動，不受 SSH 斷線影響。89 個 worker 沒有一個重試（attempts.jsonl 89 筆，每案 1 次）。
- **卡住的解析程序**：用 stdin 在遠端解析 PowerShell 語法時，`powershell -EncodedCommand` 等不到輸入結尾而卡住（PID 18308，沒做任何事）。工具第一次結束它被安全分類器擋下，使用者授權後才結束。
- **我的等待迴圈判斷錯誤**：前置檢查實際 10 秒就跑完，但等待迴圈用 `tasklist` 判斷程序結束時出錯，誤報為「25 分鐘還在跑」。結果不受影響；之後改看執行器寫出的完成標記檔。
- **傳輸包**：macOS 的 tar 多帶了 `._*` AppleDouble 檔。執行器只讀 `jobs.json` 列出的路徑，不受影響。
- **空檔案沒帶回**：0 bytes 的 `geco2-git-diff.txt` 沒有隨結果 tar 帶回；它在遠端的大小（0）已用 `Get-Item` 確認。
- **補跑工具**：`literature_geco2_cuda_setup.ps1`、`literature_geco2_cuda_precheck.py`、`literature_geco2_cuda_run.py`，以及 `literature_compare.py` 的 `export-geco2-cuda`／`ingest-geco2-cuda`。worker 改成記錄載入的 kernel 檔案，並在 Linux／Windows 正確回報峰值記憶體。測試共 43 項通過。
- **NOT RUN**：本專案、TDCM、OpenCV 都沒有重跑（驗收已確認 block 0 逐位相同）；GeCo2 沒有在負控制或 B1 上跑；沒有在第二張 GPU 上重複。

### 補跑的合併驗收（2026-10-27）

主 session 以 reviewer 身分，在補跑分支 head `5ed7804` 的乾淨 worktree 重新驗收（前後工作樹都乾淨）。判定都沒有變。

- **重跑 gate：** 單元測試 45 項與索引檢查（Python 3.12、3.14）通過；archive.json 的 37 個檔大小與 SHA-256 都相符，最大 42 MB。補跑沒有改到任何 C++。
- **封存重算：** 解開全部 raw-results（含 `raw-results-geco2-cuda.tar.gz`），用 `ingest-geco2-cuda` 匯入 89 列，再以 `--geco2-accuracy geco2_cuda` 重算，結果與 `formal-summary-geco2-cuda.json` 除 `created` 外逐位相同。報告引用的 H1(c)、H2、B2、關卡與 A／C 外接框數字也都和摘要核對過。
- **判定程式：** `hypotheses()` 在補跑中沒有修改，H1(c) 與 H2 用的是凍結時的程式。
- **突變與補測試：** 37 個突變（先前 28 個、修正後 IoU 的反向、以及 H1(c)、H2、ingest、export 共 8 個），原測試下有 4 個存活：H2 的 AP50 方向、計數 MAE 備案的方向、忽略框是否存在、H1(c) 的方向。新增 1 項測試（`ac153ee`）後 37 個全部被抓到，共 44＋2 項通過。
- **第三方獨立重算**（stdlib、自寫評分、算完才看摘要）：
  - **完全相符**：B2 的 AP50 0.7517、precision 0.783、recall 0.817、MAE 7.69、RMSE 40.0；A 組 1015（多報 1）對 901（多報 0）；C 組各家族；三種裝置的一致性（97／100、345／1190，沒有一張逐框相同）。
  - **事後診斷的區間寫法不精確**：已在上面更正。
  - **「落在色卡上」**：metadata 沒有色卡資訊，第三方無法重算。主 session 依 `testbed_generate.py` 的 `REFERENCE_CARD`（x 24–312、y 16–112）核對，28 個框的中心都在約 (70, 60) 與 (260, 60)，落在卡內。
  - **完整性**：送到實驗室的 88 張圖與 cases.json 的來源檔 SHA-256 相同；B2 的 id 與範例框和 items.json 一致；CUDA manifest 89 列全部 ok、attempt 都是 1。
- **結論表補充**：遮擋與照明兩列加上 GeCo2 的外接框結果（只比外接框，並標出多報數）。
- **NOT RUN：** GeCo2 在第二張 GPU 上的重複；實驗室端的重新執行（只用封存的輸出重算）。
