# 第50輪第1階段：與 2024–2026 文獻方法比較的可行性

日期：2026-10-26。機器：Apple M4 Pro／48 GiB，macOS 27。分支 `claude/round50-literature-comparison-e95d80`，基底 main `8f0e6a9`。結構化紀錄見 [feasibility.json](feasibility.json)，冒煙原始摘要見 [phase1-smoke-summary.json](phase1-smoke-summary.json)，GeCo2 兩次失敗嘗試見 [phase1-smoke-attempt1-geco2-scipy.json](phase1-smoke-attempt1-geco2-scipy.json) 與 [phase1-smoke-attempt2-geco2-cpu.json](phase1-smoke-attempt2-geco2-cpu.json)，下載雜湊見 [phase1-download-sha256.txt](phase1-download-sha256.txt)。

**本頁只回答「能不能跑、要怎麼跑」，沒有任何正式計時。** 冒煙都是單一新程序、沒有暖機協定、機器與其他 session 共用；跑第三方方法時（10:11–10:19Z）另一個 session 的無頭 Chromium 佔用約 9 核，CPU worker 也限制在 4 執行緒。表中的毫秒數不能當作速度結果引用。

## 結論

| 方法 | 能不能跑 | 裝置 | 冒煙 | 主要限制 |
|---|---|---|---|---|
| 穩定核心 `fpm_cli` | 可以 | CPU（UDOT） | 兩組都通過 | 固定尺度；128 物件場景（尺度 0.72–1.25）只找到 48 個、嚴格 13 個 |
| 研究路徑（混合、完整 GPU peak2/fine4，k0／k3） | 可以 | Metal／CPU＋Metal | 兩組都通過；128 物件四個 arm 都 128/128 嚴格命中、0 誤判 | — |
| OpenCV `matchTemplate`＋暴力旋轉 | 可以 | CPU | 兩組都通過 | 4096² 每張分數圖約 1.1 s，1°×3 尺度全搜尋外推約 20 分鐘/張 |
| PoseMatch-TDCM | 可以（**有官方權重**） | M4 Pro CPU 與 MPS，兩者輸出相同 | 兩組都跑通 | 官方程式只輸出單一目標；模板須 18–72 px；兩支官方腳本在示範圖上角度不同；4096² 原尺寸 CPU 峰值 RSS 約 19 GB |
| GeCo2 | 可以 | M4 Pro CPU 與 MPS（可變形注意力改用官方 PyTorch 參考實作） | 兩組都跑通（前兩次失敗已保留） | 純 CPU 需繞過上游初始化缺陷；scipy 須升到 1.16.3；CPU 與 MPS 輸出不完全相同；範例框必須在影像內，空場景與錯誤模板控制對它沒有定義 |

FSCD-147 的官方標註下載不到（兩個 Counting-DETR 連結都 404，GeCo README 的副本需要登入 Google），所以 B2 目前只能做計數，AP50（H2）要等你決定資料來源。

## 冒煙結果

兩組影像：`tdcm_demo` 是 TDCM repo 自帶的 `res/image.png`（153×150）＋`res/template.png`（31×33），GT 是作者在 `test.py` 手打的位姿（中心 45,95、角度 −50°、尺度 0.89/0.90×36）；`many128` 是第37輪 4096×4096 場景，128 個同一模板（140×102）的複本，GT 由產生器獨立給出。所有方法拿同一張影像與同一個模板檔；GeCo2 改用影像內 GT 實例的外接框當範例框（many128 用前 3 個，示範圖只有 1 個）。評分一律由 `literature_score.py` 做。

| 組 | 方法 | 輸出數 | 嚴格 TP/FP/FN（2 px／1°／2%） | 寬鬆命中（半模板半徑） | AP50（外接框） | 時間（ms，僅供參考） |
|---|---|---|---|---|---|---|
| tdcm_demo | 穩定核心 | 1 | 0/1/1 | 1 | 1.000 | match 32.8 |
| tdcm_demo | 混合 k0／k3 | 1 | 0/1/1 | 1 | 1.000 | 呼叫 18.2／16.8 |
| tdcm_demo | 完整 GPU k0／k3 | 1 | 0/1/1 | 1 | 1.000 | 呼叫 8.4／6.9 |
| tdcm_demo | OpenCV | 1 | 0/1/1 | 1 | 1.000 | 搜尋 362.9（360 張圖） |
| many128 | 穩定核心 | 48 | 13/35/115 | 48 | 0.376 | match 1364.9 |
| many128 | 混合 k0／k3 | 128 | 128/0/0 | 128 | 1.000 | 呼叫 862.0／564.6 |
| many128 | 完整 GPU k0／k3 | 128 | 128/0/0 | 128 | 1.000 | 呼叫 668.4／388.1 |
| many128 | OpenCV（只跑 6/1080 張圖） | 4 | 0/4/128 | 4 | 0.040 | 每張圖 1096.5 |
| tdcm_demo | TDCM `test_single`（CPU／MPS 相同） | 1 | 0/1/1 | 1 | 1.000 | 推論 46.0（CPU）／150.3（MPS） |
| tdcm_demo | TDCM `test_py`（CPU／MPS 相同） | 1 | 0/1/1 | 1 | 1.000 | 推論 76.3／147.1 |
| tdcm_demo | GeCo2 CPU／MPS | 7／8 | — | — | 1.000 | forward 2589.6／3079.5 |
| many128 | TDCM `test_single`（CPU／MPS） | 1 | 0/1/128 | 1 | 0.010 | 推論 43782.5／1262.8 |
| many128 | TDCM `test_py`（CPU／MPS） | 1 | 0/1/128 | 1 | 0.010 | 推論 33903.7／1266.8 |
| many128 | GeCo2 CPU／MPS | 128／128 | — | — | 1.000／1.000 | forward 4109.5／4211.9 |

- `tdcm_demo` 所有方法都偏離手打 GT 約 1.7–2.5 px，尺度也差 3.25%／1.85%（GT 寬 0.89×36=32.0 px、模板只有 31 px）：手打 GT 本身不精確，不是方法誤差，所以嚴格判定全部不通過、寬鬆全部命中。
- 角度慣例：TDCM GT −50°（`cv2.getRotationMatrix2D`，畫面上逆時針為正），我們輸出 −49.0°，兩邊同號，不需要轉換。
- `many128` 研究路徑匹配誤差中位數：位置 0.029 px、角度 0.014°、尺寸 0.16%。記憶體分開記：混合峰值 RSS 約 240 MB；完整 GPU RSS 約 94 MB，另有命名 GPU 配置約 147 MB（兩者不能相加）。
- **TDCM 在示範圖上，兩支官方腳本結果不同**：作者評估腳本 `test_single.py`（RGB 輸入，與訓練一致）得到 −78.30°，離 GT 1.19 px、28.3°；精修分數 0.899 沒超過 0.9 的門檻，所以角度沒有被改。README 指定的範例 `test.py`（BGR 輸入、`cv2.resize`、彩色精修、最後再一次 `refine_angle`）得到 −49.71°，離 GT 0.35 px、0.29°。只有一張圖，無法判斷哪支腳本才對。
- **TDCM 在 many128 只輸出一個目標**：離最近 GT 16.6 px，角度差約 180°（模板對稱），`scale_y` 2.04，卡在尺度頭的上限；這些物件 100–175 px，是 36 px 的 3–5 倍，超出這份權重的範圍。記憶體：CPU 峰值 RSS 19.1 GB；MPS 的 RSS 1.4 GB，另有驅動配置 13.1 GB。8192² 原尺寸約需 4 倍，48 GiB 放不下。
- **GeCo2 在 many128**：CPU 與 MPS 都輸出 128 個框，IoU≥0.5 下 128/128、AP50 1.000、precision 1.000；框中心誤差中位數 2.14 px（CPU）／2.63 px（MPS）。輸入縮放 0.25（4096→1024）。記憶體：CPU 峰值 RSS 10.7 GB；MPS 的 RSS 2.4 GB，另有驅動配置 9.8 GB。在示範圖上 CPU 7 個框、MPS 8 個框，都找到 GT 那一個；其他框無法判定，因為作者的 GT 只標了一個物件。
- **GeCo2 權重載入完整**：checkpoint 共 475 個鍵。缺的只有 `sam_mask.*`（149 個），與官方程式一樣由 SAM2 checkpoint 初始化；多的只有訓練用輔助頭 `class_embed_aux`／`bbox_embed_aux`（8 個）。推論用到的訓練過模組全部有載入。

## TDCM 權重

**權重存在**，不需要重新訓練就能做比較：

- 位置：repo 裡的 `dict/model.pth`（在 git 中，不是 release 附件），3,224,065 bytes，SHA256 `4bbd7178…68dc02`，MIT 授權（repo LICENSE）。
- 封存名稱 `model_epoch_30_soft_`。載入後模型有 794,333 個參數（另有 964 個 buffer 元素）。**README／論文寫 3.07M 參數，這份權重對不上**，原因不明，第2階段照實揭露。
- 只有這一份：模板固定縮成 36×36，尺度輸出頭是 `sigmoid×1.5+0.5`，只能表示 0.5–2.0 倍（相對 36 px）。論文表3的 12×12、28×52、124×124 模板模型，以及 S2.5（0.4–2.5）所需的更寬尺度範圍都沒有釋出權重。

因此原本的兩個選項改成針對 H3：

1. **改寫 H3**：只用官方 36×36 權重，測「小物件（12–18 px）落在它訓練尺度之外」的情形，並揭露這不是論文表3的設定。
2. **重新訓練 12×12 模型**：`train.py` 在 repo 中，需要 COCO train2017（約 18 GB，**尚未下載**）與未公開的 GPU 時間（論文沒寫訓練時數與 epoch；這份權重是 epoch 30）。
3. **剔除 H3** 並揭露原因。

## 已下載項目

全部放在 `build/stage50-external/data/` 各自的新目錄，沒有提交進 git。

| 項目 | 來源 | 大小（bytes） | SHA256 前 12 碼 | 完整性 | 授權 |
|---|---|---|---|---|---|
| FSC-147 影像 `FSC147_384_V2.zip` | 作者 README 的 Google Drive | 1,533,803,725 | `4f9eff24c39f` | 斷線後續傳；`unzip -t` 無 CRC 錯誤；6,146 張 | 程式碼 MIT；影像授權作者未說明 |
| FSC-147 標註（3 檔） | GitHub `LearningToCountEverything@00ea188` | 45,813,986 合計 | 見 JSON | git blob SHA-1 與 tree 完全一致 | MIT |
| COCO val2017 影像 | images.cocodataset.org | 815,585,330 | `4f7e2ccb2866` | 續傳；大小＝Content-Length；CRC 通過；5,000 張 | Flickr 使用條款 |
| COCO 2017 標註 | images.cocodataset.org | 252,907,541 | `113a836d9019` | 續傳；大小＝Content-Length；CRC 通過 | CC BY 4.0 |
| GeCo2 `GECO2FSCD.pth`（FSCD 基準權重） | 作者 HF 資產庫 `jerpelhan/geco2-assets@ed3c8ff` | 301,292,745 | `945c5d838a05` | 與 HF 公布的 LFS oid 相同 | HF 資產庫未標授權，只作研究用、不轉散 |
| GeCo2 `CNTQG_multitrain_ca44.pth`（demo 權重） | 同上 | 350,785,678 | `7841dcfd2d01` | 與 LFS oid 相同 | 同上 |
| SAM2 `sam2_hiera_base_plus.pt` | dl.fbaipublicfiles.com | 323,493,298 | `d0bb7f236400` | 續傳；大小＝Content-Length | Apache-2.0 |

另外 clone 了兩個 repo（各自固定 commit、工作樹乾淨）：TDCM `a6adc3a`（2025-10-01，MIT）、GeCo2 `67d27e8`（2026-10-12，根目錄 MIT；`Deformable-DETR/` 與 `sam2/` 為 Apache-2.0，`sam2/LICENSE_cctorch` 為 BSD-3）。

**沒有下載**：FSCD-147 標註（見下節）、COCO train2017（只在 TDCM 重訓時需要）、GeCo2 Google Drive 版權重（已用作者 HF 版，未另外比對 Drive 版）。

## B1：TDCM 論文的 COCO 合成協定

- **S2 清單**：repo 只附 `data/S2/val.csv`（9,450 列、2,557 張 val2017；`test_single.py` 用前 1,000 個樣本、種子 42）。
- **重產檢查**：用 `filterData.py` 的規則（x、y 取 floor，w、h 取 ceil，w/36、h/36 都在 0.5–2）從 `instances_val2017.json` 重產得到 9,459 列：順序相同，但多出 9 列，找不到共同原因（不是 crowd、不是尺度邊界、也不是影像邊緣）。第2階段 S2 直接用官方清單，並揭露這 0.1% 差異。
- **S1.5／S2.5**：同一規則可重產，分別 2,469 列與 13,567 列；S2.5 有一部分超出官方權重的尺度範圍。
- **S1 沒有定義**：照字面 `scale_range=(1,1)` 只剩 3 個框，論文的 S1 顯然不是這樣產生的，需要先定義。
- **樣本產生**：`utils/get_imgR.rotate_crop`（[−180°,180°) 均勻角度、含整個框的 224×224 裁切、最多重試 10 次）。固定 numpy 版本與種子時是確定性的，所以應該**用作者的程式產生一次、凍結成影像＋GT 檔**，所有方法都讀同一批檔案。
- **尺度語意**：GT `scale_x=w/36`、`scale_y=h/36`，只要 w≠h 就是非等向；我們只搜尋等向尺度清單，x／y 尺度誤差要分開報。
- **多目標協定**（320×320、4×4 格、貼上 36×36 模板）沒有釋出程式，只能照論文文字重做。

## B2：FSCD-147

- **FSC-147 已就緒**：測試集 1,190 張（高 384 px），每張 3–5 個範例框，全部在影像範圍內；範例框短邊中位數 46 px、10 百分位 19 px、最小 2 px，2.8% 小於 12 px；每張 8–3,701 個點，合計 78,836。
- **格式**：FSC-147 範例框是四角點多邊形，FSCD-147 會是 COCO instances JSON（xywh）；評分程式吃 x0,y0,x1,y1，兩者都能直接轉換。
- **FSCD-147 的來源要你決定**（研究用途、不可轉散、須引用）：
  1. 第三方 HF 鏡像 `ChipmunkG4/FSCD-147_FSCD-LVIS_temp`：`FSCD_147.zip` 1,552,158,410 bytes，LFS SHA256 `d40ab222…6c63`，2026-10-15 上傳，沒有 README 或授權。若採用，驗證方式是：影像與官方 FSC-147 zip 逐檔相同，並用官方權重重現 GeCo2 論文的 FSCD-147 數字。
  2. OpenDataLab 的 FSCD-LVIS 原始壓縮檔（據稱內含 `FSCD_147.zip`）：需要帳號，只能由你下載。
  3. 寫信向 Counting-DETR 作者要新連結。
  4. 不做 AP50，只做計數與以點為準的定位。

## GeCo2 的協定事項

- 官方門檻有兩個：FSCD 評估（`inference.py`）保留 `box_v > 0.11×max`，Gradio demo 預設 0.33。建議登錄 `GECO2FSCD.pth`＋0.11 為基準設定。
- 輸入會把長邊縮到 1024 px，且讓範例框平均不超過 80 px：4096² 與 8K 場景在偵測前會縮小 4–8 倍，這是它的設計，照實揭露。
- 範例框必須在同一張影像內，沒有獨立模板檔，所以空場景與錯誤模板兩種負控制對 GeCo2 沒有定義（H5 只比較以模板為輸入的方法）。A／C 組建議用前 3 個 GT 實例的外接框當範例框，並事前宣告這 3 個是否計分。
- 官方 `MultiScaleDeformableAttention` 只有 CUDA 版；在 Mac 上用同一檔案裡的 `ms_deform_attn_core_pytorch`。同一個參考實作在 CPU 與 MPS 的輸出就已不同（示範圖 7 對 8 個框），所以精度要固定在一個裝置上報，最好是實驗室 GPU 上有編譯 kernel 的 CUDA；GeCo2 的時間也在 CUDA 上量，只與 M4 Pro 並列，不算倍數。
- 未修改的官方程式**無法純 CPU 執行**：`models/query_generator.py` 的 `C_base` 只在形狀張量換裝置時才計算可變形注意力的參考點，CPU 上不會換裝置，所以缺 `reference_points`。worker 在 CPU 上先用該類別自己的 `get_reference_points`、同樣的參數設好這三組張量；MPS 不需要。
- `req.txt` 固定的 scipy 1.15.2 在 macOS 27 無法載入（dyld 拒絕 `_spropack` 的 `__thread_bss` 區段），GeCo2 環境改用 scipy 1.16.3。
- 官方 AP 評估需要 detectron2；本輪統一用 `literature_score.py`。

## 本輪新增的工具

- `cpp-core/tools/literature_score.py`：純 stdlib 的獨立評分程式（嚴格位姿、外接框 IoU≥0.5、COCO 101 點 AP50、計數 MAE／RMSE、空場景誤判）。測試 `cpp-core/tests/test_literature_score.py` 12 項通過，其中一項確認角點慣例與第37輪產生器的 GT 逐點相同。
- `cpp-core/tools/literature_smoke.py`：冒煙驅動程式。每個子程序啟動前檢查計時鎖，有鎖就拒絕啟動；GT 只留在父程序。
- `cpp-core/tools/literature_opencv_worker.py`：OpenCV 基準（遮罩 TM_CCOEFF_NORMED、1° 網格、與 fpm 同義的旋轉 NMS）。
- `cpp-core/tools/literature_tdcm_worker.py`：照作者腳本寫的包裝器，`--pipeline test_single`（評估腳本）與 `--pipeline test_py`（README 範例）兩種。
- `cpp-core/tools/literature_geco2_worker.py`：`demo_gradio.py` 的前處理加 `inference.py` 的後處理，記錄權重載入的缺漏鍵前綴、偏離項目與記憶體。
- 兩個 worker 都在各自的環境用 `python -I` 執行，不從網路下載任何東西。

## 事件紀錄（全部保留）

1. **網路中斷**（約當地 11:15–17:39）：FSC-147、COCO 影像與標註、GECO2FSCD、SAM2 的下載卡住或失敗，兩個環境安裝因 DNS 失敗。處理：停掉卡住的程序，用 gdown resume 與 `curl -C -` 續傳，再以 Content-Length、CRC 與公布的雜湊核對；環境重新安裝。
2. **計時鎖重疊**：第49輪 `dev_calibrate` 鎖（09:48:35Z 起）存在時，我在 09:50:07Z 跑了一次 runner 冒煙（construct 約 0.4 s，整個程序不到 1 s）。已通知第49輪 session，並讓 `literature_smoke.py` 有鎖就拒絕啟動。
3. **權限阻擋**：執行 TDCM／GeCo2 的 repo 程式碼被本 session 的自動模式分類器擋下，我沒有繞過。使用者之後同意執行（只限 `build/stage50-external` 下的 clone 與各自環境）；實際執行等第49輪正式計時鎖（09:57:23Z–10:11:19Z）解除後才開始。
4. **GeCo2 第1次（四次執行全部失敗）**：`utils.data` 匯入 `scipy.ndimage` 時，scipy 1.15.2 的 `_spropack` 被 dyld 拒絕載入。處理：只在 GeCo2 環境升級到 scipy 1.16.3。
5. **GeCo2 第2次（CPU 兩次失敗）**：`'C_base' object has no attribute 'reference_points'`（上游缺陷，見上節）；同一次的 MPS 兩次都成功。處理：worker 在 CPU 上先設好參考點，再全部重跑。

## 磁碟

`build/stage50-external` 共約 11 GB（其中解壓出的 FSC-147 密度圖 4.7 GB 用不到，可刪、zip 保留），環境約 1.3 GB，`build/stage50-ours` 35 MB。完成後剩約 95 GiB。

## 建議的第2階段調整

1. TDCM 用官方 36×36 權重，不重訓；揭露 0.80M 與 3.07M 參數不符。
2. H3 依上面三個選項擇一（建議選 1：改寫成官方權重下的小物件測試）。
3. A／C 多物件場景的 TDCM 需要我們自己寫的分數圖多峰值擷取，標成「TDCM 改編」；或 TDCM 只參加單目標組。
4. TDCM 以 `test_single.py`（評估腳本，RGB 與訓練一致）為主要流程，`test_py` 為揭露的次要流程，因為兩者在示範圖上不一致。
5. TDCM 推論前先把影像與模板縮放到模板落在 18–72 px（屬改編，需揭露）；8K 場景要分塊或縮小，原尺寸 4096² 在 CPU 上就要約 19 GB RSS。
6. GeCo2 的 CPU 與 MPS 輸出不同：精度固定在一個裝置（最好是有編譯 kernel 的 CUDA），另一個只當交叉檢查。
7. B1 跑兩種模板：原尺寸裁切，以及 TDCM 實際看到的 36×36 縮圖（我們在該設定的尺度範圍做等向搜尋）；x／y 尺度誤差分開報。
8. B1 的 S2 用官方清單、S1.5／S2.5 用官方規則、S1 先定義；1,000 個樣本用作者的 `rotate_crop`（種子 42）產生一次後凍結。
9. B2 先決定 FSCD-147 來源。
10. GeCo2 登錄 `GECO2FSCD.pth`＋門檻 0.11；負控制對它標 N/A；Mac 精度要先對照 CUDA kernel。
11. OpenCV 基準在 B1 與小場景完整跑，大場景外推或設上限。
12. 穩定核心只在固定尺度場景報告。
13. 第49輪合併後，把它的新選項加成一個 arm。

## 重現

```bash
X=/Users/Ayueh/Fastest_Image_Pattern_Matching/build
cmake -S cpp-core -B $X/stage50-ours -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_GUI=ON -DFPM_BUILD_EXPERIMENTS=ON -DFPM_WITH_OPENCV=ON
cmake --build $X/stage50-ours -j6
python-port/venv/bin/python -m unittest cpp-core/tests/test_literature_score.py
python-port/venv/bin/python -I cpp-core/tools/literature_smoke.py --external $X/stage50-external --build-root $X \
  --ours-build $X/stage50-ours --output-dir $X/stage50-external/smoke \
  --lock-root "$HOME/Library/Application Support/Claude/side-session-notes" \
  --methods ours_core ours_hybrid_k0 ours_hybrid_k3 ours_fullgpu_k0 ours_fullgpu_k3 opencv
python-port/venv/bin/python -I cpp-core/tools/literature_smoke.py --external $X/stage50-external --build-root $X \
  --ours-build $X/stage50-ours --output-dir $X/stage50-external/smoke \
  --lock-root "$HOME/Library/Application Support/Claude/side-session-notes" \
  --methods tdcm_cpu tdcm_mps tdcm_testpy_cpu tdcm_testpy_mps geco2_cpu geco2_mps
```

GeCo2 需要先把 SAM2 checkpoint 連結到 `$X/stage50-external/torch-home/hub/checkpoints/`（驅動程式會先核對 SHA256 再建立連結），環境裡的 scipy 要是 1.16.3。
