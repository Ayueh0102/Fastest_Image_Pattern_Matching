# Stage 35 真實 RGB 資料來源與使用方式：VisA 膠囊子集

本資料用來檢查「真實畫面內有多個相似物件，其中一個有局部變色」的流程。它提供原始 RGB 圖、官方異常類別與像素遮罩。這是小型研究子集，不是完整 VisA 基準，也沒有每顆物件的位置框與姿態標註。

本頁說明資料取得與評估邊界；不報算法速度或準確度。實驗結果應另附正式執行紀錄。

## 來源與授權

- 作者與資料集：Yang Zou、Jongheon Jeong、Latha Pemula、Dongqing Zhang、Onkar Dabeer；Amazon Science；VisA。
- 論文：[SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation，ECCV 2022](https://arxiv.org/abs/2207.14315)。
- 官方程式與資料說明：[amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)。本輪固定作者版本 `2a692ab575001cbde74d402d897a7286086c6199`。
- 官方完整資料檔：[VisA_20220922.tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)，大小為 1,929,840,640 bytes。本輪不下載整包。
- 逐檔原圖來源：[imaadd05/visa-anomaly-detection 公開鏡像](https://huggingface.co/datasets/imaadd05/visa-anomaly-detection)，固定 revision `912ae1aa35ac0791e64524a42342850af1605363`。
- **資料授權為 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**。官方 README 的資料授權與鏡像 `LICENSE-DATASET` 一致；官方程式的 Apache-2.0 與資料授權分開看待。

原始 JPG 與官方多類遮罩保留原始位元組，不改色、不重取樣。預覽圖、二值評估遮罩是本專案衍生產物，來源與轉換方式列於下文。對外使用資料或衍生圖時須保留作者、資料集名稱、授權連結與修改說明。

## 固定 13 張的研究協定

資料在正式 benchmark 前由 12 張調整為 13 張：增加正常圖 `050` 作為校準資料，讓模板來源 `000` 完全退出閾值擬合。

| 用途 | 正常圖 ID | 異常圖 ID | 可用於閾值擬合 |
|---|---|---|---|
| 模板來源與 self-match 檢查 | 000 | — | 否 |
| 正常閾值校準 | 050、200、400 | — | 是 |
| Pilot 異常觀察 | — | 030、036、042 | 否（正式runner不使用異常標籤擬合閾值） |
| Heldout 評估 | 100、300、500 | 033、039、044 | 否 |

這裡的 ID 分別對應 `capsules/Data/Images/Normal/<ID>.JPG` 與 `capsules/Data/Images/Anomaly/<ID>.JPG`。13 張均為 **1500×1000 RGB**，原圖總共 3,207,449 bytes，約 3.06 MiB。

`000` 在 JSON 中保留 `split="pilot"` 以維持載入相容性，但同時標記 `role="template_self_check"`、`self_match=true`、`threshold_fit_eligible=false`。**消費資料的程式應依 role 與 eligibility 排除 self-match，不能只憑 split 把全部 pilot 納入校準。**

校準與 heldout 的圖片 ID 不重複；這不是實體膠囊身分互斥的切分，也不保證拍攝序列獨立。同一收集流程可能包含相近或相關視角，因此本子集不能支持跨工廠、跨相機或跨物體泛化的結論。

最初 12 張的 manifest 已按原始位元組保存在本機 `manifest-initial-12.json`，SHA256 為 `aae0ee8d8e47edf40e9f27efa8005b0af772319f895b6e37edbe5b61967f4c0e`。這份歷史檔不再用來決定正式 benchmark 切分；從空資料夾重新取得時只產生目前的 13 張協定。

## 標註實際代表什麼

官方 [capsules 類別對照](https://github.com/amazon-science/spot-diff/blob/2a692ab575001cbde74d402d897a7286086c6199/utils/id2class.py) 為：0 背景、1 Bubble、2 Discolor、3 Scratch、4 Leak、5 Misshape。

本輪六張異常圖在官方 `image_anno.csv` 中均標為 **`bubble,discolor`**。它們不是純色差缺陷；原始多類遮罩中的值 1 與值 2 都保留。局部色差評估只取 **原遮罩等於 2** 的像素，不能把所有非零像素都說成色差。

| 異常圖 ID | Bubble 像素數 | Discolor 像素數 |
|---|---:|---:|
| 030 | 31 | 154 |
| 036 | 47 | 300 |
| 042 | 30 | 206 |
| 033 | 35 | 173 |
| 039 | 34 | 175 |
| 044 | 28 | 143 |

變色區域是局部小色斑，面積僅 143–300 像素；它不代表「整顆膠囊換成另一個顏色」。資料也沒有完整物件框、中心或姿態，故不能直接計算全場景所有膠囊的搜尋 recall。

每張資料的 `evaluation_binary_mask` 已輸出單通道 PNG：原始 mask 的 ID 2 轉為 255，其他像素轉為 0；正常圖為全零。JSON 保留尺寸、轉換規則、前景像素數與 SHA256。這些檔案只供最後評估，不得用來產生、裁切或篩選搜尋候選，再把結果宣稱為全圖搜尋成功。

## 模板選擇與前景遮罩的區別

模板從正常 pilot 圖 `Normal/000.JPG` 手動選擇一顆完整直立膠囊，原圖座標為 `bbox_xywh=[878,25,112,245]`，使用零起點半開區間。模板裁切尺寸為 112×245。此選擇未使用任何異常圖的標註。

資料準備器不製作算法的 template foreground mask。實驗 runner 可從上述正常模板的 RGB 像素建立固定前景遮罩，但須另外記錄規則、尺寸與 hash。**模板前景遮罩與 query 的異常評估遮罩用途不同，不能混用。**

## 取得與重播

從 repository 根目錄執行；基本需求為 Python 3 與 Pillow。加上預覽時另需 matplotlib、NumPy。

```bash
python3 cpp-core/tools/prepare_stage35_visa.py --preview
```

預設資料目錄為 `build/stage35-real-rgb/data/visa/`，已受 repository 的 `build/` 忽略規則保護。原始圖不逐檔加入程式碼目錄；本輪可重現封存另收錄這個小型CC BY 4.0子集及歸屬說明。準備器使用固定公開來源、固定 revision 與固定圖片 ID，沒有登入、訓練或模型依賴。

資料已存在時，可以完全離線重新驗證並產生 manifest：

```bash
python3 cpp-core/tools/prepare_stage35_visa.py --offline --preview
```

主要輸出：

| 檔案或資料夾 | 用途 |
|---|---|
| `manifest.json` | 目前 13 張協定、來源、授權、hash、切分、模板座標、評估限制 |
| `manifest-initial-12.json` | 原始 12 張歷史協定；僅在既有 12 張 manifest 升級時保留 |
| `capsules/Data/Images/` | 13 張未修改的原始 RGB 圖 |
| `capsules/Data/Masks/Anomaly/` | 全部 100 張官方膠囊多類遮罩，供來源一致性檢查 |
| `derived/evaluation_discolor_masks/` | 13 張評估專用二值遮罩，不能作搜尋輸入 |
| `provenance/` | 固定來源清單、官方類別定義、授權文字、官方資料的部分下載證據 |
| `contact_sheet.png` | 全部 13 張圖的縮小預覽，標明校準、heldout 與 self-check 角色 |
| `template_reference.png` | 正常圖上標示手動模板框的檢視圖 |
| `anomaly_review.png` | 明確標為 GT-conditioned 的局部缺陷檢視圖，只供人工 review |

所有預覽都由原圖繪製，不替代算法輸入，也不納入算法計時。

## 已驗證與尚未驗證

已完成的資料檢查：

- 13 張原圖的 SHA256 與檔案大小符合固定鏡像 revision 的 LFS metadata；全部為 1500×1000 RGB。
- `image_anno.csv` 與官方 S3 原檔的 byte range 完全一致。
- 全部 100 張 capsule 多類遮罩與官方 S3 原檔逐位元組一致；不是將二值遮罩誤認為有缺陷類別的遮罩。
- 官方 S3 僅下載分類 CSV 的 36,870 bytes，以及遮罩 tar 段的 314,880 bytes，避免取得完整大檔。
- 六張異常圖的官方類別含 `discolor`，主要非零 mask 類別為 2，並保留 `bubble` 的複合標籤。
- 13 張衍生評估遮罩只含 0、255，前景數量與原始 ID 2 像素數相同。
- 資料選擇包括三張不同正常校準圖及三張pilot異常圖；正式benchmark只以三張正常圖擬合閾值，異常圖、模板來源及六張heldout均不參與。manifest歷史欄位的pilot eligibility包括異常資料，因此消費者還必須按label過濾；正式runner以kind／split／color_anomaly／template_reference共同選擇，不單靠該欄位。離線重播可驗證相同輸入。

原圖目前驗證到固定公開鏡像的 LFS hash，**未逐張與官方完整 tar 的影像位元組比較**；分類 CSV 與全部 masks 則已直接對官方驗證。資料準備本身沒有量測算法；本輪結果另見[正式報告](../STAGE35_REAL_RGB_zh-TW.md)。全物件定位標註、純色差缺陷隔離與跨場域泛化仍未完成。
