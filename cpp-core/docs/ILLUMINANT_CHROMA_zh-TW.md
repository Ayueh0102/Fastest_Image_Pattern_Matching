# 照明正規化的同色驗證（物件內白參考＋von Kries，研究選項）

第46輪，2026-10-25，Apple M4 Pro／48 GiB。評估資料是[測試樣本集 v1](TESTBED_V1_zh-TW.md)。

測試樣本集 v1 的基準檢查顯示，既有色度驗證在中性光下可分（AUC 0.961），但白平衡偏移、局部暖光、過曝或 JPEG 下，同色接受率掉到 0–5%；RGB NCC（約 0.65）和灰階 NCC（低於 0.5）都不能判色。以 GT 對齊的探針用物件自己的米白面板做 von Kries 校正，白平衡與 Blender 暖光／混合光的 AUC 回到 1.000，局部暖光 0.986，過曝 0.898；但只用中性光定的門檻在其他照明下太緊。

本輪把這個想法做成 runner 的預設關閉方法 `chroma_vk`，在事前登錄的「校準照明／保留照明」切分下評估。既有的 `chroma`、`rgb_ncc`、`gray_ncc` 不變，定位仍是既有灰階搜尋，所有方法共用同一份偵測結果。

**結論：依事前規則不推薦。** 物件內白參考的校正讓同色與異色分得很開，但事前登錄的門檻規則在保留照明下拒絕了大部分同色物件，H1a、H2a、H3 未通過。`chroma_vk` 保留為研究選項，預設不變。

- **定位與完整性全部通過**：19 個場景的嚴格 GT 與測試樣本集 v1 相同（合成每張 32/0/0、Blender 每張 12/0/0）。每個場景所有方法的搜尋結果逐項相同。四個既有方法的輸出和改動前的 runner 逐位相同，job 裡沒有 GT。
- **異色從未被誤收**：主方法 `vk_region` 在 19 個場景、376 個找到的異色物件中，誤收 0 個；包括全部 134 個灰階相同（isogray）物件、ΔE 2 和 lightness+。既有的遮罩色度在保留照明下誤收 63/192，其中 isogray 19/68。
- **同色接受率未過門檻**：保留照明合計 31/88。白平衡偏暖 18/20 通過；局部暖光 4/20、JPEG 4/20、Blender 暖光 3/4、混合光 2/4，都未通過。過曝 0/20，因為面板全飽和，全部棄權（事前預測 P1）。
- **原因（事後診斷，非證據）**：每個沒有棄權的格子裡，同色最大距離 0.0002–0.0061，最近的 ΔE≥4 異色在 0.017（紅）／0.029（藍綠），ΔE 2 也在 0.008／0.013 以上。但事前規則「校準照明同色距離的第 95 百分位」只得到 0.00039（紅）／0.00137（藍綠）／0.00094（Blender）。JPEG、物件內的光梯度與 Blender 混合光造成的殘差超過這些門檻，但仍比最近的異色小 5–10 倍。
- **飽和處理**：面板被裁切時，rank 政策棄權而不是誤判。drop 消融在紅色白平衡偏暖時被截斷偏差拖垮（同色 0/10，rank 為 10/10）。「棄權時改用未校正距離」的後備政策在過曝下誤收 8/22 與 11/22 個異色，不安全。
- **全域參考卡（18% 灰）**：均勻照明下表現好，保留照明 57/80 同色接受、0 誤收；局部暖光 0/20，因為卡片所在處的光和物件不同。白色卡片在白平衡偏暖、偏冷與過曝下都被裁切而棄權。

## 方法

`chroma_vk` 實作在 `experiments/color_illuminant.{hpp,cpp}`，既有的 `color_verify.{hpp,cpp}` 一個位元組都沒改。完整契約見 [runner 文件](COLOR_VERIFY_RUNNER.md#illumination-normalized-chroma-chroma_vk-round-46)。

1. **參考**：模板的參考遮罩（米白面板）經由每個候選的位姿映到來源影像，取最近像素。也可以改用來源影像上固定的矩形（參考卡）。
2. **光源估計**：每個通道各取線性 RGB 的中位數，`W_source` 與模板上同一統計量 `W_template` 相比，得到 von Kries 對角增益 `gain_c = W_template,c / W_source,c`，把來源轉進模板的光。
3. **距離**：只在色彩遮罩內計算。`mean_distance` 是逐像素色度距離的平均，和既有 `chroma` 相同。`region_distance` 先把有效樣本的線性 RGB 加總，再和模板在同一批樣本上的加總比較色度；零均值雜訊不會墊高它的下限。兩者都另外輸出未校正版本。
4. **飽和（rank 政策，預設）**：位元組等於 255 視為裁切。裁切值留在中位數的排序裡，因為裁切比例低於一半時中位數不受影響。任一通道裁切比例超過 0.45、在影像內的參考樣本少於一半、或中位數太暗，就判為 status 3（棄權），校正後距離為 −1，絕不視為接受。色彩樣本只要四個雙線性鄰居中有一個裁切，就算缺少支撐。drop（先丟掉有裁切的樣本再取中位數）與 none（不處理）是消融。

對角模型無法消除單一物件內部變化的光、彩色互反射或非中性參考，也救不回已裁切的參考。

## 事前登錄

[protocol.json](stage46-illuminant-chroma/protocol.json) 與 311 個凍結源碼檔（`2a66cd6`）在任何評估前提交（`377b0c8`）。

- **資料**：16 張合成（紅、藍綠 × 8 種照明，每張 10 同色＋22 異色）與 3 張 Blender（每張 4 同色＋8 異色）。
- **遮罩**：由標籤原稿 `capsule_texture` 決定，色彩區＝家族色，參考區＝面板色 (236,234,226)，各侵蝕 5×5。只用模板設計，不用場景或 GT。兩家族版面相同，兩區相距 7 px。
- **切分**：校準＝neutral、exposure_dark、wb_cool、shading、Blender white；保留＝wb_warm、exposure_bright、spot_warm、noise_jpeg、Blender warm3200、mixed。過曝列入保留但不設門檻，因為依產生器設定，面板必然裁切（線性 0.838 × 1.35 > 1）。
- **門檻**：單類別。每個方法、每個模板群組（syn_red、syn_teal、bl_red）取校準照明下找到且未棄權的同色物件距離的第 95 百分位，原封套用到所有照明。bl_red 只用 bl_rgb_white 的 4 個同色物件校準。棄權一律算未接受。
- **門檻判定**：
  - H1：wb_warm、spot_warm、noise_jpeg 各自（兩家族合併）同色接受 ≥ 0.90；hue±、chroma−、isogray0/90 在 ΔE≥4 的拒絕率 ≥ 0.95；isogray ΔE≥4 合併 ≥ 0.95。
  - H2：Blender warm3200、mixed 各 4/4 同色接受、7/7 異色拒絕。
  - H3：8 個保留格子裡，`vk_region` 的 AUC 不低於遮罩色度與未校正區域色度。
  - lightness+ 與 ΔE 2 事前排除在門檻外，但每張表都報告。
- **事前揭露**：作者事先知道測試樣本集 v1 的探針結果與產生器的照明參數，並用它們寫下預測 P1、P2、P4 與 0.45 的裁切上限；也看過 Blender 場景的物件清單。區域距離是在任何評估之前，根據既有色度在 JPEG 下 0–5% 的接受率加入的。實作 smoke 只用換過種子的重新產生資料檢查管線，沒有看距離。

## 結果

### 主方法 `vk_region` 逐照明

「同色接受」是找到的同色物件中被接受的數量，括號是棄權數。「異色誤收」涵蓋所有方向與 ΔE。

| 照明 | 分組 | 紅 同色接受 | 紅 異色誤收 | 紅 AUC | 藍綠 同色接受 | 藍綠 異色誤收 | 藍綠 AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| neutral | 校準 | 10/10 | 0/22 | 1.000 | 10/10 | 0/22 | 1.000 |
| exposure_dark | 校準 | 10/10 | 0/22 | 1.000 | 9/10 | 0/22 | 1.000 |
| wb_cool | 校準 | 9/10 | 0/22 | 1.000 | 10/10 | 0/22 | 1.000 |
| shading | 校準 | 9/10 | 0/22 | 1.000 | 9/10 | 0/22 | 1.000 |
| wb_warm | 保留 | 10/10 | 0/22 | 1.000 | 8/10 | 0/22 | 1.000 |
| exposure_bright | 保留 | 0/10（棄10） | 0/22 | 0.500 | 0/10（棄10） | 0/22 | 0.500 |
| spot_warm | 保留 | 1/10（棄7） | 0/22 | 0.459 | 3/10（棄4） | 0/22 | 0.709 |
| noise_jpeg | 保留 | 1/10 | 0/22 | 1.000 | 3/10 | 0/22 | 1.000 |

| Blender | 分組 | 同色接受 | 異色誤收 | AUC |
|---|---|---:|---:|---:|
| white | 校準 | 3/4 | 0/8 | 1.000 |
| warm3200 | 保留 | 3/4 | 0/8 | 1.000 |
| mixed | 保留 | 2/4 | 0/8 | 1.000 |

門檻：紅 0.00039、藍綠 0.00137、Blender 0.00094。AUC 把棄權視為 +∞（最不像同色），所以全部棄權的過曝格子是 0.5，局部暖光也被大量棄權拉低。

### 事前門檻判定

| 檢查 | 條件 | 值 | 目標 | 結果 |
|---|---|---:|---:|---|
| H1a 同色接受 | wb_warm | 0.90（n=20） | ≥0.90 | 通過 |
| H1a 同色接受 | spot_warm | 0.20 | ≥0.90 | **未通過** |
| H1a 同色接受 | noise_jpeg | 0.20 | ≥0.90 | **未通過** |
| H1b ΔE≥4 異色拒絕 | 三種照明 | 1.00（各 n=30） | ≥0.95 | 通過 |
| H1c isogray ΔE≥4 | 三種照明合併 | 1.00（n=36） | ≥0.95 | 通過 |
| H2a Blender 同色 | warm3200／mixed | 3/4／2/4 | 4/4 | **未通過** |
| H2b Blender 異色 | warm3200／mixed | 7/7／7/7 | 7/7 | 通過 |
| H3 AUC ≥ 遮罩色度、未校正區域色度 | 紅／藍綠 spot_warm | 0.459／0.709 對 0.791／0.814 | 不低於 | **未通過**（4 項） |
| H3 | 其他 6 個格子 | 1.000 對 0.50–1.000 | 不低於 | 通過（12 項） |

完整性門檻全部通過，所以結論只由 H1–H3 決定：**不推薦**。逐項值在 [summary.json](stage46-illuminant-chroma/summary.json)，判定與每項失敗在 [decision-and-failures.json](stage46-illuminant-chroma/decision-and-failures.json)。

### 各方法在同一批偵測上的比較

合成＋Blender 合計，「同色」是接受數／找到數（棄權數），「異色誤收」涵蓋所有方向與 ΔE：

| 方法 | 校準 同色 | 校準 異色誤收 | 保留 同色 | 保留 異色誤收 | 保留 isogray 誤收 |
|---|---:|---:|---:|---:|---:|
| gray_ncc | 79/84 | 160/184 | 72/88 | 154/192 | 56/68 |
| rgb_ncc | 79/84 | 93/184 | 52/88 | 79/192 | 25/68 |
| chroma（既有，不遮罩） | 79/84 | 110/184 | 51/88 | 85/192 | 30/68 |
| chroma_masked（既有，遮罩） | 79/84 | 71/184 | 64/88 | 63/192 | 19/68 |
| masked_region（未校正區域距離） | 79/84 | 71/184 | 64/88 | 63/192 | 19/68 |
| vk_pixel（校正＋逐像素平均） | 79/84 | 0/184 | 28/88（棄31） | 0/192 | 0/68 |
| **vk_region（主方法）** | 79/84 | **0/184** | 31/88（棄31） | **0/192** | **0/68** |
| vk_drop_region（消融） | 79/84 | 0/184 | 21/88（棄31） | 0/192 | 0/68 |
| vk_none_region（消融） | 79/84 | 0/184 | 31/88 | 0/192 | 0/68 |
| card_white_region（僅合成） | 56/80（棄20） | 0/176 | 2/80（棄40） | 0/176 | 0/64 |
| card_gray_region（僅合成） | 76/80 | 0/176 | 57/80 | 0/176 | 0/64 |

未校正的方法必須用一個門檻涵蓋 wb_cool 的色偏，所以門檻變寬，校準照明下就已誤收 71–160 個異色。主方法在每種照明下的同色區域距離都不超過 0.0061，異色全部拒絕；問題在於同色接受。這裡的 AUC 是每個家族分開算；測試樣本集 v1 的 0.961 是兩家族合併，不能直接比較。

保留照明下各格的 AUC（棄權記為 +∞）：

| 群組／照明 | chroma_masked | vk_pixel | vk_region | vk_none_region | card_gray_region |
|---|---:|---:|---:|---:|---:|
| 紅 wb_warm | 0.864 | 1.000 | 1.000 | 1.000 | 1.000 |
| 紅 exposure_bright | 1.000 | 0.500 | 0.500 | 1.000 | 1.000 |
| 紅 spot_warm | 0.791 | 0.459 | 0.459 | 0.959 | 0.864 |
| 紅 noise_jpeg | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 藍綠 wb_warm | 0.773 | 1.000 | 1.000 | 1.000 | 1.000 |
| 藍綠 exposure_bright | 1.000 | 0.500 | 0.500 | 0.864 | 1.000 |
| 藍綠 spot_warm | 0.814 | 0.709 | 0.709 | 1.000 | 0.836 |
| 藍綠 noise_jpeg | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Blender warm3200 | 0.500 | 1.000 | 1.000 | 1.000 | — |
| Blender mixed | 0.938 | 1.000 | 1.000 | 1.000 | — |

### 方向 × ΔE（保留照明）

`vk_region`、`vk_pixel`、`vk_none_region` 與 `card_gray_region` 在每個方向、每個 ΔE（含 2）都 4/4 拒絕，Blender 每格 2/2。既有遮罩色度在保留照明的拒絕數（每格 n=4）：

| 方向 | ΔE 2 | ΔE 4 | ΔE 8 | ΔE 16 |
|---|---|---|---|---|
| hue+（紅／藍綠） | 0／1 | 1／0 | 4／3 | 4／4 |
| hue− | 0／2 | 0／1 | 3／4 | 4／4 |
| chroma− | 3／2 | 4／4 | 4／4 | 4／4 |
| isogray0 | 2／1 | 4／2 | 4／4 | 4／4 |
| isogray90 | 0／2 | 1／1 | 4／4 | 4／4 |
| lightness+ | — | 0／1 | 2／1 | — |

灰階 NCC 在 isogray 上幾乎全部誤收，這和 isogray 的定義一致：轉成 gray8 後與原色逐像素相同。逐方向的完整表在 summary 的 `directions` 與 `isogray`。

### 事前預測對照

| 預測 | 結果 |
|---|---|
| P1 過曝時主方法幾乎全部棄權；none 消融不棄權但變差 | 成立：64/64 棄權（62 個面板裁切，2 個色彩區裁切）；none 的 AUC 為 1.000／0.864，同色接受 0/20 |
| P2 白卡在過曝與白平衡偏暖下棄權，灰卡不會 | 成立；白卡在 wb_cool 也被藍色裁切而棄權，預測漏了這一項 |
| P3 全域卡片修正均勻白平衡，修不了局部暖光 | 成立：灰卡 wb_warm 9/10、10/10，spot_warm 0/10、0/10 |
| P4 drop 在白平衡偏暖時偏低 | 部分成立：紅 0/10（rank 10/10）；藍綠兩者都是 8/10 |
| P5 JPEG 下逐像素平均會拒絕多數同色，區域距離不會 | 部分成立：vk_pixel 0/20。既有色度因門檻被 wb_cool 撐寬而 20/20 接受；區域距離也受影響（4/20） |
| P6 lightness+ 很難用色度檢出 | 不成立：校正後的方法全部拒絕（固定 a\*、b\* 提高 L\* 仍會改變 RGB 色度）；未校正方法才是弱檢出 |

## 事後診斷（在 summary 之後計算，不是採用證據）

[stage46_posthoc.py](stage46-illuminant-chroma/validation-logs.tar.gz) 的輸出：

- **分離幅度**：主方法在沒有棄權的格子中，最近的 ΔE≥4 異色距離是同色最大距離的 4.7–81 倍。校準照明的同色距離中位數只有 0.0001–0.0012，門檻落在這個極小值附近。
  - 保留照明的同色中位數：JPEG 0.0014（紅）／0.0026（藍綠），局部暖光 0.0009／0.0016，Blender 混合光 0.0014。
  - 這些都比異色小一個數量級，但比門檻大。
- **局部暖光的幾何**：距離光斑中心 1.24σ 以內的物件全部棄權，因為面板紅色裁切；1.37σ 以外全部有效。有效同色物件的殘差隨著接近光斑而增加：1.37σ 為 0.0061、2.4σ 為 0.0002，符合物件內部的光梯度。
- **JPEG**：逐像素平均的同色下限從 0.0045 升到 0.009（紅），區域距離也從約 0.0002 升到 0.0014。JPEG 的 4:2:0 色度次取樣與量化本身會移動區域色度，不只是加入零均值雜訊。
- **門檻規則才是瓶頸**：p95 在乾淨的校準資料上幾乎沒有餘裕。本輪**不**採用任何事後門檻；要改成有餘裕或依雜訊估計的門檻，需要新的事前登錄輪次。

## 驗證

- **獨立重算**：`stage46_recount.py` 不匯入 harness，用自己的匹配器、百分位與判定重算。共 4,550 項檢查，零不符。負控制把一個格子的誤收數加 1，被偵測到。
- **摘要重播**：從 raw-results.tar.gz 解壓後重新 summarize，與原 summary 逐位元組相同。
- **工作重播**：從 source-snapshot.tar.gz 解壓、重新建置 runner，重跑全部 165 個候選 job，分數與搜尋結果 165/165 相同。重建出的二進位雜湊不同，因為建置路徑不同。
- **突變測試**：`color_illuminant.cpp` 的 16 個編譯突變全部被原生測試或 runner 契約測試抓到。harness 的 20 個工具突變全部被抓到；T9（p95 換成最大值）第一次存活，在實作 commit 前補強了測試，`stage46_t9_first_run.py` 在暫存複本上重現了這次存活。兩組突變在實作 commit 前跑過一次，量測 commit 後又在 HEAD 的暫存複本上重跑一次，結果相同。
- **最終測試門檻**（`b9572e0`，每項前後工作樹都乾淨）：全新 Release 研究建置；ctest 83/83（多了 `experimental_color_illuminant`）；帶 `FPM_FINE_*` 的 Python harness 661 項（含本輪 26 項）；`color_verify_runner_test.py --metal` 8 項（既有 5＋`chroma_vk` 3）；其他 Python 測試 6＋17＋26；ASan／UBSan 78/78。指令與 log 雜湊在 [validation.json](stage46-illuminant-chroma/validation.json)。
  - ASan 執行期間，作者曾在工作樹修改兩個文件檔約一分鐘（本頁 P1 用詞與判定檔），之後還原再重新套用，沒有動到源碼。

## 重現

從 repo 根目錄、使用 python-port venv。測試樣本集 v1 依 [TESTBED_V1](TESTBED_V1_zh-TW.md) 重建；不要在其他 session 進行 GPU 正式計時時執行。

```sh
PY=python-port/venv/bin/python
cmake -S cpp-core -B build/stage46-release -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_GUI=ON -DFPM_BUILD_EXPERIMENTS=ON -DFPM_WITH_OPENCV=ON
cmake --build build/stage46-release -j --target fpm_color_verify_runner
# 改動前的 runner（main 2d85d3a）另建於 build/stage46-base，用來驗證既有方法不變
$PY cpp-core/tools/benchmark_illuminant_colour.py masks --output build/stage46-illuminant/masks
$PY cpp-core/tools/benchmark_illuminant_colour.py run --testbed build/testbed-v1 \
  --runner build/stage46-release/experiments/fpm_color_verify_runner \
  --base-runner build/stage46-base/experiments/fpm_color_verify_runner \
  --masks build/stage46-illuminant/masks --protocol cpp-core/docs/stage46-illuminant-chroma/protocol.json \
  --output build/stage46-illuminant/run/manifest.json
$PY cpp-core/tools/benchmark_illuminant_colour.py summarize --manifest build/stage46-illuminant/run/manifest.json \
  --protocol cpp-core/docs/stage46-illuminant-chroma/protocol.json --output build/stage46-illuminant/summary.json
```

## 證據檔

[stage46-illuminant-chroma/](stage46-illuminant-chroma/)：

- `protocol.json`、`source-snapshot.tar.gz`、`source-sha256.json`
- `raw-results.tar.gz`（遮罩、run manifest、241 個 job、stderr 與結果），每個成員的 SHA-256 在 `raw-files-sha256.json`
- `input-sha256.json`、`summary.json`、`decision-and-failures.json`、`validation.json`
- `validation-logs.tar.gz`（重算、重播、事後診斷、突變與最終測試 log）
- `archive.json`（以上全部檔案的 SHA-256）

## 限制

- 只有渲染器資料（OpenCV 合成與 Blender Cycles），一種標籤設計、兩個色系、一個 Blender 色系，不是相機實拍。
- 參考必須是模板上印刷的中性區域，且每個物件都看得到；參考裁切時只能棄權。
- 對角模型直接作用在 sRGB 解碼值上，沒有相機特性描述，不處理彩色互反射或非中性參考。
- Blender 門檻只來自 4 個校準物件。
- 不作時間宣稱：搜尋走 Metal、驗色用 1 個 CPU 執行緒，其他 session 可能同時使用機器。

## 後續

- 事前登錄一個有餘裕的門檻規則，例如依校準雜訊與 JPEG 估計的上界，或以要偵測的最小 ΔE 決定。保留集照明不能拿來調門檻。
- 物件內的光梯度：在面板兩端各估一個參考做線性內插，或加第二塊參考區。
- 過曝：加入暗中性參考（邊框、印刷記號）作第二層，或明確把「參考裁切」當成需要人工檢查的輸出。
- 實拍驗證：真實相機、真實白平衡與壓縮。
