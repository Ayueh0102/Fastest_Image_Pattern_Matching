# 有餘裕的事前門檻與物件內光梯度模型（`chroma_vk`，研究選項）

第47輪，2026-10-25／26，Apple M4 Pro／48 GiB。

本分支 `claude/round47-chroma-margin` 先把第46輪（`claude/round46-illum-chroma`，57887db）合併到 main（c1e3b03）上（`ce057fe`）。**第46輪還不在 main 上，會和本輪一起進 main。**

第46輪的 `chroma_vk` 能把同色與異色分得很開：用物件自己的米白面板估光源，再做 von Kries 校正。但它事前登錄的門檻是「校準照明同色距離的第95百分位」，只有 0.00039／0.00137／0.00094，在保留照明下拒絕了大部分同色物件，因此未通過。

本輪事前登錄一條**有明確餘裕的門檻規則**，並加一個選用的**物件內線性光源模型**。評估資料是用全新種子重新產生的保留集，第46輪的評估場景完全沒有讀。

**結論：依事前規則，新的門檻規則推薦為研究用門檻；線性光源模型不推薦。預設全部不變。**

- **門檻規則（catch）**：門檻 τ 是「必須抓到的 ΔE00=4 變化」在無雜訊下的最小區域色度距離的一半，只由標籤原稿決定，不看任何場景：紅 0.00816、藍綠 0.01470（Blender 紅色用同一份原稿）。第46輪校準照明的資料只用來檢查可行性，要求同色殘差不超過 τ/2。
- **主方法 `vk_region` 的事前判定全部通過**（17 項）。新保留集中，有判定的同色物件 57/57 全部接受。ΔE≥4 的異色在 134 個中誤收 0，其中 isogray 0/52。所有誤收都是門檻外的 ΔE 2（保留照明 4/192）。
  - 有判定的同色物件中，最大殘差是 τ 的 0.43 倍（Blender 混合光）。
  - 最近的 ΔE≥4 異色在 τ 的 1.97 倍以上。
- **棄權要另外看**：局部暖光下，有 11/20 個同色物件因為面板紅色裁切而棄權，端到端接受只有 9/20；過曝下 64 個物件全部棄權。門檻規則救不回被裁切的參考，所以判定只對「有判定」的物件做，並設最少件數；棄權照實報告，視為需要人工檢查的輸出。
- **只看雜訊的規則不夠**：同一批物件改用校準雜訊的 Rayleigh 規則，有判定的同色接受只有 43/57；JPEG 下 0.40，局部暖光 0.89，Blender 混合光 0.75。第46輪的 p95 規則只有 34/57。
- **線性光源模型（`reference_model: "linear"`）不推薦**，兩個條件都沒達到：
  - 藍綠的校準可行性 F1 未通過：校準最大殘差 0.00911，超過 τ/2 = 0.00735。
  - 在局部暖光與 Blender 混合光下，殘差比常數模型大的有 9/13 對（符號檢定 p = 0.954）。
  - 外插會放大面板雜訊；而光梯度大的物件，正是面板被裁切而棄權的那些。
- **完整性全部通過**：412 個 worker，每個場景的定位都是 32/0/0 或 12/0/0。既有方法與第46輪的 `chroma_vk` 輸出都和改動前的 runner 逐位相同；v1 校準場景的輸出與第46輪封存 62/62 相同。job 裡沒有 GT。

## 方法

### 門檻規則（主）

```
d_catch = min( (a) 原稿色在固定 L* 的 ΔE00=4 圓上（每 1°、在色域內、不量化）的區域色度距離，
               (b) 每個受判定的原稿變體（hue±、chroma−、isogray0/90，目標 ΔE≥4）的區域色度距離 )
τ = d_catch / 2
```

- 距離公式就是 `chroma_vk` 的區域色度：線性 sRGB 色度的歐氏距離除以 √2。
- 紅色由固定 L* 圓決定：118° 方向 0.01632，最近的變體 hue+ 4 是 0.01729。藍綠由變體 isogray0 4 決定：0.02940，比固定 L* 圓的最小值 0.03123 還近。這也是規則納入 (b) 的原因（計算時看到，寫進事前登錄的形式）。
- τ 對所有色度方法、所有照明、兩種光源模型都一樣；NCC 方法沒有 catch 門檻。summarize 會重新計算 τ，與 protocol 凍結值不符就算完整性失敗。
- 兩側的餘裕：最弱的必抓變化在無雜訊下是 2τ；同色要通過可行性檢查，殘差必須不超過 τ/2。代價是 ΔE 2 正好落在門檻附近（紅 hue± ΔE 2 為 0.0082），事前就排除在判定外。

### 判定（事前登錄）

- **F（校準照明，第46輪的 v1 場景）**：每個群組的同色最大殘差 ≤ τ/2（F1），ΔE≥4 異色拒絕 ≥ 0.95（F2）。
- **H1（新保留集，合成）**：wb_warm、spot_warm、noise_jpeg 各自（兩色系合併）：
  - 有判定的同色接受 ≥ 0.90，且至少 5 個有判定（H1a）；
  - ΔE≥4 異色拒絕 ≥ 0.95（H1b，n=30）；
  - isogray 三種照明合併 ≥ 0.95（H1c，n=36）。
- **H2（新保留集，Blender）**：warm3200、mixed 各自有判定的同色全部接受，至少 2 個有判定（H2a）；7 個 ΔE≥4 異色全部拒絕（H2b）。
- 「有判定」＝找到且沒有棄權。棄權永遠不算接受；對異色來說等於拒絕。
- **只報告、不判定**：端到端接受（棄權算未接受，也就是第46輪的定義）、exposure_bright、新保留集的校準照明場景、ΔE 2、lightness+、AUC、其他方法與次要規則。

### 線性光源模型（選用方法）

模板參考像素依模板 u、v 分別排名，最低與最高的三分之一是各軸的兩端。每一端都和整塊參考用同樣方式估計（每通道中位數，同一套裁切政策與限制），並放在其背後樣本的中位數座標 (u, v) 上。兩組端點對聯立解出 u、v 兩個斜率，再把光源外插到每個色彩樣本，以 `T / W(u,v)` 校正。任何一端不可用、兩組對的行列式不為正、或外插光源 ≤ 0，都棄權。完整契約見 [runner 文件](COLOR_VERIFY_RUNNER.md#linear-reference-model-reference_model-round-47)。

色彩區（模板 x 5–67）整塊在面板（x 75–122）左邊，所以 u 方向要外插約一個面板寬。偏好規則是：線性模型必須通過自己的 F/H1/H2，而且在 spot_warm 與 Blender mixed 的配對同色殘差上，單尾符號檢定 p < .05，才會取代常數模型。

### 資料

- **校準**：test-bed v1 的 syn_{red,teal}_{neutral, exposure_dark, wb_cool, shading} 與 bl_rgb_white，共 9 張，也就是第46輪的校準照明。第46輪的保留場景完全沒有讀。
- **新保留集**：`benchmark_chroma_margin.py holdout` 以從未用過的種子重新產生：
  - 合成場景：layout 47000001／47000002、noise 47001000、background 47002000；
  - Blender：Cycles 47003000、紋理順序 47003001、位姿 47003002。
  - 標籤原稿的種子不變，25 個紋理檔與 v1 逐位元組相同。
  - 合成 16 張（2 色系 × 8 種照明），Blender 3 張（white、warm3200、mixed）。
- **切分**：保留＝新集的 wb_warm、exposure_bright、spot_warm、noise_jpeg、warm3200、mixed（10 張）。新集的校準照明 9 張只報告，不校準也不判定。
- **模板與遮罩**：模板一律用 v1 的（模板是已學好的模型），遮罩與第46輪逐位元組相同。定位設定與第46輪相同。

## 事前登錄

- 實作分三個 commit：`b879690`、`172f86a`、`190973e`。
- [protocol.json](stage47-chroma-margin/protocol.json) 與凍結源碼（`190973e`，315 檔）在保留集產生之前提交（`e78a484`）。
- 規則的形式在算出數值之前，就寫進共用筆記（2026-10-25T15:41Z）。
- 事前揭露（詳見 protocol）：
  - 作者看過第46輪的報告與事後數字（未棄權格子同色最大 0.0061、最近異色 0.017／0.029、spot 幾何），預測有用到這些。
  - 線性模型的細節是寫單元測試時定的。逐軸版本因秩相同的平手讓 u 端在 v 方向不平衡，沒通過 v 梯度測試；這個失敗起初被一個過期的測試執行檔遮住，之後改成聯立解。
  - 兩次只看管線的 smoke：用 smoke 種子產生資料，只檢查完整性、定位、狀態與有限值。看到 smoke 藍綠 spot_warm 有 18/32 個狀態 3。
  - 突變測試第一次有 3 個存活（見「驗證」）。
  - Blender 用預設種子重渲，像素與 v1 相同。

## 結果

### 主方法 `vk_region`、catch 門檻、新保留集

| 照明 | 同色 接受／找到 | 有判定 | 棄權 | ΔE≥4 異色誤收 | isogray 誤收 | 全部異色誤收 | 最大同色／τ |
|---|---:|---:|---:|---:|---:|---:|---:|
| wb_warm | 20/20 | 20/20 | 0 | 0/30 | 0/12 | 0/44 | 0.094 |
| spot_warm | 9/20 | 9/9 | 11 | 0/30 | 0/12 | 2/44（ΔE 2） | 0.285 |
| noise_jpeg | 20/20 | 20/20 | 0 | 0/30 | 0/12 | 2/44（ΔE 2） | 0.248 |
| exposure_bright（不判定） | 0/20 | 0/0 | 20 | 0/30 | 0/12 | 0/44 | — |
| Blender warm3200 | 4/4 | 4/4 | 0 | 0/7 | 0/2 | 0/8 | 0.092 |
| Blender mixed | 4/4 | 4/4 | 0 | 0/7 | 0/2 | 0/8 | 0.427 |

「最大同色／τ」是有判定同色殘差的最大值除以 τ（兩色系取大者）；最近的 ΔE≥4 異色在每格都 ≥ 1.97τ。

### 事前判定

| 檢查 | 條件 | 值 | 目標 | 結果 |
|---|---|---:|---:|---|
| F1 校準同色 ≤ τ/2 | 紅／藍綠／Blender | 0.00061／0.00178／0.00094 | 0.00408／0.00735／0.00408 | 通過 |
| F2 校準 ΔE≥4 異色拒絕 | 三群組 | 1.00（n=60／60／7） | ≥0.95 | 通過 |
| H1a 有判定同色接受 | wb_warm／spot_warm／noise_jpeg | 1.00（n=20／9／20） | ≥0.90，n≥5 | 通過 |
| H1b ΔE≥4 異色拒絕 | 三種照明 | 1.00（各 n=30） | ≥0.95 | 通過 |
| H1c isogray 合併 | 三種照明 | 1.00（n=36） | ≥0.95 | 通過 |
| H2a Blender 有判定同色 | warm3200／mixed | 4/4／4/4 | 全部，n≥2 | 通過 |
| H2b Blender 異色 | warm3200／mixed | 7/7／7/7 | 全部 | 通過 |

完整性全部通過，所以依事前規則**推薦**。逐項值在 [summary.json](stage47-chroma-margin/summary.json)，判定、每一項失敗與預測對照在 [decision-and-failures.json](stage47-chroma-margin/decision-and-failures.json)。

### 端到端（棄權算未接受，只報告）

wb_warm 20/20、noise_jpeg 20/20、Blender 4/4、4/4，spot_warm **9/20**（11 個棄權）。

事後看位置：局部暖光中，距光斑中心 1.21σ 以內的同色物件全部棄權，1.38σ 以外全部有判定，和第46輪的 1.24σ／1.37σ 一致。過曝的 64 個物件全部棄權。這兩種照明需要第二個參考（例如暗的中性區），或把「參考裁切」直接交給人工檢查。

### 同一批物件上的其他方法與規則（新保留集，88 個同色、192 個異色）

| 方法 | 規則 | 同色接受 | 有判定同色接受 | ΔE≥4 異色誤收 | 全部異色誤收 |
|---|---|---:|---:|---:|---:|
| **vk_region** | **catch** | **57/88** | **57/57** | **0/134** | **4/192** |
| vk_region | Rayleigh（校準雜訊） | 43/88 | 43/57 | 0/134 | 0/192 |
| vk_region | p95（第46輪） | 34/88 | 34/57 | 0/134 | 0/192 |
| vk_linear_region | catch | 57/88 | 57/57 | 0/134 | 7/192 |
| vk_pixel（逐像素平均） | catch | 39/88 | 39/57 | 0/134 | 1/192 |
| card_gray_region（全域灰卡，僅合成） | catch | 64/80 | 64/80 | 0/120 | 8/176 |
| vk_none_region（不處理裁切，消融） | catch | 77/88 | 77/88 | 0/134 | 9/192 |
| masked_region（不校正） | catch | 48/88 | 48/88 | 0/134 | 5/192 |
| chroma_masked（既有，不校正） | catch | 29/88 | 29/88 | 0/134 | 1/192 |
| chroma_masked | p95 | 63/88 | 63/88 | 23/134 | 63/192 |
| rgb_ncc | p95 | 55/88 | 55/88 | 39/134 | 74/192 |
| gray_ncc | p95 | 65/88 | 65/88 | 113/134 | 157/192 |

- 同一個 τ 對未校正的方法來說太緊：光一變，同色就被拒絕，而不是誤收異色。
- 不處理裁切的消融（vk_none_region）不會棄權，所以同色接受比較多，ΔE≥4 也沒有誤收。但它在裁切下沒有理論保證，而且不在事前登錄的候選裡。第46輪「棄權時改用未校正距離」的後備在過曝下誤收很多，所以這裡只記錄、不採用。
- 第46輪的 p95 規則在新保留集上，有判定的同色只接受 34/57，與第46輪在 v1 保留集的 31/88 方向一致。

### 門檻外的格子

- **ΔE 2**（無雜訊距離就在 τ 附近）：紅 hue+ 3/12、hue− 4/12，藍綠 hue+ 8/12、isogray0 7/12 被接受（三種切分合計）；其他方向 0/12。藍綠的這兩個 ΔE 2 變體無雜訊距離是 0.0139、0.0143，正好低於 τ 0.0147，預測沒有提到。
- **lightness+**：全部拒絕（合成 0/48、Blender 0/4）。

### 線性光源模型

| 配對（兩者都有判定的同色） | n | 線性較小 | 線性較大 | p（單尾） | 中位數比 線性／常數 |
|---|---:|---:|---:|---:|---:|
| spot_warm＋mixed（事前判定） | 13 | 4 | 9 | 0.954 | 1.28 |
| spot_warm | 9 | 2 | 7 | 0.980 | 1.28 |
| mixed | 4 | 2 | 2 | 0.688 | 1.17 |
| noise_jpeg | 20 | 6 | 14 | 0.979 | 2.10 |
| wb_warm | 20 | 6 | 14 | 0.979 | 1.00 |

線性模型自己的 H1、H2 全部通過，但 F1 藍綠未通過（exposure_dark、shading 的校準最大值 0.0087、0.0091）。

事後診斷（非證據）：均勻光下兩端的中位數常常落在同一個位元組，斜率剛好是 0；只要有梯度（shading、JPEG），每通道各自估出的斜率雜訊外插約一個面板寬，就會放大成色度誤差。spot_warm 中真正有大梯度的物件都已經因面板裁切而棄權，有判定的物件離光斑遠，梯度小，外插雜訊反而比梯度修正大。

### 事前預測對照

| 預測 | 結果 |
|---|---|
| P1 F1、F2 全部通過 | 成立：校準最大殘差是 τ 的 0.15／0.24／0.23 |
| P2 wb_warm、noise_jpeg 有判定同色全部接受 | 成立：20/20、20/20 |
| P3 spot_warm 通過，且是合成格子中餘裕最小的 | 成立：0.285τ，noise_jpeg 0.248τ、wb_warm 0.094τ；Blender mixed 更緊（0.427τ） |
| P4 spot_warm 端到端 < 0.90；過曝幾乎全部棄權 | 成立：9/20；64/64 |
| P5 沒有 ΔE≥4 異色被接受 | 成立 |
| P6 紅 ΔE 2 約一半被接受；lightness+ 被拒絕 | 部分成立：lightness+ 全部拒絕；紅 hue± ΔE 2 約三分之一（3/12、4/12）；漏了藍綠兩個 ΔE 2 方向（8/12、7/12） |
| P7 Rayleigh 在 JPEG、spot_warm 未通過；p95 如第46輪 | 成立：0.40、0.89（另有 mixed 0.75）；p95 0.25、0.33、0.75 |
| P8 線性模型降低梯度場景殘差，但提高均勻光殘差 | 不成立：梯度場景反而較大（4／9）；均勻光較大成立 |
| P9 Blender 兩種照明兩種模型全部接受 | 成立 |

## 驗證

- **獨立重算**：`stage47_recount.py` 不匯入 harness、`testbed_generate` 或 `many_object_accuracy`。它用自己的 sRGB／CIELAB／CIEDE2000（以 Sharma 參考數對檢查）重算 catch 門檻，再用自己的匹配器、門檻、判定、配對與裁決重算。共 11,903 項，零不符；負控制在一個格子多加一個誤收，被偵測到。
- **摘要重播**：從 raw-results.tar.gz 解壓後重新 summarize，與 summary.json 逐位元組相同。
- **job 重播**：從 source-snapshot.tar.gz 重建 runner，重跑 220 個候選 job，分數與搜尋結果 220/220 相同。二進位雜湊不同，因為建置路徑不同。
- **第46輪一致**：v1 校準場景上，7 種方法、62 個輸出與第46輪封存相同，job 除路徑外也相同。
- **保留集重建**：依 protocol 種子重新產生，91 個檔案全部一致：84 個逐位元組相同，3 張 Blender PNG 像素相同（PNG 檔含渲染中繼資料，位元組會不同），4 個場景 JSON 相同。
- **突變測試**：
  - 第一次（`172f86a`）有 3 個存活：R4 是 runner 允許 `chroma` 帶 `reference_model`，原測試同時帶了會被另外拒絕的 `reference_mask`；T13／T20 是判定與配對只用照明名稱篩選，在實際切分下等價。
  - `190973e` 補強測試後重跑，20 個編譯突變與 32 個工具突變全部被殺。量測 commit 後又在 HEAD 的暫存複本上重跑一次，結果記錄在 [validation.json](stage47-chroma-margin/validation.json)。
- 最終測試門檻（全新 Release 建置、ctest、Python harness、color runner `--metal`、其他 Python、ASan／UBSan）的指令與 log 雜湊也在 validation.json。

## 重現

從 repo 根目錄、使用 python-port venv。不要在其他 session 正式計時時執行，因為 Blender 會吃滿 CPU、搜尋會用 Metal。

```sh
PY=python-port/venv/bin/python
git archive 190973e | tar -x -C build/stage47-release-src   # 先建立目錄
cmake -S build/stage47-release-src/cpp-core -B build/stage47-release -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_GUI=ON -DFPM_BUILD_EXPERIMENTS=ON -DFPM_WITH_OPENCV=ON
cmake --build build/stage47-release -j --target fpm_color_verify_runner
# 改動前的 runner：git archive ce057fe，同樣建到 build/stage47-base
$PY cpp-core/tools/benchmark_chroma_margin.py holdout --v1 build/testbed-v1 --output build/stage47-holdout \
  --protocol cpp-core/docs/stage47-chroma-margin/protocol.json --blender /Applications/Blender.app/Contents/MacOS/Blender
$PY cpp-core/tools/benchmark_chroma_margin.py masks --output build/stage47-margin/masks
$PY cpp-core/tools/benchmark_chroma_margin.py run --v1 build/testbed-v1 --holdout build/stage47-holdout \
  --runner build/stage47-release/experiments/fpm_color_verify_runner --base-runner build/stage47-base/experiments/fpm_color_verify_runner \
  --masks build/stage47-margin/masks --protocol cpp-core/docs/stage47-chroma-margin/protocol.json --output build/stage47-margin/run/manifest.json
$PY cpp-core/tools/benchmark_chroma_margin.py summarize --manifest build/stage47-margin/run/manifest.json \
  --protocol cpp-core/docs/stage47-chroma-margin/protocol.json --output build/stage47-margin/summary.json
```

## 證據檔

[stage47-chroma-margin/](stage47-chroma-margin/)：

- `protocol.json`、`source-snapshot.tar.gz`、`source-sha256.json`
- `raw-results.tar.gz`：遮罩、run manifest、412 個 job、stderr 與結果
- `holdout-metadata.tar.gz`：新保留集的種子紀錄、GT、Blender 中繼資料與 log、`final/manifest.json`。影像本身不封存：可依種子重建，並由 SHA-256 釘住。
- `raw-files-sha256.json`、`input-sha256.json`、`summary.json`、`decision-and-failures.json`、`validation.json`
- `validation-logs.tar.gz`：重算、重播、保留集重建、第46輪一致、事後診斷、突變與最終測試 log
- `archive.json`：以上全部檔案的 SHA-256

## 限制

- 只有渲染器資料（OpenCV 合成與 Blender Cycles）；一種標籤設計、兩個色系、一個 Blender 色系，不是相機實拍。
- 門檻的品質取決於規格本身：「ΔE00=4 的色度變化要抓到」。ΔE 2 落在門檻上，亮度變化不在範圍內。
- 參考必須是每個物件都看得到的印刷中性區；參考被裁切時只能棄權。局部暖光與過曝的覆蓋率是這個方法目前真正的缺口。
- Blender 每格只有 4 個同色物件；spot_warm 只有 9 個有判定。
- 不作時間宣稱。

## 後續

- 第二層參考：用暗的中性區（邊框、印刷記號）處理面板裁切，或把參考裁切明確交給人工檢查，並事前登錄覆蓋率門檻。
- 不處理裁切的消融在本輪接受較多、也沒有誤收 ΔE≥4。要採用必須另開事前登錄的一輪，並包含過曝時的安全性檢查。
- 實拍驗證：真實相機、白平衡與壓縮，需要獨立的色差標準件。
