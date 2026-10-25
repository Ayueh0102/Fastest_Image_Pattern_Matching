# 測試樣本集 v1：Blender 渲染、合成 RGB 與同色比對

2026-10-24 建立。這是一套**測試資料與產生器**，不是新一輪演算法實驗；本頁不含任何比對器的速度或精度結論。資料本身放在不追蹤的 `build/testbed-v1/`，repo 只保存產生器、單元測試與本頁，任何人都能用固定種子重建。

## 為什麼要這一套

既有證據大多來自同模板、物件互不重疊的合成場景（第 11、34、37–41 輪），第 34 輪 Blender 只有 512×512 的正交正面平面。仍未覆蓋的情況是：

- 平面傾斜、透視相機：超出目前的相似變換（位置／角度／尺度）模型。
- 光澤反光與投影：NCC 對非線性亮度變化敏感。
- 180° 對稱模板：角度有兩個正解，候選去重（第 43 輪）需要這種反例。
- 「同色比對」：第 35 輪發現 RGB NCC 會抵消通道偏移、線性色度又會被白平衡誤報；需要有可控色差、可控照明的樣本，分開評估「找得到」和「顏色對不對」。

## 內容

### 合成 RGB 同色場景（OpenCV，16 張）

`testbed_generate.py synthetic`。兩個色系（紅 `(196,58,48)`、藍綠 `(40,140,150)`），每張 2048×1152、32 個物件：

| 物件 | 數量 | 說明 |
|---|---:|---|
| 同色（base） | 10 | 與模板同一顏色 |
| 色相 ±（hue+／hue−） | 各 4 | CIEDE2000 目標 2／4／8／16，保持 L\*、C\* |
| 彩度降低（chroma−） | 4 | 目標 2／4／8／16，保持 L\*、h |
| **灰階相同（isogray0／isogray90）** | 各 4 | 在 77R+150G+29B 固定的平面內移動，**轉成本專案 gray8 後與原色逐像素相同**；灰階比對器無法區分，只能靠顏色驗證 |
| 明度增加（lightness+） | 2 | 目標 4／8，會改變灰階 |

每個色系 8 種照明條件，版面與物件顏色都相同，只換光：`neutral`、`exposure_dark`（×0.6）、`exposure_bright`（×1.35，會飽和）、`wb_warm`、`wb_cool`（線性 RGB 對角增益）、`shading`（左到右 0.55→1.15）、`spot_warm`（局部暖光，空間上不均勻的白平衡）、`noise_jpeg`（σ=4＋JPEG q75）。

合成在線性光下進行，再轉回 8-bit sRGB 並加感測雜訊。左上角有白／18% 灰／黑三格參考卡（GT 給出位置），供「以參考白校正」的方法使用。模板從同色系單一物件的中性參考圖精確裁切。

### Blender 場景（Cycles CPU）

`testbed_blender_render.py`，固定 seed 與執行緒數（整套約 30 秒）、Standard view transform、8-bit PNG；每張都保存打包材質的 `.blend`。

| 場景 | 尺寸 | 物件 | 挑戰 |
|---|---|---:|---|
| `bl_gray_reference` | 768² | 1 | 灰階模板來源（Dst3 紋理） |
| `bl_gray_crowded` | 1536² | 36 | 尺度 0.8–1.25、任意角度 |
| `bl_gray_tilt` | 768² | 9 | 平面傾斜 0–40°（超出模型） |
| `bl_gray_perspective` | 768² | 9 | 透視相機偏離法線 25°（超出模型） |
| `bl_gray_gloss_shadow` | 768² | 9 | 光澤材質、斜向主光、3 個物件被投影 |
| `bl_sym_reference`／`bl_sym_scene` | 768² | 1／9 | 180° 對稱紋理 |
| `bl_rgb_reference` | 768² | 1 | RGB 模板來源（紅色系 base） |
| `bl_rgb_white`／`bl_rgb_warm3200`／`bl_rgb_mixed` | 1024×768 | 12 | 同一版面：4 同色＋8 異色（色相、彩度、灰階相同、明度），分別在白光、約 3200K 暖光、左暖右冷混合光下 |

## Ground truth 與標籤

- 位姿慣例沿用 `many_object_scenes._corners`：`x = cx + s(cos·u + sin·v)`、`y = cy + s(−sin·u + cos·v)`，影像座標中逆時針為正。正面場景的 Blender 投影角點在 finalize 時與這個公式比對（容許 1e-3 px）。傾斜與透視場景只提供投影四邊形，角度與尺度不代表單一相似變換。
- GT 只寫在 `*.truth.json`（合成）或場景 `*.json`（Blender）；不寫入任何 job。
- **顏色標籤在紋理層級、中性參考光下定義一次**：`same_colour`、`direction`、`delta_e00_target`、`delta_e00_achieved`（8-bit 量化後的實際值）、`gray_texture_identical_to_base`。照明條件只改光、不改物件顏色，所以同色物件在所有條件下都必須判為同色。部分 `hue−` ΔE2 變體因四捨五入恰好與原色灰階相同，標籤如實記錄。
- 色差為 CIEDE2000（sRGB D65、kL=kC=kH=1），實作以 Sharma 等人 2005 年的參考數對做單元測試。

## 重建

從 repo 根目錄、使用 python-port venv：

```sh
D=build/testbed-v1
python-port/venv/bin/python cpp-core/tools/testbed_generate.py textures --output $D/textures
python-port/venv/bin/python cpp-core/tools/testbed_generate.py synthetic --textures $D/textures --output $D/synthetic
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
  --python cpp-core/tools/testbed_blender_render.py -- --textures $D/textures --output $D/blender
python-port/venv/bin/python cpp-core/tools/testbed_generate.py finalize \
  --rendered $D/blender --synthetic $D/synthetic --output $D/final
python-port/venv/bin/python -m unittest discover -s cpp-core/tests -p test_testbed_generate.py
```

Blender 整套約 30 秒、但會吃滿 CPU；**不要在其他 session 進行 GPU 正式計時時執行**。`final/manifest.json` 記錄每個檔案的 SHA-256 與產生器雜湊。

## 建議的評估方式

1. **定位**：嚴格 GT（2 px／1°／2%），FP 與 FN 都算；傾斜／透視場景改用投影中心與四邊形 IoU，並分開報告。
2. **同色判斷**：把「找到物件」和「顏色判斷」分開。以每個 GT 物件為單位，報告同色物件在各照明下的接受率，以及每個方向 × ΔE 等級的拒絕率；`isogray` 是關鍵案例，因為它只能靠顏色區分。
3. **照明策略要分開比**：模板相對比較、參考卡校正（von Kries 對角模型）、色彩不變量（例如 Gevers & Smeulders 的 c1c2c3、l1l2l3）各自報告，不混成一個分數。

## 基準 sanity 結果（2026-10-24，非正式量測）

這是確認資料可用的一次性檢查：`testbed_evaluate.py` 以既有 `fpm_color_verify_runner`（整合搜尋模式）每個場景、每種方法各跑一個程序，只報告結果、不報告時間。凍結來源與正式 protocol 留給之後的研究輪。評估 JSON 在 `build/testbed-v1/eval/evaluation.json`。

**定位（嚴格 GT）**

| 場景 | 結果 |
|---|---|
| 16 張合成 RGB（所有照明條件） | 每張 32/0/0（TP/FP/FN），包括灰階相同的異色物件 |
| `bl_gray_crowded`（36 物件、尺度 0.8–1.25） | 36/0/0 |
| `bl_gray_gloss_shadow` | 8/1/1；遮擋＋投影的物件之一失敗 |
| `bl_gray_tilt`（中心距離 ≤10 px） | 0–35° 全部找到，分數由 0.998 降到 0.713；40° 漏失 |
| `bl_gray_perspective`（中心距離 ≤10 px） | 8/9 |
| `bl_sym_scene`（角度以 180° 取模） | 9/9 |
| `bl_rgb_white`／`warm3200`／`mixed` | 各 12/0/0 |

**同色判斷（只看已找到的物件）**

- 既有色度驗證的 `mean_distance`：中性光下 AUC 0.961，以中性光同色距離的第 95 百分位當門檻，同色接受 90%、異色（含 ΔE≈2 的灰階相同色）全部拒絕。照明一變就失效：白平衡偏移、局部暖光、過曝、JPEG 下，同色接受率掉到 0–5%；AUC 降到 0.66–0.87。
- RGB NCC 的 AUC 約 0.63–0.69，灰階 NCC 低於 0.5：兩者都不能拿來判色，與 [runner 文件](COLOR_VERIFY_RUNNER.md) 的推導一致。
- Blender：白光下同色距離 0.006、最近的異色 0.013，可以分開；3200K 暖光下同色距離 0.085 反而大於部分異色（0.078）；混合光下有一個同色物件位在暖光側，距離 0.052，大於色相差 ΔE4 的異色（0.015）。

**照明校正探針**（`testbed_colour_probe.py`，用 GT 對齊，只評估判色）：在模板的顏色區算色度，比較原始色度、全域參考卡校正，以及用物件自己的米白面板做 von Kries 校正。物件內白參考讓合成白平衡偏移的 AUC 從 0.69–0.73 回到 1.000、局部暖光從 0.78 提升到 0.986，Blender 暖光從 0.500 回到 1.000、混合光從 0.938 回到 1.000。限制：過曝時面板飽和（AUC 0.898）；只用中性光定的門檻在其他照明仍過緊，需要以多種照明下的正常樣本校準。只在顏色區計算（遮罩）也比對整個模板平均更好分；runner 已支援 chroma 遮罩。這是探針，不是新的驗證器實作；實作與事前登錄的評估見[第46輪](ILLUMINANT_CHROMA_zh-TW.md)（分得開，但事前門檻規則未通過）。

## 其他說明

- Blender 光源能量經過一次曝光校正：768² 試渲染時白色只到線性 0.10（與第 34 輪相同的幾何與能量比例），加上 `EXPOSURE_GAIN = 8.8` 後背景約 113、模板平均約 128、無飽和，渲染切片與原紋理的線性相關 0.998。
- `bl_gray_gloss_shadow` 的投影方柱從正上方也會遮住物件的一條，所以物件 1、4、7 同時有投影與部分遮擋，不是單純陰影。

## 已知限制

- 合成與 Blender 都是渲染器領域的資料，不是實際相機拍攝；Blender 的光照衰減會讓同色物件的實際顏色略有差異，評估時應另外量測渲染後的色差。
- 只有單一模板類型與兩個色系；沒有多模板庫、半透明、反光金屬或真實鏡頭畸變。
- 色差以 sRGB D65 的數學轉換計算，沒有使用相機的色彩特性描述檔。
