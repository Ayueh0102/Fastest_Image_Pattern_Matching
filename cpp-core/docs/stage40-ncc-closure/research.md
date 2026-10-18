# Stage 40：精確 Metal NCC 的文獻篩選與實驗邊界

本筆記把可參考的數學與 GPU 實作對應到本專案的細搜尋，**文獻篩選不等於在本機重現作者效能**。本階段要結束的是「既有灰階、旋轉／金字塔、每個角度 7×7 局部搜尋、與原 GPU 結果完全一致」這一組有界實驗；並不宣稱演算法已達理論極限。資料查核日期：2026-10-19。

## 為何選這三個實驗

以下保留實驗前的選題推理；三個變體現已完成比較，mode4通過主要目標、mode3較慢，正式結果與完整封存見[本階段結案報告](../GPU_NCC_CLOSURE_zh-TW.md)。

基線 `full_gpu_fine.metal` 每個候選使用 256 threads，49 個位移各由 4 threads 計算；ROI 能放進 16 KiB 時共享一次旋轉取樣，超過時可用 Stage 39 分列快取。據此選出三個可以維持精確整數 moments 的方向：

| 方向 | 改變的工作 | 本階段處理方式 |
|---|---|---|
| 有界 uint32 累加、uint64 合併 | 降低內迴圈 64 位元運算 | 實作、差分測試、正式時間／記憶體／準確度比較 |
| 重疊窗口 S/Q 統計重用 | 避免 49 次重算像素總和與平方和 | 與上一項分開比較；不能省略 dot product |
| SIMD-group 改變窗口分工 | 比較 4 lanes/window 與 32 lanes/window 的載入與歸約 | 額外對照變體；結果慢也保留 |

最接近第三項的新文獻是 Maděra、Šmelko、Kruliš 的 **Efficient GPU-accelerated parallel cross-correlation**（JPDC 199，105054，2025）。作者研究 direct correlation 的 warp shuffle、暫存器重用與工作分配，並比較不同輸入形狀下的 FFT 分界；作者實驗是 CUDA cross-correlation，不能把其加速倍率套用到本案完整 NCC。可參考的是「分工方式必須實測」，而非固定一個通用最快 kernel。[論文](https://doi.org/10.1016/j.jpdc.2025.105054)、[作者重現套件](https://github.com/asmelko/jpdc23-artifact/tree/21bfe6d09170e5f297eeff5753f74588773fbaad)

## 可直接使用的數學

以下是依照本專案資料型別推導的界限與等式，並非抄錄第三方程式。令模板面積為 n，模板像素為 T，旋轉後來源窗口為 X：

\[
 S=\sum X_i,\quad Q=\sum X_i^2,\quad P=\sum X_i T_i,
 \quad \rho=\frac{nP-S S_T}{\sqrt{(nQ-S^2)(nQ_T-S_T^2)}}.
\]

### 1. 有界 32 位元區塊

gray8 像素乘積最大 255²＝65,025。因此一個 uint32 區塊最多可以包含

\[
 b\le\left\lfloor\frac{2^{32}-1}{65025}\right\rfloor=66,051
\]

個乘積；例如 65,536 個時上界為 4,261,478,400。實作選擇沿用四條帶，當 ceil(n/4)≤66,051 時以 uint32 累加，每條帶轉 uint64 後才合併；更大模板保留 uint64 路徑，沒有另加入固定長度分塊。這個界限要套在**歸約後的總項數**，不能只有每個 lane 安全卻在 `simd_sum(uint)` 溢位。原主機對 n²·65,025 的 uint64 驗證仍須保留。最終 `fg_ncc` 的浮點運算順序不變，整數加法的重分組才能維持原 GPU 分數與 tie 決策。

### 2. 重疊統計，不是重疊 dot 的直接相減

Lewis 的經典方法把局部均值與能量改由 running sums／integral images 計算；相關分子可以另選 spatial 或 transform 實作。[作者原文：Fast Normalized Cross-Correlation](https://www.scribblethink.org/Work/nvisionInterface/nip.html)

對本案已旋轉成離散 gray8 的 ROI，可先對每列計算寬 w 的窗口和 H，再對高 h 合併：

\[
 H(x+1,y)=H(x,y)-X(x,y)+X(x+w,y),
\]
\[
 S(x,y+1)=S(x,y)-H(x,y)+H(x,y+h).
\]

把 X 換成 X² 就得到 Q。用「先減掉確定包含的項，再加新項」可避免無號中間值負溢位。本輪 mode 3 選擇較小改動：各自初始化 7 個水平窗口，再沿垂直方向滑動 S/Q；只用既有 partial storage，不配置整個 ROI 的 integral image。這些都只是 S/Q 的等價實作，必須把初始化、共享記憶體與 barrier 成本一起量測。

固定模板的 `P(x)=Σ X(x+i)T(i)` 不能只刪最左一列、加最右一列：平移後，重疊來源像素對應到不同的 T 權重。除非模板有另行證明的結構，49 組 dot 仍要計算。

Chang 等人的 **Z²-ZNCC**（2022，2025 修訂）利用 stereo 中兩張 patch 一起移動時的乘積場，進行 zigzag 更新及 register 重用。本案只移動來源窗口，故其 dot 的增量公式不直接成立；S/Q 的重用思想仍適用。[論文與公式 19–20](https://arxiv.org/html/2212.00476v2)、[作者程式](https://github.com/changqiong/z2zncc/tree/4da3d130141ee5fda189e2a144cea088df09af19)

### 3. SIMD-group 映射

本輪比較的映射是 256 threads＝8 個 32-lane groups，每組處理一個 window，49 windows 分 7 輪。它減少每個 lane 的串行 dot 長度，但增加窗口輪次與歸約，**不能先假設較快**。所有 lanes 要一致進出 SIMD 操作；跨 group 重用 ROI 必須保留 threadgroup barrier。實際 group 寬度從 pipeline 查詢，或明確驗證 32-lane 前提。[Apple：SIMD groups](https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups)、[Apple WWDC22：Scale compute workloads across Apple GPUs](https://developer.apple.com/videos/play/wwdc2022/10159/)

此變體對應作者重現套件的 [warp-per-shift 實驗](https://github.com/asmelko/jpdc23-artifact/blob/21bfe6d09170e5f297eeff5753f74588773fbaad/repo/cross-corr/src/naive_warp_per_shift.cu)。本專案自行實作 gray8 整數版本，並不直接移植其 CUDA 浮點 kernel。

## 篩選後沒有直接放進本階段的方向

| 方法 | 有價值的部分 | 未作為本階段等價替換的理由 |
|---|---|---|
| FFT correlation | 大面積密集位移搜尋、同模板頻譜重用 | 本案每角度只輸出 49 點；每個動態旋轉 ROI 仍需 transform 與 inverse。一般 FP32 FFT 也不保證原整數 moments 完全一致。尚未在本機量出 crossover，不能宣稱 FFT 永遠較慢。 |
| 矩形基底／segmented NCC | 模板可預先近似成少數矩形，積分圖快速求相關 | 少量基底時改變模板與 score；完整表示雖可精確，卻無法保證維持少量基底優勢。需另立近似誤差／漏檢實驗。 |
| 旋轉 tensor template | 將旋轉資訊預編進模板表示 | 紙上連續模型、支撐域及濾波前提不同於目前矩形 gray8 取樣；不是此 kernel 的逐位元等價重排。 |
| FP16／矩陣乘法／DL features | 批次多模板或學習特徵可能受益 | 本階段要求精確整數 moments 與原 score 排序；普通 FP32 累加在 2²⁴ 以外也不能表示每個整數。要另證明分拆後精確性，不能只因 GPU 有矩陣 API 就改寫。 |
| 提前放棄／低秩近似／稀疏取樣 | 可降低每窗口乘加量 | 會改變完整 score maps；subpixel 擬合會用鄰域分數。安全界限及必要重算尚無本輪實作，不能宣稱已完成。 |
| 跨候選／角度 memoization | 完全相同的取樣輸入可重用 | 必須含 anchor、level、angle、template 等完整 key；大量物件的主要設定 subpixel=false，不應假設大量重複。先前 profiling 與本階段資料未證明此為首要瓶頸。 |

OpenCV 官方 CUDA `match_template.cpp` 把 correlation 與 integral statistics 分離，也有 direct/FFT 選擇。這支持分開設計 numerator、normalization 與工作形狀，但其面積門檻不能直接移植到 Metal 的 7×7 輸出。[OpenCV 4.12.0 原始碼](https://github.com/opencv/opencv_contrib/blob/4.12.0/modules/cudaimgproc/src/match_template.cpp)

Briechle–Hanebeck（2001）以矩形基底近似模板；2025 的 Marušić 等人則依模板區域變異數做 split-and-merge，以矩形均值近似。兩者都是值得另開「可容許近似誤差」研究的來源，並非本輪 exact NCC 已實現的功能。[2001 作者機構 PDF](https://isas.iar.kit.edu/pdf/SPIE01_BriechleHanebeck_CrossCorr.pdf)、[2025 Segmented NCC](https://arxiv.org/abs/2502.01286)

Almira、Phelippeau、Martinez-Sanchez 的旋轉 tensor 方法將旋轉模板整合成對稱張量。其理論使用與旋轉可交換的支撐／內積運算，實際處理亦討論低通濾波；本案的矩形離散取樣不自動滿足這些等價前提。論文頁記載 2024 期刊出版，未提供本專案 Metal 性能或尺度搜尋驗證。[原文](https://arxiv.org/html/2311.07561v1)、[期刊 DOI](https://doi.org/10.1007/s12190-024-02157-6)

FlashAttention 可借鑑的是用 tiling 減少大中間矩陣與外部記憶體存取；本案已保留小 ROI 在 threadgroup memory，下一步須量 NCC 內迴圈和共享存取。它不是 NCC 演算法，不能把其 exact attention 或性能結論轉稱為本案的證明。[作者論文](https://arxiv.org/abs/2205.14135)

## 原始碼與授權查核

本研究只引用方法與短摘要；未複製上述專案程式到產品。

| 來源 | 固定版本／查核 | 使用範圍 |
|---|---|---|
| JPDC 作者套件 | commit `21bfe6d09170e5f297eeff5753f74588773fbaad`，[MIT](https://github.com/asmelko/jpdc23-artifact/blob/21bfe6d09170e5f297eeff5753f74588773fbaad/LICENSE) | 研究工作映射與實驗設計；未複製程式 |
| Z²-ZNCC | commit `4da3d130141ee5fda189e2a144cea088df09af19`，[LICENSE 檔含 MIT 文字](https://github.com/changqiong/z2zncc/blob/4da3d130141ee5fda189e2a144cea088df09af19/LICENSE)，GitHub 自動分類為 Other | 研究公式與適用前提；未複製程式 |
| OpenCV contrib | tag `4.12.0`，[root Apache-2.0](https://github.com/opencv/opencv_contrib/blob/4.12.0/LICENSE)；上述舊 CUDA 檔案另帶三條件 BSD 形式聲明 | 閱讀設計；未複製程式，未宣稱可刪檔頭授權 |
| Simon-Bertrand FFT ZNCC | commit `f1c7206811b20beb18ec2a1d5dc6a46416cb1f18`；查核時 repo tree 未見 LICENSE | 只看 [README 的 cuFFT／融合統計描述](https://github.com/Simon-Bertrand/CUDA-FastNormalizedCrossCorrelation/tree/f1c7206811b20beb18ec2a1d5dc6a46416cb1f18)；不採用程式，也未重現其性能主張 |

## 本大階段的結案標準

結案需包含三種精確變體的成敗記錄、128／512 原搜尋深度與到原圖深度、低負載控制、先前 8K 控制、相同輸入的有序結果比較、GT、時間／RSS／命名 GPU buffers、獨立 profiling 與未 profile 的正式比較，以及源碼／輸入／原始 log 的重現封存。完整實驗結果放在本階段主報告；本筆記不以文獻推論代替測量。

未驗證的 RGB 完整常駐路徑、CUDA、實拍長序列泛化、近似 tensor／FFT 改版，是**邊界外研究項目**，不是已完成。當上述有界矩陣跑完、所有失敗原因與保留模式可查核，即可結束這一大階段，避免把「還能想到新方法」誤當成永遠不能結案。
