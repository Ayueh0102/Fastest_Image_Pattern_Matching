# NCC 的記憶體分塊與 CPU／GPU 重疊研究

2026-10-14。程式分析基準為 `4092754`；本文提出可驗證的設計，**沒有將尚未量測的分塊方案當作已實現的加速成果**。本輪已另完成 [CPU／GPU 雙槽重疊實作](COARSE_OVERLAP_zh-TW.md) 與 [多物件實驗](MANY_OBJECT_STUDY_zh-TW.md)；下列分塊 kernel 與 CPU 單 arena worker 則仍是研究方案，不能混為已實作。

我們可以把 NCC 拆成較小、可重用資料的工作單元。動態的是「哪些位置、角度、尺度需要繼續搜尋」，每一批確定後的像素統計仍是規則運算。適合借用 FlashAttention 的部分是資料搬運分析、分塊及中間結果重用；它的 online softmax 公式不能直接代替 NCC 或帶抑制的多峰搜尋。

## 1. 與 FlashAttention 的關係，以及 Apple 的差別

FlashAttention 將 attention 分塊，減少慢速記憶體與晶片內儲存之間的讀寫，避免完整中間 attention 矩陣落地。FlashAttention-2 再調整工作分配，降低非矩陣運算與執行緒群之間的溝通。兩篇支持「工作怎麼分、資料怎麼重用，與總運算量同樣重要」，並不保證所有分塊都更快。[FlashAttention 原論文](https://arxiv.org/abs/2205.14135)、[FlashAttention-2 原論文](https://arxiv.org/abs/2307.08691)。

| 層級 | 能直接控制的事情 | 不能假設的事情 |
|---|---|---|
| CPU registers／L1／L2／較末層 cache | 遍歷順序、連續配置、分塊大小、SIMD、工作集大小 | 一般 C++ 配置不能指定某陣列永久留在 L1，不能把 GPU threadgroup 當成 CPU L3 |
| Metal 每個 thread 的暫存值 | 每 thread 負責的輸出數、累加器數量、資料型別 | 暫存值不保證永遠留 register；過多可能降低 occupancy 或溢出至較慢儲存 |
| Metal threadgroup memory | 顯式配置，共同載入小塊資料，必要的 barrier | 顯式複製不保證比直接 device buffer 讀取快 |
| Metal device／constant buffer 的 cache | 空間局部性、相鄰 thread 的讀取位置、tile 排序 | 不能以 API 固定指定每次讀取落在哪一層 cache |
| Apple unified memory | CPU／GPU 存取共享資源、消除不必要的額外副本 | 共用實體記憶體不代表沒有頻寬、cache、同步與 ownership 成本 |

Apple 對 Family 9 明確說明：threadgroup、device、constant 可使用同一 cache hierarchy；若工作集已命中 cache，先複製至 threadgroup 可能增加延遲。官方也說明 shader core 的不同儲存需求會共同影響 occupancy。因此本專案應同時比較「直接讀 buffer 的小型輸出 tile」和「顯式 threadgroup staging」，不能照抄 NVIDIA 的 SRAM／HBM 假設。[Shader 效能建議](https://developer.apple.com/videos/play/tech-talks/111373/)、[Apple GPU profiling 與動態記憶體](https://developer.apple.com/videos/play/tech-talks/111374/)。

實作時查詢 device 的 `maxThreadgroupMemoryLength`、pipeline 的 `staticThreadgroupMemoryLength`、`threadExecutionWidth`、`maxTotalThreadsPerThreadgroup`，並檢查額外動態配置；不硬編碼某台機器的可用 cache 大小。[配置限制](https://developer.apple.com/documentation/metal/mtldevice/maxthreadgroupmemorylength)、[靜態配置量](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/staticthreadgroupmemorylength)、[threadgroup 尺寸](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes)。

## 2. NCC 可以分塊的數學理由

令模板有 \(N=w_t h_t\) 個灰階像素，位移為 \(p\)，視窗像素為 \(I_{p,i}\)，模板為 \(T_i\)。每個位置只需三個可加總統計：

\[
S_p=\sum_i I_{p,i},\qquad Q_p=\sum_i I_{p,i}^2,\qquad P_p=\sum_i I_{p,i}T_i.
\]

模板可預先計算 \(S_T=\sum_i T_i\)、\(Q_T=\sum_i T_i^2\)。非退化情況的數學式為：

\[
\rho_p=\frac{P_p-S_p S_T/N}
 {\sqrt{(Q_p-S_p^2/N)(Q_T-S_T^2/N)}}.
\]

將模板索引分成互不重疊的 \(K_j\)，有

\[
(S_p,Q_p,P_p)=\sum_j\left(
\sum_{i\in K_j}I_{p,i},\quad
\sum_{i\in K_j}I_{p,i}^2,\quad
\sum_{i\in K_j}I_{p,i}T_i\right).
\]

這不是近似：只要整數累加不溢位，區塊大小、整數加總順序不改變三個統計。完成後再呼叫既有 CPU double 正規化即可保持當前輸出。**數學等價不等於任意浮點改寫逐位相同**；目前 `templ.mean`、`templ.norm` 的學習方式、variance guard、分母與飽和處理仍要保留原運算順序。[既有 CPU 實作](../src/ncc.cpp)、[正規化規則](../src/internal.hpp)、[Metal 統計與 CPU 正規化](../experiments/coarse_metal.mm)。

對 uint8 輸入：

\[
0\le S_p\le255N,\qquad 0\le Q_p,P_p\le65025N.
\]

故 \(65025N\le2^{32}-1\) 時可用 uint32；否則使用 uint64。非負部分和不大於總和，因此同一界限也保護分塊累加。目前 backend 另要求 \(65025N\le2^{53}\)，讓統計轉 binary64 保持精確整數。uint64 SIMD 歸約已有 16／16／32-bit limb 實作，可重用但須保留 SIMD width 與進位界限。[完整 kernels](../experiments/coarse_ncc.metal)。

## 3. 真正值得切的是相鄰視窗，而非只切單一視窗

假設一個工作塊計算 \(B_x\times B_y\) 個相鄰輸出，當前模板區塊為 \(K_x\times K_y\)。每個輸出獨自讀取時，source 像素讀取指令數為：

\[
L_{\rm independent}=B_x B_y K_x K_y.
\]

所有輸出需要的 source 聯集只有：

\[
L_{\rm union}=(B_x+K_x-1)(B_y+K_y-1).
\]

因此能在該工作塊內重用的理想倍數為：

\[
R=\frac{B_x B_y K_x K_y}
 {(B_x+K_x-1)(B_y+K_y-1)}.
\]

這是**邏輯讀取重用機會**，不是實測 DRAM 頻寬或速度倍數。現有相鄰 GPU threads 可能已被硬體 cache 服務，多個 tile 之間也會重讀 halo。若模板拆成多個 \(K_j\)，不同模板區塊的 source 聯集可能再次重疊，不能宣稱整張 source 每個像素只讀一次。

若 source 與模板皆 uint8，顯式 threadgroup staging 的基本容量為

\[
M_{\rm shared}=(B_x+K_x-1)(B_y+K_y-1)+K_xK_y
\]

bytes，加上 alignment／padding。三個統計的總累加器 payload 為

\[
M_{\rm accum}=3a B_xB_y,\qquad a\in\{4,8\}.
\]

累加器分散在 threads 的私有變數；只有真的存入 threadgroup 時才計入該 API 的 threadgroup 配額，但它們仍影響實際晶片內工作集與 occupancy。若輸入 tile 也雙緩衝，`M_shared` 約加倍，不能漏算。

例如 \(B_x=B_y=8\)、\(K_x=K_y=16\)：source 聯集 529 bytes、模板 256 bytes，uint32 累加器 payload 768 bytes。基本工作資料約 1,553 bytes，不含編譯器與控制開銷；對應的 source 獨立讀取為 16,384 次，聯集重用比例約 31 倍。此數字僅說明分塊的結構，**不代表實測能快 31 倍**。

一個低風險的初始 kernel 是每 thread 保持一個或少量相鄰輸出，對模板區塊依序累加，讓同組 threads 在同一步讀取相鄰 source 位址；先利用 implicit cache。第二個版本才將同一聯集合作搬到 threadgroup，比較節省的讀取是否足以抵銷載入、barrier、索引與 occupancy 代價。這與上一輪「一個完整 SIMD group 只處理一個視窗」是不同的工作分配。

大模板可沿模板列分條，或兩維切塊；CPU 路徑也可用同樣的 output tile／template strip，讓多個相鄰 dot products 共用 source cache lines。CPU 已有 integral sums／squares，因此 CPU 優化主要針對 dot product，不應重新逐像素計算已被積分圖省掉的統計。

## 4. 運算量、頻寬與何時可能有收益

若有 \(O\) 個輸出、模板面積 \(N\)，目前 GPU 完整統計約需 \(5ON\) 個基本整數算術操作：S 的一次加法、Q 的乘加、P 的乘加；未含位址、迴圈、歸約等成本。不能把此數字直接當成裝置宣告的 FP32 FLOPs。

對某實作的有效算術吞吐 \(F\)、瓶頸層頻寬 \(BW\)、該層實際流量 \(D\)，理想下界是

\[
t\gtrsim\max\left(\frac{5ON}{F},\frac{D}{BW}\right),
\]

再加提交、同步、barrier 與正規化等成本。切 tile 主要減少某層的 D，通常不減少 O×N。若已是整數 ALU／位址計算受限，搬到 threadgroup 不一定有用。應用 Metal counters／trace 檢查 ALU、cache／MMU、occupancy 與空白區間，而非僅由理論像素數判定頻寬受限。[Apple compute 工作分配與 locality](https://developer.apple.com/videos/play/wwdc2022/10159/)、[Metal cache 與工作集](https://developer.apple.com/videos/play/tech-talks/10580/)。

目前 `4092754` 有直接證據：六物件原 coarse mode0 device 約 3.62 ms，改成單視窗 SIMD 協作後約 16.12 ms；upload 約 0.10 ms、command 數與視窗數不變。這支持「該 SIMD 分工不划算」，**沒有證明原 mode0 是 DRAM 受限，也沒有證明所有分塊都失敗**。[既有量測與限制](COARSE_SELECTION_AND_PARALLEL_zh-TW.md)。

可以另外研究 summed-area table／滑動盒形濾波求 S、Q，只留下 P 的直接相關運算。代價是額外 pass、prefix-scan 同步及儲存；每個角度的 rotated source 都不同，要逐角度付出此成本。CPU 已用積分圖，但不能據此斷言 GPU 上也一定快。大模板且很多輸出時，FFT 求 P 可能降低算術量；其浮點誤差、padding、FFT setup、S/Q 與分數門檻都需獨立評估，不能直接宣稱等同目前逐位精確路徑。

## 5. 旋轉、尺度、RGB 與超過 100 個物件

粗搜尋在每個 angle／scale 的畫布生成後，是規則的二維滑窗；適合 output tiling。細搜尋每一層先收集還活著的候選，同尺寸 ROI 可批次處理；不同候選的座標、角度只是描述資料。下一層依賴上一層結果，須維持層間依賴，但每一層內可並行。`fineBatchSize=512` 是工作分批限制，並不等於只能找到 512 個物件。

物件數不會在固定圖尺寸／角度／尺度下直接增加 coarse 每張 map 的 NCC 視窗數；為了找更多物件通常提高 `maxTargets`，會增加粗選峰與後續 fine 工作。基準版每個 angle 最多保留 `maxTargets+5` 個粗候選，並在 NMS 前精修所有候選。若有 A 個 angle、S 個 scale，粗候選的簡單上界是 \(S A(\mathrm{maxTargets}+5)\)，實際受 threshold 與抑制影響。100 個真物件可以產生遠多於 100 個角度候選，不能用物件數線性推估時間；需要同時記錄 proposals、fine maps、fine pixels／windows、NMS 時間。

RGB 可逐通道累加。若採每通道去均值，分子是

\[
\sum_{c=1}^{3}\left(P_{p,c}-\frac{S_{p,c}S_{T,c}}{N}\right),
\]

分母為 \(\sqrt{[\sum_c(Q_{p,c}-S_{p,c}^2/N)][\sum_c(Q_{T,c}-S_{T,c}^2/N)]}\)。這與把三通道攤平成長度 3N、只減一個全域均值是不同定義，須先固定語義。可逐通道分條以維持 tile 容量，或三通道同載以重用幾何座標。運算、流量及累加器都會增加；GPU 更易有足夠工作，但**三通道本身不保證 GPU 勝 CPU**，小圖仍可能被提交成本支配。此處 RGB 是推導方向，不是宣稱目前灰階 matcher 已完整支持 RGB NCC。

旋轉與 NCC fusion 也要計算 reuse：每個視窗自行旋轉會重複做插值；先將旋轉後小 tile 共用可能較合理。現有 fine Q28 guard 會標記 rounding 邊界並由 CPU 原 double 公式修復，若直接 fusion 跳過修復，就失去目前精確契約。先保留 warp／repair 邊界，再研究已確認安全的融合範圍。[fine sampling 保護](../experiments/fine_sampling.hpp)、[fine Metal kernels](../experiments/coarse_ncc.metal)。

## 6. CPU／GPU 重疊是不同層次的改善

在基準 `4092754`，粗搜尋批次仍依序執行：CPU rotation → Metal upload／NCC／wait → CPU double normalization／peaks。有兩種可分別量測的排程：

1. **CPU 預做下一批畫布**：一個 worker 獨占 backend 完成第 n 批 upload／NCC／normalization，主執行緒生成第 n+1 批 Canvas；結果依原角度序收回後才重用 backend。這可以只增加 CPU 畫布前瞻容量，保留單組 Metal source／statistics arena。
2. **真正多個 GPU 批次在途**：第 n+1 批先提交，同時 CPU 消費 n 的結果；需要明確 source／statistics slot 或已證明的資源生命週期。雙槽可能增加一組 source arena 與 statistics arena；基準上限分別為 64 MiB、16 MiB，另須計算 CPU canvases、score maps 與 metadata。大畫布 fallback、retained capacity 及短批次需各別測試，不能把 80 MiB 當成整個流程的統一記憶體上限。

兩種方式都要保證 GPU 使用中的來源不被覆寫、GPU 完成前 CPU 不讀結果；shared storage 不會自動消除這些規則。[Apple shared resource 同步契約](https://developer.apple.com/documentation/metal/mtlstoragemode/shared)。本輪實作選擇第二種雙槽 begin／finish，讓 CPU 下一批旋轉和上一批正規化／選峰都有重疊機會；第一種單 arena worker 仍是日後降低記憶體的對照方案。

令每批 CPU 可排程工作為 C、GPU 工作為 G，m 批串行約為 \(m(C+G)\)。理想穩態每批下界為 \(\max(C,G)\)，另加管線填滿／排空；只考慮這兩類工作的加速上限為 \((C+G)/\max(C,G)\le2\)。實際還有 fine、NMS 等其他時間，以及 CPU／GPU 共享頻寬競爭，所以不是整體保證快兩倍。

host wait 是當下還沒完成的剩餘時間，重疊後可能降低，即使 device 執行時間不變。應同時記錄 CPU rotation 起訖、command GPU start/end、frame wall time，驗證時間區間真正相交。**wait 與 device time 重疊，不可相加；重疊後各 stage 的工作時間也不一定能相加還原總牆鐘。**

## 7. 可比較的後續方案與精度界線

| 次序 | 變因 | 必須一起觀察 |
|---|---|---|
| 1 | 同 kernel、同候選數，CPU 預做下一批 rotation | 總時間、CPU/GPU timeline、額外 CPU canvas／RSS、精確 ordered output |
| 2 | 一 thread／視窗維持，改 output tile／template strip | device time、cache／ALU 指標、整數統計逐值相同 |
| 3 | 對同一 tile 加 threadgroup staging | threadgroup 容量、occupancy、barrier 成本、相對步驟 2 的收益 |
| 4 | 依模板尺寸、map 大小、ROI 數分桶 | setup 成本、不同尺寸的交叉點、未參與選參的驗證集 |
| 5 | 必要時再試 GPU 多批次在途、S/Q 前綴和或融合 | arena 上限、完整生命週期、fallback、總時間，而非只看 kernel |

上表是研究拆分，並非本輪完成清單；本輪只實作上述雙槽排程，沒有新增 tiled kernel。

小／中／大模板、0°／全角度、單／多尺度、6／12／超過 100 個物件應分開列。主要性能用無 profile 的正反順序多輪結果；profile 用於原因分析。每次固定輸入、搜尋範圍、候選上限與分數門檻，保留 cold/setup、穩態時間及資料版本。先比較舊／新同 build 控制，再決定是否凍結採用。

精確路徑至少保持：三個整數統計、模板 mean/norm、CPU double 正規化運算序、退化圖與閾值邊界、signed zero／非有限值、分數同值時的 y/x 優先序、角度候選的 stable order、既有抑制矩形和 NMS。當前選峰樹雖已分 32×32 tile，仍保留完整 score map；因下一個峰依賴之前峰的抑制，不能只保留每 tile 一個初始最大值後丟掉其餘分數。若要不存完整 map，需要可重算受影響 tile 的完整設計與時間代價，這是後續不同實驗。

目前能成立的判斷是：**動態搜尋不是分塊與 GPU 並行的障礙；限制在資料重用、同步依賴、精確性與記憶體資源的平衡。先用重疊減少空等，再量測適合此 NCC 負載的 tile，比直接增加單一視窗的 threads 更有研究價值。**
