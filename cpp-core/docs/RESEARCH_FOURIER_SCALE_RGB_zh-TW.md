# NCC 的旋轉尺度與 RGB 加速推導

研究日期：2026-10-08。目標是在現有 NCC、金字塔及多目標搜尋的基礎上，減少角度／尺度枚舉與反覆影像取樣，再將規則的大量運算交給 CUDA 或 Metal。近期最適合驗證的架構是「涵蓋尺度的候選搜尋 → 局部 Fourier–Mellin 提案 → 完整 NCC 精修與驗證」。可轉向基底另作研究分支，先量測模板能否用少量基底表示。

現有正式 C++ 仍是灰階、固定尺度的比對器。此文件附數學推導、NumPy 研究原型和數值驗證；沒有把研究原型當作已完成的 GPU、多尺度或 RGB GUI 功能。

2026-10-12 後續實驗：已加入獨立 C++／Metal 批次取樣＋NCC 核心，並量測 CPU Fourier 提案與 GPU 精修的時間、記憶體及真值誤差；見 [實驗方法](EXPERIMENTS_zh-TW.md) 與 [111 組結果](EXPERIMENT_RESULTS_zh-TW.md)。本文件以下仍保留最初的推導與原型觀察；GPU 加速和失敗條件以新報告為準，正式 GUI 尚未切換研究後端。

## 相關原始研究

| 研究 | 支持的方向 | 接入本專案時的界線 |
|---|---|---|
| [Reddy 與 Chatterji 1996，An FFT-Based Technique for Translation, Rotation, and Scale-Invariant Image Registration](https://dev.ipol.im/~reyotero/bib/bib_all/1996_Reddy_Chatterji_fft_based_trans_rot_scale_invar_registr.pdf) | Fourier magnitude、log-polar、phase correlation 估計相似變換 | 整體影像配準不能直接取代背景中的多目標搜尋 |
| [Freeman 與 Adelson 1991，The Design and Use of Steerable Filters](https://people.csail.mit.edu/billf/steerpaper.html) | 基底線性組合合成任意方向的濾波器 | 有限基底的精確性取決於濾波器所屬子空間 |
| [Simoncelli 等人 1992，Shiftable Multi-Scale Transforms](https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Simoncelli91) | 位置、方向及尺度的 shiftability 與取樣條件 | 需處理混疊，不能只靠增加插值密度 |
| [Ghosh 與 Gupta 2019，Scale Steerable Filters for Locally Scale-Invariant Convolutional Neural Networks](https://arxiv.org/html/1906.03861v1) | log-radial harmonics 的尺度轉向與證明 | 可借用基底數學；其 CNN 成果不是本專案 NCC 效能證據 |
| [Padfield 2010，Masked FFT Registration](https://iacl.ece.jhu.edu/proceedings/cvpr2010/papers/1815.pdf) | 遮罩、有效交集與 NCC 統計量一起進入 FFT | 遮罩不能只靠補零代替；重疊樣本數與平均都要算對 |
| [Guizar-Sicairos 等人 2008，Efficient Subpixel Image Registration Algorithms](https://opg.optica.org/ol/abstract.cfm?URI=ol-33-2-156) | 在相關峰值附近計算局部上採樣 DFT | 解決細平移取樣成本，本身不是完整旋轉尺度搜尋 |
| [Sah 等人 2012，GPU Accelerated Real Time Rotation, Scale and Translation Invariant Image Registration Method](https://naos-be.zcu.cz/server/api/core/bitstreams/c828a969-7e8c-4617-932b-5a2004d5d5f7/content) | FFT 配準的 CUDA／OpenCL 實作先例 | 舊硬體與其 CPU 基準的加速倍數不能套到本專案 |

以上方法不是新提出的演算法。以下是按本專案目標重新整理的推導與設計；是否構成新的研究貢獻，需要更完整的文獻與實驗比較。

## 問題與假設

令模板為 T、來源影像為 I、平移為 t、旋轉為 θ、等向尺度為 s>0。推導採 x 向右、y 向上的座標，Rθ 為逆時針旋轉；程式接回畫面座標時須明確轉換 y 軸與角度。

先假設單個候選區域內的影像滿足

$$
I(x)=aT\left(s^{-1}R_{-\theta}(x-t)\right)+b+\eta(x),\qquad a>0.
$$

幾何先限於相似變換，不含透視、剪切、各向異性伸縮及非剛體變形。推導中的精確等式先在連續、無裁切的域成立；離散圖片還有插值、取樣、遮擋與背景等誤差。亮度偏移 b 在理想無限域只影響 DC；有限視窗中的偏移與窗函數則不只影響 DC，須先做背景／平均處理。

對多個目標，來源是各物件與背景的疊加，一般不存在一組 t、θ、s 能將整張來源圖變成模板。因此本研究保留位置候選與多目標篩選，只讓每個候選的幾何估計更便宜。

若相機配置能提供尺度區間、搜尋 ROI 或相鄰影格位移範圍，可將其寫成明確的參數限制；例如尺度在 [smin,smax] 時，只需搜尋 α∈[log smin,log smax]。這些先驗縮小搜尋域，並不提高域外物件的召回率。弱透視、共同顏色增益、低遮擋等假設也須用資料驗證，不能為了速度直接當成事實。

## FFT 如何精確計算 NCC

定義相關運算

$$
(f\star g)(p)=\sum_u f(p+u)\overline{g(u)}.
$$

先考慮固定姿態與二值模板遮罩 M。N=ΣM、μT=ΣMT/N，中心化模板為 Tc=M(T−μT)。來源在位置 p 的统计量為

$$
S(p)=I\star M,\qquad Q(p)=I^2\star M,\qquad C(p)=I\star T_c.
$$

由 ΣTc=0 得到

$$
\rho(p)=\frac{C(p)}{\sqrt{\left[Q(p)-S(p)^2/N\right]E_T}},
\qquad E_T=\sum_u M(u)(T(u)-\mu_T)^2.
$$

這裡分子仍是普通相關，因此

$$
C=\mathcal F^{-1}\left(\mathcal F(I)\overline{\mathcal F(T_c)}\right).
$$

S、Q 也可用相同相關定理計算。當 M 是完整軸對齊矩形時，積分圖通常是另一個值得比較的計算方式。這是 [Lewis 快速 NCC](https://scribblethink.org/Work/nvisionInterface/nip.html) 的核心分離方式；本專案已實作矩形積分圖，但相關項仍為直接點積。

FFT 必須補零至至少「來源尺寸＋核尺寸−1」，或使用正確的分塊線性相關，最後只取有效位置；直接循環相關會讓影像兩端互相匹配。FFT 改变浮點加總順序，所以數學等價不等於逐位元一致。

若旋轉模板，M 也要一起旋轉，空白角落不參與計分。任意旋轉遮罩的 S、Q 不能用一般矩形積分圖四次查表得到。若來源也有有效性遮罩，交集與 N 會隨位置改變，需使用完整的雙遮罩推導；附帶原型只驗證模板二值遮罩、來源全有效的情形。

FFT 在固定姿態下同時計算許多平移位置，並沒有自動消除所有角度與尺度。現有精搜每個姿態只有 7×7 個輸出，不能只依模板面積判斷 FFT 一定較快。

## 傅立葉如何把旋轉尺度變成位移

忽略偏移與噪聲，令 A=sRθ，採用 F(ω)=∫f(x)exp(−iωᵀx)dx。換元 x=Ay+t，且 dx=s²dy：

$$
\widehat I(\omega)=a s^2e^{-i\omega^Tt}\widehat T(A^T\omega)
=a s^2e^{-i\omega^Tt}\widehat T(sR_{-\theta}\omega).
$$

取幅值後，平移只造成的相位消失。以 ω=eᵘ(cosφ,sinφ)、α=log s 表示頻率，定義

$$
L_T(u,\phi)=\log|\widehat T(e^u\cos\phi,e^u\sin\phi)|.
$$

在非零頻譜上有

$$
L_I(u,\phi)=\log a+2\alpha+L_T(u+\alpha,\phi-\theta).
$$

移除常數項後，旋轉尺度估計變成二維位移估計：位移為 δu=−α、δφ=θ。令 H 為 log-polar 陣列的二維 Fourier transform，則

$$
P=\frac{H_I\overline{H_T}}{|H_I\overline{H_T}|},\qquad
\operatorname*{argmax}\mathcal F^{-1}(P)=(\delta_u,\delta_\phi),
$$

$$
s=e^{-\delta_u},\qquad\theta=\delta_\phi.
$$

程式中峰值位置先乘上 log-radius／角度的取樣間距。零頻譜不能相除，必須遮罩或正則化；使用 log(max(magnitude,ε))、徑向去平均等都是數值處理，會使有限陣列上的關係成為近似。

這是 [Reddy 與 Chatterji](https://dev.ipol.im/~reyotero/bib/bib_all/1996_Reddy_Chatterji_fft_based_trans_rot_scale_invar_registr.pdf) 路線在本文件座標慣例下的換元推導。取得旋轉尺度後，再用空間域／頻域相關估計平移，最後用原始像素 NCC 驗證，不能把 phase correlation 的峰值當作 NCC 分數。

實際接入時要處理四項限制。實數影像的 Fourier magnitude 滿足 M(ω)=M(−ω)，因此存在 180° 歧義，至少保留 θ 與 θ+π 兩個空間假設。尺度的 log-radius 軸不是物理上的週期，需補零、限制尺度範圍與足夠的有效頻帶。裁切及固定窗函數會破壞理想變換關係。對稱、低紋理與背景占比高的物件，也可能產生多個或不可靠的峰值。

FMM 只能為已進入候選區域的物件估計尺度。若前段仍只用固定尺度 NCC，就可能在 FMM 之前漏掉大小不同的目標。前段必須使用涵蓋 smin 到 smax 的粗尺度組、尺度可轉向特徵，或其他經驗證的候選生成方法。

## 用基底轉換模板而非反覆旋轉像素

先看旋轉。將模板表示成角向諧波

$$
T(r,\phi)=\sum_{m=-\infty}^{\infty}a_m(r)e^{im\phi}.
$$

旋轉後只需改變係數相位：

$$
T_\theta(r,\phi)=T(r,\phi-\theta)
=\sum_m e^{-im\theta}a_m(r)e^{im\phi}.
$$

令 Bm(r,φ)=am(r)exp(imφ)，Cm(p)=I⋆Bm。在本文相關定義下，旋轉模板的相關為

$$
C_\theta(p)=\sum_m e^{im\theta}C_m(p).
$$

因此可先計算各基底的平移相關，再以複數乘加合成不同角度。有限角頻寬時，角度軸還能用逆 DFT 批次合成。這與 [Freeman 與 Adelson 的 steerability](https://people.csail.mit.edu/billf/steerpaper.html) 思路一致。任意照片不會自動落在少量基底張成的空間；截斷到 |m|≤M 是近似，或必須保留足夠多的模式。

尺度可以用同樣的換元方式理解。定義二維 L2 範數保持的尺度旋轉算子

$$
(U_{s,\theta}T)(r,\phi)=s^{-1}T(r/s,\phi-\theta).
$$

係數 s⁻¹ 是因面積元素縮放 s²；對 NCC 而言，正的整體振幅本來就會被正規化消去。令 q(u,φ)=eᵘT(eᵘ,φ)，利用 dxdy=rdrdφ=e²ᵘdudφ：

$$
\|T\|_{L^2(\mathbb R^2)}^2=\int|q(u,\phi)|^2\,du\,d\phi,
\qquad q_{s,\theta}(u,\phi)=q(u-\log s,\phi-\theta).
$$

因此對 u 與 φ 展開 Fourier 基底，可得到廣義基底

$$
B_{\nu,m}(r,\phi)=r^{-1}e^{i\nu\log r}e^{im\phi},
$$

$$
U_{s,\theta}B_{\nu,m}
=e^{-i(\nu\log s+m\theta)}B_{\nu,m}.
$$

這直接把尺度與旋轉化為係數相位。若一個離散近似使用 K 個基底，

$$
T\approx\sum_{k=1}^K a_kB_k,\qquad
C_{s,\theta}(p)\approx\sum_{k=1}^K\overline{a_k}
e^{i(\nu_k\log s+m_k\theta)}(I\star B_k)(p).
$$

可預先快取模板係數與基底頻譜，以 FFT／批次矩陣乘法計算與合成響應。複數共軛的正負號由本文件的相關定義決定；若改用卷積必須一起改。

單一 r⁻¹ 諧波在整個平面上並不是普通的有限能量模板；完整表示需以廣義 Fourier–Mellin 展開解讀。實作必須限制 rmin、rmax、频率範圍並處理中心。固定有限窗 w(r) 會使 w(r/s)≠w(r)，因此不能一面固定支撐域、一面宣稱尺度轉向仍是精確的純係數乘法。[Ghosh 與 Gupta 的尺度推導](https://arxiv.org/html/1906.03861v1) 也明確包含隨尺度改變的支撐半徑。

此外，上面只推導了相關分子。NCC 的平均、能量、遮罩與支撐域也須隨姿態正確處理，不能直接把可轉向濾波輸出當成 NCC。

## 近似誤差如何變成可驗證的篩選條件

在同一姿態、同一有效區域與同一中心化／加權方式下，令 x、t 都是單位範數向量；以未重新正規化的近似 tK 代替 t，且 εK=‖t−tK‖。則由 Cauchy–Schwarz：

$$
\rho=\langle x,t\rangle,\qquad\widetilde\rho_K=\langle x,t_K\rangle,
\qquad |\rho-\widetilde\rho_K|\le\epsilon_K.
$$

所以只有在 ρ̃K+εK<最低分數時，才能據此安全排除這個姿態；否則增加基底或回到完整 NCC。若把 tK 再正規化，必須重新計算相對於 t 的誤差界。實際裁切、重取樣與旋轉遮罩會改變殘差，因此不能拿一個 log-polar 重建誤差當作所有 Cartesian 姿態共用的保證。

這個界保護的是已評估姿態的分數判斷，不保證未搜尋角度、尺度或被金字塔候選截掉的位置。它也不自動保證 NMS 後的排序一致。

## 將精修改為四參數的 NCC 最佳化

令 p=(tx,ty,θ,α)，s=eᵅ，在固定模板格點 u 上取樣

$$
b_p(u)=I(W_p(u)),\qquad W_p(u)=t+e^\alpha R_\theta u.
$$

對固定支撐域，P 為減去平均的投影，c=Pb、z=c/‖c‖；模板中心化並正規化為固定向量 τ。NCC 為 ρ=τᵀz，亦等價於最小化

$$
E(p)=\tfrac12\|z(p)-\tau\|^2=1-\rho(p).
$$

若影像插值可微，令 J 是中心化取樣值對 p 的 Jacobian，則

$$
J_z=\frac{1}{\|c\|}(I-zz^T)J,\qquad
\nabla_p\rho=J_z^T\tau.
$$

以 v=eᵅRθu、J₂=[[0,−1],[1,0]] 表示四個未中心化偏導：

$$
\frac{\partial b}{\partial t_x}=I_x,\quad
\frac{\partial b}{\partial t_y}=I_y,\quad
\frac{\partial b}{\partial\theta}=\nabla I^TJ_2v,\quad
\frac{\partial b}{\partial\alpha}=\nabla I^Tv.
$$

梯度也要減去相同有效區域的平均。令 A=JᵀJ、h=Jᵀc、g=Jᵀτ、Ec=cᵀc，可不建立完整 Jz，直接算

$$
H=J_z^TJ_z=\frac{A-hh^T/E_c}{E_c},\qquad
d=J_z^T(\tau-z)=\frac{g}{\sqrt{E_c}}-\rho\frac{h}{E_c}.
$$

解小型系統 (H+λI)Δp=d，再檢查步長、相關分數與尺度範圍。大量像素工作只需累積統計量與小矩陣，适合 GPU reduction；四參數求解成本小，可留在 GPU 以避免每轮 CPU 同步。

這是對正規化向量目標的直接微分與 Gauss–Newton 近似；[ECC image alignment](https://sites.google.com/site/georgeevangelidis/ecc) 提供相關係數幾何最佳化的研究背景，本文未宣稱上述更新逐步等同該論文所有版本。現有 uint8 取整數旋轉目標不適合直接微分，精修分支需要明確定義浮點插值與最終評分規格。

初值必須位於可收斂範圍；重复纹理、遮擋、弱梯度與相似度差會影響收斂。使用 FMM 提案、阻尼、接受／拒絕步驟與原格點搜尋退回路徑，最後在同一姿態重新評分。不能把「每次 O(N)」誤寫成一定更快，仍需比較總迭代次數與失敗退回成本。

## RGB 的目標函數與可消除的變化

保留三個通道，對每個通道分別減平均，再合併分子與能量。固定非負權重 wc 時：

$$
\rho_{RGB}=\frac{\sum_cw_c\left[C_c-S_{I,c}S_{T,c}/N\right]}
{\sqrt{\left(\sum_cw_cV_{I,c}\right)\left(\sum_cw_cV_{T,c}\right)}},
\quad V_{I,c}=Q_{I,c}-S_{I,c}^2/N.
$$

這與三個通道的 NCC 各算一次再平均不同。[OpenCV 的多通道 CCOEFF_NORMED](https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html) 也是逐通道減平均，再跨通道累加。

若 Ic=aTc+bc 且三通道共用同一個 a>0，中心化後只差共同倍數，聯合 RGB NCC=1。若白平衡使每個通道各有 ac，聯合 NCC 通常不再等於 1。逐通道 NCC 的加權平均能消除獨立正增益，但也會放棄部分相對顏色資訊；零變異通道需有明確規則，不能直接除以零或任意給滿分。

三通道可先堆疊並乘上固定白化矩陣，使通道噪聲與權重進入同一向量目標；前一節的梯度推導不變。若只使用正交色彩矩陣且保留所有三通道、等權重，內積與範數不變，NCC 也不變。換到另一個三維正交色彩座標本身不會提高辨識力或減少計算；只留亮度或改權重才會改變目標。

若要使用線性光照增益的物理假設，應使用線性 RGB；一般 sRGB 已做非線性編碼，其數學增益假設不一定等同照明變化。是否線性化必須連同模板、圖片、插值及基準一起定義，不能悄悄改變現有分數。[sRGB 與 linear-light sRGB 定義](https://www.w3.org/TR/css-color-4/#predefined-sRGB-linear)

FFT 分子可先在頻域合併通道，再做一次逆轉換：

$$
C_{RGB}=\mathcal F^{-1}\left[\sum_cw_c\widehat I_c\overline{\widehat T_{c,centered}}\right].
$$

輸入仍需各通道的前向 FFT，分母也仍需逐通道均值。三通道不代表三份独立搜尋；旋轉座標、遮罩、候選與記憶體存取可共用。

FMM 提案也可使用向量頻譜能量

$$
M_{RGB}(\omega)=\sqrt{\sum_cw_c|\widehat I_c(\omega)|^2}.
$$

在共同幾何、共同增益且去除 DC 的假設下，它滿足與灰階幅值相同的旋轉尺度關係。它避免先壓成單一灰階造成的部分色彩抵消，但不能取代完整 RGB 驗證；獨立通道增益也會改變其形狀。

## 平台通用的 GPU 與 CPU 分工

每個模板先建立金字塔、統計量、有效遮罩、RGB／灰階格式，以及選用的 FFT 頻譜、log-polar 描述或基底係數。這些資料保留在 GPU。每張來源只傳入一次或直接使用相機可共享的 GPU 影像，再建立来源金字塔與必要的來源頻譜。

候選提案先按工作量選 CPU 或 GPU。現有 Src7 頂層只有 63×48、模板 12×9，因此不能假定把每一步搬到 GPU 都划算。尺度範圍變大或模板數增加後，這個分界需要重新量測。

接著一次處理一批 ROI：取樣、FFT、log-polar 與峰值估計都留在 GPU；每個候選保留少量旋轉／尺度假設，再以融合的 warp＋NCC／Jacobian reduction 精修。迭代時只保留姿態、統計量、收斂旗標，避免每個候選、每個角度同步 CPU。最後回傳位置、角度、尺度、分數與品質資訊，再做最終 NMS 與 GUI 顯示。

對既有 7×7 精搜核，旋轉像素應以區塊取樣並在相鄰位置間重用；不要為 49 個輸出各自重做整份旋轉。對连续精修，則一個姿態掃過像素一次，合併 RGB 統計與 Jacobian 統計。完整模板通常放不進片上共享記憶體，因此需切片與 halo，不能假設整張 762×521 模板常駐片上。

| 計算項目 | CUDA | Metal |
|---|---|---|
| 取樣與相關／梯度統計 | 自有 kernels、分塊 shared memory、並行 reduction | 自有 compute kernels、threadgroup memory、並行 reduction |
| FFT | 可替換 FFT 後端；cuFFT 可作效能參考 | 可替換 Metal FFT 後端或自有 FFT kernels |
| 候選與通道批次 | grid 維度／工作佇列 | grid 維度／工作佇列 |
| 同步與重複提交 | streams／events，固定流程再評估 CUDA Graphs | command buffers／events，按批次編碼 |
| 最終資料 | 只回傳小量姿態與分數 | 共享 buffer 仍需正確同步 |

[cuFFT 的批次與 plan 重用](https://docs.nvidia.com/cuda/cufft/index.html)、[CUDA 資料移動與片上記憶體建議](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)、[CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)、[Metal 共享儲存同步](https://developer.apple.com/documentation/metal/mtlstoragemode/shared) 可支援這種排程，但不能代替實測。

完整源碼要求仍適用：自有 warp、NCC、遮罩、精修與排程須保留源碼；若採 cuFFT，它是外部 FFT 依賴，不能宣稱其內部演算法源碼也已包含。CPU 的 scalar 參考路徑仍需保留。

精度不能只看浮點型別名稱。Gray8 的平方和、點積需避免整數溢位；大面積近平坦區域的 Q−S²/N 容易因消去而受低精度影響。FP32、整数分段累加与必要的高精度統計需做誤差實驗，不預設全改 FP16。硬體線性取樣與目前雙精度插值／uint8 取整規則也可能不同。

不要配置完整的「x×y×角度×尺度」響應體積。按影像 tile、候選或角度尺度批次計算，再做局部峰值與保留候選；截斷候選的規則必須有召回率驗證。多影格流水線可讓 CPU 處理前一張結果、GPU 計算當前圖片，但吞吐量改善不等於單張延遲同樣改善。

## 加速是否成立的成本條件

固定姿態下，令 P 為輸出位置數、A 為模板像素數、C 為通道數，直接相关的主要工作量為 O(CPA)。FFT 的補零陣列若有 B 個像素，主要工作量是 O(CB log B)，另有正規化、配置與資料搬移成本。B 不等於 P，因此「大模板」與「大量輸出位置」要分開量測。模板頻譜與 FFT plan 可重用，但影像頻譜只在來源格點未改變時可重用；旋轉來源圖後通常要重新計算其頻譜。

局部枚舉的工作量約為 K Nθ Ns P A C；K 是候選數。FMM 路線希望改成 K 個 ROI 的 FFT／log-polar 成本，再加上 K H J A C 的連續精修；H 為每個候選保留的假設數，J 為迭代數。只有 HJ 足夠小、候選生成保持召回率、且 FFT／取樣／退回成本沒有抵銷時，總工作量才會下降。更複雜背景可能增加 H 與 J。

基底路線則需先支付 Kb 次基底相關，再合成角度尺度響應；只有所需基底数 Kb 與合成成本相對於完整枚舉足夠小才有利。本模板的壓縮實驗目前不能支持直接以很小的 Kb 精確替換原流程。

對 GPU，可用簡化成本模型指導實驗：

$$
T\approx T_{upload}+\max(F/P_{eff},D/B_{eff})
+n_{launch}T_{launch}+T_{sync}+T_{download}.
$$

F 是運算量、D 是記憶體流量，Peff 與 Beff 為量測到的有效算力與頻寬。這是近似排程模型，不是精確預測。數學方法主要減少 F；分塊與融合減少 D；批次化減少提交次數與同步。若只更換單一點積 kernel，剩下的旋轉、配置、同步與傳輸仍限制整體速度。

## 數值驗證與目前能得出的結論

可重跑原型位於 `../tools/research_ncc_fourier.py`，結果位於 [fourier-ncc-research-validation.json](fourier-ncc-research-validation.json)。NumPy 提供 FFT，OpenCV 只在實際模板診斷中解碼 BMP。這些依賴沒有進入正式 C++ 核心。

```sh
python-port/venv/bin/python cpp-core/tools/research_ncc_fourier.py
```

已驗證四組固定姿態 NCC：灰階／RGB，分別有無二值模板遮罩；FFT 與獨立直接公式的最大絕對差約 7.7×10⁻¹⁵。旋轉尺度廣義基底的點值等式誤差約 1.8×10⁻¹⁴。四參數 NCC 梯度與中心有限差分最大差約 4.7×10⁻¹⁰。這些檢查確認推導與座標實作，並非 GPU 精度或效能保證。

原型的 NCC 對照使用非退化樣本；常數影像、接近零的變異數與誤差飽和規則，正式後端仍需沿用並驗證既有契約。點值基底測試也只驗證 r>0 的解析等式，沒有驗證有限支撐下的完整轉向偵測器。

單物件配準使用獨立解析高斯圖案產生真值，沒有用估計器的雙線性 warp 產生真值。包含平移、0.8／1.0／1.1／1.2 倍尺度及 0°／31°／−47°／143°，灰階與 RGB 各四例。估計端只從模板像素重新取樣與驗證兩個角度假設；八例最大角度誤差约 0.080°，最大相對尺度誤差约 0.246%，最大平移誤差约 0.401 px。這是平滑、無遮擋、無背景干擾的八例，不是工業影像召回率測試，也沒有證明能取代現有格點精修。

另有三組刻意破壞假設的診斷，全部保留在結果中。加入幅度 0.4 的獨立均勻背景後，這份基本 FMM 原型的角度誤差約 59°、尺度相對誤差約 25%；大片遮擋案例的誤差約 32°／34%。兩個不同姿態的物件同時出現時，單一全圖估計也沒有找出兩個目標。這些是目前原型的失敗案例，不是所有 FMM 改良方法的上限；它們支持保留局部 ROI、遮罩、空間 NCC 驗證及搜尋退回路徑，而不是直接用全圖 FMM 替換多目標偵測。

RGB 代數案例中，共同增益加逐通道偏移得到聯合 NCC=1；獨立通道增益得到聯合 NCC 約 0.959，但逐通道 NCC 平均為 1。另一組刻意保持相同灰階的色彩差異，灰階 NCC=1、RGB NCC 約 0.882，说明保留 RGB 可以區分被灰階映射合併的訊號。

實際 Dst7 模板還做了基底壓縮診斷：先對原矩形減平均、矩形外補零，再於 720×512 的 log-polar 格點建立 q=rT。只保留 17 個角向諧波時，該格點上的相對 L2 重建誤差約 54.6%；保留 65 個仍約 22.1%。角度及 log-radius 同時各保留 −8 到 8 階，共 289 個模式時，誤差約 67.0%。這不是 Cartesian NCC 的嚴格誤差界，也不排除其他基底更有效，但已不支持「此模板用很少的這類模式就能精確取代所有旋轉尺度」的假設。

因此近期的可驗證實驗順序為：先建立含尺度與 RGB 的合成／實拍基準；維持相同搜尋內容完成批次 warp＋NCC 後端；加入局部 FMM，量測它是否減少精修假設；加入四參數連續精修並保留退回路徑；最後比較有限基底加殘差界是否值得作為早期篩選器。

驗收要同時記錄端到端延遲、各階段 GPU 時間、模板準備成本、暖機／首次執行、記憶體、每張候選數及總取樣姿態數。對照資料需包含不同尺度、角度、噪聲、模糊、遮擋、邊界、低紋理、對稱、相似錯誤物件、亮度增益、白平衡與同灰階不同顏色。比較召回率、誤報率、位置／角度／尺度誤差及最終 NCC；不能只以「都找到六個」或 kernel 時間作結論。
