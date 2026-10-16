# 第28輪：Metal fine NCC 相鄰點積的像素載入重用

2026-10-16，基準 `57bd600`，Apple M4 Pro、48 GiB、macOS 26.6.2。延續[原始算法提案B](ORIGINAL_ALGORITHM_NEXT_zh-TW.md)，獨立實作點積載入重用；第27輪的行統計模式保留，本輪不將兩者混合。

**結論：這輪有實際加速，可手動優先試mode2（chunk32）。** 128／512物件完整時間縮短約21.8%／23.5%，585,648個CPU參考窗口比較逐位一致，既有666個GT姿態零FP／FN；提交次數、工作量與顯式GPU buffer容量相同。8K少候選場景總時間沒有改善，極小模板稍慢。維持mode0及GUI預設，不聲稱所有場景或所有GPU都更快。

證據：[事前協定](fine-strip/protocol.json)、[完整摘要](fine-strip/summary.json)、[驗證與封存hash](fine-strip/validation.json)、[來源、輸入及原始結果](fine-strip/raw-results.tar.gz)。

## 正式結果與採用範圍

以下取每種方法兩個block共10次全流程呼叫的中位數；同輪凍結程式另作預設回歸比較。RSS是兩個程序峰值的中位數，包含runtime、圖像、輸出及量測程式，不能當成純工作區。所有方法都已暖機；不是讀圖到GUI顯示的時間。

| 全流程場景 | 凍結版 ms | 新建置原版 ms | strip32 ms | strip16 ms | 原版→strip32 RSS MiB |
|---|---:|---:|---:|---:|---:|
| 6物件 | 78.06 | 78.26 | 63.96 | 62.89 | 50.24→49.74 |
| 12物件 | 147.71 | 148.71 | 118.31 | 118.05 | 79.55→80.02 |
| 8K、256²模板、6物件 | 266.45 | 262.73 | 264.66 | 266.32 | 565.34→564.11 |
| 小圖、2物件 | 8.40 | 8.62 | 6.87 | 7.02 | 27.13→27.27 |
| 128物件 | 1172.83 | 1170.67 | 915.60 | 922.69 | 411.23→411.22 |
| 512物件 | 3421.75 | 3422.74 | 2619.62 | 2627.76 | 797.02→802.37 |

128的兩block原版／strip32為1170.78／914.07及1165.15／917.12ms；512為3439.75／2620.89及3416.45／2599.12ms，反轉執行順序仍保有改善。所有新模式相對原版、凍結版的時間容忍通過；所有RSS容忍通過。原版buffer容量沒有改，512的程序峰值仍增加約5.35MiB，因此不能將「沒有新增顯式buffer」寫成「所有記憶體完全相同」。

| 孤立fine模板×ROI數 | 原版→strip32 wall ms | 原版→strip32 NCC GPU ms | strip16 wall ms |
|---|---:|---:|---:|
| 3×3×1 | 0.170→0.178 | 約0.003→0.004 | 0.196 |
| 19×17×128 | 0.284→0.246 | 0.055→0.023 | 0.264 |
| 64×48×128 | 0.844→0.518 | 0.410→0.101 | 0.568 |
| 64×48×512 | 2.547→1.385 | 1.612→0.390 | 1.535 |
| 128×128×128 | 3.199→1.672 | 2.133→0.530 | 1.589 |
| 257×257×17 | 1.936→1.007 | 1.313→0.406 | 1.184 |
| 257×258×17 | 2.110→1.282 | 1.453→0.615 | 1.357 |
| 1×1024×291 | 1.090→0.886 | 0.333→0.168 | 0.932 |

strip16並非一律較好：相對strip32有5個孤立block超過時間容忍，分別在64×48×512的一個block、257×257×17及19×17×513的兩個block；這些不是相對原版的失敗。128²的wall反而strip16稍快，而GPU NCC仍strip32快，顯示wall包含旋轉、主機等待及量測波動，不宜憑單點挑全域最優分塊。

## 加速來源與下一個瓶頸

128／512的fine NCC GPU區間分別379.96→135.27ms、1221.29→430.74ms，約縮短64%～65%；包含提交／等待、正規化的fine NCC主機區間則502.20→245.48ms、1561.39→740.70ms。旋轉區間252.23→253.33ms、799.97→808.15ms，沒有同時得到類似改善。這支持收益主要出現在被替換的NCC階段。

每次fine提交仍1192／3282次，ROI總像素數及repair量完全相同。暖機fine來源上傳counter仍0，沒有透過減少搜尋或上傳換取速度；wait區間700.82→450.17ms、2195.40→1396.39ms隨NCC變短，**不能將這段等待直接稱為傳輸時間**。這些階段彼此有包含／重疊關係，不能將GPU、wait、wall與normalization直接加總。

8K少物件的fine NCC GPU雖2.99→1.06ms，但全流程262.73→264.66ms；細搜尋占比太小，不能據局部kernel倍率推導總加速。也沒有本輪全CPU性能對照、DRAM頻寬、cache miss、spill或功耗計數器。

目前對本機多候選金字塔工作量可先手動選mode2，mode3保留消融；極小任務或少候選大圖仍以實測判斷。下一步可先量「完全相同層級／anchor／角度的ROI請求」重複率，以及旋轉與CPU修補邊界的同步成本，再決定快取或融合；本輪沒有實作該下一階段，也沒有證明GPU已到極限。

## 演算法與精度契約

對一張 `(w+6)×(h+6)` 旋轉ROI，輸出仍為7×7個NCC。原模式用49個SIMD群，每群負責一個位置；模式2／3使用14群，對每個輸出y分成左4個位置與右3個位置。每群的lane分工模板的行片段，模式2每片段32個像素，模式3每片段16個像素。

片段內，固定模板值 `t(u)` 同時服務 `O=3或4` 個輸出；來源的 `O` 個值放在私有暫存，往右移一步只補進一個新像素。對每個輸出 `j`，仍完整計算自己的三項統計：

```text
Sj += I(u+j)
Qj += I(u+j)²
Pj += I(u+j) × T(u)
```

**重用的是來源／模板載入，不是相鄰點積的值。** 任意模板不能由相鄰P的簡單差值推出另一個P。新模式仍做49個完整點積，右側使用3輸出版本，沒有計算無效的第8欄。最後沿用相同整數SIMD reduction與CPU binary64正規化，分數、旋轉修補、金字塔、選峰及精修行為不改。

令 `q=ceil(w/C)`，`C`為32或16。按原始碼中的邏輯byte讀取次數計算，每張ROI：

| 工作 | 原模式 | 新strip模式 |
|---|---:|---:|
| 模板載入 | `49wh` | `14wh` |
| 來源載入 | `49wh` | `7h(2w+5q)` |
| 點積乘加 | `49wh` | `49wh` |
| SIMD群數 | 49 | 14 |

這是程式層的讀取次數，不是實測DRAM流量或cache命中率。片段會重載halo；行列索引由逐像素除法／餘數改成每片段一次，lane間步距、暫存器壓力與群數減少也會改變執行效率，不能把時間差只歸因於載入。沒有手動配置CPU L1–L3，也沒有要求GPU確實將陣列放進特定硬體暫存器；編譯器可能spill，需另外用硬體工具確認。

## 排程與資源

新模式只替換既有NCC kernel，單一encoder與command，無新增全域／threadgroup buffer或barrier。沿用原批次上限與64 MiB ROI／16 MiB輸出統計容量，**不受第27輪row arena限制**。切換到mode2／3會釋放mode1的行buffer。對相同輸入，提交次數、旋轉／repair工作與顯式GPU buffer容量應與mode0相同。

`65025wh ≤ UINT32_MAX`使用32位累加，否則64位；部分和非負且不超過完整視窗的和。64位歸約沿用原分段整數演算法，沒有轉FP32。片段尾端在讀取前限制，所有有效SIMD群均完整，補齊的群一致返回。完整map固定7×7，主機端有編譯期檢查。

## 比較方法與驗收界線

正式規則在執行前寫入。12種孤立工作量比較同建置原版／strip32／strip16，6個全流程場景另加入凍結第27輪原版。兩個block採相反順序、串行新程序；每程序至少暖機1秒，孤立10次、全流程5次正式量測，合計120程序／960次。冷啟動、首次與暖機分開記錄。凍結runner及兩個dylib的SHA與第27輪保存協定核對後才執行。

全流程固定粗NCC mode2、overlap關閉、fine batch512，其餘與第27輪相同，不混入其他輪次的精修改動。總算法時間包含每次frame prepare，排除讀圖、一次模型建置、輸出序列化與GT評估。孤立探針包含旋轉、CPU修補、NCC及正規化；`stats_device_ms`另量NCC command GPU區間，不能加到wall或wait上。

資料沿用六個既有合成場景，共666個GT姿態，含8K、128及512個同模板分離實例；不是新增獨立相機資料。GT只在父程序評估，worker只能看到圖像與搜尋參數。一對一嚴格配對門檻為位置2px、角度1°、尺度±2%。孤立map逐位對CPU參考，跨mode核對摘要；全流程完整有序提案／結果、工作量、冷／暖各自的提交／上傳／repair計數與容量皆須一致。

主實驗預先指定chunk32，chunk16為分塊大小消融。全流程時間容忍`max(10%,2ms)`、孤立`max(10%,0.1ms)`；RSS容忍`max(10%,16MiB)`。所有退步與失敗均保留，通過容忍不表示加速；不憑這個有限資料集自動改全域或GUI預設。

## 源碼、使用與重現

驗證已覆蓋Release 43項、相關ASan／UBSan 11項、兩種strip模式的Metal API／Shader Validation，以及Python binding 2項、量測工具9項。新增模板寬度1／15／16／17／31／32／33／63／64／65，高度1／3／17，15張map配batch7驗最後一批；沿用257×257／257×258整數位寬邊界、實際溢出32位的高亮模板、半像素、border、常數與同地址換圖測試。Native Session另外以fresh原版比對新模式、重複呼叫、模式切換、提交數與每種arena容量。

協作agent完成kernel與測試；後續獨立複查因服務額度中斷，量測agent因工具權限中斷，沒有交付量測程式。主agent接手完成host／API整合、源碼核查、量測工具、驗證與報告；不宣稱已完成獨立agent最終複查。移除新增strip區段後，原Metal檔逐byte等於基準，原mode0／1 kernel未改。

封存中的來源已另行重新建置兩個runner，模式2／3的33×17模板、15張map、batch7皆對CPU逐位一致，小圖完整搜尋的提案及結果也與正式量測相同。移動封存位置後可逐byte重算相同摘要；移除所有匹配、改動計數、破壞hash與重複worker的負控制均被拒絕。初次負控制僅移除單一block，斷言失敗後修正測試資料，沒有改matcher／摘要程式；另一次封存指令誤列未被Git追蹤的AGENTS.md，修正清單後完成。兩次失敗及處理記錄均保留。

[Metal kernels](../experiments/coarse_ncc.metal)、[主機排程](../experiments/coarse_metal.mm)、[C API](../experiments/coarse_ncc.h)、[Native Session](../experiments/research_pipeline.hpp)、[完整量測工具](../tools/benchmark_fine_strip.py)、[孤立C++探針](../tools/fine_reuse_probe.cpp)。

研究介面：`Options.fineNccMode=2`（chunk32）或3（chunk16）；C API為`fpm_frame_set_fine_ncc_mode`，Python為`SourceFrame.set_fine_ncc_mode`，native JSON為`fine_ncc_mode`。模式0／1與既有C ABI保留；CPU拒絕非0模式，固定尺度稀疏流程亦不接受此選項。

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake -S cpp-core -B build/fine-strip \
  -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_BUILD_TESTS=ON \
  -DFPM_BUILD_GUI=ON -DFPM_WITH_OPENCV=ON
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build build/fine-strip -j 6
ctest --test-dir build/fine-strip --output-on-failure
FPM_FINE_REUSE_BUILD=build/fine-strip python-port/venv/bin/python cpp-core/tests/test_fine_reuse_binding.py
python-port/venv/bin/python cpp-core/tests/test_fine_strip_harness.py
python-port/venv/bin/python cpp-core/tools/benchmark_fine_strip.py --output build/fine-strip-replay
```

保留`build/fine-reuse`凍結程式，不用新來源重建舊基準。全流程輸入來自`build/gui-integration-results/formal.json`；可用`--previous`指定輸入描述。離線摘要以`--summarize /解壓位置/formal.json --output summary.json`執行，只需相對job／result路徑，並驗證hash。原始worker的絕對圖像路徑是歷史記錄，重新跑搜尋需先按封存的輸入映射改為解壓路徑。
