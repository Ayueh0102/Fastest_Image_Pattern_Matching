# 第27輪：Metal 細搜尋相鄰視窗的統計重用

2026-10-16，基準 `ae1836b`，Apple M4 Pro、48 GiB、macOS 26.6.2。這輪實作[提案B](ORIGINAL_ALGORITHM_NEXT_zh-TW.md)的第一部分：共用 fine NCC 的來源總和與平方和。完整 C++／Metal 來源保留，原模式與 GUI 預設維持不變。

**結論：數值等價通過，但不採用為加速預設。** 80程序／620次正式量測中，666個既有合成GT姿態零FP／FN，提案與最終輸出和凍結版完全一致；128／512物件分別慢約2.5%／1.5%，高瘦模板退步更大。原版仍是目前建議的 fine NCC，新增模式僅作研究對照。

證據：[事前規則](fine-reuse/protocol.json)、[所有摘要與失敗](fine-reuse/summary.json)、[驗證與封存索引](fine-reuse/validation.json)、[完整來源快照與原始結果](fine-reuse/raw-results.tar.gz)。

## 正式結果

以下為兩個block合併後的正式呼叫中位數；RSS為兩個新程序峰值的中位數，不能將RSS差額解釋為某一個buffer的實際駐留量。完整block數字、setup、first與原始值都在摘要／封存。

| 全流程場景 | 凍結版 ms | 同建置原版 ms | 共享統計 ms | 原版→共享 RSS MiB |
|---|---:|---:|---:|---:|
| 6物件 | 76.59 | 74.86 | 77.60 | 49.09→49.28 |
| 12物件 | 147.37 | 145.81 | 147.09 | 80.30→79.48 |
| 8K、256²模板、6物件 | 268.28 | 268.53 | 267.37 | 565.69→564.32 |
| 小圖、2物件 | 8.05 | 8.05 | 7.94 | 26.84→27.12 |
| 128物件 | 1146.91 | 1152.31 | 1181.34 | 410.54→415.05 |
| 512物件 | 3404.99 | 3372.58 | 3424.14 | 794.38→793.10 |

8K總時間略快，但fine的GPU統計時間反而2.78→3.26ms，**不能把全流程的微小差額歸功於這個kernel**。128／512的fine GPU統計分別376.31→393.45ms、1204.65→1245.29ms；fine總NCC為486.71→511.09ms、1531.00→1583.22ms。同一套離散／連續搜尋設定隔離此次改動，並未套上其他研究輪次的最快精修組合，所以不要拿本表和固定尺度稀疏路徑的毫秒數直接相比。

| 孤立 fine：模板×ROI數 | 原版→共享 wall ms | 原版→共享 GPU統計 ms | 新增行容量 MiB |
|---|---:|---:|---:|
| 19×17×128 | 0.281→0.290 | 0.055→0.066 | 0.157 |
| 64×48×128 | 0.876→0.851 | 0.411→0.416 | 0.369 |
| 64×48×512 | 2.619→2.660 | 1.612→1.635 | 1.477 |
| 128×128×128 | 3.129→3.157 | 2.163→2.121 | 0.916 |
| 257×257×17 | 1.918→1.874 | 1.296→1.256 | 0.239 |
| 257×258×17 | 2.051→2.027 | 1.443→1.381 | 0.479 |
| 1×257×128 | 0.313→0.489 | 0.047→0.167 | 1.798 |
| 1×1024×291 | 1.070→2.441 | 0.320→1.376 | 15.952 |

高瘦最後一列從一批變兩批，command buffers由2增為4。兩個高瘦案例共4個block超過性能容忍；其餘測試通過容忍，但許多仍稍慢。所有RSS容忍通過，並不表示模式1省記憶體：例如128／512場景fine保留容量由3.968→5.882MiB，增加約1.914MiB。

11種探針共365,344個CPU參考窗口逐位一致；各暖機／正式結果也與首輪一致。全流程666個GT是六張既有場景的姿態數，不能乘上重複量測次數當作獨立精度樣本。全流程提交數、來源上傳量及工作量一致；128／512每次fine仍1192／3282次提交。暖機fine來源上傳counter為0，GPU金字塔來源已可重用；**等待時間不是傳輸時間**，且此counter不涵蓋其他階段。這輪沒有硬體頻寬、cache miss或功耗計數器，因此不能宣稱已找到硬體極限。

## 驗證與採用決定

Release 41／41、相關ASan／UBSan 9／9、Metal API與Shader Validation、Python binding 2項、量測工具7項全部通過。新增測試涵蓋32／64邊界、border／半像素、常數來源、相同地址換圖、padded輸入、錯誤mode／C結構尺寸、切回mode0釋放row buffer、metrics reset及超出16MiB明確拒絕。這不是新一輪30分鐘soak或Windows／CUDA實測。

子agent完成獨立源碼與量測審查，並補上暫存上限／重設測試與pipeline初始化的autoreleasepool。審查發現探針按時間暖機累積紀錄會干擾RSS，已在正式測試前改成固定容量。全部正式來源、二進位與輸入hash在量測期間不變。

正式完成後只修正摘要工具的兩個失敗處理問題：無正確匹配時保留空誤差欄位；離線摘要失準時非零退出。新工具重算本次成功資料與舊摘要逐byte一致，量測時的工具版本另行封存。原始試跑、尚未重新設定CMake而找不到新target的建置失敗、修正前後測試log均保留，未刪除慢樣本。

不把mode1接成GUI或全域預設，不以尺寸門檻自動啟用。下一步B2先獨立實驗單一kernel的SIMD strip：同群處理3或4個相鄰位置，共用模板／來源載入，點積各自累加；保留原批次及暫存上限，不混入本輪row pass。右側用3輸出版本，避免多算無效第8個橫向位置。需要驗證暫存器壓力、64位spill與少候選並行度，尚未實作或宣稱加速。

## 等價變換與實作範圍

每個角度先產生 `(w+6)×(h+6)` 的旋轉 ROI，再計算49個相鄰位置。原版每個位置各自累加 `S=ΣI`、`Q=ΣI²`、`P=ΣIT`。新模式將前兩項拆成可共用的行統計：

```text
R1(y,0) = Σu=0..w−1 I(y,u)
R2(y,0) = Σu=0..w−1 I(y,u)²
R1(y,x+1) = R1(y,x) − I(y,x) + I(y,x+w)
R2(y,x+1) = R2(y,x) − I(y,x)² + I(y,x+w)²
S(y,x) = Σv=0..h−1 R1(y+v,x)
Q(y,x) = Σv=0..h−1 R2(y+v,x)
```

第一個 kernel 以一個 SIMD 群處理一行，算第一個窗口並滾動到其餘六個位置；第二個 kernel 仍逐像素計算原點積 `P`，並將行統計垂直加總。兩者放在同一個 NCC command buffer 的連續 encoder，沿用 tracked shared buffer 的相依性。每批仍為旋轉一次提交／等待、NCC一次提交／等待。CPU邊界修補、binary64正規化、選峰、角度與金字塔控制不改。

這會減少 `S/Q` 重算，但**沒有消除 `49wh` 次點積，也沒有保證少讀49倍來源或少用49倍記憶體**。行統計另增加 GPU buffer 存取與一次 dispatch；收益需實測。這不是新的快取層控制、全部GPU常駐、FFT、RGB全圖或連續精修求解法。

## 精度與記憶體界線

- `65025wh ≤ UINT32_MAX` 用32位整數，否則用64位。包含32→64邊界 `257×257`／`257×258` 的高亮度測試。滾動和先減後加，所有中間和受完整視窗上界限制；64位 reduction 沿用既有分段整數加總。
- 每張ROI需 `7(h+6)×8` bytes（32位）或 `7(h+6)×16` bytes（64位）行統計。新增行暫存上限16 MiB；原ROI上限64 MiB、最終統計上限16 MiB保留。
- 高瘦模板可能因行暫存限制縮小批次，**整次搜尋可能增加提交／等待**；只保證每批不增加CPU同步。單張行統計超過16 MiB明確報錯，不默默回退或無限制配置。
- `row_buffer_bytes` 是 `statistics_buffer_bytes` 的子集，已包含於 `retained_gpu_bytes`／frame fine容量，不能重複相加。RSS是程序峰值，包含來源、輸出、runtime與量測程式，不能稱為純演算法工作區。

## 比較方法

正式協定在量測前寫入；11種孤立 fine 工作量、6個全流程場景，80個串行新程序、620次正式呼叫。每個程序暖機至少1秒，另記建模／第一輪；兩個block反轉方法順序。沒有同時跑其他本任務GPU工作。孤立探針暖機只保留固定容量的計數與一致性狀態，避免較快模式累積更多量測紀錄、墊高RSS。

孤立比較同建置 mode0／mode1，模板從3×3至257×258，含1、128、512、513個ROI、批次尾端與高瘦模板。逐張map對CPU參考做記憶體逐位比對；跨方法另核對map摘要。CPU參考與GT檢查均在計時及RSS取樣後執行。

全流程比較凍結第26輪程式、同建置原模式、共享行統計，粗NCC固定mode2、overlap關閉、fine batch512，其餘搜尋／rescue選項相同。來源prepare包含在總時間；圖片解碼、一次建模與JSON序列化不含於暖機算法時間。新模式lazy pipeline初始化包含在第一輪。

六個場景沿用第14輪合成資料，共666個GT姿態，含8K及128／512同模板分離實例；**不是新獨立資料或真實相機場景**。GT只在父程序以一對一配對評估，門檻位置2px、角度1°、尺度±2%。工作程序只收來源、模板與搜尋參數。

性能容忍預先設為全流程 `max(10%,2ms)`、孤立 `max(10%,0.1ms)`；RSS為 `max(10%,16MiB)`。通過容忍不等於更快，原始差值與失敗均保留。`stats_device_ms` 是NCC command的GPU時間，包含行統計與點積兩個dispatch，不是單一kernel純運算時間，也不是獨立傳輸時間。

## 使用與重現

原生 `fpm::research::Options::fineNccMode=1`，或C API `fpm_frame_set_fine_ncc_mode(frame,1,...)` 開啟；需要Metal fine batching。Python adapter為 `SourceFrame.set_fine_ncc_mode(1)`；native runner使用 `fine_ncc_mode: 1`。模式0釋放行buffer，CPU拒絕模式1，固定尺度稀疏路徑亦拒絕此不適用選項。新metrics使用獨立48-byte版本化結構，既有C結構布局不改。

來源：[Metal kernels](../experiments/coarse_ncc.metal)、[排程與容量](../experiments/coarse_metal.mm)、[C API](../experiments/coarse_ncc.h)、[原生整合](../experiments/research_pipeline.cpp)、[孤立探針](../tools/fine_reuse_probe.cpp)、[正式比較與離線摘要](../tools/benchmark_fine_reuse.py)。

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake -S cpp-core -B build/fine-reuse \
  -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_BUILD_TESTS=ON \
  -DFPM_BUILD_GUI=ON -DFPM_WITH_OPENCV=ON
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build build/fine-reuse -j 6
ctest --test-dir build/fine-reuse --output-on-failure
python-port/venv/bin/python cpp-core/tests/test_fine_reuse_harness.py
python-port/venv/bin/python cpp-core/tests/test_fine_reuse_binding.py
python-port/venv/bin/python cpp-core/tools/benchmark_fine_reuse.py --output build/fine-reuse-replay
```

正式比較需要事先保留的 `build/adaptive-ncc` 凍結程式與第14輪輸入；可用 `--baseline`／`--previous` 指定對應位置。不要以新程式重建舊基準。離線重算使用解壓後 `formal.json`，其job與result路徑為相對路徑；會驗證內容hash，輸出路徑由 `--output summary.json` 指定。
