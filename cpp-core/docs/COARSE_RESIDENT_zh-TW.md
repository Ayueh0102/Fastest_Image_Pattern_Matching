# Metal 常駐粗搜尋與選峰實驗

第36輪，2026-10-18，Apple M4 Pro／48 GiB，Release。這一輪測試把粗 NCC 的正規化與逐次選峰接在 GPU 統計後面，讓 CPU 只讀取候選清單。研究選項為 `coarse_peak_mode=2`，原本 `1` 的 CPU double 正規化與精確選峰樹作同建置對照。原核心與 GUI 預設保留；這個實驗及負面結果留存，供未來 CUDA 實作比較。

**結論：常駐已實作，但不採用為通用預設。** 8K少物件場景耗時降低26.4%，128／512物件卻分別耗時約6.0／11.1倍。六組場景的最終結果與有序提案均等於對照版；這不消除底層FP32選峰與double的數值差異。

## 時間 記憶體與準確度

48個新程序，48次first call與144次正式計時，另保留暖機。時間是12次正式樣本的pooled median；RSS是四程序的峰值最大值。下面的「原路徑」已使用Metal NCC、金字塔及批次細搜尋，只是正規化／選峰仍在CPU，並非全CPU算法。

| 場景 | 原路徑 ms | 常駐版 ms | 耗時變化 | RSS MiB 原→新 | 兩者正確檢出 |
|---|---:|---:|---:|---:|---:|
| 6旋轉物件 | 58.63 | 65.05 | +10.9% | 41.72→39.64 | 6／6 |
| 12旋轉物件 | 106.76 | 149.94 | +40.4% | 56.94→51.56 | 12／12 |
| 小圖完整解析度 | 5.46 | 5.20 | −4.7% | 26.92→26.00 | 2／2 |
| 8K全角度三尺度 | 255.17 | 187.88 | −26.4% | 373.23→332.91 | 6／6 |
| 128物件 | 834.79 | 4,993.26 | +498.1% | 233.48→181.27 | 128／128 |
| 512物件 | 2,373.64 | 26,361.99 | +1010.6% | 443.08→349.61 | 512／512 |

所有首次／正式呼叫皆零FP、零FN，同方法重複結果一致；完整有序proposals與matches也都與CPU選峰對照相同。最大位置誤差0.1281px、角度0.0996°、相對尺度0.2745%，兩路徑相同。這些是本輪合成與既有受控場景的結果，不是實拍泛化或所有分數門檻的證明。

8K案例四個配對block耗時降低25.1～30.1%，並非挑單一較快樣本。小圖差距只有約0.26ms，列為接近持平；6物件在先導量測曾略快，但正式交錯量測沒有重現。6／12／128／512案例未通過事前性能容忍，完整失敗紀錄保留。

512案例的程序RSS雖降低約21.1%，frame持有的粗搜尋GPU buffers卻由60.53增至76.68MiB。新版避免CPU密集double map與選峰樹，但保留GPU的FP32 map、端點表、輸出與批次來源；不能把「程序RSS下降」寫成每一類buffer都下降，也不能把GPU容量加到RSS當成總實體記憶體。

## 退步發生在哪裡

| 512物件的診斷 | 原路徑 | 常駐版 |
|---|---:|---:|
| 粗command buffers | 102 | 36 |
| 粗GPU區間 ms | 88.22 | 24,493.50 |
| host wait ms | 145.05 | 24,510.25 |
| host copy與參數準備 ms | 2.29 | 2.45 |
| CPU密集正規化 ms | 199.73 | 0 |
| fine maps | 659,376 | 659,376 |

提交數降低、密集CPU正規化消失，但GPU執行區間大幅增加；不能把這次約24秒的GPU區間歸因於2.45ms的host copy。粗搜尋視窗數、fine maps、最終提案與輸出相同，也沒有靠少搜候選交換速度。結合下面的工作分配可知，主要待改善的是GPU選峰器的平行粒度與依序成本；這不代表Metal或NCC已到極限。

## 常駐的範圍

```text
CPU 旋轉畫布與準備參數
    ↓
同一次 Metal 提交
    整數 NCC 統計 → GPU 正規化 → GPU 逐次選峰與停止
    ↓ 一次等待，CPU 只讀候選清單
CPU 候選座標轉換與排序 → 既有批次細搜尋 → 精修與 NMS
```

每批最多四個角度；較大的統計工作區在同一個 command buffer 內分段重用，不在每段之間交回 CPU。金字塔、細搜尋與精修沿用既有選項。這不是整條搜尋已全 GPU：粗旋轉、抑制端點表準備、候選交接、細搜尋控制與數值修補，以及最終 NMS 仍有 CPU 工作。

API 的 `peak_mode` 與研究 runner 的 `coarse_peak_mode` 指同一項設定；`ncc_mode` 是獨立的整數 NCC 工作分配選項。模式2沒有默默改成近似 top-K，也不會因容量不足截掉候選。CPU／GPU 非同步雙槽排程與模式2互斥，組合錯誤直接回報。

## 數學與數值契約

令模板面積為 \(N\)，來源視窗的像素和、平方和與模板點積分別為 \(S,Q,P\)，模板的像素和、平方和為 \(S_T,Q_T\)。GPU 先計算整數中心化統計：

\[
V_I=NQ-S^2,\qquad V_T=NQ_T-S_T^2,\qquad C=NP-SS_T,
\]

再以 FP32 計算 \(C/(\sqrt{V_I}\sqrt{V_T})\)。整數相減在浮點轉換之前完成，避免低對比影像在 FP32 的大數相減中丟失變異量；host 先檢查 \(65025N^2\leq\mathrm{UINT64\_MAX}\)，不讓乘法溢位。變異保護、飽和分數與常數模板行為沿用原規則的對應處理。

這是數學等價的 NCC 表示，不是原本 binary64 運算順序的逐位重現。非常接近的分數可能合併成相同 FP32 值，排序或門檻判斷可能改變。GPU 對已算出的分數使用「分數降序、y 升序、x 升序」，保留逐次選峰、抑制、再選下一個的規則。抑制端點表由 CPU 按原 double／截斷規則建立，包含 overlap=1 仍抑制已選像素。門檻向上取到最近的 FP32 值，以保持「已儲存 FP32 分數與輸入 double 門檻比較」的語意。

數值測試比對所選位置的 CPU 分數與尚未抑制區域的最大值，容許誤差為3e-6，同時明列與原有序清單的差異。完整流程另外用獨立 GT 驗證，不能用數值容差通過代替召回率通過。

## GPU 工作分配與已知成本

正規化按位置平行執行。選峰採每張 map 一個256-thread工作群組：各 thread 維護一段分數的最大值，每輪共同選出全域最大值、抑制矩形，再只重掃原勝出位置遭抑制的段。停止條件與迴圈在 GPU，沒有每選一個峰就讓 CPU 往返。

這個設計保留了依序選峰的相依關係，但一張 map 的選峰只由一個工作群組處理。每個候選仍有歸約、記憶體屏障與必要的重掃；它沒有沿用 CPU tournament tree 的對數路徑更新。因此它是常駐可行性的實驗對照，不是已充分利用全部 GPU 核心的選峰器。沒有硬體 occupancy／cache counter，不能把觀察到的成本拆成精確的快取未命中或同步百分比。

候選預算和模板大小都會改變工作量，不能只按原始影像解析度選後端：

| 案例 | 來源尺寸 | 模板尺寸 | 每尺度 maxTargets | 全部粗 NCC 視窗數 |
|---|---|---|---:|---:|
| 8K六物件 | 8192×8192 | 256×256 | 12 | 39,329,756 |
| 128物件 | 4096×4096 | 140×102 | 384 | 44,610,794 |
| 512物件 | 8192×4096 | 140×102 | 768 | 99,960,878 |

每張粗 map 的選峰上限為 `maxTargets+5`，達不到門檻時會提早停止；上述上限不是實際每張 map 都選到的峰數，也不是最終物件數。8K與128案例的粗 NCC 視窗數接近，但候選預算不同，不能把它們當成只改影像尺寸的控制實驗。

每個粗搜尋 canvas 的 gray8來源與 FP32 score map 各有64 MiB上限；統計暫存上限16 MiB。端點表、輸出、模板及金字塔另計，這些不是整個程序的記憶體上限。候選上限1～10016，來源／模板必須符合矩陣與整數範圍檢查。配置保留供重用，切回其他模式不會立即釋放已配置的研究緩衝區；retained buffer容量與程序RSS分開報告。

## 正式比較規格

六組固定案例：6旋轉物件、12旋轉物件、小圖完整解析度、8K全角度三尺度、128與512旋轉尺度物件。8K案例是六個植入物件；大量物件案例為可重建的合成場景，不是實拍串流或 RGB 異色驗收。

每案例兩個模式、四個順序交錯 block，每程序至少一秒暖機、三次正式計時，初始化與第一次呼叫獨立保留。程序串行執行。總時間包含來源準備、提案、細搜尋、有啟用的精修與NMS，不含圖片解碼、JSON序列化與GT評估。8K與小圖案例是core-only，沒有後段連續pose精修；其他案例沿用完整rescue。所有方法使用相同影像、模板、尺度、角度、候選預算與其他後端設定。

準確度採一對一配對：位置誤差≤2px、角度≤1°、相對尺度≤2%；所有首次及正式呼叫都評估。每 block 的性能容忍為 `new <= control + max(2ms, 10%)`；RSS容忍為 `new <= control + max(8MiB, 10%)`。通過容忍不等於已加速。設定、原始工作量、提案數與有序結果差異均保留。

`device_ms` 是整個粗 command buffer 的 GPU 區間，新版包含正規化與選峰；`wait_ms` 包含 GPU 執行，兩者不能相加。新版 `copy_ms` 也包含端點表準備，並不是獨立的 PCIe 傳輸時間。兩條路徑同時改了數值表示與工作位置，不能把全部速度差都歸功「省掉一次等待」。

## 驗證與重現

新測試涵蓋360張 map、非連續row stride、常數與低對比、uint32／uint64統計、統計分段、1005個同分候選、門檻、模式切換及不合法配置。所選位置最大分數差7.22211e-8；62個有序 peak項目與double參考不同，明確保留，未宣稱逐位一致。Metal API／Shader Validation啟用後同樣通過。Release完整56／56、resident測試ASan／UBSan通過；ASan關閉leak detection，不宣稱LeakSanitizer驗收。量測工具另有三項含多個負控制的測試，確認錯誤精度、改變粗搜尋工作量、CPU密集回退、速度退步、缺少block或修改結果不能被當成通過。

從封存解出331個來源／輸入檔，全部SHA256吻合；以封存源碼乾淨重建runner與resident測試成功。重建後360張map測試通過，6物件兩方法各兩次輸出與正式版的完整有序matches／proposals相同。從封存原始結果重算的summary逐位元組相同。這些是重現性檢查，不是新的性能樣本或再跑一輪完整CTest。

```sh
cmake -S cpp-core -B build/fourier-gpu -DCMAKE_BUILD_TYPE=Release \
  -DFPM_BUILD_EXPERIMENTS=ON -DFPM_BUILD_GUI=ON -DFPM_WITH_OPENCV=ON
cmake --build build/fourier-gpu --parallel 8
ctest --test-dir build/fourier-gpu --output-on-failure
python-port/venv/bin/python cpp-core/tests/test_coarse_resident_harness.py
python-port/venv/bin/python cpp-core/tools/benchmark_coarse_resident.py \
  --blocks 4 --repeats 3 --output build/stage36-coarse-resident-replay/formal/manifest.json
```

CPU仍可透過 `coarse_peak_mode=1` 選原精確路徑。Metal實驗為 `2`，且 `overlap_enabled=false`；`fine_ncc_mode=2`、`fine_fused=true`、`fine_angle_reuse=true`、`pose_resident_mode=0`、`pose_input_mode=1`、`pose_storage_mode=1` 是本輪其他共同設定。

完整源碼：[GPU核心](../experiments/coarse_resident.metal)、[Metal派工](../experiments/coarse_metal.mm)、[候選介面與細搜尋交接](../src/matcher.cpp)、[數值與邊界測試](../tests/coarse_resident_test.cpp)、[量測與重播](../tools/benchmark_coarse_resident.py)。

## 封存與外部審查

先讀[事前規格](stage36-coarse-resident/protocol.json)、[正式摘要](stage36-coarse-resident/formal-summary.json)、[採用判斷與失敗](stage36-coarse-resident/decision-and-failures.json)，再核對[驗證紀錄](stage36-coarse-resident/validation.json)與[檔案大小／SHA256清單](stage36-coarse-resident/archive.json)。

| 封存 | 內容 |
|---|---|
| [source-snapshot.tar.gz](stage36-coarse-resident/source-snapshot.tar.gz) | 量測當時C++／Metal／Python／CMake、GUI來源與331檔雜湊清單 |
| [repository-fixtures.tar.gz](stage36-coarse-resident/repository-fixtures.tar.gz) | 原repository樣圖，保留相對路徑 |
| [generated-fixtures.tar.gz](stage36-coarse-resident/generated-fixtures.tar.gz) | 本輪正式生成輸入 |
| [raw-results.tar.gz](stage36-coarse-resident/raw-results.tar.gz) | pilot／formal manifests、jobs、逐次results、stderr、量測diff與原始雜湊 |
| [validation-logs.tar.gz](stage36-coarse-resident/validation-logs.tar.gz) | Release、Metal、ASan／UBSan、負控制、早期失敗、重播與乾淨重建 |

正式48程序的1754次暖機只保存時間，不列入192次GT驗收數。先導結果與正式結果分開，所有未通過性能門檻的block均保留。早期CTest把新合法的mode2仍當非法值，修為mode3後全過；ASan第一次缺新target，重新配置後通過。這些修正前log沒有刪除。

前三份封存解到同一個空目錄；raw另解，避免更動現有工作樹。原始job內保留當時絕對輸入路徑以供雜湊審查；離線summary不使用這些圖片路徑，直接重跑單一job時則須改成解出的source／template路徑。依賴仍須依建置文件安裝，封存不包含Qt／OpenCV／系統Metal與Python套件的全部源碼。

```sh
mkdir -p build/stage36-review/source build/stage36-review/raw
for bundle in source-snapshot repository-fixtures generated-fixtures; do
  tar -xzf "cpp-core/docs/stage36-coarse-resident/$bundle.tar.gz" -C build/stage36-review/source
done
tar -xzf cpp-core/docs/stage36-coarse-resident/raw-results.tar.gz -C build/stage36-review/raw
python-port/venv/bin/python build/stage36-review/source/cpp-core/tools/benchmark_coarse_resident.py \
  --summarize build/stage36-review/raw/formal/manifest.json \
  --output build/stage36-review/summary.json
cmp cpp-core/docs/stage36-coarse-resident/formal-summary.json build/stage36-review/summary.json

cmake -S build/stage36-review/source/cpp-core -B build/stage36-review/release \
  -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_BUILD_GUI=ON -DFPM_WITH_OPENCV=ON
cmake --build build/stage36-review/release \
  --target fpm_native_pipeline_runner fpm_coarse_resident_test --parallel 8
build/stage36-review/release/experiments/fpm_coarse_resident_test
```

量測期間源碼／binary／輸入雜湊不變。量測後只更新說明文件、log分類映射，以及標頭對FP32例外的註解；算法沒有再改。來源封存以量測版本為準，不能要求其文件索引與今日工作樹完全相同。

## 未來 CUDA 對照

保留常駐中間資料、整數中心化、候選緩衝區與原始抑制規則，能讓 CUDA 版本有相同工作定義。CUDA原生FP64可另建立原double順序的對照，但仍需驗證運算合併、取整與同分順序。這份單工作群組選峰的依序成本也會存在於CUDA，換裝置不保證自行消失。

下一個值得獨立比較的架構是多工作群組的分塊最大值與失效區域更新，保留GPU端候選數與停止狀態；另一條路徑是可證明互不抑制的候選批次。若改成一次取top-K或近似NMS，應獨立列為新演算法，重新檢查擁擠與重疊場景的漏檢，不能把本輪結果視為其準確度證明。
