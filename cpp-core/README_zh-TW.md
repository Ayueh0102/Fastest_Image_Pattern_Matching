# 可移植 C++ 核心與桌面介面

這個版本把比對演算法整理成 **C++17、無第三方依賴的核心函式庫**。GUI、CLI、C ABI 共用同一核心；金字塔、旋轉插值、NCC、SIMD、BlockMax、矩形交集、次像素求解的原始碼都在此目錄。

- [歷史版本與移植稽核](docs/AUDIT_zh-TW.md)
- [演算法契約與移植順序](docs/PORTING_zh-TW.md)
- [17 組影像實測原始資料](docs/validation.json)
- [旋轉盤 3／6 個結果的設定實驗](docs/rotor-validation.json)（原因與演算法對照見稽核報告）
- [原解析度密集角度搜尋與精修檢查](docs/rotation-search-validation.json)、[對齊比較圖](docs/rotation-search-validation.png)
- [相同參數下的模式效能比較](docs/mode-benchmark.json)
- [CPU／Metal／Fourier 實驗方法與重跑](docs/EXPERIMENTS_zh-TW.md)
- [111 組時間、記憶體與精度實測](docs/EXPERIMENT_RESULTS_zh-TW.md)
- [實圖完整搜尋並排圖、旋轉尺度誤差與 RGB 批次比較](docs/REAL_IMAGE_RESULTS_zh-TW.md)
- [連續姿態精修、NCC 保守篩選與大量候選記憶體實測](docs/PERFORMANCE_ACCURACY_zh-TW.md)
- [GPU 連續 NCC 精修、CPU／Metal 時間與記憶體比較](docs/GPU_REFINEMENT_zh-TW.md)
- [CPU／Metal 自動分流、延遲初始化與效能交界曲線](docs/METAL_ROUTING_zh-TW.md)
- [Metal 傳輸、提交等待與底層並行瓶頸實測](docs/METAL_COST_ANALYSIS_zh-TW.md)
- [Metal GPU 常駐求解、停止控制與新舊版比較](docs/METAL_RESIDENT_REFINEMENT_zh-TW.md)
- [NCC 局部錯峰救援、8K／大量候選與 Metal 粗 NCC 比較](docs/NCC_RESCUE_AND_METAL_COARSE_zh-TW.md)
- [搜尋階段計時、Metal 精確金字塔、角度批次與 8K 時間／記憶體比較](docs/PIPELINE_STAGES_AND_METAL_PYRAMID_zh-TW.md)
- [同圖跨尺度金字塔共用、Metal 批次細搜尋與完整比較](docs/SHARED_FRAME_AND_FINE_METAL_zh-TW.md)
- [粗搜尋精確選峰與 GPU NCC 平行方式比較](docs/COARSE_SELECTION_AND_PARALLEL_zh-TW.md)
- [128／256／512 多物件實驗](docs/MANY_OBJECT_STUDY_zh-TW.md)
- [CPU／GPU 粗搜尋重疊](docs/COARSE_OVERLAP_zh-TW.md)
- [NCC 記憶體分塊與 FlashAttention 原理分析](docs/NCC_TILING_AND_OVERLAP_RESEARCH_zh-TW.md)

GUI 的完整原始碼位於 `../macos-cli/mainwindow.{h,cpp}` 與 `main.cpp`，使用 Qt6。CLI 可選用 OpenCV 解碼圖片。Qt、OpenCV、C++ 標準函式庫本身並未複製進專案；比對核心沒有呼叫它們的影像演算法，也沒有依賴未提供源碼的 FPM DLL／SO。

## 編譯純核心（不需要 Qt 或 OpenCV）

從 repository 根目錄執行：

```sh
cmake -S cpp-core -B build/core -DCMAKE_BUILD_TYPE=Release
cmake --build build/core --config Release -j
ctest --test-dir build/core -C Release --output-on-failure
```

產出 `fpm_core` 靜態函式庫、`fpm_c` 動態函式庫及 `fpm_cli`。沒有 OpenCV 的 CLI 支援 8-bit P5 PGM，C++／C API 直接接收記憶體中的灰階影像。Windows 多組態產物位於 `Release/`。

也可從根目錄 CMake 編譯；MSVC 預設同時保留原 MFC 工具，要只編譯新核心請加 `-DFPM_BUILD_MFC=OFF`。

## 編譯 GUI 與一般圖片 CLI

準備 C++17 編譯器、CMake 3.16+、Qt6 Widgets／Concurrent（測試另需 Qt Test）及 OpenCV 的 core／imgproc／imgcodecs。可沿用本機已安裝的 Qt6/OpenCV，或用 CMake 的 `CMAKE_PREFIX_PATH` 指定位置；Windows 原有 OpenCV vcpkg manifest 可繼續使用，Qt 需另外提供。

```sh
cmake -S cpp-core -B build/portable -DCMAKE_BUILD_TYPE=Release \
  -DFPM_BUILD_GUI=ON -DFPM_WITH_OPENCV=ON
cmake --build build/portable --config Release -j
ctest --test-dir build/portable -C Release --output-on-failure
```

macOS 啟動：

```sh
build/portable/fpm_gui.app/Contents/MacOS/fpm_gui --gui \
  --src "Test Images/Src7.bmp" --tmpl "Test Images/Dst7.bmp"
```

或在 Finder 開啟 `build/portable/fpm_gui.app`。Linux 為 `build/portable/fpm_gui`，Windows 為 `build/portable/Release/fpm_gui.exe`。這些平台共用源碼；本次實測平台為 Apple Silicon macOS，尚未實機驗證 Windows/Linux。

保留的舊建置入口：`cmake -S macos-cli -B build/desktop`，產出 macOS 的 `build/desktop/MatchTool.app`。`macos-cli/match_core.{h,cpp}` 僅作歷史比較，不再連結到新版 GUI。

GUI 提供繁中／簡中／英文、來源與模板載入／拖放、縮放、結果框與選取標示、分數／重疊／角度／金字塔設定、兩組角度區間、反色、SIMD、次像素、半解析度快速模式與 JSON 匯出。比對在背景執行，期間可縮放檢視影像；變更圖片會清除舊結果，模板與縮減面積未改變時會重用已學習的模板。

GUI 用 Qt 解碼與繪圖，**可把 `FPM_WITH_OPENCV` 設為 OFF**；此時 GUI 仍可開啟 Qt 支援的圖片，CLI 僅支援 PGM。GUI 和一般圖片 CLI 共用明確的 RGB 灰階轉換，忽略 alpha 與 EXIF 自動旋轉。

## CLI

```sh
build/portable/fpm_cli --src "Test Images/Src7.bmp" \
  --tmpl "Test Images/Dst7.bmp" --target-num 5 --score .8 --angle 180

build/portable/fpm_cli --src "Test Images/Src7.bmp" \
  --tmpl "Test Images/Dst7.bmp" --target-num 5 --score .8 \
  --angle-range 115 125 --angle-range -125 -115 --subpixel
```

輸出 JSON，包含中心、四角、角度、分數及學習／比對時間。另有 `--overlap`、`--min-area`、`--invert`、`--no-simd`、`--fast`；`--help` 可看說明。`--fast` 在半解析度停止，犧牲精度以減少運算；需要完整精度時保留預設值。

`Src7` 旋轉盤若要顯示 6 個候選，使用 `--target-num 70 --score .5 --angle 180 --overlap .1`（GUI 填入相同設定）。相鄰矩形約有 9%～10% 交疊；重疊設為 0 會剩下 3 個。另 3 個的 NCC 約 0.64～0.65，因此最低分數設為 .8 也只會剩下 3 個；詳見稽核報告。

啟用 `--subpixel` 時，最細層會加密角度並反覆重設浮點原點，以減少小模板的平移／角度耦合偏差，代價是增加時間。此圖精修後的一組交疊約 10.006%，請用 `--overlap .2` 避免卡在 .1 的去重疊界限；六個結果中仍有三個分數約 .64～.65。

### 旋轉來源圖與旋轉模板的獨立比較

`tools/audit_rotation_direction.py` 以六個已知位置比較兩個方向；這是 OpenCV 診斷工具，尚未改變 C++／GUI 的來源圖旋轉流程。兩邊均測試雙線性、雙三次、Lanczos4，並各自在種子位置 ±3 px、角度 ±1° 內做局部座標搜尋，最後步距為 0.1 px／0.01°。先用非對稱合成模板的 0°／90°／180° 精確旋轉，確認兩個方向的座標與分數正確。

旋轉模板時遮罩排除外接矩形的空白角落；為排除插值邊界影響，兩個方向一律略去模板最外側 4 px。因此下面數字與完整矩形的正式 matcher 分數稍有不同。雙線性插值的三個低分位置為：

| 位置（診斷 JSON index） | 旋轉來源圖 | 旋轉模板 |
|---|---:|---:|
| 3 | 0.648119 | 0.648296 |
| 4 | 0.647263 | 0.647257 |
| 5 | 0.637084 | 0.637101 |

六個位置雙線性分數的最大方向差約 0.00018；雙三次與 Lanczos4 也沒有讓低分三個變成高分。這支持「此樣本低分的主要原因不是旋轉哪一邊」，不代表所有影像／模板都等價。兩個方向的離散像素取樣集合不同，此局部搜尋也不是全域最大值證明，更不是速度比較。完整固定姿態與精修結果、輸入／工具雜湊保留於 [rotation-direction-validation.json](docs/rotation-direction-validation.json)。重跑：`python-port/venv/bin/python cpp-core/tools/audit_rotation_direction.py`。

## 模式、速度與 Subpixel

旋轉、尺度、FFT、RGB 與 CPU／GPU 分工的數學推導及研究原型，見 [研究文件](docs/RESEARCH_FOURIER_SCALE_RGB_zh-TW.md)。正式核心目前仍是固定尺度灰階比對；研究原型未接入 GUI。

`codex/ncc-fourier-gpu-experiments` 已提供可實跑的 C++ 多核與 Metal 浮點／RGB 姿態評分器，透過 `FPM_BUILD_EXPERIMENTS=ON` 額外編譯。實測區分常駐資料的評分時間、含 CPU 提案與同步的 ROI 管線時間、初始化與記憶體；[結果](docs/EXPERIMENT_RESULTS_zh-TW.md) 同時保留小批次 GPU 較慢、Fourier 裁切失敗及回退後仍不夠精確的案例。NumPy FFT 仍在 CPU，CUDA 是尚待實作的對應設計。

2026-10-08，Apple Silicon Release；Src7 4024×3036、Dst7 762×521，target=70、score=.5、angle=180、overlap=.2、minArea=256。各模式先暖機一次，再序列量測三次，取比對時間中位數，排除讀圖與模板學習：

| 模式 | 比對時間 ms | 此樣本數量 | 精度／實作差異 |
|---|---:|---:|---|
| 舊 macOS C++ | 177.600 | 6 | OpenCV＋舊 DOTPROD，全解析度，次像素關閉 |
| 新 C++ scalar | 719.898 | 6 | 全解析度，關閉 SIMD／次像素 |
| 新 C++ NEON 一般模式 | 344.348 | 6 | 全解析度，次像素關閉 |
| 新 C++ FastMode | 134.198 | 6 | NEON，半解析度停止，降低精度 |
| 新 C++ Subpixel | 3188.826 | 6 | NEON，全解析度＋三輪位置／角度精修 |

這些是同機、同圖、同參數的模式比較，數量相同不等於精度相同。原始結果和各次時間保留於 `docs/mode-benchmark.json`。一般模式有加速，但尚未追平舊 C++；FastMode 不能拿來當作等精度勝過舊版的證據。GUI 的背景工作避免阻塞視窗，並不是讓核心自動多執行緒加速。

已使用金字塔、縮小旋轉畫布、局部 ROI、NEON／SSE2、積分圖與依條件啟用的 BlockMax。核心尚未加入專用 DOTPROD、FFT 相關後端或多核心候選搜尋，不能宣稱已用上所有最快路徑。

Subpixel 在 GUI「次像素位置與角度精修」、CLI `--subpixel`、C++ `MatchConfig::subpixel`、C ABI `fpm_config.subpixel` 均已接通。旋轉模式使用 27 點三維二次曲面精修 x／y／角度，固定角度可用 2D fallback；QR 求解、負定 Hessian 與偏移界限檢查也都有源碼。FastMode 在半解析度結束，會跳過最細層 Subpixel，GUI 因此停用該選項。

## C++ 與其他語言介面

```cpp
#include <fpm/matcher.hpp>

fpm::ImageView source{sourceBytes, sourceWidth, sourceHeight, sourceStride};
fpm::ImageView templ{templateBytes, templateWidth, templateHeight, templateStride};
fpm::Pattern pattern(templ, 256); // 複製模板，可跨影格／執行緒重用
fpm::MatchConfig config;
config.maxTargets = 5;
config.minScore = 0.8;
config.toleranceAngle = 180;
auto results = fpm::match(source, pattern, config);
```

CMake 使用 `add_subdirectory(.../cpp-core)`，連結 `fpm_core`；純 C 或其他語言連結 `fpm_c`，介面在 `include/fpm/c_api.h`。這次也附上可直接執行的 Python ctypes 範例：

```sh
python3 cpp-core/examples/python_ctypes.py build/portable/libfpm_c.dylib
```

Linux 使用 `libfpm_c.so`，Windows 使用 `fpm_c.dll`。ctypes 範例只依賴 Python 標準函式庫。要**重新用其他語言實作**演算法時，依 `docs/PORTING_zh-TW.md` 的階段契約及測試順序移植；要**從其他語言呼叫 C++**，則使用 C ABI。

## 驗證與限制

```sh
# 需現有 Python port 的 numpy、opencv、pytest 環境
PYTHONPATH=python-port/src python-port/venv/bin/python -m pytest python-port/tests -q
python-port/venv/bin/python cpp-core/tools/compare_versions.py --build build/portable
python-port/venv/bin/python cpp-core/tools/diagnose_rotor.py --build build/portable
python-port/venv/bin/python cpp-core/tools/audit_rotation_search.py --build build/portable
python-port/venv/bin/python cpp-core/tools/benchmark_modes.py --build build/portable

# 無依賴 scalar + ASan/UBSan（Clang/GCC）
cmake -S cpp-core -B build/sanitized -DCMAKE_BUILD_TYPE=Debug \
  -DFPM_SANITIZERS=ON -DFPM_DISABLE_SIMD=ON
cmake --build build/sanitized -j
ctest --test-dir build/sanitized --output-on-failure
```

演算法是旋轉搜尋與金字塔候選精修，沒有比例縮放搜尋，也不是窮舉所有位置／角度的全域最佳解保證。高頻紋理在縮圖後可能失去辨識性；可提高 `minReducedArea` 減少金字塔層數。均勻模板沿用原版 NCC=1 的退化規則，因此不適合拿來辨識位置。

新版不是逐位元複製 OpenCV：旋轉用雙精度雙線性取樣，角度範圍嚴格受限，次像素解有有效性檢查。預設 NEON／SSE2 直接相關保留完整源碼，但目前部分資料比舊版 OpenCV/DOTPROD 組合慢；精度與速度數據見稽核報告。

原始專案及衍生程式遵循 repository 根目錄的 BSD-2-Clause `LICENSE`。CMake install 亦會安裝該授權文字。
