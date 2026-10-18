# Template Matching 專案閱讀與重現指南

這個 repository 同時保留原作者的 Windows MFC 工具、後續語言移植，以及新的可移植 C++ 核心與 CPU／Metal 研究。第一次接觸時，請先分清楚使用哪個版本，再讀對應實驗；相同的「NCC」「GPU」「RGB」名稱可能代表不同的處理階段。

**目前可使用的主體是附完整自有演算法源碼的灰階 C++ 比對核心與 Qt GUI。** 灰階旋轉與多尺度搜尋已有原生 C++／Metal 研究流程；固定尺度稀疏搜尋、RGB 局部評分及其他優化以各自的實驗介面提供。功能存在、數值一致、速度改善、真實場景可靠，是四個需要分開判斷的結論。最新採用設定以[目前建議方案](../cpp-core/docs/CURRENT_BEST_zh-TW.md)為準。

English overview: This repository preserves the original MFC application and develops a portable C++17 matcher, a Qt desktop interface, and optional CPU/Metal experiments. Start with the component map below. Historical benchmark numbers describe specific workloads and builds; they are not universal speed claims. RGB image loading, grayscale localization, and color anomaly detection are separate capabilities.

## 先選一條閱讀路線

| 目的 | 閱讀順序 |
|---|---|
| 直接使用工具 | [C++ 建置與 GUI](../cpp-core/README_zh-TW.md) → [目前建議方案](../cpp-core/docs/CURRENT_BEST_zh-TW.md) |
| 評估是否適用自己的場景 | [完成範圍](../cpp-core/docs/PROJECT_STATUS_zh-TW.md) → [實驗總索引](../cpp-core/docs/EXPERIMENT_REVIEW_zh-TW.md) → 對應輪次的完整報告 |
| 審查時間、記憶體與精度主張 | 對應報告 → protocol → summary／results → 原始 worker 與失敗 → [Log 分類索引](../cpp-core/docs/LOG_CATALOG_zh-TW.md) |
| 移植到其他語言或平台 | [版本稽核](../cpp-core/docs/AUDIT_zh-TW.md) → [移植契約](../cpp-core/docs/PORTING_zh-TW.md) → 公開標頭與測試 |

## 程式與資料在哪裡

| 元件 | 位置 | 用途與界線 |
|---|---|---|
| 原版 Windows GUI | [MatchTool](../MatchTool/) | 原作者 MFC／OpenCV 實作。保留作歷史與移植依據；本機 macOS 研究沒有直接執行 Windows MFC binary |
| 舊 macOS 核心 | [match_core.cpp](../macos-cli/match_core.cpp) | OpenCV／舊 DOTPROD 的歷史執行期參考；不是新版 GUI 的比對核心 |
| 可移植 C++ 核心 | [src](../cpp-core/src/) 與 [公開標頭](../cpp-core/include/fpm/) | 固定尺度灰階 NCC、金字塔、旋轉插值、選峰、幾何與 subpixel；核心不依賴 Qt 或 OpenCV |
| C 與 Python 呼叫入口 | [c_api.h](../cpp-core/include/fpm/c_api.h)、[ctypes 範例](../cpp-core/examples/python_ctypes.py) | 呼叫同一個新 C++ 核心；不等同重新用 Python 實作演算法 |
| 新版 Qt GUI | [mainwindow.cpp](../macos-cli/mainwindow.cpp) | 原始碼仍放在 `macos-cli`，由 `cpp-core` 建置並連結新核心；名稱不代表使用舊核心 |
| 原生研究流程 | [experiments](../cpp-core/experiments/) | CPU／Metal 後端、灰階尺度排程、稀疏提案、連續姿態精修等選配功能；部分研究選項尚未接 GUI |
| 舊 Python 移植 | [python-port](../python-port/README_zh-TW.md) | NumPy／OpenCV 與 Tk GUI 的獨立版本，搜尋與 NMS 定義存在稽核差異，不作新 C++ 等價性保證 |
| 實驗執行與分析 | [tools](../cpp-core/tools/) | Python 控制量測、產生場景與統計；C++ runner 執行原生流程。Fourier 原型的 NumPy FFT 仍屬研究工具 |
| 自動化測試與證據 | [tests](../cpp-core/tests/)、[實驗文件](../cpp-core/docs/) | 現行測試、原始結果、負面結果與來源封存；不要把舊根目錄指南「沒有單元測試」套用到新核心 |
| 原作者樣圖 | [Test Images](../Test%20Images/)、[Result Images](../Result%20Images/) | 圖片與結果示例；顯示框的位置不等於完整人工 ground truth |

「完整源碼」指專案自有演算法、GUI、量測工具與測試的實作均在 repository 或對應來源封存中。Qt、OpenCV、系統 Metal、標準函式庫與 Python 套件仍是外部依賴，沒有把其全部源碼複製到本專案。

## 主要處理流程

灰階多尺度研究流程可以簡化成：

```text
來源影像與裁切模板
    ↓
模板各尺度模型 ＋ 同一影像共用金字塔
    ↓
旋轉與 NCC 粗搜尋 → 選出候選 → 各層細搜尋
    ↓
連續位置／角度／尺度精修 → 有預算的錯峰救援
    ↓
最後驗分 → 去除重疊 → 位置、角度、尺度與分數
```

固定尺度稀疏研究路徑以前段方向特徵提案取代密集金字塔候選搜尋，仍需要 NCC 驗分與精修。兩條路徑的時間不應混在同一張表中。CPU 與 Metal 分工取決於選項。建議路徑仍以CPU精確選峰；第36輪新增[Metal常駐粗正規化與選峰](../cpp-core/docs/COARSE_RESIDENT_zh-TW.md)研究選項，只讀回候選，但大量候選時更慢。這是第36輪的範圍。第37輪另有[完整Metal灰階資料路徑](../cpp-core/docs/FULL_GPU_zh-TW.md)：金字塔、旋轉、粗細搜尋、精修／救援與NMS均留在GPU，一次提交、最後一次等待。它是獨立研究選項，採FP32數值契約；CPU仍負責靜態指令規劃與I/O，RGB驗色及稀疏路徑不包含在內。

同一模板可以重用模型。新影格可透過研究 `Session::updateSource` 更新來源，影像相關資料仍須重新準備；不能把上一張圖的金字塔當成新圖快取。每個 Session 一次只允許一個呼叫者。

## RGB 與真實場景的判讀方式

RGB 圖片可用來驗證三種不同問題：

1. **定位同一模板的所有實例**：看漏檢、誤檢，以及位置／角度／尺度誤差。不同圖案的郵票不應全部算成某一張模板的正確實例。
2. **對齊後比較顏色**：看異色物件能否被區分，並把反光、陰影與曝光變化列入負例。
3. **大量物件的成本**：同時記錄整條流程時間、初始化、記憶體，以及顏色檢查新增的成本。

原始 RGB 圖輸入灰階搜尋，不代表已完成全圖 RGB 搜尋。已知位置的 RGB 評分也不代表具備全圖召回率。以真實照片為底圖、人工改色或貼上物件的案例，應標為半合成，與未修改的實拍分開報告。

本輪密集實拍與顏色驗證的正式結論，沿[實驗總索引](../cpp-core/docs/EXPERIMENT_REVIEW_zh-TW.md)閱讀對應結果與原始資料；本入門頁不另複製會隨新證據更新的數字。

## 建置與第一次驗證

以下從 repository 根目錄建立純核心，不需要 Qt 或 OpenCV：

```sh
cmake -S cpp-core -B build/core -DCMAKE_BUILD_TYPE=Release
cmake --build build/core --config Release -j
ctest --test-dir build/core -C Release --output-on-failure
```

產物包含 `fpm_core`、`fpm_c` 與 `fpm_cli`；此設定 CLI 讀取 PGM，GUI 與一般圖片解碼需要另外啟用。GUI／Metal 研究建置請使用 [C++ README](../cpp-core/README_zh-TW.md) 的指令。新核心源碼可跨平台編譯，但目前大部分研究性能證據來自 Apple Silicon macOS；沒有 CUDA 實測可作硬體比較。

重現某一輪結果時，優先使用那輪封存的來源、輸入、protocol 與執行器。只用目前 HEAD 執行舊 protocol，不一定是在比較當時的程式。先核對封存雜湊，再重算 summary，最後才重跑較昂貴的 CPU／GPU benchmark。

## 如何讀性能與準確度數字

| 指標 | 需要一起確認的條件 |
|---|---|
| 時間 | 讀圖、建模、首次呼叫、暖機 repeat 與 GPU kernel 是否分開；整條流程是否含相同候選與精修預算 |
| 記憶體 | 程序峰值 RSS、顯式 GPU buffer 與常駐容量不能互相替代；釋放配置後 RSS 不一定立即下降 |
| 準確度 | 真正的標註種類、模板所代表的物件類別、TP／FP／FN 與幾何容忍；沒有標註的樣圖只支持觀察 |
| CPU／GPU 比較 | 是否同一建置、同一前段、同一輸入與輸出條件；不同輪次加速比不可相乘 |
| 驗證通過 | 測試、封存完整性與 benchmark gate 是不同檢查；封存驗證成功可以同時包含性能或精度失敗 |

目前的階段結論是按場景選擇已驗證的設定，保留不採用的實驗與共同失敗。影像大小、候選數、模板尺寸、旋轉／尺度範圍與顏色需求都會改變選擇；不能由單一成功樣本推論通用最快或所有實拍都可靠。
