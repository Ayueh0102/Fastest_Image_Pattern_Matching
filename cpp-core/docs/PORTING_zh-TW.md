# 移植契約：以這份 C++ 為可執行規格

## 程式邊界

| 階段 | 原始碼 | 內容 |
|---|---|---|
| 公開資料模型 | `include/fpm/matcher.hpp` | Gray8 view、擁有影像、已學習 Pattern、設定、結果 |
| 影像與模板學習 | `src/image.cpp` | 步幅檢查、複製、統計量、高斯金字塔、旋轉畫布、ROI 雙線性插值 |
| 相關與候選 | `src/ncc.cpp` | 64-bit 點積、scalar／NEON／SSE2、積分圖、NCC、BlockMax |
| 幾何與去重疊 | `src/geometry.cpp` | 四角／中心、凸多邊形裁切、面積與 greedy NMS |
| 次像素 | `src/subpixel.cpp` | 2D／3D 二次曲面、QR 最小平方、Cholesky 峰值檢查 |
| 搜尋策略 | `src/matcher.cpp` | 角度列舉、分層門檻、所有候選精修、最終排序與 NMS |
| C ABI | `include/fpm/c_api.h`、`src/c_api.cpp` | handle、POD 設定／結果、例外轉錯誤碼 |
| 輸入與輸出 | `tools/cli.cpp` | PGM／可選 OpenCV 圖片解碼、JSON |
| GUI | `../macos-cli/mainwindow.{h,cpp}` | Qt 圖片／顯示、背景工作、模板快取、JSON |

`src/internal.hpp` 揭露階段函式供測試，不是穩定 ABI。公開 C ABI 初始版本為 1；修改 struct 欄位順序或型別時須同步所有語言 binding 並提升 ABI 版本。

## 影像、座標、角度

- 輸入為單通道 unsigned 8-bit，row-major。`stride` 是 bytes，可大於 width；不支援負步幅。呼叫者負責指標所指記憶體的有效長度。影像須非空。
- 原點在左上，x 向右、y 向下。像素中心採整數座標，旋轉中心是 `((W-1)/2,(H-1)/2)`。
- `Match.corners` 順序 LT、RT、RB、LB，框的向量長度為模板 W、H。因此零角度中心為 `(left+W/2,top+H/2)`，不是 `(W-1)/2`。這保留原版輸出慣例。
- 搜尋把**來源影像**旋轉 `searchAngle`，輸出模板角度為 `-searchAngle`；輸出範圍 `(-180,180]`，正值在畫面上逆時針。
- `angleRanges` 使用**輸出角度**，與早期 MFC 區间使用來源旋轉角度不同。要移植舊區間 `[a,b]`，使用新區間 `[-b,-a]`。跨 ±180 的區間要拆開。
- GUI／CLI 的 RGB 轉灰階為 `(77R+150G+29B+128)>>8`；忽略 alpha、EXIF 旋轉。跨語言比較時直接共用同一份 Gray8 bytes，可排除不同圖片解碼器的差異。

## 模板與金字塔

`Pattern` 擁有模板像素與各層統計量；改變呼叫者原始模板不會改變 Pattern。建立新模板或調整縮減面積時重新學習。Match 不接受另一份模板或另一個 pyramid depth，避免舊版 cache 與設定不一致。

層數沿用原 `GetTopLayer`：`limit=floor(sqrt(minReducedArea))²`，原面積整除 4 直到不大於 limit。實際每層尺寸為 `ceil(W/2),ceil(H/2)`；不是每層重新以實際面積判定停止。

`pyrDown` 用分離 `[1,4,6,4,1]` 核，邊界 `REFLECT_101`，兩軸完成後 `(sum+128)/256` 取整數。單列、單行、奇數尺寸均有測試。source 的明暗反轉先於建立金字塔。

模板平均小於 128 時，頂層旋轉畫布填 255，否則填 0；局部 ROI 一律補 0，padding=3，因此 NCC response 是 7×7。

## NCC 與候選

令 N 為模板面積，ST、SI 為模板／視窗總和：

```text
numerator = sum(I*T) - SI * mean(T)
varI = max(sum(I²) - SI²/N, 0)
normalizer = sqrt(varI) * sqrt(sum((T-mean(T))²))
```

分母的原版保護規則完整保留：

1. 模板變異數小於 double epsilon：整張 response=1。
2. `varI <= min(0.5,10*float_epsilon*sum(I²))`：normalizer=0。
3. `abs(numerator)<normalizer` 時正常除法。
4. 否則小於 `1.125*normalizer` 時飽和至 ±1，其他情況為 0。

影像總和／平方和用 double 積分圖；每列點積使用 uint64，SIMD 每 4096 bytes 清空累加器，避開長列的 int32 overflow。scalar 和 SIMD 必須算出相同分數；與原 OpenCV float raw correlation 可能有少量誤差。沒有內建 FFT 路徑。

頂層每個角度最多取 `maxTargets+5` 候選。選最大值後遮掉原版規則的矩形，再找下一個；BlockMax 只更新與遮罩相交的區塊，與全圖掃描等價。**不可只從最高 k×50 個像素挑目標**。相同分數按 row-major 排序。即使 overlap=1，也至少移除已選像素。

## 搜尋與精修

角度步長 `atan(2/max(templateW,templateH))*180/pi`。各層門檻 `minScore*0.9^layer`。初始全圖搜尋後，候選座標反旋轉回來源，再逐層乘 2，測試鄰近角度並取局部 ROI。

必須先精修所有候選再做最終 NMS。粗層排名不是細層分數上界，已找到 N 個候選也不代表找到了 N 個不同物件。

新版角度範圍包含端點但不超出設定，會去掉重複的 ±180；極窄範圍及固定角度也可運作。輸入模板比原始來源任一邊大時回傳空結果，保持舊工具輸入尺寸契約。沒有尺度搜尋。

`stopLayer=1` 保留原 MFC 的半解析度 FastMode，輸出位置乘 2、框仍採原始模板尺寸。若模板只有 level 0，退回完整解析度。正常預設為 0；次像素只在 level 0 執行。

## 次像素

- 3D 使用同一空間座標周圍 3×3、三個角度的 27 點；不可把各角度各自最大值周圍的 patch 當成同一位置。
- 若最後一層最佳角度落在搜尋鄰域邊端，最多移動鄰域三次，嘗試取得能包住峰值的三個角度。仍遵守使用者角度範圍。
- 在最細層把角度間距逐次減半兩次（最終為原步長的 1/4），每次重新取样、包圍峰值，才做 3D 擬合。小模板直接以原始角度步長擬合，可能把位置量化誤差錯估成角度偏移。
- 最細層共做三輪，後兩輪以已精修的浮點位置為新原點；只有第一輪沿用金字塔的座標乘 2。重新取樣可減少整數平移網格對角度峰值的偏差。固定角度同樣允許重置原點後做 2D 精修。
- 在相對 x/y/angle-step 座標擬合十參數二次曲面，QR 求最小平方；利用 `-H` 的 Cholesky 同時驗證 Hessian 負定並求峰值。
- 只接受 finite、各軸偏移在 [-1,1] 的局部最大值；平坦、鞍點、奇異或越界解均拒絕。3D 不適用时，可用 2D 位置精修，零角度也可生效。
- 分數是最後一輪取樣峰值的 NCC，不把擬合曲面的預測值偽裝成重新量測的 NCC；仍非最終擬合座標的重新評分。
- 次像素是局部近似，不保證每個樣本都提高精度。連續紋理的獨立合成案例會驗證已知位移／旋轉下的改善。

## 幾何與 NMS

使用 Sutherland–Hodgman 凸多邊形裁切，三角形面積加總，分母固定為模板框面積。**不是 IoU**，零角度和旋轉角度共用同一規則。完整重合仍去重，即使 maxOverlap=1。先 AABB 篩掉不相交框，再算多邊形交集。

NMS 與輸出按 score 降序；平手依 center.y、center.x、angle 決定。固定輸入與設定得到可重現順序，但原版本平手順序不同不代表數值錯誤。

## 跨語言移植與驗收順序

1. 先完成 ImageView/stride、RGB 灰階、Gaussian pyramid，通過奇數尺寸／單列／單行測試。
2. 寫 scalar 點積與 NCC，對照 `tests/core_test.cpp` 的獨立直接公式，以及 `tests/opencv_test.cpp` 的 OpenCV 對照；先不要加入 FFT。
3. 做 rotate/ROI/反座標，再驗證已知 90° 的目標與四角／中心。
4. 做完整候選搜尋和 BlockMax；寬峰值加遠處第二目標的案例不可漏檢。
5. 做 NMS；0.6 的 intersection/template-area 必須與 0.42857 的 IoU 分清。
6. 完成金字塔搜尋、分層門檻、反色、區間、FastMode，再加入有有效性檢查的次像素求解。
7. 執行 `compare_versions.py` 保存每筆結果，依物件位置配對再比較，不只比較第一筆分數或總數。
8. 最後才加入 SIMD、FFT、執行緒或 GPU，並讓加速版通過同一套測試。

`tests/core_test.cpp` 是無外部測試框架的 C++ 驗證程式，Release 也執行檢查；`tests/c_api_test.c` 用真正 C 編譯器驗證 ABI；`tests/gui_test.cpp` 用 Qt Test 驗證使用者操作流程。這讓純核心能離線建置，不需要先下載 GoogleTest。
