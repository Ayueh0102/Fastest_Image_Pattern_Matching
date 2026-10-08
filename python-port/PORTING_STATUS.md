# Python 移植狀態（2026-10-08 實測更新）

目前 Python 版保留主要 NCC／金字塔／旋轉流程，**尚未達到原 C++ 的完整等價性**。舊文件的 85%／95% 與各項 100% 宣稱缺少一致驗證依據，這次改以原始碼與可重跑測試說明。

完整分析、Git 基準、效能及逐筆輸出請見：

- [版本稽核](../cpp-core/docs/AUDIT_zh-TW.md)
- [17 組實際影像結果](../cpp-core/docs/validation.json)
- [新 C++ 核心、GUI 與使用方式](../cpp-core/README_zh-TW.md)

已確認差異：

1. `_find_topk_suppressed` 的 `k*50` 預選會讓寬高分峰值排擠其他有效目標；不是原版 BlockMax 的等價實作。
2. `early_stop_count` 在 NMS 前停止精修，沒有保證得到正確的前 N 個不同物件。
3. 零角度使用 IoU、旋轉使用 intersection/template-area，兩種模式的去重疊規則不同。
4. 缺少原版雙角度區間、反色參數與半解析度 FastMode；若傳入舊的 coarse_step/fine_step/scale/topk_per_level，它們不影響現行主流程。
5. 次像素三個角度的 patch 各自以不同峰值為中心，與同一 3D 座標系不一致；零角度設定 subpixel 不會啟用 3D 精修。

實例：`Src6.jpg/Dst6.bmp`，target=15、score=.8、angle=180、overlap=0、area=256、subpixel=false，Python 找到 14 個，而新／舊 C++ 均找到 15 個。另有部分相同數量、不同位置／角度的結果；不能只用数量判斷對齊。

現有 13 個 Python 測試仍通過，主要覆蓋 Src7 與基本流程，未擋住這些反例。本次保留 Python 演算法作比較，沒有宣稱問題已在 Python 修好。要先從 Python 使用新核心，可使用完整的 [ctypes 範例](../cpp-core/examples/python_ctypes.py)；要重新移植演算法，依 [移植契約](../cpp-core/docs/PORTING_zh-TW.md) 逐階段驗證。
