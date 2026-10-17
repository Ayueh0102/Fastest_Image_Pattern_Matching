# 第35輪證據入口

先讀[場景、結論與採用決定](../STAGE35_REAL_RGB_zh-TW.md)，不要把測試通過解讀為實拍精度通過。

| 類別 | 外部可直接閱讀 | 完整封存位置 |
|---|---|---|
| 設定 | [protocol.json](protocol.json) | `evidence/formal/protocol.json` |
| 校準 | [calibration.json](calibration.json) | 同名檔；1/3正常圖有效，暫定閾值 |
| 正式彙整 | [summary.json](summary.json)（閱讀版，保留指標、省略逐pose重複陣列） | `evidence/formal/`內500組job／raw／stderr |
| 失敗與決定 | [decision.json](decision.json)、[performance-failures.json](performance-failures.json) | 原始worker保持失敗值 |
| 先導診斷 | [金字塔漏搜](../STAGE35_PYRAMID_DIAGNOSIS_zh-TW.md) | `evidence/pilot/`、`evidence/pilot-diagnosis/` |
| 資料來源 | [VisA來源](data-provenance.md) | `data/visa/`、`data/fsc/manifest.json`、`evidence/scene-manifests/` |
| 測試／重現 | [validation.json](validation.json) | [validation-logs.tar.gz](validation-logs.tar.gz) |
| 原始證據 | [raw-results.tar.gz](raw-results.tar.gz) | 含protocol、原始結果、資料、review、執行檔及build來源 |
| 完整自有源碼 | [source-and-fixtures.tar.gz](source-and-fixtures.tar.gz) | `source/`，包含C++／Metal／Qt／MFC／Python與測試素材 |
| 雜湊 | [archive.json](archive.json) | 封存內`content-sha256.json`，2213個內容檔 |

兩份主要封存須解壓至同一個新目錄，再執行：

```sh
python3 stage35-real-rgb/reviews/replay_archive.py
```

需要NumPy與OpenCV；不需GPU或原始FSC照片。輸出會同時顯示重播成功、240筆性能配對失敗及實拍色彩精度未過關，這是預期結果。

重新跑算法需要先取得FSC照片，詳報告內的固定版本下載與生成指令。FSC照片及其衍生控制圖片沒有隨Git再散布；VisA原圖／mask依CC BY4.0附作者與授權。封存二進位用於版本追溯，依賴原系統Qt等動態函式庫；在新環境應由來源重建。

全部輪次的檔案角色與封存內路徑可用[Log分類索引](../LOG_CATALOG_zh-TW.md)查詢。
