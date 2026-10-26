# 依模板資訊量選擇粗搜尋頂層（小模板召回）

第49輪，2026-10-26，Apple M4 Pro／48 GiB，macOS 27.0.1。[第35輪](STAGE35_REAL_RGB_zh-TW.md)的43×30郵票模板在原設定（`min_area=256`、proposal 0.5、limit 100）縮到11×8後只找回4/49，改成512／0.25／300可恢復49/49，但暖機時間由25.58變成100.28ms；另一個58×71的heldout模板原本就是49/49，改了只變慢。本輪測試一條預設關閉的規則：只看模板本身，預測每個候選頂層在真實位置的NCC，取預測分數仍達門檻的最深一層，從不比原設定更深。目標是小、低紋理的模板在較高解析度起搜，原本就找得到的模板維持原深度、不付時間代價。

**結論：這條規則在保留集失敗，不推薦，保留為預設關閉的研究選項。** 保留集兩個召回不足的模板（fsc2931b0 2/98、fsc2931b1 83/98，原設定共漏111個）規則一個都沒改、一個都沒救回；30個原本就足夠的模板有11個被改淺，包括第35輪那個「改了只變慢」的模板（仍98/98，但時間1.49倍）。整體TP 3025/3136，比等時間預算組（3058）、與模板無關的固定512（3040）都少，也明顯少於第35輪手動設定（3130）。事前登錄的H1、H2、H2r、H3、H4、H7未通過，H5（負控制零FP）與H6（正場景零FP）通過。事後診斷顯示兩個失敗的根源都是背景：規則假設背景在模板平均±2個標準差之內，而保留集兩個不足模板的場景背景在3.5與2.6個標準差之外（低對比模板放在對比強的背景上，頂層邊界被背景污染）；同時對背景在範圍內的模板，取兩個背景中較差者又太悲觀。GUI、核心與研究預設都不改。

## 規則

`research_pipeline` 新增 `Settings::pyramidRule`（預設0）與 `pyramidRuleMinScore`；兩個runner對應 `pyramid_rule`、`pyramid_rule_min_score`（[pyramid_rule.cpp](../experiments/pyramid_rule.cpp)）。對每個候選頂層L（0到 `min_area` 給的原頂層）：

1. 以雙線性取樣把模板畫在平坦背景上，8個固定姿態：搜尋格點角度約20／65／110／155°，各偏離格點¼個粗角度步、物件中心偏離整數格點¼個頂層像素。
2. 背景兩種：模板平均 −2 與 +2 個標準差（夾在0..255）。
3. 用比對器本身的 `pyrDown` 縮L次、`rotateCanvas` 依粗角度格點轉回，取已學習的第L層模板在物件附近±1像素內的最佳NCC。
4. 預測分數＝兩種背景中較差者的8姿態平均。選預測分數 ≥ 門檻的最深一層；都不到則取分數最高的最深一層；從不比原頂層深，深度不變時沿用原 `min_area`（所以輸出逐位與原設定相同）。

只作用於密集金字塔路徑（混合或CPU），`fullGpu` 與 `sparseOrientation` 會拒絕。成本只在建立Session時：每個尺度模型中位1.9ms、最大8.35ms，算在 `search_construct_ms`，不在搜尋時間內。

## 設計

| 組別 | 設定 | 用途 |
|---|---|---|
| original | 256／0.5／100 | 第35輪原預設（geometry_original） |
| stage35_manual | 512／0.25／300 | 第35輪手動較淺金字塔（geometry） |
| adaptive | original＋規則，門檻0.50 | 新的自動規則 |
| budget | 256／0.40／200 | 公平的等時間預算：開發集總時間不超過adaptive的最大候選預算，不用模板資訊 |
| fixed512 | 512／0.5／100 | 與模板無關的較淺深度，候選預算同原設定 |
| adaptive_r0 | original＋規則，門檻0.75 | 描述用（1個block、不判定）：被取代的R0調參目標選出的門檻 |

共用第35輪的搜尋（Metal粗搜尋＋Metal精修、尺度[1]、角度180、最終分數0.8、overlap 0.2、stop_layer 1、fine mode 2／融合／角度重用、u8／shared、8執行緒），runner為 `fpm_color_verify_runner` 的 `geometry_only`。

**資料切分**依照片、種子與檔案分開：開發集為FSC147照片2933、1913、2927、825、1924的全部3個範例框、程序紋理種子49100–49107、Dst10，以及第35輪的development clean／exposure；保留集為照片7417、2931、7222、7263、1921、1944、2928、種子49200–49207、Dst4、Dst3，以及第35輪的heldout clean／exposure（回歸檢查模板）。每個模板4個灰階場景：49個旋轉副本（7×7格、角度U(−175,175)、背景U(60,190)）、同一場景加σ=4雜訊、空場景（只有背景與雜訊）、49個無關程序紋理的干擾場景（無目標）。GT由[產生器](../tools/stage49_scenes.py)以OpenCV雙線性warp解析產生，與C++比對器無關；保留集產生器會核對protocol凍結的SHA-256，所以保留集與GT在protocol提交後才產生。開發集25個模板（50正＋48負場景），保留集32個模板（64正＋62負）。FSC照片與其衍生影像沒有確認再散布授權，沿用第35輪做法不放進git，只存雜湊、種子與產生程式。

## 開發集調參

開發集掃描把每個正場景在每個可能的頂層都跑一次（其餘參數同原設定），並收集每個模板的預測分數。25個模板中21個在原深度就98/98；不足的是第35輪原場景（25/98，背景120）、fsc2933b1（0/98）、fsc825b1（40/98），以及有旋轉對稱、在任何深度都有角度歧義的dst10。同一個2933模板在本輪背景180的場景是98/98，在第35輪背景120的場景只有25/98：成敗取決於背景。

門檻只用開發集調。掃描前提交的目標R0（`623d95c`：先比「起始層比需要深」的模板數，再比時間）選出0.75，代價是開發集時間8.2倍、25個模板改了15個，違背本輪「找得到的維持原深度」的目標。看過掃描曲線後（保留集尚不存在），改用修訂R1（`a952aaf`）：開發集總時間不超過原深度2倍的門檻中取TP最多者、再取最小門檻，選出0.50（開發集TP 2216→2364、時間1.58倍）。2倍上限是看過曲線後的判斷，已在[protocol](stage49-adaptive-pyramid/protocol.json)揭露；R0的0.75以 `adaptive_r0` 在保留集跑一次供對照。等時間預算組依事前規則取開發集總時間不超過adaptive（2174.7ms）的最大候選預算：0.40／200（1972.0ms，TP 2259；下一檔0.35／200為2256.4ms）。

開發集對保留集的預測（事前寫進protocol）：H1預期失敗（開發集只回收148/229＝65%）、H2預期失敗（22個足夠模板保持18個、時間1.45倍）、H7預期平手（固定512在開發集同為2364 TP、2256ms）。

## 正式結果

正式窗口2026-10-26 09:57–10:11 UTC，1396個worker（正場景3個block、負場景與adaptive_r0各1個），每個worker是新程序：第0次呼叫、1次暖機、3次量測。時間＝`Session::run` 的牆鐘時間；每個worker取3次中位、每個場景取各block中位、總和是64個正場景中位數的和。全部1396個worker的重複呼叫輸出完全相同，各block之間輸出也完全相同。

| 組別 | 嚴格TP／3136 | 正場景FP | 負場景FP | 搜尋時間總和 ms | 對original |
|---|---:|---:|---:|---:|---:|
| original | 3022 | 0 | 0 | 2190.6 | 1.00 |
| stage35_manual | 3130 | 0 | 0 | 6237.5 | 2.85 |
| adaptive | 3025 | 0 | 0 | 3004.7 | 1.37 |
| budget | 3058 | 0 | 0 | 3036.5 | 1.39 |
| fixed512 | 3040 | 0 | 0 | 3862.6 | 1.76 |
| adaptive_r0（描述） | 3136 | 0 | 0 | 22569.0 | 10.30 |

嚴格GT＝位置≤2px、角度≤1°、尺度比[0.98,1.02]的一對一最大配對。[正式摘要](stage49-adaptive-pyramid/formal-summary.json)

**不足模板（原設定召回<90%）只有兩個，規則都沒動：**

| 模板 | 原頂層的預測分數 | original | adaptive | budget | fixed512 | stage35_manual | adaptive_r0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fsc2931b0（40×84） | 0.649 | 2/98 | 2/98 | 28/98 | 2/98 | 92/98 | 98/98 |
| fsc2931b1（44×36） | 0.619 | 83/98 | 83/98 | 92/98 | 98/98 | 98/98 | 98/98 |

**被規則改淺的11個模板原本都已≥96/98：** 7417的3個範例框與第35輪heldout原場景（頂層3→2，預測0.09–0.34）、1944b1／b2、2928b1（3→2）、7222b2、7263b0／b1、proc49201（2→1）。其中3個物件因此多找到（7417b1 96→98、7263b1 97→98），時間在這些模板是1.35–5.6倍。其餘19個足夠模板保持原深度，輸出與original逐位相同。

**回歸檢查（第35輪heldout模板）：** 規則把頂層由3改成2，仍是98/98，時間62.2→92.6ms（1.49倍），正是本輪想避免的「改了只變慢」。

記憶體：每個程序的peak RSS中位數original 36.7、stage35_manual 46.5、adaptive 40.8、budget 36.8、fixed512 45.9 MiB（adaptive_r0 58.9，最大479.0 MiB出現在fsc2931b2改到第0層時）；命名GPU配置的粗搜尋frame中位數2.64／5.03／4.35／2.64／5.03 MiB，精細frame與pose在各組約1.0與2.8 MiB。RSS與GPU配置不相加。負場景（空場景與干擾場景）的時間總和也隨深度改變：1444.5／5456.0／2108.1／2203.0／2944.2ms。

## 假說判定

| 事前假說 | 判定 | 依據 |
|---|---|---|
| H1 不足模板回收≥80%、無場景少1個以上 | 否 | 回收0/111，規則保持兩個模板的深度 |
| H2 足夠模板≥90%保持深度、保持者輸出相同、時間≤1.05倍 | 否 | 保持19/30（保持者輸出全部相同），時間1.405倍 |
| H2r 第35輪heldout模板保持深度、49/49、時間≤1.05倍 | 否 | 頂層3→2，98/98，1.488倍 |
| H3 對等時間預算：TP不少、時間≤1.10倍 | 否 | 3025＜3058（時間0.990倍） |
| H4 對第35輪手動：時間≤0.75倍、TP不少於手動−2% | 否 | 時間0.482倍通過，但3025＜3130−62.7 |
| H5 負場景adaptive零FP、不多於original | 是 | 各組負場景FP都是0 |
| H6 正場景FP不多於original＋1% | 是 | 各組正場景FP都是0 |
| H7 對固定512：TP不少、時間不多 | 否 | 3025＜3040（時間0.778倍） |

依事前決策規則（H1、H2、H2r、H5、H6全過且H3／H7至少一項過才推薦），**不推薦**。所有未通過的判定都由與主機負載無關的條件決定（TP、保持深度的模板數）。[判定與失敗](stage49-adaptive-pyramid/decision-and-failures.json)

## 事後診斷（不影響判定）

以下是看過保留集結果後做的解釋，不是事前假說。

- **背景超出假設：** 以「場景背景離模板平均幾個標準差」來看，兩個不足模板是3.51與2.59個標準差（模板標準差只有13.4與17.2），超出規則假設的2個標準差；規則用的背景是129／182與122／191，實際背景108／112。32個保留集模板中只有5個超過2個標準差，其中就包含這兩個。這和第35輪原場景的機制相同：同一模板在背景120時頂層失敗，在背景180時沒問題。
- **對背景在範圍內的模板太悲觀：** 被改淺的11個模板背景都在2.2個標準差以內，多數在1.5以內；規則取兩個背景中較差者的預測（例如7417b0第3層0.341），但實際搜尋在第3層用的門檻是0.5×0.9³＝0.3645，單一門檻0.50也沒有反映各層真正的接受分數。
- **候選，不是分數：** fsc2931b0在原第2層（refine關閉時）粗到細的候選是0個，改第1層後49個都在GT附近、49/49；開發集的fsc2933b1在第1層則是100個候選（＝limit）被非目標峰占滿，只剩10個在GT附近。規則只預測真實位置的分數，預測不到假峰競爭。
- **R0的門檻：** 0.75在保留集找回全部3136個，但時間10.3倍；等於「幾乎都改淺」，不是有資訊的選擇。

下一個有意義的嘗試是讓背景來自實際影格（例如來源影像在模板外的強度分布）、把預測分數和該層真正的接受分數比較，並把候選數上限納入；這些都需要新的事前登錄與新的保留集。

## 量測方法與界線

- **主機：** 正式期間沒有其他session的計時鎖，也沒有建置或benchmark程序；但系統的 `mediaanalysisd` 在1194個worker前佔滿一核，`node`／`chrome-headless` 約330個、`Codex (Renderer)` 192個；1分鐘負載在624個worker前超過6（最高14.1）。只有92個worker屬於安靜狀態。protocol規定主機狀態只記錄、不影響判定；事後在少數共同的安靜案例上，安靜worker的時間比與全部block的時間比相差約1%（[host-sensitivity](stage49-adaptive-pyramid/validation.json)）。計時數字因此比前幾輪雜訊大，未通過的判定則不受影響。
- **開發集校準期間：** `mediaanalysisd` 在400個worker中的311個前佔滿一核；第50輪session在09:50:07Z的一次runner冒煙與校準重疊（3個worker記錄到 `fpm_native_pipeline_runner`）；我在校準進行中改了3個Python檔，所以該phase的源碼指紋前後不同（binary與輸入不變）。等時間預算的選擇照校準結果保留。
- **其他揭露：** 第一次開發集掃描被主機gate拒絕（mediaanalysisd 106%、負載4.3），之後只有計時phase會拒絕，負載上限改為6（系統的ANECompilerService已連續數天佔約90%一核）；第一次量測前驗證因凍結源碼樹缺少 `Test Images` 而失敗（gui_workflow與7個Python測試），補上同一commit的資料夾後全部通過；第一版單元測試的條紋週期2.4px在第0層就已混疊，實作commit改成8px。全部記錄在protocol的 `exploratory_disclosure`。
- 只有一台M4 Pro；場景是半合成的平坦背景、尺度固定1；FSC範例框不是同一模板的身份GT，本輪只用它們當模板。

## 驗證

| 檢查 | 結果與範圍 |
|---|---|
| 測試先失敗 | 紅燈commit `5d21fc8`：`research_pyramid_rule`（CPU／Metal）、`native_runner_job_keys`、`native_runner_pyramid_rule` 與color verify的決策測試失敗；實作 `0457625` 後全部通過。最終測試對紅燈空殼（突變M0）同樣失敗 |
| 編譯突變 | 隔離副本12個突變（門檻被忽略、取較好背景、面積下取整、Session忽略有效面積、不轉回、1個標準差、取最淺、接受fullGpu、兩個runner的鍵處理、深度不變卻換面積、紅燈空殼）全部被抓到，還原後測試通過 |
| 量測工具 | `test_adaptive_pyramid_harness.py` 17項負控制（深度鏡像、job不含GT、計畫、切分防護、計時鎖等待、調參R0／R1、校準、判定通過與失敗、主機旗標、保留集需要凍結的protocol） |
| 量測前Release | 凍結commit `8f8bdb9` 全新建置：ctest 87/87、Python harness `test_*.py` 751、其他Python 49、color verify `--metal` 11 |
| 獨立稽核 | 系統Python 3.9、只用stdlib、不匯入harness、不啟動GPU：1396個worker雜湊與「job不含GT」、自寫的一對一配對重算GT、756個cell與3054個數值零不符，判定與發佈相同 |
| 封存重播 | 從封存解出的原始結果用凍結driver重算正式摘要、開發集調參與校準，三份逐位相同；開發集與保留集場景由產生器重建，224個場景與模板的雜湊及GT全部相同 |
| ASan／UBSan與最終建置 | 見[validation](stage49-adaptive-pyramid/validation.json)（含experiments的scalar與SIMD兩組，以及最終head的全新Release研究建置） |

## 封存

[封存清單與SHA256](stage49-adaptive-pyramid/archive.json)：[protocol.json](stage49-adaptive-pyramid/protocol.json)（在產生保留集前提交）、source-snapshot.tar.gz與source-sha256.json（凍結commit `8f8bdb9` 的325個已提交源碼檔）、raw-results-development.tar.gz（開發集掃描與校準）、raw-results-formal.tar.gz（正式1396個worker的job、結果與stdout／stderr）、scene-metadata.tar.gz（兩個切分的cases.json與provenance，含GT與種子）、input-sha256.json（281個輸入影像的雜湊）、dev-tuning.json、dev-tuning-r0.json、dev-calibration.json、[formal-summary.json](stage49-adaptive-pyramid/formal-summary.json)、[decision-and-failures.json](stage49-adaptive-pyramid/decision-and-failures.json)、[validation.json](stage49-adaptive-pyramid/validation.json)與validation-logs.tar.gz（紅燈、建置與測試log、突變、稽核、重播、主機敏感度與事後診斷）。

[規則](../experiments/pyramid_rule.cpp)、[場景產生器](../tools/stage49_scenes.py)、[量測工具](../tools/benchmark_adaptive_pyramid.py)、[負控制](../tests/test_adaptive_pyramid_harness.py)、[規則測試](../tests/pyramid_rule_test.cpp)、[封存工具](../tools/stage49_archive.py)。重現量測應使用凍結的源碼快照與本機的FSC鏡像（`build/stage35-real-rgb/data/fsc`）。

## 未涵蓋

完整GPU路徑、多尺度、實拍場景、非平坦背景、其他Metal裝置、Windows／Linux都沒有測；規則只決定頂層，不調整候選預算。
