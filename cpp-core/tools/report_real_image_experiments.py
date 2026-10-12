"""Render measured detections and a Chinese report; never reruns the search."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAMES = dict(cpu_fixed='CPU / full search', cpu_fast='CPU / FastMode', cpu_grid='CPU / 3-scale bank',
             hybrid_cpu='CPU coarse + CPU refine', hybrid_metal='CPU coarse + GPU refine',
             fourier_metal='CPU coarse + Fourier + GPU')
ZH = dict(cpu_fixed='原有 CPU 完整搜尋', cpu_fast='原有 CPU FastMode', cpu_grid='原有 CPU 三尺度搜尋',
          hybrid_cpu='CPU 粗搜＋CPU 8 執行緒精修', hybrid_metal='CPU 粗搜＋Metal 精修',
          fourier_metal='CPU 粗搜＋Fourier＋Metal')
CASE_NAMES = dict(src7='六孔轉盤', src3='電路板', src5_45='旋轉零件', controlled_scale='真實紋理受控尺度圖')
ORDER = ['cpu_fixed', 'cpu_fast', 'cpu_grid', 'hybrid_cpu', 'hybrid_metal', 'fourier_metal']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'cpp-core/docs/real-image-experiments.json')
    parser.add_argument('--rgb', type=Path, default=ROOT/'cpp-core/docs/rgb-batch-benchmark.json')
    parser.add_argument('--output', type=Path, default=ROOT/'cpp-core/docs/REAL_IMAGE_RESULTS_zh-TW.md')
    args = parser.parse_args()
    data, rgb = json.loads(args.input.read_text()), json.loads(args.rgb.read_text())
    lookup = {(e['job']['case']['name'], e['job']['method']): e for e in data['results']}
    assets = args.output.parent/'real-image-experiments'
    assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.facecolor': '#f5f7fa'})

    def draw_panel(ax, entry, crop=None, dense=False):
        case, result = entry['job']['case'], entry['result']
        image = plt.imread(case['source'])
        ax.imshow(image, cmap='gray', vmin=0 if image.dtype.kind == 'f' else None,
                  vmax=1 if image.dtype.kind == 'f' else None)
        matches = result['runs'][-1]['matches']
        accuracy = result['accuracy']
        pass_by_index = {r['result_index']: r['pose_pass'] for r in accuracy['details'] if r['localized']} if accuracy else {}
        for truth in case.get('truth', []):
            ax.add_patch(Polygon(truth['corners'], fill=False, edgecolor='#41d9ff', linewidth=1.3, linestyle='--'))
            ax.text(truth['pos_x'], truth['pos_y']-92, f"T{truth['index']+1}", color='#06384b',
                    fontsize=9, ha='center', weight='bold', bbox=dict(facecolor='#baf0ff', alpha=.95, edgecolor='none', pad=2))
        for match in matches:
            color = ('#0cde80' if pass_by_index.get(match['index'], False) else '#ff8b21') if accuracy else '#0cde80'
            ax.add_patch(Polygon(match['corners'], fill=False, edgecolor=color, linewidth=1.35))
            ax.plot(match['pos_x'], match['pos_y'], '+', color=color, markersize=5)
            if dense:
                label = str(match['index']+1)
            elif accuracy:
                label = f"NCC {match['score']:.3f}\n{match['angle']:.1f} deg / {match['scale']:.3f}x"
            else:
                label = f"#{match['index']+1}  {match['score']:.3f}\n{match['angle']:.2f} deg"
            ax.text(match['pos_x'], match['pos_y'], label, fontsize=6.5 if dense else 8, color='white',
                    ha='center', va='center', bbox=dict(facecolor='#111b25', alpha=.82, edgecolor='none', pad=2))
        title = f"{NAMES[entry['job']['method']]}\n{result['algorithm']['median_ms']:.1f} ms  |  {len(matches)} detections"
        if accuracy:
            title += f"  |  pose pass {accuracy['pose_tp']}/6"
        ax.set_title(title, loc='left', fontsize=12, weight='bold', pad=10, color='#173047')
        if crop:
            ax.set_xlim(crop[:2]); ax.set_ylim(crop[3], crop[2])
        ax.axis('off')

    configs = [
        ('src7', ['cpu_fixed', 'cpu_fast', 'hybrid_metal', 'fourier_metal'], (2, 2), (15, 12.4),
         'Real rotor photograph / full-image search',
         'All candidates found online. NCC labels are similarity scores, not calibrated pose accuracy.'),
        ('src3', ['cpu_fixed', 'hybrid_metal'], (1, 2), (14, 6.5),
         'Real circuit-board photograph / full-image search',
         'Both methods return 36 detections at NCC >= 0.8. Labels show detection IDs.'),
        ('src5_45', ['cpu_fixed', 'hybrid_metal', 'fourier_metal'], (1, 3), (15, 5.7),
         'Real rotated component / full-image search, cropped display',
         'Search used the entire 2592 x 1944 image. Only this display is cropped around the detection.'),
        ('controlled_scale', ['cpu_fixed', 'cpu_grid', 'hybrid_metal', 'fourier_metal'], (2, 2), (16, 11.7),
         'Controlled composite / photographed texture, 6 known rotations and scales',
         'Cyan dashed: ground truth. Green: pose pass. Orange: detected but pose fails. Limits: 2 px / 1 deg / 2% scale.'),
    ]
    for name, methods, shape, size, heading, footnote in configs:
        fig, axes = plt.subplots(*shape, figsize=size, squeeze=False)
        fig.suptitle(heading, x=.035, y=.975, ha='left', fontsize=18, weight='bold', color='#173047')
        crop = None
        if name == 'src5_45':
            match = lookup[name, 'cpu_fixed']['result']['runs'][-1]['matches'][0]
            x, y = match['pos_x'], match['pos_y']
            crop = (x-250, x+250, y-220, y+220)
        for ax, method in zip(axes.flat, methods):
            draw_panel(ax, lookup[name, method], crop=crop, dense=name == 'src3')
        fig.text(.035, .02, footnote, fontsize=10, color='#34485c')
        fig.subplots_adjust(left=.035, right=.975, bottom=.065, top=.90, hspace=.19, wspace=.08)
        fig.savefig(assets/f'{name}-comparison.png', dpi=125)
        plt.close(fig)

    get = lambda c, m: lookup[c, m]['result']
    med = lambda c, m: get(c, m)['algorithm']['median_ms']
    lines = ['# 實圖完整搜尋與 RGB 批次比較', '',
             '日期：2026-10-12。Apple M4 Pro（12 核 CPU、16 核 GPU、48 GiB）；Release、NEON CPU／Metal。', '',
             '本次從整張圖重新產生候選，沒有讀入既有偵測座標或把真值交給搜尋。傳統基準是**目前可攜 C++ 金字塔 NCC**，不是重新執行 Windows MFC 原作者程式。', '',
             '每種組合用獨立程序序列執行，固定亂數打散順序；先跑一次，再暖機至少 200 ms，量測 5 次取中位數。時間包括全圖搜尋、精修、CPU／GPU 提交等待與最後抑制重疊；不含解碼、模板學習、Metal 初始化、畫框和 GUI。輸入已駐留，初始化另列。不是攝影機到 GUI 的端到端延遲。', '',
             '## 方法與比較邊界', '',
             '| 方法 | 實際流程 |', '|---|---|',
             '| 原有 CPU 完整搜尋 | 原有 SIMD 金字塔粗到精，搜尋至原解析度；尺度固定 1；關閉既有 subpixel 選項 |',
             '| 原有 CPU FastMode | 原有搜尋停在第 1 層；無原解析度精修。受控尺度圖套用相同三尺度模板庫 |',
             '| 原有 CPU 三尺度搜尋 | 尺度 0.75、1、1.25 各跑完整搜尋，再合併結果 |',
             '| CPU 粗搜＋CPU／Metal 精修 | 相同 FastMode 候選；6 輪局部位置／角度／log-scale 網格精修；各輪步長減半 |',
             '| CPU 粗搜＋Fourier＋Metal | 同樣先完整執行粗搜，對每個候選做局部 Fourier–Mellin 提案，NCC 優於原候選才採用，再精修 |', '',
             '原始實圖全部固定尺度；受控圖才開啟三尺度提案與連續尺度精修。這個多尺度搜尋是新的**研究腳本**，正式 GUI 尚未接入。搜尋／評分核心為 C++／Metal，排程、FFT、報表為 Python；目前還不是全 C++ 的 Fourier 搜尋器。', '',
             'Fourier 使用 CPU NumPy FFT、頻譜幅值 log-polar 與 phase correlation；此版本仍保留 CPU 角度枚舉，因此是在量測「額外提案」的收益，不能把它當成已消除角度／尺度枚舉的完整 Fourier 偵測器。', '',
             '原有 NCC 使用 uint8 旋轉影像；新精修使用浮點反向取樣。兩者分數的數值契約不完全相同，分數增加不能直接當成精度提高。公平隔離 GPU 的比較是「CPU 粗搜＋CPU 8 精修」對「CPU 粗搜＋Metal 精修」；FastMode 則顯示省略細層搜尋的成本。', '',
             '## 原始實圖：時間與找到的數量', '',
             '這些檔案內容都是灰階；部分容器雖有三通道，三通道數值相同。本次沒有把它們冒充真正的 RGB 實圖。原始照片沒有校準姿態真值；以下數量／一致性不能當成準確率。', '',
             '| 圖片 | 方法 | 演算法 ms | 數量 | CPU 搜尋 ms | FFT 提案 ms | 精修 ms |',
             '|---|---|---:|---:|---:|---:|---:|']
    for case in ['src7', 'src3', 'src5_45']:
        for method in ORDER:
            if (case, method) not in lookup:
                continue
            r = get(case, method)
            stage = lambda k: statistics.median(t[k] for t in r['runs'])
            lines.append(f"| {CASE_NAMES[case]} | {ZH[method]} | {r['algorithm']['median_ms']:.3f} | {len(r['runs'][-1]['matches'])} | {stage('cpu_search_ms'):.3f} | {stage('fft_ms'):.3f} | {stage('refine_ms'):.3f} |")
    lines += ['', '各階段分別取中位數，故相加不必恰好等於總時間中位數。', '',
              f"六孔圖 Metal 混合流程比原有完整搜尋快 **{med('src7','cpu_fixed')/med('src7','hybrid_metal'):.2f}×**；相同新精修流程換成 Metal，比 CPU 8 執行緒快 **{med('src7','hybrid_cpu')/med('src7','hybrid_metal'):.2f}×**。原有 FastMode 本身更快，但省略了新流程做的原解析度精修，不能混為相同精度的結論。", '',
              '![六孔轉盤完整搜尋結果](real-image-experiments/src7-comparison.png)', '',
              '![電路板結果](real-image-experiments/src3-comparison.png)', '',
              '![旋轉零件結果，僅顯示區域裁切](real-image-experiments/src5_45-comparison.png)', '',
              '| 實圖 | Metal 混合對 CPU 完整搜尋：配對數 | 最大中心差 px | 最大角度差 ° |', '|---|---:|---:|---:|']
    for case in ['src7', 'src3', 'src5_45']:
        a = get(case, 'hybrid_metal')['reference_agreement']
        lines.append(f"| {CASE_NAMES[case]} | {a['paired_within_20px']}/{a['reference_count']} | {max(p['position_px'] for p in a['pairs']):.4f} | {max(p['angle_deg'] for p in a['pairs']):.4f} |")
    lines += ['', '相同新流程的 CPU／Metal 輸出數量一致，角度與尺度一致。Src7 有兩組最後一格的中心選擇差異，最大 0.0625 px；其餘實圖與受控圖中心相同。兩組位置交给同一 CPU scorer 重新評分，確認為接近同分的候選；精確差值保留於 `real-image-validation.json`。這種局部最大值選擇差異與上述對原有演算法的差異應分開看。', '',
              '## 已知旋轉與尺度：保留失敗結果', '',
              '使用 Dst3.bmp 的真實電容紋理，透過獨立的 OpenCV 仿射轉換建立六個物件，加入 σ=1 灰階雜訊。這張是受控合成圖，不能代表真實物件的光照、透視或尺度改變。OpenCV 只用於解碼與產生獨立真值圖，搜尋本身沒有使用 OpenCV 比對。', '',
              '六個尺度依序為 0.75、0.90、1.00、1.10、1.25、0.85；角度 −35°、12°、83°、143°、−102°、46°。所有方法用相同最終 NCC ≥0.8；提案門檻 0.5。姿態通過需同時滿足中心 ≤2 px、角度 ≤1°、相對尺度誤差 ≤2%。', '',
              '| 方法 | ms | 找到 / 6 | 姿態通過 / 6 | 姿態 FP / FN |', '|---|---:|---:|---:|---:|']
    for method in ORDER:
        r = get('controlled_scale', method)
        a = r['accuracy']
        lines.append(f"| {ZH[method]} | {r['algorithm']['median_ms']:.3f} | {a['localized']} | {a['pose_tp']} | {a['pose_fp']} / {a['pose_fn']} |")
    lines += ['', '這裡的 FP 包含「位置接近真正物件，但角度／尺度尚未達標」；不代表每個 FP 都是背景誤檢。姿態 FP/FN 是完整姿態標準，不等於只看物件是否存在的 precision/recall。', '',
              '![受控旋轉尺度結果](real-image-experiments/controlled_scale-comparison.png)', '',
              '| 物件／真值 | Metal 混合：位置 px / 角度 ° / 尺度 % | Fourier 混合：位置 px / 角度 ° / 尺度 % |', '|---|---:|---:|']
    for a, b in zip(get('controlled_scale','hybrid_metal')['accuracy']['details'], get('controlled_scale','fourier_metal')['accuracy']['details']):
        fmt = lambda r: f"{r['errors']['position_px']:.3f} / {r['errors']['angle_deg']:.3f} / {r['errors']['relative_scale']*100:.3f}" if r['localized'] else '未找到'
        t = a['truth']
        lines.append(f"| T{t['index']+1}：{t['angle']}°、{t['scale']:.2f}× | {fmt(a)} | {fmt(b)} |")
    lines += ['', '六輪角度步長為 1°、0.5°、…，從單一種子可累積移動不足 2°；尺度不匹配時粗搜產生的角度偏差可能超出這個範圍。這是目前局部搜尋的可達範圍限制，不是 GPU 浮點誤差的證據。FFT 有時提供較好的起點，但仍有失敗；這批結果不足以宣稱多尺度功能已完成。', '',
              '## RGB：三通道不保證 GPU 較快', '',
              '以下另外使用三個不同的解析色彩通道；相同姿態、相同 RGB NCC，資料已駐留。這是評分核心的工作量曲線，不含全圖搜尋，也不是上述灰階實圖改成三份複製。', '',
              '| RGB 模板／姿態數 | CPU 1 ms | CPU 8 ms | Metal wall ms | CPU 8 / Metal |', '|---|---:|---:|---:|---:|']
    for name in ['64_rgb_1', '64_rgb_49', '64_rgb_512', '128_rgb_512']:
        rows = [e for e in rgb['runs'] if e['job']['case']['name'] == name]
        val = lambda backend, threads: next(e['result']['wall']['median_ms'] for e in rows if e['job']['backend'] == backend and e['job']['threads'] == threads)
        c1, c8, gpu = val('cpu',1), val('cpu',8), val('metal',1)
        lines.append(f'| {name} | {c1:.6f} | {c8:.6f} | {gpu:.6f} | {c8/gpu:.2f}× |')
    lines += ['', f"所有 RGB CPU／Metal 分數最大差 {max(r['max_score_error'] for r in rgb['consistency']):.3g}，最佳姿態索引相同。", '',
              '通道數只是工作量的一部分。近似模型為 CPU 時間 ≈ a·N·模板面積·C，GPU 時間 ≈ 提交／同步固定成本 + b·N·模板面積·C；實際還受記憶體、重用、SIMD、執行緒與批次配置影響。C=3 不會消除 GPU 的固定成本，不能直接推出 GPU 一定更快，也不會必然恰好耗時三倍。', '',
              '## 初始化與記憶體', '',
              'RSS 是獨立 Python 程序的高水位，包含解碼器、NumPy、C++ 暫存、FFT、驅動與初始化，不是純演算法配置量。明列的 host／Metal buffer 屬於精修 scorer，沒有涵蓋 CPU 粗搜或 FFT 所有暫存；shared Metal buffer 與系統 RAM 重疊，不能和 RSS 相加。', '',
              '| 圖片／方法 | 初始化 ms | 首次演算法 ms | 峰值 RSS MiB | scorer host MiB | scorer Metal MiB |', '|---|---:|---:|---:|---:|---:|']
    for case in ['src7', 'src3', 'src5_45', 'controlled_scale']:
        for method in ORDER:
            if (case, method) not in lookup:
                continue
            r = get(case, method)
            b = r['memory']['pose_scorer_buffers'] or {}
            lines.append(f"| {CASE_NAMES[case]}／{ZH[method]} | {r['setup_ms']:.2f} | {r['first_call']['algorithm_ms']:.2f} | {r['memory']['process_peak_bytes']/2**20:.2f} | {b.get('host_buffer_bytes',0)/2**20:.2f} | {b.get('gpu_buffer_bytes',0)/2**20:.2f} |")
    lines += ['', '## 判斷與下一步', '',
              '大型模板的批次 NCC 精修值得保留；小圖、單一候選、或原本已很快的搜尋不一定受益。要以「整個工作量的實測」選 CPU／GPU，而不是只看有沒有 RGB。', '',
              '目前 CPU 粗搜仍占混合流程大量時間；Fourier 提案沒有刪除原粗搜，實圖上的額外 FFT 成本大於時間收益。因此此版應定位成可重現研究原型，尚不適合直接取代正式比對器。', '',
              '優先改善方向是讓粗搜的角度／尺度不確定度決定精修範圍，並保留多個競爭種子；接著減少逐輪同步、改善 CPU SIMD／多執行緒與 GPU 取樣重用。若要靠 Fourier 真正加速，需要證明它能在可靠召回率下替代一部分粗搜尋，而非持續疊加提案。', '',
              '尚未實作：正式 GUI 多尺度與 RGB 全圖搜尋、CUDA 實機後端、GPU FFT、自動 CPU／GPU 路由。已有實作：C++／Metal 三通道浮點 NCC、平移／旋轉／尺度姿態評分、本報告中的全圖灰階研究排程。', '',
              '## 重現與原始紀錄', '',
              '依 `EXPERIMENTS_zh-TW.md` 編譯實驗庫，再用同一 Python 環境執行（NumPy、OpenCV；報表另需 Matplotlib）：', '',
              '```sh', 'python cpp-core/tools/real_image_experiments.py',
              'python cpp-core/tools/benchmark_rgb_batches.py',
              'python cpp-core/tools/validate_real_image_experiments.py',
              'python cpp-core/tools/report_real_image_experiments.py', '```', '',
              '- [完整實圖 JSON](real-image-experiments.json)：每次時間、全部偵測框／分數／姿態、分階段成本、真值、參數、來源與二進位雜湊。',
              '- [RGB JSON](rgb-batch-benchmark.json)：全部 CPU／GPU 分數與一致性驗證。',
              '- [實圖驗證 JSON](real-image-validation.json)：來源雜湊、六組 CLI 基準一致性、CPU／Metal 最後一格差異、幾何抑制重疊與灰階內容檢查。',
              '- [先前 ROI 實驗](EXPERIMENT_RESULTS_zh-TW.md)：其時間邊界較窄，不能與本次全圖延遲直接混用。', '',
              '本次多尺度抑制重疊使用交集／較小框面積；同尺度時與原本面積分母等價。它不是 IoU。原圖參數：Src7 分數 0.5、重疊 0.2、角度 ±180°；Src3 分數 0.8、重疊 0、角度 0°；Src5-45 分數 0.8、重疊 0、角度 ±180°。所有模板金字塔最小面積 256。', '']
    args.output.write_text('\n'.join(lines))
    print(args.output)


if __name__ == '__main__':
    main()
