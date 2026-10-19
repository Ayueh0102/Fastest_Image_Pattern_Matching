"""Render measured data without hand-copying timing/accuracy tables."""
import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
MIB = 1024**2


def render(report):
    lines = ["# CPU／Metal 與 Fourier–NCC 實驗結果", "",
             f"日期：{report['date']}。分支：`{report['branch']}`。本表由原始 JSON 自動產生。", "",
             f"平台：{report.get('cpu', report['machine'])}，{report['logical_cpus']} 個邏輯 CPU，"
             f"{(report.get('physical_memory_bytes') or 0)/1024**3:.0f} GiB 記憶體；{report['platform']}；NumPy {report['numpy']}。", "",
             f"每個案例／變體使用新程序，依固定亂數順序序列執行；先暖機 {report['protocol']['warmups']} 次，"
             f"再持續暖機至少 {report['protocol'].get('additional_warmup_ms', 0):g} ms，量測 {report['protocol']['repeats']} 次取中位數。"
             "原始 JSON 保留每次時間、首次呼叫、初始化成本、完整姿態、分數及來源／二進位雜湊。", "",
             "這是浮點姿態評分與**已給定 ROI 的配準實驗**。Src7 使用既有六個候選的位置建立局部姿態批次；"
             "時間不含全圖候選搜尋、NMS、讀圖或 GUI。CPU 基準是這次相同數學契約的 C++ 實驗實作，並非正式 NEON／舊 OpenCV 的替代效能宣稱。", "",
             "## 同一批姿態：CPU 與 GPU", "",
             "wall time 包含姿態轉換、緩衝區增長、提交、等待與結果複製；source/template 已常駐。初始化和讀圖另計。", "",
             "| 模板／工作量 | CPU 1 執行緒 ms | CPU 8 執行緒 ms | Metal wall ms | GPU command ms | CPU 8 / Metal |",
             "|---|---:|---:|---:|---:|---:|"]
    micros = {}
    for entry in report['runs']:
        job, result = entry['job'], entry['result']
        if job['suite'] == 'micro':
            micros.setdefault(job['case']['name'], {})[(job['backend'], job['threads'])] = result
    for name, variants in sorted(micros.items()):
        cpu1 = variants.get(('cpu', 1)); cpu8 = next((v for (b, t), v in variants.items() if b == 'cpu' and t != 1), None)
        gpu = variants.get(('metal', 1))
        if cpu1 and cpu8 and gpu:
            a, b, c = [v['wall']['median_ms'] for v in (cpu1, cpu8, gpu)]
            lines.append(f"| {name} | {a:.3f} | {b:.3f} | {c:.3f} | {gpu['gpu_command']['median_ms']:.3f} | {b/c:.2f}× |")
    maximum = max((r['max_error'] for r in report.get('micro_consistency', [])), default=0)
    lines += ["", f"以上所有 CPU／GPU 批次的最大分數差：**{maximum:.3g}**，最佳姿態索引相同。"
              "完整獨立雙精度參考驗證另見 `pose-experiment-validation.json`。", "",
              "## 記憶體與初始化", "",
              "MiB = 2²⁰ bytes。RSS 是新程序的整體高水位，含 Python、NumPy、解碼器和驅動初始化；**不是純核心記憶體**。"
              "Host／Metal buffer 是核心持有的容量，未包含 allocator、執行緒堆疊或 FFT 暫存。"
              "Metal 使用 shared memory，不能把 GPU 欄與 RSS 相加；currentAllocatedSize 是當下資源配置量，並非物理 VRAM 峰值。", "",
              "| 工作量／後端 | 初始化 ms | 首次呼叫 ms | 程序峰值 RSS MiB | Host buffer MiB | Metal buffer MiB | Metal allocated MiB |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ('128_gray_4096', '128_rgb_512', 'src7_local_162'):
        for (backend, threads), result in sorted(micros.get(name, {}).items()):
            m = result['memory']
            lines.append(f"| {name} / {backend} {threads} | {result['setup_ms']:.3f} | {result['first_call']['wall_ms']:.3f} | "
                         f"{m['process_peak_bytes']/MIB:.2f} | {m['host_buffer_bytes']/MIB:.2f} | {m['gpu_buffer_bytes']/MIB:.2f} | {m['gpu_allocated_bytes']/MIB:.2f} |")
    lines += ["", "## 搜尋策略對照", "",
              "完整支撐組使用 256×256 source/template。每個顏色組有 4 個已知真值案例，包含旋轉 0°、31°、−47°、143°，尺度 0.8、1.0、1.1、1.2。"
              "平移先驗 ±12 px、角度全周、尺度 [0.75, 1.25]；所有方法共用最後七輪 4D NCC 局部網格精修。", "",
              "Grid 用半解析度的 5,880 個位置／角度／尺度組合提出候選；Fourier 用 CPU NumPy FFT、log-polar 和 phase correlation 提出兩個 180° 分支，"
              "再用 NCC 選擇。Guarded 在候選越界或初始 NCC < 0.85 時回到 Grid。精修使用 7×81 個候選，最後回報實際評分過的姿態。", "",
              "通過標準固定為 NCC ≥ 0.8、位置誤差 ≤1 px、角度 ≤1°、尺度相對誤差 ≤1%。下表是各案例中位數再取中位數；"
              "不能解讀為完整工業資料集的成功率。", "",
              "| 方法 | 後端 | 灰階 ms / 通過 | RGB ms / 通過 | 8 案例最高 RSS MiB |",
              "|---|---|---:|---:|---:|"]
    registrations = [r for r in report['runs'] if r['job']['suite'] == 'registration']
    for method in ('grid', 'fourier', 'fourier_guarded'):
        for backend in ('cpu', 'metal'):
            rs = [r for r in registrations if r['job']['method'] == method and r['job']['backend'] == backend and r['job']['case']['name'].startswith('clean')]
            if not rs:
                continue
            values = []
            for group in ('clean_gray', 'clean_rgb'):
                selected = [r['result'] for r in rs if r['job']['case']['name'].startswith(group)]
                values.append(f"{statistics.median(r['algorithm']['median_ms'] for r in selected):.3f} / {sum(r['accuracy']['tp'] for r in selected)}/{len(selected)}")
            lines.append(f"| {method} | {'CPU 8' if backend == 'cpu' else 'Metal'} | {' | '.join(values)} | {max(r['result']['memory']['process_peak_bytes'] for r in rs)/MIB:.1f} |")
    clean = [r['result'] for r in registrations if r['job']['method'] == 'fourier' and r['job']['case']['name'].startswith('clean')]
    errors = [e for r in clean for e in r['accuracy']['errors_to_each_truth']]
    if errors:
        lines += ["", f"Fourier＋精修的 8 組完整支撐案例（兩後端）最大誤差：位置 {max(e['position_px'] for e in errors):.4f} px、"
                  f"角度 {max(e['angle_deg'] for e in errors):.4f}°、尺度 {100*max(e['relative_scale'] for e in errors):.4f}%。"]
    lines += ["", "## 保留失敗，不用分數代替準確度", "",
              "裁切組把 template 縮為 128×128 再補零到 256×256，source 不變；物件尾端被裁切，頻譜邊界因此不相同。"
              "壓力組含背景雜訊、遮擋、兩個重疊姿態和一個純雜訊負例。兩物件組超出單 ROI 單姿態模型的表達能力，仍列出 FN。", "",
              "| 資料組 | 方法 | 後端 | TP / FP / FN | 回退次數 |",
              "|---|---|---|---:|---:|"]
    for group in ('cropped', 'stress'):
        for method in ('grid', 'fourier', 'fourier_guarded'):
            for backend in ('cpu', 'metal'):
                rs = [r['result'] for r in registrations if r['job']['method'] == method and r['job']['backend'] == backend and r['job']['case']['name'].startswith(group)]
                if rs:
                    counts = ' / '.join(str(sum(r['accuracy'][k] for r in rs)) for k in ('tp', 'fp', 'fn'))
                    lines.append(f"| {group} | {method} | {backend} | {counts} | {sum(r['runs'][-1]['fallback'] for r in rs)} |")
    lines += ["", "FP 包括高於分數門檻、但姿態超出上述精度容許的輸出。這比只檢查『有找到』嚴格。"
              "本次 Grid 和 Guarded 在背景組仍有尺度偏差、在雙物件組仍有位置偏差；回退不是準確度保證。", "",
              "## 本次結論與適用範圍", "",
              "1. 合併取樣＋NCC 的 GPU 批次值得繼續；大型工作量有收益，小批次須考慮提交／同步成本。尚未實作自動 CPU／GPU 分派。",
              "2. Fourier 在符合整體相似變換假設的資料上可用；目前 CPU FFT 提案成本及暫存明顯，在這批小 ROI 的 Metal 管線中，直接批次網格搜尋更快。",
              "3. 模板裁切、背景、遮擋會破壞假設。後續應先改善 ROI、mask、窗函數與候選保留策略，再研究 GPU FFT，不能直接刪除回退搜尋。",
              "4. 本次尚未將研究後端接進正式 matcher／GUI；沒有全圖多尺度召回率、CUDA 實機結果或通用最快宣稱。CPU 尚可加入 SIMD／更好的 tile 重用；GPU 的相鄰姿態取樣也尚未共用。", "",
              "方法、數學契約、完整源碼位置與重跑指令見 [實驗說明](EXPERIMENTS_zh-TW.md)。原始資料見 [benchmark JSON](pose-experiment-benchmark.json)。", ""]
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'cpp-core/docs/pose-experiment-benchmark.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'cpp-core/docs/EXPERIMENT_RESULTS_zh-TW.md')
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.input.read_text())))
    print(args.output)
