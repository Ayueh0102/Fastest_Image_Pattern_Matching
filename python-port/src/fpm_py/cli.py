"""CLI 介面

提供命令列執行影像樣板比對流程，輸出與原專案類似之欄位。
"""

from __future__ import annotations

import argparse
import json

import cv2

from .pyramid import pyramid_match


def main() -> None:
    p = argparse.ArgumentParser(
        description="Python 版影像樣板比對（NCC/金字塔/角度搜尋）"
    )
    p.add_argument("--src", type=str, required=True, help="來源影像路徑")
    p.add_argument("--tmpl", type=str, required=True, help="樣板影像路徑")
    p.add_argument("--target-num", type=int, default=1, help="預期最大目標數量")
    p.add_argument("--overlap", type=float, default=0.5, help="最大重疊比（NMS）")
    p.add_argument("--score", type=float, default=0.5, help="最小分數門檻")
    p.add_argument(
        "--tolerance-angle",
        type=float,
        default=0.0,
        help="角度容忍度（±deg），0 表示不搜尋旋轉",
    )
    # 向後相容：保留但標記為 deprecated
    p.add_argument(
        "--coarse-step",
        type=float,
        default=15.0,
        help="(已棄用) 角度粗步進，現使用自適應步進",
    )
    p.add_argument(
        "--fine-step",
        type=float,
        default=5.0,
        help="(已棄用) 角度細步進，現使用自適應步進",
    )
    p.add_argument(
        "--method",
        type=str,
        default="ccoeff_normed",
        choices=["ccorr_normed", "ccoeff_normed"],
        help="NCC 變種（預設 ccoeff_normed）",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="opencv",
        choices=["opencv", "fft"],
        help="NCC 後端：OpenCV 或 FFT",
    )
    p.add_argument(
        "--no-subpixel",
        action="store_true",
        help="停用亞像素精化",
    )
    p.add_argument(
        "--min-area",
        type=int,
        default=256,
        help="金字塔頂層最小面積",
    )
    # 相容 macOS CLI 的 --no-display 和 --angle
    p.add_argument("--no-display", action="store_true", help="不顯示結果圖片")
    p.add_argument(
        "--angle",
        type=float,
        default=None,
        help="角度容忍度（--tolerance-angle 的別名）",
    )

    args = p.parse_args()

    src = cv2.imread(args.src, cv2.IMREAD_GRAYSCALE)
    tmpl = cv2.imread(args.tmpl, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise SystemExit(f"讀取來源影像失敗：{args.src}")
    if tmpl is None:
        raise SystemExit(f"讀取樣板影像失敗：{args.tmpl}")

    # --angle 作為 --tolerance-angle 的別名
    tol_angle = args.tolerance_angle
    if args.angle is not None:
        tol_angle = args.angle

    results = pyramid_match(
        src,
        tmpl,
        target_num=args.target_num,
        score=args.score,
        max_overlap=args.overlap,
        tolerance_angle=tol_angle,
        min_reduced_area=args.min_area,
        method=args.method,
        backend=args.backend,
        subpixel=not args.no_subpixel,
        # 向後相容
        coarse_step=args.coarse_step,
        fine_step=args.fine_step,
    )

    print(f"Matches: {len(results)}")
    for r in results:
        print(
            f"  #{r['index']}: Score={r['score']:.4f} "
            f"Angle={r['angle']:.2f}° "
            f"Pos=({r['pos_x']:.1f}, {r['pos_y']:.1f})"
        )

    if not args.no_display:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
