"""快速圖像樣板匹配 GUI

提供直觀的圖形介面進行圖像樣板匹配，繪圖邏輯對齊 C++ MFC 版本：
- 紅白虛線旋轉矩形框
- 綠色 L 形角落標記
- 綠色十字中心標記
- 綠色索引號文字
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .pyramid import pyramid_match


# ── C++ 繪圖常數（對應 MatchToolDlg.cpp） ─────────────────────────────────
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLACK = (0, 0, 0)
D2R = math.pi / 180.0


# ── C++ 繪圖函式 ──────────────────────────────────────────────────────────

def _draw_dash_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color1: tuple[int, int, int] = COLOR_RED,
    color2: tuple[int, int, int] = COLOR_WHITE,
    period: int = 3,
) -> None:
    """繪製紅白虛線，對應 C++ DrawDashLine。

    每 period 個像素中，第 1 個為 color2（白），其餘為 color1（紅）。
    """
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    dist = max(1, int(math.sqrt(dx * dx + dy * dy)))
    for i in range(dist):
        t = i / dist
        x = int(x1 + dx * t)
        y = int(y1 + dy * t)
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            c = color2 if (i % period == 0) else color1
            img[y, x] = c


def _draw_mark_cross(
    img: np.ndarray,
    cx: int,
    cy: int,
    length: int = 5,
    color: tuple[int, int, int] = COLOR_GREEN,
    thickness: int = 1,
) -> None:
    """繪製十字中心標記，對應 C++ DrawMarkCross。"""
    cv2.line(img, (cx - length, cy), (cx + length, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, cy - length), (cx, cy + length), color, thickness, cv2.LINE_AA)


def _draw_match_result(
    img: np.ndarray,
    result: dict,
    tmpl_h: int,
    tmpl_w: int,
    index: int,
    selected: bool = False,
) -> None:
    """繪製單一匹配結果，對齊 C++ MFC 繪圖邏輯。

    包含：旋轉矩形虛線框 + L 形角標 + 十字中心 + 索引號。
    """
    angle_deg = result.get("angle", 0.0)
    pos_x = result.get("pos_x", 0.0)
    pos_y = result.get("pos_y", 0.0)

    # 從 center 反推 ptLT（對齊 pyramid.py:625-639 的方向向量）
    # pyramid.py: dRAngle = output_angle * D2R
    #   u (右方向) = (cos(dRA), -sin(dRA))
    #   v (下方向) = (sin(dRA),  cos(dRA))
    dRAngle = angle_deg * D2R
    cos_a = math.cos(dRAngle)
    sin_a = math.sin(dRAngle)

    half_w = tmpl_w / 2.0
    half_h = tmpl_h / 2.0
    # center = ptLT + (w/2)*u + (h/2)*v → ptLT = center - (w/2)*u - (h/2)*v
    ptLT = (
        pos_x - half_w * cos_a - half_h * sin_a,
        pos_y + half_w * sin_a - half_h * cos_a,
    )
    # ptRT = ptLT + w*u,  ptLB = ptLT + h*v,  ptRB = ptRT + h*v
    ptRT = (
        ptLT[0] + tmpl_w * cos_a,
        ptLT[1] - tmpl_w * sin_a,
    )
    ptLB = (
        ptLT[0] + tmpl_h * sin_a,
        ptLT[1] + tmpl_h * cos_a,
    )
    ptRB = (
        ptRT[0] + tmpl_h * sin_a,
        ptRT[1] + tmpl_h * cos_a,
    )

    def to_int(pt):
        return (int(round(pt[0])), int(round(pt[1])))

    iLT = to_int(ptLT)
    iRT = to_int(ptRT)
    iRB = to_int(ptRB)
    iLB = to_int(ptLB)
    iC = (int(round(pos_x)), int(round(pos_y)))

    # ── 1. 紅白虛線旋轉矩形 ──
    line_color1 = COLOR_RED
    line_color2 = COLOR_WHITE
    if selected:
        line_color1 = (0, 200, 255)  # 選中時用橙色
    _draw_dash_line(img, iLT, iRT, line_color1, line_color2)
    _draw_dash_line(img, iRT, iRB, line_color1, line_color2)
    _draw_dash_line(img, iRB, iLB, line_color1, line_color2)
    _draw_dash_line(img, iLB, iLT, line_color1, line_color2)

    # ── 2. 綠色 L 形角落標記 ──
    def lerp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    ratio = 1.0 / 6.0  # L 形長度為邊長的 1/6
    # LT 角
    cv2.line(img, iLT, to_int(lerp(ptLT, ptRT, ratio)), COLOR_GREEN, 1, cv2.LINE_AA)
    cv2.line(img, iLT, to_int(lerp(ptLT, ptLB, ratio)), COLOR_GREEN, 1, cv2.LINE_AA)
    # RT 角
    cv2.line(img, iRT, to_int(lerp(ptRT, ptLT, ratio)), COLOR_GREEN, 1, cv2.LINE_AA)
    cv2.line(img, iRT, to_int(lerp(ptRT, ptRB, ratio)), COLOR_GREEN, 1, cv2.LINE_AA)
    # RB 角
    cv2.line(img, iRB, to_int(lerp(ptRB, ptRT, ratio)), COLOR_GREEN, 1, cv2.LINE_AA)
    cv2.line(img, iRB, to_int(lerp(ptRB, ptLB, ratio)), COLOR_GREEN, 1, cv2.LINE_AA)
    # LB 角
    cv2.line(img, iLB, to_int(lerp(ptLB, ptLT, ratio)), COLOR_GREEN, 1, cv2.LINE_AA)
    cv2.line(img, iLB, to_int(lerp(ptLB, ptRB, ratio)), COLOR_GREEN, 1, cv2.LINE_AA)

    # ── 3. 綠色十字中心標記 ──
    _draw_mark_cross(img, iC[0], iC[1], 5, COLOR_GREEN, 1)

    # ── 4. 索引號（上邊中點，綠色，較大字體） ──
    mid_top = to_int(lerp(ptLT, ptRT, 0.5))
    label = str(index)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    (tw_text, th_text), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.putText(
        img, label, (mid_top[0] - tw_text // 2, mid_top[1] - 8),
        font, font_scale, COLOR_GREEN, thickness, cv2.LINE_AA,
    )


# ── GUI 狀態 ──────────────────────────────────────────────────────────────

@dataclass
class AppState:
    src_path: str = ""
    tmpl_path: str = ""
    img_tk: ImageTk.PhotoImage | None = None
    is_processing: bool = False
    results: list[dict] | None = None
    base_bgr: np.ndarray | None = None
    source_bgr: np.ndarray | None = None
    zoom: float = 1.0
    offset_x: int = 0
    offset_y: int = 0
    selected_index: int | None = None
    src_gray: np.ndarray | None = None
    tmpl_gray: np.ndarray | None = None
    render_scale: float = 1.0
    render_origin_x: int = 0
    render_origin_y: int = 0


# ── 主視窗 ────────────────────────────────────────────────────────────────

class FpmGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("快速圖像樣板匹配 - Fast Pattern Matching")
        self.geometry("1200x800")
        self.minsize(900, 600)

        style = ttk.Style()
        try:
            for t in ("aqua", "default", "alt", "clam"):
                if t in style.theme_names():
                    style.theme_use(t)
                    break
        except Exception:
            pass
        self.configure(bg="#f7f7f7")
        style.configure("TFrame", background="#f7f7f7")
        style.configure("TLabelFrame", background="#f7f7f7", foreground="#222")
        style.configure("TLabel", background="#f7f7f7", foreground="#222")
        style.configure("TButton", foreground="#222")

        self.state = AppState()
        self.var_show_boxes = tk.BooleanVar(value=True)
        self.var_show_labels = tk.BooleanVar(value=True)
        self.var_auto_focus = tk.BooleanVar(value=True)
        self.coord_var = tk.StringVar(value="")
        self.result_info_var = tk.StringVar(value="")
        self._drag_start = None

        # 版面：先 pack BOTTOM 狀態列，再 LEFT/RIGHT 主區域
        self._create_status_bar()
        self._create_left_panel()
        self._create_canvas_area()

        self._load_default_images()
        self._preview_source()

    # ── 版面建構 ──────────────────────────────────────────────────────────

    def _create_status_bar(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=2)

        self.progress = ttk.Progressbar(bar, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.status_label = ttk.Label(bar, text="就緒", width=28, anchor="w")
        self.status_label.pack(side=tk.LEFT, padx=(0, 8))

        self.coord_label = ttk.Label(bar, textvariable=self.coord_var, width=18, anchor="e")
        self.coord_label.pack(side=tk.RIGHT)

    def _create_left_panel(self) -> None:
        # 外框：固定寬度
        outer = ttk.Frame(self, width=280)
        outer.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 0), pady=4)
        outer.pack_propagate(False)

        # ── 可捲動區域：Canvas + Scrollbar ──
        self._left_scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL)
        self._left_canvas = tk.Canvas(
            outer, highlightthickness=0, bg="#f7f7f7",
            yscrollcommand=self._left_scrollbar.set,
        )
        self._left_scrollbar.config(command=self._left_canvas.yview)
        self._left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 內部面板（所有控制元件都放在這裡）
        panel = ttk.Frame(self._left_canvas)
        self._left_win_id = self._left_canvas.create_window(
            (0, 0), window=panel, anchor="nw",
        )

        # 內容變化 → 更新捲動範圍
        def _on_panel_configure(_event=None):
            self._left_canvas.configure(scrollregion=self._left_canvas.bbox("all"))
        panel.bind("<Configure>", _on_panel_configure)

        # Canvas 大小變化 → 內部面板寬度跟隨
        def _on_canvas_configure(event):
            self._left_canvas.itemconfig(self._left_win_id, width=event.width)
        self._left_canvas.bind("<Configure>", _on_canvas_configure)

        # ── 捲動支援 ──
        # Tk 9.0 macOS 用 <TouchpadScroll>（TIP 684），舊版用 <MouseWheel>
        # delta 編碼：高 16 位 = deltaX，低 16 位 = deltaY（均為有符號 16 位）
        # 觸控板約 60 fps，需要節流（用 tk::PreciseScrollDeltas 或手動解碼）
        self._left_outer = outer
        self._left_canvas.configure(yscrollincrement=1)  # 啟用像素級捲動
        self._scroll_accum_y = 0.0  # 累積量（用於右側圖像縮放節流）

        def _is_in_left_panel(event) -> bool:
            """檢查事件所在 widget 是否屬於左側面板。"""
            try:
                w = self.winfo_containing(event.x_root, event.y_root)
                while w is not None:
                    if w is outer:
                        return True
                    w = getattr(w, "master", None)
            except (tk.TclError, AttributeError):
                pass
            return False

        def _decode_touchpad_delta(delta: int) -> tuple[float, float]:
            """解碼 TouchpadScroll delta → (dx, dy)。

            依 TIP 684：高 16 位 = deltaX，低 16 位 = deltaY。
            優先使用 tk::PreciseScrollDeltas（更精確）。
            """
            try:
                result = self.tk.call("::tk::PreciseScrollDeltas", delta)
                return float(result[0]), float(result[1])
            except (tk.TclError, ValueError, IndexError):
                pass
            # 手動解碼
            dy = delta & 0xFFFF
            if dy >= 0x8000:
                dy -= 0x10000
            dx = (delta >> 16) & 0xFFFF
            if dx >= 0x8000:
                dx -= 0x10000
            return float(dx), float(dy)

        def _on_touchpad_scroll(event):
            """Tk 9.0 macOS 觸控板捲動。"""
            _dx, dy = _decode_touchpad_delta(event.delta)
            if _is_in_left_panel(event):
                if dy != 0:
                    # 像素級捲動，速度適中（除以 3 降速）
                    pixels = int(-dy / 3)
                    if pixels != 0:
                        self._left_canvas.yview_scroll(pixels, "units")
                return "break"
            return None

        def _on_mousewheel(event):
            """傳統滑鼠滾輪（Tk 8.x / 非 macOS）。"""
            if _is_in_left_panel(event):
                self._left_canvas.yview_scroll(
                    -1 if event.delta > 0 else 1, "units",
                )
                return "break"
            return None

        self.bind_all("<MouseWheel>", _on_mousewheel)
        try:
            self.bind_all("<TouchpadScroll>", _on_touchpad_scroll)
        except tk.TclError:
            pass  # Tk 版本不支援 <TouchpadScroll>

        self.left_panel = panel

        # ── 圖像文件 ──
        grp = ttk.LabelFrame(panel, text="圖像文件", padding=8)
        grp.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(grp, text="來源影像:").grid(row=0, column=0, sticky="w")
        self.ent_src = ttk.Entry(grp, width=22)
        self.ent_src.grid(row=1, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(grp, text="瀏覽", command=self.browse_src, width=6).grid(row=1, column=1)

        ttk.Label(grp, text="樣板影像:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.ent_tmpl = ttk.Entry(grp, width=22)
        self.ent_tmpl.grid(row=3, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(grp, text="瀏覽", command=self.browse_tmpl, width=6).grid(row=3, column=1)
        grp.columnconfigure(0, weight=1)

        # ── 匹配參數 ──
        grp2 = ttk.LabelFrame(panel, text="匹配參數", padding=8)
        grp2.pack(fill=tk.X, pady=(0, 6))

        self.var_target = tk.IntVar(value=5)
        self.var_score = tk.DoubleVar(value=0.75)
        self.var_overlap = tk.DoubleVar(value=0.5)
        self.var_tol = tk.DoubleVar(value=180.0)

        for i, (label, var, hint) in enumerate([
            ("目標數量:", self.var_target, "個"),
            ("最低分數:", self.var_score, "(0-1)"),
            ("最大重疊:", self.var_overlap, "(0-1)"),
            ("角度範圍:", self.var_tol, "±度"),
        ]):
            ttk.Label(grp2, text=label).grid(row=i, column=0, sticky="w", pady=1)
            ttk.Entry(grp2, textvariable=var, width=10).grid(row=i, column=1, padx=4)
            ttk.Label(grp2, text=hint, font=("", 9)).grid(row=i, column=2, sticky="w")

        # ── 執行控制 ──
        grp3 = ttk.LabelFrame(panel, text="執行控制", padding=6)
        grp3.pack(fill=tk.X, pady=(0, 6))

        self.btn_run = ttk.Button(grp3, text="開始匹配", command=self.run_match)
        self.btn_run.pack(fill=tk.X, pady=1)
        ttk.Button(grp3, text="清除結果", command=self.clear_results).pack(fill=tk.X, pady=1)
        ttk.Button(grp3, text="匯出結果", command=self.export_results).pack(fill=tk.X, pady=1)

        # ── 顯示選項 ──
        grp4 = ttk.LabelFrame(panel, text="顯示選項", padding=4)
        grp4.pack(fill=tk.X, pady=(0, 6))

        ttk.Checkbutton(grp4, text="顯示匹配框", variable=self.var_show_boxes,
                        command=self._refresh_overlay).pack(anchor="w")
        ttk.Checkbutton(grp4, text="顯示標籤", variable=self.var_show_labels,
                        command=self._refresh_overlay).pack(anchor="w")
        ttk.Checkbutton(grp4, text="自動聚焦", variable=self.var_auto_focus).pack(anchor="w")

        # ── 結果列表（固定高度 + 自帶 scrollbar） ──
        grp5 = ttk.LabelFrame(panel, text="匹配結果", padding=4)
        grp5.pack(fill=tk.X, pady=(0, 6))

        cols = ("id", "score", "angle", "pos")
        self.tree = ttk.Treeview(grp5, columns=cols, show="headings", height=5)
        self.tree.heading("id", text="#")
        self.tree.heading("score", text="分數")
        self.tree.heading("angle", text="角度")
        self.tree.heading("pos", text="位置")
        self.tree.column("id", width=28, anchor="center")
        self.tree.column("score", width=55, anchor="center")
        self.tree.column("angle", width=55, anchor="center")
        self.tree.column("pos", width=90, anchor="center")

        sb = ttk.Scrollbar(grp5, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_result_select)
        self.tree.bind("<Double-1>", lambda _: self._focus_on_selection())

        # ── 樣板預覽 ──
        grp6 = ttk.LabelFrame(panel, text="樣板預覽", padding=4)
        grp6.pack(fill=tk.X, pady=(6, 0))
        self.tmpl_preview_label = tk.Label(grp6, text="尚未載入", bg="#f0f0f0")
        self.tmpl_preview_label.pack(fill=tk.X)
        self._tmpl_tk: ImageTk.PhotoImage | None = None

    def _create_canvas_area(self) -> None:
        frame = ttk.Frame(self)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 工具列
        toolbar = ttk.Frame(frame)
        toolbar.pack(fill=tk.X, pady=(0, 2))
        ttk.Button(toolbar, text="適合視窗", command=self.view_fit, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="100%", command=self.view_100, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="+", command=self.view_zoom_in, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="-", command=self.view_zoom_out, width=3).pack(side=tk.LEFT, padx=2)

        self.canvas = tk.Canvas(frame, bg="#e0e0e0", relief="sunken", bd=1)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>", lambda _: self._render())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        try:
            self.canvas.bind("<TouchpadScroll>", self._on_touchpad_zoom)
        except tk.TclError:
            pass
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda _: self.coord_var.set(""))

    # ── 圖像載入 ──────────────────────────────────────────────────────────

    def _load_default_images(self) -> None:
        base = os.path.dirname(__file__)
        search = [
            os.path.join(base, "..", "..", "..", "Test Images"),
            os.path.join(base, "..", "..", "Test Images"),
        ]
        src_name, tmpl_name = None, None
        for d in search:
            if not os.path.isdir(d):
                continue
            for s, t in [("Src7.bmp", "Dst7.bmp"), ("Src1.bmp", "Dst1.bmp")]:
                sp = os.path.join(d, s)
                tp = os.path.join(d, t)
                if os.path.exists(sp) and os.path.exists(tp):
                    src_name, tmpl_name = os.path.abspath(sp), os.path.abspath(tp)
                    break
            if src_name:
                break

        if src_name:
            self.ent_src.insert(0, src_name)
            self.state.src_path = src_name
        if tmpl_name:
            self.ent_tmpl.insert(0, tmpl_name)
            self.state.tmpl_path = tmpl_name
            tmpl = cv2.imread(tmpl_name, cv2.IMREAD_GRAYSCALE)
            self._update_tmpl_preview(tmpl)
            if tmpl is not None:
                self.state.tmpl_gray = tmpl

    def browse_src(self) -> None:
        p = filedialog.askopenfilename(
            title="選擇來源影像",
            filetypes=[("圖像", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("所有", "*.*")],
        )
        if p:
            self.ent_src.delete(0, tk.END)
            self.ent_src.insert(0, p)
            self.state.src_path = os.path.abspath(p)
            self.state.source_bgr = None
            self._preview_source()

    def browse_tmpl(self) -> None:
        p = filedialog.askopenfilename(
            title="選擇樣板影像",
            filetypes=[("圖像", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("所有", "*.*")],
        )
        if p:
            self.ent_tmpl.delete(0, tk.END)
            self.ent_tmpl.insert(0, p)
            self.state.tmpl_path = os.path.abspath(p)
            tmpl = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            self._update_tmpl_preview(tmpl)
            self.state.tmpl_gray = tmpl
            self._preview_source()

    def _preview_source(self) -> None:
        p = self.ent_src.get().strip()
        if not p or not os.path.exists(p):
            self._show_placeholder()
            return
        if self.state.source_bgr is None:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                self._show_placeholder()
                return
            self.state.source_bgr = img
        self.state.base_bgr = self.state.source_bgr.copy()
        self.state.results = None
        self.state.zoom = 1.0
        self.state.offset_x = 0
        self.state.offset_y = 0
        self._render()

    def _update_tmpl_preview(self, tmpl: np.ndarray | None) -> None:
        if tmpl is None:
            self.tmpl_preview_label.config(image="", text="尚未載入")
            self._tmpl_tk = None
            return
        rgb = cv2.cvtColor(tmpl, cv2.COLOR_GRAY2RGB) if tmpl.ndim == 2 else tmpl
        pil = Image.fromarray(rgb)
        pil.thumbnail((260, 160), Image.BILINEAR)
        self._tmpl_tk = ImageTk.PhotoImage(pil)
        self.tmpl_preview_label.config(image=self._tmpl_tk, text="")

    # ── 繪圖核心 ──────────────────────────────────────────────────────────

    def _show_placeholder(self) -> None:
        self.canvas.delete("all")
        self.update_idletasks()
        cx = max(50, self.canvas.winfo_width() // 2)
        cy = max(50, self.canvas.winfo_height() // 2)
        self.canvas.create_text(cx, cy, text="請選擇圖像並執行匹配",
                                font=("", 12), fill="gray")

    def _render(self) -> None:
        if self.state.base_bgr is None:
            self._show_placeholder()
            return
        self.update_idletasks()
        cw = max(100, self.canvas.winfo_width() - 4)
        ch = max(100, self.canvas.winfo_height() - 4)
        h, w = self.state.base_bgr.shape[:2]
        fit = min(cw / w, ch / h)
        scale = max(0.05, min(10.0, fit * self.state.zoom))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        rgb = cv2.cvtColor(self.state.base_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
        self.state.img_tk = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        ox = 2 + self.state.offset_x
        oy = 2 + self.state.offset_y
        self.state.render_scale = scale
        self.state.render_origin_x = ox
        self.state.render_origin_y = oy
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.state.img_tk)

    def _refresh_overlay(self) -> None:
        """重新繪製匹配結果疊圖（對齊 C++ 繪圖風格）。"""
        if self.state.source_bgr is None:
            return
        out = self.state.source_bgr.copy()

        if self.state.results and self.state.tmpl_gray is not None:
            th, tw = self.state.tmpl_gray.shape[:2]

            for i, r in enumerate(self.state.results):
                selected = (self.state.selected_index == i)

                if self.var_show_boxes.get():
                    _draw_match_result(out, r, th, tw, i, selected)
                elif selected:
                    cx = int(round(r.get("pos_x", 0)))
                    cy = int(round(r.get("pos_y", 0)))
                    _draw_mark_cross(out, cx, cy, 8, COLOR_GREEN, 2)

                if self.var_show_labels.get() and not self.var_show_boxes.get():
                    cx = int(round(r.get("pos_x", 0)))
                    cy = int(round(r.get("pos_y", 0)))
                    cv2.putText(out, str(i), (cx + 8, cy - 4),
                                cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_GREEN, 1, cv2.LINE_AA)

        self.state.base_bgr = out
        self._render()

    # ── 匹配執行 ──────────────────────────────────────────────────────────

    def run_match(self) -> None:
        if self.state.is_processing:
            return
        src_path = self.ent_src.get().strip()
        tmpl_path = self.ent_tmpl.get().strip()
        if not src_path or not tmpl_path:
            messagebox.showerror("錯誤", "請選擇來源影像和樣板影像")
            return
        if not os.path.exists(src_path) or not os.path.exists(tmpl_path):
            messagebox.showerror("錯誤", "圖像文件不存在")
            return

        src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        tmpl = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
        if src is None or tmpl is None:
            messagebox.showerror("錯誤", "無法讀取圖像")
            return
        if tmpl.shape[0] >= src.shape[0] or tmpl.shape[1] >= src.shape[1]:
            messagebox.showerror("錯誤", "樣板不能大於來源影像")
            return

        self.state.is_processing = True
        self.btn_run.config(state="disabled", text="匹配中...")
        self.progress.start()
        self.status_label.config(text="正在匹配...")

        def worker():
            try:
                print(f"[GUI] Start matching...", flush=True)
                print(f"[GUI] src shape: {src.shape}, tmpl shape: {tmpl.shape}", flush=True)
                results = pyramid_match(
                    src, tmpl,
                    target_num=int(self.var_target.get()),
                    score=float(self.var_score.get()),
                    max_overlap=float(self.var_overlap.get()),
                    tolerance_angle=float(self.var_tol.get()),
                )
                print(f"[GUI] Matching finished. results={len(results)}", flush=True)
                self.after(0, lambda: self._on_match_done(results, src, tmpl, src_path))
            except Exception as exc:
                self.after(0, lambda: self._on_match_error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_match_done(self, results: list, src: np.ndarray, tmpl: np.ndarray,
                       src_path: str) -> None:
        self.state.results = results
        self.state.src_gray = src
        self.state.tmpl_gray = tmpl
        if self.state.source_bgr is None:
            color = cv2.imread(src_path, cv2.IMREAD_COLOR)
            self.state.source_bgr = color if color is not None else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

        self.state.zoom = 1.0
        self.state.offset_x = 0
        self.state.offset_y = 0

        self._populate_tree(results)
        self._update_tmpl_preview(tmpl)

        if results:
            self.state.selected_index = 0
            self.tree.selection_set("0")
            self.tree.focus("0")
        else:
            self.state.selected_index = None

        self._refresh_overlay()

        if results and self.var_auto_focus.get():
            self._focus_on_selection()

        self.status_label.config(text=f"找到 {len(results)} 個匹配")
        self._finish()

    def _on_match_error(self, msg: str) -> None:
        messagebox.showerror("匹配失敗", msg)
        self._finish()

    def _finish(self) -> None:
        self.state.is_processing = False
        self.btn_run.config(state="normal", text="開始匹配")
        self.progress.stop()

    # ── 結果列表 ──────────────────────────────────────────────────────────

    def _populate_tree(self, results: list[dict]) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, r in enumerate(results):
            vals = (i, f"{r['score']:.3f}", f"{r['angle']:.1f}", f"({r['pos_x']:.0f},{r['pos_y']:.0f})")
            self.tree.insert("", tk.END, iid=str(i), values=vals)

    def _on_result_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel or not self.state.results:
            return
        try:
            idx = int(sel[0])
        except (ValueError, TypeError):
            return
        if 0 <= idx < len(self.state.results):
            self.state.selected_index = idx
            self._refresh_overlay()
            if self.var_auto_focus.get():
                self._focus_on_selection()

    def _focus_on_selection(self) -> None:
        if not self.state.results or self.state.selected_index is None:
            return
        r = self.state.results[self.state.selected_index]
        cx, cy = r["pos_x"], r["pos_y"]
        self._render()
        scale = max(self.state.render_scale, 1e-6)
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        self.state.offset_x = int(cw / 2 - 2 - cx * scale)
        self.state.offset_y = int(ch / 2 - 2 - cy * scale)
        self.state.zoom = min(4.0, max(self.state.zoom, 1.5))
        self._render()

    def clear_results(self) -> None:
        self.state.results = None
        self.state.selected_index = None
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._preview_source()
        self.status_label.config(text="已清除")

    def export_results(self) -> None:
        if not self.state.results:
            messagebox.showwarning("警告", "沒有結果可匯出")
            return
        fn = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("所有", "*.*")],
        )
        if fn:
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(self.state.results, f, ensure_ascii=False, indent=2)
            self.status_label.config(text=f"已匯出: {os.path.basename(fn)}")

    # ── 視圖控制 ──────────────────────────────────────────────────────────

    def view_fit(self) -> None:
        self.state.zoom = 1.0
        self.state.offset_x = 0
        self.state.offset_y = 0
        self._render()

    def view_100(self) -> None:
        if self.state.base_bgr is None:
            return
        h, w = self.state.base_bgr.shape[:2]
        cw = max(100, self.canvas.winfo_width())
        fit = min(cw / w, self.canvas.winfo_height() / h)
        self.state.zoom = max(0.1, 1.0 / fit) if fit > 0 else 1.0
        self.state.offset_x = 0
        self.state.offset_y = 0
        self._render()

    def view_zoom_in(self) -> None:
        self.state.zoom = min(10.0, self.state.zoom * 1.25)
        self._render()

    def view_zoom_out(self) -> None:
        self.state.zoom = max(0.05, self.state.zoom / 1.25)
        self._render()

    def _on_wheel(self, event) -> None:
        if event.delta > 0:
            self.view_zoom_in()
        else:
            self.view_zoom_out()

    def _on_touchpad_zoom(self, event) -> None:
        """Tk 9.0 macOS 觸控板：在圖像區域時縮放（帶節流）。"""
        try:
            result = self.tk.call("::tk::PreciseScrollDeltas", event.delta)
            dy = float(result[1])
        except (tk.TclError, ValueError, IndexError):
            dy = float(event.delta & 0xFFFF)
            if dy >= 0x8000:
                dy -= 0x10000
        self._scroll_accum_y += dy
        # 累積超過閾值才觸發縮放（避免過快）
        if self._scroll_accum_y > 30:
            self.view_zoom_in()
            self._scroll_accum_y = 0.0
        elif self._scroll_accum_y < -30:
            self.view_zoom_out()
            self._scroll_accum_y = 0.0

    def _on_press(self, event) -> None:
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event) -> None:
        if not self._drag_start:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        self.state.offset_x += dx
        self.state.offset_y += dy
        self._render()

    def _on_motion(self, event) -> None:
        if self.state.base_bgr is None or self.state.render_scale <= 0:
            self.coord_var.set("")
            return
        x = (event.x - self.state.render_origin_x) / self.state.render_scale
        y = (event.y - self.state.render_origin_y) / self.state.render_scale
        ix, iy = int(round(x)), int(round(y))
        h, w = self.state.base_bgr.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            self.coord_var.set(f"({ix}, {iy})")
        else:
            self.coord_var.set("")


def main() -> None:
    app = FpmGui()
    app.mainloop()


if __name__ == "__main__":
    main()
