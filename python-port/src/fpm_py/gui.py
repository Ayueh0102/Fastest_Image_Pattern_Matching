"""快速圖像樣板匹配 GUI

提供直觀的圖形介面進行圖像樣板匹配：
- 載入來源影像與樣板影像
- 設定匹配參數
- 執行 NCC 金字塔匹配演算法
- 視覺化顯示匹配結果
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .pyramid import pyramid_match


@dataclass
class AppState:
    src_path: str = ""
    tmpl_path: str = ""
    img_tk: ImageTk.PhotoImage | None = None
    is_processing: bool = False
    results: list[dict] | None = None
    base_bgr: np.ndarray | None = None  # 原始顯示影像（含疊圖）
    source_bgr: np.ndarray | None = None  # 原始來源彩色圖像
    zoom: float = 1.0
    offset_x: int = 0
    offset_y: int = 0
    selected_index: int | None = None
    src_gray: np.ndarray | None = None
    tmpl_gray: np.ndarray | None = None
    render_scale: float = 1.0
    render_origin_x: int = 0
    render_origin_y: int = 0
    render_width: int = 0
    render_height: int = 0


class FpmGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("快速圖像樣板匹配 - Fast Pattern Matching")
        self.geometry("1200x800")
        self.minsize(900, 600)
        
        # 設定樣式（優先使用 macOS 友善與淺色主題，避免深色模式黑底黑字）
        style = ttk.Style()
        try:
            preferred = ("aqua", "default", "alt", "vista", "clam", "classic")
            for t in preferred:
                if t in style.theme_names():
                    style.theme_use(t)
                    break
        except Exception:
            pass
        # 明確指定背景/前景，避免深色模式看不到元件
        self.configure(bg="#f7f7f7")
        style.configure("TFrame", background="#f7f7f7")
        style.configure("TLabelFrame", background="#f7f7f7", foreground="#222222")
        style.configure("TLabel", background="#f7f7f7", foreground="#222222")
        style.configure("TButton", foreground="#222222")
        style.configure("Horizontal.TProgressbar", troughcolor="#dddddd", background="#4caf50")
        
        self.state = AppState()
        self._result_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        self.var_show_boxes = tk.BooleanVar(value=True)
        self.var_show_labels = tk.BooleanVar(value=True)
        self.var_show_template_overlay = tk.BooleanVar(value=False)
        self.var_auto_focus = tk.BooleanVar(value=True)
        self.coord_var = tk.StringVar(value=self._format_coord(None, None))
        self.result_info_var = tk.StringVar(value="尚未執行匹配")
        
        # 建立主要區域
        self._create_top_bar()
        self._create_main_layout()
        self._create_image_input_section()
        self._create_match_parameters_section()
        self._create_angle_parameters_section()
        self._create_display_options_section()
        self._create_control_section()
        self._create_display_section()
        self._create_status_section()
        
        # 載入預設圖像
        self._load_default_images()
        # 初始預覽（若有來源圖像）
        self._preview_source()
        self.update_idletasks()

    def _create_top_bar(self) -> None:
        """建立頂部工具列，確保在任何主題下都有可見元件"""
        top = tk.Frame(self, bg="#ffffff")
        top.pack(side=tk.TOP, fill=tk.X)
        tk.Button(top, text="載入來源", command=self.browse_src, bg="#e8e8e8").pack(side=tk.LEFT, padx=6, pady=6)
        tk.Button(top, text="載入樣板", command=self.browse_tmpl, bg="#e8e8e8").pack(side=tk.LEFT, padx=6, pady=6)
        tk.Button(top, text="開始匹配", command=self.run_match, bg="#d7ffd9").pack(side=tk.LEFT, padx=6, pady=6)
        tk.Label(top, text=" ").pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="適合視窗", command=self.view_fit, bg="#eef7ff").pack(side=tk.LEFT, padx=2, pady=6)
        tk.Button(top, text="100%", command=self.view_100, bg="#eef7ff").pack(side=tk.LEFT, padx=2, pady=6)
        tk.Button(top, text="放大", command=self.view_zoom_in, bg="#eef7ff").pack(side=tk.LEFT, padx=2, pady=6)
        tk.Button(top, text="縮小", command=self.view_zoom_out, bg="#eef7ff").pack(side=tk.LEFT, padx=2, pady=6)

    def _create_main_layout(self) -> None:
        """建立主要佈局區域"""
        # 左側控制面板
        self.control_panel = ttk.Frame(self, width=300)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.control_panel.pack_propagate(False)
        
        # 右側顯示區域
        self.display_panel = ttk.Frame(self)
        self.display_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _create_image_input_section(self) -> None:
        """建立圖像輸入區域"""
        # 圖像文件選擇群組
        img_group = ttk.LabelFrame(self.control_panel, text="📁 圖像文件", padding=10)
        img_group.pack(fill=tk.X, pady=(0, 10))
        
        # 來源圖像
        ttk.Label(img_group, text="來源圖像 (Source):").grid(row=0, column=0, sticky="w", pady=2)
        self.ent_src = ttk.Entry(img_group, width=30)
        self.ent_src.grid(row=1, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(img_group, text="瀏覽", command=self.browse_src, width=8).grid(row=1, column=1)
        
        # 樣板圖像
        ttk.Label(img_group, text="樣板圖像 (Template):").grid(row=2, column=0, sticky="w", pady=(10, 2))
        self.ent_tmpl = ttk.Entry(img_group, width=30)
        self.ent_tmpl.grid(row=3, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(img_group, text="瀏覽", command=self.browse_tmpl, width=8).grid(row=3, column=1)
        
        img_group.columnconfigure(0, weight=1)

    def _create_match_parameters_section(self) -> None:
        """建立匹配參數區域"""
        match_group = ttk.LabelFrame(self.control_panel, text="🎯 匹配參數", padding=10)
        match_group.pack(fill=tk.X, pady=(0, 10))
        
        # 參數變數
        self.var_target = tk.IntVar(value=5)
        self.var_score = tk.DoubleVar(value=0.75)
        self.var_overlap = tk.DoubleVar(value=0.5)
        
        # 目標數量
        ttk.Label(match_group, text="目標數量:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(match_group, textvariable=self.var_target, width=15).grid(row=0, column=1, sticky="w", padx=(5, 0))
        ttk.Label(match_group, text="個", font=("", 8)).grid(row=0, column=2, sticky="w")
        
        # 匹配分數閾值
        ttk.Label(match_group, text="最低匹配分數:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(match_group, textvariable=self.var_score, width=15).grid(row=1, column=1, sticky="w", padx=(5, 0))
        ttk.Label(match_group, text="(0-1)", font=("", 8)).grid(row=1, column=2, sticky="w")
        
        # 重疊度閾值
        ttk.Label(match_group, text="最大重疊度:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(match_group, textvariable=self.var_overlap, width=15).grid(row=2, column=1, sticky="w", padx=(5, 0))
        ttk.Label(match_group, text="(0-1)", font=("", 8)).grid(row=2, column=2, sticky="w")

    def _create_angle_parameters_section(self) -> None:
        """建立角度參數區域"""
        angle_group = ttk.LabelFrame(self.control_panel, text="🔄 旋轉搜尋", padding=10)
        angle_group.pack(fill=tk.X, pady=(0, 10))
        
        # 角度參數變數
        self.var_tol = tk.DoubleVar(value=180.0)
        self.var_coarse = tk.DoubleVar(value=15.0)
        self.var_fine = tk.DoubleVar(value=5.0)
        
        # 容忍角度
        ttk.Label(angle_group, text="角度容忍範圍:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(angle_group, textvariable=self.var_tol, width=15).grid(row=0, column=1, sticky="w", padx=(5, 0))
        ttk.Label(angle_group, text="±度", font=("", 8)).grid(row=0, column=2, sticky="w")
        
        # 粗略步進
        ttk.Label(angle_group, text="粗略搜尋步進:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(angle_group, textvariable=self.var_coarse, width=15).grid(row=1, column=1, sticky="w", padx=(5, 0))
        ttk.Label(angle_group, text="度", font=("", 8)).grid(row=1, column=2, sticky="w")
        
        # 精細步進
        ttk.Label(angle_group, text="精細搜尋步進:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(angle_group, textvariable=self.var_fine, width=15).grid(row=2, column=1, sticky="w", padx=(5, 0))
        ttk.Label(angle_group, text="度", font=("", 8)).grid(row=2, column=2, sticky="w")

    def _create_display_options_section(self) -> None:
        """建立顯示選項區域"""
        display_group = ttk.LabelFrame(self.control_panel, text="👁️ 顯示選項", padding=10)
        display_group.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(
            display_group,
            text="顯示匹配框",
            variable=self.var_show_boxes,
            command=self._refresh_overlay_image,
        ).pack(anchor="w")

        ttk.Checkbutton(
            display_group,
            text="顯示標籤資訊",
            variable=self.var_show_labels,
            command=self._refresh_overlay_image,
        ).pack(anchor="w")

        ttk.Checkbutton(
            display_group,
            text="將樣板疊加到選定結果",
            variable=self.var_show_template_overlay,
            command=self._refresh_overlay_image,
        ).pack(anchor="w")

        ttk.Checkbutton(
            display_group,
            text="自動聚焦最新結果",
            variable=self.var_auto_focus,
        ).pack(anchor="w")

    def _create_control_section(self) -> None:
        """建立控制按鈕區域"""
        control_group = ttk.LabelFrame(self.control_panel, text="⚡ 執行控制", padding=10)
        control_group.pack(fill=tk.X, pady=(0, 10))
        
        # 執行按鈕
        self.btn_run = ttk.Button(control_group, text="🚀 開始匹配", command=self.run_match)
        self.btn_run.pack(fill=tk.X, pady=2)
        
        # 清除按鈕
        ttk.Button(control_group, text="🗑️ 清除結果", command=self.clear_results).pack(fill=tk.X, pady=2)
        
        # 匯出按鈕
        ttk.Button(control_group, text="💾 匯出結果", command=self.export_results).pack(fill=tk.X, pady=2)

    def _create_display_section(self) -> None:
        """建立顯示區域"""
        display_title = ttk.LabelFrame(self.display_panel, text="📊 匹配結果顯示", padding=5)
        display_title.pack(fill=tk.BOTH, expand=True)

        paned = ttk.Panedwindow(display_title, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # 左側：結果列表與樣板預覽
        side_panel = ttk.Frame(paned, width=260)
        paned.add(side_panel, weight=1)

        result_group = ttk.LabelFrame(side_panel, text="匹配結果列表", padding=5)
        result_group.pack(fill=tk.BOTH, expand=True)

        tree_container = ttk.Frame(result_group)
        tree_container.pack(fill=tk.BOTH, expand=True)

        columns = ("id", "score", "angle", "pos")
        self.result_tree = ttk.Treeview(tree_container, columns=columns, show="headings", height=12)
        self.result_tree.heading("id", text="#")
        self.result_tree.heading("score", text="分數")
        self.result_tree.heading("angle", text="角度")
        self.result_tree.heading("pos", text="位置")
        self.result_tree.column("id", width=40, anchor="center")
        self.result_tree.column("score", width=70, anchor="center")
        self.result_tree.column("angle", width=70, anchor="center")
        self.result_tree.column("pos", width=120, anchor="center")

        scrollbar = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        info_label = ttk.Label(result_group, textvariable=self.result_info_var, anchor="w", justify=tk.LEFT, wraplength=220)
        info_label.pack(fill=tk.X, padx=2, pady=(6, 2))

        btn_frame = ttk.Frame(result_group)
        btn_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btn_frame, text="定位至結果", command=self._focus_on_selection).pack(side=tk.RIGHT)

        self.result_tree.bind("<<TreeviewSelect>>", self._on_result_select)
        self.result_tree.bind("<Double-1>", lambda _e: self._focus_on_selection())

        # 樣板預覽
        preview_group = ttk.LabelFrame(side_panel, text="樣板預覽", padding=5)
        preview_group.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.template_preview = tk.Label(preview_group, text="尚未載入樣板", anchor="center", background="#f7f7f7")
        self.template_preview.pack(fill=tk.BOTH, expand=True)
        self.template_tk: ImageTk.PhotoImage | None = None

        # 右側：圖像顯示畫布
        canvas_frame = ttk.Frame(paned)
        paned.add(canvas_frame, weight=3)
        self.canvas = tk.Canvas(canvas_frame, bg="#f0f0f0", relief="sunken", bd=2)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        def on_resize(_event=None):
            if self.state.base_bgr is not None:
                self._render_current_image()
            else:
                self._show_placeholder()

        self.canvas.bind("<Configure>", on_resize)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Leave>", lambda _e: self.coord_var.set(self._format_coord(None, None)))
        self._drag_start = None

        self._show_placeholder()

    def _create_status_section(self) -> None:
        """建立狀態列"""
        status_frame = ttk.Frame(self)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        # 進度條
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        # 狀態標籤保持固定寬度避免版面抖動
        self.coord_label = ttk.Label(status_frame, textvariable=self.coord_var, width=20, anchor="center")
        self.coord_label.pack(side=tk.RIGHT, padx=(0, 10))
        self.status_label = ttk.Label(status_frame, text="就緒", width=24, anchor="w")
        self.status_label.pack(side=tk.RIGHT)

    def _load_default_images(self) -> None:
        """載入預設測試圖像"""
        # 嘗試載入預設資源
        base_dir = os.path.dirname(__file__)
        # 往上找兩級到專案根目錄，然後找 assets 或 Test Images
        possible_paths = [
            os.path.join(base_dir, "..", "..", "..", "assets"),
            os.path.join(base_dir, "..", "..", "..", "Test Images"),
            os.path.join(base_dir, "..", "..", "assets"),
            os.path.join(base_dir, "..", "..", "Test Images"),
        ]
        
        src_files = ["Src7.bmp", "Src1.bmp", "Src3.bmp"]
        tmpl_files = ["Dst7.bmp", "Dst1.bmp", "Dst3.bmp"]
        
        default_src = None
        default_tmpl = None
        
        # 尋找可用的測試圖像
        for base_path in possible_paths:
            if not os.path.exists(base_path):
                continue
            
            # 尋找來源圖像
            if not default_src:
                for src_file in src_files:
                    src_path = os.path.abspath(os.path.join(base_path, src_file))
                    if os.path.exists(src_path):
                        default_src = src_path
                        break
            
            # 尋找樣板圖像
            if not default_tmpl:
                for tmpl_file in tmpl_files:
                    tmpl_path = os.path.abspath(os.path.join(base_path, tmpl_file))
                    if os.path.exists(tmpl_path):
                        default_tmpl = tmpl_path
                        break
                        
            if default_src and default_tmpl:
                break
        
        # 載入找到的圖像
        if default_src:
            self.ent_src.insert(0, default_src)
            self.state.src_path = os.path.abspath(default_src)
            self.status_label.config(text=f"已載入預設來源圖像: {os.path.basename(default_src)}")

        if default_tmpl:
            self.ent_tmpl.insert(0, default_tmpl)
            try:
                tmpl_preview = cv2.imread(default_tmpl, cv2.IMREAD_GRAYSCALE)
            except Exception:
                tmpl_preview = None
            self._update_template_preview(tmpl_preview)
            if tmpl_preview is not None:
                self.state.tmpl_gray = tmpl_preview
            self.state.tmpl_path = os.path.abspath(default_tmpl)
            if default_src:
                self.status_label.config(text="已載入預設測試圖像")
            else:
                self.status_label.config(text=f"已載入預設樣板圖像: {os.path.basename(default_tmpl)}")
        
        if not (default_src or default_tmpl):
            self.status_label.config(text="未找到預設測試圖像，請手動選擇")

    def _show_placeholder(self) -> None:
        """顯示提示文字佔位"""
        self.canvas.delete("all")
        self.update_idletasks()
        cx = max(50, self.canvas.winfo_width() // 2)
        cy = max(50, self.canvas.winfo_height() // 2)
        self.canvas.create_text(
            cx,
            cy,
            text="請選擇圖像並執行匹配\n匹配結果將顯示在此處",
            font=("", 12),
            fill="gray",
            anchor="center",
        )

    @staticmethod
    def _format_coord(x: int | None, y: int | None) -> str:
        """格式化座標文字，維持固定寬度避免狀態列跳動"""
        if x is None or y is None:
            return "座標: -----, -----"
        return f"座標: {x:>5}, {y:>5}"

    def _render_current_image(self) -> None:
        """依 zoom/offset 將 base_bgr 繪製到畫布。"""
        if self.state.base_bgr is None:
            self._show_placeholder()
            return
        self.update_idletasks()
        cw = max(100, self.canvas.winfo_width() - 20)
        ch = max(100, self.canvas.winfo_height() - 20)
        h, w = self.state.base_bgr.shape[:2]
        # 基礎縮放為適合視窗，再套用使用者 zoom
        fit_scale = min(cw / w, ch / h)
        scale = max(0.05, min(10.0, fit_scale * self.state.zoom))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img_rgb = cv2.cvtColor(self.state.base_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb).resize((new_w, new_h), Image.BILINEAR)
        self.state.img_tk = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        x = 10 + self.state.offset_x
        y = 10 + self.state.offset_y
        self.state.render_scale = scale
        self.state.render_origin_x = x
        self.state.render_origin_y = y
        self.state.render_width = new_w
        self.state.render_height = new_h
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.state.img_tk)
        self.canvas.update_idletasks()

    def _preview_source(self) -> None:
        """在畫布上預覽目前來源圖像（若可用）"""
        p = self.ent_src.get().strip()
        if not p or not os.path.exists(p):
            self._show_placeholder()
            return
        abs_path = os.path.abspath(p)
        if self.state.source_bgr is None or abs_path != self.state.src_path:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                self._show_placeholder()
                return
            self.state.source_bgr = img
            self.state.src_path = abs_path
        if self.state.source_bgr is None:
            self._show_placeholder()
            return
        self.state.base_bgr = self.state.source_bgr.copy()
        self.state.results = None
        self.state.src_gray = None
        self._clear_result_list(keep_template=True)
        self.state.zoom = 1.0
        self.state.offset_x = 0
        self.state.offset_y = 0
        self.coord_var.set(self._format_coord(None, None))
        self._render_current_image()

    def browse_src(self) -> None:
        """選擇來源圖像"""
        p = filedialog.askopenfilename(
            title="選擇來源影像",
            filetypes=[("圖像文件", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("所有文件", "*.*")]
        )
        if p:
            self.ent_src.delete(0, tk.END)
            self.ent_src.insert(0, p)
            self.status_label.config(text=f"已選擇來源圖像: {os.path.basename(p)}")
            self._preview_source()

    def browse_tmpl(self) -> None:
        """選擇樣板圖像"""
        p = filedialog.askopenfilename(
            title="選擇樣板影像",
            filetypes=[("圖像文件", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("所有文件", "*.*")]
        )
        if p:
            self.ent_tmpl.delete(0, tk.END)
            self.ent_tmpl.insert(0, p)
            self.status_label.config(text=f"已選擇樣板圖像: {os.path.basename(p)}")
            try:
                tmpl_preview = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                self._update_template_preview(tmpl_preview)
            except Exception:
                self._update_template_preview(None)
            self.state.results = None
            self.state.src_gray = None
            self.state.tmpl_gray = None
            self.state.tmpl_path = os.path.abspath(p)
            self._clear_result_list(keep_template=True)
            self._preview_source()

    def clear_results(self) -> None:
        """清除匹配結果"""
        # 若有來源圖像，回到來源預覽；否則顯示佔位
        self._preview_source()
        self.state.results = None
        self.state.src_gray = None
        self.state.tmpl_gray = None
        self.state.selected_index = None
        self._clear_result_list(keep_template=True)
        self.status_label.config(text="已清除結果")

    def export_results(self) -> None:
        """匯出匹配結果"""
        if not self.state.results:
            messagebox.showwarning("警告", "沒有匹配結果可以匯出！")
            return
        
        filename = filedialog.asksaveasfilename(
            title="儲存匹配結果",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.state.results, f, ensure_ascii=False, indent=2)
                self.status_label.config(text=f"結果已匯出至: {os.path.basename(filename)}")
                messagebox.showinfo("成功", f"匹配結果已儲存至:\n{filename}")
            except Exception as e:
                messagebox.showerror("錯誤", f"匯出失敗: {str(e)}")

    def _clear_result_list(self, keep_template: bool = False) -> None:
        """清空結果列表與相關狀態"""
        if hasattr(self, "result_tree"):
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
        self.state.selected_index = None
        if hasattr(self, "template_preview") and not keep_template:
            self.template_preview.config(image="", text="尚未載入樣板")
            self.template_tk = None
        self._set_result_info(default_only=not keep_template)

    def _populate_result_tree(self, results: list[dict]) -> None:
        """更新結果列表顯示"""
        self._clear_result_list(keep_template=True)
        for idx, r in enumerate(results):
            pos_txt = f"({r['pos_x']:.1f}, {r['pos_y']:.1f})"
            values = (r.get('index', idx), f"{r['score']:.3f}", f"{r['angle']:.1f}°", pos_txt)
            self.result_tree.insert("", tk.END, iid=str(idx), values=values)
        if not results:
            self._set_result_info(default_only=False)

    def _update_template_preview(self, tmpl: np.ndarray | None) -> None:
        """更新樣板預覽顯示"""
        if tmpl is None:
            self.template_preview.config(image="", text="尚未載入樣板")
            self.template_tk = None
            return

        try:
            rgb = cv2.cvtColor(tmpl, cv2.COLOR_GRAY2RGB)
        except Exception:
            rgb = tmpl
        pil = Image.fromarray(rgb)
        pil.thumbnail((240, 240), Image.BILINEAR)
        self.template_tk = ImageTk.PhotoImage(pil)
        self.template_preview.config(image=self.template_tk, text="")
        if self.state.selected_index is None:
            self._set_result_info(default_only=False)

    def _set_result_info(self, default_only: bool = False) -> None:
        """更新側邊欄的匹配資訊文字"""
        if not hasattr(self, "result_info_var"):
            return
        if self.state.results and self.state.selected_index is not None:
            idx = self.state.selected_index
            if 0 <= idx < len(self.state.results):
                r = self.state.results[idx]
                text = (
                    f"選定 #{r.get('index', idx)}\n"
                    f"分數 {r['score']:.3f}  角度 {r['angle']:.1f}°\n"
                    f"座標 ({r['pos_x']:.1f}, {r['pos_y']:.1f})"
                )
                self.result_info_var.set(text)
                return
        if default_only and self.state.results:
            return
        if self.state.results:
            self.result_info_var.set("請從列表中選擇一筆結果")
        else:
            self.result_info_var.set("尚未執行匹配")

    def _cv_to_tk(self, img_bgr: np.ndarray) -> ImageTk.PhotoImage:
        """將 OpenCV 圖像轉換為 Tkinter 可顯示的格式"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        
        # 依 Canvas 尺寸等比縮放
        self.update_idletasks()  # 確保尺寸已更新
        cw = max(100, self.canvas.winfo_width() - 20)
        ch = max(100, self.canvas.winfo_height() - 20)
        pil.thumbnail((cw, ch), Image.BILINEAR)
        return ImageTk.PhotoImage(pil)

    def run_match(self) -> None:
        """執行圖像匹配"""
        if self.state.is_processing:
            messagebox.showwarning("警告", "匹配正在進行中，請稍等...")
            return
        
        # 驗證輸入
        src_path = self.ent_src.get().strip()
        tmpl_path = self.ent_tmpl.get().strip()
        
        if not src_path or not tmpl_path:
            messagebox.showerror("錯誤", "請選擇來源圖像和樣板圖像！")
            return
        
        if not (os.path.exists(src_path) and os.path.exists(tmpl_path)):
            messagebox.showerror("錯誤", "圖像文件不存在！請檢查路徑是否正確。")
            return

        # 讀取圖像
        try:
            src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            tmpl = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
            
            if src is None:
                messagebox.showerror("錯誤", f"無法讀取來源圖像: {src_path}")
                return
            if tmpl is None:
                messagebox.showerror("錯誤", f"無法讀取樣板圖像: {tmpl_path}")
                return
            
            # 檢查圖像尺寸
            if tmpl.shape[0] >= src.shape[0] or tmpl.shape[1] >= src.shape[1]:
                messagebox.showerror("錯誤", "樣板圖像尺寸不能大於或等於來源圖像！")
                return
                
        except Exception as e:
            messagebox.showerror("錯誤", f"圖像讀取失敗: {str(e)}")
            return

        # 在後台執行匹配
        self._start_matching_thread(src, tmpl, src_path, tmpl_path)

    def _start_matching_thread(self, src: np.ndarray, tmpl: np.ndarray, src_path: str, tmpl_path: str) -> None:
        """在後台線程執行匹配"""
        self.state.is_processing = True
        self.btn_run.config(state="disabled", text="🔄 匹配中...")
        self.progress.start()
        self.status_label.config(text="正在執行圖像匹配...")
        
        def match_worker():
            try:
                print("[GUI] Start matching...", flush=True)
                print(f"[GUI] src shape: {src.shape}, tmpl shape: {tmpl.shape}", flush=True)
                # 執行匹配
                results = pyramid_match(
                    src,
                    tmpl,
                    target_num=int(self.var_target.get()),
                    score=float(self.var_score.get()),
                    max_overlap=float(self.var_overlap.get()),
                    tolerance_angle=float(self.var_tol.get()),
                    coarse_step=float(self.var_coarse.get()),
                    fine_step=float(self.var_fine.get()),
                )
                print(f"[GUI] Matching finished. results={len(results)}", flush=True)
                
                # 在主線程更新 GUI
                self.after(0, lambda: self._display_results(results, src, tmpl, src_path, tmpl_path))
                
            except Exception as e:
                print(f"[GUI] Matching error: {e}", flush=True)
                self.after(0, lambda: self._handle_match_error(str(e)))
        
        # 啟動後台線程
        thread = threading.Thread(target=match_worker, daemon=True)
        thread.start()

    def _display_results(self, results: list, src: np.ndarray, tmpl: np.ndarray, 
                        src_path: str, tmpl_path: str) -> None:
        """顯示匹配結果"""
        try:
            self.state.results = results
            self.state.src_gray = src
            self.state.tmpl_gray = tmpl
            if self.state.source_bgr is None:
                try:
                    color_src = cv2.imread(src_path, cv2.IMREAD_COLOR)
                except Exception:
                    color_src = None
                if color_src is not None:
                    self.state.source_bgr = color_src
                else:
                    self.state.source_bgr = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            self.state.zoom = 1.0
            self.state.offset_x = 0
            self.state.offset_y = 0

            self._populate_result_tree(results)
            self._update_template_preview(tmpl)

            if results:
                self.state.selected_index = 0
                self.result_tree.selection_set("0")
                self.result_tree.focus("0")
            else:
                self.state.selected_index = None

            self._refresh_overlay_image()
            self._set_result_info(default_only=False)

            if results and self.var_auto_focus.get():
                self._focus_on_selection()

            self._show_results_summary(results, src_path, tmpl_path)

        except Exception as e:
            self._handle_match_error(f"結果顯示失敗: {str(e)}")
        finally:
            self._finish_processing()

    # 視圖控制
    def view_fit(self) -> None:
        self.state.zoom = 1.0
        self.state.offset_x = 0
        self.state.offset_y = 0
        self._render_current_image()

    def view_100(self) -> None:
        # 100% 指的是以畫布適合倍率為 1，再乘上比例使顯示為接近 1:1
        self.state.zoom = max(0.05, min(10.0, max(self.canvas.winfo_width() / 2, 1) / max(1, self.canvas.winfo_width() / 2)))
        self.state.offset_x = 0
        self.state.offset_y = 0
        self._render_current_image()

    def view_zoom_in(self) -> None:
        self.state.zoom = min(10.0, self.state.zoom * 1.25)
        self._render_current_image()

    def view_zoom_out(self) -> None:
        self.state.zoom = max(0.05, self.state.zoom / 1.25)
        self._render_current_image()

    def _on_mouse_wheel(self, event) -> None:
        # macOS 也使用 <MouseWheel>，delta 單位不同，但此處僅關心正負
        if event.delta > 0:
            self.view_zoom_in()
        else:
            self.view_zoom_out()

    def _on_mouse_down(self, event) -> None:
        self._drag_start = (event.x, event.y)

    def _on_mouse_drag(self, event) -> None:
        if not self._drag_start:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        self.state.offset_x += dx
        self.state.offset_y += dy
        self._render_current_image()

    def _on_mouse_move(self, event) -> None:
        if self.state.base_bgr is None or self.state.render_scale <= 0:
            self.coord_var.set(self._format_coord(None, None))
            return
        x = (event.x - self.state.render_origin_x) / self.state.render_scale
        y = (event.y - self.state.render_origin_y) / self.state.render_scale
        ix = int(round(x))
        iy = int(round(y))
        h, w = self.state.base_bgr.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            self.coord_var.set(self._format_coord(ix, iy))
        else:
            self.coord_var.set(self._format_coord(None, None))

    def _on_result_select(self, _event=None) -> None:
        if not self.state.results:
            return
        selection = self.result_tree.selection()
        if not selection:
            return
        try:
            idx = int(selection[0])
        except (ValueError, TypeError):
            return
        if idx < 0 or idx >= len(self.state.results):
            return
        self.state.selected_index = idx
        self._set_result_info(default_only=False)
        if self.var_auto_focus.get():
            self._focus_on_selection()
        else:
            self._refresh_overlay_image()

    def _focus_on_selection(self) -> None:
        if self.state.results is None:
            return
        if self.state.selected_index is None:
            if len(self.state.results) == 0:
                return
            self.state.selected_index = 0
            try:
                self.result_tree.selection_set("0")
                self.result_tree.focus("0")
            except Exception:
                pass
        self._set_result_info(default_only=False)
        self._refresh_overlay_image()
        result = self.state.results[self.state.selected_index]
        cx = float(result.get("pos_x", 0.0))
        cy = float(result.get("pos_y", 0.0))
        # 先確保渲染資訊更新
        self._render_current_image()
        scale = max(self.state.render_scale, 1e-6)
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        target_x = canvas_w / 2
        target_y = canvas_h / 2
        self.state.offset_x = int(target_x - 10 - cx * scale)
        self.state.offset_y = int(target_y - 10 - cy * scale)
        # 適度放大以觀看細節
        self.state.zoom = min(4.0, max(self.state.zoom, 1.5))
        self._render_current_image()

    def _overlay_template_on_selection(self, out: np.ndarray, result: dict) -> None:
        if self.state.tmpl_gray is None:
            return
        th, tw = self.state.tmpl_gray.shape[:2]
        cx = float(result.get("pos_x", 0.0))
        cy = float(result.get("pos_y", 0.0))
        x0 = int(round(cx - tw / 2))
        y0 = int(round(cy - th / 2))
        x1 = x0 + tw
        y1 = y0 + th
        h, w = out.shape[:2]
        clip_x0 = max(0, x0)
        clip_y0 = max(0, y0)
        clip_x1 = min(w, x1)
        clip_y1 = min(h, y1)
        if clip_x0 >= clip_x1 or clip_y0 >= clip_y1:
            return
        tmpl_crop = self.state.tmpl_gray[clip_y0 - y0:clip_y1 - y0, clip_x0 - x0:clip_x1 - x0]
        if tmpl_crop.size == 0:
            return
        region = out[clip_y0:clip_y1, clip_x0:clip_x1]
        normalized = cv2.normalize(tmpl_crop, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        heat = cv2.applyColorMap(normalized, colormap)
        cv2.addWeighted(heat, 0.55, region, 0.45, 0, region)

    def _refresh_overlay_image(self) -> None:
        if not self.state.results or self.state.src_gray is None:
            if self.state.source_bgr is not None:
                self.state.base_bgr = self.state.source_bgr.copy()
                self._render_current_image()
            return

        if self.state.source_bgr is not None:
            out = self.state.source_bgr.copy()
        else:
            out = cv2.cvtColor(self.state.src_gray, cv2.COLOR_GRAY2BGR)
        th, tw = (0, 0)
        if self.state.tmpl_gray is not None:
            th, tw = self.state.tmpl_gray.shape[:2]

        for i, r in enumerate(self.state.results):
            cx = float(r.get("pos_x", 0.0))
            cy = float(r.get("pos_y", 0.0))
            x0 = int(round(cx - tw / 2))
            y0 = int(round(cy - th / 2))
            x1 = x0 + tw
            y1 = y0 + th
            color = self._result_colors[i % len(self._result_colors)]
            thickness = 3 if self.state.selected_index == i else 2

            if self.var_show_boxes.get():
                cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
                cv2.circle(out, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)
            elif self.state.selected_index == i:
                cv2.drawMarker(
                    out,
                    (int(round(cx)), int(round(cy))),
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=16,
                    thickness=2,
                )

            if self.var_show_labels.get():
                label = f"#{r.get('index', i)} {r['score']:.3f} {r['angle']:.1f}°"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_y = max(label_size[1] + 5, y0 - 5)
                cv2.rectangle(out, (x0, label_y - label_size[1] - 3),
                              (x0 + label_size[0] + 6, label_y + 3), (0, 0, 0), -1)
                cv2.putText(out, label, (x0 + 3, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if self.var_show_template_overlay.get() and self.state.selected_index is not None:
            try:
                self._overlay_template_on_selection(out, self.state.results[self.state.selected_index])
            except Exception:
                pass

        self.state.base_bgr = out
        self._render_current_image()

    def _show_results_summary(self, results: list, src_path: str, tmpl_path: str) -> None:
        """顯示結果摘要"""
        if not results:
            summary = "未找到匹配項目"
        else:
            summary = f"找到 {len(results)} 個匹配項目:\n\n"
            for r in results:
                summary += f"#{r['index']}: 位置({r['pos_x']:.1f}, {r['pos_y']:.1f}) "
                summary += f"分數={r['score']:.3f} 角度={r['angle']:.1f}°\n"
        
        # 使用較小的對話框顯示結果
        result_window = tk.Toplevel(self)
        result_window.title("匹配結果")
        result_window.geometry("400x300")
        result_window.transient(self)
        result_window.grab_set()
        
        # 結果文字區域
        text_frame = ttk.Frame(result_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, height=15)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.insert(tk.END, summary)
        text_widget.config(state=tk.DISABLED)
        
        # 按鈕
        btn_frame = ttk.Frame(result_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(btn_frame, text="關閉", command=result_window.destroy).pack(side=tk.RIGHT)
        if results:
            ttk.Button(btn_frame, text="匯出詳細結果", 
                      command=lambda: self.export_results()).pack(side=tk.RIGHT, padx=(0, 5))

    def _handle_match_error(self, error_msg: str) -> None:
        """處理匹配錯誤"""
        messagebox.showerror("匹配失敗", f"圖像匹配過程中發生錯誤:\n{error_msg}")
        self._finish_processing()

    def _finish_processing(self) -> None:
        """完成處理，恢復 GUI 狀態"""
        self.state.is_processing = False
        self.btn_run.config(state="normal", text="🚀 開始匹配")
        self.progress.stop()
        self.status_label.config(text="匹配完成" if self.state.results else "就緒")


def main() -> None:
    app = FpmGui()
    app.mainloop()


if __name__ == "__main__":
    main()
