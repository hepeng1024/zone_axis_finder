#!/usr/bin/env python3
"""
Tkinter GUI for zone_axis_finder.py.

Run with:
  /home/hepeng/miniconda3/envs/test/bin/python zone_axis_finder_gui.py

The GUI is a thin wrapper around zone_axis_finder.py, so the original command
line script remains the single source of truth for indexing and tilt-angle
calculation.
"""

from __future__ import annotations

import contextlib
import io
import math
import queue
import re
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import BooleanVar, StringVar, filedialog, messagebox
from typing import Sequence
import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise SystemExit(
        f"Missing package: {exc.name}\n"
        "Run this GUI with the same environment used for zone_axis_finder.py, for example:\n"
        "  /home/hepeng/miniconda3/envs/test/bin/python zone_axis_finder_gui.py"
    )

try:
    import numpy as np
    import zone_axis_finder
except ImportError as exc:  # pragma: no cover - depends on launch directory.
    raise SystemExit(f"Could not import zone_axis_finder.py: {exc}") from exc

try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib import patheffects as path_effects
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise SystemExit(
        f"Missing package: {exc.name}\n"
        "The sample tilt simulator needs matplotlib in the same Python environment."
    ) from exc


APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "assets"


def asset_path(filename: str) -> Path:
    return ASSET_DIR / filename


@dataclass
class AnalysisContext:
    image_path: Path | None
    match: object
    crystal_to_zero_holder: object
    target_families: list[str]
    include_opposites: bool
    holder_order: str
    alpha_limits: tuple[float, float]
    beta_limits: tuple[float, float]
    image_to_holder_rotation_deg: float
    map_label_individual_color: bool
    map_reachable_only: bool


@dataclass
class PreviewResult:
    predicted_path: Path | None = None
    fitted_path: Path | None = None
    predicted_image: Image.Image | None = None
    fitted_image: Image.Image | None = None
    map_image: Image.Image | None = None
    analysis_context: AnalysisContext | None = None


class ScrollFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.inner.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._update_window_width)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _update_scroll_region(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _update_window_width(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(3, "units")
        elif event.delta:
            self.canvas.yview_scroll(int(-event.delta / 120), "units")


class SampleTiltSimulator(tk.Toplevel):
    tilt_step_deg = 0.1
    tilt_speed_options = (0.1, 0.5, 1.0, 2.0, 5.0)
    repeat_ms = 10

    def __init__(
        self,
        parent: tk.Toplevel,
        context: AnalysisContext,
        sample_ccw_deg: float,
        offset_deg: float,
    ) -> None:
        super().__init__(parent)
        self.title("Sample Tilt Simulator")
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.geometry(f"{min(1580, max(1180, screen_w - 80))}x{min(900, max(760, screen_h - 100))}")
        self.minsize(1120, 700)
        self.context = context
        self.sample_ccw_deg = sample_ccw_deg
        self.offset_deg = offset_deg
        self.alpha_deg = 0.0
        self.beta_deg = 0.0
        self.alpha_var = StringVar(value="0.00")
        self.beta_var = StringVar(value="0.00")
        self.tilt_speed_var = StringVar(value=self._format_tilt_speed(self.tilt_step_deg))
        self.accelerating_voltage_var = StringVar(value="200")
        self.lattice_parameter_var = StringVar(value=f"{self._initial_lattice_parameter_nm():.4f}")
        self.sample_thickness_var = StringVar(value="10")
        self.diffraction_max_index_var = StringVar(value="20")
        self.relrod_method_var = StringVar(value="Gaussian")
        self.camera_length_var = StringVar(value="200 mm")
        self.scale_bar_pixels_var = StringVar(value="")
        self.scale_bar_value_var = StringVar(value="")
        self.snap_crystal_view_var = BooleanVar(value=True)
        self.show_crystal_axes_var = BooleanVar(value=True)
        self.tilt_log: list[str] = []
        self.arrow_photos: dict[str, ImageTk.PhotoImage] = {}
        self._hold_axis: str | None = None
        self._hold_direction = 0.0
        self._repeat_after_id: str | None = None
        self.footer_texts: list[object] = []
        self.pole_range = 45.0
        self.current_elev = 0.0
        self.current_azim = -90.0
        self.current_roll = 0.0
        self._is_syncing_crystal_views = False
        self.crystal_to_zero_holder = np.asarray(context.crystal_to_zero_holder, dtype=float)
        self.crystal_display_correction = np.diag([-1.0, -1.0, 1.0])

        self.base_lamella = self.create_cuboid(width_x=8, height_y=8, depth_z=0.5)
        self.base_capping = self.create_cuboid(width_x=1, height_y=8, depth_z=0.5, offset_x=9)
        self.base_fcc = self.create_fcc_lattice(size=5.6)
        self.base_bcc = self.create_bcc_reciprocal(size=4.6)
        self.diffraction_max_index = 20
        self.diffraction_hkl = self.generate_fcc_diffraction_hkl(max_index=self.diffraction_max_index)
        self.camera_length_options = (80.0, 100.0, 150.0, 200.0, 300.0, 500.0, 800.0, 1000.0, 1500.0)

        self._build_ui()
        self.bind("<Return>", lambda _event: self.apply_tilt_entries())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._initialize_scale_bar_detection()
        self.update_plot()
        self._write_output()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        ttk.Label(self, text="Sample Tilt Simulator", style="Title.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(8, 4)
        )

        plot_frame = ttk.LabelFrame(self, padding=8)
        plot_frame.grid(row=1, column=0, sticky="nw", padx=(8, 4), pady=(0, 8))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        bg_color = self.cget("bg")
        hex_bg = (
            f"#{int(self.winfo_rgb(bg_color)[0] / 256):02x}"
            f"{int(self.winfo_rgb(bg_color)[1] / 256):02x}"
            f"{int(self.winfo_rgb(bg_color)[2] / 256):02x}"
        )
        self.plot_bg_hex = hex_bg
        self.fig = Figure(figsize=(6.05, 5.55), dpi=100, facecolor=hex_bg)
        grid = self.fig.add_gridspec(2, 2, width_ratios=[1.22, 1.0], height_ratios=[1.0, 1.0])
        self.ax = self.fig.add_subplot(grid[:, 0], projection="3d")
        self.real_ax = self.fig.add_subplot(grid[0, 1], projection="3d")
        self.recip_ax = self.fig.add_subplot(grid[1, 1], projection="3d")
        self.crystal_axes = [self.ax, self.real_ax, self.recip_ax]
        for axis in self.crystal_axes:
            axis.set_facecolor(hex_bg)
        self.fig.subplots_adjust(left=0.01, right=0.995, top=0.94, bottom=0.105, wspace=0.02, hspace=0.18)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("motion_notify_event", self._sync_crystal_cameras)

        pole_frame = ttk.LabelFrame(self, text="Dynamic Pole Figure", style="Section.TLabelframe", padding=8)
        pole_frame.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(0, 8))
        pole_frame.columnconfigure(0, weight=1)
        pole_frame.rowconfigure(0, weight=1)
        self.pole_fig = Figure(figsize=(4.7, 4.7), dpi=100, facecolor="black")
        self.pole_ax = self.pole_fig.add_subplot(111)
        self.pole_canvas = FigureCanvasTkAgg(self.pole_fig, master=pole_frame)
        self.pole_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        diffraction_frame = ttk.LabelFrame(self, text="Simulated Diffraction Pattern", style="Section.TLabelframe", padding=8)
        diffraction_frame.grid(row=1, column=2, sticky="nsew", padx=(0, 8), pady=(0, 8))
        diffraction_frame.columnconfigure(0, weight=1)
        diffraction_frame.rowconfigure(0, weight=1)
        self.diffraction_fig = Figure(figsize=(4.7, 4.7), dpi=100, facecolor="black")
        self.diffraction_ax = self.diffraction_fig.add_subplot(111)
        self.diffraction_canvas = FigureCanvasTkAgg(self.diffraction_fig, master=diffraction_frame)
        self.diffraction_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        controls = ttk.LabelFrame(self, text="Tilt Control", style="Section.TLabelframe", padding=4)
        controls.grid(row=2, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        alpha_label = ttk.Label(controls, text="Alpha tilt")
        alpha_label.grid(row=0, column=0, columnspan=2, pady=(0, 2))
        alpha_minus = ttk.Button(
            controls,
            image=self._load_arrow_photo("Negative alpha tilt arrow.png", (40, 36)),
        )
        alpha_minus.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=(0, 2), ipadx=1, ipady=0)
        alpha_plus = ttk.Button(
            controls,
            image=self._load_arrow_photo("Positive alpha tilt arrow.png", (40, 36)),
        )
        alpha_plus.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(0, 2), ipadx=1, ipady=0)
        self._bind_hold_button(alpha_minus, "alpha", -1.0)
        self._bind_hold_button(alpha_plus, "alpha", 1.0)

        beta_label = ttk.Label(controls, text="Beta tilt")
        beta_label.grid(row=0, column=2, pady=(0, 2))
        beta_plus = ttk.Button(
            controls,
            image=self._load_arrow_photo("Positive beta tilt arrow.png", (30, 40)),
        )
        beta_plus.grid(row=1, column=2, sticky="nsew", padx=(8, 0), pady=(0, 2), ipadx=9, ipady=0)
        beta_minus = ttk.Button(
            controls,
            image=self._load_arrow_photo("Negative beta tilt arrow.png", (30, 40)),
        )
        beta_minus.grid(row=2, column=2, sticky="nsew", padx=(8, 0), pady=(0, 3), ipadx=9, ipady=0)
        self._bind_hold_button(beta_plus, "beta", 1.0)
        self._bind_hold_button(beta_minus, "beta", -1.0)

        speed_frame = ttk.Frame(controls)
        speed_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(1, 2))
        speed_frame.columnconfigure(1, weight=1)
        ttk.Label(speed_frame, text="Tilt speed").grid(row=0, column=0, sticky="w")
        speed_combo = ttk.Combobox(
            speed_frame,
            textvariable=self.tilt_speed_var,
            values=[self._format_tilt_speed(value) for value in self.tilt_speed_options],
            state="readonly",
            width=13,
        )
        speed_combo.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        speed_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_tilt_speed_selected())
        ttk.Button(speed_frame, text="Fine", width=7, command=lambda: self._change_tilt_speed(-1)).grid(
            row=0, column=2, padx=(0, 5)
        )
        ttk.Button(speed_frame, text="Coarse", width=7, command=lambda: self._change_tilt_speed(1)).grid(
            row=0, column=3
        )

        entries = ttk.Frame(controls)
        entries.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        entries.columnconfigure(1, weight=1)
        entries.columnconfigure(3, weight=1)
        ttk.Label(entries, text="Set alpha").grid(row=0, column=0, sticky="w")
        alpha_entry = ttk.Entry(entries, textvariable=self.alpha_var, width=11)
        alpha_entry.grid(row=0, column=1, sticky="ew", padx=(6, 14))
        ttk.Label(entries, text="Set beta").grid(row=0, column=2, sticky="w")
        beta_entry = ttk.Entry(entries, textvariable=self.beta_var, width=11)
        beta_entry.grid(row=0, column=3, sticky="ew", padx=(6, 0))
        alpha_entry.bind("<Return>", lambda _event: self.apply_tilt_entries())
        beta_entry.bind("<Return>", lambda _event: self.apply_tilt_entries())

        ttk.Button(entries, text="Apply", command=self.apply_tilt_entries).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0), padx=(0, 8)
        )
        ttk.Button(entries, text="Reset alpha/beta", command=self.reset_tilt).grid(
            row=1, column=2, columnspan=2, sticky="ew", pady=(5, 0)
        )
        view_options = ttk.Frame(controls)
        view_options.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(3, 0))
        ttk.Checkbutton(
            view_options,
            text="Snap crystal view",
            variable=self.snap_crystal_view_var,
            command=self.update_plot,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            view_options,
            text="Show holder axes in crystals",
            variable=self.show_crystal_axes_var,
            command=self.update_plot,
        ).grid(row=0, column=1, sticky="w", padx=(14, 0))

        output = ttk.LabelFrame(self, text="Tilt Output", style="Section.TLabelframe", padding=4)
        output.grid(row=2, column=1, sticky="nsew", padx=(4, 8), pady=(0, 8))
        output.columnconfigure(0, weight=1)
        output.rowconfigure(0, weight=1)
        self.output_text = tk.Text(output, wrap="none", width=58, height=5)
        y_scroll = ttk.Scrollbar(output, orient="vertical", command=self.output_text.yview)
        x_scroll = ttk.Scrollbar(output, orient="horizontal", command=self.output_text.xview)
        self.output_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.output_text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        diffraction_params = ttk.LabelFrame(self, text="Diffraction Settings", style="Section.TLabelframe", padding=4)
        diffraction_params.grid(row=2, column=2, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self._build_diffraction_parameter_panel(diffraction_params)

    def _build_diffraction_parameter_panel(self, parent: ttk.LabelFrame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)

        ttk.Label(parent, text="Voltage kV").grid(row=0, column=0, sticky="w", padx=(0, 5), pady=(0, 3))
        voltage_entry = ttk.Entry(parent, textvariable=self.accelerating_voltage_var, width=9)
        voltage_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(0, 3))
        ttk.Label(parent, text="Lattice nm").grid(row=0, column=2, sticky="w", padx=(0, 5), pady=(0, 3))
        lattice_entry = ttk.Entry(parent, textvariable=self.lattice_parameter_var, width=9)
        lattice_entry.grid(row=0, column=3, sticky="ew", pady=(0, 3))

        ttk.Label(parent, text="Thickness nm").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(0, 3))
        thickness_entry = ttk.Entry(parent, textvariable=self.sample_thickness_var, width=9)
        thickness_entry.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(0, 3))
        ttk.Label(parent, text="Max hkl").grid(row=1, column=2, sticky="w", padx=(0, 5), pady=(0, 3))
        max_index_entry = ttk.Entry(parent, textvariable=self.diffraction_max_index_var, width=9)
        max_index_entry.grid(row=1, column=3, sticky="ew", pady=(0, 3))

        ttk.Label(parent, text="Relrod model").grid(row=2, column=0, sticky="w", padx=(0, 5), pady=(0, 3))
        relrod_combo = ttk.Combobox(
            parent,
            textvariable=self.relrod_method_var,
            values=("sinc", "Gaussian"),
            state="readonly",
            width=9,
        )
        relrod_combo.grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=(0, 3))
        relrod_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_diffraction_pattern())

        ttk.Label(parent, text="Scale px").grid(row=2, column=2, sticky="w", padx=(0, 5), pady=(0, 3))
        scale_px_entry = ttk.Entry(parent, textvariable=self.scale_bar_pixels_var, width=9)
        scale_px_entry.grid(row=2, column=3, sticky="ew", pady=(0, 3))

        ttk.Label(parent, text="Bar 1/nm").grid(row=3, column=0, sticky="w", padx=(0, 5), pady=(0, 3))
        scale_value_entry = ttk.Entry(parent, textvariable=self.scale_bar_value_var, width=9)
        scale_value_entry.grid(row=3, column=1, sticky="ew", padx=(0, 10), pady=(0, 3))

        camera_row = ttk.Frame(parent)
        camera_row.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(2, 0))
        camera_row.columnconfigure(1, weight=1)
        ttk.Label(camera_row, text="Camera length").grid(row=0, column=0, sticky="w")
        camera_combo = ttk.Combobox(
            camera_row,
            textvariable=self.camera_length_var,
            values=[self._format_camera_length(value) for value in self.camera_length_options],
            state="readonly",
            width=11,
        )
        camera_combo.grid(row=0, column=1, sticky="ew", padx=(6, 5))
        camera_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_diffraction_pattern())
        ttk.Button(camera_row, text="-", width=3, command=lambda: self._change_camera_length(-1)).grid(
            row=0, column=2, padx=(0, 4)
        )
        ttk.Button(camera_row, text="+", width=3, command=lambda: self._change_camera_length(1)).grid(row=0, column=3)

        action_row = ttk.Frame(parent)
        action_row.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        action_row.columnconfigure(0, weight=1)
        action_row.columnconfigure(1, weight=1)
        action_row.columnconfigure(2, weight=1)
        ttk.Button(action_row, text="Detect scale bar", command=self._detect_scale_bar_from_input).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(action_row, text="Estimate lattice", command=self._estimate_lattice_from_settings).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        ttk.Button(action_row, text="Apply", command=self._update_diffraction_pattern).grid(
            row=0, column=2, sticky="ew", padx=(8, 0)
        )

        note = ttk.Label(
            parent,
            text="Use Detect scale bar, then enter the printed bar value such as 5 1/nm.",
            anchor="w",
        )
        note.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(4, 0))

        for entry in (voltage_entry, lattice_entry, thickness_entry, max_index_entry, scale_px_entry, scale_value_entry):
            entry.bind("<Return>", lambda _event: self._update_diffraction_pattern())

    def _sync_crystal_cameras(self, event: object) -> None:
        if self._is_syncing_crystal_views:
            return
        source_axis = getattr(event, "inaxes", None)
        if source_axis not in self.crystal_axes:
            return

        self._is_syncing_crystal_views = True
        self.current_elev = float(getattr(source_axis, "elev", self.current_elev))
        self.current_azim = float(getattr(source_axis, "azim", self.current_azim))
        self.current_roll = float(getattr(source_axis, "roll", self.current_roll))
        for target_axis in self.crystal_axes:
            if target_axis is source_axis:
                continue
            if hasattr(target_axis, "roll"):
                target_axis.view_init(elev=self.current_elev, azim=self.current_azim, roll=self.current_roll)
            else:
                target_axis.view_init(elev=self.current_elev, azim=self.current_azim)
        self.canvas.draw_idle()
        self._is_syncing_crystal_views = False

    def _load_arrow_photo(self, filename: str, max_size: tuple[int, int]) -> ImageTk.PhotoImage:
        key = f"{filename}:{max_size[0]}x{max_size[1]}"
        if key in self.arrow_photos:
            return self.arrow_photos[key]
        image = Image.open(asset_path(filename)).convert("RGBA")
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.arrow_photos[key] = photo
        return photo

    @staticmethod
    def _format_tilt_speed(value: float) -> str:
        return f"{value:g} deg/step"

    def _tilt_speed_index(self) -> int:
        step = self._current_tilt_step()
        return min(range(len(self.tilt_speed_options)), key=lambda index: abs(self.tilt_speed_options[index] - step))

    def _current_tilt_step(self) -> float:
        match = re.match(r"\s*([0-9]*\.?[0-9]+)", self.tilt_speed_var.get())
        if match:
            with contextlib.suppress(ValueError):
                value = float(match.group(1))
                if value > 0:
                    return value
        return self.tilt_step_deg

    def _on_tilt_speed_selected(self) -> None:
        self._append_log(f"Tilt speed set to {self._current_tilt_step():g} deg/step.")
        self._write_output()

    def _change_tilt_speed(self, direction: int) -> None:
        index = min(max(self._tilt_speed_index() + direction, 0), len(self.tilt_speed_options) - 1)
        self.tilt_speed_var.set(self._format_tilt_speed(self.tilt_speed_options[index]))
        self._on_tilt_speed_selected()

    @staticmethod
    def _initial_lattice_parameter_nm() -> float:
        return 0.3524

    @staticmethod
    def _format_camera_length(value: float) -> str:
        return f"{value:g} mm"

    def _camera_length_mm(self) -> float:
        match = re.match(r"\s*([0-9]*\.?[0-9]+)", self.camera_length_var.get())
        if not match:
            return 200.0
        return max(1.0, float(match.group(1)))

    def _camera_length_index(self) -> int:
        current = self._camera_length_mm()
        return min(range(len(self.camera_length_options)), key=lambda index: abs(self.camera_length_options[index] - current))

    def _change_camera_length(self, direction: int) -> None:
        index = min(max(self._camera_length_index() + direction, 0), len(self.camera_length_options) - 1)
        self.camera_length_var.set(self._format_camera_length(self.camera_length_options[index]))
        self._update_diffraction_pattern()

    @staticmethod
    def _electron_wavelength_nm(voltage_kv: float) -> float:
        voltage_v = max(voltage_kv, 1e-6) * 1000.0
        wavelength_angstrom = 12.3986 / math.sqrt(voltage_v * (1.0 + 0.97845e-6 * voltage_v))
        return wavelength_angstrom * 0.1

    @staticmethod
    def generate_fcc_diffraction_hkl(max_index: int) -> np.ndarray:
        points: list[list[float]] = []
        for h in range(-max_index, max_index + 1):
            for k in range(-max_index, max_index + 1):
                for l in range(-max_index, max_index + 1):
                    if h == k == l == 0:
                        continue
                    if h * h + k * k + l * l > max_index * max_index:
                        continue
                    if zone_axis_finder.fcc_allowed(h, k, l):
                        points.append([float(h), float(k), float(l)])
        if not points:
            return np.zeros((3, 0), dtype=float)
        return np.asarray(points, dtype=float).T

    def _diffraction_max_index_from_var(self) -> int:
        text = self.diffraction_max_index_var.get().strip()
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError("Max hkl must be an integer.") from exc
        if value < 1 or value > 30:
            raise ValueError("Max hkl must be between 1 and 30.")
        return value

    def _ensure_diffraction_hkl(self, max_index: int) -> None:
        if max_index == self.diffraction_max_index:
            return
        self.diffraction_hkl = self.generate_fcc_diffraction_hkl(max_index=max_index)
        self.diffraction_max_index = max_index

    def _positive_float_from_var(self, variable: StringVar, label: str) -> float:
        try:
            value = float(variable.get().strip())
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric.") from exc
        if value <= 0:
            raise ValueError(f"{label} must be positive.")
        return value

    def _optional_positive_float_from_var(self, variable: StringVar, label: str) -> float | None:
        text = variable.get().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric.") from exc
        if value <= 0:
            raise ValueError(f"{label} must be positive.")
        return value

    def _optional_positive_number_from_var(self, variable: StringVar, label: str) -> float | None:
        text = variable.get().strip()
        if not text:
            return None
        match = re.search(r"([0-9]+(?:\.[0-9]*)?|\.[0-9]+)", text)
        if not match:
            raise ValueError(f"{label} must include a positive number.")
        value = float(match.group(1))
        if value <= 0:
            raise ValueError(f"{label} must be positive.")
        return value

    def _initialize_scale_bar_detection(self) -> None:
        if self.context.image_path is None:
            return
        with contextlib.suppress(Exception):
            result = zone_axis_finder.detect_scale_bar_pixels(Path(self.context.image_path))
            if result is not None:
                length_px = float(result["length_px"])
                self.scale_bar_pixels_var.set(f"{length_px:.1f}")
                self._append_log(f"Auto-detected scale bar length: {length_px:.1f} px.")

    def _detect_scale_bar_from_input(self) -> None:
        if self.context.image_path is None:
            messagebox.showinfo("No Image", "No diffraction image is available for scale-bar detection.", parent=self)
            return
        try:
            result = zone_axis_finder.detect_scale_bar_pixels(Path(self.context.image_path))
        except Exception as exc:  # pragma: no cover - defensive for unusual image files.
            messagebox.showerror("Scale Bar Detection Failed", str(exc), parent=self)
            return
        if result is None:
            messagebox.showinfo(
                "Scale Bar Not Found",
                "I could not find a bright horizontal scale bar. Enter the scale-bar length in pixels manually.",
                parent=self,
            )
            return
        length_px = float(result["length_px"])
        self.scale_bar_pixels_var.set(f"{length_px:.1f}")
        self._append_log(f"Detected scale bar length: {length_px:.1f} px.")
        self._write_output()
        messagebox.showinfo(
            "Scale Bar Detected",
            f"Detected a scale-bar line of {length_px:.1f} px.\n\n"
            "Enter the printed reciprocal-space value in Bar 1/nm, for example 5 or 5 1/nm, then click Estimate lattice.",
            parent=self,
        )

    def _estimate_lattice_from_settings(self) -> None:
        try:
            scale_bar_px = self._optional_positive_float_from_var(self.scale_bar_pixels_var, "Scale bar pixels")
            scale_bar_value = self._optional_positive_number_from_var(self.scale_bar_value_var, "Scale bar value")
        except ValueError as exc:
            messagebox.showerror("Invalid Diffraction Setting", str(exc), parent=self)
            return
        if scale_bar_px is None or scale_bar_value is None:
            messagebox.showinfo(
                "Need Scale Bar",
                "Enter both the detected scale-bar length in pixels and the printed reciprocal value in 1/nm.",
                parent=self,
            )
            return

        match = self.context.match
        scale_px_per_index = float(getattr(match, "scale", 0.0))
        if scale_px_per_index <= 0:
            messagebox.showinfo(
                "Cannot Estimate Lattice",
                "The fitted diffraction-pattern scale is unavailable. Using the lattice entry directly.",
                parent=self,
            )
            return

        lattice_nm = scale_bar_px / (scale_bar_value * scale_px_per_index)
        self.lattice_parameter_var.set(f"{lattice_nm:.4f}")
        self._append_log(f"Estimated lattice parameter a={lattice_nm:.4f} nm.")
        self._write_output()
        self._update_diffraction_pattern()

    def _bind_hold_button(self, button: ttk.Button, axis: str, direction: float) -> None:
        button.bind("<ButtonPress-1>", lambda _event: self._start_hold(axis, direction))
        button.bind("<ButtonRelease-1>", lambda _event: self._stop_hold())
        button.bind("<Leave>", lambda _event: self._stop_hold())

    def _start_hold(self, axis: str, direction: float) -> None:
        self._stop_hold()
        self._hold_axis = axis
        self._hold_direction = direction
        self._repeat_tilt()

    def _stop_hold(self) -> None:
        if self._repeat_after_id is not None:
            with contextlib.suppress(tk.TclError):
                self.after_cancel(self._repeat_after_id)
        self._repeat_after_id = None
        self._hold_axis = None
        self._hold_direction = 0.0

    def _repeat_tilt(self) -> None:
        if self._hold_axis is None:
            return
        delta = self._hold_direction * self._current_tilt_step()
        if self._nudge_tilt(self._hold_axis, delta):
            self._repeat_after_id = self.after(self.repeat_ms, self._repeat_tilt)
        else:
            self._stop_hold()

    def _axis_limits(self, axis: str) -> tuple[float, float]:
        raw = self.context.alpha_limits if axis == "alpha" else self.context.beta_limits
        return min(raw), max(raw)

    def _nudge_tilt(self, axis: str, delta: float) -> bool:
        low, high = self._axis_limits(axis)
        current = self.alpha_deg if axis == "alpha" else self.beta_deg
        proposed = current + delta
        clipped = min(max(proposed, low), high)
        limit_hit = not math.isclose(proposed, clipped, abs_tol=1e-12)
        if axis == "alpha":
            self.alpha_deg = clipped
        else:
            self.beta_deg = clipped
        self._sync_entries()
        self.update_plot()
        if limit_hit:
            limit = high if proposed > high else low
            self._append_log_once(f"{axis.capitalize()} tilt reached {limit:.2f} deg limit.")
            self._write_output()
            return False
        self._write_output()
        return True

    def apply_tilt_entries(self) -> None:
        try:
            alpha = float(self.alpha_var.get().strip())
            beta = float(self.beta_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Tilt", "Enter numeric alpha and beta tilt angles.", parent=self)
            return

        alpha_low, alpha_high = self._axis_limits("alpha")
        beta_low, beta_high = self._axis_limits("beta")
        outside: list[str] = []
        if not (alpha_low <= alpha <= alpha_high):
            outside.append(f"alpha must be between {alpha_low:.2f} and {alpha_high:.2f} deg")
        if not (beta_low <= beta <= beta_high):
            outside.append(f"beta must be between {beta_low:.2f} and {beta_high:.2f} deg")
        if outside:
            messagebox.showwarning(
                "Tilt Limit Exceeded",
                "The requested tilt is outside the holder limits:\n\n" + "\n".join(outside),
                parent=self,
            )
            self._sync_entries()
            return

        self.alpha_deg = alpha
        self.beta_deg = beta
        self._append_log(f"Applied alpha={alpha:.2f} deg, beta={beta:.2f} deg.")
        self._sync_entries()
        self.update_plot()
        self._write_output()

    def reset_tilt(self) -> None:
        self.alpha_deg = 0.0
        self.beta_deg = 0.0
        self._append_log("Reset alpha/beta tilts to 0.00 deg.")
        self._sync_entries()
        self.update_plot()
        self._write_output()

    def set_sample_rotation(self, sample_ccw_deg: float, offset_deg: float) -> None:
        self.sample_ccw_deg = sample_ccw_deg
        self.offset_deg = offset_deg
        self.update_plot()
        self._write_output()

    def actual_sample_rotation_deg(self) -> float:
        return ((self.offset_deg + self.sample_ccw_deg + 180.0) % 360.0) - 180.0

    def _sync_entries(self) -> None:
        self.alpha_var.set(f"{self.alpha_deg:.2f}")
        self.beta_var.set(f"{self.beta_deg:.2f}")

    def _append_log(self, line: str) -> None:
        self.tilt_log.append(line)
        self.tilt_log = self.tilt_log[-12:]

    def _append_log_once(self, line: str) -> None:
        if not self.tilt_log or self.tilt_log[-1] != line:
            self._append_log(line)

    def _write_output(self) -> None:
        alpha_low, alpha_high = self._axis_limits("alpha")
        beta_low, beta_high = self._axis_limits("beta")
        lines = [
            f"Current alpha: {self.alpha_deg:.2f} deg",
            f"Current beta : {self.beta_deg:.2f} deg",
            f"Sample CCW  : {self.sample_ccw_deg:.2f} deg",
            f"Offset CCW  : {self.offset_deg:.2f} deg",
            f"Tilt speed  : {self._current_tilt_step():g} deg/step",
            f"Diffraction : {self.accelerating_voltage_var.get().strip() or '200'} kV, "
            f"a={self.lattice_parameter_var.get().strip() or '0.3524'} nm, "
            f"t={self.sample_thickness_var.get().strip() or '10'} nm, "
            f"L={self._camera_length_mm():g} mm",
            f"Relrod     : {self.relrod_method_var.get().strip() or 'Gaussian'}, "
            f"max hkl={self.diffraction_max_index_var.get().strip() or '20'}",
            f"Scale bar   : {self.scale_bar_pixels_var.get().strip() or '?'} px = "
            f"{self.scale_bar_value_var.get().strip() or '?'} 1/nm",
            f"Alpha limits: {alpha_low:.2f} to {alpha_high:.2f} deg",
            f"Beta limits : {beta_low:.2f} to {beta_high:.2f} deg",
            "",
        ]
        if self.tilt_log:
            lines.extend([f"Latest: {self.tilt_log[-1]}", "", "Log:"])
            lines.extend(self.tilt_log)
        else:
            lines.append("Use the arrow buttons or type alpha/beta values and press Enter or Apply.")
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("end", "\n".join(lines) + "\n")
        self.output_text.see("1.0")

    @staticmethod
    def create_cuboid(
        width_x: float,
        height_y: float,
        depth_z: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0,
    ) -> np.ndarray:
        x = np.array([-width_x, width_x, width_x, -width_x, -width_x, width_x, width_x, -width_x]) + offset_x
        y = np.array([-height_y, -height_y, height_y, height_y, -height_y, -height_y, height_y, height_y]) + offset_y
        z = np.array([-depth_z, -depth_z, -depth_z, -depth_z, depth_z, depth_z, depth_z, depth_z]) + offset_z
        return np.vstack((x, y, z))

    @staticmethod
    def create_fcc_lattice(size: float) -> np.ndarray:
        corners = np.array(
            [
                [-size, -size, -size],
                [size, -size, -size],
                [size, size, -size],
                [-size, size, -size],
                [-size, -size, size],
                [size, -size, size],
                [size, size, size],
                [-size, size, size],
            ],
            dtype=float,
        )
        face_centers = np.array(
            [
                [0.0, 0.0, size],
                [0.0, 0.0, -size],
                [size, 0.0, 0.0],
                [-size, 0.0, 0.0],
                [0.0, size, 0.0],
                [0.0, -size, 0.0],
            ],
            dtype=float,
        )
        return np.vstack((corners, face_centers)).T

    @staticmethod
    def create_bcc_reciprocal(size: float) -> np.ndarray:
        points: list[list[float]] = [[0.0, 0.0, 0.0]]
        for i in (-1.0, 1.0):
            for j in (-1.0, 1.0):
                for k in (-1.0, 1.0):
                    points.append([i * size, j * size, k * size])
        for i in (-2.0, 2.0):
            points.append([i * size, 0.0, 0.0])
            points.append([0.0, i * size, 0.0])
            points.append([0.0, 0.0, i * size])
        return np.asarray(points, dtype=float).T

    @staticmethod
    def screen_plot_coordinates(rotated_vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rx, ry, rz = rotated_vertices[0], rotated_vertices[1], rotated_vertices[2]
        plot_x = -ry
        plot_y = rz
        plot_z = rx
        return plot_x, plot_y, plot_z

    def draw_cuboid_on_ax(self, axis: object, rotated_vertices: np.ndarray, color: str, alpha: float) -> None:
        plot_x, plot_y, plot_z = self.screen_plot_coordinates(rotated_vertices)
        faces_indices = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 3, 7, 4],
            [1, 2, 6, 5],
        ]
        faces_vertices = [
            [(plot_x[idx], plot_y[idx], plot_z[idx]) for idx in face]
            for face in faces_indices
        ]
        cuboid_faces = Poly3DCollection(faces_vertices)
        cuboid_faces.set_facecolors(color)
        cuboid_faces.set_alpha(alpha)
        cuboid_faces.set_edgecolors("black")
        axis.add_collection3d(cuboid_faces)

    def draw_fcc_lattice_on_ax(self, axis: object, rotated_points: np.ndarray) -> None:
        plot_x, plot_y, plot_z = self.screen_plot_coordinates(rotated_points)
        axis.scatter(plot_x, plot_y, plot_z, s=155, c="#0ABAB5", edgecolors="black", linewidths=0.8, depthshade=True)
        edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        for start, end in edges:
            axis.plot(
                [plot_x[start], plot_x[end]],
                [plot_y[start], plot_y[end]],
                [plot_z[start], plot_z[end]],
                color="#777777",
                linewidth=1.0,
                linestyle="--",
            )

    def draw_bcc_reciprocal_on_ax(self, axis: object, rotated_points: np.ndarray) -> None:
        plot_x, plot_y, plot_z = self.screen_plot_coordinates(rotated_points)
        sizes = [160] + [95] * 8 + [75] * 6
        axis.scatter(plot_x, plot_y, plot_z, s=sizes, c="#9B59B6", edgecolors="black", linewidths=0.8, depthshade=True)
        for idx in range(9, 15):
            axis.plot(
                [plot_x[0], plot_x[idx]],
                [plot_y[0], plot_y[idx]],
                [plot_z[0], plot_z[idx]],
                color="#777777",
                linewidth=1.0,
                linestyle=":",
            )

    def _holder_visual_matrices(self, in_plane_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        alpha = -math.radians(self.alpha_deg)
        beta = -math.radians(self.beta_deg)
        gamma = -math.radians(in_plane_deg)

        rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(alpha), -math.sin(alpha)],
                [0, math.sin(alpha), math.cos(alpha)],
            ],
            dtype=float,
        )
        ry = np.array(
            [
                [math.cos(beta), 0, math.sin(beta)],
                [0, 1, 0],
                [-math.sin(beta), 0, math.cos(beta)],
            ],
            dtype=float,
        )
        rz = np.array(
            [
                [math.cos(gamma), -math.sin(gamma), 0],
                [math.sin(gamma), math.cos(gamma), 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        return rx, ry, rz

    def _compose_holder_visual_rotation(self, rx: np.ndarray, ry: np.ndarray, rz: np.ndarray) -> np.ndarray:
        if self.context.holder_order == "yx":
            return ry @ rx @ rz
        return rx @ ry @ rz

    def _crystal_display_points(self, crystal_points: np.ndarray) -> np.ndarray:
        zero_holder_points = self.crystal_to_zero_holder @ crystal_points
        sample_rotated_points = zone_axis_finder.rotate_zero_holder_in_plane(
            zero_holder_points,
            self.sample_ccw_deg,
        )
        tilted_points = zone_axis_finder.holder_rotation(
            self.alpha_deg,
            self.beta_deg,
            self.context.holder_order,
        ) @ sample_rotated_points
        # The microscope screen convention flips the transverse view, but this
        # display-only correction must not change the physical tilt solution.
        return self.crystal_display_correction @ tilted_points

    def _current_reciprocal_vectors_in_holder(self, lattice_parameter_nm: float) -> np.ndarray:
        g_crystal = self.diffraction_hkl / lattice_parameter_nm
        zero_holder_points = self.crystal_to_zero_holder @ g_crystal
        sample_rotated_points = zone_axis_finder.rotate_zero_holder_in_plane(
            zero_holder_points,
            self.sample_ccw_deg,
        )
        return zone_axis_finder.holder_rotation(
            self.alpha_deg,
            self.beta_deg,
            self.context.holder_order,
        ) @ sample_rotated_points

    def _holder_to_image_plane(self, holder_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phi = math.radians(self.context.image_to_holder_rotation_deg)
        holder_x = holder_vectors[0]
        holder_y = holder_vectors[1]
        image_x = math.cos(phi) * holder_x - math.sin(phi) * holder_y
        image_y = math.sin(phi) * holder_x + math.cos(phi) * holder_y
        return image_x, image_y

    def _draw_diffraction_error(self, message: str) -> None:
        ax = self.diffraction_ax
        ax.clear()
        ax.set_facecolor("black")
        self.diffraction_fig.patch.set_facecolor("black")
        ax.text(
            0.5,
            0.5,
            message,
            color="#e8faff",
            ha="center",
            va="center",
            transform=ax.transAxes,
            wrap=True,
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.diffraction_canvas.draw()

    def _update_diffraction_pattern(self) -> None:
        try:
            voltage_kv = self._positive_float_from_var(self.accelerating_voltage_var, "Voltage")
            lattice_parameter_nm = self._positive_float_from_var(self.lattice_parameter_var, "Lattice parameter")
            thickness_nm = self._positive_float_from_var(self.sample_thickness_var, "Sample thickness")
            max_index = self._diffraction_max_index_from_var()
        except ValueError as exc:
            self._draw_diffraction_error(str(exc))
            return

        self._ensure_diffraction_hkl(max_index)
        wavelength_nm = self._electron_wavelength_nm(voltage_kv)
        holder_g = self._current_reciprocal_vectors_in_holder(lattice_parameter_nm)
        image_gx, image_gy = self._holder_to_image_plane(holder_g)
        g_perp2 = holder_g[0] * holder_g[0] + holder_g[1] * holder_g[1]
        excitation_error = holder_g[2] + 0.5 * wavelength_nm * g_perp2

        relrod_method = self.relrod_method_var.get().strip().lower()
        if relrod_method.startswith("gauss"):
            relrod_sigma = max(0.0025, 2.0 / thickness_nm)
            relrod_intensity = np.exp(-0.5 * (excitation_error / relrod_sigma) ** 2)
            relrod_label = "Gaussian"
        else:
            relrod_intensity = np.sinc(thickness_nm * excitation_error) ** 2
            relrod_label = "sinc"
        hkl_norm2 = np.sum(self.diffraction_hkl * self.diffraction_hkl, axis=0)
        form_factor = np.exp(-0.010 * hkl_norm2)
        intensity = np.clip(relrod_intensity * form_factor, 0.0, 1.0)

        camera_length_mm = self._camera_length_mm()
        screen_x_mm = camera_length_mm * wavelength_nm * image_gx
        screen_y_mm = camera_length_mm * wavelength_nm * image_gy
        screen_half_width_mm = 24.0
        visible = (
            (intensity > 0.012)
            & (screen_x_mm >= -screen_half_width_mm * 1.08)
            & (screen_x_mm <= screen_half_width_mm * 1.08)
            & (screen_y_mm >= -screen_half_width_mm * 1.08)
            & (screen_y_mm <= screen_half_width_mm * 1.08)
        )

        ax = self.diffraction_ax
        ax.clear()
        ax.set_facecolor("black")
        self.diffraction_fig.patch.set_facecolor("black")
        ax.set_xlim(-screen_half_width_mm, screen_half_width_mm)
        ax.set_ylim(-screen_half_width_mm, screen_half_width_mm)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, color="white", alpha=0.10, linewidth=0.8, linestyle="--")
        ax.axvline(0, color="white", alpha=0.10, linewidth=0.8, linestyle="--")

        if np.any(visible):
            spot_intensity = intensity[visible]
            colors = [
                (0.62, 0.98, 1.0, float(min(1.0, 0.20 + 0.80 * value)))
                for value in spot_intensity
            ]
            sizes = 10.0 + 210.0 * np.power(spot_intensity, 0.72)
            ax.scatter(
                screen_x_mm[visible],
                screen_y_mm[visible],
                s=sizes,
                c=colors,
                edgecolors="none",
                zorder=3,
            )
        ax.scatter([0.0], [0.0], s=135, c="#fff1a8", edgecolors="#fffbe8", linewidths=0.8, zorder=4)
        ax.text(
            0.5,
            0.025,
            f"Alpha: {self.alpha_deg:.2f} deg    Beta: {self.beta_deg:.2f} deg",
            transform=ax.transAxes,
            color="#bdefff",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.diffraction_canvas.draw()
        self._write_output()

    def update_plot(self) -> None:
        lamella_rx, lamella_ry, lamella_rz = self._holder_visual_matrices(self.actual_sample_rotation_deg())
        lamella_visual_rotation = self._compose_holder_visual_rotation(lamella_rx, lamella_ry, lamella_rz)

        for axis in self.crystal_axes:
            axis.clear()
            axis.set_facecolor(self.plot_bg_hex)

        self.ax.set_title("TEM lamella", color="#333333", fontsize=10, pad=2)
        self.real_ax.set_title("Real Space", color="#333333", fontsize=10, pad=2)
        self.recip_ax.set_title("Reciprocal Space", color="#333333", fontsize=10, pad=2)

        rot_lamella = lamella_visual_rotation @ self.base_lamella
        rot_capping = lamella_visual_rotation @ self.base_capping
        self.draw_cuboid_on_ax(self.ax, rot_lamella, color="#2ecc71", alpha=0.3)
        self.draw_cuboid_on_ax(self.ax, rot_capping, color="#f1c40f", alpha=0.5)

        self.draw_fcc_lattice_on_ax(self.real_ax, self._crystal_display_points(self.base_fcc))
        self.draw_bcc_reciprocal_on_ax(self.recip_ax, self._crystal_display_points(self.base_bcc))

        axis_length = 12.0
        base_y_axis = np.array([0, axis_length, 0], dtype=float)
        dynamic_y_axis = lamella_rx @ base_y_axis
        dyn_plot_x = -dynamic_y_axis[1]
        dyn_plot_y = dynamic_y_axis[2]
        dyn_plot_z = dynamic_y_axis[0]

        self.ax.plot([], [], [], color="blue", linewidth=2, label="X-axis (Alpha)")
        self.ax.plot([], [], [], color="pink", linewidth=2, label="Y-axis (Beta)")
        self.ax.plot([], [], [], color="gray", linestyle="dashed", linewidth=1, label="Initial Y-axis")

        if self.snap_crystal_view_var.get():
            self.current_elev = 0.0
            self.current_azim = -90.0
            self.current_roll = 0.0

        for axis in self.crystal_axes:
            show_axis_guides = axis == self.ax or self.show_crystal_axes_var.get()
            if show_axis_guides:
                axis.quiver(0, 0, 0, 0, 0, axis_length, color="blue", linewidth=2, arrow_length_ratio=0.10)
                axis.quiver(
                    0,
                    0,
                    0,
                    dyn_plot_x,
                    dyn_plot_y,
                    dyn_plot_z,
                    color="pink",
                    linewidth=2,
                    arrow_length_ratio=0.10,
                )
                axis.plot([0, -axis_length], [0, 0], [0, 0], color="gray", linestyle="dashed", linewidth=1)
            axis.set_box_aspect([1, 1, 1])
            axis.set_xlim([-12.5, 12.5])
            axis.set_ylim([-12.5, 12.5])
            axis.set_zlim([-12.5, 12.5])
            if hasattr(axis, "view_init"):
                if hasattr(axis, "roll"):
                    axis.view_init(elev=self.current_elev, azim=self.current_azim, roll=self.current_roll)
                else:
                    axis.view_init(elev=self.current_elev, azim=self.current_azim)
            axis.axis("off")

        self.ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.018, 0.965),
            bbox_transform=self.fig.transFigure,
            fontsize=8,
            borderaxespad=0.0,
        )
        self._update_footer_texts()
        self.canvas.draw()
        self._update_pole_figure()
        self._update_diffraction_pattern()

    def _update_footer_texts(self) -> None:
        for text in self.footer_texts:
            with contextlib.suppress(ValueError):
                text.remove()
        self.footer_texts = [
            self.fig.text(
                0.06,
                0.065,
                f"Image to holder: {self.context.image_to_holder_rotation_deg:.1f} deg",
                ha="left",
                va="bottom",
                fontsize=9,
            ),
            self.fig.text(
                0.49,
                0.065,
                f"In-plane rotation offset: {self.offset_deg:.2f} deg",
                ha="left",
                va="bottom",
                fontsize=9,
            ),
        ]

    def _pole_points(self) -> list[zone_axis_finder.ZoneAxisMapPoint]:
        points = zone_axis_finder.sample_rotation_map_points(
            self.context.match,  # type: ignore[arg-type]
            self.context.crystal_to_zero_holder,  # type: ignore[arg-type]
            ccw_deg=self.sample_ccw_deg,
            target_families=self.context.target_families,
            include_opposites=self.context.include_opposites,
            holder_order=self.context.holder_order,
            alpha_limits=self.context.alpha_limits,
            beta_limits=self.context.beta_limits,
        )
        if self.context.map_reachable_only:
            points = zone_axis_finder.filter_reachable_zone_axis_map_points(
                points,
                self.context.crystal_to_zero_holder,  # type: ignore[arg-type]
                self.context.holder_order,
                self.context.alpha_limits,
                self.context.beta_limits,
            )
        return points

    def _update_pole_figure(self) -> None:
        points = self._pole_points()
        axis_range = self._pole_axis_range(points)
        ax = self.pole_ax
        ax.clear()
        ax.set_facecolor("black")
        self.pole_fig.patch.set_facecolor("black")
        ax.set_xlim(-axis_range, axis_range)
        ax.set_ylim(axis_range, -axis_range)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        self._draw_pole_background(ax, axis_range)
        for point in points:
            x = self.alpha_deg - point.alpha_deg
            y = point.beta_deg - self.beta_deg
            fill = zone_axis_finder.ZONE_FAMILY_COLORS.get(point.family, (80, 80, 80))
            outline = zone_axis_finder.zone_outline_color(point.zone)
            fill_rgb = tuple(channel / 255.0 for channel in fill)
            outline_rgb = tuple(channel / 255.0 for channel in outline)
            alpha_value = 1.0 if point.within_limits else 0.45
            ax.scatter(
                [x],
                [y],
                s=84,
                marker="o",
                c=[fill_rgb],
                edgecolors=[outline_rgb],
                linewidths=2.2,
                alpha=alpha_value,
                zorder=4,
            )
            label = zone_axis_finder.format_miller(point.zone)
            label_rgb = outline_rgb if self.context.map_label_individual_color else fill_rgb
            stroke_rgb: str | tuple[float, float, float] = "white" if self.context.map_label_individual_color else outline_rgb
            text = ax.text(
                x + axis_range * 0.025,
                y - axis_range * 0.018,
                label,
                color=label_rgb,
                fontsize=9,
                fontweight="bold",
                zorder=5,
            )
            text.set_path_effects([path_effects.withStroke(linewidth=2.6, foreground=stroke_rgb)])

        center_span = axis_range * 0.035
        ax.plot([-center_span, center_span], [0, 0], color="white", linewidth=1.3, zorder=6)
        ax.plot([0, 0], [-center_span, center_span], color="white", linewidth=1.3, zorder=6)
        ax.text(
            0.5,
            0.025,
            f"Alpha: {self.alpha_deg:.2f} deg    Beta: {self.beta_deg:.2f} deg",
            transform=ax.transAxes,
            color="#e8faff",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=7,
        )
        self.pole_canvas.draw()

    def _pole_axis_range(self, points: Sequence[zone_axis_finder.ZoneAxisMapPoint]) -> float:
        values = [
            abs(self.context.alpha_limits[0]),
            abs(self.context.alpha_limits[1]),
            abs(self.context.beta_limits[0]),
            abs(self.context.beta_limits[1]),
            abs(self.alpha_deg),
            abs(self.beta_deg),
            18.0,
        ]
        for point in points:
            values.extend([abs(point.alpha_deg), abs(point.beta_deg)])
        return min(85.0, max(values) + 8.0)

    def _draw_pole_background(self, ax: object, axis_range: float) -> None:
        circle_radius = axis_range * 0.88
        theta = np.linspace(0, 2.0 * math.pi, 360)
        ax.plot(circle_radius * np.cos(theta), circle_radius * np.sin(theta), color="#555555", linewidth=1.4)
        for fraction in (0.33, 0.66):
            radius = circle_radius * fraction
            ax.plot(radius * np.cos(theta), radius * np.sin(theta), color="#272727", linewidth=0.8)
        ax.plot([-circle_radius, circle_radius], [0, 0], color="#333333", linewidth=0.9)
        ax.plot([0, 0], [-circle_radius, circle_radius], color="#333333", linewidth=0.9)

    def close(self) -> None:
        self._stop_hold()
        self.destroy()


class SampleRotationSimulator(tk.Toplevel):
    canvas_size = (560, 760)
    map_size = (520, 520)

    def __init__(self, parent: tk.Tk, context: AnalysisContext) -> None:
        super().__init__(parent)
        self.title("Sample Rotation Simulator")
        self.output_text_height = 10
        self._configure_window_for_screen()
        self.context = context
        self.absolute_angle_deg = 0.0
        self.zero_offset_deg = 0.0
        self.drag_start_mouse_angle = 0.0
        self.drag_start_absolute_angle = 0.0
        self.angle_var = StringVar(value="0.00")
        self.current_var = StringVar(value="CCW rotation: 0.00 deg")
        self.offset_var = StringVar(value="Offset in-plane rotation: 0.00 deg")
        self.holder_photo: ImageTk.PhotoImage | None = None
        self.lamella_photo: ImageTk.PhotoImage | None = None
        self.dynamic_map_photo: ImageTk.PhotoImage | None = None
        self.lamella_item: int | None = None
        self.tilt_simulator_window: SampleTiltSimulator | None = None
        self.lamella_center = (self.canvas_size[0] / 2.0, self.canvas_size[1] * 0.38)
        self.holder_image, self.lamella_base = self._load_simulator_images()
        self.map_alpha_range, self.map_beta_range = self._compute_dynamic_map_ranges()

        self._build_ui()
        self._draw_static_holder()
        self._render_lamella()
        self._update_dynamic_map()
        self._update_reachable_output()

    def _configure_window_for_screen(self) -> None:
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        window_w = min(1260, max(760, screen_w - 80))
        window_h = min(900, max(620, screen_h - 120))

        canvas_h = max(480, min(760, window_h - 60))
        canvas_w = max(354, int(canvas_h * 560 / 760))
        self.canvas_size = (canvas_w, canvas_h)

        map_side = max(260, min(520, int((window_h - 360) * 0.95)))
        self.map_size = (map_side, map_side)
        self.output_text_height = max(6, min(10, int((window_h - map_side - 250) / 18)))

        self.geometry(f"{window_w}x{window_h}")
        self.minsize(min(900, window_w), min(620, window_h))

    @staticmethod
    def _normalize_angle(angle_deg: float) -> float:
        angle = (angle_deg + 180.0) % 360.0 - 180.0
        return angle + 360.0 if angle <= -180.0 else angle

    def _load_simulator_images(self) -> tuple[Image.Image, Image.Image]:
        holder_path = asset_path("Sample holder object.png")
        lamella_path = asset_path("TEM lamella object.png")
        max_w, max_h = self.canvas_size
        holder = Image.open(holder_path).convert("RGBA")
        scale = min(max_w / holder.width, max_h / holder.height)
        holder_size = (max(1, int(holder.width * scale)), max(1, int(holder.height * scale)))
        holder = holder.resize(holder_size, Image.Resampling.LANCZOS)

        lamella = Image.open(lamella_path).convert("RGBA")
        lamella_size = (max(1, int(lamella.width * scale)), max(1, int(lamella.height * scale)))
        lamella = lamella.resize(lamella_size, Image.Resampling.LANCZOS)

        holder_left = (max_w - holder.width) / 2.0
        holder_top = (max_h - holder.height) / 2.0
        self.lamella_center = (holder_left + holder.width * 0.50, holder_top + holder.height * 0.385)
        return holder, lamella

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        canvas_frame = ttk.Frame(self, padding=12)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_size[0],
            height=self.canvas_size[1],
            bg="#d9d9d9",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        canvas_footer = ttk.Frame(canvas_frame)
        canvas_footer.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        canvas_footer.columnconfigure(0, weight=1)
        ttk.Label(canvas_footer, textvariable=self.offset_var).grid(row=0, column=0, sticky="w")
        ttk.Button(canvas_footer, text="Open tilt simulator", command=self.open_tilt_simulator).grid(
            row=0, column=1, sticky="e", padx=(10, 0)
        )

        side = ttk.Frame(self, padding=(0, 12, 12, 12))
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        side.rowconfigure(2, weight=1)

        controls = ttk.LabelFrame(side, text="Rotation Control", style="Section.TLabelframe", padding=10)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, textvariable=self.current_var).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(controls, text="CCW deg").grid(row=1, column=0, sticky="w", pady=(8, 0))
        entry = ttk.Entry(controls, textvariable=self.angle_var, width=12)
        entry.grid(row=1, column=1, sticky="ew", padx=(6, 8), pady=(8, 0))
        entry.bind("<Return>", lambda _event: self.apply_entry_angle())
        ttk.Button(controls, text="Apply", command=self.apply_entry_angle).grid(row=1, column=2, sticky="ew", pady=(8, 0))
        ttk.Button(controls, text="Set as starting point", command=self.set_starting_point).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0)
        )
        ttk.Button(controls, text="Reset position", command=self.reset_position).grid(
            row=2, column=2, columnspan=2, sticky="ew", padx=(8, 0), pady=(10, 0)
        )
        ttk.Button(controls, text="Reset starting point", command=self.reset_starting_point).grid(
            row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0)
        )

        map_frame = ttk.LabelFrame(side, text="Dynamic Zone Axis Map", style="Section.TLabelframe", padding=8)
        map_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        map_frame.columnconfigure(0, weight=1)
        self.dynamic_map_label = ttk.Label(map_frame, anchor="center")
        self.dynamic_map_label.grid(row=0, column=0, sticky="ew")

        summary = ttk.LabelFrame(side, text="Reachable Target Axes", style="Section.TLabelframe", padding=8)
        summary.grid(row=2, column=0, sticky="nsew")
        summary.columnconfigure(0, weight=1)
        summary.rowconfigure(0, weight=1)
        self.output_text = tk.Text(summary, wrap="none", width=58, height=self.output_text_height)
        y_scroll = ttk.Scrollbar(summary, orient="vertical", command=self.output_text.yview)
        x_scroll = ttk.Scrollbar(summary, orient="horizontal", command=self.output_text.xview)
        self.output_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.output_text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

    def _draw_static_holder(self) -> None:
        self.holder_photo = ImageTk.PhotoImage(self.holder_image)
        self.canvas.create_image(
            self.canvas_size[0] / 2.0,
            self.canvas_size[1] / 2.0,
            image=self.holder_photo,
            anchor="center",
        )

    def _render_lamella(self) -> None:
        rotated = self.lamella_base.rotate(self.absolute_angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
        self.lamella_photo = ImageTk.PhotoImage(rotated)
        if self.lamella_item is None:
            self.lamella_item = self.canvas.create_image(
                self.lamella_center[0],
                self.lamella_center[1],
                image=self.lamella_photo,
                anchor="center",
            )
        else:
            self.canvas.itemconfigure(self.lamella_item, image=self.lamella_photo)

    def _dynamic_map_points(self, ccw_deg: float) -> list[zone_axis_finder.ZoneAxisMapPoint]:
        points = zone_axis_finder.sample_rotation_map_points(
            self.context.match,  # type: ignore[arg-type]
            self.context.crystal_to_zero_holder,  # type: ignore[arg-type]
            ccw_deg=ccw_deg,
            target_families=self.context.target_families,
            include_opposites=self.context.include_opposites,
            holder_order=self.context.holder_order,
            alpha_limits=self.context.alpha_limits,
            beta_limits=self.context.beta_limits,
        )
        if self.context.map_reachable_only:
            points = zone_axis_finder.filter_reachable_zone_axis_map_points(
                points,
                self.context.crystal_to_zero_holder,  # type: ignore[arg-type]
                self.context.holder_order,
                self.context.alpha_limits,
                self.context.beta_limits,
            )
        return points

    def _compute_dynamic_map_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        alpha_values = [self.context.alpha_limits[0], self.context.alpha_limits[1], 0.0]
        beta_values = [self.context.beta_limits[0], self.context.beta_limits[1], 0.0]
        for ccw_deg in range(-180, 181, 10):
            for point in self._dynamic_map_points(float(ccw_deg)):
                alpha_values.append(point.alpha_deg)
                beta_values.append(point.beta_deg)
        return (
            self._expand_axis_range(zone_axis_finder.padded_range(alpha_values)),
            self._expand_axis_range(zone_axis_finder.padded_range(beta_values)),
        )

    @staticmethod
    def _expand_axis_range(axis_range: tuple[float, float], fraction: float = 0.10) -> tuple[float, float]:
        low, high = axis_range
        pad = max(2.0, (high - low) * fraction)
        return low - pad, high + pad

    def _update_dynamic_map(self) -> None:
        points = self._dynamic_map_points(self.current_ccw_deg())
        image = zone_axis_finder.zone_axis_map_image(
            points,
            alpha_limits=self.context.alpha_limits,
            beta_limits=self.context.beta_limits,
            rotation_rows=None,
            map_label_individual_color=self.context.map_label_individual_color,
            reverse_alpha_axis=False,
            reverse_beta_axis=True,
            compact_axes=True,
            alpha_range=self.map_alpha_range,
            beta_range=self.map_beta_range,
            image_size=self.map_size,
            title="",
        )
        self.dynamic_map_photo = ImageTk.PhotoImage(image)
        self.dynamic_map_label.configure(image=self.dynamic_map_photo)

    def _mouse_angle(self, event: tk.Event) -> float:
        cx, cy = self.lamella_center
        return math.degrees(math.atan2(cy - float(event.y), float(event.x) - cx))

    def _on_mouse_down(self, event: tk.Event) -> None:
        self.drag_start_mouse_angle = self._mouse_angle(event)
        self.drag_start_absolute_angle = self.absolute_angle_deg

    def _on_mouse_drag(self, event: tk.Event) -> None:
        delta = self._normalize_angle(self._mouse_angle(event) - self.drag_start_mouse_angle)
        self.absolute_angle_deg = self.drag_start_absolute_angle + delta
        self._after_angle_change()

    def current_ccw_deg(self) -> float:
        return self._normalize_angle(self.absolute_angle_deg - self.zero_offset_deg)

    def offset_ccw_deg(self) -> float:
        return self._normalize_angle(self.zero_offset_deg)

    def apply_entry_angle(self) -> None:
        try:
            ccw_deg = float(self.angle_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Rotation", "Enter a numeric CCW rotation angle.")
            return
        self.absolute_angle_deg = self.zero_offset_deg + ccw_deg
        self._after_angle_change()

    def set_starting_point(self) -> None:
        self.zero_offset_deg = self._normalize_angle(self.absolute_angle_deg)
        self.absolute_angle_deg = self.zero_offset_deg
        self._after_angle_change()

    def reset_position(self) -> None:
        self.absolute_angle_deg = self.zero_offset_deg
        self._after_angle_change()

    def reset_starting_point(self) -> None:
        self.zero_offset_deg = 0.0
        self.absolute_angle_deg = 0.0
        self._after_angle_change()

    def open_tilt_simulator(self) -> None:
        if self.tilt_simulator_window is not None and self.tilt_simulator_window.winfo_exists():
            self._sync_tilt_simulator()
            self.tilt_simulator_window.lift()
            self.tilt_simulator_window.focus_force()
            return
        self.tilt_simulator_window = SampleTiltSimulator(
            self,
            self.context,
            sample_ccw_deg=self.current_ccw_deg(),
            offset_deg=self.offset_ccw_deg(),
        )
        self.tilt_simulator_window.protocol("WM_DELETE_WINDOW", self._close_tilt_simulator)

    def _close_tilt_simulator(self) -> None:
        if self.tilt_simulator_window is not None and self.tilt_simulator_window.winfo_exists():
            self.tilt_simulator_window.close()
        self.tilt_simulator_window = None

    def _sync_tilt_simulator(self) -> None:
        if self.tilt_simulator_window is not None and self.tilt_simulator_window.winfo_exists():
            self.tilt_simulator_window.set_sample_rotation(
                self.current_ccw_deg(),
                self.offset_ccw_deg(),
            )

    def _after_angle_change(self) -> None:
        ccw_deg = self.current_ccw_deg()
        self.angle_var.set(f"{ccw_deg:.2f}")
        self.current_var.set(f"CCW rotation: {ccw_deg:.2f} deg")
        self.offset_var.set(f"Offset in-plane rotation: {self.offset_ccw_deg():.2f} deg")
        self._render_lamella()
        self._update_dynamic_map()
        self._update_reachable_output()
        self._sync_tilt_simulator()

    def _update_reachable_output(self) -> None:
        ccw_deg = self.current_ccw_deg()
        rows = zone_axis_finder.sample_rotation_reachable_rows(
            self.context.match,  # type: ignore[arg-type]
            self.context.crystal_to_zero_holder,  # type: ignore[arg-type]
            ccw_deg=ccw_deg,
            target_families=self.context.target_families,
            include_opposites=self.context.include_opposites,
            holder_order=self.context.holder_order,
            alpha_limits=self.context.alpha_limits,
            beta_limits=self.context.beta_limits,
        )
        lines = [
            f"Current CCW rotation: {ccw_deg:.2f} deg",
            "",
            f"{'family':<8} {'zone':<12} {'alpha':>9} {'beta':>9} {'angle':>9}",
            "-" * 52,
        ]
        if rows:
            for row in rows:
                lines.append(
                    f"{row['family']:<8} {row['zone']:<12} "
                    f"{float(row['alpha_deg']):9.3f} "
                    f"{float(row['beta_deg']):9.3f} "
                    f"{float(row['angle_from_current_deg']):9.3f}"
                )
        else:
            lines.append("(No selected target axes are reachable at this rotation.)")
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("end", "\n".join(lines) + "\n")
        self.output_text.see("1.0")

    def close(self) -> None:
        self._close_tilt_simulator()
        self.destroy()


class ZoneAxisFinderGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FCC Zone-Axis Finder")
        self.geometry("1580x840")
        self.minsize(1240, 700)

        self.result_queue: queue.Queue[tuple[str, str, PreviewResult]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.predicted_photo: ImageTk.PhotoImage | None = None
        self.fitted_photo: ImageTk.PhotoImage | None = None
        self.zone_map_photo: ImageTk.PhotoImage | None = None
        self.intro_photo: ImageTk.PhotoImage | None = None
        self.intro_rotation_photo: ImageTk.PhotoImage | None = None
        self.latest_previews = PreviewResult()
        self.download_buttons: dict[str, ttk.Button] = {}
        self.simulator_window: SampleRotationSimulator | None = None
        self.simulator_button: ttk.Button | None = None

        self.vars: dict[str, StringVar] = {}
        self.bool_vars: dict[str, BooleanVar] = {}
        self.family_vars: dict[str, BooleanVar] = {}

        self._configure_style()
        self._build_variables()
        self._build_ui()
        self._set_default_paths()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("TkDefaultFont", 14, "bold"))
        style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))

    def _build_variables(self) -> None:
        defaults = {
            "image": "",
            "alpha": "0",
            "beta": "0",
            "current_zone": "",
            "output_prefix": "",
            "alpha_min": "-35",
            "alpha_max": "35",
            "beta_min": "-20",
            "beta_max": "20",
            "holder_order": "xy",
            "image_to_holder_rotation_deg": "90",
            "center_x": "",
            "center_y": "",
            "n_peaks": "120",
            "min_distance_px": "",
            "spot_sigma_px": "",
            "peak_percentile": "99.0",
            "max_index": "8",
            "max_g_norm": "9.0",
            "tolerance_fraction": "0.18",
            "kikuchi_max_g_norm": "0",
        }
        self.vars = {name: StringVar(value=value) for name, value in defaults.items()}
        self.bool_vars = {
            "include_opposites": BooleanVar(value=False),
            "invert": BooleanVar(value=False),
            "rotate_pattern_180": BooleanVar(value=False),
            "map_show_target_families": BooleanVar(value=True),
            "map_label_individual_color": BooleanVar(value=True),
            "map_reachable_only": BooleanVar(value=True),
            "show_in_plane_rotation_predictions": BooleanVar(value=False),
            "show_labels": BooleanVar(value=True),
            "show_kikuchi_guides": BooleanVar(value=True),
            "export_target_csv": BooleanVar(value=False),
            "export_indexed_spots": BooleanVar(value=False),
            "export_predicted_pattern": BooleanVar(value=False),
            "export_fitted_pattern": BooleanVar(value=False),
        }
        default_target_families = {"100", "110", "111"}
        self.family_vars = {
            family: BooleanVar(value=family in default_target_families)
            for family in zone_axis_finder.SUPPORTED_ZONE_FAMILIES
        }

    def _set_default_paths(self) -> None:
        self.input_preview.configure(text="Choose an input diffraction image.", image="")

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=2)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=12)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(1, weight=1)

        header = ttk.Frame(left)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        title = ttk.Label(header, text="FCC Zone-Axis Finder", style="Title.TLabel")
        title.grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Introduction / Help", command=self.show_introduction).grid(row=0, column=1, sticky="e")

        notebook = ttk.Notebook(left)
        notebook.grid(row=1, column=0, sticky="nsew")

        basic_scroll = ScrollFrame(notebook)
        advanced_scroll = ScrollFrame(notebook)
        notebook.add(basic_scroll, text="Basic")
        notebook.add(advanced_scroll, text="Advanced")

        self._build_basic_tab(basic_scroll.inner)
        self._build_advanced_tab(advanced_scroll.inner)

        button_row = ttk.Frame(left)
        button_row.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        button_row.columnconfigure(0, weight=1)
        ttk.Checkbutton(
            button_row,
            text="Rotate pattern 180 deg",
            variable=self.bool_vars["rotate_pattern_180"],
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.run_button = ttk.Button(button_row, text="Run Analysis", command=self.run_analysis)
        self.run_button.grid(row=1, column=0, sticky="ew")
        ttk.Button(button_row, text="Show Command", command=self.show_command).grid(row=1, column=1, padx=(8, 0))

        self.status_var = StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status_var).grid(row=3, column=0, sticky="ew", pady=(8, 0))

        center = ttk.Frame(self, padding=(0, 12, 8, 12))
        center.grid(row=0, column=1, sticky="nsew")
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)

        preview_tabs = ttk.Notebook(center)
        preview_tabs.grid(row=0, column=0, sticky="nsew")
        input_frame = ttk.Frame(preview_tabs)
        predicted_frame = ttk.Frame(preview_tabs)
        fitted_frame = ttk.Frame(preview_tabs)
        zone_map_frame = ttk.Frame(preview_tabs)
        simulator_frame = ttk.Frame(preview_tabs)
        preview_tabs.add(input_frame, text="Input Image")
        preview_tabs.add(predicted_frame, text="Predicted Zone Pattern")
        preview_tabs.add(fitted_frame, text="Fitted Diffraction Pattern")
        preview_tabs.add(zone_map_frame, text="Zone Axis Map")
        preview_tabs.add(simulator_frame, text="Sample Simulators")
        input_frame.rowconfigure(0, weight=1)
        input_frame.columnconfigure(0, weight=1)
        predicted_frame.rowconfigure(1, weight=1)
        predicted_frame.columnconfigure(0, weight=1)
        fitted_frame.rowconfigure(1, weight=1)
        fitted_frame.columnconfigure(0, weight=1)
        zone_map_frame.rowconfigure(1, weight=1)
        zone_map_frame.columnconfigure(0, weight=1)
        simulator_frame.rowconfigure(0, weight=1)
        simulator_frame.columnconfigure(0, weight=1)

        self.input_preview = ttk.Label(input_frame, anchor="center")
        self.input_preview.grid(row=0, column=0, sticky="nsew")
        self._build_download_bar(predicted_frame, "predicted")
        self._build_download_bar(fitted_frame, "fitted")
        self._build_download_bar(zone_map_frame, "map")
        self.predicted_preview = ttk.Label(predicted_frame, anchor="center")
        self.predicted_preview.grid(row=1, column=0, sticky="nsew")
        self.fitted_preview = ttk.Label(fitted_frame, anchor="center")
        self.fitted_preview.grid(row=1, column=0, sticky="nsew")
        self.zone_map_preview = ttk.Label(zone_map_frame, anchor="center")
        self.zone_map_preview.grid(row=1, column=0, sticky="nsew")
        self._build_simulator_tab(simulator_frame)

        output_col = ttk.Frame(self, padding=(0, 12, 12, 12))
        output_col.grid(row=0, column=2, sticky="nsew")
        output_col.columnconfigure(0, weight=1)
        output_col.rowconfigure(0, weight=3)
        output_col.rowconfigure(1, weight=2)

        output_frame = ttk.LabelFrame(output_col, text="Main Output", style="Section.TLabelframe")
        output_frame.grid(row=0, column=0, sticky="nsew")
        self.output_text = self._build_output_text(output_frame, width=72, height=18)

        more_output_frame = ttk.LabelFrame(output_col, text="More Information", style="Section.TLabelframe")
        more_output_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.more_output_text = self._build_output_text(more_output_frame, width=72, height=10)

    def _build_download_bar(self, parent: ttk.Frame, kind: str) -> None:
        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        bar.columnconfigure(0, weight=1)
        button = ttk.Button(
            bar,
            text="Download",
            command=lambda selected=kind: self.download_preview_image(selected),
            state="disabled",
        )
        button.grid(row=0, column=1, sticky="e")
        self.download_buttons[kind] = button

    def _build_simulator_tab(self, parent: ttk.Frame) -> None:
        content = ttk.Frame(parent, padding=18)
        content.grid(row=0, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        button_box = ttk.Frame(content)
        button_box.grid(row=0, column=0)
        self.simulator_button = ttk.Button(
            button_box,
            text="Open sample rotation simulator",
            command=self.open_sample_rotation_simulator,
            state="disabled",
        )
        self.simulator_button.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            button_box,
            text="Run analysis first so the simulator can use the indexed zone-axis orientation.",
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def open_sample_rotation_simulator(self) -> None:
        context = self.latest_previews.analysis_context
        if context is None:
            messagebox.showinfo("Run Analysis First", "Run the analysis before opening the sample rotation simulator.")
            return
        if self.simulator_window is not None and self.simulator_window.winfo_exists():
            self.simulator_window.lift()
            self.simulator_window.focus_force()
            return
        self.simulator_window = SampleRotationSimulator(self, context)
        self.simulator_window.protocol("WM_DELETE_WINDOW", self._close_sample_rotation_simulator)

    def _close_sample_rotation_simulator(self) -> None:
        if self.simulator_window is not None and self.simulator_window.winfo_exists():
            self.simulator_window.close()
        self.simulator_window = None

    def _build_output_text(self, parent: ttk.LabelFrame, width: int, height: int) -> tk.Text:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        text = tk.Text(parent, wrap="none", width=width, height=height)
        y_scroll = ttk.Scrollbar(parent, orient="vertical", command=text.yview)
        x_scroll = ttk.Scrollbar(parent, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        return text

    def _build_basic_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        row = 0

        image_box = ttk.LabelFrame(parent, text="Input", style="Section.TLabelframe", padding=10)
        image_box.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        image_box.columnconfigure(1, weight=1)
        ttk.Label(image_box, text="Image").grid(row=0, column=0, sticky="w")
        ttk.Entry(image_box, textvariable=self.vars["image"], width=44).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(image_box, text="Browse", command=self.browse_image).grid(row=0, column=2, sticky="ew")
        row += 1

        required = ttk.LabelFrame(parent, text="Required Holder Angles", style="Section.TLabelframe", padding=10)
        required.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        required.columnconfigure(1, weight=1)
        required.columnconfigure(3, weight=1)
        ttk.Label(required, text="Alpha deg").grid(row=0, column=0, sticky="w")
        ttk.Entry(required, textvariable=self.vars["alpha"], width=12).grid(row=0, column=1, sticky="ew", padx=(6, 14))
        ttk.Label(required, text="Beta deg").grid(row=0, column=2, sticky="w")
        ttk.Entry(required, textvariable=self.vars["beta"], width=12).grid(row=0, column=3, sticky="ew", padx=(6, 0))
        row += 1

        indexing = ttk.LabelFrame(parent, text="Indexing", style="Section.TLabelframe", padding=10)
        indexing.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        indexing.columnconfigure(1, weight=1)
        ttk.Label(indexing, text="Known current zone").grid(row=0, column=0, sticky="w")
        ttk.Entry(indexing, textvariable=self.vars["current_zone"]).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Label(indexing, text="Example: 1,0,0. Leave blank to auto-detect.").grid(
            row=1, column=1, sticky="w", pady=(3, 0)
        )
        row += 1

        targets = ttk.LabelFrame(parent, text="Target Families", style="Section.TLabelframe", padding=10)
        targets.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        for idx, family in enumerate(zone_axis_finder.SUPPORTED_ZONE_FAMILIES):
            ttk.Checkbutton(targets, text=f"<{family}>", variable=self.family_vars[family]).grid(
                row=idx // 3, column=idx % 3, sticky="w", padx=(0, 16), pady=2
            )
        ttk.Checkbutton(
            targets,
            text="Report opposite directions separately",
            variable=self.bool_vars["include_opposites"],
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            targets,
            text="Show target families only on the map",
            variable=self.bool_vars["map_show_target_families"],
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            targets,
            text="Show in-plane rotation predictions",
            variable=self.bool_vars["show_in_plane_rotation_predictions"],
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))
        row += 1

        output = ttk.LabelFrame(parent, text="Output", style="Section.TLabelframe", padding=10)
        output.grid(row=row, column=0, columnspan=3, sticky="ew")
        output.columnconfigure(1, weight=1)
        ttk.Label(output, text="Output prefix").grid(row=0, column=0, sticky="w")
        ttk.Entry(output, textvariable=self.vars["output_prefix"]).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(output, text="Choose", command=self.browse_output_prefix).grid(row=0, column=2, sticky="ew")
        ttk.Label(output, text="Leave blank to use the input image name. Tabs are generated even when all exports are unchecked.").grid(
            row=1, column=1, sticky="w", pady=(3, 8)
        )
        ttk.Checkbutton(output, text="Target angles CSV", variable=self.bool_vars["export_target_csv"]).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=2
        )
        ttk.Checkbutton(output, text="Indexed spots CSV", variable=self.bool_vars["export_indexed_spots"]).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=2
        )
        ttk.Checkbutton(output, text="Predicted zone pattern PNG", variable=self.bool_vars["export_predicted_pattern"]).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=2
        )
        ttk.Checkbutton(output, text="Fitted diffraction pattern PNG", variable=self.bool_vars["export_fitted_pattern"]).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=2
        )

    def _build_advanced_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        row = 0

        limits = ttk.LabelFrame(parent, text="Tilt Limits", style="Section.TLabelframe", padding=10)
        limits.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        for col in range(4):
            limits.columnconfigure(col, weight=1)
        self._labeled_entry(limits, "Alpha min", "alpha_min", 0, 0)
        self._labeled_entry(limits, "Alpha max", "alpha_max", 0, 2)
        self._labeled_entry(limits, "Beta min", "beta_min", 1, 0)
        self._labeled_entry(limits, "Beta max", "beta_max", 1, 2)
        row += 1

        holder = ttk.LabelFrame(parent, text="Holder Calibration", style="Section.TLabelframe", padding=10)
        holder.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        holder.columnconfigure(1, weight=1)
        holder.columnconfigure(3, weight=1)
        ttk.Label(holder, text="Holder order").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            holder,
            textvariable=self.vars["holder_order"],
            values=("xy", "yx"),
            state="readonly",
            width=8,
        ).grid(row=0, column=1, sticky="ew", padx=(6, 14))
        ttk.Label(holder, text="Image to holder deg").grid(row=0, column=2, sticky="w")
        ttk.Entry(holder, textvariable=self.vars["image_to_holder_rotation_deg"], width=12).grid(
            row=0, column=3, sticky="ew", padx=(6, 0)
        )
        row += 1

        spot = ttk.LabelFrame(parent, text="Spot Detection", style="Section.TLabelframe", padding=10)
        spot.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        for col in range(4):
            spot.columnconfigure(col, weight=1)
        self._labeled_entry(spot, "Center x", "center_x", 0, 0)
        self._labeled_entry(spot, "Center y", "center_y", 0, 2)
        self._labeled_entry(spot, "N peaks", "n_peaks", 1, 0)
        self._labeled_entry(spot, "Peak percentile", "peak_percentile", 1, 2)
        self._labeled_entry(spot, "Min distance px", "min_distance_px", 2, 0)
        self._labeled_entry(spot, "Spot sigma px", "spot_sigma_px", 2, 2)
        ttk.Checkbutton(spot, text="Invert image", variable=self.bool_vars["invert"]).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        row += 1

        matching = ttk.LabelFrame(parent, text="Reference Matching", style="Section.TLabelframe", padding=10)
        matching.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        for col in range(4):
            matching.columnconfigure(col, weight=1)
        self._labeled_entry(matching, "Max index", "max_index", 0, 0)
        self._labeled_entry(matching, "Max |g|", "max_g_norm", 0, 2)
        self._labeled_entry(matching, "Tolerance fraction", "tolerance_fraction", 1, 0)
        row += 1

        overlay = ttk.LabelFrame(parent, text="Overlay", style="Section.TLabelframe", padding=10)
        overlay.grid(row=row, column=0, sticky="ew")
        overlay.columnconfigure(1, weight=1)
        ttk.Checkbutton(overlay, text="Show Miller-index labels", variable=self.bool_vars["show_labels"]).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Checkbutton(
            overlay,
            text="Show Kikuchi guide lines",
            variable=self.bool_vars["show_kikuchi_guides"],
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(overlay, text="Kikuchi max |g|").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(overlay, textvariable=self.vars["kikuchi_max_g_norm"], width=12).grid(
            row=2, column=1, sticky="ew", padx=(6, 0), pady=(8, 0)
        )
        row += 1

        map_settings = ttk.LabelFrame(parent, text="Map Settings", style="Section.TLabelframe", padding=10)
        map_settings.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(
            map_settings,
            text="Color-code label using individual axis color",
            variable=self.bool_vars["map_label_individual_color"],
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            map_settings,
            text="Show reachable zone axes only (including sample rotation)",
            variable=self.bool_vars["map_reachable_only"],
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

    def show_introduction(self) -> None:
        window = tk.Toplevel(self)
        window.title("Introduction and Field Guide")
        window.geometry("980x760")
        window.minsize(780, 560)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)

        scroll = ScrollFrame(window)
        scroll.grid(row=0, column=0, sticky="nsew")
        content = scroll.inner
        content.columnconfigure(0, weight=1)

        row = 0
        ttk.Label(content, text="FCC Zone-Axis Finder: Introduction", style="Title.TLabel").grid(
            row=row, column=0, sticky="w", padx=14, pady=(14, 8)
        )
        row += 1

        schematic = asset_path("Schematic of the double-tilt holder and the zone-axis of the sample.png")
        if schematic.exists():
            self.intro_photo = self._make_photo(schematic, (920, 300))
            if self.intro_photo is not None:
                ttk.Label(content, image=self.intro_photo).grid(row=row, column=0, sticky="ew", padx=14, pady=(0, 8))
                row += 1

        row = self._intro_section(
            content,
            row,
            "What This Program Does",
            "This GUI starts from one FCC zone-axis diffraction pattern and the current double-tilt holder angles. "
            "It detects the diffraction spots, compares them with analytic FCC zone-axis reference "
            "patterns, indexes the current zone, and calculates the alpha and beta angles for other symmetry-related "
            "zone axes. The output table marks whether each target falls inside your holder limits.",
        )
        row = self._intro_section(
            content,
            row,
            "Alpha and Beta",
            "The schematic above shows the convention used by the default model. Alpha is treated as a rotation about "
            "the holder X axis, and beta is treated as a rotation about the holder Y axis. The default rotation order is "
            "R = Rx(alpha) @ Ry(beta), shown as xy in the Advanced tab. This is the usual serial-holder model: the alpha "
            "axis is the holder rod axis, while the beta axis is carried by the alpha tilt and therefore moves with it. "
            "The final sample orientation is determined by the final alpha and beta values rather than by the order in "
            "which the operator changed them. If a known calibration move comes out with the opposite sign or swapped "
            "behavior, adjust Holder order or Image to holder deg in the Advanced tab.",
        )
        rotation_schematic = asset_path("Schematic of the effect of sample rotation.png")
        if rotation_schematic.exists():
            self.intro_rotation_photo = self._make_photo(rotation_schematic, (920, 480))
            if self.intro_rotation_photo is not None:
                ttk.Label(content, image=self.intro_rotation_photo).grid(
                    row=row, column=0, sticky="ew", padx=14, pady=(0, 8)
                )
                row += 1
        row = self._intro_section(
            content,
            row,
            "Sample In-Plane Rotation",
            "When the lamella is rotated in the holder before loading, the crystal directions rotate within the holder "
            "XY plane. This changes how a target zone-axis direction is split between alpha and beta tilt. A target that "
            "is slightly outside the alpha limit may become reachable after a small loading rotation because some of the "
            "required tilt is transferred into beta. When Show in-plane rotation predictions is checked, the program reports "
            "these cases as out-of-limit targets reachable by sample in-plane rotation and marks the range endpoints on the "
            "Zone Axis Map. The dashed trace between each endpoint pair shows how the required alpha/beta values sweep "
            "as the sample loading rotation changes. The printed rotation range is counterclockwise when viewed along "
            "the holder -Z axis, matching the schematic convention. The Sample Simulators tab opens an interactive rotation simulator "
            "that lets you rotate only the TEM lamella object and immediately list which selected target axes are reachable "
            "at that loading angle. Its dynamic map updates the selected zone-axis points at the same time and keeps the alpha/beta "
            "axes and tilt-limit rectangle fixed. The simulator map uses the screen-view convention: positive alpha points "
            "right and positive beta points down. Set as starting point records the current absolute lamella rotation as the in-plane offset "
            "and defines that orientation as 0 deg for subsequent interactive rotation. Reset position returns to that "
            "user-defined 0 deg orientation. Reset starting point clears the saved offset and returns the lamella to the "
            "default 0 deg position.",
        )
        row = self._intro_section(
            content,
            row,
            "Sample Tilt Simulator",
            "The rotation simulator has an Open tilt simulator button below the holder canvas. This opens a separate "
            "screen-view tilt window with a 3D tilt view, a real-space FCC crystal view, a reciprocal-space BCC lattice view, "
            "a Dynamic Pole Figure, and a synchronized simulated diffraction pattern. The tilt view uses the same visual convention as the demonstration script: a green "
            "lamella, yellow capping layer, blue fixed alpha/rod axis, pink dynamic beta axis, and gray neutral-plane reference. "
            "The FCC and reciprocal lattices are mapped through the indexed crystal orientation, so they follow the same "
            "alpha/beta tilt and in-plane sample rotation as the lamella. The beta axis is redrawn after alpha "
            "tilt, while the alpha axis remains fixed. Snap crystal view returns the three 3D views to the calibrated front view "
            "after each update; Show holder axes in crystals toggles the alpha/beta reference axes on the crystal panels. The saved "
            "in-plane rotation offset from Set as starting point is shown on the plot and combined with the current interactive "
            "sample rotation for the 3D lamella. The Dynamic Pole Figure "
            "horizontally flips the screen-view points relative to the rotation-simulator map, so zone axes move right after "
            "positive alpha tilt and up after positive beta tilt. Holding an alpha or beta arrow button changes the tilt continuously "
            "using the selected Tilt speed until release or until the Advanced-tab tilt limit is reached. Use Fine and Coarse "
            "to shift the tilt-speed gear down or up. The alpha/beta entry boxes also "
            "accept the Enter key and retain two-decimal typed values. If manually entered alpha or beta values exceed the limits, the simulator opens a warning "
            "and keeps the previous valid tilt. The diffraction panel uses the same indexed orientation as the crystal and pole figure. "
            "Voltage sets the electron wavelength, lattice parameter sets the reciprocal-lattice scale, thickness controls the relrod broadening, "
            "Max hkl controls how many FCC reciprocal-lattice reflections are considered, the relrod model chooses either a finite-thickness "
            "sinc-squared relrod envelope or the broader Gaussian-like envelope, and camera length controls the diffraction-pattern magnification. "
            "Both relrod models are still multiplied by the same high-index form-factor falloff. The lattice parameter starts from the FCC Ni fallback value. "
            "If the image has a reciprocal-space scale bar, Detect scale bar estimates the line length in pixels; enter the printed value, "
            "such as 5 1/nm, and use Estimate lattice.",
        )
        row = self._intro_section(
            content,
            row,
            "Basic Tab",
            "Image: choose the experimental diffraction image. Filenames with spaces are fine.\n\n"
            "Alpha deg and Beta deg: enter the current microscope holder angles at which the input pattern was recorded.\n\n"
            "Known current zone: optional. Leave it blank to let the program choose among the supported FCC zone axes. "
            "Enter values like 1,0,0 or 1 1 0 if you already know the present zone and want to force that assignment.\n\n"
            "Target families: choose which FCC zone-axis families to calculate. The current list includes <100>, <110>, "
            "<111>, <102>, <103>, <104>, <112>, <113>, and <114>.\n\n"
            "Report opposite directions separately: normally [uvw] and [-u -v -w] are treated as the same physical zone-axis "
            "line. Check this if you want both signs listed separately.\n\n"
            "Show target families only on the map: if checked, the Zone Axis Map tab shows only the target families selected "
            "above, plus the current zone. If unchecked, the map shows all supported families from <100> to <114>.\n\n"
            "Show reachable zone axes only: in Map Settings, hides zone-axis points that cannot enter the alpha/beta tilt limits "
            "even after trying in-plane sample rotation. This affects the Zone Axis Map, Dynamic Zone Axis Map, and Dynamic Pole Figure.\n\n"
            "Show in-plane rotation predictions: if checked, the program reports out-of-limit target axes that can become "
            "reachable after sample loading rotation and marks their CCW range endpoints as stars on the Zone Axis Map.\n\n"
            "Output prefix: optional filename prefix for exported files. Leave it blank to use the input image name.\n\n"
            "Output file checkboxes: choose exactly which CSV/PNG files to export. They are all unchecked by default; "
            "the image tabs are still generated for viewing.\n\n"
            "Rotate pattern 180 deg: check this after a first run if the 2D diffraction-pattern symmetry appears to have "
            "chosen the opposite real-space indexing direction. The next run keeps the same spot fit but rotates the "
            "in-plane indexing by 180 deg before calculating target alpha/beta angles.",
        )
        row = self._intro_section(
            content,
            row,
            "Advanced: Tilt Limits and Calibration",
            "Alpha min/max and Beta min/max: the holder range used to mark target axes as reachable or not reachable. "
            "For your double-tilt holder the default values are alpha -35 to 35 deg and beta -20 to 20 deg.\n\n"
            "Holder order: controls the rotation sequence. Use xy for R = Rx(alpha) @ Ry(beta), which is the recommended "
            "default for many serial double-tilt TEM holders because the beta axis is carried by alpha tilt. Use yx for "
            "R = Ry(beta) @ Rx(alpha) only if your own calibration shows that convention fits better.\n\n"
            "Image to holder deg: the in-plane rotation from image +x to holder +X. The default is 90 deg for the TEM "
            "you commonly use. This is important because a diffraction pattern can be rotated by the camera. A single "
            "centrosymmetric diffraction pattern cannot determine this instrument calibration by itself.",
        )
        row = self._intro_section(
            content,
            row,
            "Advanced: Spot Detection",
            "Center x and Center y: optional direct-beam center in image pixels. Leave blank to let the program use the "
            "bright peak nearest the image center.\n\n"
            "N peaks: maximum number of bright peaks kept from the image.\n\n"
            "Peak percentile: brightness threshold for candidate spots. Lower it if spots are missed; raise it if noise "
            "or annotations are being picked up.\n\n"
            "Min distance px: minimum spacing between detected peaks. Increase it if one large spot is detected multiple "
            "times; decrease it for dense patterns.\n\n"
            "Spot sigma px: Gaussian smoothing radius for spot detection. Larger values favor broad spots; smaller values "
            "favor sharp spots.\n\n"
            "Invert image: use this when the diffraction spots are dark on a bright background.",
        )
        row = self._intro_section(
            content,
            row,
            "Advanced: Reference Matching and Overlay",
            "Max index: largest absolute h, k, or l included in the analytic FCC reference pattern.\n\n"
            "Max |g|: largest reciprocal-vector length included in the reference pattern.\n\n"
            "Tolerance fraction: matching tolerance as a fraction of the first-shell spot spacing. Increase it for distorted "
            "or noisy patterns; decrease it if wrong spots are being matched.\n\n"
            "Show Miller-index labels: draw hkl labels beside the predicted spots in the generated pattern images.\n\n"
            "Show Kikuchi guide lines: draw approximate Kikuchi lines. For each reflection g, the guide lines are drawn "
            "perpendicular to g through the midpoint between O and +g and the midpoint between O and -g.\n\n"
            "Kikuchi max |g|: controls which indexed reflections draw guide-line pairs. 0 means first shell only; larger "
            "values draw more lines.\n\n"
            "Color-code label using individual axis color: in Map Settings, check this to fill each zone-axis label on "
            "the maps and Dynamic Pole Figure with the individual [uvw] outline color. Leave it unchecked to fill labels "
            "with the zone-family color while using the individual [uvw] color as the text outline.\n\n"
            "Show reachable zone axes only: leaves map axes focused near the usable holder range by removing zone axes that "
            "cannot be reached by any combination of holder tilt and sample in-plane rotation.",
        )
        row = self._intro_section(
            content,
            row,
            "Outputs",
            "The Main Output box prints the best present-zone match and target alpha/beta angles that are inside the tilt "
            "limits. When enabled, it also prints out-of-limit target axes that can be recovered by sample in-plane rotation, "
            "including the useful counterclockwise loading-rotation ranges and the alpha/beta values at each range endpoint. "
            "The More Information box keeps alternative reference "
            "scores, remaining out-of-limit target axes, export messages, and notes. The Predicted Zone Pattern tab shows the ideal indexed FCC zone before rotation "
            "fitting. The Fitted Diffraction Pattern tab overlays the fitted predicted spots, optional labels, and optional "
            "Kikuchi guide lines on top of the experimental diffraction image. The Zone Axis Map tab plots alpha versus beta "
            "for the current zone and predicted zone axes, with dashed lines marking the tilt limits. Point fill color marks "
            "the zone family, while outline color distinguishes individual [uvw] directions within a family. By default, "
            "map label fill follows the family color and the label outline follows the individual [uvw] color; the Map "
            "Settings checkbox can switch labels to the individual-color fill style. When in-plane predictions are enabled, "
            "star markers show the CCW range endpoints and dashed traces show the sweep between each min/max pair. "
            "The Sample Simulators tab opens the interactive lamella rotation simulator after a successful analysis; "
            "that window includes a separate Open tilt simulator button. Its dynamic map intentionally omits the static "
            "map's current-zone cross, endpoint stars, and sweep traces. "
            "The generated-image tabs each include a Download button, so you can save one image at a "
            "time even when the export checkboxes are off. The Show Command button "
            "prints the equivalent terminal command for the current settings.",
        )

        ttk.Button(content, text="Close", command=window.destroy).grid(row=row, column=0, sticky="e", padx=14, pady=14)

    def _intro_section(self, parent: ttk.Frame, row: int, title: str, text: str) -> int:
        frame = ttk.LabelFrame(parent, text=title, style="Section.TLabelframe", padding=10)
        frame.grid(row=row, column=0, sticky="ew", padx=14, pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=text, wraplength=880, justify="left").grid(row=0, column=0, sticky="ew")
        return row + 1

    def _labeled_entry(self, parent: ttk.Frame, label: str, key: str, row: int, col: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.vars[key], width=12).grid(
            row=row, column=col + 1, sticky="ew", padx=(6, 14), pady=3
        )

    def browse_image(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(APP_DIR),
            title="Choose diffraction image",
            filetypes=[
                ("Image files", "*.bmp *.png *.jpg *.jpeg *.tif *.tiff"),
                ("All files", "*"),
            ],
        )
        if path:
            self.vars["image"].set(path)
            self._load_input_preview(Path(path))

    def browse_output_prefix(self) -> None:
        path = filedialog.asksaveasfilename(
            initialdir=str(APP_DIR),
            title="Choose output prefix",
            defaultextension="",
            filetypes=[("Prefix", "*")],
        )
        if path:
            self.vars["output_prefix"].set(path)

    def download_preview_image(self, kind: str) -> None:
        source, suffix, title = self._preview_source_for_kind(kind)
        if source is None:
            messagebox.showinfo("No Image Available", "Run the analysis before downloading this image.")
            return

        initial_dir, initial_file = self._download_default_location(suffix)
        path = filedialog.asksaveasfilename(
            initialdir=str(initial_dir),
            initialfile=initial_file,
            title=f"Download {title}",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*")],
        )
        if not path:
            return

        try:
            if isinstance(source, Image.Image):
                source.save(path)
            else:
                with Image.open(source) as image:
                    image.save(path)
        except Exception as exc:
            messagebox.showerror("Download Failed", f"Could not save the image:\n{exc}")
            return

        self.status_var.set(f"Saved {title}: {path}")

    def _preview_source_for_kind(self, kind: str) -> tuple[Image.Image | Path | None, str, str]:
        if kind == "predicted":
            return (
                self.latest_previews.predicted_image or self.latest_previews.predicted_path,
                "_predicted_zone_pattern.png",
                "Predicted Zone Pattern",
            )
        if kind == "fitted":
            return (
                self.latest_previews.fitted_image or self.latest_previews.fitted_path,
                "_fitted_diffraction_pattern.png",
                "Fitted Diffraction Pattern",
            )
        if kind == "map":
            return self.latest_previews.map_image, "_zone_axis_map.png", "Zone Axis Map"
        return None, ".png", "Image"

    def _download_default_location(self, suffix: str) -> tuple[Path, str]:
        output_prefix = self.vars["output_prefix"].get().strip()
        if output_prefix:
            prefix = Path(output_prefix)
            return prefix.parent if str(prefix.parent) else APP_DIR, f"{prefix.name}{suffix}"

        image_text = self.vars["image"].get().strip()
        if image_text:
            image_path = Path(image_text)
            return image_path.parent if str(image_path.parent) else APP_DIR, f"{image_path.stem}{suffix}"

        return APP_DIR, f"zone_axis{suffix}"

    def _load_input_preview(self, path: Path) -> None:
        self.preview_photo = self._make_photo(path, (760, 430))
        if self.preview_photo is None:
            self.input_preview.configure(text=f"Could not preview:\n{path}", image="")
            return
        self.input_preview.configure(image=self.preview_photo, text="")

    def _load_generated_previews(self, previews: PreviewResult) -> None:
        self._load_preview_target(
            self.predicted_preview,
            "predicted_photo",
            previews.predicted_image or previews.predicted_path,
            "No predicted zone pattern is available.",
        )
        self._load_preview_target(
            self.fitted_preview,
            "fitted_photo",
            previews.fitted_image or previews.fitted_path,
            "No fitted diffraction pattern is available.",
        )
        self._load_preview_target(
            self.zone_map_preview,
            "zone_map_photo",
            previews.map_image,
            "No zone-axis map is available.",
        )

    def _load_preview_target(
        self,
        label: ttk.Label,
        photo_attr: str,
        source: Image.Image | Path | None,
        empty_text: str,
    ) -> None:
        if source is None:
            label.configure(text=empty_text, image="")
            setattr(self, photo_attr, None)
            return
        photo = self._make_photo_from_source(source, (760, 430))
        if photo is None:
            label.configure(text=f"Could not preview:\n{source}", image="")
            setattr(self, photo_attr, None)
            return
        setattr(self, photo_attr, photo)
        label.configure(image=photo, text="")

    def _make_photo(self, path: Path, max_size: tuple[int, int]) -> ImageTk.PhotoImage | None:
        return self._make_photo_from_source(path, max_size)

    def _make_photo_from_source(
        self,
        source: Image.Image | Path,
        max_size: tuple[int, int],
    ) -> ImageTk.PhotoImage | None:
        try:
            if isinstance(source, Image.Image):
                image = source.copy().convert("RGB")
            else:
                image = Image.open(source).convert("RGB")
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image)
        except Exception:
            return None

    def build_settings(self) -> dict[str, object]:
        image = self.vars["image"].get().strip()
        if not image:
            raise ValueError("Please choose an input diffraction image.")
        image_path = Path(image)
        if not image_path.exists():
            raise ValueError(f"Input image does not exist:\n{image}")

        families = [family for family, var in self.family_vars.items() if var.get()]
        if not families:
            raise ValueError("Choose at least one target family.")

        current_zone_text = self.vars["current_zone"].get().strip()
        current_zone = zone_axis_finder.parse_miller(current_zone_text) if current_zone_text else None

        center_x = self.vars["center_x"].get().strip()
        center_y = self.vars["center_y"].get().strip()
        if center_x or center_y:
            if not (center_x and center_y):
                raise ValueError("--center needs both X and Y values.")
            center = (float(center_x), float(center_y))
        else:
            center = None

        output_prefix_text = self.vars["output_prefix"].get().strip()
        return {
            "image": image_path,
            "alpha": self._float_value("alpha"),
            "beta": self._float_value("beta"),
            "target_families": families,
            "current_zone": current_zone,
            "current_zone_text": current_zone_text,
            "alpha_limits": (self._float_value("alpha_min"), self._float_value("alpha_max")),
            "beta_limits": (self._float_value("beta_min"), self._float_value("beta_max")),
            "holder_order": self.vars["holder_order"].get().strip() or "xy",
            "image_to_holder_rotation_deg": self._float_value("image_to_holder_rotation_deg", 90.0),
            "center": center,
            "n_peaks": self._int_value("n_peaks", 80),
            "min_distance_px": self._optional_float_value("min_distance_px"),
            "spot_sigma_px": self._optional_float_value("spot_sigma_px"),
            "peak_percentile": self._float_value("peak_percentile", 99.0),
            "invert": self.bool_vars["invert"].get(),
            "rotate_pattern_180": self.bool_vars["rotate_pattern_180"].get(),
            "map_show_target_families": self.bool_vars["map_show_target_families"].get(),
            "map_label_individual_color": self.bool_vars["map_label_individual_color"].get(),
            "map_reachable_only": self.bool_vars["map_reachable_only"].get(),
            "max_index": self._int_value("max_index", 8),
            "max_g_norm": self._float_value("max_g_norm", 9.0),
            "tolerance_fraction": self._float_value("tolerance_fraction", 0.18),
            "include_opposites": self.bool_vars["include_opposites"].get(),
            "output_prefix": Path(output_prefix_text) if output_prefix_text else None,
            "show_labels": self.bool_vars["show_labels"].get(),
            "show_kikuchi_guides": self.bool_vars["show_kikuchi_guides"].get(),
            "kikuchi_max_g_norm": self._float_value("kikuchi_max_g_norm", 0.0),
            "show_in_plane_rotation_predictions": self.bool_vars["show_in_plane_rotation_predictions"].get(),
            "export_files": self.selected_export_files(),
        }

    def build_argv(self) -> list[str]:
        settings = self.build_settings()
        argv = [
            str(settings["image"]),
            "--alpha",
            str(settings["alpha"]),
            "--beta",
            str(settings["beta"]),
            "--target-families",
            *settings["target_families"],  # type: ignore[arg-type]
        ]

        current_zone = settings["current_zone"]
        if current_zone is not None:
            argv.extend(["--current-zone", ",".join(str(x) for x in current_zone)])  # type: ignore[union-attr]

        alpha_limits = settings["alpha_limits"]
        beta_limits = settings["beta_limits"]
        center = settings["center"]

        argv.extend(["--alpha-limits", str(alpha_limits[0]), str(alpha_limits[1])])  # type: ignore[index]
        argv.extend(["--beta-limits", str(beta_limits[0]), str(beta_limits[1])])  # type: ignore[index]
        argv.extend(["--holder-order", str(settings["holder_order"])])
        argv.extend(["--image-to-holder-rotation-deg", str(settings["image_to_holder_rotation_deg"])])
        if center is not None:
            argv.extend(["--center", str(center[0]), str(center[1])])  # type: ignore[index]
        argv.extend(["--n-peaks", str(settings["n_peaks"])])
        if settings["min_distance_px"] is not None:
            argv.extend(["--min-distance-px", str(settings["min_distance_px"])])
        if settings["spot_sigma_px"] is not None:
            argv.extend(["--spot-sigma-px", str(settings["spot_sigma_px"])])
        argv.extend(["--peak-percentile", str(settings["peak_percentile"])])
        argv.extend(["--max-index", str(settings["max_index"])])
        argv.extend(["--max-g-norm", str(settings["max_g_norm"])])
        argv.extend(["--tolerance-fraction", str(settings["tolerance_fraction"])])
        argv.extend(["--kikuchi-max-g-norm", str(settings["kikuchi_max_g_norm"])])
        if settings["output_prefix"] is not None:
            argv.extend(["--output-prefix", str(settings["output_prefix"])])

        if settings["include_opposites"]:
            argv.append("--include-opposites")
        if settings["invert"]:
            argv.append("--invert")
        if settings["rotate_pattern_180"]:
            argv.append("--rotate-pattern-180")
        if settings["show_in_plane_rotation_predictions"]:
            argv.append("--show-in-plane-rotation-predictions")
        if not settings["show_labels"]:
            argv.append("--no-labels")
        if not settings["show_kikuchi_guides"]:
            argv.append("--no-kikuchi-guides")
        argv.append("--export-files")
        argv.extend(settings["export_files"])  # type: ignore[arg-type]

        return argv

    def selected_export_files(self) -> list[str]:
        selected: list[str] = []
        if self.bool_vars["export_target_csv"].get():
            selected.append("target_csv")
        if self.bool_vars["export_indexed_spots"].get():
            selected.append("indexed_spots")
        if self.bool_vars["export_predicted_pattern"].get():
            selected.append("predicted_pattern")
        if self.bool_vars["export_fitted_pattern"].get():
            selected.append("fitted_pattern")
        return selected

    def _float_value(self, key: str, default: float | None = None) -> float:
        value = self.vars[key].get().strip()
        if not value:
            if default is not None:
                return default
            raise ValueError(f"{key} is required.")
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{key} must be a number.") from exc

    def _optional_float_value(self, key: str) -> float | None:
        value = self.vars[key].get().strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{key} must be a number.") from exc

    def _int_value(self, key: str, default: int | None = None) -> int:
        value = self.vars[key].get().strip()
        if not value:
            if default is not None:
                return default
            raise ValueError(f"{key} is required.")
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer.") from exc

    def overlay_path_from_settings(self, settings: dict[str, object] | None = None) -> Path | None:
        if settings is None:
            settings = self.build_settings()
        if "fitted_pattern" not in settings["export_files"]:
            return None
        prefix = self.output_prefix_from_settings(settings)
        return Path(f"{prefix}_fitted_diffraction_pattern.png")

    def output_prefix_from_settings(self, settings: dict[str, object]) -> Path:
        if settings["output_prefix"] is not None:
            return settings["output_prefix"]  # type: ignore[return-value]
        image = settings["image"]
        return image.with_suffix("")  # type: ignore[union-attr]

    def preview_paths_from_settings(self, settings: dict[str, object]) -> PreviewResult:
        prefix = self.output_prefix_from_settings(settings)
        return PreviewResult(
            predicted_path=Path(f"{prefix}_predicted_zone_pattern.png") if "predicted_pattern" in settings["export_files"] else None,
            fitted_path=Path(f"{prefix}_fitted_diffraction_pattern.png") if "fitted_pattern" in settings["export_files"] else None,
        )

    def show_command(self) -> None:
        try:
            argv = self.build_argv()
        except Exception as exc:
            messagebox.showerror("Cannot Build Command", str(exc))
            return
        python_exe = sys.executable
        command = " ".join([self._quote(python_exe), self._quote(str(APP_DIR / "zone_axis_finder.py")), *map(self._quote, argv)])
        self._write_output(command + "\n")

    def _quote(self, value: str) -> str:
        if not value or any(ch.isspace() for ch in value):
            return "'" + value.replace("'", "'\"'\"'") + "'"
        return value

    def run_analysis(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            settings = self.build_settings()
            argv = self.build_argv()
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        self.run_button.configure(state="disabled")
        self.status_var.set("Running analysis...")
        if settings["export_files"]:
            self._write_output("Running analysis and exporting selected result files...\n\n")
        else:
            self._write_output("Running analysis in preview-only mode. No files will be written.\n\n")
        self.latest_previews = PreviewResult()
        self._set_download_buttons_state()
        self._set_simulator_button_state()
        self._close_sample_rotation_simulator()

        previews = self.preview_paths_from_settings(settings)
        self.worker = threading.Thread(target=self._run_worker, args=(settings, argv, previews), daemon=True)
        self.worker.start()
        self.after(100, self._poll_result_queue)

    def _run_worker(self, settings: dict[str, object], argv: list[str], previews: PreviewResult) -> None:
        stream = io.StringIO()
        status = "ok"
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            try:
                previews = self._run_gui_analysis(settings, previews)
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                if code != 0:
                    status = "error"
                    print(exc)
            except Exception:
                status = "error"
                traceback.print_exc()
        self.result_queue.put((status, stream.getvalue(), previews))

    def _run_gui_analysis(self, settings: dict[str, object], previews: PreviewResult) -> PreviewResult:
        gray, rgb = zone_axis_finder.load_grayscale(settings["image"], invert=settings["invert"])  # type: ignore[arg-type]
        peaks = zone_axis_finder.detect_spots(
            gray,
            n_peaks=settings["n_peaks"],  # type: ignore[arg-type]
            min_distance_px=settings["min_distance_px"],  # type: ignore[arg-type]
            spot_sigma_px=settings["spot_sigma_px"],  # type: ignore[arg-type]
            peak_percentile=settings["peak_percentile"],  # type: ignore[arg-type]
        )
        if len(peaks) < 4:
            raise SystemExit("Too few diffraction spots were detected. Try lowering Peak percentile.")

        best, all_results = zone_axis_finder.choose_best_match(
            peaks,
            rgb.size,
            current_zone=settings["current_zone"],  # type: ignore[arg-type]
            center_xy=settings["center"],  # type: ignore[arg-type]
            max_index=settings["max_index"],  # type: ignore[arg-type]
            max_g_norm=settings["max_g_norm"],  # type: ignore[arg-type]
            tolerance_fraction=settings["tolerance_fraction"],  # type: ignore[arg-type]
        )
        if settings["rotate_pattern_180"]:
            rotated_best = zone_axis_finder.rotate_match_in_plane(
                best,
                180.0,
                peaks,
                rgb.size,
                settings["tolerance_fraction"],  # type: ignore[arg-type]
            )
            all_results = [rotated_best if result is best else result for result in all_results]
            best = rotated_best
        crystal_to_zero_holder, _image_axes, _holder_axes = zone_axis_finder.orientation_from_match(
            best,
            alpha_deg=settings["alpha"],  # type: ignore[arg-type]
            beta_deg=settings["beta"],  # type: ignore[arg-type]
            image_to_holder_rotation_deg=settings["image_to_holder_rotation_deg"],  # type: ignore[arg-type]
            holder_order=settings["holder_order"],  # type: ignore[arg-type]
        )
        rows = zone_axis_finder.target_rows(
            best,
            crystal_to_zero_holder,
            alpha_deg=settings["alpha"],  # type: ignore[arg-type]
            beta_deg=settings["beta"],  # type: ignore[arg-type]
            target_families=settings["target_families"],  # type: ignore[arg-type]
            include_opposites=settings["include_opposites"],  # type: ignore[arg-type]
            holder_order=settings["holder_order"],  # type: ignore[arg-type]
            alpha_limits=settings["alpha_limits"],  # type: ignore[arg-type]
            beta_limits=settings["beta_limits"],  # type: ignore[arg-type]
        )
        sample_rotation_rows = []
        if settings["show_in_plane_rotation_predictions"]:
            sample_rotation_rows = zone_axis_finder.in_plane_rotation_rows(
                best,
                crystal_to_zero_holder,
                target_families=settings["target_families"],  # type: ignore[arg-type]
                include_opposites=settings["include_opposites"],  # type: ignore[arg-type]
                holder_order=settings["holder_order"],  # type: ignore[arg-type]
                alpha_limits=settings["alpha_limits"],  # type: ignore[arg-type]
                beta_limits=settings["beta_limits"],  # type: ignore[arg-type]
            )
        map_families = (
            settings["target_families"]
            if settings["map_show_target_families"]
            else zone_axis_finder.SUPPORTED_ZONE_FAMILIES
        )
        map_points = zone_axis_finder.zone_axis_map_points(
            best,
            crystal_to_zero_holder,
            alpha_deg=settings["alpha"],  # type: ignore[arg-type]
            beta_deg=settings["beta"],  # type: ignore[arg-type]
            map_families=map_families,  # type: ignore[arg-type]
            holder_order=settings["holder_order"],  # type: ignore[arg-type]
            alpha_limits=settings["alpha_limits"],  # type: ignore[arg-type]
            beta_limits=settings["beta_limits"],  # type: ignore[arg-type]
        )
        if settings["map_reachable_only"]:
            map_points = zone_axis_finder.filter_reachable_zone_axis_map_points(
                map_points,
                crystal_to_zero_holder,
                str(settings["holder_order"]),
                settings["alpha_limits"],  # type: ignore[arg-type]
                settings["beta_limits"],  # type: ignore[arg-type]
            )
        zone_map_image = zone_axis_finder.zone_axis_map_image(
            map_points,
            alpha_limits=settings["alpha_limits"],  # type: ignore[arg-type]
            beta_limits=settings["beta_limits"],  # type: ignore[arg-type]
            rotation_rows=sample_rotation_rows if settings["show_in_plane_rotation_predictions"] else None,
            map_label_individual_color=settings["map_label_individual_color"],  # type: ignore[arg-type]
        )

        zone_axis_finder.print_match_summary(best, all_results)
        if settings["rotate_pattern_180"]:
            print("\nApplied correction: fitted pattern indexing rotated by 180 deg.")
        zone_axis_finder.print_target_table(rows)
        if settings["show_in_plane_rotation_predictions"]:
            zone_axis_finder.print_in_plane_rotation_table(sample_rotation_rows)
        predicted_image = zone_axis_finder.predicted_pattern_image(
            best,
            rgb.size,
            rotated=False,
            title=f"Predicted FCC {zone_axis_finder.format_miller(best.zone)} before rotation",
            show_labels=settings["show_labels"],  # type: ignore[arg-type]
            draw_guides=settings["show_kikuchi_guides"],  # type: ignore[arg-type]
            kikuchi_max_g_norm=settings["kikuchi_max_g_norm"],  # type: ignore[arg-type]
        )
        fitted_image = zone_axis_finder.fitted_diffraction_image(
            rgb,
            best,
            show_labels=settings["show_labels"],  # type: ignore[arg-type]
            draw_guides=settings["show_kikuchi_guides"],  # type: ignore[arg-type]
            kikuchi_max_g_norm=settings["kikuchi_max_g_norm"],  # type: ignore[arg-type]
            title=f"Fitted FCC {zone_axis_finder.format_miller(best.zone)}",
        )

        prefix = self.output_prefix_from_settings(settings)
        export_files = set(settings["export_files"])  # type: ignore[arg-type]
        written: list[tuple[str, Path]] = []
        if "target_csv" in export_files:
            path = Path(f"{prefix}_target_zone_axes.csv")
            zone_axis_finder.write_targets_csv(path, rows)
            written.append(("target angles", path))
        if "indexed_spots" in export_files:
            path = Path(f"{prefix}_indexed_spots.csv")
            zone_axis_finder.write_indexed_spots_csv(path, best, peaks)
            written.append(("indexed spots", path))
        if "predicted_pattern" in export_files:
            path = Path(f"{prefix}_predicted_zone_pattern.png")
            predicted_image.save(path)
            previews.predicted_path = path
            written.append(("predicted", path))
        if "fitted_pattern" in export_files:
            path = Path(f"{prefix}_fitted_diffraction_pattern.png")
            fitted_image.save(path)
            previews.fitted_path = path
            written.append(("fitted pattern", path))

        if written:
            print("\nWrote")
            for label, path in written:
                print(f"  {label:<15}: {path}")
        else:
            print("\nPreview-only mode: no files were written.")
        if settings["current_zone"] is None:
            print(
                "\nNote: the auto-indexed [uvw] is a conventional cubic assignment. "
                "Use Known current zone if you need to force a specific equivalent index."
            )
        previews.predicted_image = predicted_image
        previews.fitted_image = fitted_image
        previews.map_image = zone_map_image
        previews.analysis_context = AnalysisContext(
            image_path=Path(settings["image"]),  # type: ignore[arg-type]
            match=best,
            crystal_to_zero_holder=crystal_to_zero_holder,
            target_families=list(settings["target_families"]),  # type: ignore[arg-type]
            include_opposites=bool(settings["include_opposites"]),
            holder_order=str(settings["holder_order"]),
            alpha_limits=settings["alpha_limits"],  # type: ignore[arg-type]
            beta_limits=settings["beta_limits"],  # type: ignore[arg-type]
            image_to_holder_rotation_deg=float(settings["image_to_holder_rotation_deg"]),
            map_label_individual_color=bool(settings["map_label_individual_color"]),
            map_reachable_only=bool(settings["map_reachable_only"]),
        )
        return previews

    def _poll_result_queue(self) -> None:
        try:
            status, output, previews = self.result_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_result_queue)
            return

        self.run_button.configure(state="normal")
        if status == "ok":
            main_output, more_output = self._split_analysis_output(output)
            self._write_output(main_output, more_output)
            self.status_var.set("Analysis complete")
            self.latest_previews = previews
            self._load_generated_previews(previews)
            self._set_download_buttons_state()
            self._set_simulator_button_state()
        else:
            self._write_output(output, "")
            self.status_var.set("Analysis failed")
            self.latest_previews = PreviewResult()
            self._set_download_buttons_state()
            self._set_simulator_button_state()
            messagebox.showerror("Analysis Failed", "The analysis did not finish. See the Output panel for details.")

    def _set_download_buttons_state(self) -> None:
        for kind, button in self.download_buttons.items():
            source, _suffix, _title = self._preview_source_for_kind(kind)
            button.configure(state="normal" if source is not None else "disabled")

    def _set_simulator_button_state(self) -> None:
        if self.simulator_button is not None:
            state = "normal" if self.latest_previews.analysis_context is not None else "disabled"
            self.simulator_button.configure(state=state)

    def _write_output(self, text: str, more_text: str = "") -> None:
        self._write_text_widget(self.output_text, text)
        self._write_text_widget(self.more_output_text, more_text)

    def _write_text_widget(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.see("1.0")

    def _split_analysis_output(self, text: str) -> tuple[str, str]:
        lines = text.splitlines()
        target_idx = self._find_line(lines, "Predicted target holder angles")
        if target_idx is None:
            return text, ""

        main_lines: list[str] = []
        more_lines: list[str] = []
        pre_target = lines[:target_idx]
        alt_idx = self._find_line(pre_target, "Alternative reference scores")
        if alt_idx is None:
            main_lines.extend(pre_target)
        else:
            correction_idx = self._find_line_containing(pre_target, "Applied correction:")
            if correction_idx is not None and correction_idx > alt_idx:
                main_lines.extend(pre_target[:alt_idx])
                more_lines.extend(self._trim_blank_edges(pre_target[alt_idx:correction_idx]))
                main_lines.extend(pre_target[correction_idx:])
            else:
                main_lines.extend(pre_target[:alt_idx])
                more_lines.extend(self._trim_blank_edges(pre_target[alt_idx:]))

        target_lines = lines[target_idx:]
        row_start = target_idx + 4
        post_idx = len(lines)
        for idx in range(row_start, len(lines)):
            if not lines[idx].strip():
                post_idx = idx
                break

        target_section = lines[target_idx:post_idx]
        post_lines = self._trim_blank_edges(lines[post_idx:])
        sample_rotation_section, post_lines = self._extract_named_section(
            post_lines,
            "Out-of-limit targets reachable by sample in-plane rotation",
        )
        sample_rotation_keys = {
            key
            for key in (self._target_row_key(line) for line in sample_rotation_section)
            if key is not None
        }
        target_header = target_section[:4]
        target_rows = [line for line in target_section[4:] if line.strip()]
        yes_rows = [line for line in target_rows if line.lstrip().startswith("yes ")]
        no_rows = [
            line
            for line in target_rows
            if line.lstrip().startswith("no ") and self._target_row_key(line) not in sample_rotation_keys
        ]

        if main_lines and main_lines[-1].strip():
            main_lines.append("")
        main_lines.extend(target_header)
        if yes_rows:
            main_lines.extend(yes_rows)
        else:
            main_lines.append("(No target zone axes are within the tilt limits.)")

        if sample_rotation_section:
            if main_lines and main_lines[-1].strip():
                main_lines.append("")
            main_lines.extend(sample_rotation_section)

        if no_rows:
            if more_lines and more_lines[-1].strip():
                more_lines.append("")
            more_lines.extend(
                [
                    "Out-of-limit target holder angles",
                    "---------------------------------",
                    *target_header[2:],
                    *no_rows,
                ]
            )
        if post_lines:
            if more_lines and more_lines[-1].strip():
                more_lines.append("")
            more_lines.extend(post_lines)

        return "\n".join(main_lines).strip() + "\n", "\n".join(more_lines).strip() + ("\n" if more_lines else "")

    def _extract_named_section(self, lines: Sequence[str], heading: str) -> tuple[list[str], list[str]]:
        section_idx = self._find_line(lines, heading)
        if section_idx is None:
            return [], list(lines)
        section_end = len(lines)
        for idx in range(section_idx + 1, len(lines)):
            if not lines[idx].strip():
                section_end = idx
                break
        section = self._trim_blank_edges(lines[section_idx:section_end])
        remainder = self._trim_blank_edges([*lines[:section_idx], *lines[section_end:]])
        return section, remainder

    def _target_row_key(self, line: str) -> tuple[str, str] | None:
        match = re.match(r"^\s*(?:yes|no)\s+(<[^>]+>)\s+(\[[^\]]+\])", line)
        if match is None:
            return None
        return match.group(1), match.group(2)

    def _find_line(self, lines: Sequence[str], target: str) -> int | None:
        for idx, line in enumerate(lines):
            if line.strip() == target:
                return idx
        return None

    def _find_line_containing(self, lines: Sequence[str], target: str) -> int | None:
        for idx, line in enumerate(lines):
            if target in line:
                return idx
        return None

    def _trim_blank_edges(self, lines: Sequence[str]) -> list[str]:
        start = 0
        end = len(lines)
        while start < end and not lines[start].strip():
            start += 1
        while end > start and not lines[end - 1].strip():
            end -= 1
        return list(lines[start:end])


def main() -> int:
    app = ZoneAxisFinderGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
