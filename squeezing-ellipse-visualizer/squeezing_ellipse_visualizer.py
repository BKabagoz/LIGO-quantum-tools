#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive vacuum, FIS, and FDS quadrature-ellipse visualizer.

The tool can read its physical squeezing and filter-cavity parameters from a
small YAML file or accept the same values through the manual controls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

try:
    import yaml
except Exception as exc:
    raise RuntimeError("This script requires PyYAML. Install with: pip install pyyaml") from exc

QT_AVAILABLE = True
QT_IMPORT_ERROR = None
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from PyQt5.QtWidgets import (
        QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
        QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QMainWindow, QMessageBox, QPushButton, QRadioButton, QScrollArea, QTabWidget,
        QVBoxLayout, QWidget
    )
except Exception as exc:
    QT_AVAILABLE = False
    QT_IMPORT_ERROR = exc
    QApplication = None
    QMainWindow = object
    QComboBox = object

C = 299_792_458.0
DEFAULT_EXPORT_STEM = "squeezing_state_ellipses"

DEFAULT_PARAMS = {
    "sqz_db": 6.0,
    "inj_loss": 0.05,
    "sqz_angle_deg": 0.0,
    "fc_L_m": 300.0,
    "fc_Ti": 8.0e-4,
    "fc_Te": 1.0e-6,
    "fc_Lrt": 1.0e-4,
    "fc_detune_Hz": -25.0,
}

COLOR_CHOICES = [
    ("vacuum navy", "#000060"),
    ("FIS lavender", "#8185ff"),
    ("FDS magenta", "#d900ff"),
    ("teal", "#008c9e"),
    ("orange", "#ff7f0e"),
    ("purple", "#7f1d6a"),
    ("green", "#228833"),
    ("red-orange", "#cc3311"),
    ("mauve", "#aa4499"),
    ("sea green", "#44aa99"),
    ("indigo", "#332288"),
    ("sand", "#ddcc77"),
    ("black", "#000000"),
    ("gray", "#777777"),
]

FONT_WEIGHTS = ["normal", "medium", "semibold", "bold"]

LINE_STYLE_CHOICES = [
    ("solid", "-"),
    ("dashed", "--"),
    ("dotted", ":"),
    ("dash-dot", "-."),
]


# ---------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------

def read_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_params_from_yaml(path: Path) -> dict:
    y = read_yaml(path)
    sqz = y.get("Squeezer", {})
    fc = sqz.get("FilterCavity", {})
    sqz_angle_rad = float(sqz.get("SQZAngle", 0.0))
    return {
        "sqz_db": float(sqz.get("AmplitudedB", 15.0)),
        "inj_loss": float(sqz.get("InjectionLoss", 0.05)),
        "sqz_angle_deg": sqz_angle_rad * 180.0 / np.pi,
        "fc_L_m": float(fc.get("L", 300.0)),
        "fc_Ti": float(fc.get("Ti", 8.0e-4)),
        "fc_Te": float(fc.get("Te", 1.0e-6)),
        "fc_Lrt": float(fc.get("Lrt", 1.0e-4)),
        "fc_detune_Hz": float(fc.get("fdetune", -25.0)),
    }


def rot2(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def covariance_from_db(sqz_db: float, angle_deg: float, eta: float = 1.0) -> np.ndarray:
    vmin = 10.0 ** (-float(sqz_db) / 10.0)
    vmax = 10.0 ** (+float(sqz_db) / 10.0)
    R = rot2(np.deg2rad(angle_deg))
    V = R @ np.diag([vmin, vmax]) @ R.T
    return float(eta) * V + (1.0 - float(eta)) * np.eye(2)


def ellipse_points_from_cov(V: np.ndarray, npts: int = 1000):
    V = 0.5 * (V + V.T)
    vals, vecs = np.linalg.eigh(V)
    vals = np.maximum(vals, 0.0)
    t = np.linspace(0.0, 2.0 * np.pi, int(npts))
    circle = np.vstack([np.cos(t), np.sin(t)])
    ell = vecs @ np.diag(np.sqrt(vals)) @ circle
    return ell[0], ell[1]


def cavity_reflection(offset_hz, L_m, Ti, Te, Lrt):
    offset_hz = np.asarray(offset_hz, dtype=float)
    r1 = np.sqrt(max(0.0, 1.0 - float(Ti)))
    r2 = np.sqrt(max(0.0, (1.0 - float(Te)) * (1.0 - float(Lrt))))
    phi = 2.0 * np.pi * offset_hz * (2.0 * float(L_m) / C)
    z = np.exp(1j * phi)
    return (r1 - r2 * z) / (1.0 - r1 * r2 * z)


def fc_rotation_grid(
    f_grid_hz, L_m, Ti, Te, Lrt, detune_Hz,
    sign=+1.0, offset_deg=0.0, zero_at_high_f=True
):
    f_grid_hz = np.asarray(f_grid_hz, dtype=float)
    rp = cavity_reflection(detune_Hz + f_grid_hz, L_m, Ti, Te, Lrt)
    rm = cavity_reflection(detune_Hz - f_grid_hz, L_m, Ti, Te, Lrt)
    theta = 0.5 * (np.unwrap(np.angle(rp)) + np.unwrap(np.angle(rm)))
    if zero_at_high_f and theta.size > 0:
        theta = theta - theta[-1]
    theta = float(sign) * theta + np.deg2rad(offset_deg)
    return theta


def fc_rotation_at_frequencies(
    freqs_hz, params: dict, sign=1.0, offset_deg=0.0,
    zero_at_high_f=True
):
    fgrid = np.geomspace(1.0, 8000.0, 4000)
    theta_grid = fc_rotation_grid(
        fgrid,
        params["fc_L_m"],
        params["fc_Ti"],
        params["fc_Te"],
        params["fc_Lrt"],
        params["fc_detune_Hz"],
        sign=sign,
        offset_deg=offset_deg,
        zero_at_high_f=zero_at_high_f,
    )
    return np.interp(np.asarray(freqs_hz, dtype=float), fgrid, theta_grid)


def parse_freq_list(text):
    vals = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    return vals


def parse_text_list(text):
    out = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(item)
    return out


def alpha_ramp(n: int, amin: float, amax: float) -> np.ndarray:
    if n <= 1:
        return np.array([amax], dtype=float)
    return np.linspace(float(amin), float(amax), int(n))


def add_color_items(combo):
    for name, hex_color in COLOR_CHOICES:
        combo.addItem(f"{name}  {hex_color}", hex_color)


def add_line_style_items(combo):
    for name, style in LINE_STYLE_CHOICES:
        combo.addItem(name, style)


def set_combo_hex(combo, hexval):
    for i in range(combo.count()):
        if str(combo.itemData(i)).lower() == str(hexval).lower():
            combo.setCurrentIndex(i)
            return


def set_combo_text(combo, text):
    i = combo.findText(str(text))
    if i >= 0:
        combo.setCurrentIndex(i)


# ---------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------

def build_horizontal_state_figure(s, fig=None, apply_figsize=True):
    if fig is None:
        fig = Figure(figsize=(s["fig_w"], s["fig_h"]), dpi=110)
    else:
        fig.clear()
        if apply_figsize:
            fig.set_size_inches(s["fig_w"], s["fig_h"], forward=False)

    font_family = str(s["font_family"]).strip()

    params = s["params"]
    sqz_db = s["visual_sqz_db"] if s["use_visual_sqz"] else params["sqz_db"]
    eta = 1.0 - params["inj_loss"] if s["include_injection_loss"] else 1.0
    eta = np.clip(eta, 0.0, 1.0)

    V_vac = np.eye(2)
    xy_vac = ellipse_points_from_cov(V_vac)

    fis_angle = params["sqz_angle_deg"] + s["fis_extra_angle_deg"]
    V_fis = covariance_from_db(sqz_db, fis_angle, eta=eta)
    xy_fis = ellipse_points_from_cov(V_fis)

    theta_fc = fc_rotation_at_frequencies(
        s["fds_freqs_hz"],
        params,
        sign=s["fc_sign"],
        offset_deg=s["fc_offset_deg"],
        zero_at_high_f=s["zero_fc_rotation_at_high_f"],
    )

    custom_fds_labels = s["fds_custom_labels"]
    fds_xy = []
    fds_labels = []
    for i, (f, th) in enumerate(zip(s["fds_freqs_hz"], theta_fc)):
        angle = params["sqz_angle_deg"] + np.rad2deg(th) + s["fds_extra_angle_deg"]
        V = covariance_from_db(sqz_db, angle, eta=eta)
        fds_xy.append(ellipse_points_from_cov(V))
        if i < len(custom_fds_labels):
            lab = custom_fds_labels[i]
        else:
            lab = s["fds_label_format"].format(
                f=f,
                theta=angle,
                theta_fc=np.rad2deg(th),
            )
        fds_labels.append(lab)

    def font_kwargs(size, weight):
        kw = {"fontsize": size, "fontweight": weight}
        if font_family:
            kw["fontfamily"] = font_family
        return kw

    def style_axis(ax, orig_idx):
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)

        if s["use_common_square_limits"]:
            lim = float(s["common_square_limit"])
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        else:
            ax.set_xlim(s["xlims"][orig_idx][0], s["xlims"][orig_idx][1])
            ax.set_ylim(s["ylims"][orig_idx][0], s["ylims"][orig_idx][1])

        if s["show_crosshairs"]:
            ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
            ax.axvline(0.0, color="0.85", lw=0.8, zorder=0)

        if s["show_grid"]:
            ax.grid(True, alpha=s["grid_alpha"])

        if s["show_ticks"]:
            ax.tick_params(labelsize=s["tick_fontsize"])
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight(s["tick_fontweight"])
                if font_family:
                    tick.set_fontfamily(font_family)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_xlabel(
            s["xlabel"],
            labelpad=s["xlabel_pad"],
            **font_kwargs(s["label_fontsize"], s["label_fontweight"]),
        )
        ax.set_ylabel(
            s["ylabel"],
            labelpad=s["ylabel_pad"],
            **font_kwargs(s["label_fontsize"], s["label_fontweight"]),
        )
        if s["titles"][orig_idx]:
            ax.set_title(
                s["titles"][orig_idx],
                pad=s["title_pad"],
                **font_kwargs(s["title_fontsize"], s["title_fontweight"]),
            )

        for spine in ax.spines.values():
            spine.set_color("0.75")
            spine.set_linewidth(0.8)

    def place_legend(ax, orig_idx):
        if not s["show_legend"][orig_idx]:
            return
        prop = {"size": s["legend_fontsize"][orig_idx], "weight": s["legend_fontweight"]}
        if font_family:
            prop["family"] = font_family
        ax.legend(
            loc=s["legend_loc"][orig_idx],
            bbox_to_anchor=(s["legend_x"][orig_idx], s["legend_y"][orig_idx]),
            prop=prop,
            frameon=s["legend_frame"],
            framealpha=s["legend_framealpha"],
        )

    panel_specs = [
        {"orig_idx": 0, "kind": "vacuum"},
        {"orig_idx": 1, "kind": "fis"},
        {"orig_idx": 2, "kind": "fds"},
    ]
    visible_specs = [spec for i, spec in enumerate(panel_specs) if s["show_panels"][i]]
    if not visible_specs:
        raise ValueError("At least one panel must be selected to display.")

    # Layout behavior when one or more panels are hidden:
    #   rescale_hidden_panels=True  -> remaining panels expand to fill the row.
    #   rescale_hidden_panels=False -> preserve the original 3-panel slots,
    #                                  leaving blank space where panels are hidden.
    if s["rescale_hidden_panels"]:
        gs = fig.add_gridspec(1, len(visible_specs), wspace=s["wspace"])
        panel_axes = [
            (fig.add_subplot(gs[0, display_idx]), spec)
            for display_idx, spec in enumerate(visible_specs)
        ]
    else:
        gs = fig.add_gridspec(1, 3, wspace=s["wspace"])
        panel_axes = [
            (fig.add_subplot(gs[0, spec["orig_idx"]]), spec)
            for spec in visible_specs
        ]

    for ax, spec in panel_axes:
        orig_idx = spec["orig_idx"]
        kind = spec["kind"]

        if kind == "vacuum":
            ax.fill(
                xy_vac[0], xy_vac[1], facecolor=s["vac_color"], edgecolor="none",
                alpha=s["vac_fill_alpha"], zorder=1,
            )
            ax.plot(
                xy_vac[0], xy_vac[1], color=s["vac_color"], lw=s["vac_lw"],
                ls=s["vac_linestyle"], alpha=s["vac_alpha"],
                label=s["vac_label"], zorder=2,
            )
        elif kind == "fis":
            # Draw the vacuum reference first so it sits beneath the FIS ellipse.
            if s["overlay_vac_on_fis"]:
                ax.plot(
                    xy_vac[0], xy_vac[1], color=s["vac_color"], lw=s["vac_lw"],
                    ls=s["vac_linestyle"], alpha=s["vac_alpha"],
                    label=s["vac_label"], zorder=0.5,
                )
            ax.fill(
                xy_fis[0], xy_fis[1], facecolor=s["fis_color"], edgecolor="none",
                alpha=s["fis_fill_alpha"], zorder=1,
            )
            ax.plot(
                xy_fis[0], xy_fis[1], color=s["fis_color"], lw=s["fis_lw"],
                ls=s["fis_linestyle"], alpha=s["fis_alpha"],
                label=s["fis_label"], zorder=2,
            )
        elif kind == "fds":
            # Draw the vacuum reference first so it sits beneath all FDS ellipses.
            if s["overlay_vac_on_fds"]:
                ax.plot(
                    xy_vac[0], xy_vac[1], color=s["vac_color"], lw=s["vac_lw"],
                    ls=s["vac_linestyle"], alpha=s["vac_alpha"],
                    label=s["vac_label"], zorder=0.5,
                )
            alphas = alpha_ramp(len(fds_xy), s["fds_alpha_min"], s["fds_alpha_max"])
            for xy, lab, a in zip(fds_xy, fds_labels, alphas):
                ax.fill(
                    xy[0], xy[1], facecolor=s["fds_color"], edgecolor="none",
                    alpha=s["fds_fill_alpha"], zorder=1,
                )
                ax.plot(
                    xy[0], xy[1], color=s["fds_color"], lw=s["fds_lw"],
                    ls=s["fds_linestyle"], alpha=float(a), label=lab, zorder=2,
                )

        style_axis(ax, orig_idx)
        place_legend(ax, orig_idx)

    fig.subplots_adjust(left=s["left"], right=s["right"], bottom=s["bottom"], top=s["top"])
    return fig, theta_fc


# ---------------------------------------------------------------------
# Qt controls
# ---------------------------------------------------------------------

class PanelLegendControls:
    def __init__(self, title: str, default_loc: str = "upper right", default_x: float = 1.0, default_y: float = 1.0):
        self.box = QGroupBox(title)
        f = QFormLayout(self.box)

        self.show = QCheckBox("Show legend")
        self.show.setChecked(True)

        self.loc = QComboBox()
        for loc in [
            "best", "upper right", "upper left", "lower right", "lower left",
            "center right", "center left", "upper center", "lower center", "center"
        ]:
            self.loc.addItem(loc)
        self.loc.setCurrentText(default_loc)

        self.x = QDoubleSpinBox()
        self.y = QDoubleSpinBox()
        self.fs = QDoubleSpinBox()

        for w, val in [(self.x, default_x), (self.y, default_y)]:
            w.setRange(-2.0, 3.0)
            w.setDecimals(2)
            w.setSingleStep(0.05)
            w.setValue(val)

        self.fs.setRange(4.0, 40.0)
        self.fs.setDecimals(1)
        self.fs.setSingleStep(0.5)
        self.fs.setValue(9.0)

        f.addRow(self.show)
        f.addRow("Legend loc", self.loc)
        f.addRow("Legend x", self.x)
        f.addRow("Legend y", self.y)
        f.addRow("Legend fontsize", self.fs)


class LimitsControls:
    def __init__(self, title: str, xlim=(-5.0, 5.0), ylim=(-5.0, 5.0)):
        self.box = QGroupBox(title)
        g = QGridLayout(self.box)

        self.xmin = QDoubleSpinBox()
        self.xmax = QDoubleSpinBox()
        self.ymin = QDoubleSpinBox()
        self.ymax = QDoubleSpinBox()

        for w, val in [
            (self.xmin, xlim[0]),
            (self.xmax, xlim[1]),
            (self.ymin, ylim[0]),
            (self.ymax, ylim[1]),
        ]:
            w.setRange(-100.0, 100.0)
            w.setDecimals(2)
            w.setSingleStep(0.10)
            w.setValue(val)

        g.addWidget(QLabel("xmin"), 0, 0)
        g.addWidget(self.xmin, 0, 1)
        g.addWidget(QLabel("xmax"), 0, 2)
        g.addWidget(self.xmax, 0, 3)
        g.addWidget(QLabel("ymin"), 1, 0)
        g.addWidget(self.ymin, 1, 1)
        g.addWidget(QLabel("ymax"), 1, 2)
        g.addWidget(self.ymax, 1, 3)


class Fig2StatePanelsWindow(QMainWindow):
    def __init__(self, yaml_path: Path | None = None):
        super().__init__()
        self.yaml_path = yaml_path
        self.params = (
            load_params_from_yaml(yaml_path)
            if yaml_path is not None
            else dict(DEFAULT_PARAMS)
        )

        self.setWindowTitle("Squeezing-state ellipse visualizer")
        self.resize(1650, 800)

        self._build_ui()
        self.redraw()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setContentsMargins(4, 4, 4, 4)
        main.setSpacing(4)

        # Left column: figure above, style layers below.  The layers therefore
        # never extend beneath the full-height model panel on the right.
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # Preview area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(2)

        # Keep the active model filename with the figure rather than in the
        # narrow full-height controls column.  Only the filename is shown;
        # hovering it reveals the complete path.
        yaml_row = QHBoxLayout()
        yaml_row.setContentsMargins(4, 0, 4, 0)
        yaml_row.setSpacing(5)
        yaml_row.addWidget(QLabel("YAML:"))
        yaml_name = self.yaml_path.name if self.yaml_path is not None else "none (manual inputs)"
        yaml_tooltip = str(self.yaml_path) if self.yaml_path is not None else "No YAML file loaded"
        self.yaml_label = QLabel(yaml_name)
        self.yaml_label.setToolTip(yaml_tooltip)
        yaml_row.addWidget(self.yaml_label)
        yaml_row.addStretch(1)
        plot_layout.addLayout(yaml_row)

        self.fig = Figure(figsize=(13.5, 3.6), dpi=110)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        left_layout.addWidget(plot_widget, stretch=1)

        # Right side: the model layer remains permanently visible.
        right_widget = QWidget()
        right_widget.setFixedWidth(500)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        controls = QWidget()
        self.ctrl = QVBoxLayout(controls)
        self.ctrl.setContentsMargins(4, 4, 4, 4)
        self._build_model_group()
        self.ctrl.addStretch(1)
        scroll.setWidget(controls)
        right_layout.addWidget(scroll, stretch=1)

        self._build_fixed_actions_group(right_layout)
        # Keep the equations fully visible at the bottom.  Only the model
        # controls above these fixed blocks are allowed to scroll.
        self._build_model_equations_group(right_layout)

        # All non-model controls live in separate layers below the figure.
        self.style_tabs = QTabWidget()
        self.style_tabs.setTabPosition(QTabWidget.South)
        self.style_tabs.setMovable(False)
        self.style_tabs.setUsesScrollButtons(True)
        self.style_tabs.setMinimumHeight(225)
        self.style_tabs.setMaximumHeight(270)

        self._add_bottom_layer("Panels / limits", self._build_limits_group)
        self._add_bottom_layer("Figure layout", self._build_figure_group)
        self._add_bottom_layer("Titles / labels / fonts", self._build_text_group)
        self._add_bottom_layer("Ellipse style", self._build_style_group)
        self._add_bottom_layer("Panel legends", self._build_legend_groups)

        left_layout.addWidget(self.style_tabs, stretch=0)

        main.addWidget(left_widget, stretch=1)
        main.addWidget(right_widget, stretch=0)

    def _add_bottom_layer(self, title, build_group):
        """Add one independently scrollable control layer to the bottom tabs."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        controls = QWidget()
        self.ctrl = QVBoxLayout(controls)
        self.ctrl.setContentsMargins(6, 6, 6, 6)
        build_group()
        self.ctrl.addStretch(1)

        scroll.setWidget(controls)
        self.style_tabs.addTab(scroll, title)

    def _build_model_group(self):
        box = QGroupBox("Model")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        self.import_yaml_button = QPushButton("Import YAML and recalculate")
        self.import_yaml_button.clicked.connect(self.import_yaml_and_recalculate)

        self.view_yaml_button = QPushButton("View active YAML")
        self.view_yaml_button.clicked.connect(self.show_active_yaml)

        self.parameter_source = QComboBox()
        self.parameter_source.addItem("YAML", "yaml")
        self.parameter_source.addItem("Full manual", "manual")
        if self.yaml_path is None:
            self.parameter_source.model().item(0).setEnabled(False)
            self.parameter_source.setCurrentIndex(1)
            self.view_yaml_button.setEnabled(False)

        self.use_visual_sqz = QCheckBox("Use visual/schematic squeezing [dB]")
        self.use_visual_sqz.setChecked(False)

        self.visual_sqz_db = QDoubleSpinBox()
        self.visual_sqz_db.setRange(0.0, 20.0)
        self.visual_sqz_db.setDecimals(2)
        self.visual_sqz_db.setValue(6.00)

        self.physical_sqz_label = QLabel(
            f"{self.params['sqz_db']:.3f}" if self.yaml_path is not None else "—"
        )

        self.manual_sqz_db = QDoubleSpinBox()
        self.manual_sqz_db.setRange(0.0, 40.0)
        self.manual_sqz_db.setDecimals(3)
        self.manual_sqz_db.setSingleStep(0.1)

        self.manual_sqz_angle_deg = QDoubleSpinBox()
        self.manual_sqz_angle_deg.setRange(-360.0, 360.0)
        self.manual_sqz_angle_deg.setDecimals(3)
        self.manual_sqz_angle_deg.setSingleStep(0.5)

        self.manual_inj_loss = QDoubleSpinBox()
        self.manual_inj_loss.setRange(0.0, 1.0)
        self.manual_inj_loss.setDecimals(6)
        self.manual_inj_loss.setSingleStep(0.005)

        self.manual_fc_L_m = QDoubleSpinBox()
        self.manual_fc_L_m.setRange(0.001, 100000.0)
        self.manual_fc_L_m.setDecimals(4)
        self.manual_fc_L_m.setSingleStep(0.1)

        self.manual_fc_Ti = QDoubleSpinBox()
        self.manual_fc_Te = QDoubleSpinBox()
        self.manual_fc_Lrt = QDoubleSpinBox()
        for spin in (self.manual_fc_Ti, self.manual_fc_Te, self.manual_fc_Lrt):
            spin.setRange(0.0, 1.0)
            spin.setDecimals(9)
            spin.setSingleStep(0.000001)

        self.manual_fc_detune_hz = QDoubleSpinBox()
        self.manual_fc_detune_hz.setRange(-10000.0, 10000.0)
        self.manual_fc_detune_hz.setDecimals(4)
        self.manual_fc_detune_hz.setSingleStep(1.0)

        self.manual_params_box = QGroupBox("Manual YAML-equivalent parameters")
        manual_grid = QGridLayout(self.manual_params_box)
        manual_grid.setHorizontalSpacing(6)
        manual_grid.setVerticalSpacing(4)
        manual_rows = [
            ("Squeeze [dB]", self.manual_sqz_db, "Angle [deg]", self.manual_sqz_angle_deg),
            ("Inj. loss", self.manual_inj_loss, "FC L [m]", self.manual_fc_L_m),
            ("FC Ti", self.manual_fc_Ti, "FC Te", self.manual_fc_Te),
            ("FC Lrt", self.manual_fc_Lrt, "Detune [Hz]", self.manual_fc_detune_hz),
        ]
        for row, (label_a, widget_a, label_b, widget_b) in enumerate(manual_rows):
            manual_grid.addWidget(QLabel(label_a), row, 0)
            manual_grid.addWidget(widget_a, row, 1)
            manual_grid.addWidget(QLabel(label_b), row, 2)
            manual_grid.addWidget(widget_b, row, 3)
        manual_grid.setColumnStretch(1, 1)
        manual_grid.setColumnStretch(3, 1)
        self._load_manual_parameters(self.params)

        self.include_inj_loss = QCheckBox("Apply selected InjectionLoss (lumped input loss)")
        self.include_inj_loss.setChecked(True)

        self.state_definition_label = QLabel()
        self.state_definition_label.setWordWrap(True)

        self.fds_freqs = QLineEdit("10,30,60,150,2000")

        self.fc_sign = QComboBox()
        self.fc_sign.addItem("+ FC rotation", +1.0)
        self.fc_sign.addItem("- FC rotation", -1.0)
        self.fc_sign.setCurrentIndex(0)

        self.fc_offset = QDoubleSpinBox()
        self.fc_offset.setRange(-180.0, 180.0)
        self.fc_offset.setDecimals(2)
        self.fc_offset.setSingleStep(0.5)
        self.fc_offset.setValue(0.00)

        self.zero_high_f = QCheckBox("Set FC rotation to zero at high frequency")
        self.zero_high_f.setChecked(True)

        self.fis_extra_angle = QDoubleSpinBox()
        self.fis_extra_angle.setRange(-180.0, 180.0)
        self.fis_extra_angle.setDecimals(2)
        self.fis_extra_angle.setValue(0.00)

        self.fds_extra_angle = QDoubleSpinBox()
        self.fds_extra_angle.setRange(-180.0, 180.0)
        self.fds_extra_angle.setDecimals(2)
        self.fds_extra_angle.setValue(0.00)

        form.addRow("Parameter source", self.parameter_source)
        form.addRow(self.import_yaml_button)
        form.addRow(self.view_yaml_button)
        form.addRow("Loaded YAML squeezing [dB]", self.physical_sqz_label)
        form.addRow(self.manual_params_box)
        form.addRow(self.use_visual_sqz)
        form.addRow("Visual squeezing [dB]", self.visual_sqz_db)
        form.addRow(self.include_inj_loss)
        form.addRow("Ellipse model", self.state_definition_label)
        form.addRow("FDS frequencies [Hz]", self.fds_freqs)
        form.addRow("FC rotation sign", self.fc_sign)
        form.addRow("FC rotation offset [deg]", self.fc_offset)
        form.addRow(self.zero_high_f)
        form.addRow("Extra FIS angle [deg]", self.fis_extra_angle)
        form.addRow("Extra FDS angle [deg]", self.fds_extra_angle)

        self.include_inj_loss.toggled.connect(self._update_state_definition)
        self.use_visual_sqz.toggled.connect(self._update_state_definition)
        self.visual_sqz_db.valueChanged.connect(self._update_state_definition)
        self.parameter_source.currentIndexChanged.connect(self._parameter_source_changed)
        for spin in (
            self.manual_sqz_db,
            self.manual_sqz_angle_deg,
            self.manual_inj_loss,
            self.manual_fc_L_m,
            self.manual_fc_Ti,
            self.manual_fc_Te,
            self.manual_fc_Lrt,
            self.manual_fc_detune_hz,
        ):
            spin.valueChanged.connect(self._update_state_definition)
        self._parameter_source_changed()
        self._update_state_definition()

        self.ctrl.addWidget(box)

    def _load_manual_parameters(self, params):
        self.manual_sqz_db.setValue(float(params["sqz_db"]))
        self.manual_sqz_angle_deg.setValue(float(params["sqz_angle_deg"]))
        self.manual_inj_loss.setValue(float(params["inj_loss"]))
        self.manual_fc_L_m.setValue(float(params["fc_L_m"]))
        self.manual_fc_Ti.setValue(float(params["fc_Ti"]))
        self.manual_fc_Te.setValue(float(params["fc_Te"]))
        self.manual_fc_Lrt.setValue(float(params["fc_Lrt"]))
        self.manual_fc_detune_hz.setValue(float(params["fc_detune_Hz"]))

    def _active_params(self):
        if self.parameter_source.currentData() != "manual":
            return dict(self.params)
        return {
            "sqz_db": float(self.manual_sqz_db.value()),
            "inj_loss": float(self.manual_inj_loss.value()),
            "sqz_angle_deg": float(self.manual_sqz_angle_deg.value()),
            "fc_L_m": float(self.manual_fc_L_m.value()),
            "fc_Ti": float(self.manual_fc_Ti.value()),
            "fc_Te": float(self.manual_fc_Te.value()),
            "fc_Lrt": float(self.manual_fc_Lrt.value()),
            "fc_detune_Hz": float(self.manual_fc_detune_hz.value()),
        }

    def _parameter_source_changed(self, *_):
        use_manual = self.parameter_source.currentData() == "manual"
        self.manual_params_box.setVisible(use_manual)
        self._update_state_definition()

    def _update_state_definition(self, *_):
        params = self._active_params()
        source = "manual" if self.parameter_source.currentData() == "manual" else "YAML"
        sqz_db = (
            float(self.visual_sqz_db.value())
            if self.use_visual_sqz.isChecked()
            else float(params["sqz_db"])
        )
        if self.include_inj_loss.isChecked():
            loss = float(np.clip(params["inj_loss"], 0.0, 1.0))
            self.state_definition_label.setText(
                f"{source}: post-loss input approximation; {sqz_db:.2f} dB, "
                f"loss {100.0 * loss:.2f}%. FDS adds FC rotation only."
            )
        else:
            self.state_definition_label.setText(
                f"{source}: ideal pre-loss OPO state; {sqz_db:.2f} dB. "
                "FDS adds FC rotation only."
            )

    def _build_model_equations_group(self, parent_layout):
        box = QGroupBox("Equations used for the ellipses")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(7, 5, 7, 5)
        layout.setSpacing(4)

        loss_formula = QLabel(
            "<b>Injection loss</b><br>"
            "V<sub>sqz</sub>(θ) = R(θ) diag(10<sup>−s/10</sup>, 10<sup>s/10</sup>) R<sup>T</sup>(θ)<br>"
            "η = 1 − L<sub>inj</sub><br>"
            "V<sub>loss</sub> = ηV<sub>sqz</sub> + (1 − η)I"
        )
        loss_formula.setWordWrap(True)

        fc_formula = QLabel(
            "<b>Filter-cavity rotation</b><br>"
            "θ<sub>FC</sub>(f) = ½[arg r(Δ+f) + arg r(Δ−f)]<br>"
            "r(δ) = (r₁ − r₂e<sup>iφ</sup>)/(1 − r₁r₂e<sup>iφ</sup>), "
            "φ = 4πLδ/c<br>"
            "r₁ = √(1−T<sub>i</sub>), "
            "r₂ = √[(1−T<sub>e</sub>)(1−L<sub>rt</sub>)]<br>"
            "θ<sub>rot</sub> = sign·[θ<sub>FC</sub>(f) − θ<sub>FC</sub>(f<sub>HF</sub>)] + θ<sub>offset</sub><br>"
            "θ<sub>FDS</sub> = θ<sub>SQZ</sub> + θ<sub>rot</sub> + θ<sub>extra</sub>"
        )
        fc_formula.setWordWrap(True)

        phase_only_note = QLabel(
            "The fHF term is omitted when high-frequency zeroing is off. "
            "FC magnitude is not propagated; this model uses its phase rotation only."
        )
        phase_only_note.setWordWrap(True)
        phase_only_note.setStyleSheet("color: #666666; font-size: 9pt;")

        layout.addWidget(loss_formula)
        layout.addWidget(fc_formula)
        layout.addWidget(phase_only_note)
        parent_layout.addWidget(box, stretch=0)

    def show_active_yaml(self):
        if self.yaml_path is None:
            QMessageBox.information(
                self,
                "No YAML loaded",
                "No YAML file is active. Import one or use the full manual inputs.",
            )
            return

        try:
            raw_yaml = self.yaml_path.read_text()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "YAML read failed",
                f"Could not read:\n{self.yaml_path}\n\n{exc}",
            )
            return

        p = self.params
        msg = QMessageBox(self)
        msg.setWindowTitle("Active YAML and ellipse inputs")
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"Active YAML:\n{self.yaml_path}")
        msg.setInformativeText(
            f"Current parameter source = {self.parameter_source.currentText()}\n"
            f"Squeezer.AmplitudedB = {p['sqz_db']:.6g} dB\n"
            f"Squeezer.InjectionLoss = {p['inj_loss']:.6g}\n"
            f"Squeezer.SQZAngle = {p['sqz_angle_deg']:.6g} deg\n"
            f"FilterCavity.fdetune = {p['fc_detune_Hz']:.6g} Hz\n\n"
            "Select Show Details to inspect the complete YAML."
        )
        msg.setDetailedText(raw_yaml)
        msg.exec_()

    def import_yaml_and_recalculate(self):
        start_dir = str(self.yaml_path.parent) if self.yaml_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import model YAML",
            start_dir,
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return

        yaml_path = Path(path).expanduser().resolve()
        try:
            params = load_params_from_yaml(yaml_path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "YAML import failed",
                f"Could not load:\n{yaml_path}\n\n{exc}",
            )
            return

        self.yaml_path = yaml_path
        self.params = params
        self.yaml_label.setText(yaml_path.name)
        self.yaml_label.setToolTip(str(yaml_path))
        self.physical_sqz_label.setText(f"{params['sqz_db']:.3f}")
        self.parameter_source.model().item(0).setEnabled(True)
        self.view_yaml_button.setEnabled(True)
        self._load_manual_parameters(params)
        self.parameter_source.setCurrentIndex(0)
        self._update_state_definition()
        self.redraw()

    def _build_visibility_group(self):
        box = QGroupBox("Visible panels")
        form = QFormLayout(box)

        self.show_panel1 = QCheckBox("Show panel 1")
        self.show_panel1.setChecked(True)
        self.show_panel2 = QCheckBox("Show panel 2")
        self.show_panel2.setChecked(True)
        self.show_panel3 = QCheckBox("Show panel 3")
        self.show_panel3.setChecked(True)

        form.addRow(self.show_panel1)
        form.addRow(self.show_panel2)
        form.addRow(self.show_panel3)

        # What should happen to the remaining panels when one is hidden?
        self.rescale_panels = QRadioButton("Rescale remaining panels to fill the row")
        self.keep_panel_slots = QRadioButton("Keep original 3-panel size / slots")
        self.rescale_panels.setChecked(True)

        self.panel_layout_group = QButtonGroup(self)
        self.panel_layout_group.setExclusive(True)
        self.panel_layout_group.addButton(self.rescale_panels)
        self.panel_layout_group.addButton(self.keep_panel_slots)

        form.addRow(QLabel("When panels are hidden:"))
        form.addRow(self.rescale_panels)
        form.addRow(self.keep_panel_slots)

        self.ctrl.addWidget(box)

    def _build_figure_group(self):
        box = QGroupBox("Figure layout")
        layout = QHBoxLayout(box)
        layout.setSpacing(10)

        self.fig_w = QDoubleSpinBox()
        self.fig_h = QDoubleSpinBox()
        for w, val in [(self.fig_w, 13.50), (self.fig_h, 3.60)]:
            w.setRange(2.0, 30.0)
            w.setDecimals(2)
            w.setSingleStep(0.1)
            w.setValue(val)

        self.wspace = QDoubleSpinBox()
        self.wspace.setRange(-0.80, 3.0)
        self.wspace.setDecimals(2)
        self.wspace.setSingleStep(0.02)
        self.wspace.setValue(0.12)

        self.left = QDoubleSpinBox()
        self.right = QDoubleSpinBox()
        self.bottom = QDoubleSpinBox()
        self.top = QDoubleSpinBox()
        for w, val in [
            (self.left, 0.045),
            (self.right, 0.88),
            (self.bottom, 0.11),
            (self.top, 0.94),
        ]:
            w.setRange(0.0, 1.5)
            w.setDecimals(2)
            w.setSingleStep(0.02)
            w.setValue(val)

        self.use_common_square_limits = QCheckBox("Override panel limits with one common square axis")
        self.use_common_square_limits.setChecked(False)
        self.use_common_square_limits.setToolTip(
            "When enabled, the common half-range replaces all individual panel limits."
        )

        self.common_square_limit = QDoubleSpinBox()
        self.common_square_limit.setRange(0.01, 100.0)
        self.common_square_limit.setDecimals(2)
        self.common_square_limit.setSingleStep(0.10)
        self.common_square_limit.setValue(5.00)

        self.show_ticks = QCheckBox("Show ticks")
        self.show_ticks.setChecked(False)

        self.show_grid = QCheckBox("Show grid")
        self.show_grid.setChecked(False)

        self.show_crosshairs = QCheckBox("Show axis crosshairs")
        self.show_crosshairs.setChecked(True)

        self.grid_alpha = QDoubleSpinBox()
        self.grid_alpha.setRange(0.0, 1.0)
        self.grid_alpha.setDecimals(2)
        self.grid_alpha.setSingleStep(0.05)
        self.grid_alpha.setValue(0.18)

        size_box = QGroupBox("Size / spacing")
        size_form = QFormLayout(size_box)
        size_form.setVerticalSpacing(4)
        size_form.addRow("Export width [in]", self.fig_w)
        size_form.addRow("Export height [in]", self.fig_h)
        size_form.addRow("Panel spacing", self.wspace)

        margins_box = QGroupBox("Margins")
        margins_form = QFormLayout(margins_box)
        margins_form.setVerticalSpacing(4)
        margins_form.addRow("Left", self.left)
        margins_form.addRow("Right", self.right)
        margins_form.addRow("Bottom", self.bottom)
        margins_form.addRow("Top", self.top)

        axes_box = QGroupBox("Axes")
        axes_form = QFormLayout(axes_box)
        axes_form.setVerticalSpacing(4)
        axes_form.addRow(self.use_common_square_limits)
        axes_form.addRow("Common half-range", self.common_square_limit)
        axes_form.addRow(self.show_ticks)
        axes_form.addRow(self.show_grid)
        axes_form.addRow(self.show_crosshairs)
        axes_form.addRow("Grid alpha", self.grid_alpha)

        self.use_common_square_limits.toggled.connect(self._sync_limit_mode)
        self._sync_limit_mode()

        layout.addWidget(size_box, stretch=1)
        layout.addWidget(margins_box, stretch=1)
        layout.addWidget(axes_box, stretch=1)

        self.ctrl.addWidget(box)

    def _sync_limit_mode(self, *_):
        """Make it explicit which set of axis-limit controls is active."""
        use_common = self.use_common_square_limits.isChecked()
        self.common_square_limit.setEnabled(use_common)
        for limits in (self.lim1, self.lim2, self.lim3):
            for spinbox in (limits.xmin, limits.xmax, limits.ymin, limits.ymax):
                spinbox.setEnabled(not use_common)

    def _build_text_group(self):
        box = QGroupBox("Titles / labels / fonts")
        layout = QHBoxLayout(box)
        layout.setSpacing(10)

        self.title1 = QLineEdit("Vacuum")
        self.title2 = QLineEdit("FIS")
        self.title3 = QLineEdit("FDS")

        self.xlabel = QLineEdit(r"$X_1$")
        self.ylabel = QLineEdit(r"$X_2$")
        self.font_family = QLineEdit("Arial")

        self.title_fs = QDoubleSpinBox()
        self.label_fs = QDoubleSpinBox()
        self.tick_fs = QDoubleSpinBox()
        for w, val in [
            (self.title_fs, 12.0),
            (self.label_fs, 11.0),
            (self.tick_fs, 9.0),
        ]:
            w.setRange(4.0, 50.0)
            w.setDecimals(1)
            w.setSingleStep(0.5)
            w.setValue(val)

        self.title_weight = QComboBox()
        self.label_weight = QComboBox()
        self.tick_weight = QComboBox()
        self.legend_weight = QComboBox()
        for combo in [self.title_weight, self.label_weight, self.tick_weight, self.legend_weight]:
            combo.addItems(FONT_WEIGHTS)
        set_combo_text(self.title_weight, "normal")
        set_combo_text(self.label_weight, "normal")
        set_combo_text(self.tick_weight, "normal")
        set_combo_text(self.legend_weight, "normal")

        self.title_pad = QDoubleSpinBox()
        self.xlabel_pad = QDoubleSpinBox()
        self.ylabel_pad = QDoubleSpinBox()
        for w, val in [
            (self.title_pad, 8.0),
            (self.xlabel_pad, 4.0),
            (self.ylabel_pad, 4.0),
        ]:
            w.setRange(-30.0, 60.0)
            w.setDecimals(1)
            w.setSingleStep(1.0)
            w.setValue(val)

        titles_box = QGroupBox("Titles")
        titles_form = QFormLayout(titles_box)
        titles_form.setVerticalSpacing(4)
        titles_form.addRow("Panel 1", self.title1)
        titles_form.addRow("Panel 2", self.title2)
        titles_form.addRow("Panel 3", self.title3)
        titles_form.addRow("Font size", self.title_fs)
        titles_form.addRow("Weight", self.title_weight)
        titles_form.addRow("Pad", self.title_pad)

        labels_box = QGroupBox("Axis labels")
        labels_form = QFormLayout(labels_box)
        labels_form.setVerticalSpacing(4)
        labels_form.addRow("x label", self.xlabel)
        labels_form.addRow("y label", self.ylabel)
        labels_form.addRow("Font size", self.label_fs)
        labels_form.addRow("Weight", self.label_weight)
        labels_form.addRow("x pad", self.xlabel_pad)
        labels_form.addRow("y pad", self.ylabel_pad)

        fonts_box = QGroupBox("Fonts / ticks / legend")
        fonts_form = QFormLayout(fonts_box)
        fonts_form.setVerticalSpacing(4)
        fonts_form.addRow("Font family", self.font_family)
        fonts_form.addRow("Tick font size", self.tick_fs)
        fonts_form.addRow("Tick weight", self.tick_weight)
        fonts_form.addRow("Legend weight", self.legend_weight)

        layout.addWidget(titles_box, stretch=1)
        layout.addWidget(labels_box, stretch=1)
        layout.addWidget(fonts_box, stretch=1)

        self.ctrl.addWidget(box)

    def _build_style_group(self):
        box = QGroupBox("Ellipse style")
        layout = QHBoxLayout(box)
        layout.setSpacing(10)

        self.vac_color = QComboBox()
        self.fis_color = QComboBox()
        self.fds_color = QComboBox()
        for combo in [self.vac_color, self.fis_color, self.fds_color]:
            add_color_items(combo)
        set_combo_hex(self.vac_color, "#000060")
        set_combo_hex(self.fis_color, "#8185ff")
        set_combo_hex(self.fds_color, "#d900ff")

        self.vac_lw = QDoubleSpinBox()
        self.fis_lw = QDoubleSpinBox()
        self.fds_lw = QDoubleSpinBox()
        for w, val in [(self.vac_lw, 1.8), (self.fis_lw, 2.2), (self.fds_lw, 2.2)]:
            w.setRange(0.1, 10.0)
            w.setDecimals(1)
            w.setSingleStep(0.2)
            w.setValue(val)

        self.vac_linestyle = QComboBox()
        self.fis_linestyle = QComboBox()
        self.fds_linestyle = QComboBox()
        for combo in [self.vac_linestyle, self.fis_linestyle, self.fds_linestyle]:
            add_line_style_items(combo)
        self.vac_linestyle.setCurrentIndex(0)
        self.fis_linestyle.setCurrentIndex(0)
        self.fds_linestyle.setCurrentIndex(0)

        self.overlay_vac_on_fis = QCheckBox("Show vacuum circle beneath FIS trace")
        self.overlay_vac_on_fis.setChecked(False)

        self.overlay_vac_on_fds = QCheckBox("Show vacuum circle beneath FDS traces")
        self.overlay_vac_on_fds.setChecked(False)

        self.vac_alpha = QDoubleSpinBox()
        self.fis_alpha = QDoubleSpinBox()
        self.fds_alpha_min = QDoubleSpinBox()
        self.fds_alpha_max = QDoubleSpinBox()
        for w, val in [
            (self.vac_alpha, 1.00),
            (self.fis_alpha, 0.95),
            (self.fds_alpha_min, 0.15),
            (self.fds_alpha_max, 0.95),
        ]:
            w.setRange(0.0, 1.0)
            w.setDecimals(2)
            w.setSingleStep(0.05)
            w.setValue(val)

        self.vac_fill_alpha = QDoubleSpinBox()
        self.fis_fill_alpha = QDoubleSpinBox()
        self.fds_fill_alpha = QDoubleSpinBox()
        for w, val in [
            (self.vac_fill_alpha, 0.08),
            (self.fis_fill_alpha, 0.10),
            (self.fds_fill_alpha, 0.08),
        ]:
            w.setRange(0.0, 1.0)
            w.setDecimals(2)
            w.setSingleStep(0.05)
            w.setValue(val)

        appearance_box = QGroupBox("Line appearance")
        appearance = QGridLayout(appearance_box)
        appearance.setVerticalSpacing(4)
        appearance.addWidget(QLabel(""), 0, 0)
        appearance.addWidget(QLabel("Vacuum"), 0, 1)
        appearance.addWidget(QLabel("FIS"), 0, 2)
        appearance.addWidget(QLabel("FDS"), 0, 3)
        appearance.addWidget(QLabel("Color"), 1, 0)
        appearance.addWidget(self.vac_color, 1, 1)
        appearance.addWidget(self.fis_color, 1, 2)
        appearance.addWidget(self.fds_color, 1, 3)
        appearance.addWidget(QLabel("Width"), 2, 0)
        appearance.addWidget(self.vac_lw, 2, 1)
        appearance.addWidget(self.fis_lw, 2, 2)
        appearance.addWidget(self.fds_lw, 2, 3)
        appearance.addWidget(QLabel("Style"), 3, 0)
        appearance.addWidget(self.vac_linestyle, 3, 1)
        appearance.addWidget(self.fis_linestyle, 3, 2)
        appearance.addWidget(self.fds_linestyle, 3, 3)
        appearance.addWidget(self.overlay_vac_on_fis, 4, 0, 1, 4)
        appearance.addWidget(self.overlay_vac_on_fds, 5, 0, 1, 4)

        opacity_box = QGroupBox("Opacity")
        opacity = QGridLayout(opacity_box)
        opacity.setVerticalSpacing(4)
        opacity.addWidget(QLabel(""), 0, 0)
        opacity.addWidget(QLabel("Vacuum"), 0, 1)
        opacity.addWidget(QLabel("FIS"), 0, 2)
        opacity.addWidget(QLabel("FDS"), 0, 3)
        opacity.addWidget(QLabel("Line alpha"), 1, 0)
        opacity.addWidget(self.vac_alpha, 1, 1)
        opacity.addWidget(self.fis_alpha, 1, 2)

        fds_alpha_widget = QWidget()
        fds_alpha_layout = QHBoxLayout(fds_alpha_widget)
        fds_alpha_layout.setContentsMargins(0, 0, 0, 0)
        fds_alpha_layout.setSpacing(3)
        fds_alpha_layout.addWidget(QLabel("min"))
        fds_alpha_layout.addWidget(self.fds_alpha_min)
        fds_alpha_layout.addWidget(QLabel("max"))
        fds_alpha_layout.addWidget(self.fds_alpha_max)
        opacity.addWidget(fds_alpha_widget, 1, 3)

        opacity.addWidget(QLabel("Fill alpha"), 2, 0)
        opacity.addWidget(self.vac_fill_alpha, 2, 1)
        opacity.addWidget(self.fis_fill_alpha, 2, 2)
        opacity.addWidget(self.fds_fill_alpha, 2, 3)

        appearance_box.setMaximumWidth(510)
        layout.addWidget(appearance_box, stretch=0)
        layout.addWidget(opacity_box, stretch=1)

        self.ctrl.addWidget(box)

    def _build_limits_group(self):
        box = QGroupBox("Panels / limits")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        limits_row = QHBoxLayout()
        limits_row.setSpacing(10)

        self.lim1 = LimitsControls("Panel 1 limits", (-5.0, 5.0), (-5.0, 5.0))
        self.lim2 = LimitsControls("Panel 2 limits", (-5.0, 5.0), (-5.0, 5.0))
        self.lim3 = LimitsControls("Panel 3 limits", (-5.0, 3.9), (-5.0, 5.0))

        self.show_panel1 = QCheckBox("Show panel 1")
        self.show_panel2 = QCheckBox("Show panel 2")
        self.show_panel3 = QCheckBox("Show panel 3")
        self.show_panel1.setChecked(True)
        self.show_panel2.setChecked(True)
        self.show_panel3.setChecked(True)

        for limits, show_checkbox in [
            (self.lim1, self.show_panel1),
            (self.lim2, self.show_panel2),
            (self.lim3, self.show_panel3),
        ]:
            limits.box.setMaximumWidth(255)
            limits.box.layout().addWidget(show_checkbox, 2, 0, 1, 4)
            limits_row.addWidget(limits.box, stretch=0)
        limits_row.addStretch(1)
        layout.addLayout(limits_row)

        hidden_box = QGroupBox("When panels are hidden")
        hidden_layout = QHBoxLayout(hidden_box)
        self.rescale_panels = QRadioButton("Rescale remaining panels")
        self.keep_panel_slots = QRadioButton("Keep original panel slots")
        self.rescale_panels.setChecked(True)

        self.panel_layout_group = QButtonGroup(self)
        self.panel_layout_group.setExclusive(True)
        self.panel_layout_group.addButton(self.rescale_panels)
        self.panel_layout_group.addButton(self.keep_panel_slots)

        hidden_layout.addWidget(self.rescale_panels)
        hidden_layout.addWidget(self.keep_panel_slots)
        hidden_layout.addStretch(1)

        hidden_row = QHBoxLayout()
        hidden_row.addWidget(hidden_box, stretch=0)
        hidden_row.addStretch(1)
        layout.addLayout(hidden_row)

        self.ctrl.addWidget(box)

    def _build_legend_groups(self):
        box = QGroupBox("Panel legends")
        layout = QHBoxLayout(box)
        layout.setSpacing(10)

        self.vac_label = QLineEdit("Vacuum")
        self.fis_label = QLineEdit("FIS")
        self.fds_label_format = QLineEdit("{f:g} Hz")
        self.fds_custom_labels = QLineEdit("")
        self.fds_custom_labels.setPlaceholderText("Optional labels")

        self.legend_frame = QCheckBox("Legend frame")
        self.legend_frame.setChecked(True)

        self.legend_framealpha = QDoubleSpinBox()
        self.legend_framealpha.setRange(0.0, 1.0)
        self.legend_framealpha.setDecimals(2)
        self.legend_framealpha.setSingleStep(0.05)
        self.legend_framealpha.setValue(0.92)

        self.leg1 = PanelLegendControls("Panel 1 legend", default_loc="upper right", default_x=1.0, default_y=1.0)
        self.leg2 = PanelLegendControls("Panel 2 legend", default_loc="upper right", default_x=1.0, default_y=1.0)
        self.leg3 = PanelLegendControls("Panel 3 legend", default_loc="lower right", default_x=1.2, default_y=0.0)
        for legend in [self.leg1, self.leg2, self.leg3]:
            legend.box.setMaximumWidth(240)
            layout.addWidget(legend.box, stretch=0)

        legend_text_box = QGroupBox("Legend text / frame")
        legend_text_form = QFormLayout(legend_text_box)
        legend_text_form.setVerticalSpacing(4)
        legend_text_form.addRow("Panel 1 label", self.vac_label)
        legend_text_form.addRow("Panel 2 label", self.fis_label)
        legend_text_form.addRow("Panel 3 format", self.fds_label_format)
        legend_text_form.addRow("Panel 3 custom", self.fds_custom_labels)
        legend_text_form.addRow(self.legend_frame)
        legend_text_form.addRow("Frame alpha", self.legend_framealpha)
        legend_text_box.setMaximumWidth(290)
        layout.addWidget(legend_text_box, stretch=0)
        layout.addStretch(1)

        self.ctrl.addWidget(box)

    def _build_fixed_actions_group(self, parent_layout):
        box = QGroupBox("Actions")
        v = QVBoxLayout(box)

        btn_redraw = QPushButton("Apply controls / redraw")
        btn_export = QPushButton("Export")
        btn_save_defaults = QPushButton("Save PDF/PNG/SVG here")

        btn_redraw.clicked.connect(self.redraw)
        btn_export.clicked.connect(self.export_figure)
        btn_save_defaults.clicked.connect(self.save_defaults_here)

        v.addWidget(btn_redraw)
        v.addWidget(btn_export)
        v.addWidget(btn_save_defaults)

        parent_layout.addWidget(box, stretch=0)

    def gather_settings(self):
        freqs = parse_freq_list(self.fds_freqs.text())
        if not freqs:
            freqs = [10.0, 30.0, 60.0, 150.0, 2000.0]

        active_params = self._active_params()

        return dict(
            params=active_params,
            parameter_source=str(self.parameter_source.currentData()),
            use_visual_sqz=self.use_visual_sqz.isChecked(),
            visual_sqz_db=float(self.visual_sqz_db.value()),
            include_injection_loss=self.include_inj_loss.isChecked(),
            fds_freqs_hz=freqs,
            fc_sign=float(self.fc_sign.currentData()),
            fc_offset_deg=float(self.fc_offset.value()),
            zero_fc_rotation_at_high_f=self.zero_high_f.isChecked(),
            fis_extra_angle_deg=float(self.fis_extra_angle.value()),
            fds_extra_angle_deg=float(self.fds_extra_angle.value()),

            show_panels=[
                self.show_panel1.isChecked(),
                self.show_panel2.isChecked(),
                self.show_panel3.isChecked(),
            ],
            rescale_hidden_panels=self.rescale_panels.isChecked(),

            fig_w=float(self.fig_w.value()),
            fig_h=float(self.fig_h.value()),
            wspace=float(self.wspace.value()),
            left=float(self.left.value()),
            right=float(self.right.value()),
            bottom=float(self.bottom.value()),
            top=float(self.top.value()),

            use_common_square_limits=self.use_common_square_limits.isChecked(),
            common_square_limit=float(self.common_square_limit.value()),

            show_ticks=self.show_ticks.isChecked(),
            show_grid=self.show_grid.isChecked(),
            show_crosshairs=self.show_crosshairs.isChecked(),
            grid_alpha=float(self.grid_alpha.value()),

            titles=[self.title1.text(), self.title2.text(), self.title3.text()],
            xlabel=self.xlabel.text(),
            ylabel=self.ylabel.text(),
            font_family=self.font_family.text(),
            title_fontsize=float(self.title_fs.value()),
            label_fontsize=float(self.label_fs.value()),
            tick_fontsize=float(self.tick_fs.value()),
            title_fontweight=self.title_weight.currentText(),
            label_fontweight=self.label_weight.currentText(),
            tick_fontweight=self.tick_weight.currentText(),
            legend_fontweight=self.legend_weight.currentText(),
            title_pad=float(self.title_pad.value()),
            xlabel_pad=float(self.xlabel_pad.value()),
            ylabel_pad=float(self.ylabel_pad.value()),

            vac_color=self.vac_color.currentData(),
            fis_color=self.fis_color.currentData(),
            fds_color=self.fds_color.currentData(),
            vac_lw=float(self.vac_lw.value()),
            fis_lw=float(self.fis_lw.value()),
            fds_lw=float(self.fds_lw.value()),
            vac_linestyle=self.vac_linestyle.currentData(),
            fis_linestyle=self.fis_linestyle.currentData(),
            fds_linestyle=self.fds_linestyle.currentData(),
            overlay_vac_on_fis=self.overlay_vac_on_fis.isChecked(),
            overlay_vac_on_fds=self.overlay_vac_on_fds.isChecked(),
            vac_alpha=float(self.vac_alpha.value()),
            fis_alpha=float(self.fis_alpha.value()),
            fds_alpha_min=float(self.fds_alpha_min.value()),
            fds_alpha_max=float(self.fds_alpha_max.value()),
            vac_fill_alpha=float(self.vac_fill_alpha.value()),
            fis_fill_alpha=float(self.fis_fill_alpha.value()),
            fds_fill_alpha=float(self.fds_fill_alpha.value()),

            vac_label=self.vac_label.text(),
            fis_label=self.fis_label.text(),
            fds_label_format=self.fds_label_format.text(),
            fds_custom_labels=parse_text_list(self.fds_custom_labels.text()),

            legend_frame=self.legend_frame.isChecked(),
            legend_framealpha=float(self.legend_framealpha.value()),

            xlims=[
                (float(self.lim1.xmin.value()), float(self.lim1.xmax.value())),
                (float(self.lim2.xmin.value()), float(self.lim2.xmax.value())),
                (float(self.lim3.xmin.value()), float(self.lim3.xmax.value())),
            ],
            ylims=[
                (float(self.lim1.ymin.value()), float(self.lim1.ymax.value())),
                (float(self.lim2.ymin.value()), float(self.lim2.ymax.value())),
                (float(self.lim3.ymin.value()), float(self.lim3.ymax.value())),
            ],

            show_legend=[
                self.leg1.show.isChecked(),
                self.leg2.show.isChecked(),
                self.leg3.show.isChecked(),
            ],
            legend_loc=[
                self.leg1.loc.currentText(),
                self.leg2.loc.currentText(),
                self.leg3.loc.currentText(),
            ],
            legend_x=[
                float(self.leg1.x.value()),
                float(self.leg2.x.value()),
                float(self.leg3.x.value()),
            ],
            legend_y=[
                float(self.leg1.y.value()),
                float(self.leg2.y.value()),
                float(self.leg3.y.value()),
            ],
            legend_fontsize=[
                float(self.leg1.fs.value()),
                float(self.leg2.fs.value()),
                float(self.leg3.fs.value()),
            ],
        )

    def redraw(self):
        try:
            s = self.gather_settings()

            if not any(s["show_panels"]):
                raise ValueError("At least one panel must be selected to display.")

            if s["use_common_square_limits"]:
                if s["common_square_limit"] <= 0:
                    raise ValueError("Common half-range must be positive.")
            else:
                for (xmin, xmax), (ymin, ymax) in zip(s["xlims"], s["ylims"]):
                    if xmin >= xmax or ymin >= ymax:
                        raise ValueError("Each panel must satisfy xmin < xmax and ymin < ymax.")

            if not (0.0 <= s["left"] < s["right"] <= 1.5):
                raise ValueError("Margins must satisfy left < right.")
            if not (0.0 <= s["bottom"] < s["top"] <= 1.5):
                raise ValueError("Margins must satisfy bottom < top.")

            self.fig, _ = build_horizontal_state_figure(s, fig=self.fig, apply_figsize=False)
            self.canvas.draw_idle()
        except Exception as exc:
            QMessageBox.critical(self, "Redraw failed", str(exc))
            raise

    def export_figure(self):
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export squeezing-state ellipse panels",
            str(Path.cwd() / f"{DEFAULT_EXPORT_STEM}.pdf"),
            "PDF (*.pdf);;PNG (*.png);;SVG (*.svg);;All files (*)",
        )
        if not path:
            return

        p = Path(path).expanduser()
        if p.suffix.lower() not in {".pdf", ".png", ".svg"}:
            filt = selected_filter.lower()
            if "png" in filt:
                p = p.with_suffix(".png")
            elif "svg" in filt:
                p = p.with_suffix(".svg")
            else:
                p = p.with_suffix(".pdf")

        try:
            # Save the exact Figure currently displayed in the Qt canvas.
            # Do NOT rebuild the figure, change its figsize, or use
            # bbox_inches="tight", because all three can change the layout.
            self.canvas.draw()
            self.fig.canvas.print_figure(
                str(p),
                dpi=300,
                format=p.suffix.lower().lstrip("."),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        QMessageBox.information(self, "Export complete", f"Saved:\n{p}")

    def save_defaults_here(self):
        try:
            s = self.gather_settings()

            # Compute FC rotations only for the summary text.
            theta_fc = fc_rotation_at_frequencies(
                s["fds_freqs_hz"],
                s["params"],
                sign=s["fc_sign"],
                offset_deg=s["fc_offset_deg"],
                zero_at_high_f=s["zero_fc_rotation_at_high_f"],
            )

            destination = Path.cwd()
            prefix = destination / DEFAULT_EXPORT_STEM

            # Save the exact Figure currently displayed in the Qt canvas.
            self.canvas.draw()
            for ext in ["pdf", "png", "svg"]:
                self.fig.canvas.print_figure(
                    f"{prefix}.{ext}",
                    dpi=300,
                    format=ext,
                )

            summary = (
                f"YAML: {self.yaml_path if self.yaml_path is not None else 'None (manual inputs)'}\n"
                f"Parameter source: {s['parameter_source']}\n"
                f"Active ellipse parameters: {s['params']}\n"
                f"Visible panels: {s['show_panels']}\n"
                f"Rescale remaining panels when hidden: {s['rescale_hidden_panels']}\n"
                f"FDS frequencies [Hz]: {s['fds_freqs_hz']}\n"
                f"FDS FC rotations [deg]: {[float(np.rad2deg(x)) for x in theta_fc]}\n"
                f"Use visual squeezing: {s['use_visual_sqz']}\n"
                f"Visual squeezing [dB]: {s['visual_sqz_db']}\n"
                f"Include injection loss: {s['include_injection_loss']}\n"
                f"Vacuum beneath FIS: {s['overlay_vac_on_fis']}\n"
                f"Vacuum beneath FDS: {s['overlay_vac_on_fds']}\n"
                f"Line styles (vac/FIS/FDS): {s['vac_linestyle']}, {s['fis_linestyle']}, {s['fds_linestyle']}\n"
                f"Horizontal wspace: {s['wspace']}\n"
                f"Use common square limits: {s['use_common_square_limits']}\n"
                f"Common half-range: {s['common_square_limit']}\n"
                f"Font family: {s['font_family']}\n"
                f"Titles: {s['titles']}\n"
            )
            Path(f"{prefix}_summary.txt").write_text(summary)
            QMessageBox.information(
                self,
                "Saved",
                f"Saved PDF/PNG/SVG and summary text to:\n{destination}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            raise


def main():
    ap = argparse.ArgumentParser(
        description="Explore vacuum, frequency-independent, and frequency-dependent squeezing ellipses."
    )
    ap.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help="Optional initial squeezer model YAML",
    )
    args = ap.parse_args()

    yaml_path = args.yaml.expanduser().resolve() if args.yaml is not None else None
    if yaml_path is not None and not yaml_path.exists():
        raise FileNotFoundError(f"Could not find YAML file: {yaml_path}")
    if not QT_AVAILABLE:
        raise RuntimeError(
            "Qt/PyQt could not be imported. Install PyQt5. "
            f"Original import error: {QT_IMPORT_ERROR}"
        )

    print("[squeezing-ellipse-visualizer] Starting GUI...", flush=True)

    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True

    win = Fig2StatePanelsWindow(yaml_path)
    win.show()
    win.raise_()
    win.activateWindow()

    app._squeezing_ellipse_visualizer = win
    globals()["_SQUEEZING_ELLIPSE_VISUALIZER"] = win
    globals()["_SQUEEZING_ELLIPSE_APP"] = app

    if created:
        return app.exec_()
    return win


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        sys.exit(result)
