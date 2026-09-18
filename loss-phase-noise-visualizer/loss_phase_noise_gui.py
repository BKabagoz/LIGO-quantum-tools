#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
loss_phase_noise_gui.py

Qt GUI to tune Figure 4: implementation limits from loss and phase noise.

Layout:
  (a) Loss mixes ordinary vacuum into a squeezed ellipse.
  (b) Phase noise/readout-axis schematic.
  (c) 10 log10(V) vs nonlinear gain, phase noise A.
  (d) 10 log10(V) vs nonlinear gain, phase noise B.

Important modeling note:
  The ellipse panels are schematic OPO/quadrature figures. They are not
  frequency-dependent unless one explicitly supplies a frequency-dependent
  squeezing angle or filter-cavity rotation. Therefore this GUI does not ask
  for an ellipse frequency.

Run from terminal:
    python loss_phase_noise_gui.py

From Spyder:
    %autoreload 0
    %gui qt
    %runfile /path/to/loss_phase_noise_gui.py --wdir
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
# Do not force a backend here; embedded FigureCanvasQTAgg is Qt-based.
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


# =============================================================================
# Physics model
# =============================================================================

def variances_from_nlg(nlg):
    """
    Ideal OPO squeezed and anti-squeezed variances from nonlinear gain G.

        V_sqz  = (sqrt(G) - sqrt(G - 1))^2
        V_anti = (sqrt(G) + sqrt(G - 1))^2

    Variances are normalized to vacuum = 1.
    """
    g = np.asarray(nlg, dtype=float)
    g = np.maximum(g, 1.0 + 1e-12)
    v_sqz = (np.sqrt(g) - np.sqrt(g - 1.0))**2
    v_anti = (np.sqrt(g) + np.sqrt(g - 1.0))**2
    return v_sqz, v_anti


def apply_loss(v, loss_fraction):
    """
    Optical loss mixes in ordinary vacuum:
        V -> eta V + (1 - eta)
    with eta = 1 - loss.
    """
    eta = 1.0 - float(loss_fraction)
    return eta * np.asarray(v) + (1.0 - eta)


def phase_noise_average(v_sqz, v_anti, phase_noise_rad):
    """
    Average measured quadrature variance over zero-mean Gaussian phase jitter.

    For delta theta with RMS sigma:
        <cos^2 delta> = 0.5 * (1 + exp(-2 sigma^2))
        <sin^2 delta> = 0.5 * (1 - exp(-2 sigma^2))
    """
    sigma = float(phase_noise_rad)
    c2 = 0.5 * (1.0 + np.exp(-2.0 * sigma**2))
    s2 = 0.5 * (1.0 - np.exp(-2.0 * sigma**2))
    return v_sqz * c2 + v_anti * s2


def observed_variance(nlg, loss_fraction, phase_noise_mrad):
    v_sqz, v_anti = variances_from_nlg(nlg)
    v_phase = phase_noise_average(v_sqz, v_anti, phase_noise_mrad * 1e-3)
    return apply_loss(v_phase, loss_fraction)


def observed_db(nlg, loss_fraction, phase_noise_mrad):
    return 10.0 * np.log10(observed_variance(nlg, loss_fraction, phase_noise_mrad))


def covariance_from_variances(v_phase, v_amp, angle_deg=0.0):
    """
    Covariance matrix in phase/amplitude plotted coordinates.

    angle_deg rotates the ellipse counterclockwise in the plotted plane.
    """
    th = np.deg2rad(angle_deg)
    r = np.array([
        [np.cos(th), -np.sin(th)],
        [np.sin(th),  np.cos(th)],
    ])
    d = np.diag([v_phase, v_amp])
    return r @ d @ r.T


def ellipse_points_from_cov(cov, npts=700, nsigma=1.0):
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)

    t = np.linspace(0.0, 2.0 * np.pi, int(npts))
    circle = np.vstack([np.cos(t), np.sin(t)])
    e = vecs @ np.diag(np.sqrt(vals)) @ circle
    return nsigma * e[0], nsigma * e[1]


def line_through_origin(angle_deg, length=4.0):
    th = np.deg2rad(angle_deg)
    x = np.array([-length, length]) * np.cos(th)
    y = np.array([-length, length]) * np.sin(th)
    return x, y


def parse_float_list(text: str):
    text = text.strip()
    if not text:
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


# =============================================================================
# Colors
# =============================================================================

DISTINGUISHABLE_COLORS = {
    # Current/default palette
    "deep navy": "#000060",
    "teal": "#008c9e",
    "orange": "#ff7f0e",
    "purple": "#7f1d6a",
    "lavender": "#8185ff",
    "magenta": "#d900ff",
    "green": "#228833",
    "red-orange": "#cc3311",
    "mauve": "#aa4499",
    "sea green": "#44aa99",
    "indigo": "#332288",
    "sand": "#ddcc77",

    # Additional distinguishable-style colors
    "sky blue": "#56B4E9",
    "bluish green": "#009E73",
    "vermillion": "#D55E00",
    "reddish purple": "#CC79A7",
    "blue": "#0072B2",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#7F7F7F",
    "brown": "#8C564B",
    "pink": "#E377C2",
    "olive": "#BCBD22",
    "cyan": "#17BECF",
    "dark green": "#006400",
    "gold": "#DAA520",
    "maroon": "#800000",
    "slate blue": "#6A5ACD",
    "dark orange": "#FF8C00",
    "forest": "#228B22",
    "hot pink": "#FF69B4",
    "dark cyan": "#008B8B",
}

CURVE_COLORS = [
    "#000060",
    "#008c9e",
    "#ff7f0e",
    "#7f1d6a",
    "#8185ff",
    "#d900ff",
    "#228833",
    "#cc3311",
    "#aa4499",
    "#44aa99",
    "#332288",
    "#ddcc77",
]

VACUUM_COLOR = "#888888"
READOUT_COLOR = "#000000"


# =============================================================================
# GUI
# =============================================================================

def write_figure(fig, path):
    """Write and verify an export without relying on Figure.savefig hooks."""
    out = Path(path).expanduser().resolve()
    fmt = out.suffix.lower().lstrip(".")
    signatures = {"pdf": b"%PDF-", "png": b"\x89PNG\r\n\x1a\n", "svg": b"<svg"}
    if fmt not in signatures:
        raise ValueError("Supported export formats are PDF, PNG, and SVG.")
    original_canvas = fig.canvas
    temporary = None
    try:
        # A fresh temporary file prevents an old output from masking a failed save.
        with tempfile.NamedTemporaryFile(dir=out.parent, suffix="." + fmt,
                                         delete=False) as handle:
            temporary = Path(handle.name)
        export_canvas = FigureCanvasAgg(fig)
        export_canvas.print_figure(temporary, format=fmt, dpi=300,
                                   bbox_inches="tight", facecolor=fig.get_facecolor())
        with temporary.open("rb") as handle:
            header = handle.read(1024)
        valid = (signatures[fmt] in header if fmt == "svg"
                 else header.startswith(signatures[fmt]))
        if not valid:
            raise OSError("The renderer did not produce a valid figure file.")
        os.replace(temporary, out)
        if not out.is_file() or out.stat().st_size == 0:
            raise OSError(f"Export file is missing or empty: {out}")
    finally:
        fig.set_canvas(original_canvas)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return out


class ExportToolbar(NavigationToolbar):
    """Use the verified export for the toolbar's save button as well."""
    def save_figure(self, *args):
        self.parent().export_figure()


class Fig4Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Figure 4 loss / phase-noise GUI v7")
        self.resize(1450, 980)

        self._build_ui()
        self.redraw()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        # Plot area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        self.fig = Figure(figsize=(7.25, 6.9), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = ExportToolbar(self.canvas, self)

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main.addWidget(plot_widget, stretch=5)

        # Layered control panel.  Each layer scrolls independently, while the
        # action buttons remain permanently visible below the tabs.
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self.control_tabs = QTabWidget()
        right_layout.addWidget(self.control_tabs, stretch=1)

        self._add_control_layer("General", (self._build_general_group,))
        self._add_control_layer("Ellipse panels", (self._build_ellipse_group,))
        self._add_control_layer("Bottom curve panels", (self._build_curve_group,))
        self._add_control_layer("Legend positions", (self._build_legend_group,))

        self._build_export_group(right_layout)
        main.addWidget(right_widget, stretch=3)

    def _add_control_layer(self, title, builders):
        """Add one independently scrollable layer to the right-hand panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        panel = QWidget()
        self.ctrl = QVBoxLayout(panel)
        self.ctrl.setContentsMargins(4, 4, 4, 4)
        self.ctrl.setSpacing(4)

        for builder in builders:
            builder()
        self.ctrl.addStretch(1)

        scroll.setWidget(panel)
        self.control_tabs.addTab(scroll, title)

    def _build_general_group(self):
        box = QGroupBox("General")
        form = QFormLayout(box)

        self.font_spin = QDoubleSpinBox()
        self.font_spin.setRange(6, 24)
        self.font_spin.setSingleStep(0.5)
        self.font_spin.setValue(15.0)

        self.legend_font_spin = QDoubleSpinBox()
        self.legend_font_spin.setRange(5, 24)
        self.legend_font_spin.setSingleStep(0.5)
        self.legend_font_spin.setValue(11.0)

        self.inset_text_font_spin = QDoubleSpinBox()
        self.inset_text_font_spin.setRange(5, 24)
        self.inset_text_font_spin.setSingleStep(0.5)
        self.inset_text_font_spin.setValue(12.0)

        self.top_wspace_spin = QDoubleSpinBox()
        self.top_wspace_spin.setRange(-0.95, 1.50)
        self.top_wspace_spin.setDecimals(2)
        self.top_wspace_spin.setSingleStep(0.02)
        self.top_wspace_spin.setValue(0.24)

        self.bottom_wspace_spin = QDoubleSpinBox()
        self.bottom_wspace_spin.setRange(-0.95, 1.50)
        self.bottom_wspace_spin.setDecimals(2)
        self.bottom_wspace_spin.setSingleStep(0.02)
        self.bottom_wspace_spin.setValue(0.34)

        self.vertical_space_spin = QDoubleSpinBox()
        self.vertical_space_spin.setRange(0.00, 1.50)
        self.vertical_space_spin.setDecimals(2)
        self.vertical_space_spin.setSingleStep(0.02)
        self.vertical_space_spin.setValue(0.38)

        self.title_a_edit = QLineEdit("Loss mixes in vacuum")
        self.title_b_edit = QLineEdit("Phase noise mixes in anti-squeezing")
        self.title_c_edit = QLineEdit("Phase noise = {phase:.0f} mrad")
        self.title_d_edit = QLineEdit("Phase noise = {phase:.0f} mrad")

        self.legend_vacuum_edit = QLineEdit("vacuum")
        self.legend_loss_edit = QLineEdit("loss = {loss_pct:.0f}%")
        self.legend_squeezed_edit = QLineEdit("squeezed state")
        self.legend_readout_edit = QLineEdit("readout axis")

        self.top_titles_cb = QCheckBox("Show top-panel titles")
        self.top_titles_cb.setChecked(True)

        self.bottom_titles_cb = QCheckBox("Show bottom-panel titles")
        self.bottom_titles_cb.setChecked(True)

        form.addRow("Base fontsize", self.font_spin)
        form.addRow("Legend fontsize", self.legend_font_spin)
        form.addRow("In-figure text fontsize", self.inset_text_font_spin)
        form.addRow("Top horizontal spacing", self.top_wspace_spin)
        form.addRow("Bottom horizontal spacing", self.bottom_wspace_spin)
        form.addRow("Vertical spacing", self.vertical_space_spin)

        form.addRow(QLabel("<b>Titles</b>"))
        form.addRow("Title panel a", self.title_a_edit)
        form.addRow("Title panel b", self.title_b_edit)
        form.addRow("Title panel c", self.title_c_edit)
        form.addRow("Title panel d", self.title_d_edit)
        form.addRow(self.top_titles_cb)
        form.addRow(self.bottom_titles_cb)

        form.addRow(QLabel("<b>Legend text</b>"))
        form.addRow("Vacuum label", self.legend_vacuum_edit)
        form.addRow("Loss label format", self.legend_loss_edit)
        form.addRow("Squeezed-state label", self.legend_squeezed_edit)
        form.addRow("Readout-axis label", self.legend_readout_edit)

        note = QLabel(
            "The ellipse panels are schematic and not frequency-dependent. "
            "A frequency label is therefore not shown."
        )
        note.setWordWrap(True)
        form.addRow(note)

        self.ctrl.addWidget(box)

    def _build_ellipse_group(self):
        box = QGroupBox("Ellipse panels")
        grid = QGridLayout(box)

        self.nlg_ellipse_spin = QDoubleSpinBox()
        self.nlg_ellipse_spin.setRange(1.01, 50.0)
        self.nlg_ellipse_spin.setDecimals(2)
        self.nlg_ellipse_spin.setValue(8.0)

        self.loss_list_ellipse_edit = QLineEdit("0.00, 0.20, 0.40, 0.60")

        self.loss_phase_panel_spin = QDoubleSpinBox()
        self.loss_phase_panel_spin.setRange(0.0, 0.95)
        self.loss_phase_panel_spin.setDecimals(3)
        self.loss_phase_panel_spin.setSingleStep(0.01)
        self.loss_phase_panel_spin.setValue(0.20)

        self.readout_angle_spin = QDoubleSpinBox()
        self.readout_angle_spin.setRange(-90.0, 90.0)
        self.readout_angle_spin.setDecimals(2)
        self.readout_angle_spin.setSingleStep(0.5)
        self.readout_angle_spin.setValue(-33.0)

        self.ellipse_angle_spin = QDoubleSpinBox()
        self.ellipse_angle_spin.setRange(-90.0, 90.0)
        self.ellipse_angle_spin.setDecimals(2)
        self.ellipse_angle_spin.setSingleStep(0.5)
        self.ellipse_angle_spin.setValue(-37.5)

        self.phase_noise_spin = QDoubleSpinBox()
        self.phase_noise_spin.setRange(0.0, 200.0)
        self.phase_noise_spin.setDecimals(1)
        self.phase_noise_spin.setValue(40.0)

        self.ellipse_color_combo = QComboBox()
        for name, hex_color in DISTINGUISHABLE_COLORS.items():
            self.ellipse_color_combo.addItem(f"{name}  {hex_color}", hex_color)
        self.ellipse_color_combo.setCurrentIndex(self.ellipse_color_combo.findData("#000060"))

        self.exmin = QDoubleSpinBox()
        self.exmax = QDoubleSpinBox()
        self.eymin = QDoubleSpinBox()
        self.eymax = QDoubleSpinBox()
        for w, val in [(self.exmin, -5.0), (self.exmax, 5.0), (self.eymin, -5.0), (self.eymax, 5.0)]:
            w.setRange(-100.0, 100.0)
            w.setDecimals(2)
            w.setSingleStep(0.1)
            w.setValue(val)

        self.ellipse_alpha_min = QDoubleSpinBox()
        self.ellipse_alpha_max = QDoubleSpinBox()
        for w, val in [(self.ellipse_alpha_min, 0.22), (self.ellipse_alpha_max, 0.95)]:
            w.setRange(0.0, 1.0)
            w.setSingleStep(0.05)
            w.setDecimals(2)
            w.setValue(val)

        self.alpha_direction_combo = QComboBox()
        self.alpha_direction_combo.addItem("increase with loss", "increase")
        self.alpha_direction_combo.addItem("decrease with loss", "decrease")

        self.show_vacuum_cb = QCheckBox("Show vacuum reference")
        self.show_vacuum_cb.setChecked(True)

        row = 0
        grid.addWidget(QLabel("Ellipse NLG"), row, 0); grid.addWidget(self.nlg_ellipse_spin, row, 1); row += 1
        grid.addWidget(QLabel("Loss list for panel (a)"), row, 0); grid.addWidget(self.loss_list_ellipse_edit, row, 1); row += 1
        grid.addWidget(QLabel("Panel (b) loss"), row, 0); grid.addWidget(self.loss_phase_panel_spin, row, 1); row += 1
        grid.addWidget(QLabel("Readout angle [deg]"), row, 0); grid.addWidget(self.readout_angle_spin, row, 1); row += 1
        grid.addWidget(QLabel("Ellipse angle [deg]"), row, 0); grid.addWidget(self.ellipse_angle_spin, row, 1); row += 1
        grid.addWidget(QLabel("Phase noise [mrad]"), row, 0); grid.addWidget(self.phase_noise_spin, row, 1); row += 1
        grid.addWidget(QLabel("Ellipse color"), row, 0); grid.addWidget(self.ellipse_color_combo, row, 1); row += 1
        grid.addWidget(QLabel("Ellipse xmin"), row, 0); grid.addWidget(self.exmin, row, 1); row += 1
        grid.addWidget(QLabel("Ellipse xmax"), row, 0); grid.addWidget(self.exmax, row, 1); row += 1
        grid.addWidget(QLabel("Ellipse ymin"), row, 0); grid.addWidget(self.eymin, row, 1); row += 1
        grid.addWidget(QLabel("Ellipse ymax"), row, 0); grid.addWidget(self.eymax, row, 1); row += 1
        grid.addWidget(QLabel("Min alpha"), row, 0); grid.addWidget(self.ellipse_alpha_min, row, 1); row += 1
        grid.addWidget(QLabel("Max alpha"), row, 0); grid.addWidget(self.ellipse_alpha_max, row, 1); row += 1
        grid.addWidget(QLabel("Alpha direction"), row, 0); grid.addWidget(self.alpha_direction_combo, row, 1); row += 1
        grid.addWidget(self.show_vacuum_cb, row, 0, 1, 2); row += 1

        self.ctrl.addWidget(box)

    def _build_curve_group(self):
        box = QGroupBox("Bottom curve panels")
        grid = QGridLayout(box)

        self.loss_list_curves_edit = QLineEdit("0.20, 0.22, 0.24, 0.26, 0.28, 0.30")

        self.nlg_min_spin = QDoubleSpinBox()
        self.nlg_max_spin = QDoubleSpinBox()
        self.nlg_oper_spin = QDoubleSpinBox()
        for w, val in [(self.nlg_min_spin, 0.0), (self.nlg_max_spin, 20.0), (self.nlg_oper_spin, 8.0)]:
            w.setRange(0.0, 100.0)
            w.setDecimals(2)
            w.setSingleStep(0.1)
            w.setValue(val)

        self.ymin_curve = QDoubleSpinBox()
        self.ymax_curve = QDoubleSpinBox()
        for w, val in [(self.ymin_curve, -8.0), (self.ymax_curve, 0.0)]:
            w.setRange(-100.0, 100.0)
            w.setDecimals(2)
            w.setSingleStep(0.1)
            w.setValue(val)

        self.phase0_spin = QDoubleSpinBox()
        self.phase1_spin = QDoubleSpinBox()
        for w, val in [(self.phase0_spin, 0.0), (self.phase1_spin, 27.0)]:
            w.setRange(0.0, 200.0)
            w.setDecimals(1)
            w.setSingleStep(1.0)
            w.setValue(val)

        self.curve_legend_cb = QCheckBox("Show bottom legends")
        self.curve_legend_cb.setChecked(True)

        row = 0
        grid.addWidget(QLabel("Loss list for curves"), row, 0); grid.addWidget(self.loss_list_curves_edit, row, 1); row += 1
        grid.addWidget(QLabel("NLG min"), row, 0); grid.addWidget(self.nlg_min_spin, row, 1); row += 1
        grid.addWidget(QLabel("NLG max"), row, 0); grid.addWidget(self.nlg_max_spin, row, 1); row += 1
        grid.addWidget(QLabel("Operating NLG"), row, 0); grid.addWidget(self.nlg_oper_spin, row, 1); row += 1
        grid.addWidget(QLabel("Curve ymin"), row, 0); grid.addWidget(self.ymin_curve, row, 1); row += 1
        grid.addWidget(QLabel("Curve ymax"), row, 0); grid.addWidget(self.ymax_curve, row, 1); row += 1
        grid.addWidget(QLabel("Phase noise panel (c) [mrad]"), row, 0); grid.addWidget(self.phase0_spin, row, 1); row += 1
        grid.addWidget(QLabel("Phase noise panel (d) [mrad]"), row, 0); grid.addWidget(self.phase1_spin, row, 1); row += 1
        grid.addWidget(self.curve_legend_cb, row, 0, 1, 2); row += 1

        self.ctrl.addWidget(box)


    def _build_legend_group(self):
        box = QGroupBox("Legend positions")
        grid = QGridLayout(box)

        note = QLabel(
            "Each legend position is controlled by loc plus bbox_to_anchor x/y. "
            "Use loc='upper right' with x=1,y=1 for the usual inside upper-right legend; "
            "use x>1 to move a legend outside the axis."
        )
        note.setWordWrap(True)
        grid.addWidget(note, 0, 0, 1, 4)

        self.legend_locs = [
            "best",
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "right",
            "center left",
            "center right",
            "lower center",
            "upper center",
            "center",
        ]

        self.leg_a_loc, self.leg_a_x, self.leg_a_y = self._legend_controls("lower right", 1.07, -0.02)
        self.leg_b_loc, self.leg_b_x, self.leg_b_y = self._legend_controls("lower right", 0.90, -0.02)
        self.leg_c_loc, self.leg_c_x, self.leg_c_y = self._legend_controls("upper right", 1.0, 1.0)
        self.leg_d_loc, self.leg_d_x, self.leg_d_y = self._legend_controls("upper right", 1.0, 1.0)

        rows = [
            ("Panel a", self.leg_a_loc, self.leg_a_x, self.leg_a_y),
            ("Panel b", self.leg_b_loc, self.leg_b_x, self.leg_b_y),
            ("Panel c", self.leg_c_loc, self.leg_c_x, self.leg_c_y),
            ("Panel d", self.leg_d_loc, self.leg_d_x, self.leg_d_y),
        ]

        grid.addWidget(QLabel("panel"), 1, 0)
        grid.addWidget(QLabel("loc"), 1, 1)
        grid.addWidget(QLabel("x"), 1, 2)
        grid.addWidget(QLabel("y"), 1, 3)

        for i, (name, loc, x, y) in enumerate(rows, start=2):
            grid.addWidget(QLabel(name), i, 0)
            grid.addWidget(loc, i, 1)
            grid.addWidget(x, i, 2)
            grid.addWidget(y, i, 3)

        self.ctrl.addWidget(box)

    def _legend_controls(self, default_loc, default_x, default_y):
        loc = QComboBox()
        for item in self.legend_locs:
            loc.addItem(item)
        loc.setCurrentText(default_loc)

        x = QDoubleSpinBox()
        y = QDoubleSpinBox()
        for w, val in [(x, default_x), (y, default_y)]:
            w.setRange(-2.0, 3.0)
            w.setDecimals(2)
            w.setSingleStep(0.05)
            w.setValue(val)

        return loc, x, y

    def _place_legend(self, ax, loc_combo, x_spin, y_spin, show=True):
        if not show:
            return
        ax.legend(
            loc=loc_combo.currentText(),
            bbox_to_anchor=(float(x_spin.value()), float(y_spin.value())),
            frameon=True,
            framealpha=0.92,
            fontsize=float(self.legend_font_spin.value()),
        )


    def _build_export_group(self, parent_layout):
        box = QGroupBox("Actions")
        v = QVBoxLayout(box)

        btn_redraw = QPushButton("Redraw figure")
        btn_export = QPushButton("Export figure")
        btn_redraw.clicked.connect(self.redraw)
        btn_export.clicked.connect(self.export_figure)

        v.addWidget(btn_redraw)
        v.addWidget(btn_export)

        # Keep actions outside the scrollable tabs so they are always visible.
        parent_layout.addWidget(box, stretch=0)

    def _safe_format(self, template, **kwargs):
        """
        Format user-supplied title/legend templates.

        Supported placeholders include:
            {loss}, {loss_pct}, {phase}, {nlg}, {readout}, {ellipse_angle}

        If formatting fails, return the raw template instead of crashing the GUI.
        """
        try:
            return str(template).format(**kwargs)
        except Exception:
            return str(template)

    def _legend_loss_label(self, loss):
        return self._safe_format(
            self.legend_loss_edit.text(),
            loss=float(loss),
            loss_pct=100.0 * float(loss),
            nlg=float(self.nlg_ellipse_spin.value()),
            phase=float(self.phase_noise_spin.value()),
            readout=float(self.readout_angle_spin.value()),
            ellipse_angle=float(self.ellipse_angle_spin.value()),
        )

    def _title_text(self, edit, phase=None):
        return self._safe_format(
            edit.text(),
            phase=0.0 if phase is None else float(phase),
            nlg=float(self.nlg_ellipse_spin.value()),
            readout=float(self.readout_angle_spin.value()),
            ellipse_angle=float(self.ellipse_angle_spin.value()),
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def redraw(self):
        try:
            self._redraw_impl()
        except Exception as exc:
            QMessageBox.critical(self, "Redraw failed", str(exc))
            raise

    def _redraw_impl(self):
        base_fs = float(self.font_spin.value())

        nlg_ellipse = float(self.nlg_ellipse_spin.value())
        loss_list_ell = sorted(parse_float_list(self.loss_list_ellipse_edit.text()))
        if not loss_list_ell:
            loss_list_ell = [0.0, 0.1, 0.2, 0.3]

        loss_b = float(self.loss_phase_panel_spin.value())
        readout_deg = float(self.readout_angle_spin.value())
        ellipse_angle_deg = float(self.ellipse_angle_spin.value())
        phase_noise_mrad = float(self.phase_noise_spin.value())
        ellipse_color = self.ellipse_color_combo.currentData()

        exmin, exmax = float(self.exmin.value()), float(self.exmax.value())
        eymin, eymax = float(self.eymin.value()), float(self.eymax.value())
        if exmin >= exmax or eymin >= eymax:
            raise ValueError("Ellipse limits must satisfy xmin < xmax and ymin < ymax")

        alpha_min = float(self.ellipse_alpha_min.value())
        alpha_max = float(self.ellipse_alpha_max.value())

        loss_list_curves = sorted(parse_float_list(self.loss_list_curves_edit.text()))
        if not loss_list_curves:
            loss_list_curves = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]

        nlg_min = float(self.nlg_min_spin.value())
        nlg_max = float(self.nlg_max_spin.value())
        nlg_oper = float(self.nlg_oper_spin.value())
        if nlg_min >= nlg_max:
            raise ValueError("NLG min must be less than NLG max")

        curve_ymin = float(self.ymin_curve.value())
        curve_ymax = float(self.ymax_curve.value())
        if curve_ymin >= curve_ymax:
            raise ValueError("Curve ymin must be less than curve ymax")

        phase0 = float(self.phase0_spin.value())
        phase1 = float(self.phase1_spin.value())

        plt.rcParams.update({
            "font.size": base_fs,
            "axes.labelsize": base_fs,
            "axes.titlesize": base_fs + 0.5,
            "xtick.labelsize": max(base_fs - 1.0, 6),
            "ytick.labelsize": max(base_fs - 1.0, 6),
            "legend.fontsize": max(base_fs - 2.0, 6),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

        self.fig.clear()

        outer = self.fig.add_gridspec(
            2, 1,
            height_ratios=[1.0, 1.0],
            hspace=float(self.vertical_space_spin.value()),
        )
        gs_top = outer[0].subgridspec(
            1, 2,
            wspace=float(self.top_wspace_spin.value()),
        )
        gs_bot = outer[1].subgridspec(
            1, 2,
            wspace=float(self.bottom_wspace_spin.value()),
        )

        ax_a = self.fig.add_subplot(gs_top[0, 0])
        ax_b = self.fig.add_subplot(gs_top[0, 1])
        ax_c = self.fig.add_subplot(gs_bot[0, 0])
        ax_d = self.fig.add_subplot(gs_bot[0, 1])

        v_sqz, v_anti = variances_from_nlg(nlg_ellipse)
        t = np.linspace(0.0, 2*np.pi, 500)
        max_line_length = max(abs(exmin), abs(exmax), abs(eymin), abs(eymax))

        # ---------------- Panel (a): loss ellipses ----------------
        if self.show_vacuum_cb.isChecked():
            ax_a.plot(np.cos(t), np.sin(t), color=VACUUM_COLOR, lw=1.3, ls="--", label=self.legend_vacuum_edit.text())

        if len(loss_list_ell) == 1:
            alphas = [alpha_max]
        else:
            if self.alpha_direction_combo.currentData() == "increase":
                alphas = np.linspace(alpha_min, alpha_max, len(loss_list_ell))
            else:
                alphas = np.linspace(alpha_max, alpha_min, len(loss_list_ell))

        # Same color; alpha direction is user-selectable.
        for loss, alpha in zip(loss_list_ell, alphas):
            vp = apply_loss(v_sqz, loss)
            va = apply_loss(v_anti, loss)
            cov = covariance_from_variances(vp, va, angle_deg=ellipse_angle_deg)
            x, y = ellipse_points_from_cov(cov)
            ax_a.plot(
                x, y,
                color=ellipse_color,
                lw=2.2,
                alpha=float(alpha),
                label=self._legend_loss_label(loss),
            )

        ax_a.set_aspect("equal", adjustable="box")
        ax_a.set_box_aspect(1)
        ax_a.set_xlim(exmin, exmax)
        ax_a.set_ylim(eymin, eymax)
        ax_a.set_xlabel("Phase quadrature")
        ax_a.set_ylabel("Amplitude quadrature")
        if self.top_titles_cb.isChecked():
            ax_a.set_title(self._title_text(self.title_a_edit))
        ax_a.grid(True, alpha=0.18)
        ax_a.text(
            0.03, 0.97,
            rf"$\mathrm{{NLG}}={nlg_ellipse:.1f}$" + "\n" +
            rf"$\phi={ellipse_angle_deg:.1f}^\circ$",
            # rf"$\theta_\mathrm{{ell}}={ellipse_angle_deg:.1f}^\circ$",
            transform=ax_a.transAxes, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0),
            fontsize=float(self.inset_text_font_spin.value()),
        )
        self._place_legend(ax_a, self.leg_a_loc, self.leg_a_x, self.leg_a_y, show=True)

        # ---------------- Panel (b): phase-noise/readout schematic ----------------
        vp = apply_loss(v_sqz, loss_b)
        va = apply_loss(v_anti, loss_b)
        cov = covariance_from_variances(vp, va, angle_deg=ellipse_angle_deg)
        x, y = ellipse_points_from_cov(cov)
        ax_b.plot(x, y, color=ellipse_color, lw=2.3, label=self.legend_squeezed_edit.text())

        if self.show_vacuum_cb.isChecked():
            ax_b.plot(np.cos(t), np.sin(t), color=VACUUM_COLOR, lw=1.2, ls="--", label=self.legend_vacuum_edit.text())

        xr, yr = line_through_origin(readout_deg, length=max_line_length)
        ax_b.plot(xr, yr, color=READOUT_COLOR, lw=1.6, label=self.legend_readout_edit.text())

        sigma_deg = np.rad2deg(phase_noise_mrad * 1e-3)
        for sign in [-1.0, 1.0]:
            xj, yj = line_through_origin(readout_deg + sign * sigma_deg, length=max_line_length)
            ax_b.plot(xj, yj, color=READOUT_COLOR, lw=1.1, ls=":")

        ax_b.set_aspect("equal", adjustable="box")
        ax_b.set_box_aspect(1)
        ax_b.set_xlim(exmin, exmax)
        ax_b.set_ylim(eymin, eymax)
        ax_b.set_xlabel("Phase quadrature")
        ax_b.set_ylabel("Amplitude quadrature")
        if self.top_titles_cb.isChecked():
            ax_b.set_title(self._title_text(self.title_b_edit))
        ax_b.grid(True, alpha=0.18)
        ax_b.text(
            0.03, 0.97,
            rf"$\zeta={readout_deg:.1f}^\circ$" + "\n" +
            rf"$\phi={ellipse_angle_deg:.1f}^\circ$" + "\n" +
            rf"$\sigma_\phi={phase_noise_mrad:.0f}\,\mathrm{{mrad}}$",
            # rf"$\theta_\mathrm{{readout}}={readout_deg:.1f}^\circ$" + "\n" +
            # rf"$\theta_\mathrm{{ell}}={ellipse_angle_deg:.1f}^\circ$" + "\n" +
            # rf"$\sigma_\theta={phase_noise_mrad:.0f}\,\mathrm{{mrad}}$",
            transform=ax_b.transAxes, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0),
            fontsize=float(self.inset_text_font_spin.value()),
        )
        self._place_legend(ax_b, self.leg_b_loc, self.leg_b_x, self.leg_b_y, show=True)

        # ---------------- Panels (c,d): separate phase-noise curves ----------------
        self._plot_curve_panel(
            ax_c, phase0, loss_list_curves, nlg_min, nlg_max, nlg_oper,
            curve_ymin, curve_ymax,
            legend_controls=(self.leg_c_loc, self.leg_c_x, self.leg_c_y),
            show_legend=self.curve_legend_cb.isChecked(),
        )
        self._plot_curve_panel(
            ax_d, phase1, loss_list_curves, nlg_min, nlg_max, nlg_oper,
            curve_ymin, curve_ymax,
            legend_controls=(self.leg_d_loc, self.leg_d_x, self.leg_d_y),
            show_legend=self.curve_legend_cb.isChecked(),
        )

        if self.bottom_titles_cb.isChecked():
            ax_c.set_title(self._title_text(self.title_c_edit, phase=phase0))
            ax_d.set_title(self._title_text(self.title_d_edit, phase=phase1))

        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.09)
        self.canvas.draw_idle()

    def _plot_curve_panel(
        self,
        ax,
        phase_noise_mrad,
        loss_list,
        nlg_min,
        nlg_max,
        nlg_oper,
        ymin,
        ymax,
        legend_controls=None,
        show_legend=True,
    ):
        nlg = np.linspace(nlg_min, nlg_max, 850)

        for i, loss in enumerate(loss_list):
            color = CURVE_COLORS[i % len(CURVE_COLORS)]
            y = observed_db(nlg, loss, phase_noise_mrad)
            ax.plot(nlg, y, color=color, lw=2.0, label=self._legend_loss_label(loss))

        ax.axhline(0.0, color="0.25", lw=0.9)
        ax.axvline(nlg_oper, color="0.25", lw=1.2, ls="--")
        ax.set_xlim(0.0, nlg_max)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Nonlinear gain")
        ax.set_ylabel(r"$10\log_{10} V$ [dB]")
        ax.grid(True, alpha=0.22)

        if show_legend:
            if legend_controls is None:
                ax.legend(loc="upper right", frameon=True, framealpha=0.92)
            else:
                loc_combo, x_spin, y_spin = legend_controls
                self._place_legend(ax, loc_combo, x_spin, y_spin, show=True)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_figure(self):
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export figure",
            str(Path.cwd() / "fig4_loss_phase_noise.pdf"),
            "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)",
        )
        if not path:
            return

        try:
            out = Path(path).expanduser().resolve()
            if not out.suffix:
                extension = {"PNG (*.png)": ".png", "SVG (*.svg)": ".svg"}.get(
                    selected_filter, ".pdf"
                )
                out = out.with_suffix(extension)
                if out.exists() and QMessageBox.question(
                    self, "Replace file?", f"Replace existing file?\n{out}",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                ) != QMessageBox.Yes:
                    return
            if out.suffix.lower() not in {".pdf", ".png", ".svg"}:
                raise ValueError("Choose a .pdf, .png, or .svg filename.")
            # Export exactly the currently displayed figure, including zoom/pan.
            # Do not rebuild it or change the embedded Qt canvas permanently.
            self.canvas.draw()
            write_figure(self.fig, out)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        QMessageBox.information(self, "Export complete", f"Saved:\n{out}")


def main():
    print("[fig4-gui] Starting Figure 4 GUI v7...", flush=True)
    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True

    win = Fig4Window()
    win.show()
    win.raise_()
    win.activateWindow()

    app._fig4_window = win
    globals()["_FIG4_WINDOW"] = win
    globals()["_FIG4_APP"] = app

    if created:
        return app.exec_()
    return win


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        sys.exit(result)
