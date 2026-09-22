#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
qnoise_decomposer.py

Qt GUI for visualizing a three-power quantum-noise decomposition and associated covariance ellipses.

Main changes in this version:
- Right-hand controls are divided into four independently scrollable layers.
- YAML/model selection and ellipse controls are together in the first layer.
- Ellipse frequencies are no longer limited to three: enter any number of frequencies.
- Ellipse subplot selectors update dynamically to match the requested frequencies.
- Bottom ellipse panels automatically wrap across rows when many frequencies are requested.
- Recompute + redraw and Export are always visible below the control layers.

Run:
    python qnoise_decomposer.py --yaml april9.yaml

Spyder:
    %autoreload 0
    %gui qt
    %runfile /path/to/qnoise_decomposer.py --wdir /path/to/working/folder --args --yaml april9.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
# Do not force a backend here. Spyder/Qt may already own the event loop.
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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

import gwinc


# =============================================================================
# Defaults
# =============================================================================

DEFAULT = dict(
    fmin=10.0,
    fmax=7000.0,
    npts=1200,

    LO_ANG_DEG=-13.9,
    arm_length=3995.0,
    PDeff=0.925,

    ArmPower=269572.0,
    SEC_Detuning_deg=-0.0414,

    InjSqz_dB=18.1414,
    InjLoss=0.05,

    FC_Detune_Hz=-27.26,
    FC_Mismatch=0.024,
    FC_Mismatch_Phase_rad=3.06,

    IFO_OMC_Mismatch=0.0114,
    IFO_OMC_Mismatch_Phase_rad=3.615,

    SQZ_OMC_Mismatch=0.0212,
    SQZ_OMC_Mismatch_Phase_rad=0.3145,

    PhaseNoise_rad=0.01984,
)

CASES = ("no_sqz", "fis", "fds")

CASE_LABELS = {
    "no_sqz": "Vacuum",
    "fis": "FIS",
    "fds": "FDS",
}

CASE_COLORS = {
    "no_sqz": "#000060",
    "fis": "#8185ff",
    "fds": "#d900ff",
}

LEGEND_LOCS = [
    "none",
    "best",
    "upper right",
    "upper left",
    "lower right",
    "lower left",
    "center right",
    "center left",
    "upper center",
    "lower center",
    "center",
]


# =============================================================================
# Numeric helpers
# =============================================================================

def auto_find_yaml() -> Path:
    candidates = [Path("april9.yaml"), Path("ifo.yaml"), Path("config.yaml"), Path("gwinc.yaml")]
    for c in candidates:
        if c.exists():
            return c.resolve()

    yamls = sorted(Path(".").glob("*.yaml"))
    if len(yamls) == 1:
        return yamls[0].resolve()

    raise FileNotFoundError(
        "Could not auto-detect a YAML file. Put your YAML in this folder as april9.yaml, or pass --yaml.\n"
        f"YAMLs found here: {[p.name for p in yamls]}"
    )


def geomspace_freq(fmin: float, fmax: float, npts: int) -> np.ndarray:
    return np.geomspace(float(fmin), float(fmax), int(npts))


def parse_ellipse_frequency_text(text: str) -> list[float]:
    """Parse a comma/semicolon/whitespace-separated list of positive frequencies."""
    raw = str(text).replace(",", " ").replace(";", " ").split()
    if not raw:
        raise ValueError("Enter at least one ellipse frequency.")

    vals = []
    for item in raw:
        value = float(item)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Ellipse frequencies must be positive finite numbers; got {item!r}.")
        vals.append(value)
    return vals


def nearest_index(f: np.ndarray, f0: float) -> int:
    return int(np.argmin(np.abs(np.asarray(f) - float(f0))))


def solve_ABC_exact3(Ps: list[float], Sps: list[np.ndarray]):
    """Solve A, B, C from S(P) = A/P + B P + C evaluated at three powers."""
    P1, P2, P3 = map(float, Ps)
    M = np.array(
        [[1.0 / P1, P1, 1.0],
         [1.0 / P2, P2, 1.0],
         [1.0 / P3, P3, 1.0]],
        dtype=float
    )
    Y = np.vstack([np.asarray(s, float) for s in Sps])
    A, B, C = np.linalg.inv(M) @ Y
    return A, B, C


def ellipse_points_from_cov(V: np.ndarray, nsig: float = 1.0, npts: int = 900) -> np.ndarray:
    """
    Return ellipse coordinates in ASD units.

    V is a covariance-like matrix in PSD units.
    Output points are in sqrt(PSD) units.
    """
    V = np.asarray(V, float)
    V = 0.5 * (V + V.T)
    w, U = np.linalg.eigh(V)
    w = np.maximum(w, 0.0)

    t = np.linspace(0.0, 2.0 * np.pi, int(npts))
    circle = np.vstack([np.cos(t), np.sin(t)])
    A = U @ np.diag(np.sqrt(w)) * float(nsig)
    return (A @ circle).T


def power10_scale(x: float) -> float:
    """Return a pure power-of-ten scale factor for a positive number."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    exponent = int(np.floor(np.log10(x)))
    return 10.0 ** exponent


def format_scale_label(scale_factor: float) -> str:
    exponent = int(np.floor(np.log10(scale_factor)))
    return rf"$10^{{{exponent}}}$"


# =============================================================================
# GWINC plumbing
# =============================================================================

def make_budget(yaml_path: Path, freq: np.ndarray, p: dict):
    b = gwinc.load_budget(str(yaml_path), freq, bname="Quantum")
    ifo = b.ifo

    if hasattr(ifo, "Optics") and hasattr(ifo.Optics, "Quadrature") and hasattr(ifo.Optics.Quadrature, "dc"):
        ifo.Optics.Quadrature.dc = np.pi / 2.0 + float(p["LO_ANG_DEG"]) * np.pi / 180.0

    if hasattr(ifo, "Optics") and hasattr(ifo.Optics, "PhotoDetectorEfficiency"):
        ifo.Optics.PhotoDetectorEfficiency = float(p["PDeff"])

    if hasattr(ifo, "Infrastructure") and hasattr(ifo.Infrastructure, "Length"):
        ifo.Infrastructure.Length = float(p["arm_length"])

    return b


def apply_case_and_params(ifo, case: str, p: dict):
    if hasattr(ifo, "Laser") and hasattr(ifo.Laser, "ArmPower"):
        ifo.Laser.ArmPower = float(p["ArmPower"])
    elif hasattr(ifo, "Power") and hasattr(ifo.Power, "ArmPower"):
        ifo.Power.ArmPower = float(p["ArmPower"])
    else:
        raise AttributeError("Cannot set ArmPower on this ifo struct")

    if hasattr(ifo, "Optics") and hasattr(ifo.Optics, "SRM") and hasattr(ifo.Optics.SRM, "Tunephase"):
        ifo.Optics.SRM.Tunephase = np.pi * float(p["SEC_Detuning_deg"]) / 180.0

    case = case.lower()

    if case == "no_sqz":
        if hasattr(ifo, "Squeezer"):
            try:
                del ifo.Squeezer.FilterCavity
            except Exception:
                pass
            try:
                del ifo.Squeezer
            except Exception:
                pass
        return

    if not hasattr(ifo, "Squeezer"):
        raise AttributeError("ifo has no Squeezer block; cannot apply fis/fds settings")

    ifo.Squeezer.AmplitudedB = float(p["InjSqz_dB"])
    ifo.Squeezer.InjectionLoss = float(p["InjLoss"])
    ifo.Squeezer.SQZAngleRMS = float(p["PhaseNoise_rad"])
    ifo.Squeezer.Type = "Freq Independent" if case == "fis" else "Freq Dependent"

    if hasattr(ifo.Squeezer, "FilterCavity"):
        ifo.Squeezer.FilterCavity.fdetune = float(p["FC_Detune_Hz"])
        ifo.Squeezer.FilterCavity.L_mm = float(p["FC_Mismatch"])
        ifo.Squeezer.FilterCavity.psi_mm = float(p["FC_Mismatch_Phase_rad"])

    if hasattr(ifo, "Optics"):
        if hasattr(ifo.Optics, "MM_IFO_OMC"):
            ifo.Optics.MM_IFO_OMC = float(p["IFO_OMC_Mismatch"])
        if hasattr(ifo.Optics, "MM_IFO_OMCphi"):
            ifo.Optics.MM_IFO_OMCphi = float(p["IFO_OMC_Mismatch_Phase_rad"])

    if hasattr(ifo.Squeezer, "MM_SQZ_OMC"):
        ifo.Squeezer.MM_SQZ_OMC = float(p["SQZ_OMC_Mismatch"])
    if hasattr(ifo.Squeezer, "MM_SQZ_OMCphi"):
        ifo.Squeezer.MM_SQZ_OMCphi = float(p["SQZ_OMC_Mismatch_Phase_rad"])
    if hasattr(ifo.Squeezer, "SQZAngle"):
        ifo.Squeezer.SQZAngle = -float(p["LO_ANG_DEG"]) * np.pi / 180.0



def params_from_yaml(yaml_path: Path, freq: np.ndarray, fallback: dict) -> dict:
    """Read the model-driving parameters used by this script from a GWINC YAML.

    Any field that is unavailable in the selected model falls back to the
    current value.  This is important because ``make_budget`` and
    ``apply_case_and_params`` explicitly set these quantities after loading
    the YAML; without updating ``p``, choosing a different YAML can appear to
    do nothing.
    """
    p = dict(fallback)
    b = gwinc.load_budget(str(yaml_path), freq, bname="Quantum")
    ifo = b.ifo

    def get_nested(obj, *names):
        cur = obj
        for name in names:
            if not hasattr(cur, name):
                return None
            cur = getattr(cur, name)
        return cur

    val = get_nested(ifo, "Optics", "Quadrature", "dc")
    if val is not None:
        p["LO_ANG_DEG"] = float((float(val) - np.pi / 2.0) * 180.0 / np.pi)

    val = get_nested(ifo, "Optics", "PhotoDetectorEfficiency")
    if val is not None:
        p["PDeff"] = float(val)

    val = get_nested(ifo, "Infrastructure", "Length")
    if val is not None:
        p["arm_length"] = float(val)

    val = get_nested(ifo, "Laser", "ArmPower")
    if val is None:
        val = get_nested(ifo, "Power", "ArmPower")
    if val is not None:
        p["ArmPower"] = float(val)

    val = get_nested(ifo, "Optics", "SRM", "Tunephase")
    if val is not None:
        p["SEC_Detuning_deg"] = float(val) * 180.0 / np.pi

    val = get_nested(ifo, "Squeezer", "AmplitudedB")
    if val is not None:
        p["InjSqz_dB"] = float(val)

    val = get_nested(ifo, "Squeezer", "InjectionLoss")
    if val is not None:
        p["InjLoss"] = float(val)

    val = get_nested(ifo, "Squeezer", "SQZAngleRMS")
    if val is not None:
        p["PhaseNoise_rad"] = float(val)

    val = get_nested(ifo, "Squeezer", "FilterCavity", "fdetune")
    if val is not None:
        p["FC_Detune_Hz"] = float(val)

    val = get_nested(ifo, "Squeezer", "FilterCavity", "L_mm")
    if val is not None:
        p["FC_Mismatch"] = float(val)

    val = get_nested(ifo, "Squeezer", "FilterCavity", "psi_mm")
    if val is not None:
        p["FC_Mismatch_Phase_rad"] = float(val)

    val = get_nested(ifo, "Optics", "MM_IFO_OMC")
    if val is not None:
        p["IFO_OMC_Mismatch"] = float(val)

    val = get_nested(ifo, "Optics", "MM_IFO_OMCphi")
    if val is not None:
        p["IFO_OMC_Mismatch_Phase_rad"] = float(val)

    val = get_nested(ifo, "Squeezer", "MM_SQZ_OMC")
    if val is not None:
        p["SQZ_OMC_Mismatch"] = float(val)

    val = get_nested(ifo, "Squeezer", "MM_SQZ_OMCphi")
    if val is not None:
        p["SQZ_OMC_Mismatch_Phase_rad"] = float(val)

    return p


def run_quantum_psd(yaml_path: Path, freq: np.ndarray, case: str, p: dict) -> np.ndarray:
    b = make_budget(yaml_path, freq, p)
    apply_case_and_params(b.ifo, case, p)
    tr = b.run()
    return np.asarray(tr.psd, float)


def get_decomp_for_case(yaml_path: Path, freq: np.ndarray, case: str, p: dict):
    """
    Three-power decomposition at reference power P0:
        S_QN(P) = A/P + B P + C
    """
    P0 = float(p["ArmPower"])
    Ps3 = [0.5 * P0, 1.0 * P0, 2.0 * P0]

    Sps3 = []
    for Pi in Ps3:
        p_i = dict(p)
        p_i["ArmPower"] = float(Pi)
        Sps3.append(run_quantum_psd(yaml_path, freq, case, p_i))

    A, B, C = solve_ABC_exact3(Ps3, Sps3)

    S_imp = A / P0
    S_ba = B * P0
    S_xcorr = C
    S_QN = Sps3[1]

    sign_xcorr = np.sign(S_xcorr)
    sign_xcorr[~np.isfinite(sign_xcorr)] = np.nan

    return dict(
        S_QN=S_QN,
        S_imp=S_imp,
        S_ba=S_ba,
        S_xcorr=S_xcorr,
        sign_xcorr=sign_xcorr,
    )


# =============================================================================
# Plotting helpers
# =============================================================================

def setup_log_axes(ax, grid_alpha=0.18):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=grid_alpha)
    ax.xaxis.grid(False, which="minor")
    ax.yaxis.grid(False, which="minor")
    ax.tick_params(axis="both", which="major", direction="in", length=5, width=0.9, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=2.5, width=0.7, top=True, right=True)


def setup_linear_strip(ax):
    ax.set_xscale("log")
    ax.set_ylim(-1.45, 1.45)
    ax.set_yticks([-1, 1])
    ax.set_yticklabels([r"$-$", r"$+$"])
    ax.axhline(0.0, color="0.75", lw=0.8)
    ax.grid(True, axis="x", which="major", alpha=0.18)
    ax.grid(False, axis="y")
    ax.tick_params(axis="x", which="both", direction="in", length=3, width=0.8, top=True, bottom=True)
    ax.tick_params(axis="y", which="major", direction="in", length=4, width=0.9, right=True)


def plot_case_decomposition(ax, f, d, case):
    """Plot the total quantum noise and its three decomposition components.

    Legends are deliberately created later in ``draw_figure`` so that the
    total-QN entry and the three component entries can live in separate,
    independently movable legend boxes.
    """
    color = CASE_COLORS[case]

    y_total = np.sqrt(np.maximum(d["S_QN"], 0.0))
    y_imp = np.sqrt(np.maximum(d["S_imp"], 0.0))
    y_ba = np.sqrt(np.maximum(d["S_ba"], 0.0))
    y_xcorr = np.sqrt(np.abs(d["S_xcorr"]))

    ax.loglog(f, y_total, color=color, lw=2.8, label=r"total QN", zorder=5)
    ax.loglog(f, y_imp, color="k", ls="-", lw=1.8, label=r"imprecision", zorder=4)
    ax.loglog(f, y_ba, color="k", ls="--", lw=1.8, label=r"back-action", zorder=3)
    ax.loglog(f, y_xcorr, color="k", ls=":", lw=2.0, label=r"$|\mathrm{correlation}|$", zorder=2)


def plot_xcorr_sign(ax, f, d):
    ax.semilogx(f, d["sign_xcorr"], color="k", lw=1.8, drawstyle="steps-mid")


def get_ellipse_points_for_case(d, freq, f_ellipse, nsig=1.0):
    idx = nearest_index(freq, f_ellipse)
    S_imp = float(d["S_imp"][idx])
    S_ba = float(d["S_ba"][idx])
    S_x = float(d["S_xcorr"][idx])

    V = np.array([[S_imp, 0.5 * S_x],
                  [0.5 * S_x, S_ba]], dtype=float)
    pts = ellipse_points_from_cov(V, nsig=nsig, npts=900)
    return pts, float(freq[idx])


def compute_ellipse_points_and_scales(data, freq, ellipse_freqs, nsig=1.0):
    all_pts = {}
    actual_freqs = {}
    panel_scales = {}

    for f_ellipse in ellipse_freqs:
        fkey = float(f_ellipse)
        all_pts[fkey] = {}
        actual_freqs[fkey] = {}
        panel_max_extent = 0.0

        for case in CASES:
            pts, f_actual = get_ellipse_points_for_case(data[case], freq, fkey, nsig=nsig)
            all_pts[fkey][case] = pts
            actual_freqs[fkey][case] = f_actual
            if pts.size:
                this_max = np.nanmax(np.abs(pts))
                if np.isfinite(this_max):
                    panel_max_extent = max(panel_max_extent, this_max)

        panel_scales[fkey] = power10_scale(panel_max_extent)

    return all_pts, actual_freqs, panel_scales


def set_ellipse_ticks(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax_abs = max(abs(xmin), abs(xmax))
    ymax_abs = max(abs(ymin), abs(ymax))
    lim = max(xmax_abs, ymax_abs)

    if lim <= 0.35:
        ticks = [-0.25, 0.0, 0.25]
    elif lim <= 0.75:
        ticks = [-0.5, 0.0, 0.5]
    elif lim <= 1.6:
        ticks = [-1.0, 0.0, 1.0]
    elif lim <= 3.5:
        ticks = [-2.0, 0.0, 2.0]
    elif lim <= 8.5:
        ticks = [-8.0, -4.0, 0.0, 4.0, 8.0]
    else:
        ticks = [-10.0, -5.0, 0.0, 5.0, 10.0]

    ax.set_xticks([t for t in ticks if xmin <= t <= xmax])
    ax.set_yticks([t for t in ticks if ymin <= t <= ymax])


def plot_combined_ellipse_panel(
    ax,
    pts_by_case,
    scale_factor,
    legend_outside=False,
    show_scale_text=True,
    freq_text=None,
    default_xlim=(-8.0, 8.0),
    default_ylim=(-8.0, 8.0),
    ellipse_text_fontsize=8.0,
    legend_fontsize=8.0,
):
    for case in CASES:
        ptsn = pts_by_case[case] / scale_factor
        ax.plot(ptsn[:, 0], ptsn[:, 1], color=CASE_COLORS[case], lw=2.2, label=CASE_LABELS[case])

    ax.axhline(0, color="0.78", lw=0.8)
    ax.axvline(0, color="0.78", lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    ax.set_xlim(*default_xlim)
    ax.set_ylim(*default_ylim)
    set_ellipse_ticks(ax)

    if show_scale_text:
        ax.text(
            0.95, 0.05, format_scale_label(scale_factor),
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=ellipse_text_fontsize, color="0.35",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.0),
        )

    if freq_text is not None:
        ax.text(
            0.03, 0.97, freq_text,
            transform=ax.transAxes,
            ha="left", va="top", fontsize=ellipse_text_fontsize, color="0.15",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.0),
        )

    if legend_outside:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            framealpha=0.92,
            fontsize=legend_fontsize,
            borderpad=0.35,
            labelspacing=0.35,
            handlelength=2.2,
        )


def set_case_title(ax, text: str, color: str, fontsize: float):
    """Set a visibly bold top-panel title with a concrete bold font family."""
    fp = FontProperties(
        family="DejaVu Sans",
        weight=700,
        size=float(fontsize),
    )
    title = ax.set_title(text, color=color, fontproperties=fp)
    title.set_fontfamily("DejaVu Sans")
    title.set_fontweight(700)
    return title


def set_axis_text_sizes(ax, title_fontsize: float, axis_fontsize: float, legend_fontsize: float):
    title_fontsize = float(title_fontsize)
    axis_fontsize = float(axis_fontsize)
    legend_fontsize = float(legend_fontsize)

    if ax.get_title():
        ax.title.set_fontproperties(
            FontProperties(
                family="DejaVu Sans",
                weight=700,
                size=title_fontsize,
            )
        )
        ax.title.set_fontfamily("DejaVu Sans")
        ax.title.set_fontweight(700)
    else:
        ax.title.set_fontsize(title_fontsize)

    ax.xaxis.label.set_fontsize(axis_fontsize)
    ax.yaxis.label.set_fontsize(axis_fontsize)

    for tick in ax.get_xticklabels():
        tick.set_fontsize(max(axis_fontsize - 1.5, 6))
    for tick in ax.get_yticklabels():
        tick.set_fontsize(max(axis_fontsize - 1.5, 6))

    leg = ax.get_legend()
    if leg is not None:
        for leg_text in leg.get_texts():
            leg_text.set_fontsize(max(legend_fontsize, 6))


# =============================================================================
# Qt GUI
# =============================================================================

class Fig3Window(QMainWindow):
    def __init__(self, yaml_path: Path, params: dict, args):
        super().__init__()

        self.yaml_path = Path(yaml_path).resolve()
        self.params = dict(params)
        self.args = args

        self.freq = geomspace_freq(args.fmin, args.fmax, args.npts)
        self.data = None

        self.ellipse_freqs = [float(x) for x in args.ellipse_freqs]
        self.ellipse_pts = None
        self.ellipse_actual_freqs = None
        self.ellipse_scales = None

        self.axes = {}
        self.axis_checkboxes = {}

        self.setWindowTitle(f"Quantum-noise decomposition GUI — {self.yaml_path.name}")
        self.resize(1550, 980)

        self._build_ui()
        self.recompute_and_redraw()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Plot area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        self.fig = Figure(figsize=(7.25, 6.3), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main_layout.addWidget(plot_widget, stretch=5)

        # Right-hand control dock:
        #   static YAML/model controls at the top,
        #   vertically-tabbed scrollable layers in the middle,
        #   static recompute/export actions at the bottom.
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # YAML/model selection is always visible.
        self.ctrl_layout = right_layout
        self._build_yaml_controls()

        self.control_tabs = QTabWidget()
        self.control_tabs.setTabPosition(QTabWidget.West)
        self.control_tabs.setMovable(False)
        right_layout.addWidget(self.control_tabs, stretch=1)

        # Layer 1 is the previous Model / ellipses layer minus the now-static YAML block.
        self._add_control_layer(
            "Ellipses",
            (self._build_ellipse_controls,),
        )
        self._add_control_layer(
            "Selected subplots",
            (self._build_axis_selector,),
        )
        self._add_control_layer(
            "Top/sign limits + fonts",
            (self._build_limits_controls,),
        )
        self._add_control_layer(
            "Legend positions",
            (self._build_legend_controls,),
        )

        self._build_export_controls(right_layout)
        main_layout.addWidget(right_widget, stretch=2)

    def _add_control_layer(self, title, builders):
        """Add one independently scrollable control layer."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        panel = QWidget()
        self.ctrl_layout = QVBoxLayout(panel)
        self.ctrl_layout.setContentsMargins(4, 4, 4, 4)
        self.ctrl_layout.setSpacing(4)

        for builder in builders:
            builder()
        self.ctrl_layout.addStretch(1)

        scroll.setWidget(panel)
        self.control_tabs.addTab(scroll, title)

    def _build_yaml_controls(self):
        group = QGroupBox("Model / GWINC YAML")
        layout = QGridLayout(group)

        self.yaml_edit = QLineEdit(str(self.yaml_path))
        self.yaml_edit.setToolTip(
            "YAML file used to build the GWINC quantum-noise model. "
            "You may type a path directly or choose one with Browse."
        )

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_yaml)

        load_btn = QPushButton("Use selected YAML + recompute")
        load_btn.clicked.connect(self.load_selected_yaml)

        layout.addWidget(QLabel("YAML file"), 0, 0)
        layout.addWidget(self.yaml_edit, 0, 1)
        layout.addWidget(browse_btn, 0, 2)
        layout.addWidget(load_btn, 1, 0, 1, 3)

        self.yaml_status = QLabel(f"Active model: {self.yaml_path.name}")
        self.yaml_status.setWordWrap(True)

        note = QLabel(
            "Browse selects a candidate file. 'Use selected YAML + recompute' "
            "loads that YAML's model parameters, recomputes all three cases, "
            "and redraws the complete figure."
        )
        note.setWordWrap(True)

        layout.addWidget(self.yaml_status, 2, 0, 1, 3)
        layout.addWidget(note, 3, 0, 1, 3)

        self.ctrl_layout.addWidget(group)

    def _build_axis_selector(self):
        group = QGroupBox("Apply changes to selected subplots")
        outer = QVBoxLayout(group)

        fixed_grid = QGridLayout()
        fixed_rows = [
            ("top_no_sqz", "Top: Vacuum"),
            ("top_fis", "Top: FIS"),
            ("top_fds", "Top: FDS"),
            ("sign_no_sqz", "Sign: Vacuum"),
            ("sign_fis", "Sign: FIS"),
            ("sign_fds", "Sign: FDS"),
        ]

        for i, (key, label) in enumerate(fixed_rows):
            cb = QCheckBox(label)
            cb.setChecked(True)
            self.axis_checkboxes[key] = cb
            fixed_grid.addWidget(cb, i // 2, i % 2)

        outer.addLayout(fixed_grid)

        ellipse_box = QGroupBox("Ellipse subplots")
        self.ellipse_selector_layout = QVBoxLayout(ellipse_box)
        outer.addWidget(ellipse_box)
        self._sync_ellipse_axis_checkboxes()

        button_grid = QGridLayout()
        buttons = [
            ("Select all", "all"),
            ("Top only", "top"),
            ("Sign only", "sign"),
            ("Ellipses only", "ell"),
            ("None", "none"),
        ]

        for j, (label, mode) in enumerate(buttons):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, m=mode: self._select_axes(m))
            button_grid.addWidget(btn, j // 2, j % 2)

        outer.addLayout(button_grid)
        self.ctrl_layout.addWidget(group)

    def _sync_ellipse_axis_checkboxes(self):
        """Make ellipse subplot selectors match the current frequency list."""
        if not hasattr(self, "ellipse_selector_layout"):
            return

        for key in [k for k in self.axis_checkboxes if k.startswith("ell_")]:
            widget = self.axis_checkboxes.pop(key)
            self.ellipse_selector_layout.removeWidget(widget)
            widget.deleteLater()

        for i, freq in enumerate(self.ellipse_freqs, start=1):
            key = f"ell_{i}"
            cb = QCheckBox(f"Ellipse {i}: {freq:g} Hz")
            cb.setChecked(True)
            self.axis_checkboxes[key] = cb
            self.ellipse_selector_layout.addWidget(cb)

    def _build_limits_controls(self):
        group = QGroupBox("Top/sign axis limits and fonts")
        layout = QGridLayout(group)

        self.xmin_edit = QLineEdit(str(self.args.top_xmin))
        self.xmax_edit = QLineEdit(str(self.args.top_xmax))
        self.ymin_edit = QLineEdit(str(self.args.top_ymin))
        self.ymax_edit = QLineEdit(str(self.args.top_ymax))

        self.title_font_spin = QDoubleSpinBox()
        self.title_font_spin.setRange(5.0, 30.0)
        self.title_font_spin.setDecimals(1)
        self.title_font_spin.setSingleStep(0.5)
        self.title_font_spin.setValue(float(self.args.title_fontsize))

        self.axis_font_spin = QDoubleSpinBox()
        self.axis_font_spin.setRange(5.0, 30.0)
        self.axis_font_spin.setDecimals(1)
        self.axis_font_spin.setSingleStep(0.5)
        self.axis_font_spin.setValue(float(self.args.axis_fontsize))

        self.legend_font_spin = QDoubleSpinBox()
        self.legend_font_spin.setRange(5.0, 30.0)
        self.legend_font_spin.setDecimals(1)
        self.legend_font_spin.setSingleStep(0.5)
        self.legend_font_spin.setValue(float(self.args.legend_fontsize))

        layout.addWidget(QLabel("xmin"), 0, 0)
        layout.addWidget(self.xmin_edit, 0, 1)
        layout.addWidget(QLabel("xmax"), 1, 0)
        layout.addWidget(self.xmax_edit, 1, 1)
        layout.addWidget(QLabel("ymin"), 2, 0)
        layout.addWidget(self.ymin_edit, 2, 1)
        layout.addWidget(QLabel("ymax"), 3, 0)
        layout.addWidget(self.ymax_edit, 3, 1)
        layout.addWidget(QLabel("title fontsize"), 4, 0)
        layout.addWidget(self.title_font_spin, 4, 1)
        layout.addWidget(QLabel("axis fontsize"), 5, 0)
        layout.addWidget(self.axis_font_spin, 5, 1)
        layout.addWidget(QLabel("legend fontsize"), 6, 0)
        layout.addWidget(self.legend_font_spin, 6, 1)

        btn_lim = QPushButton("Apply x/y limits to selected")
        btn_font = QPushButton("Apply font sizes to selected")
        btn_both = QPushButton("Apply limits + font sizes")
        btn_lim.clicked.connect(self.apply_limits_to_selected)
        btn_font.clicked.connect(self.apply_fontsize_to_selected)
        btn_both.clicked.connect(self.apply_limits_and_fontsize)

        layout.addWidget(btn_lim, 7, 0, 1, 2)
        layout.addWidget(btn_font, 8, 0, 1, 2)
        layout.addWidget(btn_both, 9, 0, 1, 2)

        self.ctrl_layout.addWidget(group)

    def _build_ellipse_controls(self):
        group = QGroupBox("Ellipse controls")
        layout = QGridLayout(group)

        self.ellipse_freqs_edit = QLineEdit(
            ", ".join(f"{f:g}" for f in self.ellipse_freqs)
        )
        self.ellipse_freqs_edit.setToolTip(
            "Comma-, semicolon-, or whitespace-separated frequencies in Hz. "
            "Any positive number of frequencies may be entered."
        )

        self.exmin_edit = QLineEdit(str(self.args.ellipse_xmin))
        self.exmax_edit = QLineEdit(str(self.args.ellipse_xmax))
        self.eymin_edit = QLineEdit(str(self.args.ellipse_ymin))
        self.eymax_edit = QLineEdit(str(self.args.ellipse_ymax))

        self.gap_spin = QDoubleSpinBox()
        self.gap_spin.setRange(0.05, 1.50)
        self.gap_spin.setDecimals(2)
        self.gap_spin.setSingleStep(0.05)
        self.gap_spin.setValue(float(self.args.ellipse_gap))

        self.ellipse_wspace_spin = QDoubleSpinBox()
        self.ellipse_wspace_spin.setRange(-0.95, 1.50)
        self.ellipse_wspace_spin.setDecimals(2)
        self.ellipse_wspace_spin.setSingleStep(0.02)
        self.ellipse_wspace_spin.setValue(float(self.args.ellipse_wspace))

        self.show_scale_cb = QCheckBox("Show ellipse scale text")
        self.show_scale_cb.setChecked(True)

        self.ellipse_text_font_spin = QDoubleSpinBox()
        self.ellipse_text_font_spin.setRange(5.0, 30.0)
        self.ellipse_text_font_spin.setDecimals(1)
        self.ellipse_text_font_spin.setSingleStep(0.5)
        self.ellipse_text_font_spin.setValue(float(self.args.ellipse_text_fontsize))

        layout.addWidget(QLabel("Ellipse frequencies [Hz]"), 0, 0)
        layout.addWidget(self.ellipse_freqs_edit, 0, 1)

        freq_note = QLabel(
            "Enter any number of frequencies, e.g. 20, 50, 160, 500, 1000. "
            "Panels wrap automatically when many frequencies are requested."
        )
        freq_note.setWordWrap(True)
        layout.addWidget(freq_note, 1, 0, 1, 2)

        layout.addWidget(QLabel("ellipse xmin"), 2, 0)
        layout.addWidget(self.exmin_edit, 2, 1)
        layout.addWidget(QLabel("ellipse xmax"), 3, 0)
        layout.addWidget(self.exmax_edit, 3, 1)
        layout.addWidget(QLabel("ellipse ymin"), 4, 0)
        layout.addWidget(self.eymin_edit, 4, 1)
        layout.addWidget(QLabel("ellipse ymax"), 5, 0)
        layout.addWidget(self.eymax_edit, 5, 1)

        layout.addWidget(QLabel("ellipse vertical gap"), 6, 0)
        layout.addWidget(self.gap_spin, 6, 1)
        layout.addWidget(QLabel("ellipse horizontal spacing"), 7, 0)
        layout.addWidget(self.ellipse_wspace_spin, 7, 1)
        layout.addWidget(QLabel("ellipse text fontsize"), 8, 0)
        layout.addWidget(self.ellipse_text_font_spin, 8, 1)
        layout.addWidget(self.show_scale_cb, 9, 0, 1, 2)

        btn_apply_ell = QPushButton("Apply ellipse limits to selected")
        btn_apply_ell.clicked.connect(self.apply_ellipse_limits_to_selected)
        layout.addWidget(btn_apply_ell, 10, 0, 1, 2)

        self.ctrl_layout.addWidget(group)

    def _build_legend_controls(self):
        group = QGroupBox("Legend positions")
        layout = QGridLayout(group)

        # Three separate legend boxes:
        #   1) total QN only
        #   2) imprecision / back-action / |correlation|
        #   3) ellipse case colors
        # Each box can either stay attached to its original axes or be detached
        # and moved anywhere in normalized figure coordinates.
        def make_mode_combo():
            combo = QComboBox()
            combo.addItems(["axes (original)", "figure (free x/y)"])
            combo.setCurrentText("axes (original)")
            return combo

        def make_loc_combo(default, allow_none=True):
            combo = QComboBox()
            for loc in LEGEND_LOCS:
                if allow_none or loc != "none":
                    combo.addItem(loc)
            combo.setCurrentText(default)
            return combo

        def make_xy(x0, y0):
            x = QDoubleSpinBox()
            y = QDoubleSpinBox()
            for spin in (x, y):
                spin.setRange(-1.0, 2.0)
                spin.setDecimals(3)
                spin.setSingleStep(0.01)
            x.setValue(x0)
            y.setValue(y0)
            return x, y

        self.total_legend_mode = make_mode_combo()
        self.total_legend_loc = make_loc_combo("upper right")
        self.total_legend_x, self.total_legend_y = make_xy(0.27, 0.415)

        self.component_legend_mode = make_mode_combo()
        self.component_legend_mode.setCurrentText("figure (free x/y)")
        self.component_legend_loc = make_loc_combo("lower right")
        self.component_legend_x, self.component_legend_y = make_xy(0.352, 0.562)

        self.ellipse_legend_mode = make_mode_combo()
        self.ellipse_legend_loc = make_loc_combo("center", allow_none=False)
        self.ellipse_legend_x, self.ellipse_legend_y = make_xy(0.72, 0.415)

        r = 0
        layout.addWidget(QLabel("Total QN placement"), r, 0)
        layout.addWidget(self.total_legend_mode, r, 1)
        layout.addWidget(QLabel("Total QN loc"), r + 1, 0)
        layout.addWidget(self.total_legend_loc, r + 1, 1)
        layout.addWidget(QLabel("Total QN x"), r + 2, 0)
        layout.addWidget(self.total_legend_x, r + 2, 1)
        layout.addWidget(QLabel("Total QN y"), r + 3, 0)
        layout.addWidget(self.total_legend_y, r + 3, 1)

        r = 4
        layout.addWidget(QLabel("Components placement"), r, 0)
        layout.addWidget(self.component_legend_mode, r, 1)
        layout.addWidget(QLabel("Components loc"), r + 1, 0)
        layout.addWidget(self.component_legend_loc, r + 1, 1)
        layout.addWidget(QLabel("Components x"), r + 2, 0)
        layout.addWidget(self.component_legend_x, r + 2, 1)
        layout.addWidget(QLabel("Components y"), r + 3, 0)
        layout.addWidget(self.component_legend_y, r + 3, 1)

        r = 8
        layout.addWidget(QLabel("Ellipse placement"), r, 0)
        layout.addWidget(self.ellipse_legend_mode, r, 1)
        layout.addWidget(QLabel("Ellipse loc"), r + 1, 0)
        layout.addWidget(self.ellipse_legend_loc, r + 1, 1)
        layout.addWidget(QLabel("Ellipse x"), r + 2, 0)
        layout.addWidget(self.ellipse_legend_x, r + 2, 1)
        layout.addWidget(QLabel("Ellipse y"), r + 3, 0)
        layout.addWidget(self.ellipse_legend_y, r + 3, 1)

        btn_gap = QPushButton("Move all legends between rows")
        btn_gap.clicked.connect(self.place_legends_between_rows)
        layout.addWidget(btn_gap, r + 4, 0, 1, 2)

        btn = QPushButton("Apply legend positions / redraw")
        btn.clicked.connect(self.draw_figure)
        layout.addWidget(btn, r + 5, 0, 1, 2)

        note = QLabel(
            "Total QN and the decomposition components are separate legend boxes. "
            "Each box keeps its entries grouped vertically. In free x/y mode, x and y "
            "are normalized figure coordinates and may be outside 0–1."
        )
        note.setWordWrap(True)
        layout.addWidget(note, r + 6, 0, 1, 2)

        self.ctrl_layout.addWidget(group)

    def place_legends_between_rows(self):
        """Move all three intact legend boxes into the inter-row whitespace."""
        gap_y = 0.415
        try:
            # The whitespace is between the bottom of the sign-strip group and
            # the top of the ellipse group.  Use its actual center.
            sign_bottom = self.axes["sign_no_sqz"].get_position().y0
            ellipse_top = self.axes[next(k for k in self.axes if k.startswith("ell_"))].get_position().y1
            gap_y = 0.5 * (sign_bottom + ellipse_top)
        except Exception:
            pass

        self.total_legend_mode.setCurrentText("figure (free x/y)")
        self.total_legend_loc.setCurrentText("center")
        self.total_legend_x.setValue(0.25)
        self.total_legend_y.setValue(gap_y)

        self.component_legend_mode.setCurrentText("figure (free x/y)")
        self.component_legend_loc.setCurrentText("center")
        self.component_legend_x.setValue(0.50)
        self.component_legend_y.setValue(gap_y)

        self.ellipse_legend_mode.setCurrentText("figure (free x/y)")
        self.ellipse_legend_loc.setCurrentText("center")
        self.ellipse_legend_x.setValue(0.76)
        self.ellipse_legend_y.setValue(gap_y)

        self.draw_figure()

    def _build_export_controls(self, parent_layout):
        group = QGroupBox("Actions")
        layout = QVBoxLayout(group)

        btn_recompute = QPushButton("Recompute GWINC + redraw")
        btn_export = QPushButton("Export figure")
        btn_recompute.clicked.connect(self.recompute_and_redraw)
        btn_export.clicked.connect(self.export_figure)

        layout.addWidget(btn_recompute)
        layout.addWidget(btn_export)

        # Keep these outside the scrollable tabs so they are always visible.
        parent_layout.addWidget(group, stretch=0)

    def browse_yaml(self):
        """Choose a candidate GWINC YAML file without changing the active model yet."""
        start_dir = str(self.yaml_path.parent) if self.yaml_path else str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GWINC YAML model",
            start_dir,
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if path:
            self.yaml_edit.setText(path)

    def load_selected_yaml(self):
        """Load a new YAML, recompute with its parameters, and redraw.

        The candidate model is evaluated before replacing the current working
        model, so a bad YAML leaves the existing plot untouched.
        """
        raw = self.yaml_edit.text().strip()
        if not raw:
            QMessageBox.warning(self, "No YAML selected", "Please select a YAML file first.")
            return

        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()

        if not candidate.exists() or not candidate.is_file():
            QMessageBox.warning(
                self,
                "YAML not found",
                f"The selected file does not exist:\n{candidate}",
            )
            return

        self.yaml_status.setText(f"Loading: {candidate.name} ...")
        QApplication.processEvents()

        try:
            # Crucial: read the parameters from the NEW YAML.  Previously the
            # newly selected YAML was loaded and then most of its model values
            # were overwritten by the old hard-coded parameter dictionary.
            candidate_params = params_from_yaml(candidate, self.freq, self.params)

            candidate_data = {
                case: get_decomp_for_case(candidate, self.freq, case, candidate_params)
                for case in CASES
            }

            candidate_ellipse_freqs = parse_ellipse_frequency_text(
                self.ellipse_freqs_edit.text()
            )
            candidate_pts, candidate_actual, candidate_scales = compute_ellipse_points_and_scales(
                candidate_data,
                self.freq,
                candidate_ellipse_freqs,
                nsig=float(self.args.nsig),
            )

        except Exception as exc:
            self.yaml_status.setText(f"Active model: {self.yaml_path.name}")
            QMessageBox.critical(
                self,
                "GWINC/YAML error",
                f"Could not use:\n{candidate}\n\n{exc}",
            )
            return

        # Commit only after the full candidate computation succeeds.
        self.yaml_path = candidate
        self.params = candidate_params
        self.data = candidate_data
        self.ellipse_freqs = candidate_ellipse_freqs
        self.ellipse_pts = candidate_pts
        self.ellipse_actual_freqs = candidate_actual
        self.ellipse_scales = candidate_scales
        self._sync_ellipse_axis_checkboxes()

        self.yaml_edit.setText(str(candidate))
        self.yaml_status.setText(f"Active model: {candidate.name}")
        self.setWindowTitle(
            f"Quantum-noise decomposition GUI — {candidate.name}"
        )

        self.draw_figure()
        # Force the Qt canvas to paint now rather than merely scheduling it.
        self.canvas.draw()
        QApplication.processEvents()

    def _select_axes(self, mode: str):
        for key, cb in self.axis_checkboxes.items():
            if mode == "all":
                cb.setChecked(True)
            elif mode == "none":
                cb.setChecked(False)
            elif mode == "top":
                cb.setChecked(key.startswith("top_"))
            elif mode == "sign":
                cb.setChecked(key.startswith("sign_"))
            elif mode == "ell":
                cb.setChecked(key.startswith("ell_"))

    def selected_axis_keys(self):
        return [k for k, cb in self.axis_checkboxes.items() if cb.isChecked()]

    # ------------------------------------------------------------------
    # Data and drawing
    # ------------------------------------------------------------------

    def recompute_and_redraw(self):
        try:
            self.data = {
                case: get_decomp_for_case(self.yaml_path, self.freq, case, self.params)
                for case in CASES
            }
        except Exception as exc:
            QMessageBox.critical(self, "GWINC error", str(exc))
            raise

        self.redraw_with_current_controls()

    def redraw_with_current_controls(self):
        try:
            self.ellipse_freqs = parse_ellipse_frequency_text(
                self.ellipse_freqs_edit.text()
            )
        except Exception as exc:
            QMessageBox.warning(self, "Invalid ellipse frequencies", str(exc))
            return

        self._sync_ellipse_axis_checkboxes()

        self.ellipse_pts, self.ellipse_actual_freqs, self.ellipse_scales = compute_ellipse_points_and_scales(
            self.data,
            self.freq,
            self.ellipse_freqs,
            nsig=float(self.args.nsig),
        )

        self.draw_figure()

    def draw_figure(self):
        self.fig.clear()
        self.axes = {}

        title_fs = float(self.title_font_spin.value())
        axis_fs = float(self.axis_font_spin.value())
        legend_fs = float(self.legend_font_spin.value())

        n_ell = max(len(self.ellipse_freqs), 1)
        ell_ncols = min(5, n_ell)
        ell_nrows = int(np.ceil(n_ell / ell_ncols))

        outer = self.fig.add_gridspec(
            2, 1,
            height_ratios=[3.85, 2.15 * ell_nrows],
            hspace=float(self.gap_spin.value()),
        )

        gs_top = outer[0].subgridspec(
            2, 3,
            height_ratios=[3.0, 0.78],
            hspace=0.08,
            wspace=0.10,
        )

        gs_bot = outer[1].subgridspec(
            ell_nrows,
            ell_ncols,
            wspace=float(self.ellipse_wspace_spin.value()),
            hspace=0.28,
        )

        top_keys = ["top_no_sqz", "top_fis", "top_fds"]
        sign_keys = ["sign_no_sqz", "sign_fis", "sign_fds"]
        ell_keys = [f"ell_{i + 1}" for i in range(n_ell)]

        for j, key in enumerate(top_keys):
            self.axes[key] = self.fig.add_subplot(gs_top[0, j])

        for j, key in enumerate(sign_keys):
            self.axes[key] = self.fig.add_subplot(gs_top[1, j], sharex=self.axes["top_no_sqz"])

        for j, key in enumerate(ell_keys):
            row = j // ell_ncols
            col = j % ell_ncols
            self.axes[key] = self.fig.add_subplot(gs_bot[row, col])

        # Top and sign rows
        for j, case in enumerate(CASES):
            ax = self.axes[top_keys[j]]
            setup_log_axes(ax)
            plot_case_decomposition(ax, self.freq, self.data[case], case)
            set_case_title(ax, CASE_LABELS[case], CASE_COLORS[case], title_fs)

            if j == 0:
                ax.set_ylabel(r"Displacement ASD [m$/\sqrt{\mathrm{Hz}}$]")
            else:
                ax.tick_params(labelleft=False)
            ax.tick_params(axis="x", labelbottom=False)

            ax2 = self.axes[sign_keys[j]]
            setup_linear_strip(ax2)
            plot_xcorr_sign(ax2, self.freq, self.data[case])

            if j == 0:
                ax2.set_ylabel(r"$\mathrm{sgn}(S_{xx}^{imp,ba})$", labelpad=3)
            else:
                ax2.tick_params(labelleft=False)
            ax2.set_xlabel("Frequency [Hz]")

        # Ellipse row
        exmin, exmax, eymin, eymax = self._parse_ellipse_limits()

        for j, f_ellipse in enumerate(self.ellipse_freqs):
            ax = self.axes[ell_keys[j]]
            fkey = float(f_ellipse)
            pts_by_case = self.ellipse_pts[fkey]
            scale_factor = self.ellipse_scales[fkey]

            actual_vals = list(self.ellipse_actual_freqs[fkey].values())
            actual_mean = float(np.mean(actual_vals)) if len(actual_vals) else fkey
            freq_text = rf"$f={actual_mean:.0f}\,\mathrm{{Hz}}$"

            plot_combined_ellipse_panel(
                ax,
                pts_by_case,
                scale_factor,
                legend_outside=(j == len(ell_keys) - 1 and self.ellipse_legend_mode.currentText() == "axes (original)"),
                show_scale_text=self.show_scale_cb.isChecked(),
                freq_text=freq_text,
                default_xlim=(exmin, exmax),
                default_ylim=(eymin, eymax),
                ellipse_text_fontsize=float(self.ellipse_text_font_spin.value()),
                legend_fontsize=legend_fs,
            )

            if (j % ell_ncols) == 0:
                ax.set_ylabel("Back-action quadrature")
            else:
                ax.tick_params(labelleft=False)
            ax.set_xlabel("Imprecision quadrature")

        # Split the top-panel legend into two independent boxes: total QN only,
        # and the three decomposition components.  Both can either remain on
        # the Vacuum axes or be detached and moved in figure coordinates.
        top_ax = self.axes["top_no_sqz"]
        top_handles, top_labels = top_ax.get_legend_handles_labels()

        total_pairs = [(h, lab) for h, lab in zip(top_handles, top_labels) if lab == "total QN"]
        component_pairs = [(h, lab) for h, lab in zip(top_handles, top_labels) if lab != "total QN"]

        def draw_split_legend(pairs, mode_combo, loc_combo, x_spin, y_spin, *, keep_artist=False):
            if not pairs or loc_combo.currentText() == "none":
                return None
            handles = [p[0] for p in pairs]
            labels = [p[1] for p in pairs]
            loc = loc_combo.currentText()

            if mode_combo.currentText() == "axes (original)":
                legend = top_ax.legend(
                    handles, labels,
                    loc=loc,
                    frameon=True, framealpha=0.92,
                    fontsize=legend_fs,
                    handlelength=2.6, borderpad=0.45, labelspacing=0.35,
                )
                if keep_artist:
                    top_ax.add_artist(legend)
                return legend

            figure_loc = "center" if loc == "best" else loc
            return self.fig.legend(
                handles, labels,
                loc=figure_loc,
                bbox_to_anchor=(x_spin.value(), y_spin.value()),
                bbox_transform=self.fig.transFigure,
                ncol=1,
                frameon=True, framealpha=0.92,
                fontsize=legend_fs,
                handlelength=2.6, borderpad=0.45, labelspacing=0.35,
            )

        # If both legends are axes-attached, add the first as an artist so the
        # second ax.legend() call does not replace it.
        both_axes = (
            self.total_legend_mode.currentText() == "axes (original)"
            and self.component_legend_mode.currentText() == "axes (original)"
        )
        draw_split_legend(
            total_pairs,
            self.total_legend_mode, self.total_legend_loc,
            self.total_legend_x, self.total_legend_y,
            keep_artist=both_axes,
        )
        draw_split_legend(
            component_pairs,
            self.component_legend_mode, self.component_legend_loc,
            self.component_legend_x, self.component_legend_y,
            keep_artist=False,
        )

        if self.ellipse_legend_mode.currentText() == "figure (free x/y)":
            handles, labels = self.axes[ell_keys[-1]].get_legend_handles_labels()
            if handles:
                self.fig.legend(
                    handles, labels,
                    loc=self.ellipse_legend_loc.currentText(),
                    bbox_to_anchor=(self.ellipse_legend_x.value(), self.ellipse_legend_y.value()),
                    bbox_transform=self.fig.transFigure,
                    ncol=1,
                    frameon=True, framealpha=0.92,
                    fontsize=legend_fs,
                    handlelength=2.2, borderpad=0.35, labelspacing=0.35,
                )

        self._set_initial_limits()

        for ax in self.axes.values():
            set_axis_text_sizes(ax, title_fs, axis_fs, legend_fs)

        # Force the three top-panel case titles *after* every other styling
        # operation.  This is deliberately the last title operation before
        # drawing, so a backend/default font cannot silently replace the bold
        # face with a regular one.
        for j, case in enumerate(CASES):
            title = self.axes[top_keys[j]].title
            title.set_fontproperties(
                FontProperties(
                    family="DejaVu Sans",
                    weight=700,
                    size=title_fs,
                )
            )
            title.set_fontfamily("DejaVu Sans")
            title.set_fontweight(700)
            title.set_color(CASE_COLORS[case])

        self.fig.subplots_adjust(left=0.085, right=0.90, top=0.955, bottom=0.080)
        self.canvas.draw()
        QApplication.processEvents()

    def _set_initial_limits(self):
        xmin = float(self.xmin_edit.text())
        xmax = float(self.xmax_edit.text())
        ymin = float(self.ymin_edit.text())
        ymax = float(self.ymax_edit.text())

        for key in ["top_no_sqz", "top_fis", "top_fds"]:
            ax = self.axes[key]
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        for key in ["sign_no_sqz", "sign_fis", "sign_fds"]:
            self.axes[key].set_xlim(xmin, xmax)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _parse_limits(self):
        def parse_float_or_none(txt: str):
            txt = txt.strip()
            if txt == "":
                return None
            return float(txt)

        return (
            parse_float_or_none(self.xmin_edit.text()),
            parse_float_or_none(self.xmax_edit.text()),
            parse_float_or_none(self.ymin_edit.text()),
            parse_float_or_none(self.ymax_edit.text()),
        )

    def _parse_ellipse_limits(self):
        return (
            float(self.exmin_edit.text()),
            float(self.exmax_edit.text()),
            float(self.eymin_edit.text()),
            float(self.eymax_edit.text()),
        )

    def apply_limits_to_selected(self):
        try:
            xmin, xmax, ymin, ymax = self._parse_limits()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid limits", str(exc))
            return

        for key in self.selected_axis_keys():
            ax = self.axes.get(key)
            if ax is None:
                continue

            if xmin is not None and xmax is not None:
                ax.set_xlim(xmin, xmax)
            if ymin is not None and ymax is not None:
                ax.set_ylim(ymin, ymax)

        self.canvas.draw_idle()

    def apply_ellipse_limits_to_selected(self):
        try:
            xmin, xmax, ymin, ymax = self._parse_ellipse_limits()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid ellipse limits", str(exc))
            return

        for key in self.selected_axis_keys():
            if not key.startswith("ell_"):
                continue
            ax = self.axes.get(key)
            if ax is None:
                continue
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            set_ellipse_ticks(ax)

        self.canvas.draw_idle()

    def apply_fontsize_to_selected(self):
        title_fontsize = float(self.title_font_spin.value())
        axis_fontsize = float(self.axis_font_spin.value())
        legend_fontsize = float(self.legend_font_spin.value())

        for key in self.selected_axis_keys():
            ax = self.axes.get(key)
            if ax is None:
                continue
            set_axis_text_sizes(ax, title_fontsize, axis_fontsize, legend_fontsize)

        self.canvas.draw()
        QApplication.processEvents()

    def apply_limits_and_fontsize(self):
        self.apply_limits_to_selected()
        self.apply_fontsize_to_selected()

    def export_figure(self):
        """Export the displayed figure without calling Figure.savefig().

        In some long-lived Spyder/Qt sessions another script can replace or
        intercept Figure.savefig().  Calling the canvas' print_figure() method
        directly bypasses that interception and uses Matplotlib's backend
        export machinery itself.
        """
        try:
            self.canvas.draw()
            QApplication.processEvents()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export failed",
                f"Could not finalize the current figure before saving.\n\n{exc}",
            )
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export figure",
            "fig3_quantum_noise_decomposition.pdf",
            "PDF (*.pdf);;PNG (*.png);;SVG (*.svg);;All files (*)",
        )
        if not path:
            return

        out = Path(path).expanduser()

        # Qt/macOS does not always append an extension from the selected filter.
        suffix = out.suffix.lower()
        if suffix not in {".pdf", ".png", ".svg"}:
            if "PNG" in selected_filter:
                suffix = ".png"
            elif "SVG" in selected_filter:
                suffix = ".svg"
            else:
                suffix = ".pdf"
            out = out.with_suffix(suffix)

        fmt = suffix.lstrip(".")

        try:
            parent = out.parent
            if not parent.exists():
                raise FileNotFoundError(
                    f"Destination folder does not exist:\n{parent}"
                )
            if not parent.is_dir():
                raise NotADirectoryError(
                    f"Destination is not a folder:\n{parent}"
                )

            # Remove a zero-byte/partial file left by a previous failed attempt.
            if out.exists() and out.stat().st_size == 0:
                out.unlink()

            print(
                f"[fig3-gui] Exporting via canvas.print_figure -> {out}",
                flush=True,
            )

            kwargs = {
                "format": fmt,
                "facecolor": self.fig.get_facecolor(),
                "edgecolor": self.fig.get_edgecolor(),
            }
            if fmt == "png":
                kwargs["dpi"] = 300

            # IMPORTANT: bypass self.fig.savefig().  If another script has
            # monkey-patched/intercepted Figure.savefig in the Spyder session,
            # canvas.print_figure still goes through the actual backend.
            self.canvas.print_figure(str(out), **kwargs)

            if not out.exists():
                raise IOError(
                    "Matplotlib returned without error, but no output file "
                    f"was created:\n{out}"
                )

            size = out.stat().st_size
            if size <= 0:
                raise IOError(
                    "Matplotlib created the output path, but the file is empty."
                )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export failed",
                f"Could not save the figure to:\n{out}\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            print(
                f"[fig3-gui] Export failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return

        print(
            f"[fig3-gui] Exported figure successfully: {out} "
            f"({out.stat().st_size} bytes)",
            flush=True,
        )
        QMessageBox.information(
            self,
            "Export complete",
            f"Saved successfully:\n{out}\n\n"
            f"Size: {out.stat().st_size / 1024:.1f} kB",
        )


# =============================================================================
# Command-line interface
# =============================================================================

def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=str, default=None)

    ap.add_argument("--fmin", type=float, default=DEFAULT["fmin"])
    ap.add_argument("--fmax", type=float, default=DEFAULT["fmax"])
    ap.add_argument("--npts", type=int, default=DEFAULT["npts"])

    ap.add_argument("--top-xmin", type=float, default=DEFAULT["fmin"])
    ap.add_argument("--top-xmax", type=float, default=DEFAULT["fmax"])
    ap.add_argument("--top-ymin", type=float, default=2e-22)
    ap.add_argument("--top-ymax", type=float, default=7e-19)

    ap.add_argument(
        "--ellipse-freqs",
        type=float,
        nargs="+",
        default=[20.0, 160.0, 1000.0],
        metavar="F",
        help="One or more ellipse frequencies in Hz",
    )
    ap.add_argument("--nsig", type=float, default=1.0)
    ap.add_argument("--ellipse-gap", type=float, default=0.25)
    ap.add_argument("--ellipse-wspace", type=float, default=-0.32)
    ap.add_argument("--ellipse-xmin", type=float, default=-4.0)
    ap.add_argument("--ellipse-xmax", type=float, default=4.0)
    ap.add_argument("--ellipse-ymin", type=float, default=-4.0)
    ap.add_argument("--ellipse-ymax", type=float, default=4.0)
    ap.add_argument("--ellipse-text-fontsize", type=float, default=16.0)

    ap.add_argument("--fontsize", type=float, default=15.0, help="Legacy base fontsize")
    ap.add_argument("--title-fontsize", type=float, default=15.0)
    ap.add_argument("--axis-fontsize", type=float, default=15.0)
    ap.add_argument("--legend-fontsize", type=float, default=13.0)
    return ap


def main():
    print("[fig3-gui] Starting Figure 3 GUI...", flush=True)
    args = build_arg_parser().parse_args()

    if args.yaml:
        yaml_path = Path(args.yaml).expanduser().resolve()
    else:
        yaml_path = auto_find_yaml()

    print(f"[fig3-gui] YAML: {yaml_path}", flush=True)
    params = dict(DEFAULT)
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titleweight": "bold",
        "axes.titlesize": args.title_fontsize,
    })

    app = QApplication.instance()
    created_app = False
    if app is None:
        print("[fig3-gui] Creating QApplication...", flush=True)
        app = QApplication(sys.argv)
        created_app = True
    else:
        print("[fig3-gui] Reusing existing QApplication...", flush=True)

    print("[fig3-gui] Creating main window. This may take a little while while GWINC runs...", flush=True)
    win = Fig3Window(yaml_path=yaml_path, params=params, args=args)

    print("[fig3-gui] Showing window...", flush=True)
    win.show()
    win.raise_()
    win.activateWindow()

    app._fig3_window = win
    globals()["_FIG3_WINDOW"] = win
    globals()["_FIG3_APP"] = app

    print("[fig3-gui] Window should now be visible.", flush=True)

    if created_app:
        print("[fig3-gui] Entering Qt event loop. Close the GUI window to return.", flush=True)
        return app.exec_()

    print("[fig3-gui] Existing Qt application detected; returning window object.", flush=True)
    return win


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        sys.exit(result)
