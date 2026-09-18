#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

GWINC_AVAILABLE = True
GWINC_IMPORT_ERROR = None
try:
    import gwinc
    from gwinc.noise import quantum as gwinc_quantum
except Exception as exc:
    GWINC_AVAILABLE = False
    GWINC_IMPORT_ERROR = exc
    gwinc = None
    gwinc_quantum = None

QT_AVAILABLE = True
QT_IMPORT_ERROR = None
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from PyQt5.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
        QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox, QTabWidget,
        QVBoxLayout, QWidget
    )
except Exception as exc:
    QT_AVAILABLE = False
    QT_IMPORT_ERROR = exc
    QApplication = None
    QMainWindow = object

C = 299_792_458.0

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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_params_from_yaml(path: Path) -> dict:
    y = read_yaml(path)
    sqz = y.get("Squeezer", {})
    fc = sqz.get("FilterCavity", {})
    sqz_angle_rad = float(sqz.get("SQZAngle", 0.0))

    laser = y.get("Laser", {})
    power = y.get("Power", {})
    infrastructure = y.get("Infrastructure", {})
    optics = y.get("Optics", {})
    srm = optics.get("SRM", {}) if isinstance(optics, dict) else {}
    mm = y.get("ModeMismatch", {})
    if not isinstance(mm, dict):
        mm = {}

    arm_power = laser.get("ArmPower", power.get("ArmPower", 269572.0))
    tunephase_rad = float(srm.get("Tunephase", 0.0)) if isinstance(srm, dict) else 0.0

    # This GWINC branch contains both the newer ModeMismatch block and the
    # legacy Optics/Squeezer MM_* fields.  Prefer ModeMismatch when present,
    # but retain the legacy keys for the YAMLs used by the existing fit code.
    mm_ifo_omc = mm.get("IFO_OMC_L", optics.get("MM_IFO_OMC", 0.0))
    mm_ifo_omc_phi = mm.get("IFO_OMC_phi", optics.get("MM_IFO_OMCphi", 0.0))
    mm_ifo_sec = mm.get("SEC_ARM_L", optics.get("MM_ARM_SRC", 0.0))
    mm_ifo_sec_phi = mm.get("SEC_ARM_phi", optics.get("MM_ARM_SRCphi", 0.0))
    mm_sqz_omc = mm.get("SQZ_OMC_L", sqz.get("MM_SQZ_OMC", 0.0))
    mm_sqz_omc_phi = mm.get("SQZ_OMC_phi", sqz.get("MM_SQZ_OMCphi", 0.0))

    return {
        "sqz_db": float(sqz.get("AmplitudedB", 15.0)),
        "inj_loss": float(sqz.get("InjectionLoss", 0.05)),
        "sqz_angle_deg": sqz_angle_rad * 180.0 / np.pi,
        "fc_L_m": float(fc.get("L", 300.0)),
        "fc_Ti": float(fc.get("Ti", 8.0e-4)),
        "fc_Te": float(fc.get("Te", 1.0e-6)),
        "fc_Lrt": float(fc.get("Lrt", 1.0e-4)),
        "fc_detune_Hz": float(fc.get("fdetune", -25.0)),
        "arm_power_W": float(arm_power),
        "arm_length_m": float(infrastructure.get("Length", 3995.0)),
        "sec_detuning_rad": tunephase_rad,
        "mm_ifo_omc": float(mm_ifo_omc),
        "mm_ifo_omc_phase_rad": float(mm_ifo_omc_phi),
        "mm_sqz_omc": float(mm_sqz_omc),
        "mm_sqz_omc_phase_rad": float(mm_sqz_omc_phi),
        "mm_ifo_sec": float(mm_ifo_sec),
        "mm_ifo_sec_phase_rad": float(mm_ifo_sec_phi),
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


def add_color_items(combo):
    for name, hex_color in COLOR_CHOICES:
        combo.addItem(f"{name}  {hex_color}", hex_color)


def set_combo_hex(combo, hexval):
    for i in range(combo.count()):
        if str(combo.itemData(i)).lower() == str(hexval).lower():
            combo.setCurrentIndex(i)
            return


def set_combo_text(combo, text):
    i = combo.findText(str(text))
    if i >= 0:
        combo.setCurrentIndex(i)


def format_frequency_label(f: float, custom_labels: list[str], idx: int, fmt: str) -> str:
    if idx < len(custom_labels):
        return custom_labels[idx]
    return str(fmt).format(f=f)


# ---------------------------------------------------------------------
# IFO propagation model helpers
# ---------------------------------------------------------------------

def _set_gwinc_arm_power(ifo, arm_power_W: float):
    if hasattr(ifo, "Laser") and hasattr(ifo.Laser, "ArmPower"):
        ifo.Laser.ArmPower = float(arm_power_W)
    elif hasattr(ifo, "Power") and hasattr(ifo.Power, "ArmPower"):
        ifo.Power.ArmPower = float(arm_power_W)
    else:
        raise AttributeError("Cannot set ArmPower on this GWINC ifo struct")


def _set_gwinc_sec_detuning(ifo, sec_detuning_rad: float):
    if hasattr(ifo, "Optics") and hasattr(ifo.Optics, "SRM") and hasattr(ifo.Optics.SRM, "Tunephase"):
        # Optics.SRM.Tunephase is represented in radians in both the YAML and GWINC.
        ifo.Optics.SRM.Tunephase = float(sec_detuning_rad)
    else:
        raise AttributeError("Cannot set Optics.SRM.Tunephase on this GWINC ifo struct")


def _set_gwinc_mode_matching_fields(ifo, s: dict):
    """Write GUI mismatch overrides into the exact fields used by this quantum.py."""
    if hasattr(ifo, "Optics"):
        ifo.Optics.MM_IFO_OMC = float(s["gwinc_mm_ifo_omc"])
        ifo.Optics.MM_IFO_OMCphi = float(s["gwinc_mm_ifo_omc_phase_rad"])
        ifo.Optics.MM_ARM_SRC = float(s["gwinc_mm_ifo_sec"])
        ifo.Optics.MM_ARM_SRCphi = float(s["gwinc_mm_ifo_sec_phase_rad"])
    if hasattr(ifo, "Squeezer"):
        ifo.Squeezer.MM_SQZ_OMC = float(s["gwinc_mm_sqz_omc"])
        ifo.Squeezer.MM_SQZ_OMCphi = float(s["gwinc_mm_sqz_omc_phase_rad"])


def _matrix_at_frequency(M, idx: int, nfreq: int) -> np.ndarray:
    A = np.asarray(M)
    if A.ndim == 2:
        return A
    if A.ndim >= 3 and A.shape[0] == nfreq:
        return A[idx]
    raise ValueError(f"Unexpected GWINC matrix shape {A.shape}; expected (N,n,n) or (n,n).")


def _embed_fundamental_covariance(V2: np.ndarray, ndim: int) -> np.ndarray:
    if ndim < 2:
        raise ValueError("GWINC optical matrix has fewer than two quadrature dimensions.")
    V = np.eye(ndim, dtype=complex)
    V[:2, :2] = np.asarray(V2, dtype=complex)
    return V


def _gwinc_fundamental_projector(olib) -> np.ndarray:
    """Return a 2 x N projector onto the GWINC readout fundamental mode.

    Once mode mismatch is enabled, GWINC promotes the optical matrices into a
    larger fundamental+HOM basis.  GWINC itself does not recover the detected
    fundamental mode by slicing the first two output rows; it uses the LO
    vectors generated by the active optical matrix library.  We do the same
    here for two orthogonal quadratures.
    """
    qrow = np.asarray(gwinc_quantum.adjoint(olib.LO(0.0)), dtype=complex)
    prow = np.asarray(gwinc_quantum.adjoint(olib.LO(np.pi / 2.0)), dtype=complex)

    # In the GWINC quantum code these are row vectors, used as LOa @ H.
    qrow = np.squeeze(qrow)
    prow = np.squeeze(prow)
    if qrow.ndim != 1 or prow.ndim != 1:
        raise ValueError(
            f"Unexpected GWINC LO-vector shapes after squeeze: {qrow.shape}, {prow.shape}"
        )
    if qrow.shape != prow.shape:
        raise ValueError(f"GWINC LO-vector shape mismatch: {qrow.shape} vs {prow.shape}")

    P = np.vstack([qrow, prow])
    gram = P @ P.conj().T
    if not np.allclose(gram, np.eye(2), rtol=1e-7, atol=1e-9):
        raise ValueError(
            "GWINC fundamental-mode quadrature projector is not orthonormal; "
            f"P P^dagger = {gram}"
        )
    return P


def _project_fundamental_symmetrized_covariance(S: np.ndarray, projector: np.ndarray) -> np.ndarray:
    """Project a full GWINC spatial-mode covariance onto fundamental X1/X2."""
    S = np.asarray(S, dtype=complex)
    P = np.asarray(projector, dtype=complex)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"Expected square covariance matrix, got {S.shape}")
    if P.ndim != 2 or P.shape[0] != 2 or P.shape[1] != S.shape[0]:
        raise ValueError(
            f"Projector/covariance dimension mismatch: projector {P.shape}, covariance {S.shape}"
        )

    S2 = P @ S @ P.conj().T
    S2 = 0.5 * (S2 + S2.conj().T)
    V2 = np.real(S2)
    V2 = 0.5 * (V2 + V2.T)
    return V2


def _build_gwinc_ifo_stage(yaml_path: Path, freqs_hz, s: dict, ideal_zero_loss_tuned: bool):
    """Return the GWINC IFO reflection matrix and IFO loss-vacuum matrices.

    The stage begins immediately before the interferometer and ends at the
    interferometer output basis used by GWINC.  Input filter-cavity preparation
    is intentionally handled separately by compute_input_covariances().

    ideal_zero_loss_tuned=True:
        same arm/SRM optical parameters and arm power as the YAML/GWINC model,
        but force IFO losses, detuning, and mode mismatch to zero.

    ideal_zero_loss_tuned=False:
        use the full IFO parameters in the GWINC model, including arm/SRC loss,
        SEC detuning, mode mismatch, Gouy phases, and the optomechanical response.
    """
    if not GWINC_AVAILABLE:
        raise RuntimeError(
            "GWINC could not be imported. This figure requires the same GWINC "
            f"environment as your quantum-noise script. Import error: {GWINC_IMPORT_ERROR}"
        )

    f = np.asarray(freqs_hz, dtype=float)
    budget = gwinc.load_budget(str(yaml_path), f, bname="Quantum")
    ifo = budget.ifo

    _set_gwinc_arm_power(ifo, s["gwinc_arm_power_W"])
    _set_gwinc_sec_detuning(ifo, s["gwinc_sec_detuning_rad"])
    if hasattr(ifo, "Infrastructure") and hasattr(ifo.Infrastructure, "Length"):
        ifo.Infrastructure.Length = float(s["gwinc_arm_length_m"])

    # IMPORTANT: put the GUI mode-mismatch overrides into the IFO Struct itself
    # before asking GWINC for its debug/precomputed result.  This gives us an
    # independent check through GWINC's own ASbudget['lossMM'] calculation.
    if not ideal_zero_loss_tuned:
        _set_gwinc_mode_matching_fields(ifo, s)

    # shotrad_debug gives us the exact suspension susceptibility and IFO power
    # precomputation used by this installed quantum.py.
    dbg = gwinc_quantum.shotrad_debug(f, ifo)

    mats = gwinc_quantum.MatsHelper()
    mats.H["AS"] = mats.olib.Id
    access = type(dbg.access)()

    bsloss = ifo.Optics.BSLoss
    mismatch = (1 - ifo.Optics.coupling) + ifo.TCS.SRCloss

    if ideal_zero_loss_tuned:
        loss_arm = 0.0
        loss_src = 0.0
        sec_length_rms = 0.0
        srm_detune_rad = 0.0
        mm_src_arm_L = 0.0
        mm_src_arm_rad = 0.0
        mm_ifo_omc_L = 0.0
        mm_ifo_omc_rad = 0.0
        mm_sqz_omc_L = 0.0
        mm_sqz_omc_rad = 0.0
        direct_mm_sqz_ifo = False
        is_opd = False
        src_gouy_rad = None
        arm_gouy_rad = None
    else:
        loss_arm = 1 - (1 - ifo.Optics.Loss)**2 * (1 - ifo.Optics.ETM.Transmittance)
        loss_src = 1 - (1 - mismatch) * (1 - bsloss)
        sec_length_rms = ifo.Optics.SRM.get("LengthRMS", 0)
        srm_detune_rad = ifo.Optics.SRM.Tunephase / 2
        src_gouy_rad = ifo.Optics.SRM.get("SRCGouy_rad", None)
        arm_gouy_rad = gwinc_quantum.arm_gouyRT(
            ifo.Optics.Curvature.ITM,
            ifo.Infrastructure.Length,
            ifo.Optics.Curvature.ETM,
        )
        mm_src_arm_L = float(s["gwinc_mm_ifo_sec"])
        mm_src_arm_rad = float(s["gwinc_mm_ifo_sec_phase_rad"])
        mm_ifo_omc_L = float(s["gwinc_mm_ifo_omc"])
        mm_ifo_omc_rad = float(s["gwinc_mm_ifo_omc_phase_rad"])

        if hasattr(ifo, "Squeezer"):
            mm_sqz_omc_L = float(s["gwinc_mm_sqz_omc"])
            mm_sqz_omc_rad = float(s["gwinc_mm_sqz_omc_phase_rad"])
            direct_mm_sqz_ifo = ifo.Squeezer.get("direct_mm_sqz_ifo", False)
        else:
            mm_sqz_omc_L = 0.0
            mm_sqz_omc_rad = 0.0
            direct_mm_sqz_ifo = True
        is_opd = ifo.Optics.get("is_OPD", False)

    gwinc_quantum.apply_interferometer(
        lambda_=ifo.Laser.Wavelength,
        F_Hz=f,
        mats=mats,
        tst_suscept=dbg.sustf.tst_suscept,
        T_ITM=ifo.Optics.ITM.Transmittance,
        T_SRM=ifo.Optics.SRM.Transmittance,
        L_SRM_m=ifo.Optics.SRM.CavityLength,
        L_ARM_m=ifo.Infrastructure.Length,
        Loss_ARM=loss_arm,
        Loss_SRC=loss_src,
        parm_W=dbg.power.parm,
        SEC_lengthRMS_m=sec_length_rms,
        ARM_detune_rad=0,
        SRM_detune_rad=srm_detune_rad,
        SRC_gouy_rad=src_gouy_rad,
        ARM_gouy_rad=arm_gouy_rad,
        MM_SRC_ARM_L=mm_src_arm_L,
        MM_SRC_ARM_rad=mm_src_arm_rad,
        MM_IFO_OMC_L=mm_ifo_omc_L,
        MM_IFO_OMC_rad=mm_ifo_omc_rad,
        MM_SQZ_OMC_L=mm_sqz_omc_L,
        MM_SQZ_OMC_rad=mm_sqz_omc_rad,
        folded=ifo.Infrastructure.get("folded", False),
        direct_mm_sqz_ifo=direct_mm_sqz_ifo,
        is_OPD=is_opd,
        access=access,
    )

    H_ifo = mats.H["AS"]
    loss_mats = dict(mats.T)
    output_olib = mats.olib
    meta = {
        "model": "ideal_zero_loss_tuned" if ideal_zero_loss_tuned else "full_gwinc_ifo",
        "loss_arm": float(loss_arm),
        "loss_src": float(loss_src),
        "sec_detuning_rad": 0.0 if ideal_zero_loss_tuned else float(s["gwinc_sec_detuning_rad"]),
        "mm_ifo_omc": float(mm_ifo_omc_L),
        "mm_ifo_omc_phase_rad": float(mm_ifo_omc_rad),
        "mm_sqz_omc": float(mm_sqz_omc_L),
        "mm_sqz_omc_phase_rad": float(mm_sqz_omc_rad),
        "mm_ifo_sec": float(mm_src_arm_L),
        "mm_ifo_sec_phase_rad": float(mm_src_arm_rad),
        "optical_basis": getattr(output_olib, "__name__", output_olib.__class__.__name__),
        "gwinc_lossMM": np.asarray(getattr(dbg, "ASbudget", {}).get("lossMM", np.zeros_like(f)), dtype=float).tolist()
            if hasattr(dbg, "ASbudget") else [float("nan") for _ in f],
    }
    return H_ifo, loss_mats, output_olib, meta


def _propagate_one_gwinc_stage(Vin_list, freqs_hz, H_stack, loss_stacks, output_olib):
    """Propagate a list of 2x2 fundamental-mode covariances through one GWINC stage."""
    P_fund = _gwinc_fundamental_projector(output_olib)
    nfreq = len(freqs_hz)
    out = []
    matrix_dims = []
    for i, Vin in enumerate(Vin_list):
        H = _matrix_at_frequency(H_stack, i, nfreq)
        if H.shape[0] != H.shape[1]:
            raise ValueError(f"Expected a square GWINC IFO matrix, got {H.shape}")
        matrix_dims.append(int(H.shape[0]))

        Vfull = _embed_fundamental_covariance(Vin, H.shape[1])
        Sout = H @ Vfull @ H.conj().T
        for Tstack in loss_stacks.values():
            T = _matrix_at_frequency(Tstack, i, nfreq)
            Sout = Sout + T @ T.conj().T

        out.append(_project_fundamental_symmetrized_covariance(Sout, P_fund))
    return out, matrix_dims, P_fund


def _relative_covariance_difference(A: np.ndarray, B: np.ndarray) -> float:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    den = max(np.linalg.norm(B, ord="fro"), 1e-15)
    return float(np.linalg.norm(A - B, ord="fro") / den)


def propagate_covariances_through_ifo(Vin_list, freqs_hz, s: dict):
    ideal = s["ifo_model"] == "ideal"
    H_stack, loss_stacks, output_olib, model_meta = _build_gwinc_ifo_stage(
        s["yaml_path"], freqs_hz, s, ideal_zero_loss_tuned=ideal
    )

    out, matrix_dims, P_fund = _propagate_one_gwinc_stage(
        Vin_list, freqs_hz, H_stack, loss_stacks, output_olib
    )

    model_meta["matrix_dims"] = matrix_dims
    model_meta["fundamental_projector_shape"] = tuple(P_fund.shape)
    model_meta["output_projection"] = "GWINC LO fundamental-mode projector"

    # Numerical mode-mismatch diagnostic.  Recompute the *same* full-GWINC
    # model with all three mismatch amplitudes set to zero, then compare the
    # resulting 2x2 output covariance.  This tells us whether the user-entered
    # mismatch is actually changing the state even if the plot looks similar.
    if ideal:
        model_meta["mm_diagnostic"] = "ideal_model_forces_mm_zero"
        model_meta["mm_cov_relative_change"] = [0.0 for _ in freqs_hz]
        model_meta["zero_mm_matrix_dims"] = list(matrix_dims)
    else:
        s0 = dict(s)
        s0["gwinc_mm_ifo_omc"] = 0.0
        s0["gwinc_mm_sqz_omc"] = 0.0
        s0["gwinc_mm_ifo_sec"] = 0.0
        H0, T0, olib0, _ = _build_gwinc_ifo_stage(
            s0["yaml_path"], freqs_hz, s0, ideal_zero_loss_tuned=False
        )
        out0, matrix_dims0, _ = _propagate_one_gwinc_stage(
            Vin_list, freqs_hz, H0, T0, olib0
        )
        rel = [_relative_covariance_difference(a, b) for a, b in zip(out, out0)]
        model_meta["mm_diagnostic"] = "compared_to_zero_mm"
        model_meta["mm_cov_relative_change"] = rel
        model_meta["zero_mm_matrix_dims"] = matrix_dims0

    return out, model_meta


def compute_input_covariances(freqs_hz: list[float], s: dict):
    params = s["params"]
    sqz_db = s["visual_sqz_db"] if s["use_visual_sqz"] else params["sqz_db"]
    eta = 1.0 - params["inj_loss"] if s["include_injection_loss"] else 1.0
    eta = np.clip(eta, 0.0, 1.0)

    state = s["input_state"]
    if state == "vacuum":
        Vin = [np.eye(2) for _ in freqs_hz]
        meta = {"fc_theta_deg": [0.0 for _ in freqs_hz], "input_angle_deg": [0.0 for _ in freqs_hz]}
        return Vin, meta

    if state == "fis":
        angle = params["sqz_angle_deg"] + s["fis_extra_angle_deg"]
        Vin = [covariance_from_db(sqz_db, angle, eta=eta) for _ in freqs_hz]
        meta = {"fc_theta_deg": [0.0 for _ in freqs_hz], "input_angle_deg": [float(angle) for _ in freqs_hz]}
        return Vin, meta

    theta_fc = fc_rotation_at_frequencies(
        freqs_hz,
        params,
        sign=s["fc_sign"],
        offset_deg=s["fc_offset_deg"],
        zero_at_high_f=s["zero_fc_rotation_at_high_f"],
    )
    Vin = []
    input_angles = []
    for th in theta_fc:
        angle = params["sqz_angle_deg"] + np.rad2deg(th) + s["fds_extra_angle_deg"]
        input_angles.append(float(angle))
        Vin.append(covariance_from_db(sqz_db, angle, eta=eta))
    meta = {"fc_theta_deg": [float(np.rad2deg(x)) for x in theta_fc], "input_angle_deg": input_angles}
    return Vin, meta


# ---------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------

def build_ifo_optical_figure(s, fig=None, apply_figsize=True):
    if fig is None:
        fig = Figure(figsize=(s["fig_w"], s["fig_h"]), dpi=110)
    else:
        fig.clear()
        if apply_figsize:
            fig.set_size_inches(s["fig_w"], s["fig_h"], forward=False)

    font_family = str(s["font_family"]).strip()

    freqs_all = parse_freq_list(s["freqs_text"])
    if not freqs_all:
        freqs_all = [10.0, 30.0, 60.0, 150.0, 300.0]
    nfreq = min(int(s["num_freqs"]), len(freqs_all), 10)
    if nfreq < 1:
        raise ValueError("Number of frequencies to plot must be at least 1.")
    freqs = [float(x) for x in freqs_all[:nfreq]]

    labels = [format_frequency_label(f, s["custom_labels"], i, s["label_format"]) for i, f in enumerate(freqs)]
    Vin_list, meta = compute_input_covariances(freqs, s)
    Vout_list, ifo_meta = propagate_covariances_through_ifo(Vin_list, freqs, s)

    def font_kwargs(size, weight):
        kw = {"fontsize": size, "fontweight": weight}
        if font_family:
            kw["fontfamily"] = font_family
        return kw

    input_xlim = (float(s["input_xmin"]), float(s["input_xmax"]))
    input_ylim = (float(s["input_ymin"]), float(s["input_ymax"]))
    output_xlim = (float(s["output_xmin"]), float(s["output_xmax"]))
    output_ylim = (float(s["output_ymin"]), float(s["output_ymax"]))

    sets_to_draw = []
    if s["show_input_panel"]:
        sets_to_draw.append({
            "kind": "input",
            "title": s["input_title"],
            "color": s["input_color"],
            "Vlist": Vin_list,
            "xlim": input_xlim,
            "ylim": input_ylim,
            "show_legend": s["show_input_legend"],
            "legend_loc": s["input_legend_loc"],
            "legend_x": s["input_legend_x"],
            "legend_y": s["input_legend_y"],
        })
    if s["show_output_panel"]:
        sets_to_draw.append({
            "kind": "output",
            "title": s["output_title"],
            "color": s["output_color"],
            "Vlist": Vout_list,
            "xlim": output_xlim,
            "ylim": output_ylim,
            "show_legend": s["show_output_legend"],
            "legend_loc": s["output_legend_loc"],
            "legend_x": s["output_legend_x"],
            "legend_y": s["output_legend_y"],
        })
    if not sets_to_draw:
        raise ValueError("At least one of input/output panels must be selected.")

    ncols = min(5, len(freqs))
    rows_per_set = int(np.ceil(len(freqs) / 5.0))
    total_rows = rows_per_set * len(sets_to_draw)
    gs = fig.add_gridspec(total_rows, ncols, wspace=s["wspace"], hspace=s["hspace"])

    set_axes = {}
    freq_mode = s.get("freq_label_mode", "titles")

    for set_idx, set_info in enumerate(sets_to_draw):
        base_row = set_idx * rows_per_set
        axes_for_this_set = []
        for i, (Vcur, lab) in enumerate(zip(set_info["Vlist"], labels)):
            local_row = i // 5
            local_col = i % 5
            global_row = base_row + local_row
            ax = fig.add_subplot(gs[global_row, local_col])
            axes_for_this_set.append(ax)

            x, y = ellipse_points_from_cov(Vcur)
            ax.fill(x, y, facecolor=set_info["color"], edgecolor="none", alpha=s["fill_alpha"], zorder=1)
            ax.plot(
                x,
                y,
                color=set_info["color"],
                lw=s["line_lw"],
                alpha=s["line_alpha"],
                label=lab,
                zorder=2,
            )

            ax.set_aspect("equal", adjustable="box")
            ax.set_box_aspect(1)
            ax.set_xlim(*set_info["xlim"])
            ax.set_ylim(*set_info["ylim"])

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

            # Shared y-axis label across side-by-side figures: only first column gets the subplot ylabel.
            if local_col == 0:
                ax.set_ylabel(
                    s["ylabel"],
                    labelpad=s["ylabel_pad"],
                    **font_kwargs(s["label_fontsize"], s["label_fontweight"]),
                )
            else:
                ax.set_ylabel("")

            # Show x label only on the bottom row of a given set.
            if local_row == rows_per_set - 1:
                ax.set_xlabel(
                    s["xlabel"],
                    labelpad=s["xlabel_pad"],
                    **font_kwargs(s["label_fontsize"], s["label_fontweight"]),
                )
            else:
                ax.set_xlabel("")

            if freq_mode == "titles":
                ax.set_title(
                    lab,
                    pad=s["title_pad"],
                    **font_kwargs(s["title_fontsize"], s["title_fontweight"]),
                )
            else:
                ax.set_title("")
                if set_info["show_legend"]:
                    prop = {"size": s["legend_fontsize"], "weight": s["legend_fontweight"]}
                    if font_family:
                        prop["family"] = font_family
                    ax.legend(
                        loc=set_info["legend_loc"],
                        bbox_to_anchor=(set_info["legend_x"], set_info["legend_y"]),
                        prop=prop,
                        frameon=s["legend_frame"],
                        framealpha=s["legend_framealpha"],
                    )

            for spine in ax.spines.values():
                spine.set_color("0.75")
                spine.set_linewidth(0.8)

        set_axes[set_info["kind"]] = axes_for_this_set

    fig.subplots_adjust(left=s["left"], right=s["right"], bottom=s["bottom"], top=s["top"])

    # Shared set titles centered over each set block.
    for set_info in sets_to_draw:
        axes = set_axes[set_info["kind"]]
        if not axes:
            continue
        x0 = min(ax.get_position().x0 for ax in axes)
        x1 = max(ax.get_position().x1 for ax in axes)
        y1 = max(ax.get_position().y1 for ax in axes)
        set_title_gap = s["input_set_title_gap"] if set_info["kind"] == "input" else s["output_set_title_gap"]
        fig.text(
            0.5 * (x0 + x1),
            min(0.995, y1 + set_title_gap),
            set_info["title"],
            ha="center",
            va="bottom",
            **font_kwargs(s["title_fontsize"] + 1.0, s["title_fontweight"]),
        )

    summary_rows = []
    for f, fc_theta, input_angle, Vin, Vout in zip(
        freqs, meta["fc_theta_deg"], meta["input_angle_deg"], Vin_list, Vout_list
    ):
        summary_rows.append({
            "f_Hz": float(f),
            "fc_theta_deg": float(fc_theta),
            "input_angle_deg": float(input_angle),
            "ifo_model": ifo_meta["model"],
            "ifo_loss_arm": ifo_meta["loss_arm"],
            "ifo_loss_src": ifo_meta["loss_src"],
            "ifo_sec_detuning_rad": ifo_meta["sec_detuning_rad"],
            "ifo_optical_basis": ifo_meta.get("optical_basis", "unknown"),
            "ifo_matrix_dim": int(ifo_meta.get("matrix_dims", [2] * len(freqs))[len(summary_rows)]),
            "ifo_output_projection": ifo_meta.get("output_projection", "unknown"),
            "ifo_mm_cov_relative_change": float(ifo_meta.get("mm_cov_relative_change", [0.0] * len(freqs))[len(summary_rows)]),
            "ifo_zero_mm_matrix_dim": int(ifo_meta.get("zero_mm_matrix_dims", ifo_meta.get("matrix_dims", [2] * len(freqs)))[len(summary_rows)]),
            "ifo_mm_diagnostic": ifo_meta.get("mm_diagnostic", "unknown"),
            "ifo_gwinc_lossMM": float(ifo_meta.get("gwinc_lossMM", [float("nan")] * len(freqs))[len(summary_rows)]),
            "input_xlim": [float(input_xlim[0]), float(input_xlim[1])],
            "input_ylim": [float(input_ylim[0]), float(input_ylim[1])],
            "output_xlim": [float(output_xlim[0]), float(output_xlim[1])],
            "output_ylim": [float(output_ylim[0]), float(output_ylim[1])],
            "V_in": Vin.tolist(),
            "V_out": Vout.tolist(),
        })

    return fig, summary_rows


# ---------------------------------------------------------------------
# Qt controls
# ---------------------------------------------------------------------

class LegendControls:
    def __init__(self, title: str, default_loc: str = "upper right", default_x: float = 1.0, default_y: float = 1.0):
        self.box = QGroupBox(title)
        f = QFormLayout(self.box)
        f.setRowWrapPolicy(QFormLayout.WrapLongRows)

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
        for w, val in [(self.x, default_x), (self.y, default_y)]:
            w.setRange(-2.0, 3.0)
            w.setDecimals(2)
            w.setSingleStep(0.05)
            w.setValue(val)

        f.addRow(self.show)
        f.addRow("Legend loc", self.loc)
        f.addRow("Legend x", self.x)
        f.addRow("Legend y", self.y)


class IFOOpticalPropagationWindow(QMainWindow):
    def __init__(self, yaml_path: Path):
        super().__init__()
        self.yaml_path = yaml_path
        self.params = load_params_from_yaml(yaml_path)

        self.setWindowTitle("Optical Quadrature Propagation through the IFO")
        self.resize(1680, 920)

        self._build_ui()
        self.redraw()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setContentsMargins(4, 4, 4, 4)
        main.setSpacing(4)

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(11.5, 8.5), dpi=110)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main.addWidget(plot_widget, stretch=1)

        right_widget = QWidget()
        right_widget.setFixedWidth(460)
        self.right_widget = right_widget
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self.control_tabs = QTabWidget()
        right_layout.addWidget(self.control_tabs, stretch=1)

        self._add_control_layer(
            "Input / FC",
            (self._build_model_group, self._build_input_group),
        )
        self._add_control_layer(
            "IFO",
            (self._build_ifo_group,),
        )
        self._add_control_layer(
            "Layout",
            (self._build_figure_group,),
        )
        self._add_control_layer(
            "Text",
            (self._build_text_group,),
        )
        self._add_control_layer(
            "Style",
            (self._build_style_group, self._build_legend_group),
        )

        self._build_actions_group(right_layout)
        main.addWidget(right_widget, stretch=0)

    def _add_control_layer(self, title, builders):
        """Add one independently scrollable control layer to the right panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        controls = QWidget()
        self.ctrl = QVBoxLayout(controls)
        self.ctrl.setContentsMargins(4, 4, 4, 4)
        self.ctrl.setSpacing(4)
        for build_group in builders:
            build_group()
        self.ctrl.addStretch(1)

        scroll.setWidget(controls)
        self.control_tabs.addTab(scroll, title)

    def _build_model_group(self):
        box = QGroupBox("Squeezer / filter-cavity model")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        self.yaml_label = QLabel(self.yaml_path.name)
        self.yaml_label.setToolTip(str(self.yaml_path))

        self.select_yaml_button = QPushButton("Select base YAML and reload")
        self.select_yaml_button.clicked.connect(self.select_base_yaml)

        self.use_visual_sqz = QCheckBox("Use visual/schematic squeezing [dB]")
        self.use_visual_sqz.setChecked(False)

        self.visual_sqz_db = QDoubleSpinBox()
        self.visual_sqz_db.setRange(0.0, 20.0)
        self.visual_sqz_db.setDecimals(2)
        self.visual_sqz_db.setValue(6.00)

        self.physical_sqz_label = QLabel(f"{self.params['sqz_db']:.3f}")

        self.include_inj_loss = QCheckBox("Include injection loss in input ellipses")
        self.include_inj_loss.setChecked(True)

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

        form.addRow("Base YAML", self.yaml_label)
        form.addRow(self.select_yaml_button)
        form.addRow("Physical squeezing from YAML [dB]", self.physical_sqz_label)
        form.addRow(self.use_visual_sqz)
        form.addRow("Visual squeezing [dB]", self.visual_sqz_db)
        form.addRow(self.include_inj_loss)
        form.addRow("FC rotation sign", self.fc_sign)
        form.addRow("FC rotation offset [deg]", self.fc_offset)
        form.addRow(self.zero_high_f)
        form.addRow("Extra FIS angle [deg]", self.fis_extra_angle)
        form.addRow("Extra FDS angle [deg]", self.fds_extra_angle)

        self.ctrl.addWidget(box)

    def _build_input_group(self):
        box = QGroupBox("Input-state / frequency controls")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        self.input_state = QComboBox()
        self.input_state.addItem("Vacuum", "vacuum")
        self.input_state.addItem("FIS", "fis")
        self.input_state.addItem("FDS", "fds")
        self.input_state.setCurrentIndex(2)

        self.freqs = QLineEdit("10,30,60,150,300,500,1000,2000")

        self.num_freqs = QSpinBox()
        self.num_freqs.setRange(1, 10)
        self.num_freqs.setValue(5)

        self.show_input_panel = QCheckBox("Show input panel")
        self.show_input_panel.setChecked(True)
        self.show_output_panel = QCheckBox("Show output panel")
        self.show_output_panel.setChecked(True)

        form.addRow("Input state family", self.input_state)
        form.addRow("Frequencies [Hz] (up to 10)", self.freqs)
        form.addRow("How many to plot", self.num_freqs)
        form.addRow(self.show_input_panel)
        form.addRow(self.show_output_panel)

        self.ctrl.addWidget(box)

    def _build_ifo_group(self):
        box = QGroupBox("IFO propagation model")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        self.ifo_model = QComboBox()
        self.ifo_model.addItem("Zero-loss tuned IFO (ideal)", "ideal")
        self.ifo_model.addItem("Full GWINC IFO", "gwinc")
        self.ifo_model.setCurrentIndex(1)

        self.gwinc_arm_power = QDoubleSpinBox()
        self.gwinc_arm_power.setRange(1.0, 5.0e6)
        self.gwinc_arm_power.setDecimals(1)
        self.gwinc_arm_power.setSingleStep(1000.0)
        self.gwinc_arm_power.setValue(float(self.params["arm_power_W"]))

        self.gwinc_arm_length = QDoubleSpinBox()
        self.gwinc_arm_length.setRange(1000.0, 10000.0)
        self.gwinc_arm_length.setDecimals(2)
        self.gwinc_arm_length.setSingleStep(1.0)
        self.gwinc_arm_length.setValue(float(self.params["arm_length_m"]))

        self.gwinc_sec_detuning = QDoubleSpinBox()
        self.gwinc_sec_detuning.setRange(-np.pi, np.pi)
        self.gwinc_sec_detuning.setDecimals(7)
        self.gwinc_sec_detuning.setSingleStep(0.001)
        self.gwinc_sec_detuning.setValue(float(self.params["sec_detuning_rad"]))

        note = QLabel(
            "Ideal: uses GWINC's own interferometer matrix with IFO loss, SEC detuning, "
            "and mode mismatch forced to zero.\n"
            "Full GWINC: uses the actual IFO matrix from your installed quantum.py, "
            "including IFO loss, SEC detuning, mismatch, Gouy phases, and internal loss vacua.\n"
            "When mismatch promotes the model into a HOM basis, the output ellipse is "
            "projected back onto the readout fundamental mode using GWINC's own LO vectors.\n"
            "The plotted input state is defined immediately before the IFO."
        )
        note.setWordWrap(True)

        form.addRow("IFO model", self.ifo_model)
        form.addRow("Arm power [W]", self.gwinc_arm_power)
        form.addRow("Arm length [m]", self.gwinc_arm_length)
        form.addRow("SEC detuning [rad]", self.gwinc_sec_detuning)
        form.addRow(note)

        self.ctrl.addWidget(box)

        mm_box = QGroupBox("Mode-matching overrides")
        mm_grid = QGridLayout(mm_box)
        mm_grid.setContentsMargins(6, 6, 6, 6)
        mm_grid.setHorizontalSpacing(6)
        mm_grid.setVerticalSpacing(4)

        mm_grid.addWidget(QLabel("Path"), 0, 0)
        mm_grid.addWidget(QLabel("Mismatch"), 0, 1)
        mm_grid.addWidget(QLabel("Phase [rad]"), 0, 2)

        def mismatch_spin(value, tooltip):
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setDecimals(6)
            spin.setSingleStep(0.001)
            spin.setValue(float(value))
            spin.setToolTip(tooltip)
            return spin

        def phase_spin(value, tooltip):
            spin = QDoubleSpinBox()
            spin.setRange(-2.0 * np.pi, 2.0 * np.pi)
            spin.setDecimals(6)
            spin.setSingleStep(0.01)
            spin.setValue(float(value))
            spin.setToolTip(tooltip)
            return spin

        self.gwinc_mm_ifo_omc = mismatch_spin(
            self.params["mm_ifo_omc"], "Overrides YAML Optics.MM_IFO_OMC"
        )
        self.gwinc_mm_ifo_omc_phase = phase_spin(
            self.params["mm_ifo_omc_phase_rad"], "Overrides YAML Optics.MM_IFO_OMCphi"
        )
        self.gwinc_mm_sqz_omc = mismatch_spin(
            self.params["mm_sqz_omc"], "Overrides YAML Squeezer.MM_SQZ_OMC"
        )
        self.gwinc_mm_sqz_omc_phase = phase_spin(
            self.params["mm_sqz_omc_phase_rad"], "Overrides YAML Squeezer.MM_SQZ_OMCphi"
        )
        self.gwinc_mm_ifo_sec = mismatch_spin(
            self.params["mm_ifo_sec"], "Overrides YAML Optics.MM_ARM_SRC"
        )
        self.gwinc_mm_ifo_sec_phase = phase_spin(
            self.params["mm_ifo_sec_phase_rad"], "Overrides YAML Optics.MM_ARM_SRCphi"
        )

        for row, (label, mismatch, phase) in enumerate([
            ("IFO–OMC", self.gwinc_mm_ifo_omc, self.gwinc_mm_ifo_omc_phase),
            ("SQZ–OMC", self.gwinc_mm_sqz_omc, self.gwinc_mm_sqz_omc_phase),
            ("IFO–SEC", self.gwinc_mm_ifo_sec, self.gwinc_mm_ifo_sec_phase),
        ], start=1):
            mm_grid.addWidget(QLabel(label), row, 0)
            mm_grid.addWidget(mismatch, row, 1)
            mm_grid.addWidget(phase, row, 2)

        mm_note = QLabel(
            "Loaded from the selected YAML. Edits override the Full GWINC model; "
            "the ideal model forces all three mismatches to zero."
        )
        mm_note.setWordWrap(True)
        mm_grid.addWidget(mm_note, 4, 0, 1, 3)

        self.ctrl.addWidget(mm_box)

    def _build_figure_group(self):
        box = QGroupBox("Figure layout")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        self.fig_w = QDoubleSpinBox()
        self.fig_h = QDoubleSpinBox()
        for w, val in [(self.fig_w, 12.50), (self.fig_h, 9.00)]:
            w.setRange(2.0, 30.0)
            w.setDecimals(2)
            w.setSingleStep(0.1)
            w.setValue(val)

        self.wspace = QDoubleSpinBox()
        self.wspace.setRange(-0.80, 3.0)
        self.wspace.setDecimals(2)
        self.wspace.setSingleStep(0.02)
        self.wspace.setValue(0.28)

        self.hspace = QDoubleSpinBox()
        self.hspace.setRange(-0.80, 3.0)
        self.hspace.setDecimals(2)
        self.hspace.setSingleStep(0.02)
        self.hspace.setValue(0.32)

        self.left = QDoubleSpinBox()
        self.right = QDoubleSpinBox()
        self.bottom = QDoubleSpinBox()
        self.top = QDoubleSpinBox()
        for w, val in [
            (self.left, 0.08),
            (self.right, 0.92),
            (self.bottom, 0.14),
            (self.top, 0.84),
        ]:
            w.setRange(0.0, 1.5)
            w.setDecimals(2)
            w.setSingleStep(0.02)
            w.setValue(val)

        def axis_limit_spin(value):
            spin = QDoubleSpinBox()
            spin.setRange(-1000.0, 1000.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.25)
            spin.setValue(float(value))
            return spin

        # Shared limits: every input subplot uses these four values, and every
        # output subplot uses the second set.  x and y may be asymmetric.
        self.input_xmin = axis_limit_spin(-5.0)
        self.input_xmax = axis_limit_spin(+5.0)
        self.input_ymin = axis_limit_spin(-5.0)
        self.input_ymax = axis_limit_spin(+5.0)
        self.output_xmin = axis_limit_spin(-5.0)
        self.output_xmax = axis_limit_spin(+5.0)
        self.output_ymin = axis_limit_spin(-5.0)
        self.output_ymax = axis_limit_spin(+5.0)

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

        form.addRow("Export width [in]", self.fig_w)
        form.addRow("Export height [in]", self.fig_h)
        form.addRow("Horizontal spacing (wspace)", self.wspace)
        form.addRow("Vertical spacing (hspace)", self.hspace)
        form.addRow("Left margin", self.left)
        form.addRow("Right margin", self.right)
        form.addRow("Bottom margin", self.bottom)
        form.addRow("Top margin", self.top)
        input_limits_label = QLabel("Input axes — shared by all input panels")
        input_limits_label.setStyleSheet("font-weight: 600;")
        output_limits_label = QLabel("Output axes — shared by all output panels")
        output_limits_label.setStyleSheet("font-weight: 600;")
        form.addRow(input_limits_label)
        form.addRow("Input x min", self.input_xmin)
        form.addRow("Input x max", self.input_xmax)
        form.addRow("Input y min", self.input_ymin)
        form.addRow("Input y max", self.input_ymax)
        form.addRow(output_limits_label)
        form.addRow("Output x min", self.output_xmin)
        form.addRow("Output x max", self.output_xmax)
        form.addRow("Output y min", self.output_ymin)
        form.addRow("Output y max", self.output_ymax)
        form.addRow(self.show_ticks)
        form.addRow(self.show_grid)
        form.addRow(self.show_crosshairs)
        form.addRow("Grid alpha", self.grid_alpha)

        self.ctrl.addWidget(box)

    def _build_text_group(self):
        box = QGroupBox("Titles / labels / fonts")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        self.input_title = QLineEdit("Input optical quadratures")
        self.output_title = QLineEdit("After IFO propagation")

        self.freq_label_mode = QComboBox()
        self.freq_label_mode.addItem("Show frequencies in subplot titles", "titles")
        self.freq_label_mode.addItem("Show frequencies in legends", "legends")
        self.freq_label_mode.setCurrentIndex(0)

        self.label_format = QLineEdit("{f:g} Hz")
        self.custom_labels = QLineEdit("")
        self.custom_labels.setPlaceholderText("Optional custom frequency labels, comma-separated")

        self.xlabel = QLineEdit(r"$X_1$")
        self.ylabel = QLineEdit(r"$X_2$")
        self.font_family = QLineEdit("Arial")

        self.title_fs = QDoubleSpinBox()
        self.label_fs = QDoubleSpinBox()
        self.tick_fs = QDoubleSpinBox()
        self.legend_fs = QDoubleSpinBox()
        for w, val in [
            (self.title_fs, 12.0),
            (self.label_fs, 11.0),
            (self.tick_fs, 9.0),
            (self.legend_fs, 9.0),
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

        self.input_set_title_gap = QDoubleSpinBox()
        self.output_set_title_gap = QDoubleSpinBox()
        for w in [self.input_set_title_gap, self.output_set_title_gap]:
            w.setRange(-0.05, 0.20)
            w.setDecimals(3)
            w.setSingleStep(0.005)
            w.setValue(0.050)

        form.addRow("Input panel title", self.input_title)
        form.addRow("Output panel title", self.output_title)
        form.addRow("Frequency label placement", self.freq_label_mode)
        form.addRow("Frequency label format", self.label_format)
        form.addRow("Custom frequency labels", self.custom_labels)
        form.addRow("x label", self.xlabel)
        form.addRow("y label", self.ylabel)
        form.addRow("Font family", self.font_family)
        form.addRow("Title fontsize", self.title_fs)
        form.addRow("Title weight", self.title_weight)
        form.addRow("Axis-label fontsize", self.label_fs)
        form.addRow("Axis-label weight", self.label_weight)
        form.addRow("Tick fontsize", self.tick_fs)
        form.addRow("Tick weight", self.tick_weight)
        form.addRow("Legend fontsize", self.legend_fs)
        form.addRow("Legend weight", self.legend_weight)
        form.addRow("Frequency-title pad", self.title_pad)
        form.addRow("Input set-title gap", self.input_set_title_gap)
        form.addRow("Output set-title gap", self.output_set_title_gap)
        form.addRow("x-label pad", self.xlabel_pad)
        form.addRow("y-label pad", self.ylabel_pad)

        self.ctrl.addWidget(box)

    def _build_style_group(self):
        box = QGroupBox("Ellipse style")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        self.input_color = QComboBox()
        self.output_color = QComboBox()
        for combo in [self.input_color, self.output_color]:
            add_color_items(combo)
        set_combo_hex(self.input_color, "#d900ff")
        set_combo_hex(self.output_color, "#008c9e")

        self.line_lw = QDoubleSpinBox()
        self.line_lw.setRange(0.1, 10.0)
        self.line_lw.setDecimals(1)
        self.line_lw.setSingleStep(0.2)
        self.line_lw.setValue(2.2)

        self.line_alpha = QDoubleSpinBox()
        self.fill_alpha = QDoubleSpinBox()
        for w, val in [
            (self.line_alpha, 1.00),
            (self.fill_alpha, 0.08),
        ]:
            w.setRange(0.0, 1.0)
            w.setDecimals(2)
            w.setSingleStep(0.05)
            w.setValue(val)

        form.addRow("Input panel color", self.input_color)
        form.addRow("Output panel color", self.output_color)
        form.addRow("Linewidth", self.line_lw)
        form.addRow("Line alpha (all frequencies)", self.line_alpha)
        form.addRow("Fill alpha", self.fill_alpha)

        self.ctrl.addWidget(box)

    def _build_legend_group(self):
        box = QGroupBox("Legend settings")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        self.legend_frame = QCheckBox("Legend frame")
        self.legend_frame.setChecked(True)

        self.legend_framealpha = QDoubleSpinBox()
        self.legend_framealpha.setRange(0.0, 1.0)
        self.legend_framealpha.setDecimals(2)
        self.legend_framealpha.setSingleStep(0.05)
        self.legend_framealpha.setValue(0.92)

        self.input_leg = LegendControls("Input legend", default_loc="upper right", default_x=1.0, default_y=1.0)
        self.output_leg = LegendControls("Output legend", default_loc="upper right", default_x=1.0, default_y=1.0)

        form.addRow(self.legend_frame)
        form.addRow("Legend frame alpha", self.legend_framealpha)
        self.ctrl.addWidget(box)
        self.ctrl.addWidget(self.input_leg.box)
        self.ctrl.addWidget(self.output_leg.box)

    def _load_yaml_values_into_controls(self, params):
        """Refresh every visible control whose starting value comes from YAML."""
        self.physical_sqz_label.setText(f"{params['sqz_db']:.3f}")
        self.gwinc_arm_power.setValue(float(params["arm_power_W"]))
        self.gwinc_arm_length.setValue(float(params["arm_length_m"]))
        self.gwinc_sec_detuning.setValue(float(params["sec_detuning_rad"]))
        self.gwinc_mm_ifo_omc.setValue(float(params["mm_ifo_omc"]))
        self.gwinc_mm_ifo_omc_phase.setValue(float(params["mm_ifo_omc_phase_rad"]))
        self.gwinc_mm_sqz_omc.setValue(float(params["mm_sqz_omc"]))
        self.gwinc_mm_sqz_omc_phase.setValue(float(params["mm_sqz_omc_phase_rad"]))
        self.gwinc_mm_ifo_sec.setValue(float(params["mm_ifo_sec"]))
        self.gwinc_mm_ifo_sec_phase.setValue(float(params["mm_ifo_sec_phase_rad"]))

    def select_base_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select base GWINC YAML",
            str(self.yaml_path.parent),
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return

        yaml_path = Path(path).expanduser().resolve()
        try:
            params = load_params_from_yaml(yaml_path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load YAML", str(exc))
            return

        self.yaml_path = yaml_path
        self.params = params
        self.yaml_label.setText(yaml_path.name)
        self.yaml_label.setToolTip(str(yaml_path))
        self._load_yaml_values_into_controls(params)
        self.redraw()

    def _build_actions_group(self, parent_layout):
        box = QGroupBox("Actions")
        v = QVBoxLayout(box)

        btn_redraw = QPushButton("Redraw")
        btn_export = QPushButton("Export")
        btn_save_defaults = QPushButton("Save PDF/PNG/SVG here")

        btn_redraw.clicked.connect(self.redraw)
        btn_export.clicked.connect(self.export_figure)
        btn_save_defaults.clicked.connect(self.save_defaults_here)

        self.model_status = QLabel("GWINC basis status will appear after redraw.")
        self.model_status.setWordWrap(True)

        v.addWidget(btn_redraw)
        v.addWidget(btn_export)
        v.addWidget(btn_save_defaults)
        v.addWidget(self.model_status)
        parent_layout.addWidget(box, stretch=0)

    def gather_settings(self):
        return dict(
            params=self.params,
            use_visual_sqz=self.use_visual_sqz.isChecked(),
            visual_sqz_db=float(self.visual_sqz_db.value()),
            include_injection_loss=self.include_inj_loss.isChecked(),
            fc_sign=float(self.fc_sign.currentData()),
            fc_offset_deg=float(self.fc_offset.value()),
            zero_fc_rotation_at_high_f=self.zero_high_f.isChecked(),
            fis_extra_angle_deg=float(self.fis_extra_angle.value()),
            fds_extra_angle_deg=float(self.fds_extra_angle.value()),

            input_state=str(self.input_state.currentData()),
            freqs_text=self.freqs.text(),
            num_freqs=int(self.num_freqs.value()),
            show_input_panel=self.show_input_panel.isChecked(),
            show_output_panel=self.show_output_panel.isChecked(),
            input_title=self.input_title.text(),
            output_title=self.output_title.text(),
            freq_label_mode=str(self.freq_label_mode.currentData()),
            label_format=self.label_format.text(),
            custom_labels=parse_text_list(self.custom_labels.text()),

            ifo_model=str(self.ifo_model.currentData()),
            yaml_path=self.yaml_path,
            gwinc_arm_power_W=float(self.gwinc_arm_power.value()),
            gwinc_arm_length_m=float(self.gwinc_arm_length.value()),
            gwinc_sec_detuning_rad=float(self.gwinc_sec_detuning.value()),
            gwinc_mm_ifo_omc=float(self.gwinc_mm_ifo_omc.value()),
            gwinc_mm_ifo_omc_phase_rad=float(self.gwinc_mm_ifo_omc_phase.value()),
            gwinc_mm_sqz_omc=float(self.gwinc_mm_sqz_omc.value()),
            gwinc_mm_sqz_omc_phase_rad=float(self.gwinc_mm_sqz_omc_phase.value()),
            gwinc_mm_ifo_sec=float(self.gwinc_mm_ifo_sec.value()),
            gwinc_mm_ifo_sec_phase_rad=float(self.gwinc_mm_ifo_sec_phase.value()),

            fig_w=float(self.fig_w.value()),
            fig_h=float(self.fig_h.value()),
            wspace=float(self.wspace.value()),
            hspace=float(self.hspace.value()),
            left=float(self.left.value()),
            right=float(self.right.value()),
            bottom=float(self.bottom.value()),
            top=float(self.top.value()),
            input_xmin=float(self.input_xmin.value()),
            input_xmax=float(self.input_xmax.value()),
            input_ymin=float(self.input_ymin.value()),
            input_ymax=float(self.input_ymax.value()),
            output_xmin=float(self.output_xmin.value()),
            output_xmax=float(self.output_xmax.value()),
            output_ymin=float(self.output_ymin.value()),
            output_ymax=float(self.output_ymax.value()),
            show_ticks=self.show_ticks.isChecked(),
            show_grid=self.show_grid.isChecked(),
            show_crosshairs=self.show_crosshairs.isChecked(),
            grid_alpha=float(self.grid_alpha.value()),

            xlabel=self.xlabel.text(),
            ylabel=self.ylabel.text(),
            font_family=self.font_family.text(),
            title_fontsize=float(self.title_fs.value()),
            label_fontsize=float(self.label_fs.value()),
            tick_fontsize=float(self.tick_fs.value()),
            legend_fontsize=float(self.legend_fs.value()),
            title_fontweight=self.title_weight.currentText(),
            label_fontweight=self.label_weight.currentText(),
            tick_fontweight=self.tick_weight.currentText(),
            legend_fontweight=self.legend_weight.currentText(),
            title_pad=float(self.title_pad.value()),
            input_set_title_gap=float(self.input_set_title_gap.value()),
            output_set_title_gap=float(self.output_set_title_gap.value()),
            xlabel_pad=float(self.xlabel_pad.value()),
            ylabel_pad=float(self.ylabel_pad.value()),

            input_color=self.input_color.currentData(),
            output_color=self.output_color.currentData(),
            line_lw=float(self.line_lw.value()),
            line_alpha=float(self.line_alpha.value()),
            fill_alpha=float(self.fill_alpha.value()),

            legend_frame=self.legend_frame.isChecked(),
            legend_framealpha=float(self.legend_framealpha.value()),
            show_input_legend=self.input_leg.show.isChecked(),
            input_legend_loc=self.input_leg.loc.currentText(),
            input_legend_x=float(self.input_leg.x.value()),
            input_legend_y=float(self.input_leg.y.value()),
            show_output_legend=self.output_leg.show.isChecked(),
            output_legend_loc=self.output_leg.loc.currentText(),
            output_legend_x=float(self.output_leg.x.value()),
            output_legend_y=float(self.output_leg.y.value()),
        )

    def redraw(self):
        try:
            s = self.gather_settings()
            if not (s["input_xmin"] < s["input_xmax"]):
                raise ValueError("Input x min must be smaller than input x max.")
            if not (s["input_ymin"] < s["input_ymax"]):
                raise ValueError("Input y min must be smaller than input y max.")
            if not (s["output_xmin"] < s["output_xmax"]):
                raise ValueError("Output x min must be smaller than output x max.")
            if not (s["output_ymin"] < s["output_ymax"]):
                raise ValueError("Output y min must be smaller than output y max.")
            if not (0.0 <= s["left"] < s["right"] <= 1.5):
                raise ValueError("Margins must satisfy left < right.")
            if not (0.0 <= s["bottom"] < s["top"] <= 1.5):
                raise ValueError("Margins must satisfy bottom < top.")

            freqs = parse_freq_list(s["freqs_text"])
            if len(freqs) == 0:
                raise ValueError("Please enter at least one frequency.")
            if len(freqs) > 10:
                raise ValueError("Provide at most 10 frequencies in the list.")
            if s["num_freqs"] > len(freqs):
                raise ValueError("'How many to plot' cannot exceed the number of entered frequencies.")
            if not (s["show_input_panel"] or s["show_output_panel"]):
                raise ValueError("Select at least one of input/output panels.")

            self.fig, summary_rows = build_ifo_optical_figure(s, fig=self.fig, apply_figsize=False)
            if summary_rows:
                row0 = summary_rows[0]
                mmvals = (
                    s["gwinc_mm_ifo_omc"],
                    s["gwinc_mm_sqz_omc"],
                    s["gwinc_mm_ifo_sec"],
                )
                if s["ifo_model"] == "ideal":
                    status = (
                        "WARNING: Ideal IFO selected — all mode mismatches are forced to zero. | "
                        f"Arm power is still active: {s['gwinc_arm_power_W']:.1f} W"
                    )
                else:
                    mm_changes = [r.get("ifo_mm_cov_relative_change", 0.0) for r in summary_rows]
                    max_dv = max(mm_changes) if mm_changes else 0.0
                    status = (
                        f"Full GWINC | MM requested (IFO-OMC, SQZ-OMC, IFO-SEC) = "
                        f"({mmvals[0]:.6g}, {mmvals[1]:.6g}, {mmvals[2]:.6g}) | "
                        f"basis dim current/zero-MM = {row0.get('ifo_matrix_dim','?')}/"
                        f"{row0.get('ifo_zero_mm_matrix_dim','?')} | "
                        f"max relative ΔV vs zero-MM = {max_dv:.3e} | "
                        f"GWINC lossMM@{row0.get('f_Hz','?'):g}Hz = {row0.get('ifo_gwinc_lossMM', float('nan')):.3e}"
                    )
                self.model_status.setText(status)
                print("[MM diagnostic] " + status, flush=True)
                for r in summary_rows:
                    print(
                        f"[MM diagnostic] f={r['f_Hz']:g} Hz  "
                        f"rel_dV={r.get('ifo_mm_cov_relative_change',0.0):.6e}  "
                        f"dim={r.get('ifo_matrix_dim','?')}  "
                        f"zeroMM_dim={r.get('ifo_zero_mm_matrix_dim','?')}",
                        flush=True,
                    )
            self.canvas.draw_idle()
        except Exception as exc:
            QMessageBox.critical(self, "Redraw failed", str(exc))
            raise

    def _save_live_figure(self, path, dpi=300):
        """Save exactly the figure panel currently visible in the GUI.

        This path must not rebuild the model and must not reflow the layout.
        We therefore capture the *already-rendered* Qt canvas exactly as shown.
        For raster formats this is a literal pixel snapshot of the current panel.
        For PDF/SVG we embed that same snapshot, so the export matches the panel
        rather than retypesetting the figure.
        """
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure the visible canvas is fully up to date.
        self.canvas.draw()
        self.canvas.repaint()
        QApplication.processEvents()

        # Grab the exact rendered canvas contents.
        width, height = self.canvas.get_width_height()
        if width <= 0 or height <= 0:
            raise RuntimeError("Canvas has invalid size; cannot export current panel view.")
        rgba = np.asarray(self.canvas.buffer_rgba()).copy()
        suffix = path.suffix.lower()

        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
            try:
                from PIL import Image
            except Exception as exc:
                raise RuntimeError(
                    "Pillow is required for exact-panel raster export. Install with: pip install pillow"
                ) from exc
            Image.fromarray(rgba).save(str(path))
        else:
            # For vector-ish outputs like PDF/SVG, preserve the exact live appearance
            # by placing the captured raster snapshot into a clean one-image figure.
            from matplotlib.figure import Figure as _Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            snap_dpi = float(dpi) if dpi and dpi > 0 else 100.0
            snap_fig = _Figure(figsize=(width / snap_dpi, height / snap_dpi), dpi=snap_dpi, frameon=False)
            snap_canvas = FigureCanvasAgg(snap_fig)
            ax = snap_fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.imshow(rgba)
            ax.set_axis_off()
            # IMPORTANT: call the canvas-level print_figure to bypass any savefig monkey-patch.
            snap_canvas.print_figure(str(path), dpi=snap_dpi, facecolor='white', edgecolor='white')

        if not path.exists() or path.stat().st_size == 0:
            raise OSError(f"Matplotlib did not create a non-empty file: {path}")
        return path

    def export_figure(self):
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export IFO optical quadrature figure",
            "ifo_optical_quadratures.pdf",
            "PDF (*.pdf);;PNG (*.png);;SVG (*.svg);;All files (*)",
        )
        if not path:
            return

        # Add an extension if the user typed a bare filename.
        p = Path(path).expanduser()
        if p.suffix == "":
            if "PNG" in selected_filter:
                p = p.with_suffix(".png")
            elif "SVG" in selected_filter:
                p = p.with_suffix(".svg")
            else:
                p = p.with_suffix(".pdf")

        try:
            saved = self._save_live_figure(p, dpi=300)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", f"{type(exc).__name__}: {exc}")
            print(f"[export failed] {type(exc).__name__}: {exc}", flush=True)
            return

        print(f"[export] Saved live figure to: {saved}", flush=True)
        QMessageBox.information(self, "Export complete", f"Saved:\n{saved}")

    def save_defaults_here(self):
        try:
            s = self.gather_settings()
            # Do not rebuild the model here either. Save the exact live figure.
            prefix = "ifo_optical_quadratures"
            saved_paths = []
            for ext in ["pdf", "png", "svg"]:
                saved_paths.append(self._save_live_figure(Path(f"{prefix}.{ext}"), dpi=300))

            # Reuse the current settings for the text summary.  The per-frequency
            # numerical diagnostics are already printed on every redraw; saving the
            # figure must never trigger a second GWINC calculation.
            lines = [
                f"YAML: {self.yaml_path}",
                f"Input state: {s['input_state']}",
                f"Frequencies text: {s['freqs_text']}",
                f"How many plotted: {s['num_freqs']}",
                f"IFO model: {s['ifo_model']}",
                f"GWINC arm power [W]: {s['gwinc_arm_power_W']}",
                f"GWINC arm length [m]: {s['gwinc_arm_length_m']}",
                f"GWINC SEC detuning [rad]: {s['gwinc_sec_detuning_rad']}",
                f"IFO-OMC mismatch: {s['gwinc_mm_ifo_omc']}",
                f"IFO-OMC phase [rad]: {s['gwinc_mm_ifo_omc_phase_rad']}",
                f"SQZ-OMC mismatch: {s['gwinc_mm_sqz_omc']}",
                f"SQZ-OMC phase [rad]: {s['gwinc_mm_sqz_omc_phase_rad']}",
                f"IFO-SEC mismatch: {s['gwinc_mm_ifo_sec']}",
                f"IFO-SEC phase [rad]: {s['gwinc_mm_ifo_sec_phase_rad']}",
                f"Show input panel: {s['show_input_panel']}",
                f"Show output panel: {s['show_output_panel']}",
                f"Frequency label placement: {s['freq_label_mode']}",
                f"Input set-title gap: {s['input_set_title_gap']}",
                f"Output set-title gap: {s['output_set_title_gap']}",
                f"Input x limits: [{s['input_xmin']}, {s['input_xmax']}]",
                f"Input y limits: [{s['input_ymin']}, {s['input_ymax']}]",
                f"Output x limits: [{s['output_xmin']}, {s['output_xmax']}]",
                f"Output y limits: [{s['output_ymin']}, {s['output_ymax']}]",
                f"Horizontal spacing (wspace): {s['wspace']}",
                f"Vertical spacing (hspace): {s['hspace']}",
                "",
                "Per-frequency summary:",
            ]
            lines.append("Per-frequency numerical diagnostics are printed in the console on Redraw.")
            Path(f"{prefix}_summary.txt").write_text("\n".join(lines))
            QMessageBox.information(
                self,
                "Saved",
                "Saved the currently displayed figure as PDF/PNG/SVG plus summary txt."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            raise


def main():
    ap = argparse.ArgumentParser(
        description="Visualize squeezed-state quadrature ellipses before and after GWINC IFO propagation."
    )
    ap.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="Path to a GWINC YAML model. If omitted, a file picker opens.",
    )
    args = ap.parse_args()

    if not QT_AVAILABLE:
        raise RuntimeError(
            "Qt/PyQt could not be imported. Install PyQt5. "
            f"Original import error: {QT_IMPORT_ERROR}"
        )
    if not GWINC_AVAILABLE:
        raise RuntimeError(
            "GWINC could not be imported. Run this script in the same environment as your "
            f"GWINC quantum-noise code. Original import error: {GWINC_IMPORT_ERROR}"
        )

    print("[ifo-optical-gui] Starting optical quadrature propagation GUI...", flush=True)

    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True

    if args.yaml:
        yaml_path = Path(args.yaml).expanduser().resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"Could not find YAML file: {yaml_path}")
    else:
        selected, _ = QFileDialog.getOpenFileName(
            None,
            "Select base GWINC YAML",
            str(Path.cwd()),
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not selected:
            return 0
        yaml_path = Path(selected).expanduser().resolve()

    win = IFOOpticalPropagationWindow(yaml_path)
    win.show()
    win.raise_()
    win.activateWindow()

    app._ifo_optical_gui = win
    globals()["_IFO_OPTICAL_GUI"] = win
    globals()["_IFO_OPTICAL_APP"] = app

    if created:
        return app.exec_()
    return win


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        sys.exit(result)
