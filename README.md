# LIGO Quantum Tools

A small collection of interactive visualization and figure-building tools developed for studying squeezed-light quantum noise in LIGO-like precision measurements.

Together, the tools follow the squeezed state from preparation at the input, through interferometer propagation, to practical degradation from loss and phase noise. Parameters can be changed interactively, intermediate quantities can be visualized, and publication-style figures can be generated directly from the same models used in the analysis.

Each tool lives in its own directory and has a dedicated README describing its equations, assumptions, controls, and dependencies.

## Tools

### `squeezing-ellipse-visualizer`

Interactive GUI for visualizing the **input squeezed quantum state** before interferometer propagation.

The tool compares vacuum, frequency-independent squeezing (FIS), and frequency-dependent squeezing (FDS) as quadrature covariance ellipses. It applies the selected squeezing level and angle, optional injection loss, and—in the FDS case—the frequency-dependent rotation produced by the filter-cavity model.

The model is intentionally limited to preparation of the injected state. It does **not** include interferometer propagation, OMC/readout loss, phase noise, or spatial mode mismatch. This makes it useful for isolating the physics of squeezed-state preparation before introducing the detector response.

See [`squeezing-ellipse-visualizer/README.md`](squeezing-ellipse-visualizer/README.md) for the equations and assumptions used.

---

### `output-qstate-ellipse-visualizer`

Interactive GUI for visualizing optical quadrature covariance ellipses before and after propagation through a GWINC interferometer model.

The input state can be vacuum, frequency-independent squeezing (FIS), or frequency-dependent squeezing (FDS). The GUI can use the filter-cavity parameters from a GWINC YAML model to construct the input state, then propagate that state through the interferometer using GWINC's quantum-noise matrices.

The interface includes controls for squeezing and filter-cavity parameters, frequency selection, arm power and arm length, signal-extraction-cavity detuning, IFO–OMC / SQZ–OMC / IFO–SEC mode mismatch, plot layout, axis limits, labels, colors, legends, and exact-view figure export.

When spatial mode mismatch is enabled, the calculation uses GWINC's higher-order-mode representation and projects the propagated covariance back onto the detected fundamental mode using GWINC's local-oscillator vectors.

See [`output-qstate-ellipse-visualizer/README.md`](output-qstate-ellipse-visualizer/README.md) for the full model description.

---

### `loss-phase-noise-visualizer`

Interactive GUI illustrating how optical loss and phase noise limit observed squeezing.

The upper panels provide schematic quadrature-space views of vacuum admixture and readout-axis jitter. The lower panels calculate the measured variance as a function of nonlinear gain for selectable optical losses and phase-noise levels.

The model includes ideal OPO squeezing and anti-squeezing versus nonlinear gain, scalar optical loss modeled as vacuum admixture, Gaussian RMS phase noise, covariance-ellipse construction, independent legend and layout controls, and PDF/PNG/SVG export.

This tool is deliberately compact and does not include interferometer propagation or filter-cavity dynamics.

See [`loss-phase-noise-visualizer/README.md`](loss-phase-noise-visualizer/README.md) for the equations and assumptions used.

## Repository layout

```text
LIGO-quantum-tools/
├── README.md
├── squeezing-ellipse-visualizer/
│   ├── README.md
│   ├── squeezing_ellipse_visualizer.py
│   └── assets/
├── output-qstate-ellipse-visualizer/
│   ├── README.md
│   ├── ifo_optical_quadratures.py
│   ├── requirements.txt
│   └── ui.png
└── loss-phase-noise-visualizer/
    ├── README.md
    ├── loss_phase_noise_gui.py
    ├── requirements.txt
    └── ui.png
```

Detector-specific GWINC YAML files can be kept outside the repository and selected through the quadrature-propagation GUI.

## Requirements

Dependencies are listed separately for each tool.

The quadrature-propagation GUI requires a compatible GWINC installation because it uses internal functions from `gwinc.noise.quantum`.

The loss/phase-noise GUI only requires the standard Python packages listed in its own `requirements.txt`.

## Research context

These tools were developed for a **Contemporary Physics review article currently in preparation** on squeezed-light quantum measurement in LIGO.

The visualizations are intended to support an intuitive progression from the preparation of squeezed optical states, through their propagation in a realistic interferometer, to practical limitations from optical loss and phase noise.

Related experimental work:

**B. Kabagöz et al., _Observing and Evading Quantum Back-Action on a Kilogram-Scale Oscillator_**

[arXiv:2609.19317](https://arxiv.org/abs/2609.19317)

## Use of language models

OpenAI language models were used as programming assistants during development of parts of this repository, including code organization, debugging, interface iteration, and documentation.

All scientific model choices, equations, parameter definitions, physical interpretation, and validation of the calculations were performed by the author(s). The language model was not used as an independent source of scientific validation.

## Status

These are research tools rather than a general-purpose software package. Interfaces and internal implementation may change as the associated analysis develops.
