# 1D Quantum Dynamics via QFT/QST Split-Operator Trotterisation

This repository contains a reproducible, notebook-based workflow for simulating one-dimensional quantum dynamics on power-of-two qubit grids. The project covers two systems:

- A one-dimensional harmonic oscillator evolved with a periodic QFT/FFT split operator.
- A one-dimensional infinite potential well evolved with the boundary-compatible quantum-sine-transform (QST) family, implemented numerically as an orthonormal DST-II.

The workflow is designed for publication-style numerical experiments: exact eigenstate-expansion references, fidelity diagnostics, circuit diagrams, transpiled gate counts, CSV tables, and PNG/PDF figures are generated directly from the notebooks.

## Repository Layout

```text
.
|-- figures/                         # Generated PNG/PDF publication figures
|-- notebooks/
|   |-- 00_environment_setup.ipynb
|   |-- 00_common_qft_split_functions.ipynb
|   |-- 01_harmonic_oscillator_qft_split_operator.ipynb
|   |-- 02_infinite_well_qft_split_operator.ipynb
|   |-- 03_circuit_resource_analysis.ipynb
|   `-- 04_publication_exports.ipynb
|-- tables/                          # Generated CSV parameter, fidelity, and resource tables
|-- README.md
`-- requirements.txt
```

All executable scientific code is kept inside notebooks. Shared functions live in `notebooks/00_common_qft_split_functions.ipynb`, which is loaded by the physics and resource-analysis notebooks.

## Physical Method

The split-operator step uses the second-order symmetric form

$$
U_2(\Delta t) = e^{-iV\Delta t/(2\hbar)} F^{-1} e^{-iT\Delta t/\hbar} F e^{-iV\Delta t/(2\hbar)} ,
$$

where $F$ is the Fourier-family transform used by the grid representation. For the harmonic oscillator, $F$ is the periodic QFT/FFT transform. For the infinite well, $F$ is the Dirichlet sine-transform/QST representation compatible with hard-wall boundaries.

The exact reference solution is computed by eigenstate expansion,

$$
\psi_{\mathrm{ref}}(x,t) = \sum_n c_n \phi_n(x)\,e^{-iE_n t/\hbar}, \qquad c_n = \int \phi_n^*(x)\psi(x,0)\,dx .
$$

The reported fidelity is

$$
\mathcal{F}(t) = \left| \Delta x \sum_j \psi_{\mathrm{ref}}^*(x_j,t)\, \psi_{\mathrm{split}}(x_j,t) \right|^2 .
$$

## Harmonic Oscillator

The harmonic oscillator Hamiltonian is

$$
H = \frac{p^2}{2m} + \frac{1}{2}m\omega^2x^2 .
$$

The reference spectrum uses analytical Hermite-function eigenstates with

$$
E_n = \hbar\omega\left(n+\frac{1}{2}\right).
$$

Default simulation parameters are stored near the top of `01_harmonic_oscillator_qft_split_operator.ipynb`: $N=64$, $x\in[-8,8)$, $\hbar=m=\omega=1$, $t_{\max}=2\pi$, and `r = 100`.

## Infinite Potential Well

The infinite-well domain is $x\in(0,L)$ with Dirichlet boundary conditions. The initial state is a sine-windowed Gaussian,

$$
\psi(x,0) \propto \exp\left[-\frac{(x-x_0)^2}{4\sigma^2}\right] e^{ik_0(x-x_0)} \sin\left(\frac{\pi x}{L}\right),
$$

so that it is compatible with the hard-wall boundaries. The exact reference uses

$$
\phi_n(x)=\sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right),
\qquad
E_n=\frac{\hbar^2\pi^2n^2}{2mL^2}.
$$

Default simulation parameters are stored near the top of `02_infinite_well_qft_split_operator.ipynb`: $L=10$, $N=64$, $\hbar=m=1$, $t_{\max}=6$, and `r = 100`.

A periodic QFT represents periodic boundary conditions. The infinite-well notebook therefore uses the sine-transform/QST route, which is the Fourier-family transform matched to Dirichlet hard walls. Since the ideal well has zero interior potential, the step-count sweep mainly changes repeated-circuit resource cost; the fidelity curve is expected to remain nearly flat.

## Circuit and Resource Analysis

`03_circuit_resource_analysis.ipynb` builds a single logical split step for each system, renders compact and transpiled circuit diagrams, transpiles to rotational gates plus CX, and records:

- 1-qubit gate count
- 2-qubit gate count
- circuit depth
- gate breakdown by operation name
- total resource estimates over selected Trotter-step counts

The resource tables are combined with the fidelity sweeps to generate fidelity-vs-gate-count plots.

## Setup on Windows PowerShell

Create and activate a local virtual environment:

```powershell
cd "choose the directory path"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name codex-qft-split --display-name codex-qft-split
```

If `.venv` is already present locally, activate it and install from `requirements.txt` if needed.

## Notebook Execution Order

Run the notebooks in this order:

1. `notebooks/00_environment_setup.ipynb`
2. `notebooks/00_common_qft_split_functions.ipynb`
3. `notebooks/01_harmonic_oscillator_qft_split_operator.ipynb`
4. `notebooks/02_infinite_well_qft_split_operator.ipynb`
5. `notebooks/03_circuit_resource_analysis.ipynb`
6. `notebooks/04_publication_exports.ipynb`

To execute the full workflow from PowerShell:

```powershell
$env:JUPYTER_PATH = ".\.venv\share\jupyter"
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=codex-qft-split notebooks\00_environment_setup.ipynb notebooks\00_common_qft_split_functions.ipynb notebooks\01_harmonic_oscillator_qft_split_operator.ipynb notebooks\02_infinite_well_qft_split_operator.ipynb notebooks\03_circuit_resource_analysis.ipynb notebooks\04_publication_exports.ipynb
```

## Generated Outputs

The notebooks write publication-ready outputs automatically:

- `figures/`: 600 dpi PNG and PDF figures for probability snapshots, fidelity curves, spatial convergence, gate counts, fidelity-vs-gate-count plots, and circuit diagrams.
- `tables/`: CSV files for parameters, fidelity time series, Trotter-step sweeps, spatial convergence, circuit validation, gate breakdowns, resource totals, publication manifest, and final checklist.

The final verification state is recorded in `tables/final_checklist.csv`.
