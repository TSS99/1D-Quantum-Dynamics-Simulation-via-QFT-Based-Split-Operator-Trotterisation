# QFT/QST Split-Operator Notebook Project for 1D Quantum Dynamics

This project implements a publication-oriented workflow for one-dimensional quantum dynamics on qubit-compatible grids. It now separates the boundary models explicitly:

- The harmonic oscillator uses the standard periodic QFT/FFT split-operator method on a finite box, with boundary leakage and spatial convergence checks.
- The infinite well uses a Dirichlet sine-transform split step, i.e. the boundary-compatible Fourier/QFT-family transform for hard-wall systems. This replaces the earlier periodic-ring approximation and removes the periodic-vs-Dirichlet mismatch.

Exact references are built from analytical energy eigenstates with numerically integrated expansion coefficients. Circuit-resource analysis includes transform-convention validation and records the synthesis model used for every resource table.

## Folder Structure

```text
Codex/
|-- .venv/
|-- figures/
|-- notebooks/
|   |-- 00_environment_setup.ipynb
|   |-- 01_harmonic_oscillator_qft_split_operator.ipynb
|   |-- 02_infinite_well_qft_split_operator.ipynb
|   |-- 03_circuit_resource_analysis.ipynb
|   `-- 04_publication_exports.ipynb
|-- scripts/
|   `-- build_paper_ready_notebooks.py
|-- src/
|   `-- qftsplit/
|       |-- __init__.py
|       `-- core.py
|-- tables/
|-- README.md
`-- requirements.txt
```

## Create and Activate the Virtual Environment in Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name codex-qft-split --display-name codex-qft-split
```

This project already includes a local `.venv` in the project root. If you recreate it manually, install the frozen package set afterward and register the kernel as shown above.

## Notebook Execution Order

Run the notebooks in this order:

1. `notebooks/00_environment_setup.ipynb`
2. `notebooks/01_harmonic_oscillator_qft_split_operator.ipynb`
3. `notebooks/02_infinite_well_qft_split_operator.ipynb`
4. `notebooks/03_circuit_resource_analysis.ipynb`
5. `notebooks/04_publication_exports.ipynb`

Or execute the complete workflow from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=codex-qft-split notebooks\00_environment_setup.ipynb notebooks\01_harmonic_oscillator_qft_split_operator.ipynb notebooks\02_infinite_well_qft_split_operator.ipynb notebooks\03_circuit_resource_analysis.ipynb notebooks\04_publication_exports.ipynb
```

## Notebook Summary

- `00_environment_setup.ipynb`: verifies the Python executable, package versions, project root, shared source module, and output folders.
- `01_harmonic_oscillator_qft_split_operator.ipynb`: simulates harmonic-oscillator dynamics with a periodic QFT split operator, compares against the analytical eigenbasis reference, exports fidelity and spatial-convergence diagnostics, and checks boundary leakage.
- `02_infinite_well_qft_split_operator.ipynb`: simulates the infinite well with a Dirichlet sine-transform split operator and compares against the exact sine-basis reference.
- `03_circuit_resource_analysis.ipynb`: validates Qiskit's QFT convention against the NumPy FFT step, validates the sine-transform step, records resource counts/estimates, and exports circuit/resource figures.
- `04_publication_exports.ipynb`: verifies all required figures/tables, regenerates missing exports, checks numerical quality thresholds, and writes the final manifest/checklist.

## Boundary Model Fix

The earlier infinite-well workflow used a periodic FFT/QFT kinetic step while comparing to a Dirichlet hard-wall reference. That is physically a ring-vs-box mismatch. The corrected workflow uses an orthonormal DST-II sine transform on a midpoint grid for the infinite well. The generated `tables/infinite_well_parameters.csv` records `periodic_boundary_mismatch=False` and `boundary_model=Dirichlet_hard_wall`.

## Generated Figures and Tables

The notebooks save publication-style PNG/PDF figures into `figures/` and CSV tables into `tables/`. Outputs include density snapshots, fidelity-vs-time curves, Trotter-step sweeps, spatial-convergence tables, circuit validation tables, resource-count tables, fidelity-vs-gate-count plots, and a final publication manifest/checklist.

The current executed workflow passes the final checklist in `tables/final_checklist.csv`.
