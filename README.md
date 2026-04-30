# QFT/QST Split-Operator Notebook Project for 1D Quantum Dynamics

This project provides a reproducible notebook-based workflow for one-dimensional quantum dynamics on power-of-two qubit grids. It covers the harmonic oscillator with a periodic QFT/FFT split operator and the infinite well with the boundary-compatible quantum-sine-transform (QST) family, implemented numerically as an orthonormal DST-II and benchmarked against exact sine-eigenstate expansion.

## Folder Structure

```text
.
|-- .venv/
|-- figures/
|-- notebooks/
|   |-- 00_environment_setup.ipynb
|   |-- 00_common_qft_split_functions.ipynb
|   |-- 01_harmonic_oscillator_qft_split_operator.ipynb
|   |-- 02_infinite_well_qft_split_operator.ipynb
|   |-- 03_circuit_resource_analysis.ipynb
|   `-- 04_publication_exports.ipynb
|-- tables/
|-- README.md
`-- requirements.txt
```

## Windows PowerShell Setup

```powershell
cd "D:\CDAC Projects\Simulating_1D_Sytems_Trotterization\EPJ Plus paper\Codex"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name codex-qft-split --display-name codex-qft-split
```

The project already includes a local `.venv`. If you recreate it, install `requirements.txt` and register the kernel again.

## Notebook Execution Order

Run the notebooks in this order:

1. `notebooks/00_environment_setup.ipynb`
2. `notebooks/00_common_qft_split_functions.ipynb`
3. `notebooks/01_harmonic_oscillator_qft_split_operator.ipynb`
4. `notebooks/02_infinite_well_qft_split_operator.ipynb`
5. `notebooks/03_circuit_resource_analysis.ipynb`
6. `notebooks/04_publication_exports.ipynb`

From PowerShell:

```powershell
$env:JUPYTER_PATH = ".\.venv\share\jupyter"
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=codex-qft-split notebooks\00_environment_setup.ipynb notebooks\00_common_qft_split_functions.ipynb notebooks\01_harmonic_oscillator_qft_split_operator.ipynb notebooks\02_infinite_well_qft_split_operator.ipynb notebooks\03_circuit_resource_analysis.ipynb notebooks\04_publication_exports.ipynb
```

## Notebook Summary

- `00_environment_setup.ipynb`: verifies the Python executable, package versions, project root, common-methods notebook, and output folders.
- `00_common_qft_split_functions.ipynb`: stores the shared scientific, plotting, saving, validation, and circuit helper functions used by the workflow.
- `01_harmonic_oscillator_qft_split_operator.ipynb`: runs the harmonic-oscillator QFT split-operator simulation with `r=100`, exact eigenstate reference, fidelity diagnostics, spatial convergence, and publication figures.
- `02_infinite_well_qft_split_operator.ipynb`: runs the infinite-well hard-wall simulation with a sine-windowed Gaussian initial state, `r=100`, exact sine-eigenstate reference, fidelity diagnostics, and publication figures.
- `03_circuit_resource_analysis.ipynb`: builds compact and transpiled one-step circuits, extracts 1q/2q/depth metrics, and exports gate-count and fidelity-vs-gate-count plots.
- `04_publication_exports.ipynb`: verifies all required figures/tables, checks numerical/resource quality thresholds, and writes the final manifest/checklist.

## Infinite-Well Boundary Model

A standard periodic QFT kinetic step models a ring and does not enforce hard-wall Dirichlet boundaries. The infinite-well notebook therefore uses the boundary-compatible QST/DST-II route and records this explicitly in `tables/infinite_well_parameters.csv`. Because the ideal well has zero interior potential, the step-count sweep mainly changes circuit repetition cost; the fidelity-vs-gate-count curve is expected to be nearly flat and is reported honestly.

## Generated Outputs

The notebooks save polished PNG/PDF figures at 600 dpi into `figures/` and CSV tables into `tables/`. Outputs include probability-density snapshots, fidelity-vs-time curves, spatial-convergence plots, compact/transpiled circuit diagrams, 1q/2q gate-count plots, fidelity-vs-gate-count plots, resource tables, and a final publication manifest/checklist.
