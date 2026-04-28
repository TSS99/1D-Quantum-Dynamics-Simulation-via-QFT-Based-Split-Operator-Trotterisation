from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def write_notebook(name: str, cells: list[dict]) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "codex-qft-split",
            "language": "python",
            "name": "codex-qft-split",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    nbf.write(nb, NOTEBOOKS / name)


def md(source: str) -> dict:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> dict:
    return nbf.v4.new_code_cell(dedent(source).strip())


COMMON_IMPORTS = r"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "notebooks").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qftsplit.core import (
    build_infinite_well_circuit,
    build_periodic_resource_circuit,
    configure_matplotlib,
    dirichlet_midpoint_grid,
    draw_and_save_circuit,
    fft_momentum_grid,
    fidelity_summary,
    harmonic_potential,
    periodic_grid,
    plot_convergence,
    plot_density_snapshots,
    plot_fidelity_vs_gate_count,
    plot_fidelity,
    qft_split_circuit_validation,
    run_harmonic_case,
    run_infinite_well_case,
    save_dataframe,
    save_publication_figure,
    sine_mode_energies,
    sine_transform_validation,
    transpile_and_extract_metrics,
)

FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
configure_matplotlib()
"""


write_notebook(
    "00_environment_setup.ipynb",
    [
        md(
            """
            # 00. Environment Setup

            This notebook verifies the local environment, the shared `src/qftsplit` implementation, and the output folders used by the publication workflow.
            """
        ),
        code(
            r"""
            from pathlib import Path
            import importlib.metadata as ilmd
            import platform
            import sys

            import pandas as pd
            from IPython.display import Markdown, display

            EXPECTED_ROOT = Path(r"D:\CDAC Projects\Simulating_1D_Sytems_Trotterization\EPJ Plus paper\Codex").resolve()
            PROJECT_ROOT = Path.cwd().resolve()
            if not (PROJECT_ROOT / "notebooks").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent.resolve()
            if PROJECT_ROOT != EXPECTED_ROOT:
                raise RuntimeError(f"Project root mismatch. Expected {EXPECTED_ROOT}, found {PROJECT_ROOT}.")

            SRC_DIR = PROJECT_ROOT / "src"
            if str(SRC_DIR) not in sys.path:
                sys.path.insert(0, str(SRC_DIR))

            from qftsplit.core import configure_matplotlib, dirichlet_midpoint_grid, periodic_grid

            FIGURES_DIR = PROJECT_ROOT / "figures"
            TABLES_DIR = PROJECT_ROOT / "tables"
            for directory in (FIGURES_DIR, TABLES_DIR):
                directory.mkdir(parents=True, exist_ok=True)

            packages = ["qiskit", "numpy", "scipy", "matplotlib", "pandas", "jupyter", "ipykernel", "pylatexenc"]
            version_df = pd.DataFrame([{"package": package, "version": ilmd.version(package)} for package in packages])
            summary_df = pd.DataFrame(
                [
                    {
                        "python_executable": sys.executable,
                        "python_version": sys.version.split()[0],
                        "platform": platform.platform(),
                        "project_root": str(PROJECT_ROOT),
                        "source_module": str(SRC_DIR / "qftsplit" / "core.py"),
                        "figures_dir": str(FIGURES_DIR),
                        "tables_dir": str(TABLES_DIR),
                    }
                ]
            )

            display(Markdown("## Environment Summary"))
            display(summary_df)
            display(Markdown("## Package Versions"))
            display(version_df)

            # Smoke-test both supported spectral grids.
            periodic_grid(-8.0, 8.0, 64)
            dirichlet_midpoint_grid(10.0, 64)
            configure_matplotlib()

            print("Environment setup check completed successfully.")
            """
        ),
        md(
            """
            The remaining notebooks import the shared implementation from `src/qftsplit/core.py`. Generated figures are written to `figures/`, and generated tables are written to `tables/`.
            """
        ),
    ],
)


write_notebook(
    "01_harmonic_oscillator_qft_split_operator.ipynb",
    [
        md(
            r"""
            # 01. Harmonic Oscillator via QFT Split-Operator Evolution

            This notebook uses the standard periodic QFT/FFT split-operator method for a harmonic oscillator on a finite box. The exact reference is reconstructed from the analytical harmonic-oscillator eigenbasis. The finite periodic box is treated as a numerical approximation and checked through boundary and spatial-convergence diagnostics.
            """
        ),
        code(COMMON_IMPORTS),
        code(
            r"""
            # Main parameters
            N = 64
            x_min = -8.0
            x_max = 8.0
            hbar = 1.0
            m = 1.0
            omega = 1.0

            x0 = 2.0
            k0 = 0.0
            sigma = 1.0

            t_max = 2.0 * np.pi
            r = 200
            snapshot_count = 5

            reference_grid_size = 4096
            reference_tail_tolerance = 1e-10
            reference_basis_cap_min = 96

            fidelity_sweep_r = [50, 100, 150, 200, 250, 300, 400]
            spatial_convergence_N = [32, 64, 128]
            """
        ),
        code(
            r"""
            result = run_harmonic_case(
                grid_size=N,
                x_left=x_min,
                x_right=x_max,
                x0=x0,
                k0=k0,
                sigma=sigma,
                t_max=t_max,
                steps=r,
                mass=m,
                omega=omega,
                hbar=hbar,
                reference_grid_size=reference_grid_size,
                reference_tail_tolerance=reference_tail_tolerance,
                reference_basis_cap=max(reference_basis_cap_min, 2 * N),
            )
            summary = fidelity_summary(result)

            display(Markdown("## Main Run Summary"))
            display(pd.DataFrame([summary]))
            display(
                Markdown(
                    f"Reference basis kept **{result['reference_data']['n_keep']}** states out of cap "
                    f"**{result['reference_data']['basis_cap']}**, with discarded tail estimate "
                    f"**{result['reference_data']['tail_weight']:.2e}**."
                )
            )
            """
        ),
        code(
            r"""
            parameter_df = pd.DataFrame(
                [
                    {
                        "system": "harmonic_oscillator",
                        "numerical_transform": "periodic_QFT_or_FFT",
                        "boundary_model": "finite_periodic_box_checked_by_edge_probability",
                        "N": N,
                        "n_qubits": int(np.log2(N)),
                        "domain": f"[{x_min}, {x_max})",
                        "x0": x0,
                        "k0": k0,
                        "sigma": sigma,
                        "t_max": t_max,
                        "r": r,
                        "dt": result["dt"],
                        "hbar": hbar,
                        "m": m,
                        "omega": omega,
                        "reference_grid_size": result["reference_data"]["dense_grid_size"],
                        "reference_basis_cap": result["reference_data"]["basis_cap"],
                        "reference_basis_kept": result["reference_data"]["n_keep"],
                        "reference_tail_weight": result["reference_data"]["tail_weight"],
                        "max_edge_probability": result["max_edge_probability"],
                        "fidelity_sweep_r": ",".join(str(value) for value in fidelity_sweep_r),
                        "spatial_convergence_N": ",".join(str(value) for value in spatial_convergence_N),
                    }
                ]
            )
            fidelity_df = pd.DataFrame(
                {
                    "time": result["times"],
                    "fidelity": result["fidelity"],
                    "split_norm": result["split_norms"],
                    "reference_norm": result["reference_norms"],
                }
            )

            save_dataframe(parameter_df, TABLES_DIR, "harmonic_parameters.csv")
            save_dataframe(fidelity_df, TABLES_DIR, "harmonic_fidelity_vs_time.csv")

            display(Markdown("## Simulation Parameters"))
            display(parameter_df)
            """
        ),
        code(
            r"""
            snapshot_figure = plot_density_snapshots(
                x=result["x"],
                times=result["times"],
                reference_states=result["reference_states"],
                split_states=result["split_states"],
                snapshot_count=snapshot_count,
                title="Harmonic oscillator probability density",
            )
            fidelity_figure = plot_fidelity(result["times"], result["fidelity"], "Harmonic oscillator fidelity")

            save_publication_figure(snapshot_figure, FIGURES_DIR, "harmonic_density_snapshots")
            save_publication_figure(fidelity_figure, FIGURES_DIR, "harmonic_fidelity_vs_time")
            plt.close(snapshot_figure)
            plt.close(fidelity_figure)

            print("Saved harmonic density and fidelity figures.")
            """
        ),
        code(
            r"""
            sweep_records = []
            for r_value in fidelity_sweep_r:
                local = run_harmonic_case(
                    grid_size=N,
                    x_left=x_min,
                    x_right=x_max,
                    x0=x0,
                    k0=k0,
                    sigma=sigma,
                    t_max=t_max,
                    steps=int(r_value),
                    mass=m,
                    omega=omega,
                    hbar=hbar,
                    reference_grid_size=reference_grid_size,
                    reference_tail_tolerance=reference_tail_tolerance,
                    reference_basis_cap=max(reference_basis_cap_min, 2 * N),
                )
                local_summary = fidelity_summary(local)
                sweep_records.append(
                    {
                        "system": "harmonic_oscillator",
                        "r": int(r_value),
                        "dt": local["dt"],
                        "final_time_fidelity": local_summary["final_time_fidelity"],
                        "minimum_fidelity": local_summary["minimum_fidelity"],
                        "mean_fidelity": local_summary["mean_fidelity"],
                    }
                )

            fidelity_sweep_df = pd.DataFrame(sweep_records)
            save_dataframe(fidelity_sweep_df, TABLES_DIR, "harmonic_fidelity_vs_r.csv")
            display(Markdown("## Trotter-Step Convergence"))
            display(fidelity_sweep_df)
            """
        ),
        code(
            r"""
            convergence_records = []
            for n_value in spatial_convergence_N:
                local = run_harmonic_case(
                    grid_size=int(n_value),
                    x_left=x_min,
                    x_right=x_max,
                    x0=x0,
                    k0=k0,
                    sigma=sigma,
                    t_max=t_max,
                    steps=r,
                    mass=m,
                    omega=omega,
                    hbar=hbar,
                    reference_grid_size=reference_grid_size,
                    reference_tail_tolerance=reference_tail_tolerance,
                    reference_basis_cap=max(reference_basis_cap_min, 2 * int(n_value)),
                )
                local_summary = fidelity_summary(local)
                convergence_records.append(
                    {
                        "system": "harmonic_oscillator",
                        "N": int(n_value),
                        "n_qubits": int(np.log2(n_value)),
                        "dx": local["dx"],
                        "final_time_fidelity": local_summary["final_time_fidelity"],
                        "minimum_fidelity": local_summary["minimum_fidelity"],
                        "max_edge_probability": local_summary["max_edge_probability"],
                    }
                )

            spatial_convergence_df = pd.DataFrame(convergence_records)
            save_dataframe(spatial_convergence_df, TABLES_DIR, "harmonic_spatial_convergence.csv")
            convergence_figure = plot_convergence(
                spatial_convergence_df,
                x_column="N",
                y_column="final_time_fidelity",
                title="Harmonic oscillator spatial convergence",
                xlabel="Grid points, N",
            )
            save_publication_figure(convergence_figure, FIGURES_DIR, "harmonic_spatial_convergence")
            plt.close(convergence_figure)

            display(Markdown("## Spatial Convergence"))
            display(spatial_convergence_df)
            """
        ),
    ],
)


write_notebook(
    "02_infinite_well_qft_split_operator.ipynb",
    [
        md(
            r"""
            # 02. Infinite Well via Dirichlet Sine-Transform Split Evolution

            The previous periodic-QFT treatment evolved a free particle on a ring and compared it with a hard-wall sine-basis reference. This corrected notebook uses a Dirichlet sine transform, the boundary-compatible Fourier/QFT-family transform for the infinite well. The numerical evolution and analytical reference now use the same hard-wall boundary condition.
            """
        ),
        code(COMMON_IMPORTS),
        code(
            r"""
            L = 10.0
            N = 64
            hbar = 1.0
            m = 1.0

            x0 = L / 2.0
            k0 = 2.0
            sigma = 0.8

            t_max = 6.0
            r = 700
            snapshot_count = 6

            reference_grid_size = 4097
            reference_tail_tolerance = 1e-10
            reference_basis_cap_min = 192

            fidelity_sweep_r = [100, 200, 300, 400, 500, 600, 700]
            spatial_convergence_N = [32, 64, 128]
            """
        ),
        code(
            r"""
            result = run_infinite_well_case(
                grid_size=N,
                length=L,
                x0=x0,
                k0=k0,
                sigma=sigma,
                t_max=t_max,
                steps=r,
                mass=m,
                hbar=hbar,
                reference_grid_size=reference_grid_size,
                reference_tail_tolerance=reference_tail_tolerance,
                reference_basis_cap=max(reference_basis_cap_min, 4 * N),
            )
            summary = fidelity_summary(result)

            display(Markdown("## Main Run Summary"))
            display(pd.DataFrame([summary]))
            display(
                Markdown(
                    f"Reference basis kept **{result['reference_data']['n_keep']}** sine states out of cap "
                    f"**{result['reference_data']['basis_cap']}**, with discarded tail estimate "
                    f"**{result['reference_data']['tail_weight']:.2e}**."
                )
            )
            """
        ),
        code(
            r"""
            parameter_df = pd.DataFrame(
                [
                    {
                        "system": "infinite_well",
                        "numerical_transform": "orthonormal_DST-II_quantum_sine_transform_family",
                        "boundary_model": "Dirichlet_hard_wall",
                        "periodic_boundary_mismatch": False,
                        "N": N,
                        "n_qubits": int(np.log2(N)),
                        "domain": f"(0, {L}) midpoint grid",
                        "L": L,
                        "x0": x0,
                        "k0": k0,
                        "sigma": sigma,
                        "t_max": t_max,
                        "r": r,
                        "dt": result["dt"],
                        "hbar": hbar,
                        "m": m,
                        "reference_grid_size": result["reference_data"]["dense_grid_size"],
                        "reference_basis_cap": result["reference_data"]["basis_cap"],
                        "reference_basis_kept": result["reference_data"]["n_keep"],
                        "reference_tail_weight": result["reference_data"]["tail_weight"],
                        "initial_state_modifier": "sin(pi x / L)",
                        "boundary_caveat": "resolved_by_Dirichlet_sine_transform_not_periodic_FFT",
                        "fidelity_sweep_r": ",".join(str(value) for value in fidelity_sweep_r),
                        "spatial_convergence_N": ",".join(str(value) for value in spatial_convergence_N),
                    }
                ]
            )
            fidelity_df = pd.DataFrame(
                {
                    "time": result["times"],
                    "fidelity": result["fidelity"],
                    "split_norm": result["split_norms"],
                    "reference_norm": result["reference_norms"],
                }
            )

            save_dataframe(parameter_df, TABLES_DIR, "infinite_well_parameters.csv")
            save_dataframe(fidelity_df, TABLES_DIR, "infinite_well_fidelity_vs_time.csv")

            display(Markdown("## Simulation Parameters"))
            display(parameter_df)
            """
        ),
        code(
            r"""
            snapshot_figure = plot_density_snapshots(
                x=result["x"],
                times=result["times"],
                reference_states=result["reference_states"],
                split_states=result["split_states"],
                snapshot_count=snapshot_count,
                title="Infinite-well probability density",
            )
            fidelity_figure = plot_fidelity(result["times"], result["fidelity"], "Infinite-well fidelity")

            for stale_path in list(FIGURES_DIR.glob("infinite_well_density_snapshots_page_*.png")) + list(FIGURES_DIR.glob("infinite_well_density_snapshots_page_*.pdf")):
                stale_path.unlink()

            save_publication_figure(snapshot_figure, FIGURES_DIR, "infinite_well_density_snapshots")
            save_publication_figure(fidelity_figure, FIGURES_DIR, "infinite_well_fidelity_vs_time")
            plt.close(snapshot_figure)
            plt.close(fidelity_figure)

            print("Saved infinite-well density and fidelity figures.")
            """
        ),
        code(
            r"""
            sweep_records = []
            for r_value in fidelity_sweep_r:
                local = run_infinite_well_case(
                    grid_size=N,
                    length=L,
                    x0=x0,
                    k0=k0,
                    sigma=sigma,
                    t_max=t_max,
                    steps=int(r_value),
                    mass=m,
                    hbar=hbar,
                    reference_grid_size=reference_grid_size,
                    reference_tail_tolerance=reference_tail_tolerance,
                    reference_basis_cap=max(reference_basis_cap_min, 4 * N),
                )
                local_summary = fidelity_summary(local)
                sweep_records.append(
                    {
                        "system": "infinite_well",
                        "r": int(r_value),
                        "dt": local["dt"],
                        "final_time_fidelity": local_summary["final_time_fidelity"],
                        "minimum_fidelity": local_summary["minimum_fidelity"],
                        "mean_fidelity": local_summary["mean_fidelity"],
                    }
                )

            fidelity_sweep_df = pd.DataFrame(sweep_records)
            save_dataframe(fidelity_sweep_df, TABLES_DIR, "infinite_well_fidelity_vs_r.csv")
            display(Markdown("## Trotter-Step Convergence"))
            display(fidelity_sweep_df)
            """
        ),
        code(
            r"""
            convergence_records = []
            for n_value in spatial_convergence_N:
                local = run_infinite_well_case(
                    grid_size=int(n_value),
                    length=L,
                    x0=x0,
                    k0=k0,
                    sigma=sigma,
                    t_max=t_max,
                    steps=r,
                    mass=m,
                    hbar=hbar,
                    reference_grid_size=reference_grid_size,
                    reference_tail_tolerance=reference_tail_tolerance,
                    reference_basis_cap=max(reference_basis_cap_min, 4 * int(n_value)),
                )
                local_summary = fidelity_summary(local)
                convergence_records.append(
                    {
                        "system": "infinite_well",
                        "N": int(n_value),
                        "n_qubits": int(np.log2(n_value)),
                        "dx": local["dx"],
                        "final_time_fidelity": local_summary["final_time_fidelity"],
                        "minimum_fidelity": local_summary["minimum_fidelity"],
                    }
                )

            spatial_convergence_df = pd.DataFrame(convergence_records)
            save_dataframe(spatial_convergence_df, TABLES_DIR, "infinite_well_spatial_convergence.csv")
            convergence_figure = plot_convergence(
                spatial_convergence_df,
                x_column="N",
                y_column="final_time_fidelity",
                title="Infinite-well spatial convergence",
                xlabel="Grid points, N",
            )
            save_publication_figure(convergence_figure, FIGURES_DIR, "infinite_well_spatial_convergence")
            plt.close(convergence_figure)

            display(Markdown("## Spatial Convergence"))
            display(spatial_convergence_df)
            """
        ),
    ],
)


write_notebook(
    "03_circuit_resource_analysis.ipynb",
    [
        md(
            r"""
            # 03. Circuit Resource Analysis and Transform Validation

            This notebook separates two claims: (1) the Qiskit QFT convention matches the NumPy FFT split step used for the harmonic oscillator, and (2) resource counts are small-system exact-synthesis or analytical estimates, not asymptotic optimal decompositions. The infinite-well resource model is reported for a quantum-sine-transform implementation rather than a periodic QFT circuit.
            """
        ),
        code(COMMON_IMPORTS),
        code(
            r"""
            N = 64
            hbar = 1.0
            m = 1.0
            omega = 1.0

            harmonic_domain = (-8.0, 8.0)
            harmonic_t_max = 2.0 * np.pi
            harmonic_r_default = 200

            well_L = 10.0
            well_t_max = 6.0
            well_r_default = 700

            basis_gates = ["rx", "ry", "rz", "cx"]
            """
        ),
        code(
            r"""
            validation_records = []

            x_h, dx_h = periodic_grid(harmonic_domain[0], harmonic_domain[1], N)
            p_h = fft_momentum_grid(N, dx=dx_h, hbar=hbar)
            dt_h = harmonic_t_max / harmonic_r_default
            position_phase_h = np.exp(-0.5j * harmonic_potential(x_h, mass=m, omega=omega) * dt_h / hbar)
            momentum_phase_h = np.exp(-1j * (p_h**2 / (2.0 * m)) * dt_h / hbar)
            validation_records.append(qft_split_circuit_validation(position_phase_h, momentum_phase_h))

            dt_w = well_t_max / well_r_default
            validation_records.append(sine_transform_validation(N, length=well_L, mass=m, hbar=hbar, dt=dt_w))

            validation_df = pd.DataFrame(validation_records)
            save_dataframe(validation_df, TABLES_DIR, "circuit_validation.csv")
            display(Markdown("## Transform Validation"))
            display(validation_df)

            if not validation_df["passed"].all():
                raise RuntimeError("At least one transform validation failed.")
            """
        ),
        code(
            r"""
            harmonic_logical = build_periodic_resource_circuit(position_phase_h, momentum_phase_h)
            harmonic_transpiled, harmonic_metrics, harmonic_breakdown = transpile_and_extract_metrics(harmonic_logical, basis_gates)

            n_qubits = int(np.log2(N))
            qst_extension_qubits = n_qubits + 2
            qft_controlled_phases = qst_extension_qubits * (qst_extension_qubits - 1) // 2
            kinetic_controlled_phases = n_qubits * (n_qubits - 1) // 2
            qst_blocks_per_step = 2
            total_controlled_phases = qst_blocks_per_step * qft_controlled_phases + kinetic_controlled_phases
            well_metrics = {
                "single_step_1q_count": int(qst_blocks_per_step * qst_extension_qubits + 3 * total_controlled_phases + n_qubits),
                "single_step_2q_count": int(2 * total_controlled_phases),
                "single_step_depth": int(qst_blocks_per_step * qst_extension_qubits**2 + 4 * kinetic_controlled_phases),
            }

            single_step_metrics_df = pd.DataFrame(
                [
                    {
                        "system": "harmonic_oscillator",
                        "N": N,
                        "n_qubits": n_qubits,
                        "dt": dt_h,
                        **harmonic_metrics,
                        "transform": "periodic_QFT",
                        "synthesis_model": "Qiskit_exact_generic_DiagonalGate_small_N_upper_bound",
                    },
                    {
                        "system": "infinite_well",
                        "N": N,
                        "n_qubits": n_qubits,
                        "dt": dt_w,
                        **well_metrics,
                        "transform": "Dirichlet_quantum_sine_transform",
                        "synthesis_model": "analytical_all_to_all_QST_via_QFT_extension_estimate",
                    },
                ]
            )

            gate_breakdown_records = [
                {"system": "harmonic_oscillator", "gate_name": gate_name, "count": gate_count}
                for gate_name, gate_count in sorted(harmonic_breakdown.items())
            ]
            gate_breakdown_records.extend(
                [
                    {"system": "infinite_well", "gate_name": "QST_blocks", "count": qst_blocks_per_step},
                    {"system": "infinite_well", "gate_name": "QFT_extension_controlled_phases_per_QST", "count": qft_controlled_phases},
                    {"system": "infinite_well", "gate_name": "kinetic_quadratic_phase_controlled_phases", "count": kinetic_controlled_phases},
                    {"system": "infinite_well", "gate_name": "estimated_cx", "count": well_metrics["single_step_2q_count"]},
                ]
            )
            gate_breakdown_df = pd.DataFrame(gate_breakdown_records)

            save_dataframe(single_step_metrics_df, TABLES_DIR, "circuit_single_step_metrics.csv")
            save_dataframe(gate_breakdown_df, TABLES_DIR, "circuit_gate_breakdown.csv")

            display(Markdown("## Single-Step Metrics"))
            display(single_step_metrics_df)
            display(Markdown("## Gate Breakdown"))
            display(gate_breakdown_df)
            """
        ),
        code(
            r"""
            draw_and_save_circuit(harmonic_logical, FIGURES_DIR, "harmonic_single_step_logical_circuit", scale=0.75, fold=40)
            draw_and_save_circuit(harmonic_transpiled, FIGURES_DIR, "harmonic_single_step_transpiled_circuit", scale=0.55, fold=50)
            
            infinite_well_logical = build_infinite_well_circuit(grid_size=N, length=well_L, mass=m, hbar=hbar, dt=dt_w)
            draw_and_save_circuit(infinite_well_logical, FIGURES_DIR, "infinite_well_single_step_logical_circuit_qiskit", scale=0.75, fold=40)
            
            infinite_well_transpiled, _, _ = transpile_and_extract_metrics(infinite_well_logical, basis_gates)
            draw_and_save_circuit(infinite_well_transpiled, FIGURES_DIR, "infinite_well_single_step_transpiled_circuit_qiskit", scale=0.55, fold=50)



            def save_block_diagram(stem: str, title: str, blocks: list[str]) -> None:
                fig, axis = plt.subplots(figsize=(10.0, 2.0), constrained_layout=True)
                axis.set_axis_off()
                
                box_style = {
                    "boxstyle": "round,pad=0.6",
                    "facecolor": "#ebf5fb", 
                    "edgecolor": "#2874a6", 
                    "linewidth": 2.0,
                }
                
                if len(blocks) == 3:
                    x_positions = np.linspace(0.25, 0.75, len(blocks))
                else:
                    x_positions = np.linspace(0.12, 0.88, len(blocks))
                
                for x_pos, label in zip(x_positions, blocks):
                    axis.text(
                        x_pos, 0.5, label,
                        ha="center", va="center",
                        fontsize=13, fontweight="bold", color="#1b4f72",
                        bbox=box_style,
                    )
                    
                for i in range(len(blocks) - 1):
                    left_text = blocks[i]
                    right_text = blocks[i+1]
                    
                    left_hw = max(0.04, len(left_text) * 0.005 + 0.03)
                    right_hw = max(0.04, len(right_text) * 0.005 + 0.03)
                    
                    axis.annotate(
                        "", 
                        xy=(x_positions[i+1] - right_hw, 0.5), 
                        xytext=(x_positions[i] + left_hw, 0.5), 
                        arrowprops={
                            "arrowstyle": "-|>,head_length=0.8,head_width=0.4", 
                            "lw": 2.5, 
                            "color": "#2874a6"
                        }
                    )
                    
                axis.set_title(title, fontsize=15, fontweight="bold", pad=15)
                save_publication_figure(fig, FIGURES_DIR, stem)
                plt.close(fig)

            save_block_diagram(
                "infinite_well_single_step_logical_circuit",
                "Infinite-well Dirichlet spectral step",
                ["QST", "T(dt)", "QST^-1"],
            )
            save_block_diagram(
                "infinite_well_single_step_transpiled_circuit",
                "Analytical QST resource model",
                ["QFT extension", "odd/sine symmetry", "quadratic phase", "inverse QST"],
            )

            print("Saved circuit and resource-model diagrams.")
            """
        ),
        code(
            r"""
            fidelity_tables = {
                "harmonic_oscillator": TABLES_DIR / "harmonic_fidelity_vs_r.csv",
                "infinite_well": TABLES_DIR / "infinite_well_fidelity_vs_r.csv",
            }

            resource_totals_frames = []
            for system_key, fidelity_path in fidelity_tables.items():
                if not fidelity_path.exists():
                    raise FileNotFoundError(f"Missing {fidelity_path.name}. Run the corresponding dynamics notebook first.")
                fidelity_df = pd.read_csv(fidelity_path)
                metrics_row = single_step_metrics_df.loc[single_step_metrics_df["system"] == system_key].iloc[0]
                totals_df = fidelity_df.copy()
                totals_df["single_step_1q_count"] = int(metrics_row["single_step_1q_count"])
                totals_df["single_step_2q_count"] = int(metrics_row["single_step_2q_count"])
                totals_df["single_step_depth"] = int(metrics_row["single_step_depth"])
                totals_df["synthesis_model"] = metrics_row["synthesis_model"]
                totals_df["total_1q_count"] = totals_df["single_step_1q_count"] * totals_df["r"]
                totals_df["total_2q_count"] = totals_df["single_step_2q_count"] * totals_df["r"]
                totals_df["total_gate_count"] = totals_df["total_1q_count"] + totals_df["total_2q_count"]
                totals_df["estimated_total_depth"] = totals_df["single_step_depth"] * totals_df["r"]
                resource_totals_frames.append(totals_df)
                save_dataframe(totals_df, TABLES_DIR, f"{system_key}_resource_totals_vs_r.csv")

            resource_totals_df = pd.concat(resource_totals_frames, ignore_index=True)
            save_dataframe(resource_totals_df, TABLES_DIR, "resource_totals_vs_r.csv")
            display(Markdown("## Resource Totals Across Fidelity Sweeps"))
            display(resource_totals_df)
            """
        ),
        code(
            r"""
            harmonic_resource_df = resource_totals_df.loc[resource_totals_df["system"] == "harmonic_oscillator"].reset_index(drop=True)
            well_resource_df = resource_totals_df.loc[resource_totals_df["system"] == "infinite_well"].reset_index(drop=True)

            figures = [
                (plot_convergence(harmonic_resource_df, "r", "total_2q_count", "Harmonic oscillator CX count", "Trotter steps, r"), "harmonic_gate_counts_vs_steps"),
                (plot_convergence(well_resource_df, "r", "total_2q_count", "Infinite-well estimated CX count", "Trotter steps, r"), "infinite_well_gate_counts_vs_steps"),
                (plot_fidelity_vs_gate_count(harmonic_resource_df, "total_2q_count", "Total CX count", "Harmonic fidelity vs CX count"), "harmonic_fidelity_vs_two_qubit_gate_count"),
                (plot_fidelity_vs_gate_count(harmonic_resource_df, "total_gate_count", "Total 1q + 2q gate count", "Harmonic fidelity vs total gate count"), "harmonic_fidelity_vs_total_gate_count"),
                (plot_fidelity_vs_gate_count(well_resource_df, "total_2q_count", "Estimated total CX count", "Infinite-well fidelity vs CX count"), "infinite_well_fidelity_vs_two_qubit_gate_count"),
                (plot_fidelity_vs_gate_count(well_resource_df, "total_gate_count", "Estimated total 1q + 2q gate count", "Infinite-well fidelity vs total gate count"), "infinite_well_fidelity_vs_total_gate_count"),
            ]
            for figure, stem in figures:
                save_publication_figure(figure, FIGURES_DIR, stem)
                plt.close(figure)

            print("Saved gate-count and fidelity-vs-gate-count figures.")
            """
        ),
    ],
)


write_notebook(
    "04_publication_exports.ipynb",
    [
        md(
            r"""
            # 04. Publication Exports and Final Checklist

            This notebook verifies that all required exports exist and that the numerical quality checks pass. Missing upstream exports are regenerated in notebook order.
            """
        ),
        code(
            r"""
            from pathlib import Path
            import subprocess
            import sys

            import pandas as pd
            from IPython.display import Markdown, display

            PROJECT_ROOT = Path.cwd().resolve()
            if not (PROJECT_ROOT / "notebooks").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent.resolve()
            NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
            FIGURES_DIR = PROJECT_ROOT / "figures"
            TABLES_DIR = PROJECT_ROOT / "tables"

            def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
                path = TABLES_DIR / filename
                df.to_csv(path, index=False)
                if not path.exists():
                    raise FileNotFoundError(f"Failed to save {path}.")
                return path

            def execute_notebook(notebook_name: str) -> None:
                notebook_path = NOTEBOOK_DIR / notebook_name
                if not notebook_path.exists():
                    raise FileNotFoundError(f"Missing notebook {notebook_path}.")
                command = [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    "--ExecutePreprocessor.timeout=3600",
                    "--ExecutePreprocessor.kernel_name=codex-qft-split",
                    str(notebook_path),
                ]
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            """
        ),
        code(
            r"""
            expected_outputs = [
                ("figures/harmonic_density_snapshots.png", "01_harmonic_oscillator_qft_split_operator.ipynb"),
                ("figures/harmonic_density_snapshots.pdf", "01_harmonic_oscillator_qft_split_operator.ipynb"),
                ("figures/harmonic_fidelity_vs_time.png", "01_harmonic_oscillator_qft_split_operator.ipynb"),
                ("figures/harmonic_fidelity_vs_time.pdf", "01_harmonic_oscillator_qft_split_operator.ipynb"),
                ("figures/harmonic_spatial_convergence.png", "01_harmonic_oscillator_qft_split_operator.ipynb"),
                ("figures/harmonic_spatial_convergence.pdf", "01_harmonic_oscillator_qft_split_operator.ipynb"),
                ("tables/harmonic_parameters.csv", "01_harmonic_oscillator_qft_split_operator.ipynb"),
                ("tables/harmonic_fidelity_vs_time.csv", "01_harmonic_oscillator_qft_split_operator.ipynb"),
                ("tables/harmonic_fidelity_vs_r.csv", "01_harmonic_oscillator_qft_split_operator.ipynb"),
                ("tables/harmonic_spatial_convergence.csv", "01_harmonic_oscillator_qft_split_operator.ipynb"),
                ("figures/infinite_well_density_snapshots.png", "02_infinite_well_qft_split_operator.ipynb"),
                ("figures/infinite_well_density_snapshots.pdf", "02_infinite_well_qft_split_operator.ipynb"),
                ("figures/infinite_well_fidelity_vs_time.png", "02_infinite_well_qft_split_operator.ipynb"),
                ("figures/infinite_well_fidelity_vs_time.pdf", "02_infinite_well_qft_split_operator.ipynb"),
                ("figures/infinite_well_spatial_convergence.png", "02_infinite_well_qft_split_operator.ipynb"),
                ("figures/infinite_well_spatial_convergence.pdf", "02_infinite_well_qft_split_operator.ipynb"),
                ("tables/infinite_well_parameters.csv", "02_infinite_well_qft_split_operator.ipynb"),
                ("tables/infinite_well_fidelity_vs_time.csv", "02_infinite_well_qft_split_operator.ipynb"),
                ("tables/infinite_well_fidelity_vs_r.csv", "02_infinite_well_qft_split_operator.ipynb"),
                ("tables/infinite_well_spatial_convergence.csv", "02_infinite_well_qft_split_operator.ipynb"),
                ("figures/harmonic_single_step_logical_circuit.png", "03_circuit_resource_analysis.ipynb"),
                ("figures/harmonic_single_step_logical_circuit.pdf", "03_circuit_resource_analysis.ipynb"),
                ("figures/harmonic_single_step_transpiled_circuit.png", "03_circuit_resource_analysis.ipynb"),
                ("figures/harmonic_single_step_transpiled_circuit.pdf", "03_circuit_resource_analysis.ipynb"),
                ("figures/infinite_well_single_step_logical_circuit.png", "03_circuit_resource_analysis.ipynb"),
                ("figures/infinite_well_single_step_logical_circuit.pdf", "03_circuit_resource_analysis.ipynb"),
                ("figures/infinite_well_single_step_transpiled_circuit.png", "03_circuit_resource_analysis.ipynb"),
                ("figures/infinite_well_single_step_transpiled_circuit.pdf", "03_circuit_resource_analysis.ipynb"),
                ("figures/harmonic_gate_counts_vs_steps.png", "03_circuit_resource_analysis.ipynb"),
                ("figures/harmonic_gate_counts_vs_steps.pdf", "03_circuit_resource_analysis.ipynb"),
                ("figures/infinite_well_gate_counts_vs_steps.png", "03_circuit_resource_analysis.ipynb"),
                ("figures/infinite_well_gate_counts_vs_steps.pdf", "03_circuit_resource_analysis.ipynb"),
                ("figures/harmonic_fidelity_vs_two_qubit_gate_count.png", "03_circuit_resource_analysis.ipynb"),
                ("figures/harmonic_fidelity_vs_two_qubit_gate_count.pdf", "03_circuit_resource_analysis.ipynb"),
                ("figures/harmonic_fidelity_vs_total_gate_count.png", "03_circuit_resource_analysis.ipynb"),
                ("figures/harmonic_fidelity_vs_total_gate_count.pdf", "03_circuit_resource_analysis.ipynb"),
                ("figures/infinite_well_fidelity_vs_two_qubit_gate_count.png", "03_circuit_resource_analysis.ipynb"),
                ("figures/infinite_well_fidelity_vs_two_qubit_gate_count.pdf", "03_circuit_resource_analysis.ipynb"),
                ("figures/infinite_well_fidelity_vs_total_gate_count.png", "03_circuit_resource_analysis.ipynb"),
                ("figures/infinite_well_fidelity_vs_total_gate_count.pdf", "03_circuit_resource_analysis.ipynb"),
                ("tables/circuit_validation.csv", "03_circuit_resource_analysis.ipynb"),
                ("tables/circuit_single_step_metrics.csv", "03_circuit_resource_analysis.ipynb"),
                ("tables/circuit_gate_breakdown.csv", "03_circuit_resource_analysis.ipynb"),
                ("tables/resource_totals_vs_r.csv", "03_circuit_resource_analysis.ipynb"),
            ]
            """
        ),
        code(
            r"""
            missing_by_notebook = {}
            for relative_path, source_notebook in expected_outputs:
                path = PROJECT_ROOT / relative_path
                if not path.exists():
                    missing_by_notebook.setdefault(source_notebook, []).append(path.name)

            if missing_by_notebook:
                display(Markdown("## Regenerating Missing Exports"))
                for notebook_name in [
                    "01_harmonic_oscillator_qft_split_operator.ipynb",
                    "02_infinite_well_qft_split_operator.ipynb",
                    "03_circuit_resource_analysis.ipynb",
                ]:
                    if notebook_name in missing_by_notebook:
                        print(f"Re-executing {notebook_name} because required exports were missing.")
                        execute_notebook(notebook_name)

            manifest_df = pd.DataFrame(
                [
                    {
                        "path": relative_path,
                        "exists": (PROJECT_ROOT / relative_path).exists(),
                        "source_notebook": source_notebook,
                    }
                    for relative_path, source_notebook in expected_outputs
                ]
            )
            if not manifest_df["exists"].all():
                raise FileNotFoundError("At least one required publication export is still missing after regeneration.")

            save_dataframe(manifest_df, "publication_manifest.csv")
            display(Markdown("## Publication Manifest"))
            display(manifest_df)
            """
        ),
        code(
            r"""
            harmonic_params = pd.read_csv(TABLES_DIR / "harmonic_parameters.csv")
            well_params = pd.read_csv(TABLES_DIR / "infinite_well_parameters.csv")
            harmonic_fidelity = pd.read_csv(TABLES_DIR / "harmonic_fidelity_vs_time.csv")
            well_fidelity = pd.read_csv(TABLES_DIR / "infinite_well_fidelity_vs_time.csv")
            validation = pd.read_csv(TABLES_DIR / "circuit_validation.csv")
            resource_totals = pd.read_csv(TABLES_DIR / "resource_totals_vs_r.csv")

            quality_records = [
                {"check": "Harmonic final fidelity above 0.999", "status": float(harmonic_fidelity["fidelity"].iloc[-1]) > 0.999},
                {"check": "Harmonic minimum fidelity above 0.999", "status": float(harmonic_fidelity["fidelity"].min()) > 0.999},
                {"check": "Harmonic edge probability below 1e-4", "status": float(harmonic_params["max_edge_probability"].iloc[0]) < 1e-4},
                {"check": "Infinite-well final fidelity above 0.999", "status": float(well_fidelity["fidelity"].iloc[-1]) > 0.999},
                {"check": "Infinite-well minimum fidelity above 0.999", "status": float(well_fidelity["fidelity"].min()) > 0.999},
                {"check": "Infinite-well periodic mismatch removed", "status": bool(well_params["periodic_boundary_mismatch"].iloc[0]) is False},
                {"check": "Transform validations passed", "status": bool(validation["passed"].all())},
                {"check": "Resource table records synthesis model", "status": "synthesis_model" in resource_totals.columns},
            ]
            quality_df = pd.DataFrame(quality_records)
            display(Markdown("## Numerical Quality Checks"))
            display(quality_df)

            if not quality_df["status"].all():
                raise RuntimeError("At least one numerical quality check failed.")
            """
        ),
        code(
            r"""
            summary_md = '''## Concise Methods Summary

            **Harmonic oscillator:** second-order symmetric split-operator evolution on a periodic 64-point QFT grid, benchmarked against a truncated analytical harmonic-oscillator eigenbasis and accompanied by boundary and spatial-convergence diagnostics.

            **Infinite well:** corrected hard-wall simulation using an orthonormal Dirichlet sine transform, the boundary-compatible Fourier/QFT-family transform. This removes the earlier periodic-ring mismatch and is benchmarked against the analytical sine-basis reference.

            **Circuit resources:** the harmonic oscillator uses a validated Qiskit QFT convention and exact small-N generic diagonal synthesis. The infinite well reports an analytical quantum-sine-transform resource estimate, explicitly separated from the harmonic generic-synthesis counts.
            '''
            display(Markdown(summary_md))
            display(Markdown("## Key Tables"))
            display(harmonic_params)
            display(well_params)
            display(pd.read_csv(TABLES_DIR / "circuit_single_step_metrics.csv"))
            display(resource_totals.head())
            """
        ),
        code(
            r"""
            checklist_df = pd.DataFrame(
                [
                    {"check": "Environment notebook present", "status": (NOTEBOOK_DIR / "00_environment_setup.ipynb").exists()},
                    {"check": "Harmonic dynamics notebook present", "status": (NOTEBOOK_DIR / "01_harmonic_oscillator_qft_split_operator.ipynb").exists()},
                    {"check": "Infinite-well dynamics notebook present", "status": (NOTEBOOK_DIR / "02_infinite_well_qft_split_operator.ipynb").exists()},
                    {"check": "Circuit notebook present", "status": (NOTEBOOK_DIR / "03_circuit_resource_analysis.ipynb").exists()},
                    {"check": "Publication notebook present", "status": (NOTEBOOK_DIR / "04_publication_exports.ipynb").exists()},
                    {"check": "All required figures and tables exist", "status": bool(manifest_df["exists"].all())},
                    {"check": "CSV manifest saved", "status": (TABLES_DIR / "publication_manifest.csv").exists()},
                    {"check": "All numerical quality checks pass", "status": bool(quality_df["status"].all())},
                    {"check": "Boundary model corrected for infinite well", "status": bool(well_params["periodic_boundary_mismatch"].iloc[0]) is False},
                ]
            )
            save_dataframe(checklist_df, "final_checklist.csv")
            display(Markdown("## Final Checklist"))
            display(checklist_df)

            if not checklist_df["status"].all():
                raise RuntimeError("The final checklist contains at least one failed item.")
            print("Publication export check completed successfully.")
            """
        ),
    ],
)


print("Paper-ready notebooks rebuilt.")
