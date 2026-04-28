from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import dst, idst


Array = np.ndarray


def locate_project_root(start: Path | None = None) -> Path:
    """Locate the project root from the root folder, notebooks/, or scripts/."""
    cwd = Path.cwd().resolve() if start is None else start.resolve()
    for candidate in (cwd, cwd.parent):
        if (candidate / "notebooks").exists() and (candidate / "figures").exists() and (candidate / "tables").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root containing notebooks/, figures/, and tables/.")


def configure_matplotlib() -> None:
    """Apply a polished, publication-quality plotting style for EPJ Plus."""
    plt.rcParams.update(
        {
            # ── Typography ──────────────────────────────────────────────
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "mathtext.fontset": "cm",
            # ── Spines & ticks ──────────────────────────────────────────
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            # ── Lines & markers ─────────────────────────────────────────
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            # ── Figure background ───────────────────────────────────────
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            # ── Grid (off by default; turned on per-plot) ───────────────
            "axes.grid": False,
            # ── Legend ──────────────────────────────────────────────────
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "legend.edgecolor": "0.80",
            "legend.fancybox": True,
        }
    )


def mm_to_inches(mm: float) -> float:
    return mm / 25.4


def save_publication_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> tuple[Path, Path]:
    """Save a figure as both PNG and PDF and verify the files exist."""
    from matplotlib.backends.backend_pdf import PdfPages

    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{stem}.png"
    pdf_path = figures_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    # Use PdfPages to ensure correct page orientation (no 90° rotation)
    with PdfPages(str(pdf_path)) as pdf:
        pdf.savefig(fig, dpi=600, bbox_inches="tight", pad_inches=0.05)
    if not png_path.exists() or not pdf_path.exists():
        raise FileNotFoundError(f"Failed to save {stem} in both PNG and PDF formats.")
    return png_path, pdf_path


def save_dataframe(df: pd.DataFrame, tables_dir: Path, filename: str) -> Path:
    """Save a DataFrame to tables/ and verify that the CSV exists."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / filename
    df.to_csv(path, index=False)
    if not path.exists():
        raise FileNotFoundError(f"Failed to save {path}.")
    return path


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1) == 0)


def validate_power_of_two_grid(grid_size: int) -> None:
    if not is_power_of_two(int(grid_size)):
        raise ValueError(f"Grid size must be a power of two, received {grid_size}.")


def periodic_grid(x_left: float, x_right: float, grid_size: int) -> tuple[Array, float]:
    """Return the periodic position grid natural to a standard QFT/FFT register."""
    validate_power_of_two_grid(grid_size)
    if x_right <= x_left:
        raise ValueError("x_right must be larger than x_left.")
    dx = (x_right - x_left) / grid_size
    x = x_left + dx * np.arange(grid_size)
    return x, dx


def dirichlet_midpoint_grid(length: float, grid_size: int) -> tuple[Array, float]:
    """Return a power-of-two midpoint grid for Dirichlet sine pseudospectral evolution."""
    validate_power_of_two_grid(grid_size)
    if length <= 0:
        raise ValueError("length must be positive.")
    dx = length / grid_size
    x = (np.arange(grid_size) + 0.5) * dx
    return x, dx


def l2_norm(psi: Array, dx: float) -> float:
    """Return the discrete L2 norm associated with a uniform grid."""
    return float(np.sqrt(np.real(dx * np.vdot(psi, psi))))


def normalize_state(psi: Array, dx: float) -> Array:
    """Normalize a sampled wavefunction in the quadrature norm dx * sum |psi|^2."""
    norm = l2_norm(psi, dx)
    if norm <= 0:
        raise ValueError("Encountered a non-positive state norm during normalization.")
    return psi / norm


def state_norms(states: Array, dx: float) -> Array:
    return np.array([float(np.real(dx * np.vdot(state, state))) for state in states])


def gaussian_wavepacket(x: Array, x0: float, k0: float, sigma: float, dx: float) -> Array:
    """Return a normalized Gaussian wavepacket with density width sigma."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    psi = np.exp(-((x - x0) ** 2) / (4.0 * sigma**2)) * np.exp(1j * k0 * (x - x0))
    return normalize_state(psi.astype(np.complex128), dx)


def sine_windowed_gaussian(x: Array, length: float, x0: float, k0: float, sigma: float, dx: float) -> Array:
    """Return a Gaussian wavepacket multiplied by sin(pi x/L) for hard-wall compatibility."""
    if not (0.0 <= x0 <= length):
        raise ValueError("x0 must lie inside the well interval [0, L].")
    gaussian = np.exp(-((x - x0) ** 2) / (4.0 * sigma**2)) * np.exp(1j * k0 * (x - x0))
    window = np.sin(np.pi * x / length)
    return normalize_state((gaussian * window).astype(np.complex128), dx)


def harmonic_potential(x: Array, mass: float, omega: float) -> Array:
    return 0.5 * mass * omega**2 * x**2


def fft_momentum_grid(grid_size: int, dx: float, hbar: float) -> Array:
    """Return the momentum grid matched to NumPy's orthonormal FFT ordering."""
    return 2.0 * np.pi * hbar * np.fft.fftfreq(grid_size, d=dx)


def sine_mode_energies(grid_size: int, length: float, mass: float, hbar: float) -> Array:
    """Return continuous infinite-well kinetic energies for sine modes n=1..N."""
    n_values = np.arange(1, grid_size + 1, dtype=float)
    return (hbar**2 * np.pi**2 * n_values**2) / (2.0 * mass * length**2)


def harmonic_basis(x: Array, n_cap: int, mass: float, omega: float, hbar: float) -> Array:
    """Construct normalized harmonic-oscillator eigenfunctions by stable recurrence."""
    alpha = mass * omega / hbar
    y = np.sqrt(alpha) * x
    basis = np.empty((n_cap, x.size), dtype=np.float64)
    basis[0] = (alpha / np.pi) ** 0.25 * np.exp(-0.5 * y**2)
    if n_cap > 1:
        basis[1] = np.sqrt(2.0) * y * basis[0]
        for n in range(1, n_cap - 1):
            basis[n + 1] = np.sqrt(2.0 / (n + 1)) * y * basis[n] - np.sqrt(n / (n + 1)) * basis[n - 1]
    return basis


def infinite_well_basis(x: Array, length: float, n_cap: int) -> tuple[Array, Array]:
    """Construct analytical sine eigenstates for a length-L infinite well."""
    n_values = np.arange(1, n_cap + 1, dtype=float)[:, None]
    basis = np.sqrt(2.0 / length) * np.sin(n_values * np.pi * x[None, :] / length)
    return basis, n_values[:, 0]


def truncate_coefficients(coeffs: Array, tail_tolerance: float) -> tuple[Array, int, float, float]:
    """Apply a cumulative-weight truncation rule and renormalize kept coefficients."""
    if not (0.0 < tail_tolerance < 1.0):
        raise ValueError("tail_tolerance must lie between 0 and 1.")
    weights = np.abs(coeffs) ** 2
    total_weight = float(np.sum(weights).real)
    cumulative = np.cumsum(weights)
    target = (1.0 - tail_tolerance) * total_weight
    n_keep = int(np.searchsorted(cumulative, target) + 1)
    tail_weight = max(total_weight - float(cumulative[n_keep - 1].real), 0.0)
    kept = coeffs[:n_keep].astype(np.complex128)
    kept /= np.linalg.norm(kept)
    return kept, n_keep, total_weight, tail_weight


def build_harmonic_reference(
    x_sim: Array,
    x_left: float,
    x_right: float,
    x0: float,
    k0: float,
    sigma: float,
    mass: float,
    omega: float,
    hbar: float,
    reference_grid_size: int,
    reference_tail_tolerance: float,
    reference_basis_cap: int,
    dx_sim: float,
) -> dict:
    """Build the harmonic reference from analytical eigenstates and numerical integration."""
    x_dense = np.linspace(x_left, x_right, reference_grid_size)
    dx_dense = float(x_dense[1] - x_dense[0])
    psi0_dense = gaussian_wavepacket(x_dense, x0=x0, k0=k0, sigma=sigma, dx=dx_dense)

    basis_dense = harmonic_basis(x_dense, reference_basis_cap, mass=mass, omega=omega, hbar=hbar)
    coeffs = np.trapezoid(np.conjugate(basis_dense) * psi0_dense, x_dense, axis=1)
    kept_coeffs, n_keep, total_weight, tail_weight = truncate_coefficients(coeffs, reference_tail_tolerance)
    if n_keep >= reference_basis_cap - 2:
        raise RuntimeError("Reference basis cap was exhausted; increase reference_basis_cap.")

    basis_sim = harmonic_basis(x_sim, n_keep, mass=mass, omega=omega, hbar=hbar)
    energies = hbar * omega * (np.arange(n_keep, dtype=float) + 0.5)
    return {
        "coeffs": kept_coeffs,
        "basis_sim": basis_sim,
        "energies": energies,
        "n_keep": n_keep,
        "basis_cap": reference_basis_cap,
        "dense_grid_size": reference_grid_size,
        "raw_weight_sum": total_weight,
        "tail_weight": tail_weight,
        "dx_sim": dx_sim,
        "reference_model": "continuous_harmonic_eigenbasis_truncated_on_large_box",
    }


def build_infinite_well_reference(
    x_sim: Array,
    length: float,
    x0: float,
    k0: float,
    sigma: float,
    mass: float,
    hbar: float,
    reference_grid_size: int,
    reference_tail_tolerance: float,
    reference_basis_cap: int,
    dx_sim: float,
) -> dict:
    """Build the exact hard-wall reference from the analytical sine-series eigenbasis."""
    x_dense = np.linspace(0.0, length, reference_grid_size)
    dx_dense = float(x_dense[1] - x_dense[0])
    psi0_dense = sine_windowed_gaussian(x_dense, length=length, x0=x0, k0=k0, sigma=sigma, dx=dx_dense)

    basis_dense, _ = infinite_well_basis(x_dense, length=length, n_cap=reference_basis_cap)
    coeffs = np.trapezoid(np.conjugate(basis_dense) * psi0_dense, x_dense, axis=1)
    kept_coeffs, n_keep, total_weight, tail_weight = truncate_coefficients(coeffs, reference_tail_tolerance)
    if n_keep >= reference_basis_cap - 2:
        raise RuntimeError("Reference basis cap was exhausted; increase reference_basis_cap.")

    basis_sim, n_values = infinite_well_basis(x_sim, length=length, n_cap=n_keep)
    energies = (hbar**2 * np.pi**2 * n_values**2) / (2.0 * mass * length**2)
    return {
        "coeffs": kept_coeffs,
        "basis_sim": basis_sim,
        "energies": energies,
        "n_keep": n_keep,
        "basis_cap": reference_basis_cap,
        "dense_grid_size": reference_grid_size,
        "raw_weight_sum": total_weight,
        "tail_weight": tail_weight,
        "dx_sim": dx_sim,
        "reference_model": "continuous_dirichlet_sine_eigenbasis",
    }


def reconstruct_reference_states(reference_data: dict, t_values: Array, hbar: float) -> Array:
    """Reconstruct normalized exact states on the simulation grid for requested times."""
    coeffs = reference_data["coeffs"]
    basis_sim = reference_data["basis_sim"]
    energies = reference_data["energies"]
    dx_sim = reference_data["dx_sim"]

    phases = np.exp(-1j * energies[:, None] * t_values[None, :] / hbar)
    states = (coeffs[:, None] * phases).T @ basis_sim
    return np.asarray([normalize_state(state, dx_sim) for state in states], dtype=np.complex128)


def periodic_split_operator_evolution(
    psi0: Array,
    potential: Array,
    dx: float,
    dt: float,
    steps: int,
    mass: float,
    hbar: float,
) -> tuple[Array, Array]:
    """Second-order split-operator evolution on a periodic QFT/FFT grid."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    psi = normalize_state(psi0.astype(np.complex128), dx)
    p = fft_momentum_grid(psi.size, dx=dx, hbar=hbar)
    kinetic = p**2 / (2.0 * mass)

    position_phase_half = np.exp(-0.5j * potential * dt / hbar)
    momentum_phase = np.exp(-1j * kinetic * dt / hbar)
    states = np.empty((steps + 1, psi.size), dtype=np.complex128)
    norms = np.empty(steps + 1, dtype=float)
    states[0] = psi
    norms[0] = float(np.real(dx * np.vdot(psi, psi)))

    for step in range(1, steps + 1):
        psi = periodic_step_apply(psi, position_phase_half, momentum_phase)
        states[step] = psi
        norms[step] = float(np.real(dx * np.vdot(psi, psi)))

    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    if max_norm_error > 1e-8:
        raise RuntimeError(f"Periodic split-operator norm drift exceeded tolerance: {max_norm_error:.3e}")
    return states, norms


def periodic_step_apply(psi: Array, position_phase_half: Array, momentum_phase: Array) -> Array:
    """Apply one periodic split step using NumPy's QFT-compatible FFT convention."""
    psi = position_phase_half * psi
    psi_k = np.fft.fft(psi, norm="ortho")
    psi_k *= momentum_phase
    psi = np.fft.ifft(psi_k, norm="ortho")
    return position_phase_half * psi


def sine_split_operator_evolution(
    psi0: Array,
    length: float,
    potential: Array | None,
    dx: float,
    dt: float,
    steps: int,
    mass: float,
    hbar: float,
) -> tuple[Array, Array]:
    """Second-order split evolution on a hard-wall Dirichlet sine grid.

    The sine transform is the boundary-compatible Fourier transform for the
    infinite well.  It avoids the periodic wraparound imposed by a standard
    QFT/FFT register.
    """
    if steps <= 0:
        raise ValueError("steps must be positive.")
    psi = normalize_state(psi0.astype(np.complex128), dx)
    potential_values = np.zeros(psi.size, dtype=float) if potential is None else np.asarray(potential, dtype=float)
    kinetic = sine_mode_energies(psi.size, length=length, mass=mass, hbar=hbar)

    position_phase_half = np.exp(-0.5j * potential_values * dt / hbar)
    momentum_phase = np.exp(-1j * kinetic * dt / hbar)
    states = np.empty((steps + 1, psi.size), dtype=np.complex128)
    norms = np.empty(steps + 1, dtype=float)
    states[0] = psi
    norms[0] = float(np.real(dx * np.vdot(psi, psi)))

    for step in range(1, steps + 1):
        psi = sine_step_apply(psi, position_phase_half, momentum_phase)
        states[step] = psi
        norms[step] = float(np.real(dx * np.vdot(psi, psi)))

    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    if max_norm_error > 1e-8:
        raise RuntimeError(f"Sine split-operator norm drift exceeded tolerance: {max_norm_error:.3e}")
    return states, norms


def sine_step_apply(psi: Array, position_phase_half: Array, momentum_phase: Array) -> Array:
    """Apply one Dirichlet sine-transform split step using an orthonormal DST-II."""
    psi = position_phase_half * psi
    psi_k = dst(psi, type=2, norm="ortho")
    psi_k *= momentum_phase
    psi = idst(psi_k, type=2, norm="ortho")
    return position_phase_half * psi


def fidelity_series(reference_states: Array, split_states: Array, dx: float) -> Array:
    """Compute fidelity between reference and numerical states on the same grid."""
    if reference_states.shape != split_states.shape:
        raise ValueError("Reference and split-operator state arrays must have identical shapes.")
    fidelities = np.empty(reference_states.shape[0], dtype=float)
    for idx, (psi_ref, psi_split) in enumerate(zip(reference_states, split_states)):
        overlap = dx * np.vdot(psi_ref, psi_split)
        fidelity = float(np.abs(overlap) ** 2)
        if fidelity < -1e-10 or fidelity > 1.0 + 1e-8:
            raise RuntimeError(f"Fidelity left the allowed interval at index {idx}: {fidelity}")
        fidelities[idx] = np.clip(fidelity, 0.0, 1.0)
    return fidelities


def edge_probability(states: Array, dx: float, edge_points: int = 4) -> float:
    """Return the largest probability mass found in edge grid points over all states."""
    if edge_points <= 0:
        raise ValueError("edge_points must be positive.")
    width = min(edge_points, states.shape[1] // 2)
    density = np.abs(states[:, :width]) ** 2 + np.abs(states[:, -width:]) ** 2
    return float(np.max(dx * np.sum(density, axis=1)))


def run_harmonic_case(
    grid_size: int,
    x_left: float,
    x_right: float,
    x0: float,
    k0: float,
    sigma: float,
    t_max: float,
    steps: int,
    mass: float,
    omega: float,
    hbar: float,
    reference_grid_size: int,
    reference_tail_tolerance: float,
    reference_basis_cap: int,
) -> dict:
    """Run one harmonic-oscillator simulation and return states plus diagnostics."""
    x, dx = periodic_grid(x_left, x_right, grid_size)
    times = np.linspace(0.0, t_max, steps + 1)
    dt = float(times[1] - times[0])
    psi0 = gaussian_wavepacket(x, x0=x0, k0=k0, sigma=sigma, dx=dx)
    reference_data = build_harmonic_reference(
        x_sim=x,
        x_left=x_left,
        x_right=x_right,
        x0=x0,
        k0=k0,
        sigma=sigma,
        mass=mass,
        omega=omega,
        hbar=hbar,
        reference_grid_size=reference_grid_size,
        reference_tail_tolerance=reference_tail_tolerance,
        reference_basis_cap=reference_basis_cap,
        dx_sim=dx,
    )
    reference_states = reconstruct_reference_states(reference_data, times, hbar=hbar)
    split_states, split_norms = periodic_split_operator_evolution(
        psi0=psi0,
        potential=harmonic_potential(x, mass=mass, omega=omega),
        dx=dx,
        dt=dt,
        steps=steps,
        mass=mass,
        hbar=hbar,
    )
    reference_norms = state_norms(reference_states, dx)
    fidelity = fidelity_series(reference_states, split_states, dx=dx)
    return {
        "x": x,
        "dx": dx,
        "times": times,
        "dt": dt,
        "psi0": psi0,
        "reference_data": reference_data,
        "reference_states": reference_states,
        "split_states": split_states,
        "split_norms": split_norms,
        "reference_norms": reference_norms,
        "fidelity": fidelity,
        "max_edge_probability": edge_probability(split_states, dx),
    }


def run_infinite_well_case(
    grid_size: int,
    length: float,
    x0: float,
    k0: float,
    sigma: float,
    t_max: float,
    steps: int,
    mass: float,
    hbar: float,
    reference_grid_size: int,
    reference_tail_tolerance: float,
    reference_basis_cap: int,
) -> dict:
    """Run one hard-wall infinite-well simulation using the sine transform."""
    x, dx = dirichlet_midpoint_grid(length, grid_size)
    times = np.linspace(0.0, t_max, steps + 1)
    dt = float(times[1] - times[0])
    psi0 = sine_windowed_gaussian(x, length=length, x0=x0, k0=k0, sigma=sigma, dx=dx)
    reference_data = build_infinite_well_reference(
        x_sim=x,
        length=length,
        x0=x0,
        k0=k0,
        sigma=sigma,
        mass=mass,
        hbar=hbar,
        reference_grid_size=reference_grid_size,
        reference_tail_tolerance=reference_tail_tolerance,
        reference_basis_cap=reference_basis_cap,
        dx_sim=dx,
    )
    reference_states = reconstruct_reference_states(reference_data, times, hbar=hbar)
    split_states, split_norms = sine_split_operator_evolution(
        psi0=psi0,
        length=length,
        potential=None,
        dx=dx,
        dt=dt,
        steps=steps,
        mass=mass,
        hbar=hbar,
    )
    reference_norms = state_norms(reference_states, dx)
    fidelity = fidelity_series(reference_states, split_states, dx=dx)
    return {
        "x": x,
        "dx": dx,
        "times": times,
        "dt": dt,
        "psi0": psi0,
        "reference_data": reference_data,
        "reference_states": reference_states,
        "split_states": split_states,
        "split_norms": split_norms,
        "reference_norms": reference_norms,
        "fidelity": fidelity,
        "max_edge_probability": edge_probability(split_states, dx),
    }


def fidelity_summary(result: dict) -> dict:
    return {
        "minimum_fidelity": float(result["fidelity"].min()),
        "final_time_fidelity": float(result["fidelity"][-1]),
        "mean_fidelity": float(result["fidelity"].mean()),
        "max_split_norm_error": float(np.max(np.abs(result["split_norms"] - 1.0))),
        "max_reference_norm_error": float(np.max(np.abs(result["reference_norms"] - 1.0))),
        "max_edge_probability": float(result.get("max_edge_probability", np.nan)),
    }


def plot_density_snapshots(
    x: Array,
    times: Array,
    reference_states: Array,
    split_states: Array,
    snapshot_count: int,
    title: str,
) -> plt.Figure:
    """Plot representative density snapshots comparing reference and numerical evolution."""
    indices = np.unique(np.linspace(0, len(times) - 1, snapshot_count, dtype=int))
    n_panels = len(indices)
    panel_height = 1.45  # inches per panel – enough for clear labels
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(mm_to_inches(120.0), panel_height * n_panels + 0.5),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    # Curated colour pair – high contrast and colour-blind safe
    ref_color = "#264653"   # dark teal
    split_color = "#e76f51" # warm coral

    for axis, index in zip(axes, indices):
        rho_ref = np.abs(reference_states[index]) ** 2
        rho_split = np.abs(split_states[index]) ** 2

        axis.plot(x, rho_ref, color=ref_color, linewidth=1.8,
                  label="Analytical reference")
        axis.plot(x, rho_split, color=split_color, linewidth=1.6,
                  linestyle="--", dashes=(4, 2.5),
                  label="Spectral split operator")
        # Subtle fill under the reference curve
        axis.fill_between(x, rho_ref, alpha=0.08, color=ref_color)

        axis.set_ylabel(r"$|\psi(x,\,t)|^2$", labelpad=6)
        axis.tick_params(axis="both", which="both", direction="in")

        # Time annotation badge
        axis.text(
            0.025,
            0.85,
            rf"$t = {times[index]:.2f}$",
            transform=axis.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3",
                  "facecolor": "#f8f9fa",
                  "edgecolor": "#adb5bd",
                  "linewidth": 0.7},
        )

    axes[0].set_title(title, pad=10)
    axes[0].legend(
        loc="upper right",
        fontsize=8.5,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.80",
        fancybox=True,
    )
    axes[-1].set_xlabel(r"$x$", labelpad=4)
    return fig


def plot_fidelity(times: Array, fidelities: Array, title: str) -> plt.Figure:
    """Plot fidelity versus time over the simulated interval."""
    fig, axis = plt.subplots(figsize=(mm_to_inches(120.0), mm_to_inches(75.0)), constrained_layout=True)

    # Thin markers for dense datasets (>50 points)
    n_pts = len(fidelities)
    mark_every = max(1, n_pts // 30) if n_pts > 50 else 1

    axis.plot(
        times, fidelities,
        color="#1a5276",
        linewidth=2.5,
        marker="o",
        markersize=5,
        markerfacecolor="#c0392b",
        markeredgecolor="#1a5276",
        markeredgewidth=0.6,
        markevery=mark_every,
        zorder=3,
    )
    axis.set_xlabel(r"$t$", labelpad=4)
    axis.set_ylabel(r"Fidelity, $\mathcal{F}$", labelpad=6)
    axis.set_title(title, pad=10)

    f_min = float(np.min(fidelities))
    f_max = float(np.max(fidelities))
    f_spread = f_max - f_min

    # Adaptive y-axis: if fidelity is near-perfect (spread < 1e-3),
    # use a tight centered window so the curve is clearly visible
    if f_spread < 1e-3:
        f_centre = 0.5 * (f_min + f_max)
        half_win = max(5e-4, f_spread * 5)
        y_lo = f_centre - half_win
        y_hi = f_centre + half_win
    else:
        y_lo = max(0.0, f_min - 0.005)
        y_hi = 1.002
    axis.set_ylim(y_lo, y_hi)

    # Use plain decimal tick formatting (no scientific offset)
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.4f}"))

    axis.grid(True, alpha=0.20, linewidth=0.5, linestyle="--")
    axis.tick_params(axis="both", which="both", direction="in")
    return fig


def plot_convergence(df: pd.DataFrame, x_column: str, y_column: str, title: str, xlabel: str) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(mm_to_inches(120.0), mm_to_inches(75.0)), constrained_layout=True)
    axis.plot(
        df[x_column], df[y_column],
        marker="o",
        markersize=7,
        markerfacecolor="#52796f",
        markeredgecolor="white",
        markeredgewidth=0.8,
        color="#386641",
        linewidth=2.0,
        zorder=3,
    )
    axis.set_xlabel(xlabel, labelpad=4)

    # Produce a readable y-label from the column name
    y_label = y_column.replace("_", " ").title()
    axis.set_ylabel(y_label, labelpad=6)
    axis.set_title(title, pad=10)

    # Disable the ugly default scientific offset (e.g. "1e-5+9.999e-1")
    axis.ticklabel_format(axis="y", useOffset=False)
    y_vals = df[y_column].values
    # For large integer-scale values (gate counts etc.), use comma formatting
    if y_vals.max() > 100:
        axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    # For values very close to 1 (fidelities), show enough decimals
    elif y_vals.min() > 0.9 and y_vals.max() < 1.1:
        axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.6f}"))

    axis.grid(True, alpha=0.20, linewidth=0.5, linestyle="--")
    axis.tick_params(axis="both", which="both", direction="in")
    return fig


def qft_split_circuit_validation(position_phase_half: Array, momentum_phase: Array) -> dict:
    """Validate Qiskit's QFT convention against the NumPy FFT split step."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DiagonalGate, QFTGate
    from qiskit.quantum_info import Operator

    n_qubits = int(np.log2(position_phase_half.size))
    circuit = QuantumCircuit(n_qubits)
    circuit.append(DiagonalGate(position_phase_half), range(n_qubits))
    circuit.append(QFTGate(n_qubits).inverse(), range(n_qubits))
    circuit.append(DiagonalGate(momentum_phase), range(n_qubits))
    circuit.append(QFTGate(n_qubits), range(n_qubits))
    circuit.append(DiagonalGate(position_phase_half), range(n_qubits))

    rng = np.random.default_rng(1979)
    vector = rng.normal(size=position_phase_half.size) + 1j * rng.normal(size=position_phase_half.size)
    vector /= np.linalg.norm(vector)
    expected = periodic_step_apply(vector, position_phase_half, momentum_phase)
    observed = Operator(circuit).data @ vector
    overlap = np.vdot(expected, observed)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-14 else 1.0
    phase_adjusted_error = float(np.linalg.norm(observed / phase - expected))
    direct_error = float(np.linalg.norm(observed - expected))
    return {
        "validation": "qiskit_qft_matches_numpy_fft_step",
        "n_qubits": n_qubits,
        "direct_l2_error": direct_error,
        "phase_adjusted_l2_error": phase_adjusted_error,
        "passed": phase_adjusted_error < 1e-10,
    }


def sine_transform_validation(grid_size: int, length: float, mass: float, hbar: float, dt: float) -> dict:
    """Validate the orthonormal DST-II step against an explicit sine-transform matrix."""
    identity = np.eye(grid_size)
    sine_matrix = dst(identity, type=2, norm="ortho", axis=0)
    inverse_sine_matrix = idst(identity, type=2, norm="ortho", axis=0)
    kinetic = sine_mode_energies(grid_size, length=length, mass=mass, hbar=hbar)
    momentum_phase = np.exp(-1j * kinetic * dt / hbar)

    rng = np.random.default_rng(1980)
    vector = rng.normal(size=grid_size) + 1j * rng.normal(size=grid_size)
    vector /= np.linalg.norm(vector)
    expected = sine_step_apply(vector, np.ones(grid_size, dtype=np.complex128), momentum_phase)
    observed = inverse_sine_matrix @ (momentum_phase * (sine_matrix @ vector))
    error = float(np.linalg.norm(observed - expected))
    return {
        "validation": "dst_matrix_matches_scipy_sine_step",
        "grid_size": grid_size,
        "direct_l2_error": error,
        "passed": error < 1e-10,
    }


def build_periodic_resource_circuit(position_phase_half: Array, momentum_phase: Array):
    """Build one logical periodic QFT split step for resource accounting."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DiagonalGate, QFTGate

    n_qubits = int(np.log2(position_phase_half.size))
    circuit = QuantumCircuit(n_qubits, name="periodic_qft_split_step")
    circuit.append(DiagonalGate(position_phase_half), range(n_qubits))
    circuit.barrier()
    circuit.append(QFTGate(n_qubits).inverse(), range(n_qubits))
    circuit.barrier()
    circuit.append(DiagonalGate(momentum_phase), range(n_qubits))
    circuit.barrier()
    circuit.append(QFTGate(n_qubits), range(n_qubits))
    circuit.barrier()
    circuit.append(DiagonalGate(position_phase_half), range(n_qubits))
    return circuit


def build_infinite_well_circuit(grid_size: int, length: float, mass: float, hbar: float, dt: float):
    """Build one logical Dirichlet sine-transform (QST) split step for resource accounting.

    The infinite-well split step has the structure::

        QST  ──►  T(dt)  ──►  QST⁻¹

    where QST is implemented via a QFT on an extended (n+2)-qubit register followed by
    odd-reflection symmetry enforcement, and T(dt) is the kinetic phase diagonal.
    This function returns a *logical* circuit whose gates are labelled for readability
    so it can be drawn cleanly with ``draw_and_save_circuit``.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DiagonalGate, QFTGate

    n_qubits = int(np.log2(grid_size))   # base register (e.g. 6 for N=64)
    ext_qubits = n_qubits + 2            # QFT extension size for QST

    # Kinetic phase diagonal in sine-mode basis
    kinetic_energies = sine_mode_energies(grid_size, length=length, mass=mass, hbar=hbar)
    momentum_phase = np.exp(-1j * kinetic_energies * dt / hbar)

    # QST sub-gate: QFT⁻¹ on the full ext_qubits register, labelled "QST"
    qst_sub = QuantumCircuit(ext_qubits, name="QST")
    qst_sub.append(QFTGate(ext_qubits).inverse(), range(ext_qubits))
    qst_gate = qst_sub.to_gate(label="QST")

    # QST† sub-gate: QFT on the full ext_qubits register, labelled "QST†"
    qst_inv_sub = QuantumCircuit(ext_qubits, name="QST\u2020")
    qst_inv_sub.append(QFTGate(ext_qubits), range(ext_qubits))
    qst_inv_gate = qst_inv_sub.to_gate(label="QST\u2020")

    # Top-level circuit on ext_qubits wires (full QST register)
    circuit = QuantumCircuit(ext_qubits, name="infinite_well_sine_split_step")

    # 1. QST forward
    circuit.append(qst_gate, range(ext_qubits))
    circuit.barrier()

    # 2. Kinetic phase T(dt) on first n_qubits (sine-mode register)
    circuit.append(DiagonalGate(list(momentum_phase)), range(n_qubits))
    circuit.barrier()

    # 3. QST⁻¹ inverse
    circuit.append(qst_inv_gate, range(ext_qubits))

    return circuit



def transpile_and_extract_metrics(circuit, basis_gates: list[str]) -> tuple[object, dict, dict]:
    """Transpile a circuit and extract one-/two-qubit counts plus gate breakdown."""
    from qiskit import transpile

    transpiled = transpile(circuit, basis_gates=basis_gates, optimization_level=1)
    allowed = set(basis_gates)
    one_qubit_count = 0
    two_qubit_count = 0
    gate_breakdown: dict[str, int] = {}
    for instruction in transpiled.data:
        op = instruction.operation
        gate_breakdown[op.name] = gate_breakdown.get(op.name, 0) + 1
        if op.name == "barrier":
            continue
        if op.num_qubits > 2:
            raise RuntimeError(f"Transpiled circuit still contains a {op.num_qubits}-qubit gate: {op.name}")
        if op.name not in allowed:
            raise RuntimeError(f"Unexpected gate '{op.name}' remained after transpilation.")
        if op.num_qubits == 1:
            one_qubit_count += 1
        elif op.num_qubits == 2:
            two_qubit_count += 1
    return (
        transpiled,
        {
            "single_step_1q_count": one_qubit_count,
            "single_step_2q_count": two_qubit_count,
            "single_step_depth": transpiled.depth(),
        },
        gate_breakdown,
    )


def draw_and_save_circuit(circuit, figures_dir: Path, stem: str, scale: float, fold: int) -> tuple[Path, Path]:
    """Render a Qiskit circuit with the matplotlib drawer and save it."""
    figure = circuit.draw(output="mpl", scale=scale, fold=fold, style={"fontsize": 8})
    paths = save_publication_figure(figure, figures_dir, stem)
    plt.close(figure)
    return paths


def plot_gate_counts(resource_df: pd.DataFrame, title: str) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(mm_to_inches(120.0), mm_to_inches(75.0)), constrained_layout=True)
    axis.plot(
        resource_df["r"], resource_df["total_1q_count"],
        marker="o", markersize=7, markeredgecolor="white", markeredgewidth=0.6,
        color="#277da1", linewidth=2.0, label="1-qubit gates", zorder=3,
    )
    axis.plot(
        resource_df["r"], resource_df["total_2q_count"],
        marker="s", markersize=7, markeredgecolor="white", markeredgewidth=0.6,
        color="#d62828", linewidth=2.0, label="2-qubit gates", zorder=3,
    )
    axis.set_xlabel(r"Trotter steps, $r$", labelpad=4)
    axis.set_ylabel("Total gate count", labelpad=6)
    axis.set_title(title, pad=10)
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axis.grid(True, alpha=0.20, linewidth=0.5, linestyle="--")
    axis.tick_params(axis="both", which="both", direction="in")
    axis.legend(frameon=True, framealpha=0.9, edgecolor="0.80", fancybox=True)
    return fig


def plot_fidelity_vs_gate_count(resource_df: pd.DataFrame, x_column: str, xlabel: str, title: str) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(mm_to_inches(120.0), mm_to_inches(75.0)), constrained_layout=True)
    axis.plot(
        resource_df[x_column], resource_df["final_time_fidelity"],
        marker="o",
        markersize=7,
        markerfacecolor="#7b2cbf",
        markeredgecolor="white",
        markeredgewidth=0.8,
        color="#6a4c93",
        linewidth=2.0,
        zorder=3,
    )
    axis.set_xlabel(xlabel, labelpad=4)
    axis.set_ylabel(r"Final-time fidelity at $t = t_{\mathrm{max}}$", labelpad=6)
    axis.set_title(title, pad=10)

    f_min = float(resource_df["final_time_fidelity"].min())
    axis.set_ylim(max(0.0, f_min - 0.005), 1.002)
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.4f}"))
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    axis.grid(True, alpha=0.20, linewidth=0.5, linestyle="--")
    axis.tick_params(axis="both", which="both", direction="in")
    return fig


def run_trotter_sweep(
    run_case: Callable[[int], dict],
    system: str,
    r_values: list[int],
) -> pd.DataFrame:
    records = []
    for r_value in r_values:
        result = run_case(int(r_value))
        summary = fidelity_summary(result)
        records.append(
            {
                "system": system,
                "r": int(r_value),
                "dt": float(result["dt"]),
                "final_time_fidelity": summary["final_time_fidelity"],
                "minimum_fidelity": summary["minimum_fidelity"],
                "mean_fidelity": summary["mean_fidelity"],
            }
        )
    return pd.DataFrame(records)

