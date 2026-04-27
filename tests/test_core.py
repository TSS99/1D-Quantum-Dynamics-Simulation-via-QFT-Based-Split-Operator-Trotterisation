from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qftsplit.core import (  # noqa: E402
    fidelity_summary,
    fft_momentum_grid,
    harmonic_potential,
    periodic_grid,
    qft_split_circuit_validation,
    run_harmonic_case,
    run_infinite_well_case,
    sine_transform_validation,
)


class SplitOperatorCoreTests(unittest.TestCase):
    def test_harmonic_reference_fidelity_is_high(self) -> None:
        result = run_harmonic_case(
            grid_size=64,
            x_left=-8.0,
            x_right=8.0,
            x0=2.0,
            k0=0.0,
            sigma=1.0,
            t_max=2.0 * np.pi,
            steps=200,
            mass=1.0,
            omega=1.0,
            hbar=1.0,
            reference_grid_size=4096,
            reference_tail_tolerance=1e-10,
            reference_basis_cap=128,
        )
        summary = fidelity_summary(result)
        self.assertGreater(summary["minimum_fidelity"], 0.999)
        self.assertLess(summary["max_edge_probability"], 1e-4)

    def test_infinite_well_uses_dirichlet_sine_transform(self) -> None:
        result = run_infinite_well_case(
            grid_size=64,
            length=10.0,
            x0=5.0,
            k0=2.0,
            sigma=0.8,
            t_max=6.0,
            steps=700,
            mass=1.0,
            hbar=1.0,
            reference_grid_size=4097,
            reference_tail_tolerance=1e-10,
            reference_basis_cap=256,
        )
        summary = fidelity_summary(result)
        self.assertGreater(summary["minimum_fidelity"], 0.999)

    def test_transform_convention_validations_pass(self) -> None:
        x, dx = periodic_grid(-8.0, 8.0, 64)
        dt = 2.0 * np.pi / 200
        p = fft_momentum_grid(64, dx=dx, hbar=1.0)
        position_phase = np.exp(-0.5j * harmonic_potential(x, mass=1.0, omega=1.0) * dt)
        momentum_phase = np.exp(-1j * (p**2 / 2.0) * dt)
        self.assertTrue(qft_split_circuit_validation(position_phase, momentum_phase)["passed"])
        self.assertTrue(sine_transform_validation(64, length=10.0, mass=1.0, hbar=1.0, dt=6.0 / 700)["passed"])


if __name__ == "__main__":
    unittest.main()
