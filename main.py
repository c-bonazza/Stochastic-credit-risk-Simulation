"""
main.py — Entry point for the Monte Carlo Credit Risk Simulation project
======================================================================

Orchestrates the full pipeline:
    1. Generate synthetic portfolio (data_generation.py)
    2. Monte Carlo simulation (Base + Stress) (simulator.py)
    3. Export interactive dashboard (visualize.py)
"""

import subprocess
import sys
import time

PYTHON = sys.executable

STEPS = [
    ("1/3 — Generate portfolio", "data_generation.py"),
    ("2/3 — Monte Carlo simulation (Base + Stress)", "simulator.py"),
    ("3/3 — Interactive dashboard", "visualize.py"),
]


def main():
    print("=" * 60)
    print("  MONTE CARLO CREDIT RISK SIMULATION")
    print("  One-factor Gaussian copula model (Vasicek)")
    print("=" * 60)

    t0 = time.time()

    for label, script in STEPS:
        print(f"\n> {label}  [{script}]")
        print("-" * 60)
        result = subprocess.run([PYTHON, script], check=True)
        if result.returncode != 0:
            print(f"  ✗ Erreur lors de l'exécution de {script}")
            sys.exit(1)

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("  Pipeline finished in {elapsed:.1f}s")
    print("  Generated files:")
    print("    • portfolio.csv              — Loan portfolio")
    print("    • stress_test_comparison.png — Base vs Stress plot")
    print("    • dashboard.html             — Interactive Plotly dashboard")
    print("=" * 60)


if __name__ == "__main__":
    main()
