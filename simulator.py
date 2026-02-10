"""
simulator.py — Monte Carlo Credit Risk Simulation Engine
========================================================

Implements a one-factor Merton/Vasicek credit risk model to estimate the
loss distribution of a loan portfolio via Monte Carlo simulation.

Theoretical model:
    Each borrower i has a latent variable:
        X_i = sqrt(ρ) · Z + sqrt(1 − ρ) · ε_i
    where Z ~ N(0,1) is the common systematic factor and ε_i ~ N(0,1)
    is the idiosyncratic factor. A default occurs if X_i < Φ⁻¹(PD_i).

Reference: Vasicek, O.A. (2002). "The Distribution of Loan Portfolio Value."
"""

import yaml
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(path="config.yaml"):
    """Load simulation parameters from a YAML file.

    Parameters
    ----------
    path : str
        Path to the configuration file.

    Returns
    -------
    dict
        Dictionary containing parameters: n_simulations, confidence_level,
        rho_base, rho_stress, lgd_multiplier, etc.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_portfolio(filepath):
    """Load the loan portfolio and compute default thresholds.

    Each asset gets a default threshold corresponding to the quantile of
    the standard normal distribution for its PD: threshold_i = Φ⁻¹(PD_i).

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing columns Asset_ID, Sector, EAD, PD, LGD.

    Returns
    -------
    portfolio : pd.DataFrame
        Portfolio DataFrame.
    ead : np.ndarray
        Exposure at Default vector.
    pd_values : np.ndarray
        Probability of default vector.
    lgd : np.ndarray
        Loss given default vector.
    default_threshold : np.ndarray
        Default thresholds Φ⁻¹(PD_i).
    """
    portfolio = pd.read_csv(filepath)
    ead = portfolio["EAD"].values
    pd_values = portfolio["PD"].values
    lgd = portfolio["LGD"].values
    default_threshold = norm.ppf(pd_values)
    return portfolio, ead, pd_values, lgd, default_threshold


def run_simulation(rho, lgd_values, ead, default_threshold, n_simulations,
                   confidence, label="Simulation"):
    """Run a Monte Carlo simulation for the one-factor Vasicek model.

    For each scenario s ∈ {1, …, N}:
      1. Draw systematic factor Z_s ~ N(0,1).
      2. Draw idiosyncratic factors ε_{s,i} ~ N(0,1).
      3. Compute latent variable:
             X_{s,i} = √ρ · Z_s + √(1−ρ) · ε_{s,i}
      4. Default if X_{s,i} < Φ⁻¹(PD_i).
      5. Scenario loss: L_s = Σ_i EAD_i · LGD_i · 𝟙{default_i}.

    Metrics computed:
      - Expected Loss (EL): E[L] — mean of simulated losses.
      - Value-at-Risk (VaR) at level α: quantile satisfying P(L ≤ VaR_α) = α.
      - Expected Shortfall (ES) at level α (CVaR): ES_α = E[L | L ≥ VaR_α].

    Parameters
    ----------
    rho : float
        Asset correlation (systemic parameter of the Vasicek model).
    lgd_values : np.ndarray
        LGD vector (possibly stressed).
    ead : np.ndarray
        Exposure at Default vector.
    default_threshold : np.ndarray
        Default thresholds Φ⁻¹(PD_i).
    n_simulations : int
        Number of Monte Carlo scenarios.
    confidence : float
        Confidence level (e.g. 0.99 for 99% VaR).
    label : str
        Scenario label for console output.

    Returns
    -------
    losses : np.ndarray, shape (n_simulations,)
        Vector of total simulated losses.
    """
    n_assets = len(ead)

    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"  {n_simulations:,} simulations | {n_assets} actifs | rho = {rho}")
    print(f"{'=' * 55}")

    Z = np.random.standard_normal((n_simulations, 1))
    epsilon = np.random.standard_normal((n_simulations, n_assets))

    X = np.sqrt(rho) * Z + np.sqrt(1 - rho) * epsilon

    default_indicator = (X < default_threshold).astype(np.float64)

    losses = default_indicator @ (ead * lgd_values)

    var = np.percentile(losses, confidence * 100)
    es = losses[losses >= var].mean()
    expected_loss = losses.mean()

    print(f"  Expected Loss (EL)        : {expected_loss / 1e6:>10,.2f} M€")
    print(f"  VaR ({confidence:.0%})                : {var / 1e6:>10,.2f} M€")
    print(f"  Expected Shortfall ({confidence:.0%})  : {es / 1e6:>10,.2f} M€")
    print(f"  Min loss                  : {losses.min() / 1e6:>10,.2f} M€")
    print(f"  Max loss                  : {losses.max() / 1e6:>10,.2f} M€")
    print(f"  Loss std. dev.            : {losses.std() / 1e6:>10,.2f} M€")

    return losses


def run_stress_test(lgd, ead, default_threshold, cfg):
    """Exécute le scénario de stress « Liquidity Crunch ».

    Simule une crise de liquidité en augmentant simultanément :
      - La corrélation systémique ρ (contagion inter-actifs accrue).
      - La LGD via un multiplicateur (baisse des taux de recouvrement
        due à des ventes forcées et une illiquidité du marché).

    Ce type de stress test est conforme aux recommandations du Comité
    de Bâle (BCBS 239) sur les tests de résistance des portefeuilles
    de crédit.

    Parameters
    ----------
    lgd : np.ndarray
        Vecteur LGD de base du portefeuille.
    ead : np.ndarray
        Vecteur des expositions (EAD).
    default_threshold : np.ndarray
        Seuils de défaut Φ⁻¹(PD_i).
    cfg : dict
        Configuration chargée depuis config.yaml.

    Returns
    -------
    losses : np.ndarray
        Vecteur des pertes totales simulées sous stress.
    """
    rho_stress = cfg["correlation"]["rho_stress"]
    lgd_mult = cfg["stress_test"]["lgd_multiplier"]
    stress_label = cfg["stress_test"]["label"]
    n_sim = cfg["simulation"]["n_simulations"]
    conf = cfg["simulation"]["confidence_level"]

    stressed_lgd = np.clip(lgd * lgd_mult, 0.0, 1.0)
    return run_simulation(
        rho=rho_stress,
        lgd_values=stressed_lgd,
        ead=ead,
        default_threshold=default_threshold,
        n_simulations=n_sim,
        confidence=conf,
        label=f"STRESS CASE — {stress_label} (rho={rho_stress}, LGD×{lgd_mult})",
    )


# =====================================================================
#  Main
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)

    cfg = load_config()
    n_sim = cfg["simulation"]["n_simulations"]
    conf = cfg["simulation"]["confidence_level"]
    rho_base = cfg["correlation"]["rho_base"]
    rho_stress = cfg["correlation"]["rho_stress"]
    portfolio_file = cfg["portfolio"]["file"]

    portfolio, ead, pd_values, lgd, default_threshold = load_portfolio(portfolio_file)

    # --- Scénario de base ---
    base_losses = run_simulation(
        rho=rho_base, lgd_values=lgd, ead=ead,
        default_threshold=default_threshold,
        n_simulations=n_sim, confidence=conf,
        label=f"BASE CASE (rho={rho_base})",
    )

    # --- Scénario de stress ---
    stress_losses = run_stress_test(lgd, ead, default_threshold, cfg)

    # --- Graphique comparatif ---
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(base_losses / 1e6, bins=200, density=True, alpha=0.5,
            label=f"Base Case (ρ={rho_base})", color="#2196F3")
    ax.hist(stress_losses / 1e6, bins=200, density=True, alpha=0.5,
            label=f"Stress Case (ρ={rho_stress}, LGD×{cfg['stress_test']['lgd_multiplier']})",
            color="#E53935")

    var_base = np.percentile(base_losses, conf * 100) / 1e6
    var_stress = np.percentile(stress_losses, conf * 100) / 1e6
    ax.axvline(var_base, color="#1565C0", linestyle="--", linewidth=1.5,
               label=f"VaR {conf:.0%} Base: {var_base:,.1f} M€")
    ax.axvline(var_stress, color="#B71C1C", linestyle="--", linewidth=1.5,
               label=f"VaR {conf:.0%} Stress: {var_stress:,.1f} M€")

    ax.set_xlabel("Total portfolio loss (M€)")
    ax.set_ylabel("Density")
    ax.set_title("Loss distribution — Base Case vs Stress Case")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("stress_test_comparison.png", dpi=150)
    print("\nGraphique sauvegardé : stress_test_comparison.png")
