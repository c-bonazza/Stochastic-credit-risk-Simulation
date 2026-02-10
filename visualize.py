"""
visualize.py — Interactive credit risk dashboard (Plotly)
=========================================================

Generates a standalone HTML dashboard showing:
    1. Comparative histogram of loss distributions (Base vs Stress).
    2. Summary table of metrics (EL, VaR, ES).
    3. Bar chart of aggregated exposure (EAD) by sector.

Simulations are executed via functions in simulator.py and parameters are
read from config.yaml.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator import load_config, load_portfolio, run_simulation, run_stress_test


def compute_metrics(losses, confidence):
    """Calcule les métriques de risque sur un vecteur de pertes simulées.

    Parameters
    ----------
    losses : np.ndarray
        Vecteur des pertes totales par scénario.
    confidence : float
        Niveau de confiance (ex. 0.99).

    Returns
    -------
    dict
        Dictionnaire avec Expected Loss, VaR et ES.
    """
    var = np.percentile(losses, confidence * 100)
    return {
        "Expected Loss": losses.mean(),
        "VaR 99%": var,
        "ES 99%": losses[losses >= var].mean(),
    }


def build_dashboard(portfolio, base_losses, stress_losses, cfg):
    """Construit et exporte le dashboard Plotly en HTML.

    Parameters
    ----------
    portfolio : pd.DataFrame
        DataFrame du portefeuille (colonnes Sector, EAD, etc.).
    base_losses : np.ndarray
        Pertes simulées — scénario de base.
    stress_losses : np.ndarray
        Pertes simulées — scénario de stress.
    cfg : dict
        Configuration chargée depuis config.yaml.
    """
    conf = cfg["simulation"]["confidence_level"]
    rho_base = cfg["correlation"]["rho_base"]
    rho_stress = cfg["correlation"]["rho_stress"]
    lgd_mult = cfg["stress_test"]["lgd_multiplier"]
    n_sim = cfg["simulation"]["n_simulations"]

    m_base = compute_metrics(base_losses, conf)
    m_stress = compute_metrics(stress_losses, conf)

    # --- Layout des sous-graphiques ---
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"colspan": 2, "type": "xy"}, None],
            [{"type": "xy"}, {"type": "domain"}],
        ],
        subplot_titles=(
            "Loss distribution — Base Case vs Stress Case",
            "Exposure (EAD) by sector",
            "",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
    )

    # --- 1. Histogramme interactif des pertes ---
    fig.add_trace(
        go.Histogram(
            x=base_losses / 1e6,
            nbinsx=200,
            name=f"Base Case (ρ={rho_base})",
            marker_color="#2196F3",
            opacity=0.6,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=stress_losses / 1e6,
            nbinsx=200,
            name=f"Stress Case (ρ={rho_stress}, LGD×{lgd_mult})",
            marker_color="#E53935",
            opacity=0.6,
        ),
        row=1, col=1,
    )

    fig.add_vline(
        x=m_base["VaR 99%"] / 1e6, line_dash="dash", line_color="#1565C0",
        line_width=2,
        annotation_text=f"VaR Base: {m_base['VaR 99%']/1e6:,.1f}M",
        annotation_position="top left", row=1, col=1,
    )
    fig.add_vline(
        x=m_stress["VaR 99%"] / 1e6, line_dash="dash", line_color="#B71C1C",
        line_width=2,
        annotation_text=f"VaR Stress: {m_stress['VaR 99%']/1e6:,.1f}M",
        annotation_position="top right", row=1, col=1,
    )

    fig.update_xaxes(title_text="Loss (millions)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)

    # --- 2. Bar chart EAD par secteur ---
    ead_sector = portfolio.groupby("Sector")["EAD"].sum().sort_values(ascending=True)

    fig.add_trace(
        go.Bar(
            x=ead_sector.values / 1e6,
            y=ead_sector.index,
            orientation="h",
            marker_color=["#26A69A", "#42A5F5", "#AB47BC", "#FFA726", "#EF5350"],
            name="EAD by sector",
            text=[f"{v:,.1f}M" for v in ead_sector.values / 1e6],
            textposition="outside",
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.update_xaxes(title_text="Total EAD (millions)", row=2, col=1)

    # --- 3. Table récapitulative ---
    fmt = lambda v: f"{v/1e6:,.2f} M€"

    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Base Case</b>",
                        "<b>Stress Case</b>", "<b>Change</b>"],
                fill_color="#37474F",
                font=dict(color="white", size=13),
                align="center",
                height=35,
            ),
            cells=dict(
                values=[
                    ["Expected Loss", "VaR 99%", "Expected Shortfall 99%"],
                    [fmt(m_base["Expected Loss"]), fmt(m_base["VaR 99%"]),
                     fmt(m_base["ES 99%"])],
                    [fmt(m_stress["Expected Loss"]), fmt(m_stress["VaR 99%"]),
                     fmt(m_stress["ES 99%"])],
                    [
                        f"+{(m_stress['Expected Loss']/m_base['Expected Loss']-1)*100:.0f}%",
                        f"×{m_stress['VaR 99%']/m_base['VaR 99%']:.1f}",
                        f"×{m_stress['ES 99%']/m_base['ES 99%']:.1f}",
                    ],
                ],
                fill_color=[["#ECEFF1"] * 3, ["#E3F2FD"] * 3,
                            ["#FFEBEE"] * 3, ["#FFF3E0"] * 3],
                font=dict(size=12),
                align="center",
                height=30,
            ),
        ),
        row=2, col=2,
    )

    # --- Layout global ---
    fig.update_layout(
        title=dict(
            text=(
                "<b>Monte Carlo Credit Risk Dashboard</b><br>"
                f"<sup>Portfolio of {len(portfolio):,} loans — "
                f"{n_sim:,} simulations — Vasicek model</sup>"
            ),
            x=0.5,
            font=dict(size=20),
        ),
        barmode="overlay",
        height=850,
        width=1300,
        template="plotly_white",
        legend=dict(x=0.55, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=100),
    )

    output_file = "dashboard.html"
    fig.write_html(output_file, include_plotlyjs=True)
    print(f"\nDashboard saved: {output_file}")


# =====================================================================
#  Main
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)

    cfg = load_config()
    n_sim = cfg["simulation"]["n_simulations"]
    conf = cfg["simulation"]["confidence_level"]
    rho_base = cfg["correlation"]["rho_base"]
    portfolio_file = cfg["portfolio"]["file"]

    portfolio, ead, pd_values, lgd, default_threshold = load_portfolio(portfolio_file)

    print("Simulation Base Case...")
    base_losses = run_simulation(
        rho=rho_base, lgd_values=lgd, ead=ead,
        default_threshold=default_threshold,
        n_simulations=n_sim, confidence=conf,
        label=f"BASE CASE (rho={rho_base})",
    )

    print("Simulation Stress Case...")
    stress_losses = run_stress_test(lgd, ead, default_threshold, cfg)

    build_dashboard(portfolio, base_losses, stress_losses, cfg)
