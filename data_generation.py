import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000

# --- Asset IDs ---
asset_ids = np.arange(1, N + 1)

# --- Sectors with distribution ---
sectors = ["Technology", "Energy", "Healthcare", "Real Estate", "Retail"]
weights = [0.25, 0.20, 0.20, 0.15, 0.20]
sector_assignments = np.random.choice(sectors, size=N, p=weights)

# --- EAD (Exposure at Default) : Log-normale entre 100k et 5M ---
# Paramètres log-normaux calibrés pour centrer autour de ~800k
mu_log, sigma_log = 13.2, 0.8
raw_ead = np.random.lognormal(mean=mu_log, sigma=sigma_log, size=N)
ead = np.clip(raw_ead, 100_000, 5_000_000)

# --- PD (Probability of Default) calibrated by sector ---
pd_params = {
    "Technology":   (0.020, 0.005),
    "Energy":       (0.025, 0.006),
    "Healthcare":   (0.018, 0.004),
    "Real Estate":  (0.040, 0.007),  # PD la plus élevée
    "Retail":       (0.030, 0.006),
}

# Génération vectorisée des PD par secteur (sans boucle for)
mean_pds = np.array([pd_params[s][0] for s in sector_assignments])
std_pds = np.array([pd_params[s][1] for s in sector_assignments])
pd_values = np.random.normal(mean_pds, std_pds)
pd_values = np.clip(pd_values, 0.01, 0.05)

# --- LGD (Loss Given Default): mean 0.40, std dev 0.05 ---
lgd_values = np.random.normal(loc=0.40, scale=0.05, size=N)
lgd_values = np.clip(lgd_values, 0.0, 1.0)

# --- Construction du DataFrame ---
portfolio = pd.DataFrame({
    "Asset_ID": asset_ids,
    "Sector": sector_assignments,
    "EAD": np.round(ead, 2),
    "PD": np.round(pd_values, 4),
    "LGD": np.round(lgd_values, 4),
})

# --- Save ---
portfolio.to_csv("portfolio.csv", index=False)

print(f"Generated portfolio: {len(portfolio)} loans")
print(f"Saved file: portfolio.csv\n")
print(portfolio.head(10))
print(f"\n--- Sector statistics ---")
print(portfolio.groupby("Sector")[ ["EAD", "PD", "LGD"] ].mean().round(4))
