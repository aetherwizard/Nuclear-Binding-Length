import pandas as pd
import numpy as np
import statsmodels.api as sm

# ============================================================
#  Holonomy/QMU nuclear binding ledger and residual generator
# ============================================================

def get_shell_occupancy(nucleon_count: int):
    """
    Returns (S_dom, occupancy_dict) where occupancy_dict = {s: n_s}
    Shell capacities follow the holonomy rule:
        s = 1 : capacity = 2
        s >= 2: capacity = 4*s - 2
    Cumulative: N_<=S = 2*S**2
    """
    if nucleon_count <= 0:
        return 0, {}

    occupancy = {}
    remaining = nucleon_count
    s = 1
    while remaining > 0:
        cap = 2 if s == 1 else 4*s - 2
        fill = min(cap, remaining)
        occupancy[s] = fill
        remaining -= fill
        s += 1

    S = max(occupancy.keys()) if occupancy else 0
    return S, occupancy


def calculate_B_kappa(occupancy_dict):
    """Curvature index: sum_s n_s * (s-1)^2"""
    return sum(n * (s - 1)**2 for s, n in occupancy_dict.items())


def last_shell_fraction(occupancy_dict):
    """
    Return fractional occupancy of the last (highest) shell:
        f = n_last / cap_last
    where cap_last = 2 (s=1) or 4*s - 2 (s>=2).
    """
    if not occupancy_dict:
        return 0.0
    s_max = max(occupancy_dict.keys())
    n_last = occupancy_dict[s_max]
    cap_last = 2 if s_max == 1 else 4*s_max - 2
    if cap_last <= 0:
        return 0.0
    return n_last / cap_last


def build_holonomy_invariants(Z: int, N: int):
    """
    Compute all holonomy/QMU invariants for a given (Z,N):
      - B_k_p, B_k_n, B_k_tot, dB_k
      - f_p, f_n, Phi_partial
      - S_dom, Phi_Coul
      - A
    plus the feature vector used for the linear fit.
    """
    Z = int(Z)
    N = int(N)
    A = Z + N

    S_p, occ_p = get_shell_occupancy(Z)
    S_n, occ_n = get_shell_occupancy(N)
    S_dom = max(S_p, S_n) if max(Z, N) > 0 else 1

    B_k_p = calculate_B_kappa(occ_p)
    B_k_n = calculate_B_kappa(occ_n)
    B_k_tot = B_k_p + B_k_n
    dB_k = B_k_p - B_k_n

    f_p = last_shell_fraction(occ_p)
    f_n = last_shell_fraction(occ_n)
    partial_idx = f_p**2 + f_n**2

    coul_idx = Z*(Z-1)/S_dom if (Z > 1 and S_dom > 0) else 0.0

    return {
        "Z": Z,
        "N": N,
        "A": A,
        "B_k_p": B_k_p,
        "B_k_n": B_k_n,
        "B_k_tot": B_k_tot,
        "dB_k": dB_k,
        "f_p": f_p,
        "f_n": f_n,
        "partial_idx": partial_idx,
        "S_dom": S_dom,
        "coul_idx": coul_idx,
    }


def main(csv_path: str = "clean_binding_data.csv",
         out_path: str = "qmu_holonomy_ledger.csv"):

    # Load experimental binding energy ledger (MeV)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} nuclei from {csv_path}")

    # Build full invariants table
    invariants_list = []
    for _, row in df.iterrows():
        inv = build_holonomy_invariants(row["Z"], row["N"])
        inv["BindingEnergy"] = float(row["BindingEnergy"])
        invariants_list.append(inv)

    inv_df = pd.DataFrame(invariants_list)

    # Construct feature matrix for OLS fit
    M = inv_df[["B_k_tot", "dB_k", "partial_idx", "coul_idx", "A"]].copy()
    M.columns = ["curv_tot", "curv_diff", "partial_idx", "coul_idx", "A"]

    y = inv_df["BindingEnergy"].values

    # Fit holonomy binding-length model: E = beta * X
    model = sm.OLS(y, M).fit()
    beta = model.params

    print("\nFitted coefficients (MeV units):")
    print(beta)
    print(f"\nR^2  = {model.rsquared:.10f}")
    print(f"RMS  = {np.sqrt(model.mse_resid):.6f} MeV")

    # Predictions and residuals
    y_pred = model.predict(M)
    inv_df["E_exp"] = inv_df["BindingEnergy"]
    inv_df["E_pred"] = y_pred
    inv_df["Residual"] = inv_df["E_pred"] - inv_df["E_exp"]

    # Save full ledger for plotting
    inv_df.to_csv(out_path, index=False)
    print(f"\nWrote holonomy ledger with residuals to {out_path}")


if __name__ == "__main__":
    main()
