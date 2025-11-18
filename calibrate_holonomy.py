import pandas as pd
import numpy as np
import statsmodels.api as sm

# ============================================================
#  Non-SEMF QMU / Holonomy binding-length calibration
#
#  E_bind ≈ C0 * (B_k_p + B_k_n)
#           + C1 * (B_k_p - B_k_n)
#           + C2 * Φ_partial
#           + C3 * Φ_Coul
#           + C4 * A
#
#  where:
#    B_k_x      = ∑_s n_{s,x} (s-1)^2  (curvature index per species)
#    Φ_partial  = f_p^2 + f_n^2, f_x = n_last_x / cap_last_x
#    Φ_Coul     = Z(Z-1)/S_dom, S_dom = max(S_p, S_n)
#    A          = Z + N
#
#  No SEMF volume/surface/asym/pairing language — just holonomy indices.
# ============================================================

def get_shell_occupancy(nucleon_count: int):
    """
    Returns (S_dom, occupancy_dict) where occupancy_dict = {s: n_s}
    Shell capacities follow the holonomy rule:
        s = 1 : capacity = 2
        s ≥ 2 : capacity = 4*s - 2
    Cumulative: N_≤S = 2 S^2
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
    """Curvature index: ∑ n_s * (s-1)^2"""
    return sum(n * (s - 1)**2 for s, n in occupancy_dict.items())


def last_shell_fraction(occupancy_dict):
    """
    Return fractional occupancy of the last (highest) shell:
        f = n_last / cap_last
    where cap_last = 2 (s=1) or 4*s - 2 (s≥2).
    """
    if not occupancy_dict:
        return 0.0
    s_max = max(occupancy_dict.keys())
    n_last = occupancy_dict[s_max]
    cap_last = 2 if s_max == 1 else 4*s_max - 2
    if cap_last <= 0:
        return 0.0
    return n_last / cap_last


def build_holonomy_features(Z: int, N: int):
    """
    Build the non-SEMF holonomy feature set:

        H0 = curv_tot    = B_k_p + B_k_n
        H1 = curv_diff   = B_k_p - B_k_n
        H2 = partial_idx = f_p^2 + f_n^2
        H3 = coul_idx    = Z(Z-1)/S_dom
        H4 = A           = Z + N

    All are directly defined from holonomy shell structure and simple
    ledger counts; no "volume/surface/asymmetry/pairing" semantics.
    """
    Z = int(Z)
    N = int(N)
    A = Z + N

    S_p, occ_p = get_shell_occupancy(Z)
    S_n, occ_n = get_shell_occupancy(N)
    S_dom = max(S_p, S_n) if max(Z, N) > 0 else 1

    B_k_p = calculate_B_kappa(occ_p)
    B_k_n = calculate_B_kappa(occ_n)

    curv_tot = B_k_p + B_k_n
    curv_diff = B_k_p - B_k_n

    f_p = last_shell_fraction(occ_p)
    f_n = last_shell_fraction(occ_n)
    partial_idx = f_p**2 + f_n**2

    coul_idx = Z*(Z-1)/S_dom if (Z > 1 and S_dom > 0) else 0.0

    return {
        "curv_tot": curv_tot,
        "curv_diff": curv_diff,
        "partial_idx": partial_idx,
        "coul_idx": coul_idx,
        "A": float(A),
    }


def run_full_calibration(csv_path: str = "clean_binding_data.csv"):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} nuclei from {csv_path}")

    # Build feature matrix
    features = df.apply(
        lambda row: build_holonomy_features(row["Z"], row["N"]),
        axis=1
    )
    M = pd.DataFrame(features.tolist())
    y = df["BindingEnergy"].values

    # Holonomy-ledger model: OLS through origin
    # E_bind ≈ β · X  with X = [curv_tot, curv_diff, partial_idx, coul_idx, A]
    model = sm.OLS(y, M).fit()
    beta = model.params

    C0 = beta["curv_tot"]
    C1 = beta["curv_diff"]
    C2 = beta["partial_idx"]
    C3 = beta["coul_idx"]
    C4 = beta["A"]

    print("\n" + "="*72)
    print("  QMU / HOLONOMY BINDING-LENGTH FIT (MeV) – NON-SEMF MODEL")
    print("  (Global OLS through origin, holonomy-only invariants)")
    print("="*72)
    print("Model:")
    print("  E_bind(Z,N) ≈ C0 * (B_k_p + B_k_n)")
    print("                 + C1 * (B_k_p - B_k_n)")
    print("                 + C2 * (f_p^2 + f_n^2)")
    print("                 + C3 * [Z(Z-1)/S_dom]")
    print("                 + C4 * (Z+N)")
    print("where:")
    print("  B_k_x   = ∑_s n_{s,x} (s-1)^2")
    print("  f_x     = n_last_x / cap_last_x")
    print("  S_dom   = max(S_p, S_n)")
    print("------------------------------------------------------------")
    print(f"C0 (curv_tot)   : {C0:+.6f} MeV per index")
    print(f"C1 (curv_diff)  : {C1:+.6f} MeV per index")
    print(f"C2 (partial_idx): {C2:+.6f} MeV")
    print(f"C3 (coul_idx)   : {C3:+.6f} MeV")
    print(f"C4 (A-count)    : {C4:+.6f} MeV per nucleon")
    print("------------------------------------------------------------")
    print(f"R² (all nuclei) : {model.rsquared:.10f}")
    print(f"RMS deviation   : {np.sqrt(model.mse_resid):.6f} MeV")
    print("============================================================\n")

    _sanity_check(df, model)
    return model


def _sanity_check(df: pd.DataFrame, model):
    print("Sanity check – selected nuclei (MeV):")
    df_idx = df.set_index(["Z", "N"])

    test_nuclei = [
        ("⁴He",    2,   2),
        ("¹⁶O",    8,   8),
        ("⁴⁰Ca",  20,  20),
        ("⁴⁸Ca",  20,  28),
        ("⁵⁶Ni",  28,  28),
        ("¹⁰⁰Sn", 50,  50),
        ("¹³²Sn", 50,  82),
        ("²⁰⁸Pb", 82, 126),
    ]

    for name, Z, N in test_nuclei:
        if (Z, N) not in df_idx.index:
            continue
        feat = build_holonomy_features(Z, N)
        x = np.array([
            feat["curv_tot"],
            feat["curv_diff"],
            feat["partial_idx"],
            feat["coul_idx"],
            feat["A"],
        ])
        beta = model.params[["curv_tot", "curv_diff", "partial_idx", "coul_idx", "A"]].values
        pred = float(np.dot(beta, x))
        exp = float(df_idx.loc[(Z, N), "BindingEnergy"])
        print(f"{name:6}  exp {exp:8.3f}   pred {pred:8.3f}   Δ {pred-exp:8.3f}")


if __name__ == "__main__":
    run_full_calibration("clean_binding_data.csv")

