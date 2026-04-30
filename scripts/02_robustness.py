"""
Inference robustness checks for Beaman et al. (2012).

Tests whether the statistical conclusions survive proper inference:
  1. Wild cluster bootstrap p-values (critical for 20-GP twice-reserved cell)
  2. Multiple testing corrections (Bonferroni, Benjamini-Hochberg)
  3. Alternative clustering (GP vs. village vs. household)
  4. Randomization inference / permutation tests
  5. Joint test across outcomes

These are standard referee concerns for any diff-in-diff with few clusters.

Outputs:
  - tabs/robustness_wild_bootstrap.md
  - tabs/robustness_multiple_testing.md
  - tabs/robustness_clustering.md
  - tabs/robustness_randomization.md
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from codebook import FORMULA, OUTCOMES
from utils import (
    fit,
    hr,
    load_and_merge,
    wild_cluster_bootstrap,
    write_table,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Test 1: Wild cluster bootstrap
# ---------------------------------------------------------------------------


def test_wild_bootstrap(teen: pd.DataFrame, B: int = 999) -> None:
    """Wild cluster bootstrap for the twice-reserved gap difference."""
    hr(f"TEST: Wild Cluster Bootstrap (B={B})")
    print("\nWith only 20 GPs in the twice-reserved cell, asymptotic cluster")
    print("inference is unreliable; this is the recommended small-cluster fix.")
    print("(Cameron, Gelbach, & Miller 2008)")

    print(f"\n{'Outcome':<18}{'Coef':<10}{'SE_GP':<10}{'p (cluster)':<14}{'p (wild)':<14}")
    print("-" * 66)

    rows = []
    for y in OUTCOMES:
        df = teen.dropna(subset=[y]).copy()
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        m = fit(teen, y)
        coef = m.params["Twice:Female"]
        se = m.bse["Twice:Female"]
        p_cl = m.pvalues["Twice:Female"]

        _, p_wb, _ = wild_cluster_bootstrap(df, FORMULA.format(y=y), B=B)
        print(f"{y:<18}{coef:+.3f}     {se:.3f}     {p_cl:.3f}         {p_wb:.3f}")

        rows.append({
            "Outcome": y,
            "Coefficient": f"{coef:+.3f}",
            "SE (cluster)": f"{se:.3f}",
            "p (cluster)": f"{p_cl:.3f}",
            "p (wild bootstrap)": f"{p_wb:.3f}",
        })

    df_out = pd.DataFrame(rows)
    write_table(df_out, "robustness_wild_bootstrap", "Wild Cluster Bootstrap P-values")


# ---------------------------------------------------------------------------
# Test 2: Multiple testing correction
# ---------------------------------------------------------------------------


def test_multiple_testing(teen: pd.DataFrame) -> None:
    """Multiple testing correction for the four outcomes."""
    hr("TEST: Multiple Testing Correction")
    print("\nBonferroni: multiply p by number of tests")
    print("Benjamini-Hochberg: control false discovery rate")

    rs = []
    for y in OUTCOMES:
        m = fit(teen, y)
        rs.append((y, m.params["Twice:Female"], m.pvalues["Twice:Female"]))

    rs.sort(key=lambda x: x[2])
    ps = sorted(p for _, _, p in rs)

    print(f"\n{'Outcome':<18}{'Coef':<10}{'p (raw)':<10}{'p (Bonf)':<12}{'p (BH)':<10}")
    print("-" * 60)

    rows = []
    for y, b, p in rs:
        bonf = min(p * len(rs), 1.0)
        bh = min(p * len(rs) / (ps.index(p) + 1), 1.0)
        print(f"{y:<18}{b:+.3f}     {p:.3f}     {bonf:.3f}        {bh:.3f}")
        rows.append({
            "Outcome": y,
            "Coefficient": f"{b:+.3f}",
            "p (raw)": f"{p:.3f}",
            "p (Bonferroni)": f"{bonf:.3f}",
            "p (BH)": f"{bh:.3f}",
        })

    df_out = pd.DataFrame(rows)
    write_table(df_out, "robustness_multiple_testing", "Multiple Testing Corrections")


# ---------------------------------------------------------------------------
# Test 3: Alternative clustering
# ---------------------------------------------------------------------------


def test_clustering(teen: pd.DataFrame) -> None:
    """Sensitivity to clustering choice: GP vs. household vs. village."""
    hr("TEST: Clustering Choice (GP / Household / Village)")
    print("\nTreatment assigned at GP level, so GP clustering is correct.")
    print("HH and village shown for comparison.")

    print(f"\n{'Outcome':<18}{'Coef':<10}{'SE(GP)':<14}{'SE(HH)':<14}{'SE(Village)':<14}")
    print("-" * 70)

    rows = []
    for y in OUTCOMES:
        df = teen.dropna(subset=[y]).copy()
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        f = FORMULA.format(y=y)
        m_gp = smf.ols(f, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["AA0_2b"]}
        )
        m_hh = smf.ols(f, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["hhserialid"]}
        )
        m_v = smf.ols(f, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["village_id"]}
        )

        coef = m_gp.params["Twice:Female"]
        print(
            f"{y:<18}{coef:+.3f}     {m_gp.bse['Twice:Female']:.3f}        "
            f"{m_hh.bse['Twice:Female']:.3f}        {m_v.bse['Twice:Female']:.3f}"
        )
        rows.append({
            "Outcome": y,
            "Coefficient": f"{coef:+.3f}",
            "SE (GP)": f"{m_gp.bse['Twice:Female']:.3f}",
            "SE (Household)": f"{m_hh.bse['Twice:Female']:.3f}",
            "SE (Village)": f"{m_v.bse['Twice:Female']:.3f}",
        })

    df_out = pd.DataFrame(rows)
    write_table(df_out, "robustness_clustering", "Clustering Sensitivity")


# ---------------------------------------------------------------------------
# Test 4: Randomization inference
# ---------------------------------------------------------------------------


def test_randomization_inference(
    teen: pd.DataFrame, B: int = 999, seed: int = 42
) -> None:
    """GP-level treatment permutation test."""
    hr(f"TEST: Randomization Inference (B={B})")
    print("\nPermute treatment assignment at GP level, preserving cell sizes.")

    gp_trt = teen[["AA0_2b", "twice_res", "once_res", "never_res"]].drop_duplicates(
        "AA0_2b"
    ).reset_index(drop=True)

    n_never = int(gp_trt["never_res"].sum())
    n_once = int(gp_trt["once_res"].sum())
    n_twice = int(gp_trt["twice_res"].sum())

    rng = np.random.default_rng(seed)

    print(f"\nGP cell sizes: Never={n_never}, Once={n_once}, Twice={n_twice}")
    print(f"\n{'Outcome':<18}{'Coef':<10}{'p (perm)':<12}{'5%-95% null'}")
    print("-" * 60)

    rows = []
    for y in OUTCOMES:
        df = teen.dropna(subset=[y]).copy()
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        m_obs = smf.ols(FORMULA.format(y=y), data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["AA0_2b"]}
        )
        obs = m_obs.params["Twice:Female"]

        gps = gp_trt["AA0_2b"].values
        null = []

        for _ in range(B):
            perm = rng.permutation(len(gps))
            once_p = np.zeros(len(gps))
            twice_p = np.zeros(len(gps))
            once_p[perm[n_never : n_never + n_once]] = 1
            twice_p[perm[n_never + n_once :]] = 1

            gp_perm = pd.DataFrame({"AA0_2b": gps, "_o": once_p, "_t": twice_p})
            dp = df.merge(gp_perm, on="AA0_2b", how="left").copy()
            dp["Once"] = dp["_o"]
            dp["Twice"] = dp["_t"]

            try:
                mp = smf.ols(FORMULA.format(y=y), data=dp).fit()
                null.append(mp.params["Twice:Female"])
            except Exception:
                continue

        null = np.asarray(null)
        p_perm = float((np.abs(null) >= np.abs(obs)).mean())
        lo, hi = np.percentile(null, 5), np.percentile(null, 95)

        print(f"{y:<18}{obs:+.3f}     {p_perm:.3f}      [{lo:+.3f}, {hi:+.3f}]")
        rows.append({
            "Outcome": y,
            "Coefficient": f"{obs:+.3f}",
            "p (permutation)": f"{p_perm:.3f}",
            "95% null CI": f"[{lo:+.3f}, {hi:+.3f}]",
        })

    df_out = pd.DataFrame(rows)
    write_table(df_out, "robustness_randomization", "Randomization Inference")


# ---------------------------------------------------------------------------
# Test 5: Joint test across outcomes
# ---------------------------------------------------------------------------


def test_joint(teen: pd.DataFrame) -> None:
    """Stacked SUR-style estimation across all 4 outcomes."""
    hr("TEST: Joint Test (Stacked Estimation)")
    print("\nStack all 4 outcomes and estimate with outcome fixed effects.")

    df_long = []
    for y in OUTCOMES:
        sub = teen.dropna(subset=[y])[
            ["AA0_2b", "girl", "once_res", "twice_res", "hhserialid"]
        ].copy()
        sub["outcome_name"] = y
        sub["Y"] = teen.dropna(subset=[y])[y].values
        df_long.append(sub)

    df_long = pd.concat(df_long, ignore_index=True)
    df_long["Female"] = df_long["girl"]
    df_long["Once"] = df_long["once_res"]
    df_long["Twice"] = df_long["twice_res"]

    m = smf.ols(
        "Y ~ C(outcome_name) * (Once + Twice + Female + Once:Female + Twice:Female)",
        data=df_long,
    ).fit(cov_type="cluster", cov_kwds={"groups": df_long["AA0_2b"]})

    print(f"\nN observations (stacked across 4 outcomes): {len(df_long):,}")

    print(f"\nImplied Twice:Female effect at each outcome:")
    ref = sorted(df_long["outcome_name"].unique())[0]
    cov = m.cov_params()
    ref_eff = m.params["Twice:Female"]

    print(f"  {ref} (reference): {ref_eff:+.3f} ({m.bse['Twice:Female']:.3f})")

    for o in sorted(df_long["outcome_name"].unique()):
        if o == ref:
            continue
        inter = f"C(outcome_name)[T.{o}]:Twice:Female"
        if inter in m.params.index:
            b = ref_eff + m.params[inter]
            var_sum = (
                cov.loc["Twice:Female", "Twice:Female"]
                + cov.loc[inter, inter]
                + 2 * cov.loc["Twice:Female", inter]
            )
            se = float(np.sqrt(var_sum))
            print(f"  {o}: {b:+.3f} ({se:.3f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


ALL_TESTS = {
    1: ("wild-bootstrap", test_wild_bootstrap, "teen-B"),
    2: ("multiple-testing", test_multiple_testing, "teen"),
    3: ("clustering", test_clustering, "teen"),
    4: ("randomization-inf", test_randomization_inference, "teen-B"),
    5: ("joint-test", test_joint, "teen"),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the .dta files.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=999,
        help="Number of bootstrap and permutation draws.",
    )
    parser.add_argument(
        "--tests",
        type=int,
        nargs="*",
        help=f"Subset of tests to run (1-{len(ALL_TESTS)}). Default: all.",
    )
    args = parser.parse_args()

    tests_to_run = args.tests if args.tests else list(ALL_TESTS.keys())

    teen, _ = load_and_merge(args.data_dir, include_adult=False)

    for k in tests_to_run:
        if k not in ALL_TESTS:
            continue
        _, fn, kind = ALL_TESTS[k]
        if kind == "teen":
            fn(teen)
        elif kind == "teen-B":
            fn(teen, B=args.bootstrap_reps)


if __name__ == "__main__":
    main()
