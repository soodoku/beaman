"""
Sensitivity analysis for Beaman et al. (2012).

Tests whether results are fragile to specification choices:
  1. Leave-one-out by GP (which GPs drive the result?)
  2. No-ceiling index (drop items where boys hit ceiling)
  3. Age placebo: adult women 35+ (socialized before reservations)
  4. Unused E items: 10 gender-attitude questions the paper didn't use
  5. Time use: did adolescents' chores change?

These go beyond standard robustness to ask "what if we did it differently?"

Outputs:
  - tabs/sensitivity_loo.md
  - tabs/sensitivity_ceiling.md
  - tabs/sensitivity_age_placebo.md
  - tabs/sensitivity_e_items.md
  - tabs/sensitivity_time_use.md
  - figs/loo_sensitivity.png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from codebook import E_LABELS, E_SIGNS, FORMULA, OUTCOMES
from utils import (
    hr,
    load_and_merge,
    make_index,
    save_figure,
    wild_cluster_bootstrap,
    write_table,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Test 1: Leave-one-out
# ---------------------------------------------------------------------------


def test_loo(teen: pd.DataFrame) -> None:
    """Leave-one-out for the 20 twice-reserved GPs."""
    hr("TEST: Leave-One-Out (Twice-Reserved GPs)")
    print("\nHow sensitive is each outcome to dropping single GPs?")

    twice_gps = teen[teen["twice_res"] == 1]["AA0_2b"].unique()

    print(f"\n{'Outcome':<20}{'Original p':<14}{'LOO p>0.05':<22}{'Worst p':<10}")
    print("-" * 66)

    rows = []
    loo_results = {}

    for y in OUTCOMES:
        df_full = teen.dropna(subset=[y]).copy()
        df_full["Female"] = df_full["girl"]
        df_full["Once"] = df_full["once_res"]
        df_full["Twice"] = df_full["twice_res"]

        m_full = smf.ols(FORMULA.format(y=y), data=df_full).fit(
            cov_type="cluster", cov_kwds={"groups": df_full["AA0_2b"]}
        )
        p_full = m_full.pvalues["Twice:Female"]

        ps = []
        for gp in twice_gps:
            d = df_full[df_full["AA0_2b"] != gp]
            try:
                m = smf.ols(FORMULA.format(y=y), data=d).fit(
                    cov_type="cluster", cov_kwds={"groups": d["AA0_2b"]}
                )
                ps.append(m.pvalues["Twice:Female"])
            except Exception:
                continue

        n_above = sum(1 for p in ps if p > 0.05)
        worst = max(ps) if ps else float("nan")

        print(f"{y:<20}{p_full:<14.3f}{n_above}/{len(ps):<22}{worst:.3f}")
        rows.append({
            "Outcome": y,
            "Original p": f"{p_full:.3f}",
            "LOO p > 0.05": f"{n_above}/{len(ps)}",
            "Worst p": f"{worst:.3f}",
        })
        loo_results[y] = ps

    df_out = pd.DataFrame(rows)
    write_table(df_out, "sensitivity_loo", "Leave-One-Out Sensitivity")

    plot_loo(loo_results, twice_gps)


def plot_loo(loo_results: dict, twice_gps) -> None:
    """Plot LOO p-values for each outcome."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, (y, ps) in enumerate(loo_results.items()):
        ax = axes[i]
        ax.scatter(range(len(ps)), sorted(ps), alpha=0.7)
        ax.axhline(0.05, color="red", linestyle="--", label="p=0.05")
        ax.set_xlabel("GP index (sorted by p-value)")
        ax.set_ylabel("p-value")
        ax.set_title(y)
        ax.legend()
        ax.set_ylim(0, max(0.3, max(ps) + 0.05) if ps else 0.3)

    fig.suptitle("Leave-One-Out P-values (Twice:Female coefficient)", fontsize=12)
    plt.tight_layout()
    save_figure(fig, "loo_sensitivity")


# ---------------------------------------------------------------------------
# Test 2: No-ceiling index
# ---------------------------------------------------------------------------


def test_no_ceiling_index(teen: pd.DataFrame, B: int = 999) -> None:
    """Index with and without ceiling-bound items."""
    hr("TEST: Index Without Ceiling-Bound Items")
    print("\nCeiling items: no_housewife (boys 0.998), marry_after18 (boys 0.980)")
    print("These items have little room for boys to improve.")

    combos = [
        (
            "All four (replication)",
            ["no_housewife", "wish_graduate", "marry_after18", "wish_pradhan"],
        ),
        ("Ceiling only", ["no_housewife", "marry_after18"]),
        ("Non-ceiling only", ["wish_graduate", "wish_pradhan"]),
    ]

    print(
        f"\n{'Index spec':<28}{'N':<8}{'Twice:Female':<18}{'p (cluster)':<14}{'p (wild)':<10}"
    )
    print("-" * 78)

    rows = []
    for label, cols in combos:
        df = teen.dropna(subset=cols).copy()
        df = make_index(df, cols)
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        formula = "idx ~ Once + Twice + Female + Once:Female + Twice:Female"
        m = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["AA0_2b"]}
        )

        coef = m.params["Twice:Female"]
        se = m.bse["Twice:Female"]
        p_cl = m.pvalues["Twice:Female"]

        _, p_wb, _ = wild_cluster_bootstrap(df, formula, B=B)

        print(f"{label:<28}{len(df):<8}{coef:+.3f} ({se:.3f})   {p_cl:.3f}         {p_wb:.3f}")
        rows.append({
            "Index specification": label,
            "N": str(len(df)),
            "Twice:Female (SE)": f"{coef:+.3f} ({se:.3f})",
            "p (cluster)": f"{p_cl:.3f}",
            "p (wild)": f"{p_wb:.3f}",
        })

    df_out = pd.DataFrame(rows)
    write_table(df_out, "sensitivity_ceiling", "Ceiling Effects Analysis")


# ---------------------------------------------------------------------------
# Test 3: Age placebo
# ---------------------------------------------------------------------------


def test_age_placebo(adult: pd.DataFrame) -> None:
    """Adult women's attitudes (E_1-E_10) by age group."""
    hr("TEST: Age Placebo (Women 35+ vs. Under 35)")
    print("\nIf 'role-model' channel works on adolescents, women aged 35+ should")
    print("be unaffected (they were socialized before 1993 reservation began).")

    old = adult[(adult["Female"] == 1) & (adult["age"] >= 35)].copy()
    young = adult[(adult["Female"] == 1) & (adult["age"] < 35)].copy()

    print(f"\nN women 35+: {len(old)}, women under 35: {len(young)}")

    all_rows = []
    for label, df in [("Women 35+", old), ("Women under 35", young)]:
        print(f"\n--- {label} ---")
        print(f"{'Item':<6}{'Description':<35}{'Once':<14}{'Twice':<14}")
        print("-" * 69)

        n_sig = 0
        for e in E_SIGNS:
            y = f"{e}_agree"
            d = df.dropna(subset=[y]).copy()
            d["Once"] = d["once_res"]
            d["Twice"] = d["twice_res"]

            if len(d) < 200:
                continue

            m = smf.ols(f"{y} ~ Once + Twice", data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["AA0_2b"]}
            )

            bo = m.params["Once"]
            so = m.bse["Once"]
            bt = m.params["Twice"]
            st = m.bse["Twice"]
            sig = "*" if m.pvalues["Twice"] < 0.05 else ""

            if m.pvalues["Twice"] < 0.05:
                n_sig += 1

            print(
                f"  {e:<6}{E_LABELS[e]:<35}{bo:+.3f}({so:.3f})  {bt:+.3f}({st:.3f}){sig}"
            )
            all_rows.append({
                "Age group": label,
                "Item": e,
                "Description": E_LABELS[e],
                "Once (SE)": f"{bo:+.3f} ({so:.3f})",
                "Twice (SE)": f"{bt:+.3f} ({st:.3f}){sig}",
            })

        print(f"  N significant at p<0.05: {n_sig}/10")

    df_out = pd.DataFrame(all_rows)
    write_table(df_out, "sensitivity_age_placebo", "Age Placebo (E Items by Age Group)")


# ---------------------------------------------------------------------------
# Test 4: Unused E items
# ---------------------------------------------------------------------------


def test_e_section(teen: pd.DataFrame) -> None:
    """Ten gender-attitude items (E_1-E_10) the paper did not use."""
    hr("TEST: Unused Gender-Attitude Items (E_1 to E_10)")
    print("\nThese 10 items were collected but not reported in the main paper.")
    print("Coding: agree (top-2) vs disagree (bottom-2 box).")
    print("Coefficient is Twice:Female gap difference.")

    print(f"\n{'Item':<6}{'Description':<40}{'Dir':<6}{'Twice gap-diff':<18}{'p':<8}")
    print("-" * 78)

    n_sig = 0
    rows = []
    for e in E_SIGNS:
        y = f"{e}_agree"
        df = teen.dropna(subset=[y]).copy()
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        m = smf.ols(FORMULA.format(y=y), data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["AA0_2b"]}
        )

        bd = m.params["Twice:Female"]
        sd = m.bse["Twice:Female"]
        p = m.pvalues["Twice:Female"]

        sign = "+" if E_SIGNS[e] > 0 else ("-" if E_SIGNS[e] < 0 else "P")
        star = "*" if p < 0.05 else ""

        if p < 0.05:
            n_sig += 1

        print(f"{e:<6}{E_LABELS[e]:<40}{sign:<6}{bd:+.3f} ({sd:.3f}){star:<3}   {p:.3f}")
        rows.append({
            "Item": e,
            "Description": E_LABELS[e],
            "Direction": sign,
            "Twice:Female (SE)": f"{bd:+.3f} ({sd:.3f}){star}",
            "p-value": f"{p:.3f}",
        })

    print(f"\nN of 10 significant at p<0.05: {n_sig} (expected ~0.5 under null)")

    df_out = pd.DataFrame(rows)
    write_table(df_out, "sensitivity_e_items", "Unused E Items (Teen Gender Attitudes)")


# ---------------------------------------------------------------------------
# Test 5: Time use
# ---------------------------------------------------------------------------


def test_time_use(teen: pd.DataFrame) -> None:
    """Did adolescents do each chore yesterday?"""
    hr("TEST: Time Use (Chores Yesterday)")
    print("\nDid treatment change how adolescents spend their time?")

    activities = [
        ("C1_1", "Went to school yest"),
        ("C1_3", "Helped to cook"),
        ("C1_7", "Cleaned home"),
        ("C1_8", "Cleaned clothes"),
        ("C1_9", "Cared for kids/elderly"),
        ("C1_10", "Gathered fuel/firewood"),
    ]

    print(
        f"\n{'Activity':<30}{'Boys NR':<12}{'Girls NR':<12}{'Twice:Fem coef':<20}{'p':<8}"
    )
    print("-" * 82)

    rows = []
    for q, lbl in activities:
        if q not in teen.columns:
            continue

        teen[f"{q}_yes"] = (teen[q] == 1).astype(int)
        nr = teen[teen["never_res"] == 1]
        bm = nr[nr["boy"] == 1][f"{q}_yes"].mean()
        gm = nr[nr["girl"] == 1][f"{q}_yes"].mean()

        df = teen.copy()
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        m = smf.ols(
            f"{q}_yes ~ Once + Twice + Female + Once:Female + Twice:Female", data=df
        ).fit(cov_type="cluster", cov_kwds={"groups": df["AA0_2b"]})

        bd = m.params["Twice:Female"]
        sd = m.bse["Twice:Female"]
        p = m.pvalues["Twice:Female"]
        star = "*" if p < 0.05 else ""

        print(f"  {lbl:<28}{bm:<12.3f}{gm:<12.3f}{bd:+.3f} ({sd:.3f}){star:<3} {p:.3f}")
        rows.append({
            "Activity": lbl,
            "Boys (NR)": f"{bm:.3f}",
            "Girls (NR)": f"{gm:.3f}",
            "Twice:Female (SE)": f"{bd:+.3f} ({sd:.3f}){star}",
            "p-value": f"{p:.3f}",
        })

    df_out = pd.DataFrame(rows)
    write_table(df_out, "sensitivity_time_use", "Time Use (Chores Yesterday)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


ALL_TESTS = {
    1: ("leave-one-out", test_loo, "teen"),
    2: ("no-ceiling-index", test_no_ceiling_index, "teen-B"),
    3: ("age-placebo", test_age_placebo, "adult"),
    4: ("unused-E-section", test_e_section, "teen"),
    5: ("time-use", test_time_use, "teen"),
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
        help="Number of bootstrap draws.",
    )
    parser.add_argument(
        "--tests",
        type=int,
        nargs="*",
        help=f"Subset of tests to run (1-{len(ALL_TESTS)}). Default: all.",
    )
    args = parser.parse_args()

    tests_to_run = args.tests if args.tests else list(ALL_TESTS.keys())
    needs_adult = any(ALL_TESTS[k][2] == "adult" for k in tests_to_run if k in ALL_TESTS)

    teen, adult = load_and_merge(args.data_dir, include_adult=needs_adult)

    for k in tests_to_run:
        if k not in ALL_TESTS:
            continue
        _, fn, kind = ALL_TESTS[k]
        if kind == "teen":
            fn(teen)
        elif kind == "teen-B":
            fn(teen, B=args.bootstrap_reps)
        elif kind == "adult":
            if adult is not None:
                fn(adult)
            else:
                print(f"Skipping test {k}: adult data not available")


if __name__ == "__main__":
    main()
