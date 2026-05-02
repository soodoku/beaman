"""
Pure replication of Tables 1, 2, and 3 from:

    Beaman, L., Duflo, E., Pande, R., & Topalova, P. (2012).
    "Female Leadership Raises Aspirations and Educational Attainment for Girls:
    A Policy Experiment in India." Science, 335(6068), 582-586.
    https://doi.org/10.1126/science.1212382

This script reproduces the core findings:
  - Table 1: Parents' aspirations for their children
  - Table 2: Adolescents' aspirations
  - Table 3: Educational outcomes and time use

Outputs:
  - tabs/table2_baselines.md
  - tabs/table2_coefficients.md
  - tabs/table3_baselines.md
  - tabs/table3_time_use.md
  - figs/gender_gap_by_treatment.png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from codebook import EDU_OUTCOMES, OUTCOMES
from utils import (
    fit,
    gap_components,
    hr,
    load_and_merge,
    save_figure,
    write_table,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Table 1: Parents' aspirations
# ---------------------------------------------------------------------------

PARENT_OUTCOMES = [
    "want_daughter_graduate",
    "want_daughter_marry_after18",
    "want_daughter_pradhan",
]

PARENT_PAPER_BASELINES = {
    "want_daughter_graduate": (0.318, 0.176),
    "want_daughter_marry_after18": (0.992, 0.756),
    "want_daughter_pradhan": (0.688, 0.588),
}


def print_table1(adult: pd.DataFrame) -> None:
    """Reproduce Table 1: Parents' aspirations for their children."""
    hr("TABLE 1: Parents' Aspirations for Their Children")

    women = adult[adult["Female"] == 1].copy()
    n_gp = women["AA0_2b"].nunique()
    print(f"Sample: {len(women):,} female adults (mothers) in {n_gp} GPs")

    nr = women[women["never_res"] == 1]

    print("\nPanel A: Never-reserved baselines (paper vs. reproduction)")
    print(f"{'Outcome':<32}{'Boys (paper)':<15}{'Boys (mine)':<15}"
          f"{'Girls (paper)':<15}{'Girls (mine)':<15}")
    print("-" * 92)

    for y in PARENT_OUTCOMES:
        if y not in women.columns:
            continue
        pb, pg = PARENT_PAPER_BASELINES.get(y, (np.nan, np.nan))

        boys_mean = nr["want_son_graduate"].mean() if y == "want_daughter_graduate" else np.nan
        girls_mean = nr[y].mean()

        print(f"{y:<32}{pb:<15.3f}{boys_mean if not np.isnan(boys_mean) else 'N/A':<15}"
              f"{pg:<15.3f}{girls_mean:<15.3f}")

    print("\nPanel B: Treatment effects on parents' aspirations for DAUGHTERS")
    print("(Women-only sample, OLS with GP-clustered SEs)")
    print(f"\n{'Outcome':<32}{'Once (SE)':<20}{'Twice (SE)':<20}")
    print("-" * 72)

    has_data = False
    for y in PARENT_OUTCOMES:
        if y not in women.columns:
            continue
        use = women.dropna(subset=[y])
        if len(use) < 10:
            continue
        has_data = True
        use = use.copy()
        use["Once"] = use["once_res"]
        use["Twice"] = use["twice_res"]

        formula = f"{y} ~ Once + Twice"
        m = smf.ols(formula, data=use).fit(
            cov_type="cluster", cov_kwds={"groups": use["AA0_2b"]}
        )

        b_once = m.params["Once"]
        se_once = m.bse["Once"]
        b_twice = m.params["Twice"]
        se_twice = m.bse["Twice"]

        star_once = "*" if m.pvalues["Once"] < 0.05 else ""
        star_twice = "*" if m.pvalues["Twice"] < 0.05 else ""

        print(f"{y:<32}{b_once:+.3f} ({se_once:.3f}){star_once:<4}   "
              f"{b_twice:+.3f} ({se_twice:.3f}){star_twice:<4}")

    if not has_data:
        print("  (Parent aspiration data not available in this dataset)")
        print("  Note: A2_* variables may be in a separate module not included here.")


# ---------------------------------------------------------------------------
# Table 2: Adolescents' aspirations
# ---------------------------------------------------------------------------

PAPER_BASELINES = {
    "no_housewife": (0.998, 0.600),
    "wish_graduate": (0.296, 0.195),
    "marry_after18": (0.980, 0.660),
    "wish_pradhan": (0.499, 0.485),
}


def print_table2(teen: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Table 2: Adolescents' aspirations."""
    hr(f"TABLE 2: Adolescents' Aspirations\n"
       f"N = {len(teen):,} adolescents in {teen['AA0_2b'].nunique()} GPs.  "
       f"Twice-reserved cell: {(teen['twice_res']==1).sum()} adolescents in "
       f"{teen[teen['twice_res']==1]['AA0_2b'].nunique()} GPs.")

    nr = teen[teen["never_res"] == 1]

    print("\nPanel A: Never-reserved baselines (paper vs. reproduction)")
    print(f"{'Outcome':<18}{'Boys (paper)':<15}{'Boys (mine)':<15}"
          f"{'Girls (paper)':<15}{'Girls (mine)':<15}")
    print("-" * 78)

    baseline_rows = []
    for y in OUTCOMES:
        boys_mean = nr[nr["boy"] == 1][y].mean()
        girls_mean = nr[nr["girl"] == 1][y].mean()
        pb, pg = PAPER_BASELINES[y]
        print(f"{y:<18}{pb:<15.3f}{boys_mean:<15.3f}{pg:<15.3f}{girls_mean:<15.3f}")
        baseline_rows.append({
            "Outcome": y,
            "Boys (paper)": f"{pb:.3f}",
            "Boys (reproduced)": f"{boys_mean:.3f}",
            "Girls (paper)": f"{pg:.3f}",
            "Girls (reproduced)": f"{girls_mean:.3f}",
        })

    baseline_df = pd.DataFrame(baseline_rows)
    write_table(baseline_df, "table2_baselines", "Table 2: Adolescent Aspirations - Baselines")

    print("\nPanel B: Treatment effects (cluster-robust SEs at GP level)")
    print(f"{'Outcome':<18}{'Cycles':<8}{'Boys (SE)':<22}{'Girls (SE)':<22}{'Gap diff (SE)':<22}")
    print("-" * 92)

    coef_rows = []
    for y in OUTCOMES:
        m = fit(teen, y)
        comp = gap_components(m)
        for cycles in ("once", "twice"):
            (bb, sb), (bg, sg), (bd, sd) = (
                comp[cycles]["boys"],
                comp[cycles]["girls"],
                comp[cycles]["diff"],
            )
            stars_b = "*" if abs(bb / sb) > 1.96 else ""
            stars_g = "*" if abs(bg / sg) > 1.96 else ""
            stars_d = "*" if abs(bd / sd) > 1.96 else ""
            label = "Once" if cycles == "once" else "Twice"
            print(
                f"{y:<18}{label:<8}"
                f"{bb:+.3f} ({sb:.3f}){stars_b:<3}     "
                f"{bg:+.3f} ({sg:.3f}){stars_g:<3}     "
                f"{bd:+.3f} ({sd:.3f}){stars_d:<3}"
            )
            coef_rows.append({
                "Outcome": y,
                "Cycles": label,
                "Boys": f"{bb:+.3f} ({sb:.3f}){stars_b}",
                "Girls": f"{bg:+.3f} ({sg:.3f}){stars_g}",
                "Gap diff": f"{bd:+.3f} ({sd:.3f}){stars_d}",
            })
        print()

    coef_df = pd.DataFrame(coef_rows)
    write_table(
        coef_df, "table2_coefficients", "Table 2: Adolescent Aspirations - Treatment Effects"
    )

    return teen


def plot_gender_gap(teen: pd.DataFrame) -> None:
    """Plot gender gap by treatment cell for each outcome."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, y in enumerate(OUTCOMES):
        ax = axes[i]

        cells = []
        for cell, label in [("never_res", "Never"), ("once_res", "Once"), ("twice_res", "Twice")]:
            sub = teen[teen[cell] == 1]
            boys = sub[sub["boy"] == 1][y].mean()
            girls = sub[sub["girl"] == 1][y].mean()
            gap = girls - boys
            cells.append((label, boys, girls, gap))

        labels = [c[0] for c in cells]
        boys_vals = [c[1] for c in cells]
        girls_vals = [c[2] for c in cells]

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width/2, boys_vals, width, label="Boys", color="#4C72B0")
        ax.bar(x + width/2, girls_vals, width, label="Girls", color="#DD8452")

        ax.set_ylabel("Mean")
        ax.set_title(y)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 1.1)

    fig.suptitle("Adolescent Aspirations by Treatment Cell and Gender", fontsize=12)
    plt.tight_layout()
    save_figure(fig, "gender_gap_by_treatment")


# ---------------------------------------------------------------------------
# Table 3: Educational outcomes and time use
# ---------------------------------------------------------------------------

EDU_OUTCOMES_LABELS = [
    ("attends_school", "Attends school"),
    ("can_read_write", "Can read and write"),
    ("grade_completed", "Grade completed"),
]

PAPER_EDU_BASELINES = {
    "attends_school": (0.744, 0.684),
    "can_read_write": (0.947, 0.908),
    "grade_completed": (5.538, 5.409),
    "time_domestic": (29.298, 108.231),
}


def build_edu_outcomes(teen: pd.DataFrame) -> pd.DataFrame:
    """Build educational outcomes from household roster data.

    Column mappings from roster:
    - A1_8: Attending school/SSK/MSK/anganwadi (1=yes, 2=no)
    - A1_5: Can read and/or write (1=cannot, 2=read only, 3=read and write)
    - A1_7: Highest education achieved (grade level; 16=NA/not applicable)
    """
    teen = teen.copy()

    if "A1_8" in teen.columns:
        teen["attends_school"] = (teen["A1_8"] == 1).astype(int)
        teen.loc[teen["A1_8"].isna(), "attends_school"] = np.nan
    else:
        teen["attends_school"] = np.nan

    if "A1_5" in teen.columns:
        teen["can_read_write"] = (teen["A1_5"] == 3).astype(int)
        teen.loc[teen["A1_5"].isna(), "can_read_write"] = np.nan
    else:
        teen["can_read_write"] = np.nan

    if "A1_7" in teen.columns:
        teen["grade_completed"] = teen["A1_7"].replace([16, 99, 999], np.nan)
    else:
        teen["grade_completed"] = np.nan

    chore_cols = []
    for col in ["C1_2", "C1_3", "C1_7", "C1_8", "C1_9", "C1_10"]:
        if col in teen.columns:
            time_col = col.replace("C1_", "C2_")
            if time_col in teen.columns:
                chore_cols.append(time_col)

    if chore_cols:
        teen["time_domestic"] = teen[chore_cols].fillna(0).sum(axis=1)
    else:
        teen["time_domestic"] = np.nan

    return teen


def print_table3(teen: pd.DataFrame) -> None:
    """Reproduce Table 3: Educational outcomes and time use."""
    hr("TABLE 3: Adolescents' Educational Outcomes and Time Use")

    teen = build_edu_outcomes(teen)
    nr = teen[teen["never_res"] == 1]

    print("\nPanel A: Never-reserved baselines (paper vs. reproduction)")
    print(f"{'Outcome':<25}{'Boys (paper)':<15}{'Boys (mine)':<15}"
          f"{'Girls (paper)':<15}{'Girls (mine)':<15}")
    print("-" * 85)

    baseline_rows = []
    for col, label in EDU_OUTCOMES_LABELS + [("time_domestic", "Time on chores (min)")]:
        if col not in teen.columns or teen[col].isna().all():
            print(f"{label:<25}(not available in merged data)")
            continue
        boys_mean = nr[nr["boy"] == 1][col].mean()
        girls_mean = nr[nr["girl"] == 1][col].mean()
        pb, pg = PAPER_EDU_BASELINES.get(col, (np.nan, np.nan))
        print(f"{label:<25}{pb:<15.3f}{boys_mean:<15.3f}{pg:<15.3f}{girls_mean:<15.3f}")
        baseline_rows.append({
            "Outcome": label,
            "Boys (paper)": f"{pb:.3f}",
            "Boys (reproduced)": f"{boys_mean:.3f}",
            "Girls (paper)": f"{pg:.3f}",
            "Girls (reproduced)": f"{girls_mean:.3f}",
        })

    if baseline_rows:
        baseline_df = pd.DataFrame(baseline_rows)
        write_table(baseline_df, "table3_baselines", "Table 3: Educational Outcomes - Baselines")

    print("\nPanel B: Education treatment effects (gap-difference specification)")
    print("y ~ Once + Twice + Female + Once:Female + Twice:Female")
    print("Twice:Female = gender gap closure effect")
    print(f"\n{'Outcome':<25}{'Boys NR':<12}{'Girls NR':<12}{'Gap NR':<12}"
          f"{'Twice:Fem (SE)':<22}{'p-value':<10}")
    print("-" * 93)

    edu_rows = []
    for y in EDU_OUTCOMES:
        if y not in teen.columns or teen[y].isna().all():
            print(f"{y:<25}(not available)")
            continue

        boys_nr = nr[nr["boy"] == 1][y].mean()
        girls_nr = nr[nr["girl"] == 1][y].mean()
        gap_nr = girls_nr - boys_nr

        df = teen.dropna(subset=[y]).copy()
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        m = smf.ols(
            f"{y} ~ Once + Twice + Female + Once:Female + Twice:Female", data=df
        ).fit(cov_type="cluster", cov_kwds={"groups": df["AA0_2b"]})

        bd = m.params["Twice:Female"]
        sd = m.bse["Twice:Female"]
        p = m.pvalues["Twice:Female"]
        star = "*" if p < 0.05 else ""

        print(f"{y:<25}{boys_nr:<12.3f}{girls_nr:<12.3f}{gap_nr:<+12.3f}"
              f"{bd:+.3f} ({sd:.3f}){star:<4}{p:.3f}")
        edu_rows.append({
            "Outcome": y,
            "Boys (NR)": f"{boys_nr:.3f}",
            "Girls (NR)": f"{girls_nr:.3f}",
            "Gap (NR)": f"{gap_nr:+.3f}",
            "Twice:Female": f"{bd:+.3f} ({sd:.3f}){star}",
            "p-value": f"{p:.3f}",
        })

    if edu_rows:
        edu_df = pd.DataFrame(edu_rows)
        write_table(edu_df, "table3_education", "Table 3: Educational Outcomes - Treatment Effects")

    print("\nPanel C: Time use treatment effects")
    activities = [
        ("C1_1", "Went to school yesterday"),
        ("C1_3", "Helped to cook"),
        ("C1_7", "Cleaned home"),
        ("C1_8", "Cleaned clothes"),
        ("C1_9", "Cared for kids/elderly"),
        ("C1_10", "Gathered fuel/firewood"),
    ]

    print(f"\n{'Activity':<30}{'Boys NR':<12}{'Girls NR':<12}{'Twice:Fem':<18}{'p-value':<10}")
    print("-" * 82)

    time_rows = []
    for q, lbl in activities:
        if q not in teen.columns:
            continue
        teen[f"{q}_yes"] = (teen[q] == 1).astype(int)
        nr = teen[teen["never_res"] == 1]
        bm = nr[nr["boy"] == 1][f"{q}_yes"].mean()
        gm = nr[nr["girl"] == 1][f"{q}_yes"].mean()

        df = teen.dropna(subset=[f"{q}_yes"]).copy()
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
        time_rows.append({
            "Activity": lbl,
            "Boys (NR)": f"{bm:.3f}",
            "Girls (NR)": f"{gm:.3f}",
            "Twice:Female": f"{bd:+.3f} ({sd:.3f}){star}",
            "p-value": f"{p:.3f}",
        })

    if time_rows:
        time_df = pd.DataFrame(time_rows)
        write_table(time_df, "table3_time_use", "Table 3: Time Use - Treatment Effects")


# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------


def print_summary(teen: pd.DataFrame, adult: pd.DataFrame | None) -> None:
    """Print a summary comparison with paper's main findings."""
    hr("REPLICATION SUMMARY")

    print("\nPaper's headline finding (Table 2, normalized average):")
    print("  - Gender gap in adolescent aspirations: -0.507 SD in never-reserved")
    print("  - Gap difference in twice-reserved: +0.166 SD (p<0.02)")
    print("  - This represents a 32% closure of the gender gap")

    teen_idx = teen.dropna(subset=OUTCOMES).copy()
    nr = teen_idx[teen_idx["never_res"] == 1]

    for y in OUTCOMES:
        mu = nr[y].mean()
        sd = nr[y].std()
        if sd > 0:
            teen_idx[f"{y}_z"] = (teen_idx[y] - mu) / sd

    z_cols = [f"{y}_z" for y in OUTCOMES]
    teen_idx["idx"] = teen_idx[z_cols].mean(axis=1)
    teen_idx["Female"] = teen_idx["girl"]
    teen_idx["Once"] = teen_idx["once_res"]
    teen_idx["Twice"] = teen_idx["twice_res"]

    m = smf.ols("idx ~ Once + Twice + Female + Once:Female + Twice:Female", data=teen_idx).fit(
        cov_type="cluster", cov_kwds={"groups": teen_idx["AA0_2b"]}
    )

    print("\nReproduced normalized average:")
    print(f"  - Gap in never-reserved (Female coef): {m.params['Female']:.3f}")
    print(f"  - Gap difference in twice-reserved: {m.params['Twice:Female']:+.3f} "
          f"(SE={m.bse['Twice:Female']:.3f}, p={m.pvalues['Twice:Female']:.3f})")

    gap_nr = m.params["Female"]
    gap_diff = m.params["Twice:Female"]
    if gap_nr != 0:
        closure = abs(gap_diff / gap_nr) * 100
        print(f"  - Gap closure: {closure:.1f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
        "--tables",
        type=int,
        nargs="*",
        choices=[1, 2, 3],
        help="Which tables to reproduce (default: all).",
    )
    args = parser.parse_args()

    tables_to_run = args.tables if args.tables else [1, 2, 3]

    needs_adult = 1 in tables_to_run
    teen, adult = load_and_merge(args.data_dir, include_adult=needs_adult)

    if 1 in tables_to_run and adult is not None:
        print_table1(adult)

    if 2 in tables_to_run:
        print_table2(teen)
        plot_gender_gap(teen)

    if 3 in tables_to_run:
        print_table3(teen)

    print_summary(teen, adult)


if __name__ == "__main__":
    main()
