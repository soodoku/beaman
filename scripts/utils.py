"""
Shared utilities for Beaman et al. (2012) replication.

This module contains:
- Data loading and merging functions
- Outcome construction for teens and adults
- Estimation primitives (OLS with cluster-robust SEs)
- Wild cluster bootstrap implementation
- Index construction
- Table and figure output utilities
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf

from codebook import (
    E_LABELS,
    E_SIGNS,
    FORMULA,
    HOUSEWIFE_STRINGS,
    OUTCOMES,
)

# Re-export for backwards compatibility
__all__ = [
    "E_LABELS",
    "E_SIGNS",
    "FORMULA",
    "HOUSEWIFE_STRINGS",
    "OUTCOMES",
]

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
TABS_DIR = ROOT / "tabs"
FIGS_DIR = ROOT / "figs"


def ensure_output_dirs() -> None:
    """Create tabs/ and figs/ directories if they don't exist."""
    TABS_DIR.mkdir(exist_ok=True)
    FIGS_DIR.mkdir(exist_ok=True)


def write_table(df: pd.DataFrame, filename: str, title: str | None = None) -> Path:
    """
    Write DataFrame to tabs/{filename}.md as markdown table.

    Parameters
    ----------
    df : DataFrame
        Table data
    filename : str
        Filename without extension
    title : str, optional
        Table title (added as H2 header)

    Returns
    -------
    Path : Path to written file
    """
    ensure_output_dirs()
    path = TABS_DIR / f"{filename}.md"

    lines = []
    if title:
        lines.append(f"## {title}\n")

    lines.append(df.to_markdown(index=False))
    lines.append("")

    path.write_text("\n".join(lines))
    return path


def save_figure(fig: plt.Figure, filename: str, dpi: int = 150) -> Path:
    """
    Save figure to figs/{filename}.png.

    Parameters
    ----------
    fig : matplotlib Figure
        Figure to save
    filename : str
        Filename without extension
    dpi : int
        Resolution (default 150)

    Returns
    -------
    Path : Path to written file
    """
    ensure_output_dirs()
    path = FIGS_DIR / f"{filename}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Outcome construction
# ---------------------------------------------------------------------------


def _binary_top2(s: pd.Series) -> np.ndarray:
    """Convert 5-point Likert to binary: top-2 = 1, bottom-2 = 0, else NaN."""
    return np.where(s.isin([1, 2]), 1.0, np.where(s.isin([4, 5]), 0.0, np.nan))


def build_teen_outcomes(teen: pd.DataFrame) -> pd.DataFrame:
    """Add outcome columns to the teenager dataframe."""
    teen = teen.copy()

    occ = teen["B1_3"].astype(str).str.strip()
    teen["no_housewife"] = (~occ.isin(HOUSEWIFE_STRINGS)).astype(int)

    age = teen["B1_2"].replace([99, 999], np.nan)
    teen["marry_after18"] = (age >= 19).astype(int)
    teen.loc[age.isna(), "marry_after18"] = np.nan

    teen["wish_graduate"] = (teen["B1_1"] == 13).astype(int)
    teen.loc[teen["B1_1"].isna(), "wish_graduate"] = np.nan

    teen["wish_pradhan"] = teen["B1_4"].map({1: 1, 2: 0})

    for e in E_SIGNS:
        teen[f"{e}_agree"] = _binary_top2(teen[e])

    return teen


def build_adult_outcomes(adult: pd.DataFrame) -> pd.DataFrame:
    """Add outcome columns to the adult dataframe."""
    adult = adult.copy()

    outcome_mappings = [
        ("approached", "D1_7", {1: 1, 2: 0}),
        ("easy_meet", "D1_9", {1: 1, 2: 0}),
        ("knows_pradhan", "D1_1a_rem", {1: 1, 2: 0, 999: 0}),
        ("knows_prev_pradhan", "D1_1b_rem", {1: 1, 2: 0, 999: 0}),
        ("knows_mla", "D1_1c_rem", {1: 1, 2: 0, 999: 0}),
        ("heard_gs", "D3_1", {1: 1, 2: 0}),
        ("spoke_gs", "D3_12", {1: 1, 2: 0}),
        ("voted_gp", "D1_3", {1: 1, 2: 0}),
        ("reads_paper", "D1_2", {1: 1, 2: 0}),
    ]

    for col, src, recode in outcome_mappings:
        adult[col] = adult[src].map(recode)

    attend_cols = ["D3_11_1", "D3_11_2", "D3_11_3", "D3_11_4", "D3_11_5", "D3_11_6"]
    adult["any_gs_activity"] = adult[attend_cols].apply(
        lambda r: 1.0 if any(v == 1 for v in r) else 0.0, axis=1
    )

    for e in E_SIGNS:
        adult[f"{e}_agree"] = _binary_top2(adult[e])

    adult["Male"] = (adult["gendercode"] == 1).astype(int)
    adult["Female"] = (adult["gendercode"] == 2).astype(int)

    return adult


def build_parent_outcomes(adult: pd.DataFrame) -> pd.DataFrame:
    """Add parent aspiration outcomes to adult dataframe (for Table 1).

    Note: A2_* variables may not be present in all versions of the data.
    The parent aspiration questions were asked in a separate module.
    """
    adult = adult.copy()

    if "A2_1" in adult.columns:
        adult["want_daughter_graduate"] = (adult["A2_1"] == 13).astype(int)
        adult.loc[adult["A2_1"].isna(), "want_daughter_graduate"] = np.nan
    else:
        adult["want_daughter_graduate"] = np.nan

    if "A2_2" in adult.columns:
        adult["want_son_graduate"] = (adult["A2_2"] == 13).astype(int)
        adult.loc[adult["A2_2"].isna(), "want_son_graduate"] = np.nan
    else:
        adult["want_son_graduate"] = np.nan

    if "A2_3" in adult.columns:
        age = adult["A2_3"].replace([99, 999], np.nan)
        adult["want_daughter_marry_after18"] = (age >= 19).astype(int)
        adult.loc[age.isna(), "want_daughter_marry_after18"] = np.nan
    else:
        adult["want_daughter_marry_after18"] = np.nan

    if "A2_5" in adult.columns:
        adult["want_daughter_pradhan"] = adult["A2_5"].map({1: 1, 2: 0})
    else:
        adult["want_daughter_pradhan"] = np.nan

    return adult


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def load_and_merge(
    data_dir: Path, include_adult: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load the .dta files and build the analysis dataframes.

    Returns:
        teen: DataFrame with adolescent data and treatment indicators
        adult: DataFrame with adult data (or None if include_adult=False)
    """
    teen, _ = pyreadstat.read_dta(
        str(data_dir / "powerful_women_in_india_teenager_survey.dta")
    )
    trt, _ = pyreadstat.read_dta(
        str(data_dir / "powerful_women_in_india_pradhan_seats_reserved_for_women.dta")
    )
    roster, _ = pyreadstat.read_dta(
        str(data_dir / "powerful_women_in_india_household_roster.dta")
    )
    hh, _ = pyreadstat.read_dta(
        str(data_dir / "powerful_women_in_india_household_survey.dta")
    )

    trt["twice_res"] = (
        (trt["res_woman"] == 1) & (trt["prev_res_woman"] == 1)
    ).astype(int)
    trt["ever_res"] = ((trt["res_woman"] == 1) | (trt["prev_res_woman"] == 1)).astype(
        int
    )
    trt["once_res"] = ((trt["ever_res"] == 1) & (trt["twice_res"] == 0)).astype(int)
    trt = trt[["AA0_2b", "twice_res", "once_res"]]

    roster_cols = ["serialid", "membercode", "A1_2", "A1_4_year"]
    for col in ["A1_5", "A1_7", "A1_8"]:
        if col in roster.columns:
            roster_cols.append(col)
    teen = teen.merge(
        roster[roster_cols],
        left_on=["hhserialid", "hhmembercode"],
        right_on=["serialid", "membercode"],
        how="left",
    )
    teen = teen[(teen["A1_4_year"] >= 11) & (teen["A1_4_year"] <= 15)]
    teen["girl"] = (teen["A1_2"] == 2).astype(int)
    teen["boy"] = (teen["A1_2"] == 1).astype(int)

    hh_loc = hh[["serialid", "AA0_3b", "AA0_1b"]].rename(
        columns={"AA0_3b": "village_jl", "AA0_1b": "block_id"}
    )
    teen = teen.merge(
        hh_loc, left_on="hhserialid", right_on="serialid", how="left", suffixes=("", "_hh")
    )

    teen = teen.merge(trt, on="AA0_2b", how="left").dropna(
        subset=["twice_res", "girl", "boy"]
    )
    teen = teen[(teen["girl"] == 1) | (teen["boy"] == 1)].copy()
    teen["never_res"] = ((teen["twice_res"] == 0) & (teen["once_res"] == 0)).astype(int)
    teen["village_id"] = (
        teen["AA0_2b"].astype(int).astype(str)
        + "_"
        + teen["village_jl"].astype(int).astype(str)
    )
    teen = build_teen_outcomes(teen)

    adult = None
    if include_adult:
        adult, _ = pyreadstat.read_dta(
            str(data_dir / "powerful_women_in_india_adult_survey.dta")
        )
        adult = adult.merge(trt, on="AA0_2b", how="left")
        adult["never_res"] = (
            (adult["twice_res"] == 0) & (adult["once_res"] == 0)
        ).astype(int)
        adult = adult.merge(
            roster[["serialid", "membercode", "A1_4_year"]],
            left_on=["serialid", "AA0_5"],
            right_on=["serialid", "membercode"],
            how="left",
            suffixes=("", "_r"),
        )
        adult["age"] = adult["A1_4_year"]
        adult = build_adult_outcomes(adult)
        adult = build_parent_outcomes(adult)

    return teen, adult


# ---------------------------------------------------------------------------
# Estimation primitives
# ---------------------------------------------------------------------------


def fit(df: pd.DataFrame, y: str, cluster: str = "AA0_2b"):
    """Fit the Beaman et al. specification with cluster-robust SEs."""
    use = df.dropna(subset=[y]).copy()
    use["Female"] = use["girl"]
    use["Once"] = use["once_res"]
    use["Twice"] = use["twice_res"]
    return smf.ols(FORMULA.format(y=y), data=use).fit(
        cov_type="cluster",
        cov_kwds={"groups": use[cluster]},
    )


def fit_simple(df: pd.DataFrame, y: str, regressors: list[str], cluster: str = "AA0_2b"):
    """Fit a simple regression with cluster-robust SEs."""
    use = df.dropna(subset=[y]).copy()
    formula = f"{y} ~ {' + '.join(regressors)}"
    return smf.ols(formula, data=use).fit(
        cov_type="cluster",
        cov_kwds={"groups": use[cluster]},
    )


def gap_components(model) -> dict:
    """Pull boys, girls, and gap-difference coefficients with SEs."""
    cov = model.cov_params()
    out = {}
    for label, t in [("once", "Once"), ("twice", "Twice")]:
        b_b = model.params[t]
        s_b = model.bse[t]
        b_g = model.params[t] + model.params[f"{t}:Female"]
        s_g = float(
            np.sqrt(
                cov.loc[t, t]
                + cov.loc[f"{t}:Female", f"{t}:Female"]
                + 2 * cov.loc[t, f"{t}:Female"]
            )
        )
        b_d = model.params[f"{t}:Female"]
        s_d = model.bse[f"{t}:Female"]
        out[label] = dict(boys=(b_b, s_b), girls=(b_g, s_g), diff=(b_d, s_d))
    return out


def make_index(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Construct a z-score index normalized to never-reserved baseline."""
    df = df.copy()
    nr = df[df["never_res"] == 1]
    z_cols = []
    for c in cols:
        mu, sd = nr[c].mean(), nr[c].std()
        if sd > 0:
            df[f"{c}_z"] = (df[c] - mu) / sd
            z_cols.append(f"{c}_z")
    df["idx"] = df[z_cols].mean(axis=1)
    return df


# ---------------------------------------------------------------------------
# Wild cluster bootstrap
# ---------------------------------------------------------------------------


def wild_cluster_bootstrap(
    df: pd.DataFrame,
    formula: str,
    coef_name: str = "Twice:Female",
    cluster_col: str = "AA0_2b",
    B: int = 999,
    seed: int = 42,
) -> tuple[float, float, int]:
    """
    Wild cluster bootstrap p-value for `coef_name`, restricting to the null.
    Rademacher weights drawn at the cluster level.

    Returns: (t_observed, two-sided p-value, number of completed bootstrap draws)
    """
    use = df.copy()
    m_full = smf.ols(formula, data=use).fit(
        cov_type="cluster", cov_kwds={"groups": use[cluster_col]}
    )
    t_obs = m_full.params[coef_name] / m_full.bse[coef_name]

    formula_null = formula.replace(f" + {coef_name}", "")
    m_null = smf.ols(formula_null, data=use).fit()
    fitted, resid = m_null.fittedvalues.values, m_null.resid.values

    rng = np.random.default_rng(seed)
    cl = use[cluster_col].values
    unique = np.unique(cl)
    boot = []
    y_name = formula.split("~")[0].strip()

    for _ in range(B):
        w = rng.choice([-1.0, 1.0], size=len(unique))
        cw = dict(zip(unique, w))
        wvec = np.fromiter((cw[c] for c in cl), dtype=float, count=len(cl))
        use["_y_b"] = fitted + wvec * resid
        try:
            mb = smf.ols(formula.replace(y_name, "_y_b"), data=use).fit(
                cov_type="cluster", cov_kwds={"groups": use[cluster_col]}
            )
            boot.append(mb.params[coef_name] / mb.bse[coef_name])
        except Exception:
            continue

    boot = np.asarray(boot)
    return float(t_obs), float((np.abs(boot) >= np.abs(t_obs)).mean()), len(boot)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def hr(title: str, level: int = 1) -> None:
    """Print a horizontal rule with title."""
    bar = "=" * 82 if level == 1 else "-" * 82
    print(f"\n{bar}\n{title}\n{bar}")
