"""
Mechanism tests for Beaman et al. (2012).

Tests whether the proposed "role model" causal mechanism leaves behavioral traces:
  1. Pradhan contact rates by treatment
  2. Pradhan recognition (can respondents NAME the pradhan?)
  3. Gram Sabha attendance and speaking
  4. Political engagement: voting and newspaper reading

The paper claims aspirations changed through exposure to female leaders as role models.
If this mechanism operates, we should see behavioral traces in adults' engagement
with the political system.

Outputs:
  - tabs/mechanisms_pradhan_contact.md
  - tabs/mechanisms_recognition.md
  - tabs/mechanisms_gram_sabha.md
  - tabs/mechanisms_political.md
  - tabs/mechanisms_summary.md
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from utils import fit_simple, hr, load_and_merge, write_table

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Test 1: Pradhan contact
# ---------------------------------------------------------------------------


def test_pradhan_contact(adult: pd.DataFrame) -> None:
    """Pradhan contact (D1_7, D1_9): adult women's behavioral footprint."""
    hr("TEST: Pradhan Contact")
    print("\nDid women in reserved areas interact more with the female pradhan?")

    print(f"\nFraction approaching pradhan, by treatment cell and gender:")
    print(f"{'Cell':<12}{'Men':<12}{'Women':<12}")
    print("-" * 36)

    rows = []
    for cell, label in [
        ("never_res", "Never"),
        ("once_res", "Once"),
        ("twice_res", "Twice"),
    ]:
        sub = adult[adult[cell] == 1]
        m_men = sub[sub["Male"] == 1]["approached"].mean()
        m_wom = sub[sub["Female"] == 1]["approached"].mean()
        print(f"  {label:<10}{m_men:<12.3f}{m_wom:<12.3f}")
        rows.append({
            "Treatment Cell": label,
            "Men": f"{m_men:.3f}",
            "Women": f"{m_wom:.3f}",
        })

    df_contact = pd.DataFrame(rows)
    write_table(df_contact, "mechanisms_pradhan_contact", "Pradhan Contact by Treatment Cell")

    print(f"\nWomen-only regressions:")
    women = adult[adult["Female"] == 1]

    for v, lbl in [
        ("approached", "Approached pradhan"),
        ("easy_meet", "Easy to meet pradhan"),
    ]:
        m = fit_simple(women, v, ["once_res", "twice_res"])
        print(
            f"  {lbl:<25}: Once={m.params['once_res']:+.3f} (p={m.pvalues['once_res']:.2f}), "
            f"Twice={m.params['twice_res']:+.3f} (p={m.pvalues['twice_res']:.2f})"
        )


# ---------------------------------------------------------------------------
# Test 2: Pradhan recognition
# ---------------------------------------------------------------------------


def test_recognition(adult: pd.DataFrame) -> None:
    """Pradhan name recognition (D1_1a)."""
    hr("TEST: Pradhan Name Recognition")
    print("\nCritical mechanism check: can respondents NAME the pradhan?")
    print("If role model channel operates, recognition should increase.")

    print(f"\nFraction CORRECTLY naming current pradhan, by cell and gender:")
    print(f"{'Cell':<14}{'Men':<14}{'Women':<14}")
    print("-" * 42)

    rows = []
    for cell, label in [
        ("never_res", "Never"),
        ("once_res", "Once"),
        ("twice_res", "Twice"),
    ]:
        sub = adult[adult[cell] == 1]
        m = sub[sub["Male"] == 1]["knows_pradhan"].mean()
        w = sub[sub["Female"] == 1]["knows_pradhan"].mean()
        print(f"  {label:<12}{m:<14.3f}{w:<14.3f}")
        rows.append({
            "Treatment Cell": label,
            "Men": f"{m:.3f}",
            "Women": f"{w:.3f}",
        })

    df_recog = pd.DataFrame(rows)

    print(f"\nWomen-only regressions (does treatment increase recognition?):")
    women = adult[adult["Female"] == 1]

    print(f"{'Outcome':<26}{'Baseline':<10}{'Once':<18}{'Twice':<18}")
    print("-" * 72)

    reg_rows = []
    for v, lbl in [
        ("knows_pradhan", "Knows curr pradhan"),
        ("knows_prev_pradhan", "Knows prev pradhan"),
        ("knows_mla", "Knows MLA"),
    ]:
        m = fit_simple(women, v, ["once_res", "twice_res"])
        base = women[women["never_res"] == 1][v].mean()
        print(
            f"  {lbl:<26}{base:<10.3f}{m.params['once_res']:+.3f}(p={m.pvalues['once_res']:.2f})    "
            f"{m.params['twice_res']:+.3f}(p={m.pvalues['twice_res']:.2f})"
        )
        reg_rows.append({
            "Outcome": lbl,
            "Baseline (NR)": f"{base:.3f}",
            "Once (coef, p)": f"{m.params['once_res']:+.3f} (p={m.pvalues['once_res']:.2f})",
            "Twice (coef, p)": f"{m.params['twice_res']:+.3f} (p={m.pvalues['twice_res']:.2f})",
        })

    df_recog["Recognition"] = "Raw means"
    df_reg = pd.DataFrame(reg_rows)
    write_table(df_reg, "mechanisms_recognition", "Pradhan Recognition (Women Only)")


# ---------------------------------------------------------------------------
# Test 3: Gram Sabha participation
# ---------------------------------------------------------------------------


def test_gram_sabha(adult: pd.DataFrame) -> None:
    """Gram Sabha participation (women-only)."""
    hr("TEST: Gram Sabha Participation")
    print("\nDid women in reserved areas participate more in local government?")

    women = adult[adult["Female"] == 1]

    print(f"\n{'Outcome':<26}{'Baseline':<10}{'Once':<18}{'Twice':<18}")
    print("-" * 72)

    rows = []
    for v, lbl in [
        ("heard_gs", "Heard of GS"),
        ("any_gs_activity", "Any GS activity"),
        ("spoke_gs", "Spoke at last GS"),
    ]:
        m = fit_simple(women, v, ["once_res", "twice_res"])
        base = women[women["never_res"] == 1][v].mean()
        print(
            f"  {lbl:<26}{base:<10.3f}{m.params['once_res']:+.3f}(p={m.pvalues['once_res']:.2f})    "
            f"{m.params['twice_res']:+.3f}(p={m.pvalues['twice_res']:.2f})"
        )
        rows.append({
            "Outcome": lbl,
            "Baseline (NR)": f"{base:.3f}",
            "Once (coef, p)": f"{m.params['once_res']:+.3f} (p={m.pvalues['once_res']:.2f})",
            "Twice (coef, p)": f"{m.params['twice_res']:+.3f} (p={m.pvalues['twice_res']:.2f})",
        })

    df_out = pd.DataFrame(rows)
    write_table(df_out, "mechanisms_gram_sabha", "Gram Sabha Participation (Women Only)")


# ---------------------------------------------------------------------------
# Test 4: Political engagement
# ---------------------------------------------------------------------------


def test_political_engagement(adult: pd.DataFrame) -> None:
    """Voting and newspaper reading (women-only)."""
    hr("TEST: Political Engagement")
    print("\nDid women in reserved areas engage more with politics?")

    women = adult[adult["Female"] == 1]

    print(f"\n{'Outcome':<26}{'Baseline':<10}{'Once':<18}{'Twice':<18}")
    print("-" * 72)

    rows = []
    for v, lbl in [
        ("voted_gp", "Voted in last GP"),
        ("reads_paper", "Reads newspaper"),
    ]:
        m = fit_simple(women, v, ["once_res", "twice_res"])
        base = women[women["never_res"] == 1][v].mean()
        print(
            f"  {lbl:<26}{base:<10.3f}{m.params['once_res']:+.3f}(p={m.pvalues['once_res']:.2f})    "
            f"{m.params['twice_res']:+.3f}(p={m.pvalues['twice_res']:.2f})"
        )
        rows.append({
            "Outcome": lbl,
            "Baseline (NR)": f"{base:.3f}",
            "Once (coef, p)": f"{m.params['once_res']:+.3f} (p={m.pvalues['once_res']:.2f})",
            "Twice (coef, p)": f"{m.params['twice_res']:+.3f} (p={m.pvalues['twice_res']:.2f})",
        })

    df_out = pd.DataFrame(rows)
    write_table(df_out, "mechanisms_political", "Political Engagement (Women Only)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_mechanism_summary(adult: pd.DataFrame) -> None:
    """Summarize mechanism evidence."""
    hr("MECHANISM SUMMARY")

    women = adult[adult["Female"] == 1]

    print("\nKey question: If 'role model' channel works, do we see behavioral traces?")
    print("\nExpected if mechanism operates:")
    print("  - Women contact pradhan more in reserved areas")
    print("  - Women recognize pradhan's name more in reserved areas")
    print("  - Women participate more in gram sabhas")

    print("\nObserved (twice-reserved vs. never-reserved, women only):")

    outcomes = [
        ("approached", "Approached pradhan"),
        ("knows_pradhan", "Knows pradhan name"),
        ("spoke_gs", "Spoke at gram sabha"),
    ]

    rows = []
    for v, lbl in outcomes:
        m = fit_simple(women, v, ["once_res", "twice_res"])
        base = women[women["never_res"] == 1][v].mean()
        twice = women[women["twice_res"] == 1][v].mean()
        coef = m.params["twice_res"]
        p = m.pvalues["twice_res"]
        direction = "+" if coef > 0 else "-"
        sig = "*" if p < 0.05 else ""

        print(f"  {lbl:<25}: {base:.1%} -> {twice:.1%} ({direction}{abs(coef):.1%}, p={p:.2f}){sig}")
        rows.append({
            "Outcome": lbl,
            "Never-reserved": f"{base:.1%}",
            "Twice-reserved": f"{twice:.1%}",
            "Coefficient": f"{direction}{abs(coef):.3f}",
            "p-value": f"{p:.2f}",
            "Sig": sig,
        })

    df_out = pd.DataFrame(rows)
    write_table(df_out, "mechanisms_summary", "Mechanism Evidence Summary")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


ALL_TESTS = {
    1: ("pradhan-contact", test_pradhan_contact),
    2: ("recognition", test_recognition),
    3: ("gram-sabha", test_gram_sabha),
    4: ("political-engagement", test_political_engagement),
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
        "--tests",
        type=int,
        nargs="*",
        help=f"Subset of tests to run (1-{len(ALL_TESTS)}). Default: all.",
    )
    args = parser.parse_args()

    tests_to_run = args.tests if args.tests else list(ALL_TESTS.keys())

    _, adult = load_and_merge(args.data_dir, include_adult=True)

    if adult is None:
        print("ERROR: Adult survey data not found.")
        return

    for k in tests_to_run:
        if k not in ALL_TESTS:
            continue
        _, fn = ALL_TESTS[k]
        fn(adult)

    print_mechanism_summary(adult)


if __name__ == "__main__":
    main()
