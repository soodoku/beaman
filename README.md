# Beaman et al. (2012) Replication

A Python reanalysis of:

> Beaman, L., Duflo, E., Pande, R., & Topalova, P. (2012).
> "Female Leadership Raises Aspirations and Educational Attainment for Girls:
> A Policy Experiment in India." *Science*, 335(6068), 582-586.
> [doi:10.1126/science.1212382](https://doi.org/10.1126/science.1212382)

## Paper Summary

The paper examines whether exposure to female political leaders affects adolescent girls' aspirations in rural India. It exploits random assignment of panchayat (village council) seats to women under India's 1993 reservation policy. The main claim: two cycles of female leadership closes one-third of the gender gap in adolescent aspirations.

## Data

**Sources:**
- [Harvard Dataverse (J-PAL)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O3UKFO)
- [Stanford Redivis](https://stanford.redivis.com/datasets/q707-5rwxymgw0/tables)

**Required files:**

| File | Rows | Description |
|------|------|-------------|
| `powerful_women_in_india_teenager_survey.dta` | 3,680 | Adolescent aspirations |
| `powerful_women_in_india_pradhan_seats_reserved_for_women.dta` | 165 | Treatment assignment |
| `powerful_women_in_india_household_roster.dta` | 37,263 | Demographics |
| `powerful_women_in_india_household_survey.dta` | 7,425 | Location data |
| `powerful_women_in_india_adult_survey.dta` | 13,508 | Adult attitudes (for mechanisms) |

**Treatment cells:**
- Twice-reserved: 20 GPs (both current and previous cycle)
- Once-reserved: 70 GPs (exactly one cycle)
- Never-reserved: 72 GPs (control)

## Replication Results

### Table 2: Adolescent Aspirations

Baselines in never-reserved villages:

| Outcome | Boys (paper) | Boys (reproduced) | Girls (paper) | Girls (reproduced) |
|---------|--------------|-------------------|---------------|---------------------|
| no_housewife | 0.998 | 0.998 | 0.600 | 0.585 |
| wish_graduate | 0.296 | 0.290 | 0.195 | 0.190 |
| marry_after18 | 0.980 | 0.980 | 0.660 | 0.662 |
| wish_pradhan | 0.499 | 0.504 | 0.485 | 0.487 |

Coefficients match within rounding tolerance.

See [tabs/table2_baselines.md](tabs/table2_baselines.md) and [tabs/table2_coefficients.md](tabs/table2_coefficients.md) for full tables.

### Paper Headline

The paper reports a +0.166 SD gap difference in twice-reserved areas (p < 0.02), representing 32% closure of the gender gap in aspirations.

## Robustness Checks

### Wild Cluster Bootstrap

With only 20 GPs in the twice-reserved cell, asymptotic cluster inference is unreliable. Wild cluster bootstrap (Cameron, Gelbach & Miller 2008) provides small-sample-corrected p-values.

See [tabs/robustness_wild_bootstrap.md](tabs/robustness_wild_bootstrap.md).

### Multiple Testing

Four outcomes tested separately. Bonferroni and Benjamini-Hochberg corrections applied.

See [tabs/robustness_multiple_testing.md](tabs/robustness_multiple_testing.md).

### Alternative Clustering

Treatment assigned at GP level. Standard errors compared across clustering at GP, household, and village levels.

See [tabs/robustness_clustering.md](tabs/robustness_clustering.md).

### Randomization Inference

GP-level treatment permutation test preserving cell sizes.

See [tabs/robustness_randomization.md](tabs/robustness_randomization.md).

## Mechanism Evidence

The paper proposes a "role model" channel: girls observe female pradhans and update aspirations. If this mechanism operates, adults (especially women) should show behavioral traces: more contact with the pradhan, higher recognition, increased political participation.

### Pradhan Contact and Recognition

Women's contact rates with the pradhan across treatment cells.

See [tabs/mechanisms_pradhan_contact.md](tabs/mechanisms_pradhan_contact.md) and [tabs/mechanisms_recognition.md](tabs/mechanisms_recognition.md).

### Gram Sabha Participation

Whether women in reserved areas participated more in local governance.

See [tabs/mechanisms_gram_sabha.md](tabs/mechanisms_gram_sabha.md).

### Political Engagement

Voting and newspaper reading rates.

See [tabs/mechanisms_political.md](tabs/mechanisms_political.md).

## Sensitivity Analysis

### Leave-One-Out

Drops each of the 20 twice-reserved GPs one at a time. Reports how many LOO specifications yield p > 0.05.

See [tabs/sensitivity_loo.md](tabs/sensitivity_loo.md) and [figs/loo_sensitivity.png](figs/loo_sensitivity.png).

### Ceiling Effects

Two of four outcomes (no_housewife, marry_after18) have boys near ceiling (>98%). This restricts the scope for boys to improve, mechanically inflating the gap-closure estimate.

See [tabs/sensitivity_ceiling.md](tabs/sensitivity_ceiling.md).

### Age Placebo

If the role-model channel operates on impressionable adolescents, adult women over 35 (socialized before the 1993 reservation policy) should show no treatment effects on gender attitudes.

See [tabs/sensitivity_age_placebo.md](tabs/sensitivity_age_placebo.md).

### Unused Gender-Attitude Items

The survey collected 10 gender-attitude questions (E_1 to E_10) not reported in the main paper.

See [tabs/sensitivity_e_items.md](tabs/sensitivity_e_items.md).

### Time Use

Did treatment change how adolescents allocate their time (school attendance, domestic chores)?

See [tabs/sensitivity_time_use.md](tabs/sensitivity_time_use.md).

## Running the Code

```bash
pip install -r requirements.txt

# Reproduce Tables 1, 2, 3
python scripts/01_replicate.py --data-dir data

# Inference robustness checks
python scripts/02_robustness.py --data-dir data

# Mechanism tests
python scripts/03_mechanisms.py --data-dir data

# Sensitivity analysis
python scripts/04_sensitivity.py --data-dir data

# Run specific tests
python scripts/02_robustness.py --data-dir data --tests 1 2
python scripts/04_sensitivity.py --data-dir data --tests 1 4

# More bootstrap iterations
python scripts/02_robustness.py --data-dir data --bootstrap-reps 4999
```

Output tables are written to `tabs/` as markdown files. Figures are saved to `figs/` as PNG.

## Codebook

Variable definitions and recoding logic are centralized in [scripts/codebook.py](scripts/codebook.py).

**Adolescent outcomes (Table 2):**

| Outcome | Source | Coding |
|---------|--------|--------|
| `no_housewife` | `B1_3` (free text) | 1 if not housewife/in-laws |
| `wish_graduate` | `B1_1` | 1 if `B1_1 == 13` (graduate) |
| `marry_after18` | `B1_2` | 1 if `B1_2 >= 19` |
| `wish_pradhan` | `B1_4` | 1 if `B1_4 == 1` |

**Adult outcomes:**

| Outcome | Source | Coding |
|---------|--------|--------|
| `approached` | `D1_7` | 1 = approached pradhan |
| `knows_pradhan` | `D1_1a_rem` | 1 = named correctly |
| `spoke_gs` | `D3_12` | 1 = spoke at gram sabha |
| `voted_gp` | `D1_3` | 1 = voted in GP election |

**Treatment:**

| Variable | Construction | N GPs |
|----------|--------------|-------|
| `twice_res` | `res_woman == 1` AND `prev_res_woman == 1` | 20 |
| `once_res` | exactly one cycle reserved | 70 |
| `never_res` | neither cycle reserved | 72 |

## Caveats

- 162 of 165 GPs match between treatment and household files
- Wild bootstrap uses Rademacher weights imposing the null
- Bootstrap/permutation p-values shift slightly across runs; use `--bootstrap-reps 4999` for stability
- The "high-education job" outcome requires hand-coding free-text occupation field `B1_3`

## License

MIT
