# Soul mates milestone 1 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the simulated data pipeline for the soul mate talk: diaries, the PyMC model, P_find, the tests and the walkthrough notebook.

**Architecture:** `simulate.py` makes POLYMOD style diaries with known true values. `model.py` holds the band definition, the PyMC model and the numpy function for P_find. The tests and the notebook only call these three public functions.

**Tech Stack:** Python 3.13, uv, PyMC 6.4, PyTensor 3.3.7, ArviZ 1.3, pandas, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-soulmates-design.md`

## Global constraints

- Version floors: `pymc>=6.4`, `pytensor>=3.3.7`, `arviz>=1.3`. PyTensor 2.31 breaks every compile on this machine with `ld: library 'd64' not found`.
- ArviZ 1.x API: `az.hdi(..., prob=0.94)`, not `hdi_prob`. `az.from_dict({"posterior": {...}})` needs numpy arrays, not lists.
- Run everything through uv: `uv run pytest`, `uv run python`.
- Do not commit. The user commits when they ask.
- Comments, docstrings and markdown follow the user's writing rules. No em dashes or en dashes, no hyphens in words ("non centred", "made up") and no contractions.
- Mark every made up value as made up or as a guess where it appears.

## File structure

```
.gitignore
pyproject.toml
src/soulmates/__init__.py      # empty
src/soulmates/model.py         # N_BANDS, age_band, build_model, p_find
src/soulmates/simulate.py      # simulate_diaries
tests/test_recovery.py         # test_p_find_by_hand, test_recovers_true_values
notebooks/01_simulated.ipynb   # the talk walkthrough
```

---

### Task 1: Project setup and P_find

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `src/soulmates/__init__.py`, `src/soulmates/model.py`
- Test: `tests/test_recovery.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `p_find(idata, pool_size) -> xarray.DataArray`. It reads `idata.posterior["eye_contacts"]` and `idata.posterior["p_recog"]` and returns P_find with the same `chain` and `draw` dims.

- [ ] **Step 1: Write the project files**

`.gitignore`:

```
.venv/
__pycache__/
.ipynb_checkpoints/
.pytest_cache/
```

`pyproject.toml`:

```toml
[project]
name = "soulmates"
version = "0.1.0"
description = "Bayesian odds of finding your soul mate, for a PyCon talk"
requires-python = ">=3.13"
dependencies = [
    "arviz>=1.3",
    "matplotlib>=3.11",
    "pandas>=3.1",
    "pymc>=6.4",
    "pytensor>=3.3.7",
]

[dependency-groups]
dev = ["ipykernel", "nbconvert", "pytest"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

`src/soulmates/__init__.py`: an empty file.

- [ ] **Step 2: Install**

Run: `uv sync`
Expected: creates `.venv` and installs `soulmates` in editable mode. Then run `uv run python -c "import pytensor; print(pytensor.__version__)"` and expect `3.3.7` or later.

- [ ] **Step 3: Write the failing test**

`tests/test_recovery.py`:

```python
import arviz as az
import numpy as np
import pytest

from soulmates.model import p_find


def test_p_find_by_hand():
    # 2 eye contacts, sure recognition, pool of 4: 1 − (3/4)² = 7/16
    idata = az.from_dict({"posterior": {"eye_contacts": np.array([[2.0]]), "p_recog": np.array([[1.0]])}})
    assert p_find(idata, pool_size=4).item() == pytest.approx(7 / 16)
```

- [ ] **Step 4: Run it and see it fail**

Run: `uv run pytest tests/test_recovery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'soulmates.model'`

- [ ] **Step 5: Write the minimal code**

`src/soulmates/model.py`:

```python
"""PyMC model for new contacts per day, and the chance of finding your soul mate."""
import numpy as np


def p_find(idata, pool_size):
    post = idata.posterior
    # 1 − (1 − p_recog / M) ^ eye_contacts, written so a tiny probability does not round to zero
    return -np.expm1(post["eye_contacts"] * np.log1p(-post["p_recog"] / pool_size))
```

- [ ] **Step 6: Run it and see it pass**

Run: `uv run pytest tests/test_recovery.py -v`
Expected: `test_p_find_by_hand PASSED`

- [ ] **Step 7: Stop for review.** Do not commit.

---

### Task 2: Simulator, model and recovery test

**Files:**
- Create: `src/soulmates/simulate.py`
- Modify: `src/soulmates/model.py` (add `N_BANDS`, `age_band`, `build_model`)
- Test: `tests/test_recovery.py` (add `test_recovers_true_values`)

**Interfaces:**
- Consumes: `p_find` from Task 1, unchanged.
- Produces:
  - `N_BANDS = 17` and `age_band(age) -> numpy int array`, where the band is `min(age // 5, 16)`.
  - `simulate_diaries(n_people, band_means, alpha, seed) -> pandas.DataFrame` with int columns `age` (0 to 84) and `new_contacts`.
  - `build_model(df, search_ages=(18, 60)) -> pymc.Model` with free variables `mu_pop`, `sigma_band`, `z` (dim `band`), `alpha`, `p_eye`, `p_recog`, and deterministics `band_mean` (dim `band`), `lifetime_new`, `eye_contacts`.

- [ ] **Step 1: Write the failing test**

Replace `tests/test_recovery.py` with:

```python
import arviz as az
import numpy as np
import pymc as pm
import pytest

from soulmates.model import build_model, p_find
from soulmates.simulate import simulate_diaries

# Made up true values for the test, one mean per 5 year band, peaking at ages 20 to 24
TRUE_BAND_MEANS = [1.0, 1.5, 2.0, 3.0, 4.0, 3.5, 3.0, 2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0]
TRUE_ALPHA = 2.0
# Ages 18 to 60 by hand: 18 to 19 is 2 × 3.0, then 5 years each of 4.0, 3.5, 3.0, 2.5, 2.5, 2.0, 2.0, 1.5,
# then age 60 is 1 × 1.5. That is 6 + 20 + 17.5 + 15 + 12.5 + 12.5 + 10 + 10 + 7.5 + 1.5 = 112.5, times 365 days.
TRUE_LIFETIME_NEW = 41062.5


def test_recovers_true_values():
    df = simulate_diaries(3000, TRUE_BAND_MEANS, TRUE_ALPHA, seed=1)
    with build_model(df):
        post = pm.sample(random_seed=2, progressbar=False).posterior
    # Limits from a trial over six seeds, where the largest errors were 14%, 4.5% and 26%
    assert float(post["alpha"].mean()) == pytest.approx(TRUE_ALPHA, rel=0.25)
    assert float(post["lifetime_new"].mean()) == pytest.approx(TRUE_LIFETIME_NEW, rel=0.10)
    assert post["band_mean"].mean(("chain", "draw")).values == pytest.approx(TRUE_BAND_MEANS, rel=0.40)


def test_p_find_by_hand():
    # 2 eye contacts, sure recognition, pool of 4: 1 − (3/4)² = 7/16
    idata = az.from_dict({"posterior": {"eye_contacts": np.array([[2.0]]), "p_recog": np.array([[1.0]])}})
    assert p_find(idata, pool_size=4).item() == pytest.approx(7 / 16)
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_recovery.py -v`
Expected: collection error with `ImportError: cannot import name 'build_model' from 'soulmates.model'`

- [ ] **Step 3: Write the model**

`src/soulmates/model.py`, full file:

```python
"""PyMC model for new contacts per day, and the chance of finding your soul mate."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N_BANDS = 17  # 5 year bands, the last one holds ages 80 and over


def age_band(age):
    return np.minimum(np.asarray(age) // 5, N_BANDS - 1)


def build_model(df, search_ages=(18, 60)):
    band = age_band(df["age"].to_numpy())
    search = np.arange(search_ages[0], search_ages[1] + 1)
    years_per_band = np.bincount(age_band(search), minlength=N_BANDS)  # search years that fall in each band
    with pm.Model(coords={"band": np.arange(N_BANDS)}) as model:
        mu_pop = pm.Normal("mu_pop", np.log(2), 1)  # guess: about 2 new people per day, wide on purpose
        sigma_band = pm.HalfNormal("sigma_band", 0.5)
        z = pm.Normal("z", 0, 1, dims="band")  # non centred form
        band_mean = pm.Deterministic("band_mean", pt.exp(mu_pop + sigma_band * z), dims="band")
        alpha = pm.HalfNormal("alpha", 5)
        pm.NegativeBinomial(
            "new_contacts", mu=band_mean[band], alpha=alpha, observed=df["new_contacts"].to_numpy()
        )
        lifetime_new = pm.Deterministic("lifetime_new", 365 * pt.sum(years_per_band * band_mean))
        p_eye = pm.Beta("p_eye", 2, 2)  # guess, no data informs it
        pm.Deterministic("eye_contacts", lifetime_new * p_eye)
        pm.Beta("p_recog", 2, 2)  # guess, no data informs it
    return model


def p_find(idata, pool_size):
    post = idata.posterior
    # 1 − (1 − p_recog / M) ^ eye_contacts, written so a tiny probability does not round to zero
    return -np.expm1(post["eye_contacts"] * np.log1p(-post["p_recog"] / pool_size))
```

- [ ] **Step 4: Write the simulator**

`src/soulmates/simulate.py`:

```python
"""Simulated POLYMOD style contact diaries with known true values."""
import numpy as np
import pandas as pd

from soulmates.model import age_band


def simulate_diaries(n_people, band_means, alpha, seed):
    rng = np.random.default_rng(seed)
    age = rng.integers(0, 85, size=n_people)
    mu = np.asarray(band_means)[age_band(age)]
    # numpy uses n and p for the negative binomial: n = alpha and p = alpha / (alpha + mu) give mean mu
    new_contacts = rng.negative_binomial(alpha, alpha / (alpha + mu))
    return pd.DataFrame({"age": age, "new_contacts": new_contacts})
```

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv run pytest tests/test_recovery.py -v`
Expected: both tests PASSED, in about 20 to 40 seconds. In the scratch trial the sampler reported 0 divergences and r_hat 1.00.

- [ ] **Step 6: Stop for review.** Do not commit.

---

### Task 3: Walkthrough notebook

**Files:**
- Create: `notebooks/01_simulated.ipynb`

**Interfaces:**
- Consumes: `age_band`, `build_model`, `p_find` from `soulmates.model`, and `simulate_diaries` from `soulmates.simulate`, as defined in Task 2.
- Produces: an executed notebook with four figures and the printed P_find summary.

- [ ] **Step 1: Build the notebook**

Run this once from the repo root. It writes the notebook with nbformat, which nbconvert installs.

```bash
mkdir -p notebooks && uv run python - <<'EOF'
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Soul mates with Bayes, on simulated data\n\n"
       "Everyone has one soul mate, drawn at random from the M people born within 5 years of them. "
       "You find them only if you lock eyes and recognize each other. This notebook estimates how many "
       "new people you meet per day, turns that into lifetime eye contacts, and then into the chance of "
       "finding your soul mate. The data is simulated with known true values, so we can check the model."),
    code("import arviz as az\n"
         "import matplotlib.pyplot as plt\n"
         "import numpy as np\n"
         "import pymc as pm\n\n"
         "from soulmates.model import age_band, build_model, p_find\n"
         "from soulmates.simulate import simulate_diaries\n\n"
         "# Made up true values: mean new contacts per day for each 5 year band, peaking at ages 20 to 24\n"
         "TRUE_BAND_MEANS = [1.0, 1.5, 2.0, 3.0, 4.0, 3.5, 3.0, 2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0]\n"
         "TRUE_ALPHA = 2.0\n"
         "df = simulate_diaries(3000, TRUE_BAND_MEANS, TRUE_ALPHA, seed=1)\n"
         "df.head()"),
    code("observed = df.groupby(age_band(df[\"age\"]))[\"new_contacts\"].mean()\n"
         "fig, ax = plt.subplots()\n"
         "ax.bar(observed.index * 5, observed.values, width=4, label=\"simulated diaries\")\n"
         "ax.plot(np.arange(17) * 5, TRUE_BAND_MEANS, \"ko\", label=\"true band mean\")\n"
         "ax.set(xlabel=\"start of the 5 year age band\", ylabel=\"new contacts per day\", "
         "title=\"New contacts per day by age band\")\n"
         "ax.legend();"),
    md("## Prior predictive check\n\n"
       "Before fitting, we draw fake diaries from the priors alone. They should cover the simulated "
       "diaries without putting much weight on absurd days."),
    code("model = build_model(df)\n"
         "with model:\n"
         "    prior = pm.sample_prior_predictive(random_seed=3)\n"
         "prior_counts = prior.prior_predictive[\"new_contacts\"].values.ravel()\n"
         "bins = np.arange(52)\n"
         "fig, ax = plt.subplots()\n"
         "ax.hist(np.minimum(prior_counts, 50), bins=bins, density=True, alpha=0.5, label=\"prior predictive\")\n"
         "ax.hist(np.minimum(df[\"new_contacts\"], 50), bins=bins, density=True, alpha=0.5, label=\"simulated diaries\")\n"
         "ax.set(xlabel=\"new contacts on one day (50 stands for 50 or more)\", ylabel=\"share of days\", "
         "title=\"Prior predictive against the simulated diaries\")\n"
         "ax.legend();"),
    md("## Fit and recovery\n\nThe posterior band means should sit close to the true values."),
    code("with model:\n"
         "    idata = pm.sample(random_seed=2)\n"
         "az.summary(idata, var_names=[\"mu_pop\", \"sigma_band\", \"alpha\", \"lifetime_new\"])"),
    code("hdi = az.hdi(idata, var_names=[\"band_mean\"], prob=0.94)[\"band_mean\"].values\n"
         "mean = idata.posterior[\"band_mean\"].mean((\"chain\", \"draw\")).values\n"
         "ages = np.arange(17) * 5\n"
         "fig, ax = plt.subplots()\n"
         "ax.errorbar(ages, mean, yerr=[mean - hdi[:, 0], hdi[:, 1] - mean], fmt=\"o\", "
         "label=\"posterior mean and 94% HDI\")\n"
         "ax.plot(ages, TRUE_BAND_MEANS, \"kx\", label=\"true band mean\")\n"
         "ax.set(xlabel=\"start of the 5 year age band\", ylabel=\"new contacts per day\", "
         "title=\"Posterior band means against the true values\")\n"
         "ax.legend();"),
    md("## The chance of finding your soul mate\n\n"
       "P_find = 1 − (1 − p_recog / M) ^ eye_contacts. M is a placeholder here until the UN World "
       "Population Prospects cohort sizes are in."),
    code("POOL_SIZE = 1e8  # PLACEHOLDER until the UN WPP cohort sizes arrive\n"
         "pf = p_find(idata, POOL_SIZE).values.ravel()\n"
         "fig, ax = plt.subplots()\n"
         "ax.hist(pf, bins=60)\n"
         "ax.set(xlabel=\"P(find your soul mate)\", ylabel=\"posterior draws\", "
         "title=f\"P_find with a placeholder pool of {POOL_SIZE:.0e} people\")\n"
         "print(f\"P_find mean {pf.mean():.2e}, 94% interval {np.percentile(pf, 3):.2e} to {np.percentile(pf, 97):.2e}\")"),
]
nbf.write(nbf.v4.new_notebook(cells=cells), "notebooks/01_simulated.ipynb")
EOF
```

- [ ] **Step 2: Execute it**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_simulated.ipynb`
Expected: exits without error. In the scratch trial the last cell printed `P_find mean 9.84e-05, 94% interval 1.14e-05 to 2.43e-04`. Different library builds may change the last digits.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: 2 passed.

- [ ] **Step 4: Stop for review.** Do not commit.
