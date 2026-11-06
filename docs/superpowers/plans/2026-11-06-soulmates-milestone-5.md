# Soul mates milestone 5 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test the one soul mate premise against HCMST 2017. Fit a free match chance q per new person you talk to and compare it with the soul mate model by LOO. Then report K, the number of possible partners in the world pool.

**Architecture:** `src/soulmates/hcmst.py` loads age, ever had a partner and the weight. `src/soulmates/match.py` turns the POLYMOD posterior into the lifetime curve N(a) and builds the two models. `model.py` gains `recog_prior`, so both modules build the p_recog prior the same way. Notebook 03 runs the comparison.

**Tech Stack:** as milestones 1 to 4. No new dependency: pandas reads the Stata file.

**Spec:** `docs/superpowers/specs/2026-11-06-soulmates-milestone-5-design.md`

## Global constraints

- Everything from the earlier plans still holds. That covers the version floors, the ArviZ 1.x API, `uv run` for every command, the writing rules and marking made up values as made up.
- Do not commit. The user commits when they ask.
- No download script for HCMST. Each user accepts the data use agreement on the Stanford page. The file is already in `data/raw/`.
- `python3` on this machine is now the system Python 3.9. Run the writing checker with `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`, and everything else with `uv run`.
- Every code block below already ran in a scratch copy of the repo on 2026-11-06.

## File structure

```
src/soulmates/model.py        # changed: recog_prior helper, used by build_model
src/soulmates/hcmst.py        # new: load_hcmst
src/soulmates/match.py        # new: lifetime_curve, build_match_model, build_soulmate_model
tests/test_hcmst.py           # new: loader test on a tiny Stata file
tests/test_match.py           # new: lifetime_curve by hand, recovery of q
notebooks/03_partners.ipynb   # new: the comparison
```

---

### Task 1: HCMST loader

**Files:**
- Create: `src/soulmates/hcmst.py`
- Test: `tests/test_hcmst.py`

**Interfaces:**
- Consumes: the HCMST Stata file.
- Produces: `load_hcmst(path) -> pandas.DataFrame` with `age` (int64), `ever_partnered` (bool) and `weight` (float64), for ages 18 to 60.

- [ ] **Step 1: Write the failing test**

`tests/test_hcmst.py`:

```python
import pandas as pd

from soulmates.hcmst import load_hcmst


def test_load_hcmst(tmp_path):
    # Made up rows: ages 17 and 61 fall outside 18 to 60, status 4 means "never had a partner"
    raw = pd.DataFrame({
        "w1_ppage": [17, 18, 30, 60, 61],
        "w1_partnership_status": [1, 4, 1, 3, 2],
        "w1_weight_combo": [1.0, 0.5, 1.5, 2.0, 1.0],
        "other_column": [0, 0, 0, 0, 0],
    })
    path = tmp_path / "hcmst.dta"
    raw.to_stata(path, write_index=False)
    expected = pd.DataFrame({
        "age": [18, 30, 60],
        "ever_partnered": [False, True, True],
        "weight": [0.5, 1.5, 2.0],
    })
    pd.testing.assert_frame_equal(load_hcmst(path), expected)
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_hcmst.py -v`
Expected: collection error with `ModuleNotFoundError: No module named 'soulmates.hcmst'`

- [ ] **Step 3: Write the loader**

`src/soulmates/hcmst.py`:

```python
"""Load the 2017 wave of How Couples Meet and Stay Together: age, ever had a partner, survey weight."""
import warnings

import pandas as pd

COLUMNS = ["w1_ppage", "w1_partnership_status", "w1_weight_combo"]


def load_hcmst(path):
    with warnings.catch_warnings():
        # some string columns fall back to latin-1, none of the three columns used here is a string
        warnings.simplefilter("ignore", UnicodeWarning)
        d = pd.read_stata(path, convert_categoricals=False, columns=COLUMNS)
    d = d[d["w1_ppage"].between(18, 60)]
    return pd.DataFrame({
        "age": d["w1_ppage"].astype(int).to_numpy(),
        "ever_partnered": (d["w1_partnership_status"] != 4).to_numpy(),  # 4 is "never had a partner"
        "weight": d["w1_weight_combo"].astype(float).to_numpy(),
    })
```

- [ ] **Step 4: Run it and see it pass**

Run: `uv run pytest tests/test_hcmst.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run it on the real file**

Run: `uv run python -c "from soulmates.hcmst import load_hcmst; d = load_hcmst('data/raw/HCMST 2017 to 2022 small public version 2.2.dta'); print(d.shape, int(d.ever_partnered.sum()))"`
Expected: `(2520, 3) 2422`

- [ ] **Step 6: Stop for review.** Do not commit.

---

### Task 2: Match models

**Files:**
- Modify: `src/soulmates/model.py` (add `recog_prior`, use it in `build_model`)
- Create: `src/soulmates/match.py`
- Test: `tests/test_match.py`

**Interfaces:**
- Consumes: `age_band` from `model.py`, the POLYMOD posterior, and a DataFrame with `age` and `ever_partnered`.
- Produces:
  - `recog_prior(bounds=(0.1, 0.6), mass=0.9) -> (alpha, beta)`, two floats. For the defaults it returns about (3.056, 5.543).
  - `lifetime_curve(posterior, start_age=15) -> numpy array` of length 101, N(a) for ages 0 to 100.
  - `build_match_model(df, curve) -> pymc.Model` with `log10_q` and the deterministic `q`.
  - `build_soulmate_model(df, curve, pool_size, recog_bounds=(0.1, 0.6), recog_mass=0.9) -> pymc.Model` with `p_recog` and the deterministic `q`.

- [ ] **Step 1: Write the failing tests**

`tests/test_match.py`:

```python
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from soulmates.match import build_match_model, lifetime_curve


def test_lifetime_curve_by_hand():
    # Every band mean is 1 new person per day and there is no weekend effect, so a year adds 365
    posterior = az.from_dict(
        {"posterior": {"band_mean": np.ones((1, 1, 17)), "beta_weekend": np.zeros((1, 1))}},
        dims={"band_mean": ["band"]},
    ).posterior
    curve = lifetime_curve(posterior, start_age=15)
    assert curve[14] == 0
    assert curve[17] == pytest.approx(3 * 365)  # ages 15, 16 and 17


def test_recovers_q():
    # Made up: q = 1 in 2000, and 300 new people per year from age 15, so about 9% never find a partner
    true_q = 1 / 2000
    curve = np.maximum(np.arange(101) - 14, 0) * 300.0
    rng = np.random.default_rng(1)
    age = rng.integers(18, 61, size=3000)
    ever = rng.random(3000) < 1 - (1 - true_q) ** curve[age]
    df = pd.DataFrame({"age": age, "ever_partnered": ever})
    with build_match_model(df, curve):
        q = pm.sample(random_seed=2, progressbar=False).posterior["q"].median().item()
    # Limit from a trial over six seeds, where the largest error was 3.3%
    assert q == pytest.approx(true_q, rel=0.10)
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_match.py -v`
Expected: collection error with `ModuleNotFoundError: No module named 'soulmates.match'`

- [ ] **Step 3: Add recog_prior to the model**

`src/soulmates/model.py`, full file:

```python
"""PyMC model for new contacts per day, and the chance of finding your soul mate."""
import numpy as np
import preliz as pz
import pymc as pm
import pytensor.tensor as pt

N_BANDS = 17  # 5 year bands, the last one holds ages 80 and over


def age_band(age):
    return np.minimum(np.asarray(age) // 5, N_BANDS - 1)


def recog_prior(bounds=(0.1, 0.6), mass=0.9):
    # The speaker's guess, "I am mass sure p_recog lies between the bounds", turned into the
    # flattest Beta prior that meets it (maximum entropy), so the answer is unique
    recog = pz.Beta()
    pz.maxent(recog, lower=bounds[0], upper=bounds[1], mass=mass, plot=False)
    return float(recog.alpha), float(recog.beta)


def build_model(df, search_ages=(18, 60), recog_bounds=(0.1, 0.6), recog_mass=0.9):
    band = age_band(df["age"].to_numpy())
    weekend = df["weekend"].to_numpy().astype(float)
    search = np.arange(search_ages[0], search_ages[1] + 1)
    years_per_band = np.bincount(age_band(search), minlength=N_BANDS)  # search years that fall in each band
    with pm.Model(coords={"band": np.arange(N_BANDS)}) as model:
        mu_pop = pm.Normal("mu_pop", np.log(2), 1)  # guess: about 2 new people per day, wide on purpose
        sigma_band = pm.HalfNormal("sigma_band", 0.5)
        z = pm.Normal("z", 0, 1, dims="band")  # non centred form
        band_mean = pm.Deterministic("band_mean", pt.exp(mu_pop + sigma_band * z), dims="band")
        beta_weekend = pm.Normal("beta_weekend", 0, 0.5)  # log of the weekend multiplier, centred on no effect
        alpha = pm.HalfNormal("alpha", 5)
        pm.NegativeBinomial(
            "new_contacts",
            mu=band_mean[band] * pt.exp(beta_weekend * weekend),
            alpha=alpha,
            observed=df["new_contacts"].to_numpy(),
        )
        week = 5 / 7 + 2 / 7 * pt.exp(beta_weekend)  # average day of a week with 5 weekdays and 2 weekend days
        pm.Deterministic("lifetime_new", 365 * week * pt.sum(years_per_band * band_mean))
        a, b = recog_prior(recog_bounds, recog_mass)
        pm.Beta("p_recog", alpha=a, beta=b)
    return model


def p_find(idata, pool_size):
    post = idata.posterior
    # 1 − (1 − p_recog / M) ^ lifetime_new, written so a tiny probability does not round to zero
    return -np.expm1(post["lifetime_new"] * np.log1p(-post["p_recog"] / pool_size))
```

- [ ] **Step 4: Write the match module**

`src/soulmates/match.py`:

```python
"""Who has had a partner by which age: a free match chance q against the one soul mate premise."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from soulmates.model import age_band, recog_prior

AGES = np.arange(101)


def lifetime_curve(posterior, start_age=15):
    # N(a): new people met from start_age up to and including age a, from the POLYMOD posterior means
    band_mean = posterior["band_mean"].mean(("chain", "draw")).values
    week = 5 / 7 + 2 / 7 * np.exp(float(posterior["beta_weekend"].mean()))
    per_year = 365 * week * band_mean[age_band(AGES)]
    per_year[AGES < start_age] = 0
    return np.cumsum(per_year)


def _likelihood(df, curve, q):
    n = curve[df["age"].to_numpy()]
    # 1 − (1 − q) ^ N(a), written so a tiny q does not round to zero
    pm.Bernoulli("ever_partnered", p=-pt.expm1(n * pt.log1p(-q)), observed=df["ever_partnered"].to_numpy().astype(int))


def build_match_model(df, curve):
    with pm.Model() as model:
        log10_q = pm.Normal("log10_q", -4, 2)  # wide: 1 in 100 million to near certain
        _likelihood(df, curve, pm.Deterministic("q", 10**log10_q))
    return model


def build_soulmate_model(df, curve, pool_size, recog_bounds=(0.1, 0.6), recog_mass=0.9):
    with pm.Model() as model:
        a, b = recog_prior(recog_bounds, recog_mass)
        p_recog = pm.Beta("p_recog", alpha=a, beta=b)
        _likelihood(df, curve, pm.Deterministic("q", p_recog / pool_size))  # one soul mate in the pool
    return model
```

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv run pytest -v`
Expected: 11 passed, in about 55 seconds. In the scratch trial over six seeds, the recovery of q had errors of at most 3.3% and 0 divergences.

- [ ] **Step 6: Stop for review.** Do not commit.

---

### Task 3: Notebook 03

**Files:**
- Create: `notebooks/03_partners.ipynb`

**Interfaces:**
- Consumes: `load_hcmst` from Task 1, everything from Task 2, `load_polymod`, `build_model`, `pool_size` and `data/wpp_pool.csv`.
- Produces: the executed notebook with one figure, the LOO table and the headline K.

- [ ] **Step 1: Build the notebook**

Save this script as a scratch file outside the repo, for example `build_nb03.py`, and run it from the repo root with `uv run python build_nb03.py`.

```python
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# How many soul mates do you need?\n\n"
       "The soul mate premise gives each person one possible partner in a pool of about 1.3 billion. "
       "The How Couples Meet and Stay Together survey (HCMST, Rosenfeld and colleagues, Stanford) shows how many "
       "people actually have had a partner by each age. This notebook asks how large the number of possible "
       "partners has to be for the two to match.\n\n"
       "To get the data, read and accept the data use agreement on https://data.stanford.edu/hcmst2017, then "
       "save `HCMST 2017 to 2022 small public version 2.2.dta` to `data/raw/`. The agreement asks you to tell "
       "the HCMST contact person about your work and to cite the data."),
    code("import arviz as az\n"
         "import matplotlib.pyplot as plt\n"
         "import numpy as np\n"
         "import pandas as pd\n"
         "import pymc as pm\n\n"
         "from soulmates.hcmst import load_hcmst\n"
         "from soulmates.match import build_match_model, build_soulmate_model, lifetime_curve\n"
         "from soulmates.model import build_model, recog_prior\n"
         "from soulmates.polymod import load_polymod\n"
         "from soulmates.wpp import pool_size\n\n"
         "hcmst = load_hcmst(\"../data/raw/HCMST 2017 to 2022 small public version 2.2.dta\")\n"
         "bands = pd.cut(hcmst[\"age\"], [17, 24, 29, 34, 39, 49, 60])\n"
         "observed = hcmst.groupby(bands, observed=True).apply(lambda g: pd.Series({\n"
         "    \"people\": len(g),\n"
         "    \"never had a partner\": int((~g[\"ever_partnered\"]).sum()),\n"
         "    \"ever, share\": g[\"ever_partnered\"].mean(),\n"
         "    \"ever, weighted share\": np.average(g[\"ever_partnered\"], weights=g[\"weight\"]),\n"
         "})).astype({\"people\": int, \"never had a partner\": int})\n"
         "observed.round(3)"),
    md("## New people met by each age\n\n"
       "N(a) is the number of new people you have met and talked to from age 15 up to age a. It comes from the "
       "POLYMOD fit of notebook 02, through the posterior means of the band means and the weekend effect."),
    code("polymod = load_polymod(\"../data/raw\")\n"
         "with build_model(polymod):\n"
         "    contacts = pm.sample(random_seed=2, progressbar=False)\n"
         "curve = lifetime_curve(contacts.posterior, start_age=15)\n"
         "print({age: round(curve[age]) for age in (18, 25, 40, 60)})"),
    md("## The free model\n\n"
       "Each new person you talk to is a possible partner with chance q. The chance of having had a partner by "
       "age a is 1 − (1 − q) ^ N(a). The prior on q is wide, from 1 in 100 million to near certain."),
    code("with build_match_model(hcmst, curve):\n"
         "    free = pm.sample(random_seed=2, idata_kwargs={\"log_likelihood\": True})\n"
         "q = free.posterior[\"q\"].values.ravel()\n"
         "print(f\"q: 1 in {1 / np.median(q):,.0f}, 94% interval 1 in {1 / np.percentile(q, 97):,.0f} \"\n"
         "      f\"to 1 in {1 / np.percentile(q, 3):,.0f}, divergences {int(free.sample_stats['diverging'].sum())}\")"),
    code("chance = 1 - (1 - q[:, None]) ** curve[hcmst[\"age\"].to_numpy()][None, :]\n"
         "predicted = pd.Series(chance.mean(axis=0)).groupby(bands.to_numpy(), observed=True).mean()\n"
         "labels = [str(b) for b in observed.index]\n"
         "fig, ax = plt.subplots()\n"
         "ax.plot(labels, observed[\"ever, share\"], \"o\", label=\"HCMST, share\")\n"
         "ax.plot(labels, observed[\"ever, weighted share\"], \"s\", label=\"HCMST, weighted share\")\n"
         "ax.plot(labels, predicted.values, \"k-\", label=\"free model\")\n"
         "ax.set(xlabel=\"age band\", ylabel=\"share who ever had a partner\", title=\"Free model against HCMST\")\n"
         "ax.legend();"),
    md("## The soul mate model\n\n"
       "Now the premise itself: one possible partner in the world pool M, recognized with chance p_recog. Then "
       "q = p_recog / M. p_recog keeps the guess from notebook 02, 90% sure between 0.1 and 0.6."),
    code("wpp = pd.read_csv(\"../data/wpp_pool.csv\")\n"
         "M = pool_size(wpp)\n"
         "with build_soulmate_model(hcmst, curve, M):\n"
         "    soulmate = pm.sample(random_seed=2, idata_kwargs={\"log_likelihood\": True})\n"
         "share = (1 - (1 - soulmate.posterior[\"q\"].values.ravel()[:, None]) ** curve[hcmst[\"age\"].to_numpy()][None, :]).mean()\n"
         "print(f\"soul mate model: predicted share who ever had a partner {share:.1e}, observed {hcmst['ever_partnered'].mean():.3f}\")\n"
         "print(f\"divergences {int(soulmate.sample_stats['diverging'].sum())}\")"),
    md("## Which model predicts better?\n\n"
       "LOO (leave one out cross validation) scores how well each model predicts each person it did not see. "
       "elpd is that score summed over people, and higher is better."),
    code("comparison = az.compare({\"free q\": free, \"one soul mate\": soulmate})\n"
         "comparison"),
    md("## How many soul mates?\n\n"
       "K = q × M / p_recog is the number of people in the world pool who would have worked for you. q comes from "
       "the free model, and p_recog from its prior, because no data informs it."),
    code("a, b = recog_prior()\n"
         "p_recog = np.random.default_rng(4).beta(a, b, size=q.size)\n"
         "K = q * M / p_recog\n"
         "print(f\"1 in {1 / np.median(q):,.0f} new people you talk to could be a partner\")\n"
         "print(f\"K: {np.median(K):,.0f} possible partners in the world pool, \"\n"
         "      f\"94% interval {np.percentile(K, 3):,.0f} to {np.percentile(K, 97):,.0f}\")"),
    md("## Known limits\n\n"
       "1. The contact rates come from Europe in 2006 and the partner data from the US in 2017.\n"
       "2. The survey weights are not in the likelihood. The weighted shares are shown as a check.\n"
       "3. N(a) is treated as known, without its uncertainty of about ±4%.\n"
       "4. q is the same at every age. The model predicts about 69% partnered at 18 to 24, where HCMST shows "
       "81%, so it misses the youngest band by about 12 points.\n"
       "5. \"Ever had a partner\" counts any past partner, so a short relationship counts the same as a life "
       "long one.\n"
       "6. K uses the p_recog prior, because no data informs p_recog. The spread of K comes mostly from it."),
]
nbf.write(nbf.v4.new_notebook(cells=cells), "notebooks/03_partners.ipynb")
print("wrote notebooks/03_partners.ipynb")
```

Expected: `wrote notebooks/03_partners.ipynb`

- [ ] **Step 2: Execute it**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_partners.ipynb`
Expected: exits without error after about 2 minutes. In the scratch trial it printed:

```
{18: 1420, 25: 5044, 40: 12794, 60: 21722}
q: 1 in 2,419, 94% interval 1 in 2,230 to 1 in 2,609, divergences 0
soul mate model: predicted share who ever had a partner 9.7e-06, observed 0.961
divergences 0
1 in 2,419 new people you talk to could be a partner
K: 1,589,085 possible partners in the world pool, 94% interval 817,943 to 5,532,943
```

The LOO table ranks "free q" first, with an elpd of about −350. "One soul mate" comes second, with an elpd difference of about −27,900 and a standard error of about 140.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: 11 passed.

- [ ] **Step 4: Stop for review.** Do not commit.
