# Soul mates milestone 4 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the premise to "people you talk to" and remove `p_eye`. Build the `p_recog` prior from a stated guess with PreliZ `maxent`, and add a check without Germany.

**Architecture:** `build_model` loses `p_eye` and gains `recog_bounds` and `recog_mass`, which PreliZ turns into the Beta prior for `p_recog`. `p_find` reads `lifetime_new` directly. `load_polymod` gains a `country` column from the household file. Both notebooks get the new wording, and notebook 02 gets the prior figure and the fit without Germany.

**Tech Stack:** as milestones 1 to 3, plus PreliZ 0.28.

**Spec:** `docs/superpowers/specs/2026-11-05-soulmates-milestone-4-design.md`

## Global constraints

- Everything from the earlier plans still holds. That covers the version floors, the ArviZ 1.x API, `uv run` for every command, the writing rules and marking made up values as made up.
- Do not commit. The user commits when they ask.
- New dependency: `preliz>=0.28`. Do not use `pm.find_constrained_prior`, which is deprecated.
- Every code block below already ran in a scratch copy of the repo on 2026-11-05.

## File structure

```
pyproject.toml, uv.lock           # changed: preliz added
src/soulmates/model.py            # changed: no p_eye, p_recog prior from PreliZ maxent, p_find reads lifetime_new
tests/test_recovery.py            # changed: hand test uses lifetime_new, new prior test
src/soulmates/polymod.py          # changed: country column
scripts/download_polymod.py       # changed: fourth file, the household file
tests/test_polymod.py             # changed: household rows and country column
notebooks/01_simulated.ipynb      # changed: wording, rerun
notebooks/02_polymod.ipynb        # changed: wording, prior figure, fit without Germany, known limits
```

---

### Task 1: The p_recog prior and the model without p_eye

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (through `uv add`)
- Modify: `src/soulmates/model.py`
- Test: `tests/test_recovery.py`

**Interfaces:**
- Consumes: `age_band` and `N_BANDS`, unchanged.
- Produces:
  - `build_model(df, search_ages=(18, 60), recog_bounds=(0.1, 0.6), recog_mass=0.9) -> pymc.Model`. Free variables: `mu_pop`, `sigma_band`, `z`, `beta_weekend`, `alpha`, `p_recog`. Deterministics: `band_mean`, `lifetime_new`. There is no `p_eye` and no `eye_contacts`.
  - `p_find(idata, pool_size) -> xarray.DataArray`, which reads `lifetime_new` and `p_recog`.

- [ ] **Step 1: Add PreliZ**

Run: `uv add "preliz>=0.28"`
Expected: `pyproject.toml` lists `preliz>=0.28`. Then `uv run python -c "import preliz, pymc, pytensor, arviz; print(preliz.__version__, pymc.__version__, pytensor.__version__, arviz.__version__)"` prints `0.28.0 6.4.0 3.3.7 1.3.0` or newer patch versions.

- [ ] **Step 2: Write the failing tests**

Replace `tests/test_recovery.py` with:

```python
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from soulmates.model import build_model, p_find
from soulmates.simulate import simulate_diaries

# Made up true values for the test, one mean per 5 year band, peaking at ages 20 to 24
TRUE_BAND_MEANS = [1.0, 1.5, 2.0, 3.0, 4.0, 3.5, 3.0, 2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0]
TRUE_ALPHA = 2.0
TRUE_WEEKEND = 1.4  # made up: a weekend day has 1.4 times the new contacts of a weekday
TRUE_BETA_WEEKEND = 0.3365  # log(1.4)
# Ages 18 to 60 by hand: 18 to 19 is 2 × 3.0, then 5 years each of 4.0, 3.5, 3.0, 2.5, 2.5, 2.0, 2.0, 1.5,
# then age 60 is 1 × 1.5. That is 6 + 20 + 17.5 + 15 + 12.5 + 12.5 + 10 + 10 + 7.5 + 1.5 = 112.5, times 365 days
# is 41062.5 weekday contacts. An average day of the week is (5 × 1 + 2 × 1.4) / 7 = 7.8 / 7 of a weekday,
# so the lifetime count is 41062.5 × 7.8 / 7.
TRUE_LIFETIME_NEW = 45755.357


def test_recovers_true_values():
    df = simulate_diaries(3000, TRUE_BAND_MEANS, TRUE_ALPHA, seed=1, weekend_effect=TRUE_WEEKEND)
    with build_model(df):
        post = pm.sample(random_seed=2, progressbar=False).posterior
    # Limits from trials over six seeds, where the largest errors were 14%, 4.5%, 26% and 0.06
    assert float(post["alpha"].mean()) == pytest.approx(TRUE_ALPHA, rel=0.25)
    assert float(post["lifetime_new"].mean()) == pytest.approx(TRUE_LIFETIME_NEW, rel=0.10)
    assert post["band_mean"].mean(("chain", "draw")).values == pytest.approx(TRUE_BAND_MEANS, rel=0.40)
    assert float(post["beta_weekend"].mean()) == pytest.approx(TRUE_BETA_WEEKEND, abs=0.15)


def test_p_find_by_hand():
    # 2 new people, sure recognition, pool of 4: 1 − (3/4)² = 7/16
    idata = az.from_dict({"posterior": {"lifetime_new": np.array([[2.0]]), "p_recog": np.array([[1.0]])}})
    assert p_find(idata, pool_size=4).item() == pytest.approx(7 / 16)


def test_recog_prior_matches_the_guess():
    # The guess "90% sure p_recog lies between 0.1 and 0.6" must put 90% of the prior draws in that range.
    # With 4000 draws, chance alone moves the share by about 0.005.
    df = pd.DataFrame({"age": [30], "new_contacts": [1], "weekend": [False]})
    with build_model(df, recog_bounds=(0.1, 0.6), recog_mass=0.9):
        p_recog = pm.sample_prior_predictive(draws=4000, random_seed=3).prior["p_recog"].values
    assert ((p_recog > 0.1) & (p_recog < 0.6)).mean() == pytest.approx(0.90, abs=0.02)
```

- [ ] **Step 3: Run them and see them fail**

Run: `uv run pytest tests/test_recovery.py -v`
Expected: `test_p_find_by_hand` FAILED with a `KeyError` on `eye_contacts`, and `test_recog_prior_matches_the_guess` FAILED with `TypeError: build_model() got an unexpected keyword argument 'recog_bounds'`. `test_recovers_true_values` still passes.

- [ ] **Step 4: Change the model**

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
        # The speaker's guess, "I am recog_mass sure p_recog lies between the bounds", turned into the
        # flattest Beta prior that meets it (maximum entropy), so the answer is unique
        recog = pz.Beta()
        pz.maxent(recog, lower=recog_bounds[0], upper=recog_bounds[1], mass=recog_mass, plot=False)
        pm.Beta("p_recog", alpha=float(recog.alpha), beta=float(recog.beta))
    return model


def p_find(idata, pool_size):
    post = idata.posterior
    # 1 − (1 − p_recog / M) ^ lifetime_new, written so a tiny probability does not round to zero
    return -np.expm1(post["lifetime_new"] * np.log1p(-post["p_recog"] / pool_size))
```

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv run pytest tests/test_recovery.py -v`
Expected: 3 passed, with no `FutureWarning`.

- [ ] **Step 6: Stop for review.** Do not commit.

---

### Task 2: Country column

**Files:**
- Modify: `scripts/download_polymod.py`
- Modify: `src/soulmates/polymod.py`
- Test: `tests/test_polymod.py`

**Interfaces:**
- Consumes: the four POLYMOD files in `data/raw/`.
- Produces: `load_polymod(raw_dir, max_drop=0.05) -> pandas.DataFrame` with the columns `age`, `new_contacts`, `weekend` and `country` (str, the two letter code from the household file).

- [ ] **Step 1: Add the household file to the download script**

In `scripts/download_polymod.py`, change "three" to "four" in the first line of the docstring. Add this entry as the last line of `FILES`:

```python
    "2008_Mossong_POLYMOD_hh_common.csv": "d7fb1359ad84dba8cce4c444063940aa",
```

Run: `uv run python scripts/download_polymod.py`
Expected: four lines ending in `: ok`.

- [ ] **Step 2: Write the failing test**

Replace `tests/test_polymod.py` with:

```python
import pandas as pd
import pytest

from soulmates.polymod import load_polymod

# Made up rows. Participant 3 has no age and participant 4 has no day of the week, so both are dropped.
PARTICIPANTS = "part_id,hh_id,part_age\n1,11,30\n2,12,25\n3,13,\n4,14,40\n5,15,50\n6,16,60\n7,17,70\n"
HOUSEHOLDS = "hh_id,country\n11,DE\n12,GB\n13,IT\n14,PL\n15,DE\n16,FI\n17,LU\n"
SDAY = "part_id,dayofweek\n1,0\n2,3\n3,2\n4,\n5,6\n6,1\n7,5\n"
# Participant 1 has two first time contacts, one daily contact and one contact with no frequency.
# Participant 2 has a contact met less often and one with no frequency. Participants 5 to 7 have none.
CONTACTS = "part_id,cont_id,frequency_multi\n1,1,5\n1,2,5\n1,3,1\n1,4,\n2,5,4\n2,6,\n"


@pytest.fixture
def raw_dir(tmp_path):
    files = {"participant_common": PARTICIPANTS, "sday": SDAY, "contact_common": CONTACTS, "hh_common": HOUSEHOLDS}
    for name, text in files.items():
        (tmp_path / f"2008_Mossong_POLYMOD_{name}.csv").write_text(text)
    return tmp_path


def test_load_polymod(raw_dir):
    df = load_polymod(raw_dir, max_drop=1.0)
    expected = pd.DataFrame({
        "age": [30, 25, 50, 60, 70],
        "new_contacts": [2, 0, 0, 0, 0],
        "weekend": [True, False, True, False, False],
        "country": ["DE", "GB", "DE", "FI", "LU"],
    })
    pd.testing.assert_frame_equal(df, expected)


def test_load_polymod_raises_above_max_drop(raw_dir):
    with pytest.raises(ValueError, match="no part_age"):
        load_polymod(raw_dir)
```

- [ ] **Step 3: Run it and see it fail**

Run: `uv run pytest tests/test_polymod.py -v`
Expected: `test_load_polymod` FAILED, because the result has no `country` column. `test_load_polymod_raises_above_max_drop` still passes.

- [ ] **Step 4: Change the loader**

`src/soulmates/polymod.py`, full file:

```python
"""Load the POLYMOD contact diaries into one row per participant."""
import pandas as pd

PREFIX = "2008_Mossong_POLYMOD_"


def load_polymod(raw_dir, max_drop=0.05):
    def read(name):
        return pd.read_csv(f"{raw_dir}/{PREFIX}{name}.csv", low_memory=False)

    part = (
        read("participant_common")
        .merge(read("sday")[["part_id", "dayofweek"]], on="part_id", how="left")
        .merge(read("hh_common")[["hh_id", "country"]], on="hh_id", how="left")
    )
    cont = read("contact_common")
    no_age, no_day = part["part_age"].isna(), part["dayofweek"].isna()
    rules = [("no part_age", no_age), ("no dayofweek", no_day), ("no frequency_multi", cont["frequency_multi"].isna())]
    for rule, mask in rules:
        print(f"{rule}: {mask.sum()} of {len(mask)} rows ({mask.mean():.1%})")
        if mask.mean() > max_drop:
            raise ValueError(f"{rule} removes {mask.mean():.1%} of rows, more than max_drop = {max_drop:.1%}")
    part = part[~(no_age | no_day)]
    # A missing frequency_multi is never equal to 5, so those contacts do not count as new
    new = cont[cont["frequency_multi"] == 5].groupby("part_id").size()
    return pd.DataFrame({
        "age": part["part_age"].astype(int).to_numpy(),
        "new_contacts": part["part_id"].map(new).fillna(0).astype(int).to_numpy(),
        "weekend": part["dayofweek"].isin([0, 6]).to_numpy(),  # 0 is Sunday, 6 is Saturday
        "country": part["country"].to_numpy(),
    })
```

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv run pytest tests/test_polymod.py -v`
Expected: 2 passed.

- [ ] **Step 6: Stop for review.** Do not commit.

---

### Task 3: Notebooks

**Files:**
- Modify: `notebooks/01_simulated.ipynb` (one cell)
- Modify: `notebooks/02_polymod.ipynb` (five cells changed, four cells added)

**Interfaces:**
- Consumes: `build_model` and `p_find` from Task 1, `load_polymod` from Task 2, and `pools` from the P_find cell of notebook 02.
- Produces: both notebooks, executed.

- [ ] **Step 1: Edit the notebooks**

Save this script as a scratch file outside the repo, for example `edit_nbs_m4.py`, and run it from the repo root with `uv run python edit_nbs_m4.py`. Each cell is found by how its text starts, and the script fails loudly if it does not find exactly one match.

```python
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def find(nb, prefix):
    hits = [i for i, c in enumerate(nb.cells) if c.source.startswith(prefix)]
    assert len(hits) == 1, f"expected one cell starting with {prefix!r}, found {len(hits)}"
    return hits[0]


# Notebook 01: new wording of the premise
path = "notebooks/01_simulated.ipynb"
nb = nbf.read(path, as_version=4)
nb.cells[find(nb, "# Soul mates with Bayes, on simulated data")] = md(
    "# Soul mates with Bayes, on simulated data\n\n"
    "Everyone has one soul mate, drawn at random from the M people born within 5 years of them. "
    "You find them only if you meet and talk to them and you both recognize each other. This notebook "
    "estimates how many new people you meet per day, turns that into new people met in a lifetime, and "
    "then into the chance of finding your soul mate. The data is simulated with known true values, so we "
    "can check the model.")
nbf.write(nb, path)
print("updated", path)

# Notebook 02
path = "notebooks/02_polymod.ipynb"
nb = nbf.read(path, as_version=4)
nb.cells[find(nb, "# Soul mates with Bayes, on the POLYMOD survey")] = md(
    "# Soul mates with Bayes, on the POLYMOD survey\n\n"
    "POLYMOD asked 7290 people in 8 European countries in 2005 and 2006 to write down everyone they met "
    "on one day (Mossong et al. 2008, Zenodo record 3874557, licence CC BY-NC 4.0). A contact counts when "
    "two people touch skin to skin, or talk face to face for at least three words. For each contact, "
    "people said how often they usually meet that person, and a contact marked \"first time\" is a new "
    "person. So this notebook counts new people you meet and talk to, not people you only glance at. "
    "Run `uv run python scripts/download_polymod.py` and `uv run python scripts/download_wpp.py` once "
    "before this notebook.")
nb.cells[find(nb, "model = build_model(df)")] = code(
    "# The speaker's guess: \"I am 90% sure that p_recog lies between 0.1 and 0.6\"\n"
    "RECOG_BOUNDS = (0.1, 0.6)\n"
    "RECOG_MASS = 0.9\n\n"
    "model = build_model(df, recog_bounds=RECOG_BOUNDS, recog_mass=RECOG_MASS)\n"
    "with model:\n"
    "    idata = pm.sample(random_seed=2)\n"
    "az.summary(idata, var_names=[\"mu_pop\", \"sigma_band\", \"alpha\", \"beta_weekend\", \"lifetime_new\", \"p_recog\"])")
i = find(nb, "## The chance of finding your soul mate")
nb.cells[i:i] = [
    md("## The guess behind p_recog\n\n"
       "p_recog is the chance that, when you meet your soul mate and talk for the first time, you both "
       "realize it then. No data informs it. The guess \"I am 90% sure it lies between 0.1 and 0.6\" "
       "becomes a Beta prior through PreliZ `maxent`, which picks the flattest Beta curve that puts 90% "
       "between the two bounds. Mossong et al. report that about three quarters of first time contacts "
       "lasted less than 15 minutes, which is why the guess sits low. The posterior of p_recog equals "
       "its prior, so the draws below show the prior."),
    code("p_recog = idata.posterior[\"p_recog\"].values.ravel()\n"
         "inside = ((p_recog > RECOG_BOUNDS[0]) & (p_recog < RECOG_BOUNDS[1])).mean()\n"
         "fig, ax = plt.subplots()\n"
         "ax.hist(p_recog, bins=60, density=True)\n"
         "for bound in RECOG_BOUNDS:\n"
         "    ax.axvline(bound, color=\"k\", linestyle=\"--\")\n"
         "ax.set(xlabel=\"p_recog\", ylabel=\"density\", "
         "title=f\"Prior for p_recog, {inside:.0%} of draws between the bounds\")\n"
         "print(f\"p_recog mean {p_recog.mean():.3f}, share between the bounds {inside:.3f}\")"),
]
nb.cells[find(nb, "## The chance of finding your soul mate")] = md(
    "## The chance of finding your soul mate\n\n"
    "P_find = 1 − (1 − p_recog / M) ^ lifetime_new. M is everyone alive in 2023 aged 25 to 35, from the UN "
    "World Population Prospects 2024 (licence CC BY 3.0 IGO). p_recog is the only guessed number left, and "
    "it drives the spread of the result.")
i = find(nb, "## Known limits")
nb.cells[i:i] = [
    md("## How much does Germany matter?\n\n"
       "Diaries in Belgium, Germany, Finland and the Netherlands asked people to give an estimate of their "
       "professional contacts instead of listing each one (Mossong et al. 2008, Discussion). Those contacts "
       "have no frequency, so none of them can count as first time. Germany records far fewer new contacts "
       "than the other countries, and it is the largest sample. This cell fits the model again without "
       "Germany."),
    code("no_de = df[df[\"country\"] != \"DE\"]\n"
         "with build_model(no_de, recog_bounds=RECOG_BOUNDS, recog_mass=RECOG_MASS):\n"
         "    idata_no_de = pm.sample(random_seed=2)\n"
         "for label, d in [(\"all 8 countries\", idata), (\"without Germany\", idata_no_de)]:\n"
         "    life = d.posterior[\"lifetime_new\"].values.ravel()\n"
         "    pf = p_find(d, pools[\"world\"]).values.ravel()\n"
         "    print(f\"{label}: lifetime new contacts {life.mean():.0f} \"\n"
         "          f\"({np.percentile(life, 3):.0f} to {np.percentile(life, 97):.0f}), \"\n"
         "          f\"world P_find mean {pf.mean():.2e}, 1 in {1 / pf.mean():,.0f}\")"),
]
nb.cells[find(nb, "## Known limits")] = md(
    "## Known limits\n\n"
    "1. Four countries asked for an estimate of professional contacts instead of a list, and Germany "
    "records far fewer new contacts than the rest. The cell above shows how much that moves the result.\n"
    "2. The recovery test runs at alpha = 2, while this data sits near alpha = 0.23. The model is not "
    "tested at that level of spread.\n"
    "3. All 8 countries are pooled into one sample.\n"
    "4. The pool counts everyone and ignores gender and orientation.\n"
    "5. M is a fixed number from the WPP estimates, without uncertainty.\n"
    "6. Strangers you only glance at are not counted. The premise is people you meet and talk to.")
nbf.write(nb, path)
print("updated", path)
```

Expected: `updated notebooks/01_simulated.ipynb` and `updated notebooks/02_polymod.ipynb`

- [ ] **Step 2: Execute both notebooks**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_simulated.ipynb`
Expected: exits without error.

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_polymod.ipynb`
Expected: exits without error after about 4 minutes. In the scratch trial it printed:

```
p_recog mean 0.356, share between the bounds 0.904
world: M = 1,328,956,521, P_find mean 5.53e-06 (1 in 180,740), 94% interval 1.60e-06 to 1.04e-05
8 POLYMOD countries: M = 39,411,641, P_find mean 1.87e-04 (1 in 5,361), 94% interval 5.39e-05 to 3.49e-04
all 8 countries: lifetime new contacts 20665 (19212 to 22180), world P_find mean 5.53e-06, 1 in 180,740
without Germany: lifetime new contacts 24129 (22455 to 25920), world P_find mean 6.41e-06, 1 in 155,990
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: 8 passed.

- [ ] **Step 4: Stop for review.** Do not commit.
