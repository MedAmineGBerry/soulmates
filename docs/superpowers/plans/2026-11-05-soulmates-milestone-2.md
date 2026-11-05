# Soul mates milestone 2 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the simulated diaries with the real POLYMOD survey: download script, loader, weekend term in the model, and a second notebook.

**Architecture:** `scripts/download_polymod.py` fetches three Zenodo files into `data/raw/`. `src/soulmates/polymod.py` turns them into one row per participant with `age`, `new_contacts` and `weekend`. `build_model` and `simulate_diaries` gain the weekend term. The notebook calls only these functions.

**Tech Stack:** as milestone 1. Python 3.13, uv, PyMC 6.4, PyTensor 3.3.7, ArviZ 1.3, pandas, matplotlib, pytest. The download uses only the standard library.

**Spec:** `docs/superpowers/specs/2026-11-05-soulmates-milestone-2-design.md`

## Global constraints

- Everything from the milestone 1 plan still holds. That covers the version floors, the ArviZ 1.x API and `uv run` for every command. It also covers the writing rules for comments and markdown, and marking made up values as made up.
- Do not commit. The user commits when they ask.
- `data/` stays out of git. It is already in `.gitignore`.
- Every code block below already ran in a scratch copy of the repo on 2026-11-05.

## File structure

```
scripts/download_polymod.py    # new: download and MD5 check
src/soulmates/polymod.py       # new: load_polymod
src/soulmates/model.py         # changed: beta_weekend in build_model
src/soulmates/simulate.py      # changed: weekend_effect in simulate_diaries
tests/test_polymod.py          # new: loader tests on hand written CSV rows
tests/test_recovery.py         # changed: weekend in the recovery test
notebooks/02_polymod.ipynb     # new: the real data walkthrough
notebooks/01_simulated.ipynb   # rerun only, its numbers change
```

---

### Task 1: Download script

**Files:**
- Create: `scripts/download_polymod.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the three files `2008_Mossong_POLYMOD_participant_common.csv`, `2008_Mossong_POLYMOD_contact_common.csv` and `2008_Mossong_POLYMOD_sday.csv` in `data/raw/`, each matching its MD5 checksum.

This script talks to the network, so it has no unit test. Step 2 checks it by hand.

- [ ] **Step 1: Write the script**

`scripts/download_polymod.py`:

```python
"""Download the three POLYMOD files from Zenodo into data/raw and check their MD5 checksums."""
import hashlib
import pathlib
import urllib.request

URL = "https://zenodo.org/api/records/3874557/files/{}/content"  # POLYMOD version 2, licence CC BY-NC 4.0
FILES = {
    "2008_Mossong_POLYMOD_participant_common.csv": "ac784c281a71e15c7e64eaf2d6365709",
    "2008_Mossong_POLYMOD_contact_common.csv": "52baf32033bf780786fae6604043ec00",
    "2008_Mossong_POLYMOD_sday.csv": "148c5fc86ca10d385e3d047014e14a09",
}
RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, md5 in FILES.items():
        path = RAW / name
        if not path.exists():
            urllib.request.urlretrieve(URL.format(name), path)
        if hashlib.md5(path.read_bytes()).hexdigest() != md5:
            raise SystemExit(f"{name}: checksum does not match Zenodo. Delete {path} and run again.")
        print(f"{name}: ok")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python scripts/download_polymod.py`
Expected: three lines ending in `: ok`. The files are already in `data/raw/` from the design step, so the script only checks them. In the scratch trial it also downloaded them fresh into an empty folder. After a file was changed by hand, it stopped with exit code 1.

- [ ] **Step 3: Stop for review.** Do not commit.

---

### Task 2: Loader

**Files:**
- Create: `src/soulmates/polymod.py`
- Test: `tests/test_polymod.py`

**Interfaces:**
- Consumes: the three CSV files in a folder, as written by Task 1.
- Produces: `load_polymod(raw_dir, max_drop=0.05) -> pandas.DataFrame` with columns `age` (int64), `new_contacts` (int64) and `weekend` (bool), one row per kept participant, in the order of `participant_common.csv`. It raises `ValueError` naming the rule when a rule removes a larger share than `max_drop`.

- [ ] **Step 1: Write the failing test**

`tests/test_polymod.py`:

```python
import pandas as pd
import pytest

from soulmates.polymod import load_polymod

# Made up rows. Participant 3 has no age and participant 4 has no day of the week, so both are dropped.
PARTICIPANTS = "part_id,part_age\n1,30\n2,25\n3,\n4,40\n5,50\n6,60\n7,70\n"
SDAY = "part_id,dayofweek\n1,0\n2,3\n3,2\n4,\n5,6\n6,1\n7,5\n"
# Participant 1 has two first time contacts, one daily contact and one contact with no frequency.
# Participant 2 has a contact met less often and one with no frequency. Participants 5 to 7 have none.
CONTACTS = "part_id,cont_id,frequency_multi\n1,1,5\n1,2,5\n1,3,1\n1,4,\n2,5,4\n2,6,\n"


@pytest.fixture
def raw_dir(tmp_path):
    for name, text in [("participant_common", PARTICIPANTS), ("sday", SDAY), ("contact_common", CONTACTS)]:
        (tmp_path / f"2008_Mossong_POLYMOD_{name}.csv").write_text(text)
    return tmp_path


def test_load_polymod(raw_dir):
    df = load_polymod(raw_dir, max_drop=1.0)
    expected = pd.DataFrame({
        "age": [30, 25, 50, 60, 70],
        "new_contacts": [2, 0, 0, 0, 0],
        "weekend": [True, False, True, False, False],
    })
    pd.testing.assert_frame_equal(df, expected)


def test_load_polymod_raises_above_max_drop(raw_dir):
    with pytest.raises(ValueError, match="no part_age"):
        load_polymod(raw_dir)
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_polymod.py -v`
Expected: collection error with `ModuleNotFoundError: No module named 'soulmates.polymod'`

- [ ] **Step 3: Write the loader**

`src/soulmates/polymod.py`:

```python
"""Load the POLYMOD contact diaries into one row per participant."""
import pandas as pd

PREFIX = "2008_Mossong_POLYMOD_"


def load_polymod(raw_dir, max_drop=0.05):
    def read(name):
        return pd.read_csv(f"{raw_dir}/{PREFIX}{name}.csv", low_memory=False)

    part = read("participant_common").merge(read("sday")[["part_id", "dayofweek"]], on="part_id", how="left")
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
    })
```

- [ ] **Step 4: Run it and see it pass**

Run: `uv run pytest tests/test_polymod.py -v`
Expected: both tests PASSED.

- [ ] **Step 5: Run it on the real files**

Run: `uv run python -c "from soulmates.polymod import load_polymod; df = load_polymod('data/raw'); print(df.shape, df.new_contacts.sum())"`
Expected:

```
no part_age: 65 of 7290 rows (0.9%)
no dayofweek: 136 of 7290 rows (1.9%)
no frequency_multi: 1407 of 97904 rows (1.4%)
(7105, 3) 7359
```

- [ ] **Step 6: Stop for review.** Do not commit.

---

### Task 3: Weekend term in the model and the simulator

**Files:**
- Modify: `src/soulmates/model.py` (`build_model` only)
- Modify: `src/soulmates/simulate.py`
- Test: `tests/test_recovery.py`

**Interfaces:**
- Consumes: `age_band` and `p_find` from milestone 1, unchanged.
- Produces:
  - `build_model(df, search_ages=(18, 60)) -> pymc.Model`. `df` must have `age`, `new_contacts` and `weekend`. The model gains the free variable `beta_weekend`. All other names stay as in milestone 1.
  - `simulate_diaries(n_people, band_means, alpha, seed, weekend_effect=1.0) -> pandas.DataFrame` with columns `age`, `new_contacts` and `weekend`.

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
    # 2 eye contacts, sure recognition, pool of 4: 1 − (3/4)² = 7/16
    idata = az.from_dict({"posterior": {"eye_contacts": np.array([[2.0]]), "p_recog": np.array([[1.0]])}})
    assert p_find(idata, pool_size=4).item() == pytest.approx(7 / 16)
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_recovery.py -v`
Expected: `test_recovers_true_values` FAILED with `TypeError: simulate_diaries() got an unexpected keyword argument 'weekend_effect'`. `test_p_find_by_hand` still passes.

- [ ] **Step 3: Change the simulator**

`src/soulmates/simulate.py`, full file:

```python
"""Simulated POLYMOD style contact diaries with known true values."""
import numpy as np
import pandas as pd

from soulmates.model import age_band


def simulate_diaries(n_people, band_means, alpha, seed, weekend_effect=1.0):
    rng = np.random.default_rng(seed)
    age = rng.integers(0, 85, size=n_people)
    weekend = rng.random(n_people) < 2 / 7
    mu = np.asarray(band_means)[age_band(age)] * np.where(weekend, weekend_effect, 1.0)
    # numpy uses n and p for the negative binomial: n = alpha and p = alpha / (alpha + mu) give mean mu
    new_contacts = rng.negative_binomial(alpha, alpha / (alpha + mu))
    return pd.DataFrame({"age": age, "new_contacts": new_contacts, "weekend": weekend})
```

- [ ] **Step 4: Change the model**

In `src/soulmates/model.py`, replace the whole `build_model` function with:

```python
def build_model(df, search_ages=(18, 60)):
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
        lifetime_new = pm.Deterministic("lifetime_new", 365 * week * pt.sum(years_per_band * band_mean))
        p_eye = pm.Beta("p_eye", 2, 2)  # guess, no data informs it
        pm.Deterministic("eye_contacts", lifetime_new * p_eye)
        pm.Beta("p_recog", 2, 2)  # guess, no data informs it
    return model
```

`N_BANDS`, `age_band` and `p_find` stay unchanged.

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv run pytest -v`
Expected: 4 passed, in about 25 seconds. In the scratch trial over six seeds, the sampler reported 0 divergences every time.

- [ ] **Step 6: Stop for review.** Do not commit.

---

### Task 4: Real data notebook

**Files:**
- Create: `notebooks/02_polymod.ipynb`
- Rerun: `notebooks/01_simulated.ipynb`

**Interfaces:**
- Consumes: `load_polymod` from Task 2, and `age_band`, `build_model` and `p_find` from Task 3.
- Produces: an executed notebook with three figures and printed summaries.

- [ ] **Step 1: Build the notebook**

Run this once from the repo root:

```bash
uv run python - <<'EOF'
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Soul mates with Bayes, on the POLYMOD survey\n\n"
       "POLYMOD asked 7290 people in 8 European countries in 2005 and 2006 to write down everyone they met "
       "on one day (Mossong et al. 2008, Zenodo record 3874557, licence CC BY-NC 4.0). For each contact they "
       "said how often they usually meet that person. A contact marked \"first time\" is a new person. "
       "Run `uv run python scripts/download_polymod.py` once before this notebook."),
    code("import arviz as az\n"
         "import matplotlib.pyplot as plt\n"
         "import numpy as np\n"
         "import pymc as pm\n\n"
         "from soulmates.model import age_band, build_model, p_find\n"
         "from soulmates.polymod import load_polymod\n\n"
         "df = load_polymod(\"../data/raw\")\n"
         "df.head()"),
    code("raw = df.groupby([age_band(df[\"age\"]), \"weekend\"])[\"new_contacts\"].mean().unstack()\n"
         "fig, ax = plt.subplots()\n"
         "ax.plot(raw.index * 5, raw[False], \"o-\", label=\"weekday\")\n"
         "ax.plot(raw.index * 5, raw[True], \"s-\", label=\"weekend\")\n"
         "ax.set(xlabel=\"start of the 5 year age band\", ylabel=\"new contacts per day\", "
         "title=\"Raw mean of new contacts by age band, POLYMOD\")\n"
         "ax.legend();"),
    md("## Fit\n\nThe model is the one from the simulated notebook, plus a weekend term `beta_weekend`. "
       "A weekend day has exp(beta_weekend) times the new contacts of a weekday."),
    code("model = build_model(df)\n"
         "with model:\n"
         "    idata = pm.sample(random_seed=2)\n"
         "az.summary(idata, var_names=[\"mu_pop\", \"sigma_band\", \"alpha\", \"beta_weekend\", \"lifetime_new\"])"),
    code("hdi = az.hdi(idata, var_names=[\"band_mean\"], prob=0.94)[\"band_mean\"].values\n"
         "mean = idata.posterior[\"band_mean\"].mean((\"chain\", \"draw\")).values\n"
         "ages = np.arange(17) * 5\n"
         "fig, ax = plt.subplots()\n"
         "ax.errorbar(ages, mean, yerr=[mean - hdi[:, 0], hdi[:, 1] - mean], fmt=\"o\", "
         "label=\"posterior mean and 94% HDI\")\n"
         "ax.plot(raw.index * 5, raw[False], \"kx\", label=\"raw weekday mean\")\n"
         "ax.set(xlabel=\"start of the 5 year age band\", ylabel=\"new contacts per weekday\", "
         "title=\"Posterior band means, POLYMOD\")\n"
         "ax.legend();"),
    code("multiplier = np.exp(idata.posterior[\"beta_weekend\"].values.ravel())\n"
         "lifetime = idata.posterior[\"lifetime_new\"].values.ravel()\n"
         "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))\n"
         "ax1.hist(multiplier, bins=60)\n"
         "ax1.set(xlabel=\"weekend multiplier exp(beta_weekend)\", ylabel=\"posterior draws\", "
         "title=\"Weekend against weekday\")\n"
         "ax2.hist(lifetime, bins=60)\n"
         "ax2.set(xlabel=\"new people met between ages 18 and 60\", ylabel=\"posterior draws\", "
         "title=\"Lifetime new contacts\")\n"
         "print(f\"weekend multiplier {multiplier.mean():.2f}, 94% interval {np.percentile(multiplier, 3):.2f} "
         "to {np.percentile(multiplier, 97):.2f}\")\n"
         "print(f\"lifetime new contacts {lifetime.mean():.0f}, 94% interval {np.percentile(lifetime, 3):.0f} "
         "to {np.percentile(lifetime, 97):.0f}\")"),
    md("## The chance of finding your soul mate\n\n"
       "P_find = 1 − (1 − p_recog / M) ^ eye_contacts. M is still a placeholder until the UN World "
       "Population Prospects cohort sizes are in, and p_eye and p_recog are still guesses."),
    code("POOL_SIZE = 1e8  # PLACEHOLDER until the UN WPP cohort sizes arrive\n"
         "pf = p_find(idata, POOL_SIZE).values.ravel()\n"
         "print(f\"P_find mean {pf.mean():.2e}, 94% interval {np.percentile(pf, 3):.2e} to {np.percentile(pf, 97):.2e}\")"),
    md("## Known limits\n\n"
       "1. Some participants may have reported extra contacts only as a total, without a frequency. If so, "
       "those contacts are missing here and the count of new contacts is too low. This has not been checked. "
       "The columns would be in `2008_Mossong_POLYMOD_participant_extra.csv`, which is not downloaded.\n"
       "2. The recovery test runs at alpha = 2, while this data sits near alpha = 0.23. The model is not "
       "tested at that level of spread.\n"
       "3. All 8 countries are pooled into one sample."),
]
nbf.write(nbf.v4.new_notebook(cells=cells), "notebooks/02_polymod.ipynb")
EOF
```

- [ ] **Step 2: Execute both notebooks**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_polymod.ipynb`
Expected: exits without error. In the scratch trial the same code printed `weekend multiplier 1.25, 94% interval 1.10 to 1.41`, `lifetime new contacts 20688, 94% interval 19264 to 22210` and `P_find mean 5.12e-05, 94% interval 5.73e-06 to 1.29e-04`, with 0 divergences. Different library builds may change the last digits.

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_simulated.ipynb`
Expected: exits without error. Its numbers change from milestone 1, because the simulator now draws the weekend flag before the counts, which shifts the random stream.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: 4 passed.

- [ ] **Step 4: Stop for review.** Do not commit.
