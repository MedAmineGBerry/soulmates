# Soul mates milestone 7 implementation plan: fixing the calculus

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Count only new people inside the age window and give each country its own contact level. Let people differ in the partner model. Rebuild the demo and the notebooks on top of that.

**Architecture:** `polymod.py` and the new `france.py` load diaries with an age window. `model.py` gains a country level and three lifetime counts. `match.py` gains `partner_chance` and a model with q and k. `roulette.py` gives each life its own contact rate and a country. The notebooks and the demo inputs are rebuilt from these.

**Tech Stack:** as milestones 1 to 6. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-10-soulmates-milestone-7-design.md`

## Global constraints

- Everything from the earlier plans still holds. That covers the version floors, the ArviZ 1.x API, `uv run` for every command, the writing rules and marking made up values as made up.
- Do not commit. The user commits when they ask.
- `python3` on this machine is the system Python 3.9. Run the writing checker with `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`, and everything else with `uv run`.
- The France files are already in `data/raw/france2012/`. `data/` stays out of git, apart from `data/demo_inputs.json`.
- Every code block below ran in a scratch copy of the repo on 2026-11-10: 19 tests passed and the three notebooks executed. The new app file was linted with `marimo check`, and its `spin` logic was run, but the app was not clicked through in a browser. Task 5 does that.

## File structure

```
src/soulmates/polymod.py          # changed: contact_age, age window in load_polymod
src/soulmates/france.py           # new: load_france_days, load_diaries
scripts/download_france.py        # new: download and MD5 check
src/soulmates/model.py            # changed: country level, lifetime_typical, lifetime_country, lifetime_anywhere
src/soulmates/simulate.py         # changed: country_factors
src/soulmates/match.py            # changed: partner_chance, model with q and k, flat model; soul mate model removed
src/soulmates/roulette.py         # changed: personal rate, country, people_met
scripts/export_demo_inputs.py     # changed: country model, k draws, country factors
data/demo_inputs.json             # regenerated
demo/roulette.py                  # changed: country choice, new texts
tests/test_polymod.py, test_france.py, test_recovery.py, test_match.py, test_roulette.py
notebooks/01_simulated.ipynb, 02_polymod.ipynb, 03_partners.ipynb   # rebuilt
.gitignore                        # changed: .DS_Store
```

Between tasks the repo is not fully consistent. After Task 4 the old notebook 03 and the old export script no longer run. Tasks 5 and 6 repair them.

---

### Task 1: Age window in the POLYMOD loader

**Files:**
- Modify: `src/soulmates/polymod.py`
- Test: `tests/test_polymod.py`

**Interfaces:**
- Consumes: the four POLYMOD files in `data/raw/`.
- Produces:
  - `contact_age(cont) -> pandas.Series`, the exact contact age or the middle of the estimated range.
  - `load_polymod(raw_dir, window=5, max_drop=0.05) -> pandas.DataFrame` with the columns `age`, `new_contacts`, `weekend`, `country`. `new_contacts` counts first time contacts within `window` years of the participant's age.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_polymod.py` with:

```python
import pandas as pd
import pytest

from soulmates.polymod import load_polymod

# Made up rows. Participant 3 has no age and participant 4 has no day of the week, so both are dropped.
PARTICIPANTS = "part_id,hh_id,part_age\n1,11,30\n2,12,25\n3,13,\n4,14,40\n5,15,50\n6,16,60\n7,17,70\n"
HOUSEHOLDS = "hh_id,country\n11,DE\n12,GB\n13,IT\n14,PL\n15,DE\n16,FI\n17,LU\n"
SDAY = "part_id,dayofweek\n1,0\n2,3\n3,2\n4,\n5,6\n6,1\n7,5\n"
# Participant 1 is 30. Their first time contacts (code 5): one aged 31, one aged 36, one with an estimated
# range of 28 to 34 (middle 31) and one with no age. They also have a daily contact and one with no frequency.
# Participant 2 has a contact met less often and one with no frequency. Participants 5 to 7 have none.
CONTACTS = (
    "part_id,cont_id,cnt_age_exact,cnt_age_est_min,cnt_age_est_max,frequency_multi\n"
    "1,1,31,,,5\n1,2,36,,,5\n1,3,,28,34,5\n1,4,,,,5\n1,5,30,,,1\n1,6,30,,,\n2,7,25,,,4\n2,8,25,,,\n"
)


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
        "new_contacts": [2, 0, 0, 0, 0],  # the contacts aged 31 and about 31; 36 is outside 5 years
        "weekend": [True, False, True, False, False],
        "country": ["DE", "GB", "DE", "FI", "LU"],
    })
    pd.testing.assert_frame_equal(df, expected)


def test_a_wider_window_counts_the_contact_aged_36(raw_dir):
    assert load_polymod(raw_dir, window=10, max_drop=1.0)["new_contacts"].tolist() == [3, 0, 0, 0, 0]


def test_load_polymod_raises_above_max_drop(raw_dir):
    with pytest.raises(ValueError, match="no part_age"):
        load_polymod(raw_dir)
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_polymod.py -v`
Expected: 2 failed, 1 passed. `test_load_polymod` fails with `AssertionError: DataFrame.iloc[:, 1] (column name="new_contacts") are different`. `test_a_wider_window_counts_the_contact_aged_36` fails with `TypeError: load_polymod() got an unexpected keyword argument 'window'`.

- [ ] **Step 3: Change the loader**

`src/soulmates/polymod.py`, full file:

```python
"""Load the POLYMOD contact diaries into one row per participant."""
import pandas as pd

PREFIX = "2008_Mossong_POLYMOD_"


def contact_age(cont):
    # the exact age where it is given, else the middle of the estimated range
    return cont["cnt_age_exact"].fillna((cont["cnt_age_est_min"] + cont["cnt_age_est_max"]) / 2)


def load_polymod(raw_dir, window=5, max_drop=0.05):
    def read(name):
        return pd.read_csv(f"{raw_dir}/{PREFIX}{name}.csv", low_memory=False)

    part = (
        read("participant_common")
        .merge(read("sday")[["part_id", "dayofweek"]], on="part_id", how="left")
        .merge(read("hh_common")[["hh_id", "country"]], on="hh_id", how="left")
    )
    cont = read("contact_common").merge(part[["part_id", "part_age"]], on="part_id")
    no_age, no_day = part["part_age"].isna(), part["dayofweek"].isna()
    rules = [("no part_age", no_age), ("no dayofweek", no_day), ("no frequency_multi", cont["frequency_multi"].isna())]
    for rule, mask in rules:
        print(f"{rule}: {mask.sum()} of {len(mask)} rows ({mask.mean():.1%})")
        if mask.mean() > max_drop:
            raise ValueError(f"{rule} removes {mask.mean():.1%} of rows, more than max_drop = {max_drop:.1%}")
    part = part[~(no_age | no_day)]
    # New means "first time" (code 5) and within `window` years of the participant's age, because the
    # soul mate pool only holds people of that age. A missing frequency or age compares as False.
    in_window = (contact_age(cont) - cont["part_age"]).abs() <= window
    new = cont[(cont["frequency_multi"] == 5) & in_window].groupby("part_id").size()
    return pd.DataFrame({
        "age": part["part_age"].astype(int).to_numpy(),
        "new_contacts": part["part_id"].map(new).fillna(0).astype(int).to_numpy(),
        "weekend": part["dayofweek"].isin([0, 6]).to_numpy(),  # 0 is Sunday, 6 is Saturday
        "country": part["country"].to_numpy(),
    })
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/test_polymod.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run it on the real files**

Run: `uv run python -c "from soulmates.polymod import load_polymod; df = load_polymod('data/raw'); print(df.shape, int(df.new_contacts.sum()))"`
Expected: the three lines about dropped rows as before, then `(7105, 4) 1760`.

- [ ] **Step 6: Stop for review.** Do not commit.

---

### Task 2: France 2012

**Files:**
- Create: `scripts/download_france.py`
- Create: `src/soulmates/france.py`
- Test: `tests/test_france.py`

**Interfaces:**
- Consumes: `contact_age` and `load_polymod` from Task 1.
- Produces:
  - `load_france_days(raw_dir, window=5) -> pandas.DataFrame`, one row per participant and diary day, with the columns `part_id`, `wave`, `day`, `age`, `new_contacts`, `weekend`, `country` ("FR").
  - `load_diaries(raw_dir, window=5) -> pandas.DataFrame`, POLYMOD and France together with the columns `age`, `new_contacts`, `weekend`, `country`. It expects the France files in `raw_dir/france2012`.

- [ ] **Step 1: Write the download script**

`scripts/download_france.py`:

```python
"""Download the four France 2012 files from Zenodo into data/raw/france2012 and check their MD5 checksums."""
import hashlib
import pathlib
import urllib.request

URL = "https://zenodo.org/api/records/3886590/files/{}/content"  # Béraud et al. 2015, version 3, CC BY-NC-ND 4.0
FILES = {
    "2015_Beraud_France_participant_common.csv": "8564fb60b58670c3e302e3aae58537f6",
    "2015_Beraud_France_contact_common.csv": "bf8ea2b9cb95b772868b1ceb5931e8d5",
    "2015_Beraud_France_contact_extra.csv": "1446fb7027517e731af91d9cba954af9",
    "2015_Beraud_France_sday.csv": "1fb1fb75df818a479e5e9e2f0a88635e",
}
RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "france2012"


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

Run: `uv run python scripts/download_france.py`
Expected: four lines ending in `: ok`. The files are already there from the design step, so the script only checks them.

- [ ] **Step 2: Write the failing test**

`tests/test_france.py`:

```python
import pandas as pd

from soulmates.france import load_france_days

# Made up rows. Participant 1 is 30 and kept a diary on two days: a Sunday (code 0) and a Monday (code 1).
PARTICIPANTS = "part_id,part_age\n1,30\n2,40\n"
SDAY = "part_id,wave,studyDay,dayofweek\n1,1,1,0\n1,1,2,1\n2,1,1,3\n"
# Day 1 of participant 1: two first time contacts, aged 29 and 50. Day 2: one first time contact aged 33.
CONTACTS = (
    "part_id,cont_id,cnt_age_exact,cnt_age_est_min,cnt_age_est_max,frequency_multi\n"
    "1,101,29,,,5\n1,102,50,,,5\n1,103,33,,,5\n2,104,41,,,1\n"
)
CONTACT_EXTRA = "cont_id,wave,studyDay\n101,1,1\n102,1,1\n103,1,2\n104,1,1\n"


def test_load_france_days(tmp_path):
    files = {"participant_common": PARTICIPANTS, "sday": SDAY, "contact_common": CONTACTS, "contact_extra": CONTACT_EXTRA}
    for name, text in files.items():
        (tmp_path / f"2015_Beraud_France_{name}.csv").write_text(text)
    expected = pd.DataFrame({
        "part_id": [1, 1, 2],
        "wave": [1, 1, 1],
        "day": [1, 2, 1],
        "age": [30, 30, 40],
        "new_contacts": [1, 1, 0],  # the contact aged 50 is outside 5 years
        "weekend": [True, False, False],
        "country": ["FR", "FR", "FR"],
    })
    pd.testing.assert_frame_equal(load_france_days(tmp_path), expected)
```

- [ ] **Step 3: Run it and see it fail**

Run: `uv run pytest tests/test_france.py -v`
Expected: collection error with `ModuleNotFoundError: No module named 'soulmates.france'`

- [ ] **Step 4: Write the loader**

`src/soulmates/france.py`:

```python
"""Load the France 2012 contact diaries (Béraud et al. 2015): one row per participant and diary day."""
import pandas as pd

from soulmates.polymod import contact_age, load_polymod

PREFIX = "2015_Beraud_France_"
COLUMNS = ["age", "new_contacts", "weekend", "country"]


def load_france_days(raw_dir, window=5):
    def read(name):
        return pd.read_csv(f"{raw_dir}/{PREFIX}{name}.csv", low_memory=False)

    part = read("participant_common")[["part_id", "part_age"]]
    days = read("sday").merge(part, on="part_id").dropna(subset=["part_age", "dayofweek"])
    # contact_extra says on which diary day (wave and studyDay) a contact was met
    cont = read("contact_common").merge(read("contact_extra"), on="cont_id").merge(part, on="part_id")
    in_window = (contact_age(cont) - cont["part_age"]).abs() <= window
    new = cont[(cont["frequency_multi"] == 5) & in_window].groupby(["part_id", "wave", "studyDay"]).size()
    days = days.merge(new.rename("new_contacts").reset_index(), on=["part_id", "wave", "studyDay"], how="left")
    return pd.DataFrame({
        "part_id": days["part_id"].to_numpy(),
        "wave": days["wave"].to_numpy(),
        "day": days["studyDay"].to_numpy(),
        "age": days["part_age"].astype(int).to_numpy(),
        "new_contacts": days["new_contacts"].fillna(0).astype(int).to_numpy(),
        "weekend": days["dayofweek"].isin([0, 6]).to_numpy(),  # 0 is Sunday, 6 is Saturday, as in POLYMOD
        "country": "FR",
    })


def load_diaries(raw_dir, window=5):
    # the 8 POLYMOD countries plus France, in the columns the contact model needs
    return pd.concat(
        [load_polymod(raw_dir, window)[COLUMNS], load_france_days(f"{raw_dir}/france2012", window)[COLUMNS]],
        ignore_index=True,
    )
```

- [ ] **Step 5: Run the test and see it pass**

Run: `uv run pytest tests/test_france.py -v`
Expected: 1 passed.

- [ ] **Step 6: Run it on the real files**

Run: `uv run python -c "from soulmates.france import load_diaries, load_france_days; f = load_france_days('data/raw/france2012'); d = load_diaries('data/raw'); print(f.shape, int(f.new_contacts.sum()), d.shape, int(d.new_contacts.sum()))"`
Expected, after the lines about dropped rows: `(4002, 7) 939 (11107, 4) 2699`

- [ ] **Step 7: Stop for review.** Do not commit.

---

### Task 3: Contact model with a country level

**Files:**
- Modify: `src/soulmates/model.py`
- Modify: `src/soulmates/simulate.py`
- Test: `tests/test_recovery.py`

**Interfaces:**
- Consumes: a DataFrame with `age`, `new_contacts`, `weekend`, `country`.
- Produces:
  - `recog_prior(bounds=(0.3, 0.9), mass=0.9) -> (alpha, beta)`.
  - `build_model(df, search_ages=(15, 60), recog_bounds=(0.3, 0.9), recog_mass=0.9) -> pymc.Model`. Free variables: `mu_pop`, `sigma_band`, `z`, `sigma_country`, `log_factor` (dim `country`), `beta_weekend`, `alpha`, `z_new`, `p_recog`. Deterministics: `band_mean` (dim `band`), `country_factor` (dim `country`), `lifetime_typical`, `lifetime_country` (dim `country`), `lifetime_anywhere`. There is no `lifetime_new` any more.
  - `p_find(idata, pool_size, lifetime="lifetime_anywhere") -> xarray.DataArray`.
  - `simulate_diaries(n_people, band_means, alpha, seed, weekend_effect=1.0, country_factors=(1.0,)) -> pandas.DataFrame` with the extra column `country` ("C0", "C1" and so on).

- [ ] **Step 1: Write the failing tests**

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
TRUE_ALPHA = 0.2  # close to the spread of the real diaries
TRUE_WEEKEND = 1.4  # made up: a weekend day has 1.4 times the new contacts of a weekday
TRUE_BETA_WEEKEND = 0.3365  # log(1.4)
# Made up: four countries from quiet to talkative. Their logs add up to zero, so 1.0 is the typical country.
TRUE_COUNTRY_FACTORS = [0.5, 0.8, 1.25, 2.0]
# Ages 15 to 60 by hand: 5 years each of 3.0, 4.0, 3.5, 3.0, 2.5, 2.5, 2.0, 2.0, 1.5, then age 60 is 1 × 1.5.
# That is 15 + 20 + 17.5 + 15 + 12.5 + 12.5 + 10 + 10 + 7.5 + 1.5 = 121.5, times 365 days is 44347.5
# weekday contacts. An average day of the week is (5 × 1 + 2 × 1.4) / 7 = 7.8 / 7 of a weekday.
TRUE_LIFETIME_TYPICAL = 49415.786  # 44347.5 × 7.8 / 7
TRUE_LIFETIME_COUNTRY = [24707.893, 39532.629, 61769.732, 98831.571]  # the line above times each factor


def test_recovers_true_values():
    df = simulate_diaries(6000, TRUE_BAND_MEANS, TRUE_ALPHA, seed=1, weekend_effect=TRUE_WEEKEND,
                          country_factors=TRUE_COUNTRY_FACTORS)
    with build_model(df):
        post = pm.sample(random_seed=2, progressbar=False).posterior
    mean = lambda name: post[name].mean(("chain", "draw")).values
    # Limits from a trial over six seeds, where the largest errors were 6%, 4%, 10%, 34% and 0.14
    assert float(mean("alpha")) == pytest.approx(TRUE_ALPHA, rel=0.25)
    assert float(mean("lifetime_typical")) == pytest.approx(TRUE_LIFETIME_TYPICAL, rel=0.12)
    assert mean("lifetime_country") == pytest.approx(TRUE_LIFETIME_COUNTRY, rel=0.25)
    assert mean("band_mean") == pytest.approx(TRUE_BAND_MEANS, rel=0.60)
    assert float(mean("beta_weekend")) == pytest.approx(TRUE_BETA_WEEKEND, abs=0.25)


def test_p_find_by_hand():
    # 2 new people, sure recognition, pool of 4: 1 − (3/4)² = 7/16
    idata = az.from_dict({"posterior": {"lifetime_anywhere": np.array([[2.0]]), "p_recog": np.array([[1.0]])}})
    assert p_find(idata, pool_size=4).item() == pytest.approx(7 / 16)


def test_recog_prior_matches_the_guess():
    # The guess "90% sure p_recog lies between 0.3 and 0.9" must put 90% of the prior draws in that range.
    # With 4000 draws, chance alone moves the share by about 0.005.
    df = pd.DataFrame({"age": [30, 30], "new_contacts": [1, 0], "weekend": [False, True], "country": ["A", "B"]})
    with build_model(df, recog_bounds=(0.3, 0.9), recog_mass=0.9):
        p_recog = pm.sample_prior_predictive(draws=4000, random_seed=3).prior["p_recog"].values
    assert ((p_recog > 0.3) & (p_recog < 0.9)).mean() == pytest.approx(0.90, abs=0.02)
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_recovery.py -v`
Expected: 2 failed, 1 passed. `test_recovers_true_values` fails with `TypeError: simulate_diaries() got an unexpected keyword argument 'country_factors'`. `test_p_find_by_hand` fails with a `KeyError` on `lifetime_new`. The prior test already passes, because the old model accepts the bounds.

- [ ] **Step 3: Change the model**

`src/soulmates/model.py`, full file:

```python
"""PyMC model for new contacts per day by age band and country, and the chance of finding your soul mate."""
import numpy as np
import preliz as pz
import pymc as pm
import pytensor.tensor as pt

N_BANDS = 17  # 5 year bands, the last one holds ages 80 and over


def age_band(age):
    return np.minimum(np.asarray(age) // 5, N_BANDS - 1)


def recog_prior(bounds=(0.3, 0.9), mass=0.9):
    # The speaker's guess, "I am mass sure p_recog lies between the bounds", turned into the
    # flattest Beta prior that meets it (maximum entropy), so the answer is unique
    recog = pz.Beta()
    pz.maxent(recog, lower=bounds[0], upper=bounds[1], mass=mass, plot=False)
    return float(recog.alpha), float(recog.beta)


def build_model(df, search_ages=(15, 60), recog_bounds=(0.3, 0.9), recog_mass=0.9):
    band = age_band(df["age"].to_numpy())
    weekend = df["weekend"].to_numpy().astype(float)
    countries, country = np.unique(df["country"].to_numpy().astype(str), return_inverse=True)
    search = np.arange(search_ages[0], search_ages[1] + 1)
    years_per_band = np.bincount(age_band(search), minlength=N_BANDS)  # search years that fall in each band
    with pm.Model(coords={"band": np.arange(N_BANDS), "country": countries}) as model:
        mu_pop = pm.Normal("mu_pop", np.log(0.4), 1)  # guess: about 0.4 new people of your age per day
        sigma_band = pm.HalfNormal("sigma_band", 0.5)
        z = pm.Normal("z", 0, 1, dims="band")  # non centred form
        band_mean = pm.Deterministic("band_mean", pt.exp(mu_pop + sigma_band * z), dims="band")  # typical country
        sigma_country = pm.HalfNormal("sigma_country", 0.5)  # how far countries sit from the typical one
        # The country terms sum to zero, so mu_pop is the average of the observed countries. Without
        # that, the overall level and the country levels can trade off and the sampler crawls.
        log_factor = pm.ZeroSumNormal("log_factor", sigma=sigma_country, dims="country")
        country_factor = pm.Deterministic("country_factor", pt.exp(log_factor), dims="country")
        beta_weekend = pm.Normal("beta_weekend", 0, 0.5)  # log of the weekend multiplier, centred on no effect
        alpha = pm.HalfNormal("alpha", 5)
        pm.NegativeBinomial(
            "new_contacts",
            mu=band_mean[band] * country_factor[country] * pt.exp(beta_weekend * weekend),
            alpha=alpha,
            observed=df["new_contacts"].to_numpy(),
        )
        week = 5 / 7 + 2 / 7 * pt.exp(beta_weekend)  # average day of a week with 5 weekdays and 2 weekend days
        typical = pm.Deterministic("lifetime_typical", 365 * week * pt.sum(years_per_band * band_mean))
        pm.Deterministic("lifetime_country", typical * country_factor, dims="country")
        # a country we have no data for: a new draw from the distribution the observed countries came from
        z_new = pm.Normal("z_new", 0, 1)
        pm.Deterministic("lifetime_anywhere", typical * pt.exp(sigma_country * z_new))
        a, b = recog_prior(recog_bounds, recog_mass)
        pm.Beta("p_recog", alpha=a, beta=b)
    return model


def p_find(idata, pool_size, lifetime="lifetime_anywhere"):
    post = idata.posterior
    # 1 − (1 − p_recog / M) ^ lifetime, written so a tiny probability does not round to zero
    return -np.expm1(post[lifetime] * np.log1p(-post["p_recog"] / pool_size))
```

- [ ] **Step 4: Change the simulator**

`src/soulmates/simulate.py`, full file:

```python
"""Simulated POLYMOD style contact diaries with known true values."""
import numpy as np
import pandas as pd

from soulmates.model import age_band


def simulate_diaries(n_people, band_means, alpha, seed, weekend_effect=1.0, country_factors=(1.0,)):
    rng = np.random.default_rng(seed)
    age = rng.integers(0, 85, size=n_people)
    weekend = rng.random(n_people) < 2 / 7
    country = rng.integers(len(country_factors), size=n_people)
    mu = np.asarray(band_means)[age_band(age)] * np.where(weekend, weekend_effect, 1.0) * np.asarray(country_factors)[country]
    # numpy uses n and p for the negative binomial: n = alpha and p = alpha / (alpha + mu) give mean mu
    new_contacts = rng.negative_binomial(alpha, alpha / (alpha + mu))
    return pd.DataFrame({"age": age, "new_contacts": new_contacts, "weekend": weekend, "country": [f"C{c}" for c in country]})
```

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv run pytest tests/test_recovery.py -v`
Expected: 3 passed, in about 70 seconds. In the scratch trial over six seeds every fit had 0 divergences.

- [ ] **Step 6: Stop for review.** Do not commit.

---

### Task 4: Partner model with q and k

**Files:**
- Modify: `src/soulmates/match.py`
- Test: `tests/test_match.py`

**Interfaces:**
- Consumes: `age_band` from `model.py`, the posterior of the contact model, and a DataFrame with `age` and `ever_partnered`.
- Produces:
  - `lifetime_curve(posterior, start_age=15) -> numpy array` of length 101, unchanged.
  - `partner_chance(n, q, k) -> float or numpy array`, the chance of having had a partner after `n` people for the average over all personal rates.
  - `build_match_model(df, curve) -> pymc.Model` with the free variables `log10_theta` and `k` and the deterministic `q`.
  - `build_flat_model(df, curve) -> pymc.Model` with `log10_theta` and the deterministic `q`.
  - `build_soulmate_model` no longer exists.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_match.py` with:

```python
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from soulmates.match import build_match_model, lifetime_curve, partner_chance


def test_lifetime_curve_by_hand():
    # Every band mean is 1 new person per day and there is no weekend effect, so a year adds 365
    posterior = az.from_dict(
        {"posterior": {"band_mean": np.ones((1, 1, 17)), "beta_weekend": np.zeros((1, 1))}},
        dims={"band_mean": ["band"]},
    ).posterior
    curve = lifetime_curve(posterior, start_age=15)
    assert curve[14] == 0
    assert curve[17] == pytest.approx(3 * 365)  # ages 15, 16 and 17


def test_partner_chance_by_hand():
    # q = 1 − 1/e makes theta = 1. With n = 1 and k = 1 the chance is 1 − (1 + 1) ^ (−1) = 0.5
    assert partner_chance(1, 1 - np.exp(-1), 1) == pytest.approx(0.5)
    assert partner_chance(4800, 0.0, 2) == 0  # nobody can become a partner
    assert partner_chance(10, 1.0, 2) == 1  # everybody can


def test_recovers_q_and_k():
    # Made up: q = 1 in 400, k = 3, and 150 new people per year from age 15
    true_q, true_k = 1 / 400, 3.0
    curve = np.maximum(np.arange(101) - 14, 0) * 150.0
    rng = np.random.default_rng(1)
    age = rng.integers(18, 61, size=6000)
    lam = rng.gamma(true_k, 1 / true_k, size=6000)  # how many people this person meets, relative to the average
    ever = rng.random(6000) < 1 - (1 - true_q) ** (lam * curve[age])
    with build_match_model(pd.DataFrame({"age": age, "ever_partnered": ever}), curve):
        post = pm.sample(random_seed=2, progressbar=False, target_accept=0.95).posterior
    # Limits from a trial over four seeds, where the largest errors were 6% for q and 10% for k
    assert post["q"].median().item() == pytest.approx(true_q, rel=0.20)
    assert post["k"].median().item() == pytest.approx(true_k, rel=0.35)
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_match.py -v`
Expected: collection error with `ImportError: cannot import name 'partner_chance' from 'soulmates.match'`

- [ ] **Step 3: Change the module**

`src/soulmates/match.py`, full file:

```python
"""Who has had a partner by which age: a match chance q, and a spread k in how many people each person meets."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from soulmates.model import age_band

AGES = np.arange(101)


def lifetime_curve(posterior, start_age=15):
    # N(a): new people met from start_age up to and including age a in a typical country, from the posterior means
    band_mean = posterior["band_mean"].mean(("chain", "draw")).values
    week = 5 / 7 + 2 / 7 * np.exp(float(posterior["beta_weekend"].mean()))
    per_year = 365 * week * band_mean[age_band(AGES)]
    per_year[AGES < start_age] = 0
    return np.cumsum(per_year)


def partner_chance(n, q, k):
    # Each person meets lam × n people, where lam follows a Gamma distribution with shape k and mean 1.
    # Averaging (1 − q) ^ (lam × n) over lam gives (1 + theta × n / k) ^ (−k), with theta = −log(1 − q).
    with np.errstate(divide="ignore"):  # q = 1 gives theta = inf, which still yields a chance of 1
        theta = -np.log1p(-np.asarray(q, dtype=float))
    return -np.expm1(-k * np.log1p(theta * n / k))


def _observe(df, x):
    # x = −log(chance of no partner yet). The chance of a partner is 1 − exp(−x). Passing its logit,
    # x + log(1 − exp(−x)), keeps the likelihood finite when x is large and the chance rounds to 1.
    pm.Bernoulli("ever_partnered", logit_p=x + pt.log1mexp(-x), observed=df["ever_partnered"].to_numpy().astype(int))


def _theta():
    # theta = −log(1 − q) can take any positive value, so q = 1 − exp(−theta) always stays below 1
    theta = 10 ** pm.Normal("log10_theta", -3, 2)  # wide: from 1 in 10 million to near certain
    pm.Deterministic("q", -pt.expm1(-theta))
    return theta


def build_match_model(df, curve):
    n = curve[df["age"].to_numpy()]
    with pm.Model() as model:
        theta = _theta()
        k = pm.LogNormal("k", 0, 1)  # centred on 1, from strong differences between people to almost none
        _observe(df, k * pt.log1p(theta * n / k))  # no partner yet with chance (1 + theta × n / k) ^ (−k)
    return model


def build_flat_model(df, curve):
    # the simpler model for comparison: everyone meets exactly N(a) people
    n = curve[df["age"].to_numpy()]
    with pm.Model() as model:
        _observe(df, _theta() * n)  # no partner yet with chance (1 − q) ^ n
    return model
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/test_match.py -v -W error::RuntimeWarning`
Expected: 3 passed, in about 15 seconds.

- [ ] **Step 5: Stop for review.** Do not commit.

---

### Task 5: Demo

**Files:**
- Modify: `src/soulmates/roulette.py`
- Test: `tests/test_roulette.py`
- Modify: `scripts/export_demo_inputs.py`
- Regenerate: `data/demo_inputs.json`
- Modify: `demo/roulette.py`

**Interfaces:**
- Consumes: `recog_prior` from Task 3, `partner_chance`, `build_match_model` and `lifetime_curve` from Task 4, `load_diaries` from Task 2.
- Produces:
  - `load_inputs(path) -> dict` with `curve`, `q`, `k`, `countries` (code to factor) and `pools`.
  - `spin(inputs, age, pool, recog_bounds, n_lives, seed, country="typical", recog_mass=0.9) -> pandas.DataFrame` with the columns `q`, `k`, `p_recog`, `people_met` (int), `found_partner`, `age_met` (Int64) and `found_soul_mate`. It raises `ValueError` for an age outside 18 to 60, an unknown pool or an unknown country.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_roulette.py` with:

```python
import numpy as np
import pytest

from soulmates.match import partner_chance
from soulmates.roulette import spin

# Made up inputs: 300 new people per year from age 15, a world pool of 1e9 and a European pool of 4e7 at
# every age, and one country that is twice as talkative as the typical one
CURVE = np.maximum(np.arange(101) - 14, 0) * 300.0
POOLS = {"world": {age: 1e9 for age in range(18, 61)}, "europe": {age: 4e7 for age in range(18, 61)}}
ARGS = dict(age=30, pool="world", recog_bounds=(0.3, 0.9), n_lives=1000, seed=1)


def inputs_with(q, k):
    return {"curve": CURVE, "q": np.full(4000, q), "k": np.full(4000, k), "countries": {"XX": 2.0}, "pools": POOLS}


def test_nobody_finds_anyone_with_q_zero():
    lives = spin(inputs_with(0.0, 3.0), **ARGS)
    assert not lives["found_partner"].any()
    assert lives["age_met"].isna().all()


def test_everyone_who_meets_someone_finds_a_partner_with_q_one():
    lives = spin(inputs_with(1.0, 3.0), **ARGS)
    met = lives["people_met"] >= 1
    assert lives.loc[met, "found_partner"].all()
    assert not lives.loc[~met, "found_partner"].any()
    assert lives.loc[met, "age_met"].between(15, 30).all()


def test_share_found_matches_the_formula():
    # At age 30 the average life has met 16 × 300 = 4800 people. With q = 1 in 4800 and k = 2, the formula
    # gives the share below. With 1000 lives, chance alone moves the share by about 0.016.
    q, k = 1 / 4800, 2.0
    lives = spin(inputs_with(q, k), **ARGS)
    assert lives["found_partner"].mean() == pytest.approx(partner_chance(4800, q, k), abs=0.06)
    assert lives.loc[lives["found_partner"], "age_met"].between(15, 30).all()


def test_a_country_factor_of_two_doubles_the_people_met():
    # k = 1e9 makes every life meet almost exactly the average: 4800 people, or 9600 in country XX
    typical = spin(inputs_with(0.0, 1e9), **ARGS)
    talkative = spin(inputs_with(0.0, 1e9), country="XX", **ARGS)
    assert typical["people_met"].median() == pytest.approx(4800, rel=0.01)
    assert talkative["people_met"].median() == pytest.approx(9600, rel=0.01)


def test_rejects_age_pool_and_country_outside_the_inputs():
    with pytest.raises(ValueError, match="age"):
        spin(inputs_with(0.0, 3.0), **{**ARGS, "age": 17})
    with pytest.raises(ValueError, match="pool"):
        spin(inputs_with(0.0, 3.0), **{**ARGS, "pool": "mars"})
    with pytest.raises(ValueError, match="country"):
        spin(inputs_with(0.0, 3.0), country="ZZ", **ARGS)
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_roulette.py -v`
Expected: 4 failed, 1 passed. The failures are `KeyError: 'people_met'` and `TypeError: spin() got an unexpected keyword argument 'country'`.

- [ ] **Step 3: Change the module**

`src/soulmates/roulette.py`, full file:

```python
"""Soul Mate Roulette: play out simulated lives under soul mate rules and under real world rules."""
import json

import numpy as np
import pandas as pd

from soulmates.model import recog_prior

POOLS = ("world", "europe")


def load_inputs(path):
    with open(path) as f:
        raw = json.load(f)
    return {
        "curve": np.asarray(raw["curve"], dtype=float),
        "q": np.asarray(raw["q"], dtype=float),
        "k": np.asarray(raw["k"], dtype=float),
        "countries": {name: float(factor) for name, factor in raw["countries"].items()},
        "pools": {pool: {int(age): float(m) for age, m in raw["pools"][pool].items()} for pool in POOLS},
    }


def spin(inputs, age, pool, recog_bounds, n_lives, seed, country="typical", recog_mass=0.9):
    if not 18 <= age <= 60:
        raise ValueError(f"age must be between 18 and 60, got {age}")
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
    if country != "typical" and country not in inputs["countries"]:
        raise ValueError(f"country must be 'typical' or one of {sorted(inputs['countries'])}, got {country!r}")
    rng = np.random.default_rng(seed)
    curve = inputs["curve"]
    draw = rng.integers(len(inputs["q"]), size=n_lives)  # q and k come as pairs from the same posterior draw
    q, k = inputs["q"][draw], inputs["k"][draw]
    a, b = recog_prior(recog_bounds, recog_mass)
    p_recog = rng.beta(a, b, size=n_lives)
    # People differ: each life gets its own multiplier with mean 1, and the chosen country scales everyone
    rate = rng.gamma(k, 1 / k) * (1.0 if country == "typical" else inputs["countries"][country])
    n = np.floor(rate * curve[age])  # people of your own age this life has met and talked to by `age`

    # One uniform draw decides both whether a partner was found and at which meeting: inverting the
    # geometric distribution at u gives the first match, and u < 1 − (1 − q) ^ n keeps it within n meetings
    u = rng.random(n_lives)
    age_met = pd.array([pd.NA] * n_lives, dtype="Int64")
    with np.errstate(divide="ignore", invalid="ignore"):  # q = 1 gives log1p(−1) = −inf, which still yields the right answer
        found_partner = (n >= 1) & (u < -np.expm1(n * np.log1p(-q)))
        first = np.ceil(np.log1p(-u[found_partner]) / np.log1p(-q[found_partner]))
    first = np.maximum(first, 1)  # q = 1 gives 0 here, and the first meeting is number 1
    # the first age at which this life's own count, rate × N(a), reaches that meeting
    age_met[found_partner] = np.searchsorted(curve, first / rate[found_partner])

    m = inputs["pools"][pool][age]
    found_soul_mate = rng.random(n_lives) < -np.expm1(n * np.log1p(-p_recog / m))
    return pd.DataFrame({
        "q": q,
        "k": k,
        "p_recog": p_recog,
        "people_met": n.astype(int),
        "found_partner": found_partner,
        "age_met": age_met,
        "found_soul_mate": found_soul_mate,
    })
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/test_roulette.py -v -W error::RuntimeWarning`
Expected: 5 passed.

- [ ] **Step 5: Change the export script and run it**

`scripts/export_demo_inputs.py`, full file:

```python
"""Write data/demo_inputs.json for Soul Mate Roulette: the lifetime curve, the country factors, the q and k
draws and the pool sizes.

It fits the contact model and the partner model with the same seeds as the notebooks, so it needs the
POLYMOD, France and HCMST files in data/raw. The JSON holds model results and no survey rows, and it is
committed to git.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pymc as pm

from soulmates.france import load_diaries
from soulmates.hcmst import load_hcmst
from soulmates.match import build_match_model, lifetime_curve
from soulmates.model import build_model
from soulmates.wpp import POLYMOD_COUNTRIES, pool_size

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def main():
    with build_model(load_diaries(DATA / "raw")):
        contacts = pm.sample(random_seed=2, progressbar=False).posterior
    curve = lifetime_curve(contacts, start_age=15)
    factor = contacts["country_factor"].mean(("chain", "draw"))
    countries = {str(c): float(factor.sel(country=c)) for c in factor["country"].values}
    hcmst = load_hcmst(DATA / "raw" / "HCMST 2017 to 2022 small public version 2.2.dta")
    with build_match_model(hcmst, curve):
        partners = pm.sample(random_seed=2, progressbar=False, target_accept=0.95).posterior
    q, k = partners["q"].values.ravel(), partners["k"].values.ravel()
    wpp = pd.read_csv(DATA / "wpp_pool.csv")
    ages = range(18, 61)
    pools = {
        "world": {age: pool_size(wpp, age=age) for age in ages},
        "europe": {age: pool_size(wpp, age=age, locations=POLYMOD_COUNTRIES) for age in ages},
    }
    path = DATA / "demo_inputs.json"
    path.write_text(json.dumps({"curve": curve.tolist(), "q": q.tolist(), "k": k.tolist(), "countries": countries, "pools": pools}))
    print(f"wrote {path}: {len(q)} draws, q 1 in {1 / np.median(q):,.0f}, k {np.median(k):.2f}, N(30) = {curve[30]:,.0f}, "
          f"N(60) = {curve[60]:,.0f}, countries {({c: round(f, 2) for c, f in countries.items()})}")


if __name__ == "__main__":
    main()
```

Run: `uv run python scripts/export_demo_inputs.py`
Expected, after about 2 minutes, as the last line:

```
wrote /Users/a.jebari/Desktop/Study/soulmates/data/demo_inputs.json: 4000 draws, q 1 in 424, k 2.95, N(30) = 2,725, N(60) = 5,783, countries {'BE': 1.1, 'DE': 0.31, 'FI': 1.17, 'FR': 1.15, 'GB': 0.88, 'IT': 1.66, 'LU': 1.15, 'NL': 1.51, 'PL': 0.92}
```

The file is about 172 KB.

- [ ] **Step 6: Change the app**

`demo/roulette.py`, full file:

```python
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from soulmates.roulette import load_inputs, spin

    # written by scripts/export_demo_inputs.py, committed to git so the demo runs from a fresh clone
    inputs = load_inputs(mo.notebook_dir().parent / "data" / "demo_inputs.json")
    return inputs, mo, np, plt, spin


@app.cell
def _(inputs, mo, np):
    one_in = round(1 / np.median(inputs["q"]), -1)
    mo.md(f"""
    # Soul Mate Roulette

    Each spin plays out simulated lives. Under soul mate rules, one person in the pool is your match, and you
    have to meet them, talk to them and recognize each other. Under real world rules, about 1 in {one_in:,.0f}
    of the people your age you talk to could be a partner, as fitted on the How Couples Meet and Stay Together
    survey. Every life meets its own number of people, because some of us talk to many strangers and some to few.
    """)
    return


@app.cell
def _(inputs, mo):
    names = {"BE": "Belgium", "DE": "Germany", "FI": "Finland", "FR": "France", "GB": "Great Britain",
             "IT": "Italy", "LU": "Luxembourg", "NL": "the Netherlands", "PL": "Poland"}
    places = {"a typical country": "typical"} | {names.get(code, code): code for code in sorted(inputs["countries"])}
    age = mo.ui.slider(18, 60, value=30, label="Your age", show_value=True)
    country = mo.ui.dropdown(places, value="a typical country", label="You live in")
    pool = mo.ui.dropdown(
        {"the whole world": "world", "the 8 POLYMOD countries": "europe"},
        value="the whole world",
        label="Your soul mate lives in",
    )
    lower = mo.ui.slider(0.01, 0.99, step=0.01, value=0.3, label="Lower bound for p_recog", show_value=True)
    upper = mo.ui.slider(0.01, 0.99, step=0.01, value=0.9, label="Upper bound for p_recog", show_value=True)
    one = mo.ui.run_button(label="Spin 1 life")
    many = mo.ui.run_button(label="Spin 1,000 lives")
    mo.vstack([mo.hstack([age, country, pool]), mo.hstack([lower, upper]), mo.hstack([one, many])])
    return age, country, lower, many, one, pool, upper


@app.cell
def _(lower, mo, upper):
    mo.stop(lower.value >= upper.value, mo.callout("The lower bound has to be below the upper bound.", kind="warn"))
    bounds = (lower.value, upper.value)
    mo.md(f"The guess: I am 90% sure that p_recog lies between {lower.value:.2f} and {upper.value:.2f}.")
    return (bounds,)


@app.cell
def _(age, bounds, country, inputs, mo, one, pool, spin):
    mo.stop(not one.value, mo.md("Press *Spin 1 life* to play out one life."))
    life = spin(inputs, age.value, pool.value, bounds, n_lives=1, seed=None, country=country.value).iloc[0]
    real = f"met a possible partner at age {life.age_met}" if life.found_partner else "did not meet a possible partner"
    soul = "found your soul mate" if life.found_soul_mate else "did not find your soul mate"
    mo.md(f"""
    By age {age.value} this life has met and talked to {life.people_met:,} new people of its own age.

    - Real world rules: you {real}.
    - Soul mate rules: you {soul}.
    """)
    return


@app.cell
def _(age, bounds, country, inputs, many, mo, plt, pool, spin):
    mo.stop(not many.value, mo.md("Press *Spin 1,000 lives* to play out a thousand lives."))
    lives = spin(inputs, age.value, pool.value, bounds, n_lives=1000, seed=None, country=country.value)
    partners, soul_mates = int(lives["found_partner"].sum()), int(lives["found_soul_mate"].sum())
    possible = (lives["q"] * lives["people_met"]).mean()
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh(["soul mate rules", "real world rules"], [soul_mates, partners])
    ax.set(xlim=(0, 1000), xlabel="lives out of 1,000 that found someone")
    fig.tight_layout()
    mo.vstack([
        mo.md(
            f"Out of 1,000 lives, {partners:,} met a possible partner under real world rules "
            f"and {soul_mates:,} found their soul mate. The lives met between {int(lives['people_met'].min()):,} "
            f"and {int(lives['people_met'].max()):,} people of their own age, and on average about "
            f"{possible:.0f} of them could have been a partner."
        ),
        fig,
    ])
    return


if __name__ == "__main__":
    app.run()
```

Run: `uv run marimo check demo/roulette.py`
Expected: no output, exit code 0.

- [ ] **Step 7: Click through it**

Run in the background: `uv run marimo run demo/roulette.py --headless --port 2718 --no-token`
Open `http://localhost:2718` in the browser pane. The controls appear after about 25 seconds. Then check:

1. The sliders show 30, 0.3 and 0.9, and there are two dropdowns, "You live in" and "Your soul mate lives in".
2. "Spin 1 life" names how many people this life has met, and that number changes from spin to spin.
3. "Spin 1,000 lives" in a typical country shows about 970 lives with a possible partner and 0 or 1 soul mates. In a run of 100,000 lives in the scratch copy, 967 per 1,000 met a possible partner. The median life met 2,425 people, and the lives averaged 6.6 possible partners.
4. With "You live in" set to Italy the figure rises to about 990 per 1,000. With Germany it falls to about 780.

The dropdowns are native selects inside a shadow root, and in the browser pane they do not take keyboard input. Set them from the page with JavaScript if needed. Stop the server afterwards.

- [ ] **Step 8: Stop for review.** Do not commit.

---

### Task 6: Notebooks and housekeeping

**Files:**
- Rebuild: `notebooks/01_simulated.ipynb`, `notebooks/02_polymod.ipynb`, `notebooks/03_partners.ipynb`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: everything from Tasks 1 to 5.
- Produces: the three executed notebooks.

- [ ] **Step 1: Build the notebooks**

Save this script as a scratch file outside the repo, for example `build_notebooks.py`, and run it from the repo root with `uv run python build_notebooks.py`.

```python
"""Build the three notebooks of the soul mates project. Run from the repo root, then execute them with nbconvert."""
from textwrap import dedent

import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write(name, cells):
    nbf.write(nbf.v4.new_notebook(cells=cells), f"notebooks/{name}.ipynb")
    print(f"wrote notebooks/{name}.ipynb")


# ---------------------------------------------------------------- 01 simulated
write("01_simulated", [
    md("""
        # Soul mates with Bayes, on simulated data

        Everyone has one soul mate, drawn at random from the M people born within 5 years of them. You find them
        only if you meet and talk to them and you both recognize each other. This notebook estimates how many new
        people of your own age you meet per day, turns that into new people met between 15 and 60, and then into
        the chance of finding your soul mate. The data is simulated with known true values, so we can check the model.
    """),
    code("""
        import arviz as az
        import matplotlib.pyplot as plt
        import numpy as np
        import pymc as pm

        from soulmates.model import age_band, build_model, p_find
        from soulmates.simulate import simulate_diaries

        # Made up true values: mean new contacts per day for each 5 year band, peaking at ages 20 to 24
        TRUE_BAND_MEANS = [1.0, 1.5, 2.0, 3.0, 4.0, 3.5, 3.0, 2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0]
        TRUE_ALPHA = 0.2
        TRUE_WEEKEND = 1.4
        TRUE_COUNTRY_FACTORS = [0.5, 0.8, 1.25, 2.0]  # four made up countries, from quiet to talkative
        df = simulate_diaries(6000, TRUE_BAND_MEANS, TRUE_ALPHA, seed=1, weekend_effect=TRUE_WEEKEND,
                              country_factors=TRUE_COUNTRY_FACTORS)
        df.head()
    """),
    code("""
        observed = df.groupby(age_band(df["age"]))["new_contacts"].mean()
        fig, ax = plt.subplots()
        ax.bar(observed.index * 5, observed.values, width=4)
        ax.set(xlabel="start of the 5 year age band", ylabel="new contacts per day",
               title="Simulated diaries: new contacts per day by age band, all countries")
        df.groupby("country")["new_contacts"].mean().round(2)
    """),
    md("""
        ## Prior predictive check

        Before fitting, we draw fake diaries from the priors alone. They should cover the simulated diaries
        without putting much weight on absurd days.
    """),
    code("""
        model = build_model(df)
        with model:
            prior = pm.sample_prior_predictive(random_seed=3)
        prior_counts = prior.prior_predictive["new_contacts"].values.ravel()
        bins = np.arange(52)
        fig, ax = plt.subplots()
        ax.hist(np.minimum(prior_counts, 50), bins=bins, density=True, alpha=0.5, label="prior predictive")
        ax.hist(np.minimum(df["new_contacts"], 50), bins=bins, density=True, alpha=0.5, label="simulated diaries")
        ax.set(xlabel="new contacts on one day (50 stands for 50 or more)", ylabel="share of days",
               title="Prior predictive against the simulated diaries")
        ax.legend();
    """),
    md("""
        ## Fit and recovery

        The model has a level for each age band and a factor for each country. The posterior should sit close
        to the true values.
    """),
    code("""
        with model:
            idata = pm.sample(random_seed=2)
        az.summary(idata, var_names=["mu_pop", "sigma_band", "sigma_country", "alpha", "beta_weekend", "lifetime_typical"])
    """),
    code("""
        hdi = az.hdi(idata, var_names=["band_mean"], prob=0.94)["band_mean"].values
        mean = idata.posterior["band_mean"].mean(("chain", "draw")).values
        ages = np.arange(17) * 5
        fig, ax = plt.subplots()
        ax.errorbar(ages, mean, yerr=[mean - hdi[:, 0], hdi[:, 1] - mean], fmt="o", label="posterior mean and 94% HDI")
        ax.plot(ages, TRUE_BAND_MEANS, "kx", label="true band mean")
        ax.set(xlabel="start of the 5 year age band", ylabel="new contacts per weekday, typical country",
               title="Posterior band means against the true values")
        ax.legend();
    """),
    code("""
        factor = idata.posterior["country_factor"]
        for name, true in zip(factor["country"].values, TRUE_COUNTRY_FACTORS):
            draws = factor.sel(country=name).values.ravel()
            print(f"{name}: true factor {true}, posterior {draws.mean():.2f} "
                  f"({np.percentile(draws, 3):.2f} to {np.percentile(draws, 97):.2f})")
    """),
    md("""
        ## The chance of finding your soul mate

        P_find = 1 − (1 − p_recog / M) ^ lifetime. The pool M is made up here, because the data is simulated.
        `lifetime_anywhere` is the count for a person in a country we have no data for.
    """),
    code("""
        POOL_SIZE = 1e8  # made up, the real pool comes from the UN data in notebook 02
        pf = p_find(idata, POOL_SIZE).values.ravel()
        fig, ax = plt.subplots()
        ax.hist(np.log10(pf), bins=60)
        ax.set(xlabel="log10 of P(find your soul mate)", ylabel="posterior draws",
               title=f"P_find with a made up pool of {POOL_SIZE:.0e} people")
        print(f"P_find mean {pf.mean():.2e}, 94% interval {np.percentile(pf, 3):.2e} to {np.percentile(pf, 97):.2e}")
    """),
])

# ---------------------------------------------------------------- 02 contacts and odds
write("02_polymod", [
    md("""
        # Soul mates with Bayes, on real contact diaries

        Two surveys asked people to write down everyone they met on a day. POLYMOD covers 8 European countries
        in 2005 and 2006 (Mossong et al. 2008, Zenodo record 3874557, licence CC BY-NC 4.0). The French survey
        of 2012 adds a ninth country (Béraud et al. 2015, Zenodo record 3886590, licence CC BY-NC-ND 4.0).
        A contact counts when two people touch skin to skin, or talk face to face for at least three words. For
        each contact, people said how often they usually meet that person, and a contact marked "first time" is
        a new person.

        Your soul mate is drawn from the people within 5 years of your age. So this notebook counts only new
        people within 5 years of the participant's age. Strangers you only glance at are not counted.

        Run these once before this notebook: `uv run python scripts/download_polymod.py`,
        `uv run python scripts/download_france.py` and `uv run python scripts/download_wpp.py`.
    """),
    code("""
        import arviz as az
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import pymc as pm

        from soulmates.france import load_diaries
        from soulmates.model import age_band, build_model, p_find, recog_prior
        from soulmates.wpp import POLYMOD_COUNTRIES, pool_size

        df = load_diaries("../data/raw")
        adults = df[df["age"].between(18, 60)]
        adults.groupby("country")["new_contacts"].agg(diaries="size", new_people_per_day="mean").round(2)
    """),
    code("""
        raw = df.groupby([age_band(df["age"]), "weekend"])["new_contacts"].mean().unstack()
        fig, ax = plt.subplots()
        ax.plot(raw.index * 5, raw[False], "o-", label="weekday")
        ax.plot(raw.index * 5, raw[True], "s-", label="weekend")
        ax.set(xlabel="start of the 5 year age band", ylabel="new people of your age per day",
               title="Raw mean of new contacts by age band, 9 countries")
        ax.legend();
    """),
    md("""
        ## Fit

        The model has a level for each age band and a factor for each country. The factors come from one shared
        distribution, so the model also says how far a country we have no data for could sit from the average.
        A weekend day has exp(beta_weekend) times the new contacts of a weekday.
    """),
    code("""
        # The speaker's guess: "I am 90% sure that p_recog lies between 0.3 and 0.9"
        RECOG_BOUNDS = (0.3, 0.9)
        RECOG_MASS = 0.9

        model = build_model(df, recog_bounds=RECOG_BOUNDS, recog_mass=RECOG_MASS)
        with model:
            idata = pm.sample(random_seed=2)
        print("divergences", int(idata.sample_stats["diverging"].sum()))
        az.summary(idata, var_names=["mu_pop", "sigma_band", "sigma_country", "alpha", "beta_weekend", "p_recog"])
    """),
    code("""
        hdi = az.hdi(idata, var_names=["band_mean"], prob=0.94)["band_mean"].values
        mean = idata.posterior["band_mean"].mean(("chain", "draw")).values
        ages = np.arange(17) * 5
        fig, ax = plt.subplots()
        ax.errorbar(ages, mean, yerr=[mean - hdi[:, 0], hdi[:, 1] - mean], fmt="o", label="posterior mean and 94% HDI")
        ax.set(xlabel="start of the 5 year age band", ylabel="new people of your age per weekday",
               title="Posterior band means in a typical country")
        ax.legend();
    """),
    md("""
        ## The guess behind p_recog

        p_recog is the chance that you both realize it at some point after you have met and talked. No data
        informs it. The guess "I am 90% sure it lies between 0.3 and 0.9" becomes a Beta prior through PreliZ
        `maxent`, which picks the flattest Beta curve that puts 90% between the two bounds. The posterior of
        p_recog equals its prior, so the draws below show the prior.
    """),
    code("""
        p_recog = idata.posterior["p_recog"].values.ravel()
        inside = ((p_recog > RECOG_BOUNDS[0]) & (p_recog < RECOG_BOUNDS[1])).mean()
        fig, ax = plt.subplots()
        ax.hist(p_recog, bins=60, density=True)
        for bound in RECOG_BOUNDS:
            ax.axvline(bound, color="k", linestyle="--")
        ax.set(xlabel="p_recog", ylabel="density", title=f"Prior for p_recog, {inside:.0%} of draws between the bounds")
        a, b = recog_prior(RECOG_BOUNDS, RECOG_MASS)
        print(f"Beta({a:.2f}, {b:.2f}), mean {p_recog.mean():.3f}, share between the bounds {inside:.3f}")
    """),
    md("""
        ## The odds, country by country

        P_find = 1 − (1 − p_recog / M) ^ lifetime. M is everyone alive in 2023 aged 25 to 35, from the UN World
        Population Prospects 2024 (licence CC BY 3.0 IGO). Your soul mate can live anywhere, so every country
        uses the world pool. The lifetime count is the number of new people of your age you talk to between
        15 and 60.
    """),
    code("""
        wpp = pd.read_csv("../data/wpp_pool.csv")  # written by scripts/download_wpp.py
        M = pool_size(wpp)
        NAMES = {"BE": "Belgium", "DE": "Germany", "FI": "Finland", "FR": "France", "GB": "Great Britain",
                 "IT": "Italy", "LU": "Luxembourg", "NL": "Netherlands", "PL": "Poland"}


        def row(label, life, pf):
            return {"where you live": label, "new people, 15 to 60": round(life.mean()),
                    "from": round(np.percentile(life, 3)), "to": round(np.percentile(life, 97)),
                    "odds: 1 in": round(1 / pf.mean())}


        life_c, pf_c = idata.posterior["lifetime_country"], p_find(idata, M, lifetime="lifetime_country")
        rows = [row(NAMES[c], life_c.sel(country=c).values.ravel(), pf_c.sel(country=c).values.ravel())
                for c in life_c["country"].values]
        table = pd.DataFrame(rows).sort_values("odds: 1 in").reset_index(drop=True)
        table
    """),
    code("""
        fig, ax = plt.subplots(figsize=(7, 4))
        y = np.arange(len(table))[::-1]
        ax.errorbar(table["new people, 15 to 60"], y, xerr=[table["new people, 15 to 60"] - table["from"],
                    table["to"] - table["new people, 15 to 60"]], fmt="o")
        ax.set_yticks(y, table["where you live"])
        ax.set(xlabel="new people of your age you talk to between 15 and 60", title="Countries differ in how much people talk to strangers")
        fig.tight_layout()
    """),
    md("""
        ## The headline

        For a person in a country we have no data for, the model draws a new country from the distribution the
        9 countries came from. That count carries the spread between countries.
    """),
    code("""
        pools = {"world": M, "8 POLYMOD countries": pool_size(wpp, locations=POLYMOD_COUNTRIES)}
        for label in ("lifetime_anywhere", "lifetime_typical"):
            life = idata.posterior[label].values.ravel()
            print(f"{label}: median {np.median(life):,.0f}, 94% interval {np.percentile(life, 3):,.0f} to {np.percentile(life, 97):,.0f}")
        fig, ax = plt.subplots()
        for label, m in pools.items():
            pf = p_find(idata, m).values.ravel()
            print(f"{label}: M = {m:,.0f}, P_find mean {pf.mean():.2e} (1 in {1 / pf.mean():,.0f}), "
                  f"94% interval 1 in {1 / np.percentile(pf, 3):,.0f} to 1 in {1 / np.percentile(pf, 97):,.0f}")
            ax.hist(np.log10(pf), bins=60, alpha=0.5, label=label)
        ax.set(xlabel="log10 of P(find your soul mate)", ylabel="posterior draws",
               title="P_find for a person somewhere in the world, age 30 in 2023")
        ax.legend()
        headline = p_find(idata, M).values.mean()
        print(f"people of your generation who find their one soul mate: about {M * headline:,.0f} of {M:,.0f}")
    """),
    md("""
        ## The best case

        This is a best case and is labelled as one. It takes the most talkative country, first with the guess
        for p_recog and then with certain recognition.
    """),
    code("""
        best = life_c.mean(("chain", "draw")).idxmax().item()
        life_best = life_c.sel(country=best).values.ravel()
        print(f"most talkative country: {NAMES[best]}, {life_best.mean():,.0f} new people of your age between 15 and 60")
        print(f"with the guess for p_recog: 1 in {1 / pf_c.sel(country=best).values.mean():,.0f}")
        print(f"if recognition were certain: 1 in {M / life_best.mean():,.0f}")
    """),
    md("""
        ## Known limits

        1. The model cannot tell culture from survey method. Four countries, Germany among them, asked for an
           estimate of professional contacts instead of a list. Germany's low level may come from that.
        2. The diaries are from 2005 to 2012. They cannot see people who meet online.
        3. The French participants kept a diary on two or more days. The model treats those days as separate
           diaries, which makes the French estimate a little too sure of itself.
        4. The pool counts everyone and ignores gender and orientation.
        5. M is a fixed number from the WPP estimates, without uncertainty.
        6. Strangers you only glance at are not counted. The premise is people you meet and talk to.
        7. p_recog is a guess. It drives most of the spread of the odds within one country.
    """),
])

# ---------------------------------------------------------------- 03 partners
write("03_partners", [
    md("""
        # How many of them could be the one?

        The soul mate premise gives each person one possible partner in a pool of about 1.3 billion. The How
        Couples Meet and Stay Together survey (HCMST) shows how many people actually have had a partner by each
        age. This notebook asks what share of the people you talk to would have to be possible partners for the
        two to match.

        To get the data, read and accept the data use agreement on https://data.stanford.edu/hcmst2017, then
        save `HCMST 2017 to 2022 small public version 2.2.dta` to `data/raw/`. The agreement asks you to tell
        the HCMST contact person about your work and to cite the data.

        Data citation: Rosenfeld, Michael J., Reuben J. Thomas, and Sonia Hausen. 2023. How Couples Meet and Stay
        Together 2017-2020-2022 combined dataset. [Computer files]. Stanford, CA: Stanford University Libraries.
        https://data.stanford.edu/hcmst2017
    """),
    code("""
        import arviz as az
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import pymc as pm

        from soulmates.france import load_diaries, load_france_days
        from soulmates.hcmst import load_hcmst
        from soulmates.match import build_flat_model, build_match_model, lifetime_curve, partner_chance
        from soulmates.model import build_model

        hcmst = load_hcmst("../data/raw/HCMST 2017 to 2022 small public version 2.2.dta")
        bands = pd.cut(hcmst["age"], [17, 24, 29, 34, 39, 49, 60])
        observed = hcmst.groupby(bands, observed=True).apply(lambda g: pd.Series({
            "people": len(g),
            "never had a partner": int((~g["ever_partnered"]).sum()),
            "ever, share": g["ever_partnered"].mean(),
            "ever, weighted share": np.average(g["ever_partnered"], weights=g["weight"]),
        })).astype({"people": int, "never had a partner": int})
        observed.round(3)
    """),
    md("""
        ## People differ in how many others they meet

        In the French survey every participant kept a diary on two days. If the number of new people only
        depended on the day, the two counts would be unrelated. The cell looks at new people of any age first,
        and then at new people within 5 years of the participant's age, which are fewer and noisier.
    """),
    code("""
        for label, window in [("new people of any age", 200), ("new people of your own age", 5)]:
            france = load_france_days("../data/raw/france2012", window=window)
            adults = france[france["age"].between(18, 60)]
            two_days = adults.pivot_table(index=["part_id", "wave"], columns="day", values="new_contacts").dropna()
            none_one_day = ((two_days[1] == 0).mean() + (two_days[2] == 0).mean()) / 2
            none_both = ((two_days[1] == 0) & (two_days[2] == 0)).mean()
            between = np.cov(two_days[1], two_days[2])[0, 1]  # how much of the spread stays with the person
            mean = two_days.to_numpy().mean()
            print(f"{label}: {len(two_days)} adults with two diary days, {mean:.2f} per day")
            print(f"  correlation between the two days: {two_days[1].corr(two_days[2]):.2f}")
            print(f"  nobody new on a day: {none_one_day:.1%}. On both days: {none_both:.1%}. If the days were unrelated: {none_one_day ** 2:.1%}")
            print(f"  spread of personal rates implied by the two days: k = {mean ** 2 / between:.2f}")
    """),
    md("""
        A small k means that people differ a lot. The two days lie in the same survey wave, so they also share
        what was going on in that person's life in those weeks. These values of k therefore overstate lasting
        differences. They are a lower bound for k, and the partner model below estimates k from HCMST.

        ## New people met by each age

        N(a) is the number of new people of your own age you have met and talked to from age 15 up to age a, in
        a typical country. It comes from the contact model of notebook 02.
    """),
    code("""
        with build_model(load_diaries("../data/raw")):
            contacts = pm.sample(random_seed=2, progressbar=False)
        curve = lifetime_curve(contacts.posterior, start_age=15)
        print({age: round(curve[age]) for age in (18, 25, 40, 60)})
    """),
    md("""
        ## The partner model

        Each person you talk to is a possible partner with chance q. Each person meets their own number of
        people: the average N(a) times a personal factor with mean 1. The number k says how much people differ.
        A small k means large differences. The chance of having had a partner by age a is then
        1 − (1 + θ × N(a) / k) ^ (−k), with θ = −log(1 − q).

        The simpler model for comparison gives everyone exactly N(a) people.
    """),
    code("""
        with build_match_model(hcmst, curve):
            match = pm.sample(random_seed=2, target_accept=0.95, idata_kwargs={"log_likelihood": True})
        with build_flat_model(hcmst, curve):
            flat = pm.sample(random_seed=2, idata_kwargs={"log_likelihood": True})
        q, k = match.posterior["q"].values.ravel(), match.posterior["k"].values.ravel()
        q_flat = flat.posterior["q"].values.ravel()
        print(f"q: 1 in {1 / np.median(q):,.0f}, 94% interval 1 in {1 / np.percentile(q, 97):,.0f} to 1 in {1 / np.percentile(q, 3):,.0f}")
        print(f"k: {np.median(k):.2f}, 94% interval {np.percentile(k, 3):.2f} to {np.percentile(k, 97):.2f}")
        print(f"one q for everyone: 1 in {1 / np.median(q_flat):,.0f}")
        print(f"divergences: {int(match.sample_stats['diverging'].sum())} and {int(flat.sample_stats['diverging'].sum())}")
    """),
    code("""
        n = curve[hcmst["age"].to_numpy()]
        by_band = lambda chance: pd.Series(chance.mean(axis=0)).groupby(bands.to_numpy(), observed=True).mean()
        predicted = by_band(partner_chance(n[None, :], q[:, None], k[:, None]))
        predicted_flat = by_band(1 - (1 - q_flat[:, None]) ** n[None, :])
        labels = [str(b) for b in observed.index]
        fig, ax = plt.subplots()
        ax.plot(labels, observed["ever, share"], "o", label="HCMST, share")
        ax.plot(labels, observed["ever, weighted share"], "s", label="HCMST, weighted share")
        ax.plot(labels, predicted.values, "k-", label="q and k: people differ")
        ax.plot(labels, predicted_flat.values, "k:", label="one q: everyone meets the same number")
        ax.set(xlabel="age band", ylabel="share who ever had a partner", title="Two partner models against HCMST")
        ax.legend()
        pd.DataFrame({"observed": observed["ever, share"].values, "q and k": predicted.values,
                      "one q": predicted_flat.values}, index=labels).round(3)
    """),
    md("""
        ## Which model predicts better?

        LOO (leave one out cross validation) scores how well each model predicts each person it did not see.
        elpd is that score summed over people, and higher is better. `dse` is the standard error of the
        difference.
    """),
    code("""
        az.compare({"q and k": match, "one q": flat})
    """),
    md("""
        ## How many of them could be the one?

        The average person has talked to N(60) people of their own age by 60. A share q of them could have
        become a partner.
    """),
    code("""
        possible = q * curve[60]
        alone = 1 - partner_chance(curve[60], q, k)
        print(f"1 in {1 / np.median(q):,.0f} of the people your age you talk to could be a partner")
        print(f"possible partners met by 60: {np.median(possible):.0f}, 94% interval {np.percentile(possible, 3):.0f} to {np.percentile(possible, 97):.0f}")
        print(f"share of people the model leaves without a partner by 60: {alone.mean():.1%}")
    """),
    md("""
        ## Known limits

        1. The contact rates come from Europe between 2005 and 2012 and the partner data from the US in 2017.
        2. The survey weights are not in the likelihood. The weighted shares are shown as a check.
        3. N(a) is the curve of a typical country and is treated as known.
        4. q is the same at every age.
        5. "Ever had a partner" counts any past partner, so a short relationship counts the same as a life
           long one.
        6. The model only counts people within 5 years of your age. In HCMST, 73% of couples are that close
           in age.
        7. The diaries cannot see people who meet online. In HCMST, 37% of the couples who met in 2010 or
           later met online.
    """),
])
```

Expected: three lines starting with `wrote notebooks/`.

- [ ] **Step 2: Execute them**

Run each of these:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_simulated.ipynb
```

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_polymod.ipynb
```

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_partners.ipynb
```

Expected: each exits without error, in about 45, 70 and 90 seconds. In the scratch trial they printed, among other lines:

```
01: C0: true factor 0.5, posterior 0.50 (0.45 to 0.55)
01: C3: true factor 2.0, posterior 1.89 (1.70 to 2.09)
02: divergences 0
02: lifetime_anywhere: median 5,817, 94% interval 2,003 to 16,976
02: lifetime_typical: median 5,778, 94% interval 5,247 to 6,362
02: world: M = 1,328,956,521, P_find mean 3.07e-06 (1 in 325,806), 94% interval 1 in 1,526,676 to 1 in 114,995
02: 8 POLYMOD countries: M = 39,411,641, P_find mean 1.03e-04 (1 in 9,663), 94% interval 1 in 45,276 to 1 in 3,411
02: people of your generation who find their one soul mate: about 4,079 of 1,328,956,521
02: most talkative country: Italy, 9,617 new people of your age between 15 and 60
02: with the guess for p_recog: 1 in 231,870
02: if recognition were certain: 1 in 138,194
03: new people of any age: 661 adults with two diary days, 1.63 per day
03:   correlation between the two days: 0.40
03: new people of your own age: 661 adults with two diary days, 0.41 per day
03:   correlation between the two days: 0.06
03: q: 1 in 424, 94% interval 1 in 302 to 1 in 554
03: k: 2.95, 94% interval 1.97 to 4.84
03: one q for everyone: 1 in 794
03: divergences: 0 and 0
03: possible partners met by 60: 14, 94% interval 10 to 19
03: share of people the model leaves without a partner by 60: 0.6%
```

The country table of notebook 02 runs from Italy (9,617 people, 1 in 231,870) to Germany (1,777 people, 1 in 1,255,282). The comparison table of notebook 03 ranks "q and k" first, 10.0 points of elpd ahead of "one q", with a `dse` of 6.2.

- [ ] **Step 3: Ignore .DS_Store**

Add this line to `.gitignore`:

```
.DS_Store
```

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -v -W error::RuntimeWarning`
Expected: 19 passed.

- [ ] **Step 5: Stop for review.** Do not commit.
