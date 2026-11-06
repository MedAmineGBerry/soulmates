# Soul mates milestone 6 implementation plan: Soul Mate Roulette

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the live demo. A marimo app plays out simulated lives under soul mate rules and under real world rules.

**Architecture:** `src/soulmates/roulette.py` holds `load_inputs` and `spin`, the only logic. `scripts/export_demo_inputs.py` writes `data/demo_inputs.json` from the model fits, and that file is committed. `demo/roulette.py` is a marimo app that wires sliders and buttons to `spin`.

**Tech Stack:** as milestones 1 to 5, plus marimo 0.25.2.

**Spec:** `docs/superpowers/specs/2026-11-06-soulmates-milestone-6-design.md`

## Global constraints

- Everything from the earlier plans still holds. That covers the version floors, the ArviZ 1.x API, `uv run` for every command, the writing rules and marking made up values as made up.
- Do not commit. The user commits when they ask.
- `python3` on this machine is the system Python 3.9. Run the writing checker with `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`, and everything else with `uv run`.
- Every code block below already ran in a scratch copy of the repo on 2026-11-06. The app was also clicked through in the browser pane.

## File structure

```
src/soulmates/roulette.py         # new: load_inputs, spin
tests/test_roulette.py            # new: q = 0, q = 1, the share for a known q, bad inputs
scripts/export_demo_inputs.py     # new: writes data/demo_inputs.json
.gitignore                        # changed: data/* plus an exception for data/demo_inputs.json
data/demo_inputs.json             # new, committed: curve, q draws, pools
pyproject.toml, uv.lock           # changed: marimo added
demo/roulette.py                  # new: the marimo app
```

---

### Task 1: Spin function

**Files:**
- Create: `src/soulmates/roulette.py`
- Test: `tests/test_roulette.py`

**Interfaces:**
- Consumes: `recog_prior` from `soulmates.model`.
- Produces:
  - `load_inputs(path) -> dict` with `curve` (numpy array of 101 floats), `q` (numpy array) and `pools` (a dict from `"world"` and `"europe"` to a dict from age to pool size).
  - `spin(inputs, age, pool, recog_bounds, n_lives, seed, recog_mass=0.9) -> pandas.DataFrame` with the columns `q`, `p_recog`, `found_partner` (bool), `age_met` (Int64, missing when no partner was found) and `found_soul_mate` (bool). It raises `ValueError` for an age outside 18 to 60 or a pool other than `"world"` and `"europe"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_roulette.py`:

```python
import numpy as np
import pytest

from soulmates.roulette import spin

# Made up inputs: 300 new people per year from age 15, a world pool of 1e9 and a European pool of 4e7 at every age
CURVE = np.maximum(np.arange(101) - 14, 0) * 300.0
POOLS = {"world": {age: 1e9 for age in range(18, 61)}, "europe": {age: 4e7 for age in range(18, 61)}}


def inputs_with_q(q):
    return {"curve": CURVE, "q": np.full(4000, q), "pools": POOLS}


def test_nobody_finds_anyone_with_q_zero():
    lives = spin(inputs_with_q(0.0), age=30, pool="world", recog_bounds=(0.1, 0.6), n_lives=1000, seed=1)
    assert not lives["found_partner"].any()
    assert lives["age_met"].isna().all()


def test_everyone_finds_someone_at_15_with_q_one():
    # the first meeting already matches, and the first new person is met at age 15
    lives = spin(inputs_with_q(1.0), age=30, pool="world", recog_bounds=(0.1, 0.6), n_lives=1000, seed=1)
    assert lives["found_partner"].all()
    assert (lives["age_met"] == 15).all()


def test_share_found_matches_the_formula():
    # At age 30 a life has met 16 × 300 = 4800 people. This q makes 1 − (1 − q) ^ 4800 exactly 0.5.
    # With 1000 lives, chance alone moves the share by about 0.016.
    q = 1 - 0.5 ** (1 / 4800)
    lives = spin(inputs_with_q(q), age=30, pool="world", recog_bounds=(0.1, 0.6), n_lives=1000, seed=1)
    assert lives["found_partner"].mean() == pytest.approx(0.5, abs=0.06)
    assert lives.loc[lives["found_partner"], "age_met"].between(15, 30).all()


def test_rejects_age_and_pool_outside_the_inputs():
    with pytest.raises(ValueError, match="age"):
        spin(inputs_with_q(0.0), age=17, pool="world", recog_bounds=(0.1, 0.6), n_lives=1, seed=1)
    with pytest.raises(ValueError, match="pool"):
        spin(inputs_with_q(0.0), age=30, pool="mars", recog_bounds=(0.1, 0.6), n_lives=1, seed=1)
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_roulette.py -v`
Expected: collection error with `ModuleNotFoundError: No module named 'soulmates.roulette'`

- [ ] **Step 3: Write the module**

`src/soulmates/roulette.py`:

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
        "pools": {pool: {int(age): float(m) for age, m in raw["pools"][pool].items()} for pool in POOLS},
    }


def spin(inputs, age, pool, recog_bounds, n_lives, seed, recog_mass=0.9):
    if not 18 <= age <= 60:
        raise ValueError(f"age must be between 18 and 60, got {age}")
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
    rng = np.random.default_rng(seed)
    curve = inputs["curve"]
    n = curve[age]  # every life meets exactly N(age) new people, as the HCMST fit assumed
    q = rng.choice(inputs["q"], size=n_lives)
    a, b = recog_prior(recog_bounds, recog_mass)
    p_recog = rng.beta(a, b, size=n_lives)

    # One uniform draw decides both whether a partner was found and at which meeting: inverting the
    # geometric distribution at u gives the first match, and u < 1 − (1 − q) ^ N keeps it within N meetings
    u = rng.random(n_lives)
    age_met = pd.array([pd.NA] * n_lives, dtype="Int64")
    with np.errstate(divide="ignore"):  # q = 1 gives log1p(−1) = −inf, which still yields the right answer
        found_partner = u < -np.expm1(n * np.log1p(-q))
        first = np.ceil(np.log1p(-u[found_partner]) / np.log1p(-q[found_partner]))
    first = np.maximum(first, 1)  # q = 1 gives 0 here, and the first meeting is number 1
    age_met[found_partner] = np.searchsorted(curve, first)  # first age a with N(a) >= that meeting

    m = inputs["pools"][pool][age]
    found_soul_mate = rng.random(n_lives) < -np.expm1(n * np.log1p(-p_recog / m))
    return pd.DataFrame({
        "q": q,
        "p_recog": p_recog,
        "found_partner": found_partner,
        "age_met": age_met,
        "found_soul_mate": found_soul_mate,
    })
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/test_roulette.py -v -W error::RuntimeWarning`
Expected: 4 passed. The flag turns warnings into errors, so this also checks that q = 1 does not warn.

- [ ] **Step 5: Stop for review.** Do not commit.

---

### Task 2: Demo inputs

**Files:**
- Create: `scripts/export_demo_inputs.py`
- Modify: `.gitignore`
- Create: `data/demo_inputs.json` (written by the script)

**Interfaces:**
- Consumes: `load_polymod`, `build_model`, `load_hcmst`, `lifetime_curve`, `build_match_model`, `pool_size` and `POLYMOD_COUNTRIES`, all unchanged.
- Produces: `data/demo_inputs.json` with the keys `curve`, `q` and `pools`, readable by `load_inputs` from Task 1.

- [ ] **Step 1: Write the script**

`scripts/export_demo_inputs.py`:

```python
"""Write data/demo_inputs.json for Soul Mate Roulette: the lifetime curve, the q draws and the pool sizes.

It fits the POLYMOD model and the free HCMST model with the same seeds as notebook 03, so it needs the
HCMST file in data/raw. The JSON holds model results and no survey rows, and it is committed to git.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pymc as pm

from soulmates.hcmst import load_hcmst
from soulmates.match import build_match_model, lifetime_curve
from soulmates.model import build_model
from soulmates.polymod import load_polymod
from soulmates.wpp import POLYMOD_COUNTRIES, pool_size

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def main():
    with build_model(load_polymod(DATA / "raw")):
        contacts = pm.sample(random_seed=2, progressbar=False)
    curve = lifetime_curve(contacts.posterior, start_age=15)
    hcmst = load_hcmst(DATA / "raw" / "HCMST 2017 to 2022 small public version 2.2.dta")
    with build_match_model(hcmst, curve):
        q = pm.sample(random_seed=2, progressbar=False).posterior["q"].values.ravel()
    wpp = pd.read_csv(DATA / "wpp_pool.csv")
    ages = range(18, 61)
    pools = {
        "world": {age: pool_size(wpp, age=age) for age in ages},
        "europe": {age: pool_size(wpp, age=age, locations=POLYMOD_COUNTRIES) for age in ages},
    }
    path = DATA / "demo_inputs.json"
    path.write_text(json.dumps({"curve": curve.tolist(), "q": q.tolist(), "pools": pools}))
    print(f"wrote {path}: {len(q)} q draws, q 1 in {1 / np.median(q):,.0f}, N(30) = {curve[30]:,.0f}, "
          f"world pool at 30 = {pools['world'][30]:,.0f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Let git see the one data file**

In `.gitignore`, replace the line `data/` with these two lines:

```
data/*
!data/demo_inputs.json
```

`data/` on its own would hide the whole folder, and git cannot bring back a file inside a hidden folder. `data/*` hides only what is inside, so the exception works. This was tested in a throwaway repo.

- [ ] **Step 3: Run the script**

Run: `uv run python scripts/export_demo_inputs.py`
Expected, after 1 to 4 minutes, as the last line:

```
wrote /Users/a.jebari/Desktop/Study/soulmates/data/demo_inputs.json: 4000 q draws, q 1 in 2,419, N(30) = 7,965, world pool at 30 = 1,328,956,521
```

The file is about 97 KB.

- [ ] **Step 4: Check what git sees**

Run: `git status --short --untracked-files=all data .gitignore`
Expected: `.gitignore` as modified and `data/demo_inputs.json` as untracked, and nothing else from `data/`.

- [ ] **Step 5: Stop for review.** Do not commit.

---

### Task 3: The marimo app

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (through `uv add`)
- Create: `demo/roulette.py`

**Interfaces:**
- Consumes: `load_inputs` and `spin` from Task 1, and `data/demo_inputs.json` from Task 2.
- Produces: an app that starts with `uv run marimo run demo/roulette.py`.

- [ ] **Step 1: Add marimo**

Run: `uv add "marimo>=0.25.2"`
Expected: `pyproject.toml` lists `marimo>=0.25.2`. Then `uv run python -c "import marimo, pymc, pytensor, arviz, preliz; print(marimo.__version__, pymc.__version__, pytensor.__version__, arviz.__version__, preliz.__version__)"` prints `0.25.2 6.4.0 3.3.7 1.3.0 0.28.0` or newer patch versions.

- [ ] **Step 2: Write the app**

`demo/roulette.py`:

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
def _(mo):
    mo.md("""
    # Soul Mate Roulette

    Each spin plays out simulated lives. Under soul mate rules, one person in the pool is your match, and you
    have to meet them, talk to them and recognize them. Under real world rules, about 1 in 2,400 of the new
    people you talk to could be a partner, as fitted on the How Couples Meet and Stay Together survey.
    """)
    return


@app.cell
def _(mo):
    age = mo.ui.slider(18, 60, value=30, label="Your age", show_value=True)
    pool = mo.ui.dropdown(
        {"the whole world": "world", "the 8 POLYMOD countries": "europe"},
        value="the whole world",
        label="Your soul mate lives in",
    )
    lower = mo.ui.slider(0.01, 0.99, step=0.01, value=0.1, label="Lower bound for p_recog", show_value=True)
    upper = mo.ui.slider(0.01, 0.99, step=0.01, value=0.6, label="Upper bound for p_recog", show_value=True)
    one = mo.ui.run_button(label="Spin 1 life")
    many = mo.ui.run_button(label="Spin 1,000 lives")
    mo.vstack([mo.hstack([age, pool]), mo.hstack([lower, upper]), mo.hstack([one, many])])
    return age, lower, many, one, pool, upper


@app.cell
def _(lower, mo, upper):
    mo.stop(lower.value >= upper.value, mo.callout("The lower bound has to be below the upper bound.", kind="warn"))
    bounds = (lower.value, upper.value)
    mo.md(f"The guess: I am 90% sure that p_recog lies between {lower.value:.2f} and {upper.value:.2f}.")
    return (bounds,)


@app.cell
def _(age, bounds, inputs, mo, one, pool, spin):
    mo.stop(not one.value, mo.md("Press *Spin 1 life* to play out one life."))
    life = spin(inputs, age.value, pool.value, bounds, n_lives=1, seed=None).iloc[0]
    met = round(inputs["curve"][age.value])
    real = f"met a possible partner at age {life.age_met}" if life.found_partner else "did not meet anyone"
    soul = "found your soul mate" if life.found_soul_mate else "did not find your soul mate"
    mo.md(f"""
    By age {age.value} you have met and talked to about {met:,} new people.

    - Real world rules: you {real}.
    - Soul mate rules: you {soul}.
    """)
    return


@app.cell
def _(age, bounds, inputs, many, mo, np, plt, pool, spin):
    mo.stop(not many.value, mo.md("Press *Spin 1,000 lives* to play out a thousand lives."))
    lives = spin(inputs, age.value, pool.value, bounds, n_lives=1000, seed=None)
    partners, soul_mates = int(lives["found_partner"].sum()), int(lives["found_soul_mate"].sum())
    pool_size = inputs["pools"][pool.value][age.value]
    k = np.median(lives["q"] * pool_size / lives["p_recog"])
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh(["soul mate rules", "real world rules"], [soul_mates, partners])
    ax.set(xlim=(0, 1000), xlabel="lives out of 1,000 that found someone")
    fig.tight_layout()
    mo.vstack([
        mo.md(
            f"Out of 1,000 lives, {partners:,} met a possible partner under real world rules "
            f"and {soul_mates:,} found their soul mate. The real world rules mean about {k:,.0f} "
            f"possible partners in a pool of {pool_size:,.0f} people."
        ),
        fig,
    ])
    return


if __name__ == "__main__":
    app.run()
```

- [ ] **Step 3: Lint it**

Run: `uv run marimo check demo/roulette.py`
Expected: no output, exit code 0.

- [ ] **Step 4: Click through it**

Run in the background: `uv run marimo run demo/roulette.py --headless --port 2718 --no-token`
Open `http://localhost:2718` in the browser pane. The first cell imports PyMC and PreliZ, so the controls appear after about 25 seconds. Then check:

1. The sliders show their values: 30, 0.1 and 0.6.
2. "Spin 1 life" shows about 7,965 new people by age 30. It then shows either an age at which a possible partner was met, or "did not meet anyone".
3. "Spin 1,000 lives" with the whole world shows about 970 lives with a partner, 0 or 1 soul mates, and about 1.6 million possible partners. In the scratch trial it showed 972 and 974 partners, 0 soul mates, and 1,592,510 and 1,638,266 possible partners.
4. With the 8 POLYMOD countries it shows about 965 partners and about 47,000 possible partners. In the scratch trial it showed 965 partners, 1 soul mate and 46,924 possible partners.

The pool dropdown is a native select inside a shadow root, and in the browser pane it does not take keyboard input. Set it from the page with JavaScript if needed. Stop the server afterwards.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: 15 passed.

- [ ] **Step 6: Stop for review.** Do not commit.
