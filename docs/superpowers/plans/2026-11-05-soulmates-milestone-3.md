# Soul mates milestone 3 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder pool of 1e8 people with the real pool size M from the UN World Population Prospects 2024.

**Architecture:** `scripts/download_wpp.py` downloads the WPP single age file, checks it and writes the small `data/wpp_pool.csv`. `src/soulmates/wpp.py` holds `pool_size`, which turns those rows into M. Notebook 02 uses M for the world and for the 8 POLYMOD countries.

**Tech Stack:** as milestones 1 and 2. The download uses `urllib` and `hashlib` from the standard library.

**Spec:** `docs/superpowers/specs/2026-11-05-soulmates-milestone-3-design.md`

## Global constraints

- Everything from the milestone 1 and 2 plans still holds. That covers the version floors, the ArviZ 1.x API, `uv run` for every command, the writing rules and marking made up values as made up.
- Do not commit. The user commits when they ask.
- `data/` stays out of git, and that includes `data/wpp_pool.csv`.
- Every code block below already ran in a scratch copy of the repo on 2026-11-05.

## File structure

```
scripts/download_wpp.py        # new: download, MD5 check, write data/wpp_pool.csv
src/soulmates/wpp.py           # new: POLYMOD_COUNTRIES, pool_size
tests/test_wpp.py              # new: pool_size tests on hand written rows
notebooks/02_polymod.ipynb     # changed: real M in the P_find section, two more known limits
```

---

### Task 1: WPP download script

**Files:**
- Create: `scripts/download_wpp.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `data/wpp_pool.csv` with the columns `location` (str, "World" or an ISO3 code), `year` (int), `age` (int, 0 to 100) and `pop_thousands` (float). It has 67,266 rows: 9 locations × 74 years × 101 ages.

This script talks to the network, so it has no unit test. Step 2 checks it by hand.

- [ ] **Step 1: Write the script**

`scripts/download_wpp.py`:

```python
"""Download the WPP 2024 single age file, check its MD5 and keep the rows for the soul mate pool."""
import hashlib
import pathlib
import urllib.request

import pandas as pd

NAME = "WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv.gz"
URL = f"https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/{NAME}"
MD5 = "f5e699a447a166783588c8919f18cc8b"  # recorded on 2026-11-05, the UN publishes no checksum
COUNTRIES = ["BEL", "DEU", "FIN", "GBR", "ITA", "LUX", "NLD", "POL"]  # the 8 POLYMOD countries
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def main():
    raw = DATA / "raw" / NAME
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        urllib.request.urlretrieve(URL, raw)
    if hashlib.md5(raw.read_bytes()).hexdigest() != MD5:
        raise SystemExit(f"{NAME}: checksum does not match the one recorded on 2026-11-05. "
                         f"The UN may have replaced the file. Delete {raw} and run again, or check the new file.")
    cols = ["Location", "ISO3_code", "Time", "AgeGrpStart", "PopTotal"]
    wpp = pd.read_csv(raw, encoding="utf-8-sig", usecols=cols, low_memory=False)  # the file starts with a BOM
    keep = wpp[(wpp["Location"] == "World") | wpp["ISO3_code"].isin(COUNTRIES)]
    pool = pd.DataFrame({
        "location": keep["ISO3_code"].fillna("World"),
        "year": keep["Time"],
        "age": keep["AgeGrpStart"],
        "pop_thousands": keep["PopTotal"],
    })
    pool.to_csv(DATA / "wpp_pool.csv", index=False)
    print(f"wrote {len(pool)} rows to {DATA / 'wpp_pool.csv'}")
    for label, locs in [("World", ["World"]), ("8 POLYMOD countries", COUNTRIES)]:
        rows = pool[pool["location"].isin(locs) & (pool["year"] == 2023) & pool["age"].between(25, 35)]
        print(f"{label}, aged 25 to 35 in 2023: {rows['pop_thousands'].sum() * 1000:,.0f} people")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python scripts/download_wpp.py`
Expected, in about 3 seconds, because the 62 MB file is already in `data/raw/` from the design step:

```
wrote 67266 rows to /Users/a.jebari/Desktop/Study/soulmates/data/wpp_pool.csv
World, aged 25 to 35 in 2023: 1,328,956,521 people
8 POLYMOD countries, aged 25 to 35 in 2023: 39,411,641 people
```

- [ ] **Step 3: Stop for review.** Do not commit.

---

### Task 2: Pool function

**Files:**
- Create: `src/soulmates/wpp.py`
- Test: `tests/test_wpp.py`

**Interfaces:**
- Consumes: a DataFrame with the columns of `data/wpp_pool.csv`, as written by Task 1.
- Produces: `POLYMOD_COUNTRIES` (tuple of 8 ISO3 codes) and `pool_size(wpp, age=30, year=2023, window=5, locations=("World",)) -> float`, the number of people. It raises `ValueError` when it finds fewer than `len(locations) × (2 × window + 1)` rows.

- [ ] **Step 1: Write the failing test**

`tests/test_wpp.py`:

```python
import pandas as pd
import pytest

from soulmates.wpp import pool_size

# Made up rows: 1 to 5 thousand people at ages 28 to 32 in 2023, the same for World and DEU
WPP = pd.DataFrame({
    "location": ["World"] * 5 + ["DEU"] * 5,
    "year": [2023] * 10,
    "age": [28, 29, 30, 31, 32] * 2,
    "pop_thousands": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
})


def test_pool_size_sums_the_window():
    # ages 29 to 31 hold 2 + 3 + 4 = 9 thousand people
    assert pool_size(WPP, age=30, window=1) == 9000


def test_pool_size_adds_locations():
    assert pool_size(WPP, age=30, window=1, locations=("World", "DEU")) == 18000


def test_pool_size_raises_on_missing_rows():
    with pytest.raises(ValueError, match="expected 3"):
        pool_size(WPP, age=30, year=2030, window=1)
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_wpp.py -v`
Expected: collection error with `ModuleNotFoundError: No module named 'soulmates.wpp'`

- [ ] **Step 3: Write the function**

`src/soulmates/wpp.py`:

```python
"""Pool size M from the UN World Population Prospects: everyone alive whose age is close to yours."""
POLYMOD_COUNTRIES = ("BEL", "DEU", "FIN", "GBR", "ITA", "LUX", "NLD", "POL")


def pool_size(wpp, age=30, year=2023, window=5, locations=("World",)):
    rows = wpp[
        wpp["location"].isin(locations)
        & (wpp["year"] == year)
        & wpp["age"].between(age - window, age + window)
    ]
    expected = len(locations) * (2 * window + 1)
    if len(rows) != expected:
        # a typo in a location or a year outside 1950 to 2023 would otherwise give M = 0
        raise ValueError(f"found {len(rows)} rows for {locations} in {year}, expected {expected}")
    return rows["pop_thousands"].sum() * 1000
```

- [ ] **Step 4: Run it and see it pass**

Run: `uv run pytest tests/test_wpp.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run it on the real rows**

Run: `uv run python -c "import pandas as pd; from soulmates.wpp import pool_size, POLYMOD_COUNTRIES; w = pd.read_csv('data/wpp_pool.csv'); print(f'{pool_size(w):,.0f}', f'{pool_size(w, locations=POLYMOD_COUNTRIES):,.0f}')"`
Expected: `1,328,956,521 39,411,641`

- [ ] **Step 6: Stop for review.** Do not commit.

---

### Task 3: Real M in notebook 02

**Files:**
- Modify: `notebooks/02_polymod.ipynb` (three cells)

**Interfaces:**
- Consumes: `pool_size` and `POLYMOD_COUNTRIES` from Task 2, `data/wpp_pool.csv` from Task 1, and `idata`, `p_find`, `np` and `plt` from the earlier cells of the notebook.
- Produces: the executed notebook. The P_find section prints both pools and plots both P_find posteriors.

- [ ] **Step 1: Edit the notebook**

Run this once from the repo root. It replaces three cells, each found by how its text starts, and fails loudly if it does not find exactly one match.

```bash
uv run python - notebooks/02_polymod.ipynb <<'EOF'
import sys

import nbformat as nbf

path = sys.argv[1]
nb = nbf.read(path, as_version=4)


def replace(prefix, cell):
    hits = [i for i, c in enumerate(nb.cells) if c.source.startswith(prefix)]
    assert len(hits) == 1, f"expected one cell starting with {prefix!r}, found {len(hits)}"
    nb.cells[hits[0]] = cell


replace("## The chance of finding your soul mate", nbf.v4.new_markdown_cell(
    "## The chance of finding your soul mate\n\n"
    "P_find = 1 − (1 − p_recog / M) ^ eye_contacts. M is everyone alive in 2023 aged 25 to 35, from the UN "
    "World Population Prospects 2024 (licence CC BY 3.0 IGO). Run `uv run python scripts/download_wpp.py` "
    "once before this section. p_eye and p_recog are still guesses with Beta(2, 2) priors, and they drive "
    "the result."))
replace("POOL_SIZE = 1e8", nbf.v4.new_code_cell(
    "import pandas as pd\n"
    "from soulmates.wpp import POLYMOD_COUNTRIES, pool_size\n\n"
    "wpp = pd.read_csv(\"../data/wpp_pool.csv\")  # written by scripts/download_wpp.py\n"
    "pools = {\"world\": pool_size(wpp), \"8 POLYMOD countries\": pool_size(wpp, locations=POLYMOD_COUNTRIES)}\n"
    "fig, ax = plt.subplots()\n"
    "for label, m in pools.items():\n"
    "    pf = p_find(idata, m).values.ravel()\n"
    "    print(f\"{label}: M = {m:,.0f}, P_find mean {pf.mean():.2e} (1 in {1 / pf.mean():,.0f}), \"\n"
    "          f\"94% interval {np.percentile(pf, 3):.2e} to {np.percentile(pf, 97):.2e}\")\n"
    "    ax.hist(np.log10(pf), bins=60, alpha=0.5, label=label)\n"
    "ax.set(xlabel=\"log10 of P(find your soul mate)\", ylabel=\"posterior draws\", "
    "title=\"P_find for two pools, age 30 in 2023\")\n"
    "ax.legend();"))
replace("## Known limits", nbf.v4.new_markdown_cell(
    "## Known limits\n\n"
    "1. Some participants may have reported extra contacts only as a total, without a frequency. If so, "
    "those contacts are missing here and the count of new contacts is too low. This has not been checked. "
    "The columns would be in `2008_Mossong_POLYMOD_participant_extra.csv`, which is not downloaded.\n"
    "2. The recovery test runs at alpha = 2, while this data sits near alpha = 0.23. The model is not "
    "tested at that level of spread.\n"
    "3. All 8 countries are pooled into one sample.\n"
    "4. The pool counts everyone and ignores gender and orientation.\n"
    "5. M is a fixed number from the WPP estimates, without uncertainty."))
nbf.write(nb, path)
print("updated", path)
EOF
```

Expected: `updated notebooks/02_polymod.ipynb`

- [ ] **Step 2: Execute it**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_polymod.ipynb`
Expected: exits without error after about 2 minutes. In the scratch trial the last cell printed:

```
world: M = 1,328,956,521, P_find mean 3.85e-06 (1 in 259,501), 94% interval 4.31e-07 to 9.70e-06
8 POLYMOD countries: M = 39,411,641, P_find mean 1.30e-04 (1 in 7,696), 94% interval 1.45e-05 to 3.27e-04
```

The figure shows two humps on the log10 axis, the world one centred near −5.5 and the 8 country one near −4.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: 7 passed.

- [ ] **Step 4: Stop for review.** Do not commit.
