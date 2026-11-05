# Soul mates with Bayes, milestone 3

This milestone replaces the placeholder pool of 1e8 people with real cohort sizes from the UN World Population Prospects. It adds a download script, a function for the pool size M, and the real P_find in the POLYMOD notebook. The earlier specs are `2026-11-05-soulmates-design.md` and `2026-11-05-soulmates-milestone-2-design.md`. This spec only describes what changes.

## The data

World Population Prospects 2024 (WPP 2024) is the latest edition of the UN population estimates. We found no 2026 edition on 2026-11-05. The file we use is

`https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv.gz`

It is 62,082,217 bytes and was last changed on 13 December 2024. Its MD5 on 2026-11-05 was `f5e699a447a166783588c8919f18cc8b`. The UN publishes no checksum, so this value is our own record. Our World in Data and the `wpp2024` R package list the licence as CC BY 3.0 IGO, with the UN as licensor. We did not find the UN's own licence page.

The file starts with a byte order mark, so pandas must read it with `encoding="utf-8-sig"`. It has 4,148,070 rows. The columns we use:

| Column | Meaning |
|---|---|
| `Location` | Name of the country or region, "World" for the whole world |
| `ISO3_code` | Three letter country code, empty for regions |
| `Time` | Year, from 1950 to 2023 |
| `AgeGrpStart` | Age in years, from 0 to 100, where 100 stands for 100 and over |
| `PopTotal` | Population in thousands, both sexes |

The world total for 2023 reads 8,091,735 thousand, which is 8.09 billion people. The 8 POLYMOD countries appear with the codes BEL, DEU, FIN, GBR, ITA, LUX, NLD and POL. None of their rows, and none of the world rows, has a missing `PopTotal`.

## Pool definition

M is the number of people alive in a reference year whose age is within `window` years of yours. The defaults are age 30, year 2023 and window 5, so the pool is everyone aged 25 to 35 in 2023.

| Pool | M |
|---|---|
| World | 1,328,956,521 |
| The 8 POLYMOD countries | 39,411,641 |

We count people alive, and not people born in those years, because a person who died as a child can never be met. M counts everyone, whatever their gender or orientation, as the premise of the chapter does. M is a fixed number. WPP gives single estimates for past years. An error of a few percent in M is small next to the guessed priors for `p_eye` and `p_recog`.

## Download

`scripts/download_wpp.py` does four things:

1. It downloads the file into `data/raw/` with `urllib` if it is not there yet.
2. It checks the MD5 against `f5e699a447a166783588c8919f18cc8b` and stops with an error if it does not match.
3. It keeps the world rows and the rows of the 8 countries, for every year and age. It writes them to `data/wpp_pool.csv` with the columns `location` ("World" or the ISO3 code), `year`, `age` and `pop_thousands`. That is 9 × 74 × 101 = 67,266 rows.
4. It prints the world pool and the 8 country pool for the defaults, as a check against the table above.

The notebook reads only `data/wpp_pool.csv`.

## Pool function

`src/soulmates/wpp.py` holds:

```python
POLYMOD_COUNTRIES = ("BEL", "DEU", "FIN", "GBR", "ITA", "LUX", "NLD", "POL")

def pool_size(wpp, age=30, year=2023, window=5, locations=("World",)):
    ...
```

`wpp` is the DataFrame read from `data/wpp_pool.csv`. The function sums `pop_thousands` over the given locations, the given year and the ages from `age − window` to `age + window`, and multiplies by 1000. It expects exactly `len(locations) × (2 × window + 1)` rows. If it finds fewer, it raises `ValueError`. Without that check, a typo in a location or a year outside 1950 to 2023 would give M = 0 and a broken P_find.

## Tests

`tests/test_wpp.py` uses rows written by hand inside the test:

1. The World rows hold 1, 2, 3, 4 and 5 thousand people at ages 28 to 32 in 2023. Then `pool_size(wpp, age=30, window=1)` returns (2 + 3 + 4) × 1000 = 9000.
2. With the same rows for a second location, `pool_size(wpp, age=30, window=1, locations=("World", "DEU"))` returns 18000.
3. `pool_size(wpp, age=30, year=2030, window=1)` raises `ValueError`, because the rows have no year 2030.

The tests from milestones 1 and 2 stay as they are.

## Notebook

In `notebooks/02_polymod.ipynb`, the P_find section changes:

1. It reads `data/wpp_pool.csv` and computes M for the world and for the 8 countries with `pool_size`.
2. It prints P_find for both pools, as the posterior mean and the 94% interval, and also as "1 in N" for the mean.
3. It plots both P_find posteriors on a log scale.
4. The markdown says that `p_eye` and `p_recog` are still guesses and drive the result.

The rough expectation before any run is about 3.9e-06 for the world and 1.3e-04 for the 8 countries. Both come from 20,688 × 0.25 / M. Here 20,688 is the posterior mean of lifetime new contacts from milestone 2, and 0.25 is the product of the prior means of `p_eye` and `p_recog`.

The section "Known limits" gains two points:

1. The pool counts everyone and ignores gender and orientation.
2. M is a fixed number from WPP estimates, without uncertainty.

`notebooks/01_simulated.ipynb` keeps the placeholder pool, because its data is simulated.

## Left out on purpose

1. Uncertainty on M. It was on the list for this milestone and is dropped for the reason given under "Pool definition".
2. Better priors for `p_eye` and `p_recog`. They are milestone 4.
3. Pools that change over the search years. M is counted once, at the reference age.
