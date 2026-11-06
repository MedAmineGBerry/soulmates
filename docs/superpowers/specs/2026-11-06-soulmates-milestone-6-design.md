# Soul mates with Bayes, milestone 6: Soul Mate Roulette

This milestone builds the live demo for the talk. Each spin plays out simulated lives and checks them under two sets of rules. One is a single soul mate in the pool. The other is the real world match chance q from HCMST. The audience sees that soul mate lives almost never end in a match, while real world lives almost always do. The earlier specs are in this folder. This spec only describes what is new.

## What one spin does

A spin plays out a number of lives, 1 or 1,000. For each life:

1. Draw one q from the posterior of the free HCMST model of milestone 5.
2. Draw one p_recog from the prior built from the current guess with PreliZ `maxent`.
3. Take N = N(age), the number of new people met and talked to from age 15 up to the chosen age, from the POLYMOD curve.
4. Real world rules: the life found someone with chance 1 − (1 − q) ^ N. If it did, draw the number of the first matching meeting from a geometric distribution with chance q. Turn that number into an age through the curve, the first age a with N(a) at or above it.
5. Soul mate rules: the life found its soul mate with chance 1 − (1 − p_recog / M) ^ N. M is the pool size for the chosen age and pool.

Steps 1 and 2 put the uncertainty about q and p_recog into the lives. The rest of the randomness is who the N people turn out to be.

## Why every life meets exactly N(age) people

POLYMOD records one day per person. The spread in its counts therefore mixes day to day noise for the same person with lasting differences between people. The data cannot separate the two. If the spread were all day to day noise, a life adds up about 16,000 days of draws, and lives would barely differ. If it were all lasting differences, alpha near 0.23 would leave many people who meet very few new people in their whole life.

The free model of milestone 5 assumed that everyone of age a met exactly N(a) people. A demo with lasting differences would find fewer partners than HCMST shows and would contradict the model it presents. So every simulated life meets exactly N(age) people. The notebook and the talk name this as a simplification.

## Demo inputs

`scripts/export_demo_inputs.py` writes `data/demo_inputs.json`:

| Key | Content |
|---|---|
| `curve` | N(a) for ages 0 to 100, from `lifetime_curve` with `start_age=15` |
| `q` | 4,000 posterior draws of q from the free HCMST model |
| `pools` | For each age from 18 to 60, the world pool and the 8 POLYMOD country pool, from `pool_size` with window 5 in 2023 |

The script fits the POLYMOD model of milestone 4 and the free model of milestone 5. It uses the same seeds as the notebooks, so the numbers match them. It needs the HCMST file, so only the speaker runs it. In the trial on 2026-11-06 it took about 4 minutes.

The JSON holds model results and no survey rows, so it is committed to git. `.gitignore` ignores `data/`, so it gains an exception for this one file. The demo then runs from a fresh clone, without the HCMST agreement and without any download.

## The spin function

`src/soulmates/roulette.py` holds:

```python
def load_inputs(path): ...
def spin(inputs, age, pool, recog_bounds, n_lives, seed, recog_mass=0.9): ...
```

`load_inputs` reads the JSON into numpy arrays. `spin` returns a pandas DataFrame with one row per life and the columns `q`, `p_recog`, `found_partner` (bool), `age_met` (int, or missing when no partner was found) and `found_soul_mate` (bool). `pool` is `"world"` or `"europe"`. `spin` raises `ValueError` for an age outside 18 to 60 or an unknown pool.

## The app

`demo/roulette.py` is a marimo app. marimo is a reactive Python notebook: when a slider moves, every cell that depends on it runs again. It has these controls:

1. Age, a slider from 18 to 60, default 30.
2. Pool, world or the 8 POLYMOD countries, default world.
3. The p_recog guess, two sliders for the lower and upper bound, default 0.1 and 0.6. Next to them stands the sentence "I am 90% sure p_recog lies between L and U".
4. Two buttons, "Spin 1 life" and "Spin 1,000 lives".

It shows:

1. For one life: the number of new people met by that age, and whether and at which age the real world rules found someone. Next to it, whether the soul mate rules did.
2. For 1,000 lives: the tally under both rules, as two numbers and a bar chart.
3. K = q × M / p_recog for the current guess and pool, as a median.

The app has no logic of its own. It calls `load_inputs` and `spin`.

It starts offline with `uv run marimo run demo/roulette.py`. Its first cell imports PyMC and PreliZ, so in the trial the controls appeared after about 25 seconds. The speaker starts the app before the talk.

In the trial, 1,000 lives at age 30 gave 972 and 974 possible partners and 0 soul mates for the world pool. For the 8 POLYMOD countries they gave 965 partners and 1 soul mate.

marimo 0.25.2 becomes a new dependency. On 2026-11-06 it installed next to PyMC 6.4.0, PyTensor 3.3.7, ArviZ 1.3.0 and PreliZ 0.28.0 without changing them, and `uv pip check` reported no conflicts.

## Tests

`tests/test_roulette.py` uses inputs made by hand, never the real JSON:

1. With every q draw at 0, no life finds a partner.
2. With every q draw at 1, every life finds a partner at the first meeting. The age is the first age with N(a) at least 1, which is 15 when every year adds people from 15 on.
3. Set every q draw to a known value and N to a known count. The share of lives that found a partner must then land near 1 − (1 − q) ^ N. With 1,000 lives, chance alone moves that share by about 0.016 at a share of 0.5, so the tolerance is 0.06.
4. An age of 17 and an unknown pool each raise `ValueError`.

The marimo file gets no test. The export script gets no test, as it only calls code that already has tests.

## Left out on purpose

1. A static browser version of the demo, which marimo can export. It is a nice extra for later.
2. Lasting differences between people in how many others they meet.
3. Any model fitting inside the demo.
