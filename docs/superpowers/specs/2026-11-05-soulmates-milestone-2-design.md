# Soul mates with Bayes, milestone 2

This milestone replaces the simulated diaries with the real POLYMOD contact survey. It adds a download script, a loader that turns the raw files into one row per participant, and a weekend term in the model. The milestone 1 spec is in `2026-11-05-soulmates-design.md`. This spec only describes what changes.

## The data

POLYMOD is a contact diary survey from 8 European countries, run in 2005 and 2006 (Mossong et al. 2008). Each participant recorded every person they met on one day. The data is on Zenodo as record 3874557, version 2, published on 3 June 2020, under the licence CC BY-NC 4.0. The licence allows non commercial use with attribution.

Three of the seven files are needed:

| File | Rows | Columns we use |
|---|---|---|
| `2008_Mossong_POLYMOD_participant_common.csv` | 7290 | `part_id`, `part_age` |
| `2008_Mossong_POLYMOD_contact_common.csv` | 97904 | `part_id`, `frequency_multi` |
| `2008_Mossong_POLYMOD_sday.csv` | 7290 | `part_id`, `dayofweek` |

`frequency_multi` says how often the participant usually meets that contact: 1 daily, 2 weekly, 3 monthly, 4 less often, 5 first time. A new contact is a contact with code 5. There are 7540 of them.

`dayofweek` runs from 0 for Sunday to 6 for Saturday. We checked this against the diary dates. The code matches the date in 7149 of 7154 diaries that have both.

## What the raw data shows

These numbers come from a first look at the files, before any model.

1. 68% of participants recorded no new contact. The largest count is 42.
2. On weekdays the mean is 0.99 new contacts per day and the variance is 6.49. On weekend days the mean is 1.16 and the variance is 8.08. People record more new contacts on weekends, so the weekend term must be free to take either sign.
3. A negative binomial with a mean of 1 and alpha near 0.2 gives about 70% zeros, close to the 68% observed. The model therefore needs no separate zero inflated part.
4. The sum of the raw band means over ages 18 to 60, times 365, is about 20,600 new people in a lifetime. This is only a quick check on the raw numbers. The model result comes later.

## Download

`scripts/download_polymod.py` downloads the three files from Zenodo with `urllib` into `data/raw/`. It checks each file against its MD5 checksum and stops with an error if one does not match. The checksums are written in the script:

| File | MD5 |
|---|---|
| `2008_Mossong_POLYMOD_participant_common.csv` | `ac784c281a71e15c7e64eaf2d6365709` |
| `2008_Mossong_POLYMOD_contact_common.csv` | `52baf32033bf780786fae6604043ec00` |
| `2008_Mossong_POLYMOD_sday.csv` | `148c5fc86ca10d385e3d047014e14a09` |

The URL pattern is `https://zenodo.org/api/records/3874557/files/<file>/content`. The folder `data/` is in `.gitignore`, so the raw files never enter the repo. The household file is not needed while the countries are pooled.

## Loader

`load_polymod(raw_dir, max_drop=0.05)` in `src/soulmates/polymod.py` returns a pandas DataFrame with one row per participant:

| Column | Type | Meaning |
|---|---|---|
| `age` | int | `part_age` |
| `new_contacts` | int | Number of the participant's contacts with `frequency_multi` = 5 |
| `weekend` | bool | `dayofweek` is 0 or 6 |

It applies three rules and prints how many rows each rule removes:

1. Drop participants with no `part_age`. This removes 65 of 7290 (0.9%).
2. Drop participants with no `dayofweek`. This removes 136 of 7290 (1.9%).
3. Do not count contacts with no `frequency_multi`. This leaves out 1407 of 97904 contacts (1.4%).

The first two rules overlap, so 7105 participants remain. If any rule removes a larger share than `max_drop`, the loader raises a `ValueError` that names the rule. A participant with no contact rows gets 0 new contacts. This applies to 36 participants.

All 8 countries are pooled into one sample.

## Model change

The model from milestone 1 gains one term for the day of the week:

```
# 2. Daily counts (READ from the diary)
beta_weekend   ~ Normal(0, 0.5)    # log of the weekend multiplier, centred on no effect
new_contacts[i] ~ NegativeBinomial(mean = band_mean[band[i]] × exp(beta_weekend × weekend[i]), alpha)

# 3. Lifetime counts (computed inside the model)
lifetime_new = Σ over ages 18 to 60 of 365 × band_mean[band(age)] × (5/7 + 2/7 × exp(beta_weekend))
```

`weekend` is read from the data. `beta_weekend` is estimated by the fit. The factor (5/7 + 2/7 × exp(beta_weekend)) averages a week of 5 weekdays and 2 weekend days. Everything else stays as in milestone 1.

`build_model(df, search_ages=(18, 60))` keeps its signature. The DataFrame must now carry the column `weekend`.

## Simulator change

`simulate_diaries(n_people, band_means, alpha, seed, weekend_effect=1.0)` marks each person as a weekend diary with probability 2/7. Their mean is multiplied by `weekend_effect`. The returned DataFrame gains the column `weekend`.

## Tests

1. `tests/test_recovery.py` sets the true weekend multiplier to 1.4, a made up value. The true lifetime count becomes 41062.5 × (5 + 2 × 1.4) / 7 = 45755.4. The test holds it as a literal, with the hand calculation in a comment. The recovery test also checks the posterior mean of `beta_weekend` against log(1.4) = 0.336. The tolerance comes from a trial over six seeds, as in milestone 1. The checks for alpha, band means and the lifetime count keep their limits, unless the same trial shows they no longer hold.
2. `tests/test_polymod.py` writes a few CSV rows into a temporary folder with pytest's `tmp_path`, so it never needs the real download. It checks:
   1. Only contacts with code 5 count as new.
   2. A contact with no `frequency_multi` does not count.
   3. A participant with no contact rows gets 0.
   4. `dayofweek` 0 and 6 give `weekend` True, and 1 to 5 give False.
   5. A participant with no age or no day of the week is dropped when `max_drop=1.0`.
   6. The same small data raises `ValueError` with the default `max_drop=0.05`.
3. `tests/test_p_find_by_hand` stays as it is.

## Notebook

`notebooks/02_polymod.ipynb` loads the data and plots the raw mean of new contacts per age band, split by weekday and weekend. It fits the model and shows the posterior band means, the posterior of the weekend multiplier exp(beta_weekend), and the posterior of the lifetime count. P_find still uses the placeholder pool of 1e8 people. The notebook states two known limits in plain words:

1. Some participants may have reported extra contacts only as a total, without a frequency. If so, those contacts are missing here and the count of new contacts is too low. This has not been checked. The columns would be in `2008_Mossong_POLYMOD_participant_extra.csv`, which is not downloaded.
2. The recovery test runs at alpha = 2, while the real data sits near alpha = 0.2. The model is not tested at that level of spread.

## Left out on purpose

1. A country effect. All 8 countries are pooled until the talk needs a single country.
2. A second recovery run at realistic values such as alpha = 0.2. It would double the test time.
3. The check of `participant_extra.csv` for contacts reported only as a total.
4. Uncertainty on M and the WPP data. Both belong to milestone 3.
