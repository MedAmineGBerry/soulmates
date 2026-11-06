# Soul mates with Bayes, milestone 5

This milestone tests the soul mate premise against real partnership data. It asks how many possible partners a person would need for the model to match how many people actually find one. The data is the 2017 wave of How Couples Meet and Stay Together (HCMST). The earlier specs are in this folder. This spec only describes what is new.

## The idea

Each new person you talk to is a possible partner with a small chance q. By age a you have talked to N(a) new people. The chance that you have had a partner by age a is

```
P_partner(a) = 1 − (1 − q) ^ N(a)
```

The soul mate model is the special case with exactly one possible partner in a pool of M people, whom you recognize with chance p_recog. It fixes q = p_recog / M, about 1 in 3.7 billion for the world pool. The free model lets the HCMST data choose q.

The number of possible partners in the world pool is

```
K = q × M / p_recog
```

The soul mate premise says K = 1. The talk shows how large K has to be.

Take a small example. With q = 1 in 2,000 and N = 7,000 new people by the mid twenties, P_partner = 1 − (1 − 1/2000)^7000. That is about 0.97.

## The data

HCMST 2017 to 2022, small public version 2.2, from the Stanford data page `https://data.stanford.edu/hcmst2017`. The file `HCMST 2017 to 2022 small public version 2.2.dta` is 4,198,913 bytes. Its MD5 on 2026-11-06 was `fe0000cbf379bebcb86ba8d7b00cbd9a`. The codebook `HCMST 2017 to 2022 v2.2 codebook.pdf` sits next to it in `data/raw/`.

The page names no licence. Downloading means accepting a data use agreement with four conditions:

1. The data is not used to identify individuals.
2. No fee is charged when it is passed on.
3. The user tells the contact person for the dataset about the work.
4. The user cites the data. The speaker accepted it on 2026-11-06. The speaker still has to tell the HCMST contact person about the talk.

Each user has to accept the agreement, so the repo has no download script for HCMST. The notebook tells people to accept it on the Stanford page and save the Stata file to `data/raw/`.

The 2017 wave has 3,510 respondents and 725 columns. The columns we use:

| Column | Meaning |
|---|---|
| `w1_ppage` | Age at the 2017 interview, 18 to 93, none missing |
| `w1_partnership_status` | 1 married (2,085), 2 partnered and not married (777), 3 no partner now but a past partner (541), 4 never had a partner (107) |
| `w1_weight_combo` | Survey weight, from 0.059 to 3.111, none missing |

Pandas reads the file with `pd.read_stata(path, convert_categoricals=False)`. It warns that some strings fall back to latin-1. None of the three columns above is a string, so the warning does not affect them.

## What the raw data shows

For ages 18 to 60 there are 2,520 people. 98 of them never had a partner.

| Age | People | Never had a partner | Ever had one | Ever, weighted |
|---|---|---|---|---|
| 18 to 24 | 254 | 49 | 0.807 | 0.768 |
| 25 to 29 | 410 | 31 | 0.924 | 0.910 |
| 30 to 34 | 251 | 3 | 0.988 | 0.985 |
| 35 to 39 | 272 | 7 | 0.974 | 0.970 |
| 40 to 49 | 505 | 5 | 0.990 | 0.990 |
| 50 to 60 | 828 | 3 | 0.996 | 0.996 |

The weighted shares are at most 4 points lower than the unweighted ones, and only at the youngest ages.

## Lifetime curve

N(a) comes from the POLYMOD fit of milestone 4. For each age it adds up the new people met per year, from `start_age` up to and including age a:

```
per_year(age) = 365 × band_mean[band(age)] × (5/7 + 2/7 × exp(beta_weekend))
N(a)          = Σ over ages start_age to a of per_year(age)
```

`band_mean` and `beta_weekend` are the posterior means. N(a) is treated as known, without its uncertainty of about ±4%. A joint model would let the HCMST answers pull on the POLYMOD contact rates, which we do not want.

`start_age` defaults to 15. In a trial, starting at 18 gave an 18 year old only 355 new people. That made the 81% partnered at 18 to 24 hard to explain. The LOO score preferred 15: an expected log score (elpd) of −347 against −391 for 18, where higher is better.

## Models

The free model:

```
log10_q ~ Normal(−4, 2)                                  # wide: 1 in 100 million to near certain
ever_partnered[i] ~ Bernoulli(1 − (1 − 10^log10_q) ^ N(age[i]))
```

The soul mate model:

```
p_recog ~ Beta from PreliZ maxent, the same guess as milestone 4
ever_partnered[i] ~ Bernoulli(1 − (1 − p_recog / M) ^ N(age[i]))
```

Both are computed as `−expm1(N × log1p(−q))`, so a tiny q does not round to zero. Both keep the pointwise log likelihood, so ArviZ can compare them with LOO (leave one out cross validation). M is the world pool from milestone 3, 1,328,956,521.

K is computed from the draws. q comes from the posterior of the free model, and p_recog from its Beta prior with the same number of draws.

## Trial results

A trial on 2026-11-06, with `start_age=15`:

1. q is about 1 in 2,400, with a 94% interval from 1 in 2,230 to 1 in 2,610. The sampler had 0 divergences.
2. K, using the prior mean 0.356 for p_recog, is about 1.5 million.

A second trial ran the whole notebook in a scratch copy on 2026-11-06:

1. q is 1 in 2,419, with a 94% interval from 1 in 2,230 to 1 in 2,609.
2. The soul mate model predicts that a share of 9.7e-06 of people ever had a partner. HCMST observes 0.961.
3. LOO ranks the free model first, with an elpd of about −350. The soul mate model is about 27,900 lower, with a standard error of about 140.
4. K, from the draws, has a median of 1,589,085 and a 94% interval from 817,943 to 5,532,943.
5. Both models had 0 divergences.
6. The free model predicts about 69% partnered at 18 to 24, where HCMST shows 81%. From 25 on it matches the observed shares within about 2 points.

## Files and interfaces

```
src/soulmates/hcmst.py      # load_hcmst
src/soulmates/match.py      # lifetime_curve, build_match_model, build_soulmate_model
tests/test_hcmst.py         # loader test on a tiny Stata file
tests/test_match.py         # lifetime_curve by hand, recovery of q
notebooks/03_partners.ipynb # the comparison
```

`load_hcmst(path)` returns a DataFrame with `age` (int), `ever_partnered` (bool, status is not 4) and `weight` (float), for ages 18 to 60.

`lifetime_curve(posterior, start_age=15)` returns a numpy array with N(a) for ages 0 to 100. Ages below `start_age` get 0.

`build_match_model(df, curve)` returns the free model. `build_soulmate_model(df, curve, pool_size, recog_bounds=(0.1, 0.6), recog_mass=0.9)` returns the soul mate model.

## Tests

1. `lifetime_curve` on a posterior made by hand with every band mean 1 and `beta_weekend` 0. Then per_year is 365, and age 17 counted from 15 gives 3 × 365 = 1095. Age 14 gives 0.
2. `load_hcmst` on a tiny Stata file that the test writes with `DataFrame.to_stata` into a temporary folder. It checks the age filter, the mapping of status 4 to False and the weight column.
3. Recovery of q. Simulate `ever_partnered` for 3000 people with a made up q of 1 in 2,000 and a made up curve. Fit the free model and check that the posterior median of q lies within a tolerance. The tolerance comes from a trial over several seeds, as in the earlier milestones.

## Notebook

`notebooks/03_partners.ipynb`:

1. A note on the data use agreement and where to save the file.
2. The observed shares by age band, weighted and unweighted.
3. The POLYMOD fit and the lifetime curve.
4. The free model: q, and the predicted share by age band next to the observed share.
5. The soul mate model and the LOO comparison of the two.
6. The headline: K with its interval, and "1 in X people you talk to could be a partner".
7. Known limits:
   1. The contact rates come from Europe in 2006 and the partner data from the US in 2017.
   2. The survey weights are not in the likelihood. The weighted shares are shown as a check.
   3. N(a) is treated as known.
   4. q is the same at every age.
   5. "Ever had a partner" counts any past partner, so a short relationship counts the same as a life long one.

## Left out on purpose

1. A q that changes with age.
2. The later HCMST waves of 2020 and 2022.
3. Survey weights inside the model.
4. The Soul Mate Roulette demo, which is the next milestone.
