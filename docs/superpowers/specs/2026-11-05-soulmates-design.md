# Soul mates with Bayes, milestone 1

This milestone builds the whole pipeline on simulated data. It simulates contact diaries, fits a PyMC model for new contacts per day, and turns the fit into the probability of finding your soul mate. A test shows that the model recovers known true values. It is the base for a PyCon talk inspired by the "Soul Mates" chapter of Randall Munroe's What If?.

## The question

Everyone has exactly one soul mate, drawn at random from the people born around the same time. You find them only if you lock eyes and recognize each other. What is the chance that this happens in a lifetime?

Two numbers answer it. M is the pool size: everyone born within 5 years of you, before or after. N is the number of new people you make eye contact with between ages 18 and 60. If each of those people is your soul mate with probability 1/M, the chance of finding them is

```
P_find = 1 − (1 − p_recog / M) ^ N
```

where p_recog is the probability that you recognize your soul mate when you lock eyes. When N is tiny next to M this is close to N × p_recog / M.

## Scope

In scope for milestone 1:

1. A simulator for contact diaries with known true values.
2. The PyMC model for daily new contacts by age band, with lifetime eye contacts computed inside the model.
3. A numpy function that turns posterior draws into P_find for a given M.
4. A recovery test and a hand calculated test for P_find.
5. One notebook that walks from simulated data to the P_find posterior.

Later milestones, each with its own spec:

1. Real POLYMOD diaries from Zenodo, with weekday and weekend effects and input validation.
2. M from UN World Population Prospects, with uncertainty on M.
3. Model comparison against real partnership data from How Couples Meet and Stay Together.
4. The Soul Mate Roulette interactive demo.

The "people never change" growth model is a separate repo and is not part of this project.

## Data shape

One row per participant, as in a POLYMOD diary:

| Column | Type | Meaning |
|---|---|---|
| `age` | int | Age of the participant in years |
| `new_contacts` | int | Contacts the participant marked as "first time" on their diary day |

POLYMOD records one day per participant, so a row is one person on one day. Only "first time" contacts count, because every repeat contact was a first time contact once and counting it again would count the same person twice.

## Age bands

Ages fall into 5 year bands: `band = min(age // 5, 16)`. That gives 17 bands, from 0 to 4 up to a last band for ages 80 and over. The search years 18 to 60 cover bands 3 to 12.

## Model

```
# 1. Band means (estimated)
mu_pop     ~ Normal(log(2), 1)    # average log of new contacts per day, over all bands
sigma_band ~ HalfNormal(0.5)      # how far bands sit from that average
z[band]    ~ Normal(0, 1)
log_mu[band] = mu_pop + sigma_band × z[band]

# 2. Daily counts (READ from the diary)
new_contacts[i] ~ NegativeBinomial(mean = exp(log_mu[band[i]]), alpha)
alpha ~ HalfNormal(5)

# 3. Lifetime counts (computed inside the model)
lifetime_new = Σ over ages 18 to 60 of 365 × exp(log_mu[band(age)])
p_eye        ~ Beta(2, 2)
eye_contacts = lifetime_new × p_eye

# 4. Recognition (prior only)
p_recog      ~ Beta(2, 2)
```

Where each name comes from:

| Name | Origin |
|---|---|
| `age`, `new_contacts` | Read from the data |
| `band` | Computed from `age` |
| `mu_pop`, `sigma_band`, `z`, `alpha` | Estimated by the fit |
| `p_eye`, `p_recog` | Prior only. No data informs them, so their posterior equals their prior |
| `band_mean` (that is exp(log_mu)), `lifetime_new`, `eye_contacts` | Computed in the model as `pm.Deterministic`, so they appear in the trace |

The prior on `mu_pop` centres on log(2), which is a made up guess of about 2 new people per day. It is wide on purpose. `alpha` is the dispersion of the negative binomial: a small alpha means more days with very many contacts. Writing the band means through `z` is the non centred form. It is the same model, and the sampler handles it better when bands differ only a little.

`p_eye ~ Beta(2, 2)` and `p_recog ~ Beta(2, 2)` are guesses centred on 0.5. They are marked as guesses in the code. A later talk section turns elicited guesses into better priors.

## P_find

`p_find(idata, pool_size)` reads the `eye_contacts` and `p_recog` draws from the posterior and returns

```
P_find = −expm1(eye_contacts × log1p(−p_recog / pool_size))
```

This is the same formula as above. `log1p` and `expm1` keep a tiny probability from rounding to zero.

M stays outside the sampler, so a change of pool size needs no refit. In milestone 1 the notebook uses M = 1e8 and labels it as a placeholder until the WPP data arrives.

## Files and interfaces

```
pyproject.toml                 # uv project, Python 3.13, pymc, arviz, pandas, matplotlib, pytest
src/soulmates/simulate.py      # simulate_diaries
src/soulmates/model.py         # build_model, p_find
tests/test_recovery.py         # recovery test and P_find test
notebooks/01_simulated.ipynb   # the talk walkthrough
```

`simulate_diaries(n_people, band_means, alpha, seed)` draws ages uniformly from 0 to 84, maps them to bands, and draws `new_contacts` from a negative binomial with the band mean and alpha. It returns a pandas DataFrame with the columns `age` and `new_contacts`.

Version floors: PyMC 6.4, PyTensor 3.3.7 and ArviZ 1.3. PyTensor 2.31 passes the linker flag `-ld64` on macOS 15 and later. The Apple clang 21 linker on this machine rejects it, so every model compile fails. PyTensor 3.3.7 skips the flag for clang 21. ArviZ 1.x renamed `hdi_prob` to `prob` and no longer installs matplotlib, so matplotlib is a direct dependency.

`build_model(df, search_ages=(18, 60))` returns a `pm.Model` built as in the model section.

`p_find(idata, pool_size)` returns the P_find draws as an xarray DataArray with the same chains and draws as the posterior.

## Tests

`test_recovery.py` holds two tests.

1. Recovery. Simulate 3000 people with a fixed seed, 17 fixed band means between 1 and 4 per day and alpha = 2. These true values are made up for the test. Fit the model and check that each posterior mean lies close to its true value:
   1. `alpha` within 25%.
   2. `lifetime_new` within 10%. The test holds the true value 41062.5 as a literal, with the hand calculation in a comment. The code under test never computes it.
   3. Every one of the 17 band means within 40%.

   These limits come from a trial over six seeds in a scratch environment. The largest errors were 14% for alpha, 4.5% for `lifetime_new` and 26% for a band mean. An earlier version checked whether each true value lies inside its 94% HDI (highest density interval, the narrowest range that holds 94% of the posterior). That check fails by design in about 6% of runs for each value. In the trial, two of four seeds failed the band mean version of it, so it was dropped.
2. P_find by hand. With 2 eye contacts, p_recog = 1 and M = 4, the result is 1 − (3/4)² = 7/16. The test builds a small InferenceData by hand that holds exactly those draws, so it needs no sampling.

## Notebook

`01_simulated.ipynb` first simulates the diaries, plots new contacts by age and runs a prior predictive check. Then it fits the model, compares the posterior band means with the true ones and plots the P_find posterior. Each step uses the functions above and nothing else.

## Left out on purpose

1. Weekday and weekend effects. They belong to the real POLYMOD milestone.
2. Input validation beyond what pandas gives. Real data is where bad rows appear.
3. Uncertainty on M. It arrives with the WPP data.
