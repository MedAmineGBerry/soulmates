# Soul mates with Bayes, milestone 4

This milestone changes the premise to "people you talk to", removes the guessed `p_eye`, and builds the prior for `p_recog` from a stated guess. It also adds a check of how much the result depends on Germany. The earlier specs are `2026-11-05-soulmates-design.md`, `2026-11-05-soulmates-milestone-2-design.md` and `2026-11-05-soulmates-milestone-3-design.md`. This spec only describes what changes.

## Why the premise changes

POLYMOD counts a contact in two cases (Mossong et al. 2008, Methods). Either two people touch skin to skin, such as in a kiss or a handshake. Or they hold a face to face conversation of at least three words. Strangers you only glance at are not in the data. Our lifetime count of new people therefore counts new people you meet and talk to.

The talk uses that version of the premise. You find your soul mate if you meet and talk to them and you both recognize each other. Munroe's version counts every glance, which would need a count of glances per day that no data here gives.

`p_eye` was the share of new contacts that include eye contact. A face to face conversation almost always does, so `p_eye` is close to 1. Its old prior had a mean of 0.5, which halved the lifetime count without a reason. This milestone removes it.

## The p_recog prior

`p_recog` is the chance that, when you meet your soul mate and talk for the first time, you both realize it then. No data informs it. The speaker states the guess as a sentence:

"I am 90% sure that p_recog lies between 0.1 and 0.6."

The talk gives this reason. Mossong et al. report that about three quarters of contacts with people not met before lasted less than 15 minutes. A short first talk is a weak moment to recognize anyone.

PreliZ `maxent` turns the sentence into a Beta prior. A Beta has two parameters and the sentence gives one condition, so many Beta curves meet it. `maxent` picks the one with maximum entropy, the flattest curve that puts 90% between 0.1 and 0.6, so the answer is unique. For this guess it is Beta(3.056, 5.543), with a mean of 0.355. Its tails are uneven: 2.9% below 0.1 and 7.1% above 0.6.

We first tried `pm.find_constrained_prior` and dropped it, for two reasons. PyMC 6.4 marks it as deprecated and points to PreliZ `maxent` instead. Its answer also depends on the starting guess: from Beta(2, 2) it returned Beta(2.732, 5.671), and from Beta(1, 5) it returned Beta(2.904, 5.537).

PreliZ 0.28 becomes a new dependency, `preliz>=0.28`. It installs next to PyMC 6.4.0, PyTensor 3.3.7 and ArviZ 1.3.0 without changing them, and `uv pip check` reports no conflicts. It brings numba with it.

## Model change

1. `p_eye` and `eye_contacts` are removed.
2. `build_model(df, search_ages=(18, 60), recog_bounds=(0.1, 0.6), recog_mass=0.9)` computes the Beta parameters with PreliZ `maxent` and adds `p_recog ~ Beta(alpha, beta)`.
3. Everything else stays as in milestone 2, including the weekend term.

## P_find change

`p_find(idata, pool_size)` reads `lifetime_new` and `p_recog` from the posterior and returns

```
P_find = −expm1(lifetime_new × log1p(−p_recog / pool_size))
```

which is 1 − (1 − p_recog / M) ^ lifetime_new.

## Loader and download change

1. `scripts/download_polymod.py` adds `2008_Mossong_POLYMOD_hh_common.csv` with the MD5 `d7fb1359ad84dba8cce4c444063940aa`. The file is 122,891 bytes and is already in `data/raw/`.
2. `load_polymod` joins the household file on `hh_id` and returns a fourth column, `country`. It holds the two letter code from the file: BE, DE, FI, GB, IT, LU, NL or PL.

## Germany check

Four countries, BE, DE, FI and NL, asked people to give an estimate of their professional contacts instead of listing each one (Mossong et al. 2008, Discussion). Those contacts have no frequency, so none of them can count as first time.

For ages 18 to 60, the mean of new contacts per day is 1.00 in those four countries and 1.58 in the other four. The gap is mostly Germany, at 0.35 per day. BE (1.32), FI (1.52) and NL (1.62) sit close to GB (1.27) and PL (1.25). Germany is also the largest sample, with 1341 participants. The data cannot tell whether Germany is low because of the diary rule or for another reason.

The notebook fits the model a second time without Germany and prints the lifetime count and the world P_find for both fits.

## Tests

1. `test_p_find_by_hand` uses `lifetime_new` in place of `eye_contacts`. With 2 new people, p_recog = 1 and M = 4, the result stays 1 − (3/4)² = 7/16.
2. A new test `test_recog_prior_matches_the_guess` builds the model on a tiny data frame and draws 4000 prior samples. It checks that the share of `p_recog` draws between 0.1 and 0.6 is 0.90 within 0.02. With 4000 draws, chance alone moves that share by about 0.005, so the limit is four times wider.
3. The loader test adds a small household file and checks the `country` column.
4. The recovery test does not change. It never used `p_eye` or `p_recog`.

## Notebooks

`notebooks/02_polymod.ipynb`:

1. The opening text and the P_find section say "meet and talk" in place of "lock eyes". They describe the POLYMOD contact definition in their own words.
2. The fit cell states the guess as a sentence and passes the two bounds as named constants. A new cell before the P_find section plots the `p_recog` draws with the two bounds marked. The posterior of `p_recog` equals its prior, so those draws show the prior.
3. The P_find cell keeps both pools, the world and the 8 POLYMOD countries.
4. A new cell fits the model without Germany and prints both lifetime counts and both world P_find values.
5. The known limits drop the limit on contacts reported only as a total, because the paper now answers it. They gain this one: four countries estimated professional contacts instead of listing them, and Germany records far fewer new contacts than the rest.

`notebooks/01_simulated.ipynb` gets the same change of wording in its opening text and runs again.

## Results from the scratch trial

A trial run of both notebooks in a scratch copy on 2026-11-05 gave these numbers:

1. Lifetime new contacts: 20,665, with a 94% interval from 19,212 to 22,180. Removing `p_eye` changes the random stream of the sampler, so this differs a little from the 20,688 of milestone 3.
2. World pool: P_find has a posterior mean of 5.53e-06, about 1 in 180,700. The 94% interval runs from 1.60e-06 to 1.04e-05. Milestone 3 gave 1 in 259,500, with an interval that spanned a factor of 22. The new interval spans a factor of 6.5.
3. The 8 POLYMOD countries: the posterior mean is 1.87e-04, about 1 in 5,400.
4. Without Germany: lifetime new contacts rise to 24,129, and the world figure becomes about 1 in 156,000.
5. Both fits, with and without Germany, had 0 divergences and r_hat of at most 1.00. The fit without Germany was checked in the repo after the build.

## Left out on purpose

1. Glances with strangers. They need a prior with no data behind it.
2. A country effect in the model. The Germany check is only a second fit.
3. The comparison with How Couples Meet and Stay Together and the Soul Mate Roulette demo. Both are later milestones.
