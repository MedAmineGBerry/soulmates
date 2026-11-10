# Soul mates with Bayes, milestone 7: fixing the calculus

This milestone corrects one error in the model and several weak spots around it. It also gives every country its own contact level. It changes both headline numbers of the talk. The earlier specs are in this folder. Their numbers are superseded by the ones here.

## What was wrong

1. N and M counted different people. N, the number of new people you talk to, counted every age. M, the pool your soul mate comes from, counts only people within 5 years of your age. Only 24.9% of the first time contacts of adults in POLYMOD fall inside that window. The France 2012 survey gives 25.2%.
2. The odds counted new people from age 18 and the partner model counted from 15. In HCMST, 17.8% of people met their partner before 18.
3. The partner model had one match chance q for every age and gave everyone the same number of contacts. It missed ages 18 to 24 by 12 points. It also predicted 100% partnered at 50 to 60, where 3 of 828 people never had a partner.
4. The comparison of the one soul mate model with the free model was no contest. One side predicted that 1 in 100,000 people ever has a partner, and the data shows 96%.
5. The figure of 1.6 million possible partners in the world divided by a guess. It also assumed that the people you meet are a random draw from the whole world.
6. p_recog was defined for the first conversation only.
7. All countries shared one contact level, although they differ by a factor of up to five.

## Decisions

| Topic | Decision |
|---|---|
| Age window | Count only first time contacts within 5 years of the participant's age. |
| Start age | 15 everywhere. |
| Countries | Each of 9 countries gets its own contact level, drawn from a shared distribution. The 9 are the 8 POLYMOD countries and France. Germany stays in. |
| Headline | The odds for a person in a country we have no data for. This carries the spread between countries. |
| Best case | The most talkative country, shown next to the headline and labelled as the best case. |
| p_recog | The chance that you both realize it at some point after you have met and talked. The guess: "I am 90% sure it lies between 0.3 and 0.9." |
| Partner model | A match chance q and a number k for how much people differ in how many others they meet. |
| Comparison | The model with q and k against the model with one q. The one soul mate model is removed. |
| Twist | "About 14 of the people your age you will talk to." The world figure of 1.6 million is removed. |

The guess is higher because p_recog no longer has to happen in the first conversation. In HCMST, 62% of couples began their romance in the same year they met, and 80% within one year.

## Contacts inside the age window

`load_polymod(raw_dir, window=5, max_drop=0.05)` counts a first time contact only when the contact's age is within `window` years of the participant's age. The contact age is `cnt_age_exact`. Where that is missing, it is the middle of `cnt_age_est_min` and `cnt_age_est_max`. A contact with no age at all is not counted. In POLYMOD that concerns 4 of 7,540 first time contacts. The returned columns stay `age`, `new_contacts`, `weekend` and `country`.

## France 2012

The survey of Béraud et al. 2015 is on Zenodo as record 3886590, version 3, under CC BY-NC-ND 4.0. It has 1,755 participants, 4,066 diary days and 38,881 contacts, of which 4,728 are first time contacts. Each participant kept a diary on two days of a survey wave, and 278 participants took part in two waves. The 8 files are in `data/raw/france2012/` and match their Zenodo checksums. Its `dayofweek` runs from 0 for Sunday to 6 for Saturday, as in POLYMOD. We checked this against the diary dates.

`scripts/download_france.py` downloads the four files we use and checks their MD5 values. They hold the participants, the contacts, the diary days, and the link from each contact to its diary day.

`src/soulmates/france.py` holds `load_france_days(raw_dir, window=5)`. It returns one row per participant and diary day with the columns `part_id`, `wave`, `day`, `age`, `new_contacts`, `weekend` and `country`, where `country` is always "FR".

France serves three purposes:

1. It is the ninth country in the contact model. After dropping days without a weekday it adds 4,002 diary days.
2. It replicates POLYMOD. The share of adult first time contacts inside the age window is 25.2%, against 24.9%.
3. It shows that people differ. For adults, the counts of new people of any age on a person's two days correlate at 0.40, which corresponds to k = 0.60. For new people of the person's own age the counts are small and noisy. They correlate at 0.06, which corresponds to k = 1.62.

The two diary days lie in the same survey wave, so they also share whatever was going on in that person's life in those weeks. France therefore overstates lasting differences. Its values of k are a lower bound and are not used as a prior. The partner model estimates k from HCMST.

The contact model treats the diary days of one French participant as separate rows. That makes the French estimate a little too sure of itself.

## Contact model with a country level

```
# 1. Levels (estimated)
mu_pop        ~ Normal(log(0.4), 1)     # guess: about 0.4 new people of your age per day
sigma_band    ~ HalfNormal(0.5)         # how far age bands sit from the average
sigma_country ~ HalfNormal(0.5)         # how far countries sit from the average
z[band]       ~ Normal(0, 1)
log_factor[c] ~ ZeroSumNormal(sigma_country)                # the country terms add up to zero
band_mean[band]   = exp(mu_pop + sigma_band × z[band])      # in a typical country
country_factor[c] = exp(log_factor[c])

# 2. Daily counts (READ from the diaries)
beta_weekend ~ Normal(0, 0.5)
alpha        ~ HalfNormal(5)
new_contacts[i] ~ NegativeBinomial(mean = band_mean[band[i]] × country_factor[country[i]] × exp(beta_weekend × weekend[i]), alpha)

# 3. Lifetime counts, ages 15 to 60 (computed inside the model)
lifetime_typical    = Σ over ages of 365 × band_mean[band(age)] × (5/7 + 2/7 × exp(beta_weekend))
lifetime_country[c] = lifetime_typical × country_factor[c]
z_new               ~ Normal(0, 1)
lifetime_anywhere   = lifetime_typical × exp(sigma_country × z_new)

# 4. Recognition (prior only)
p_recog ~ Beta from PreliZ maxent, 90% between 0.3 and 0.9, which is Beta(3.61, 2.47)
```

The country terms add up to zero, so the typical country is the average of the 9 observed ones. A first version let them float freely. Then the overall level and the country levels could trade off, and a fit took two to four times as long.

`lifetime_anywhere` is the count for a person in a country we have no data for. It draws a new country from the same distribution the 9 countries came from, so it carries the spread between countries.

The old variable `lifetime_new` is replaced by `lifetime_typical`, `lifetime_country` and `lifetime_anywhere`.

`build_model(df, search_ages=(15, 60), recog_bounds=(0.3, 0.9), recog_mass=0.9)` needs the column `country`. `recog_prior` gets the same default bounds. `p_find(idata, pool_size, lifetime="lifetime_anywhere")` reads the named lifetime variable. For `lifetime_country` it returns one column of odds per country.

`simulate_diaries(n_people, band_means, alpha, seed, weekend_effect=1.0, country_factors=(1.0,))` assigns each person to a country at random and returns a `country` column.

The model cannot tell culture from survey method. Four countries, Germany among them, asked for an estimate of professional contacts instead of a list (Mossong et al. 2008, Discussion). Germany's low level may come from that. The model reads it as a quiet country.

## Partner model with differences between people

Each person has a personal multiplier λ on the number of new people they meet. λ follows a Gamma distribution with shape k and mean 1. A small k means that people differ a lot. A large k means that everyone is close to the average.

Given λ, the chance of having had no partner by age a is (1 − q) ^ (λ × N(a)). Averaged over all people, this gives

```
P_partner(a) = 1 − (1 + θ × N(a) / k) ^ (−k),   with θ = −log(1 − q)
```

As a hand check, θ × N(a) = 1 and k = 1 give P_partner = 1 − 1/2 = 0.5.

The model:

```
log10_theta ~ Normal(−3, 2)          # theta can be any positive number
q           = 1 − exp(−theta)        # so q always stays below 1
k           ~ LogNormal(0, 1)
ever_partnered[i] ~ Bernoulli(P_partner(age[i]))
```

N(a) comes from the typical country, through `band_mean` and `beta_weekend`.

The likelihood is written through the logit of P_partner. Written directly, P_partner rounds to exactly 1 for large N(a), and a person without a partner then gets an infinite penalty. In a trial that left one chain of the one q model stuck.

`src/soulmates/match.py` then holds:

1. `partner_chance(n, q, k)`, a numpy function for the formula above. The notebook and the demo use it.
2. `build_match_model(df, curve)`, the model with q and k.
3. `build_flat_model(df, curve)`, the old model with one q, kept for the comparison.
4. `lifetime_curve(posterior, start_age=15)`, unchanged.

`build_soulmate_model` is deleted.

## Results from the scratch trial

The whole milestone was built and run in a scratch copy on 2026-11-10, on 11,107 diary rows from 9 countries. All fits had 0 divergences and r_hat of at most 1.01. `sigma_country` came out at 0.54, alpha at 0.15, and a weekend day has 1.23 times the new contacts of a weekday.

| Where you live | New people of your age, 15 to 60 | Odds, world pool |
|---|---|---|
| A country without data (headline) | 5,817 median (2,003 to 16,976) | 1 in 325,800 (1 in 1,526,700 to 1 in 115,000) |
| A typical country | 5,778 median (5,247 to 6,362) | not computed |
| Italy | 9,617 (7,775 to 11,827) | 1 in 231,900 |
| Netherlands | 8,778 (6,047 to 12,488) | 1 in 253,300 |
| Finland | 6,767 (5,503 to 8,265) | 1 in 329,500 |
| France | 6,642 (5,805 to 7,551) | 1 in 335,800 |
| Luxembourg | 6,623 (5,433 to 8,065) | 1 in 336,500 |
| Belgium | 6,337 (5,028 to 7,955) | 1 in 352,000 |
| Poland | 5,318 (4,323 to 6,493) | 1 in 419,700 |
| Great Britain | 5,105 (4,107 to 6,250) | 1 in 437,000 |
| Germany | 1,777 (1,363 to 2,266) | 1 in 1,255,300 |

For the 8 POLYMOD countries as the pool, the headline is 1 in 9,700.

The old headline was 1 in 180,700. With the age window alone, and the old guess for p_recog, the odds were 1 in 670,000.

## Best case

The talk shows a best case next to the headline and labels it as one:

1. Italy, with the stated guess for p_recog: 1 in 231,900.
2. Italy, if recognition were certain: 1 in 138,200.
3. Online meeting can only add to this. In HCMST, 37% of the couples who met in 2010 or later met online. The diaries of 2005 to 2012 cannot see such meetings, and we have no number for them.

The headline also has a warmer reading. At 1 in 325,800, about 4,100 of the 1.33 billion people of your generation do find their one soul mate.

With exactly one soul mate, no choice in this spec brings the odds above about 1 in 138,000.

## Partner model results

On the curve of the typical country, which reaches 5,783 people by 60:

| Quantity | q and k | One q |
|---|---|---|
| q | 1 in 424 (1 in 302 to 1 in 554) | 1 in 794 |
| k | 2.95 (1.97 to 4.84) | none |
| Predicted share with a partner, 18 to 24 | 78.5% | 67.8% |
| Observed share, 18 to 24 | 80.7% | 80.7% |

The model with q and k beats the one q model by 10.0 points of LOO score, with a standard error of 6.2. That is moderate evidence. The clearer gain is the fit at ages 18 to 24.

The average person talks to about 14 possible partners of their own age by 60, with an interval from 10 to 19. The model leaves 0.6% of people without a partner by 60.

## Demo

`spin` draws q and k together from the posterior. For each life it draws λ from a Gamma distribution with shape k and mean 1. The life then meets λ × N(age) people, times the factor of the chosen country. The real world rule finds a partner with chance 1 − (1 − q) ^ (λ × N(age)). The age at which the partner was met comes from the curve scaled by λ. The soul mate rule uses the same number of people.

The app gains a choice "where you live": a typical country, or one of the 9 countries. `data/demo_inputs.json` gains the k draws and the mean factor of each country. Its curve is the one of the typical country, from age 15. The app text drops the world figure of 1.6 million and shows how many possible partners the 1,000 lives met on average.

The old limit "every simulated life meets exactly the average number of people" no longer holds and leaves the notebook and the talk.

## Tests

1. `test_polymod.py`: the hand written contact rows gain ages. A first time contact 6 years older than the participant is not counted with `window=5` and is counted with `window=10`. A contact with only an estimated age range uses the middle of the range.
2. `test_france.py`: `load_france_days` on tiny files written by hand. Two days for one person give two rows.
3. `test_match.py`:
   1. `partner_chance(1, 1 − exp(−1), 1)` is 0.5.
   2. Recovery: simulate 6,000 people with a known q and k, fit `build_match_model`, and check both within a tolerance from a trial over four seeds.
   3. The hand check of `lifetime_curve` stays.
4. `test_recovery.py`: the recovery test simulates 6,000 people in four countries with made up factors. It runs at alpha = 0.2, close to the 0.15 of the real data. It checks alpha, the weekend effect, the band means and the lifetime counts of the typical country and of each country. The tolerances come from a trial over six seeds. The prior test uses the bounds 0.3 and 0.9. The hand test of `p_find` uses `lifetime_anywhere`.
5. `test_roulette.py`: with q = 0 nobody finds a partner. With q = 1 every life with at least one contact finds a partner at the first meeting. With a known q and k, the share of lives with a partner lands near `partner_chance`. A country factor of 2 doubles the number of people a life meets.

## Notebooks and housekeeping

All three notebooks run again. Notebook 02 shows the table of countries, the headline and the best case. Notebook 03 gains the France section and the comparison of the two partner models. `.DS_Store` goes into `.gitignore`.

## Left out on purpose

1. Countries outside Europe. Zenodo holds surveys in the same format for Taiwan, China, Peru and others. Each needs its own check first. In the first rows we read, 7% of contacts in Taiwan and 0.4% in Peru are first time contacts.
2. A comparison of today with 2006. One survey published in 2026 has a "never met before" answer. We have not checked which country it covers.
3. The European Social Survey, round 3, 2006, with its questions on ever having lived with a partner. It would replace the US partner data with European data from the same year as POLYMOD. It needs a registration by the speaker.
4. Differences in loneliness between countries. Only France has two diary days per person, so k cannot vary by country.
5. The speed dating data and the rework of the deck. Both follow this milestone.
6. A match chance q that changes with age.
