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
