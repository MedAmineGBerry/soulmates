"""Who has had a partner by which age: a free match chance q against the one soul mate premise."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from soulmates.model import age_band, recog_prior

AGES = np.arange(101)


def lifetime_curve(posterior, start_age=15):
    # N(a): new people met from start_age up to and including age a, from the POLYMOD posterior means
    band_mean = posterior["band_mean"].mean(("chain", "draw")).values
    week = 5 / 7 + 2 / 7 * np.exp(float(posterior["beta_weekend"].mean()))
    per_year = 365 * week * band_mean[age_band(AGES)]
    per_year[AGES < start_age] = 0
    return np.cumsum(per_year)


def _likelihood(df, curve, q):
    n = curve[df["age"].to_numpy()]
    # 1 − (1 − q) ^ N(a), written so a tiny q does not round to zero
    pm.Bernoulli("ever_partnered", p=-pt.expm1(n * pt.log1p(-q)), observed=df["ever_partnered"].to_numpy().astype(int))


def build_match_model(df, curve):
    with pm.Model() as model:
        log10_q = pm.Normal("log10_q", -4, 2)  # wide: 1 in 100 million to near certain
        _likelihood(df, curve, pm.Deterministic("q", 10**log10_q))
    return model


def build_soulmate_model(df, curve, pool_size, recog_bounds=(0.1, 0.6), recog_mass=0.9):
    with pm.Model() as model:
        a, b = recog_prior(recog_bounds, recog_mass)
        p_recog = pm.Beta("p_recog", alpha=a, beta=b)
        _likelihood(df, curve, pm.Deterministic("q", p_recog / pool_size))  # one soul mate in the pool
    return model
