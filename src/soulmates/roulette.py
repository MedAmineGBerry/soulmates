"""Soul Mate Roulette: play out simulated lives under soul mate rules and under real world rules."""
import json

import numpy as np
import pandas as pd

from soulmates.model import recog_prior

POOLS = ("world", "europe")


def load_inputs(path):
    with open(path) as f:
        raw = json.load(f)
    return {
        "curve": np.asarray(raw["curve"], dtype=float),
        "q": np.asarray(raw["q"], dtype=float),
        "pools": {pool: {int(age): float(m) for age, m in raw["pools"][pool].items()} for pool in POOLS},
    }


def spin(inputs, age, pool, recog_bounds, n_lives, seed, recog_mass=0.9):
    if not 18 <= age <= 60:
        raise ValueError(f"age must be between 18 and 60, got {age}")
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
    rng = np.random.default_rng(seed)
    curve = inputs["curve"]
    n = curve[age]  # every life meets exactly N(age) new people, as the HCMST fit assumed
    q = rng.choice(inputs["q"], size=n_lives)
    a, b = recog_prior(recog_bounds, recog_mass)
    p_recog = rng.beta(a, b, size=n_lives)

    # One uniform draw decides both whether a partner was found and at which meeting: inverting the
    # geometric distribution at u gives the first match, and u < 1 − (1 − q) ^ N keeps it within N meetings
    u = rng.random(n_lives)
    age_met = pd.array([pd.NA] * n_lives, dtype="Int64")
    with np.errstate(divide="ignore"):  # q = 1 gives log1p(−1) = −inf, which still yields the right answer
        found_partner = u < -np.expm1(n * np.log1p(-q))
        first = np.ceil(np.log1p(-u[found_partner]) / np.log1p(-q[found_partner]))
    first = np.maximum(first, 1)  # q = 1 gives 0 here, and the first meeting is number 1
    age_met[found_partner] = np.searchsorted(curve, first)  # first age a with N(a) >= that meeting

    m = inputs["pools"][pool][age]
    found_soul_mate = rng.random(n_lives) < -np.expm1(n * np.log1p(-p_recog / m))
    return pd.DataFrame({
        "q": q,
        "p_recog": p_recog,
        "found_partner": found_partner,
        "age_met": age_met,
        "found_soul_mate": found_soul_mate,
    })
