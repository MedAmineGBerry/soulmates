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
        "k": np.asarray(raw["k"], dtype=float),
        "countries": {name: float(factor) for name, factor in raw["countries"].items()},
        "pools": {pool: {int(age): float(m) for age, m in raw["pools"][pool].items()} for pool in POOLS},
    }


def spin(inputs, age, pool, recog_bounds, n_lives, seed, country="typical", recog_mass=0.9):
    if not 18 <= age <= 60:
        raise ValueError(f"age must be between 18 and 60, got {age}")
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
    if country != "typical" and country not in inputs["countries"]:
        raise ValueError(f"country must be 'typical' or one of {sorted(inputs['countries'])}, got {country!r}")
    rng = np.random.default_rng(seed)
    curve = inputs["curve"]
    draw = rng.integers(len(inputs["q"]), size=n_lives)  # q and k come as pairs from the same posterior draw
    q, k = inputs["q"][draw], inputs["k"][draw]
    a, b = recog_prior(recog_bounds, recog_mass)
    p_recog = rng.beta(a, b, size=n_lives)
    # People differ: each life gets its own multiplier with mean 1, and the chosen country scales everyone
    rate = rng.gamma(k, 1 / k) * (1.0 if country == "typical" else inputs["countries"][country])
    n = np.floor(rate * curve[age])  # people of your own age this life has met and talked to by `age`

    # One uniform draw decides both whether a partner was found and at which meeting: inverting the
    # geometric distribution at u gives the first match, and u < 1 − (1 − q) ^ n keeps it within n meetings
    u = rng.random(n_lives)
    age_met = pd.array([pd.NA] * n_lives, dtype="Int64")
    with np.errstate(divide="ignore", invalid="ignore"):  # q = 1 gives log1p(−1) = −inf, which still yields the right answer
        found_partner = (n >= 1) & (u < -np.expm1(n * np.log1p(-q)))
        first = np.ceil(np.log1p(-u[found_partner]) / np.log1p(-q[found_partner]))
    first = np.maximum(first, 1)  # q = 1 gives 0 here, and the first meeting is number 1
    # the first age at which this life's own count, rate × N(a), reaches that meeting
    age_met[found_partner] = np.searchsorted(curve, first / rate[found_partner])

    m = inputs["pools"][pool][age]
    found_soul_mate = rng.random(n_lives) < -np.expm1(n * np.log1p(-p_recog / m))
    return pd.DataFrame({
        "q": q,
        "k": k,
        "p_recog": p_recog,
        "people_met": n.astype(int),
        "found_partner": found_partner,
        "age_met": age_met,
        "found_soul_mate": found_soul_mate,
    })
