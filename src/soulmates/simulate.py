"""Simulated POLYMOD style contact diaries with known true values."""
import numpy as np
import pandas as pd

from soulmates.model import age_band


def simulate_diaries(n_people, band_means, alpha, seed):
    rng = np.random.default_rng(seed)
    age = rng.integers(0, 85, size=n_people)
    mu = np.asarray(band_means)[age_band(age)]
    # numpy uses n and p for the negative binomial: n = alpha and p = alpha / (alpha + mu) give mean mu
    new_contacts = rng.negative_binomial(alpha, alpha / (alpha + mu))
    return pd.DataFrame({"age": age, "new_contacts": new_contacts})
