"""Simulated POLYMOD style contact diaries with known true values."""
import numpy as np
import pandas as pd

from soulmates.model import age_band


def simulate_diaries(n_people, band_means, alpha, seed, weekend_effect=1.0, country_factors=(1.0,)):
    rng = np.random.default_rng(seed)
    age = rng.integers(0, 85, size=n_people)
    weekend = rng.random(n_people) < 2 / 7
    country = rng.integers(len(country_factors), size=n_people)
    mu = np.asarray(band_means)[age_band(age)] * np.where(weekend, weekend_effect, 1.0) * np.asarray(country_factors)[country]
    # numpy uses n and p for the negative binomial: n = alpha and p = alpha / (alpha + mu) give mean mu
    new_contacts = rng.negative_binomial(alpha, alpha / (alpha + mu))
    return pd.DataFrame({"age": age, "new_contacts": new_contacts, "weekend": weekend, "country": [f"C{c}" for c in country]})
