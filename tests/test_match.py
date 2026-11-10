import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from soulmates.match import build_match_model, lifetime_curve, partner_chance


def test_lifetime_curve_by_hand():
    # Every band mean is 1 new person per day and there is no weekend effect, so a year adds 365
    posterior = az.from_dict(
        {"posterior": {"band_mean": np.ones((1, 1, 17)), "beta_weekend": np.zeros((1, 1))}},
        dims={"band_mean": ["band"]},
    ).posterior
    curve = lifetime_curve(posterior, start_age=15)
    assert curve[14] == 0
    assert curve[17] == pytest.approx(3 * 365)  # ages 15, 16 and 17


def test_partner_chance_by_hand():
    # q = 1 − 1/e makes theta = 1. With n = 1 and k = 1 the chance is 1 − (1 + 1) ^ (−1) = 0.5
    assert partner_chance(1, 1 - np.exp(-1), 1) == pytest.approx(0.5)
    assert partner_chance(4800, 0.0, 2) == 0  # nobody can become a partner
    assert partner_chance(10, 1.0, 2) == 1  # everybody can


def test_recovers_q_and_k():
    # Made up: q = 1 in 400, k = 3, and 150 new people per year from age 15
    true_q, true_k = 1 / 400, 3.0
    curve = np.maximum(np.arange(101) - 14, 0) * 150.0
    rng = np.random.default_rng(1)
    age = rng.integers(18, 61, size=6000)
    lam = rng.gamma(true_k, 1 / true_k, size=6000)  # how many people this person meets, relative to the average
    ever = rng.random(6000) < 1 - (1 - true_q) ** (lam * curve[age])
    with build_match_model(pd.DataFrame({"age": age, "ever_partnered": ever}), curve):
        post = pm.sample(random_seed=2, progressbar=False, target_accept=0.95).posterior
    # Limits from a trial over four seeds, where the largest errors were 6% for q and 10% for k
    assert post["q"].median().item() == pytest.approx(true_q, rel=0.20)
    assert post["k"].median().item() == pytest.approx(true_k, rel=0.35)
