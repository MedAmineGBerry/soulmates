import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from soulmates.match import build_match_model, lifetime_curve


def test_lifetime_curve_by_hand():
    # Every band mean is 1 new person per day and there is no weekend effect, so a year adds 365
    posterior = az.from_dict(
        {"posterior": {"band_mean": np.ones((1, 1, 17)), "beta_weekend": np.zeros((1, 1))}},
        dims={"band_mean": ["band"]},
    ).posterior
    curve = lifetime_curve(posterior, start_age=15)
    assert curve[14] == 0
    assert curve[17] == pytest.approx(3 * 365)  # ages 15, 16 and 17


def test_recovers_q():
    # Made up: q = 1 in 2000, and 300 new people per year from age 15, so about 9% never find a partner
    true_q = 1 / 2000
    curve = np.maximum(np.arange(101) - 14, 0) * 300.0
    rng = np.random.default_rng(1)
    age = rng.integers(18, 61, size=3000)
    ever = rng.random(3000) < 1 - (1 - true_q) ** curve[age]
    df = pd.DataFrame({"age": age, "ever_partnered": ever})
    with build_match_model(df, curve):
        q = pm.sample(random_seed=2, progressbar=False).posterior["q"].median().item()
    # Limit from a trial over six seeds, where the largest error was 3.3%
    assert q == pytest.approx(true_q, rel=0.10)
