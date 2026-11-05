import arviz as az
import numpy as np
import pymc as pm
import pytest

from soulmates.model import build_model, p_find
from soulmates.simulate import simulate_diaries

# Made up true values for the test, one mean per 5 year band, peaking at ages 20 to 24
TRUE_BAND_MEANS = [1.0, 1.5, 2.0, 3.0, 4.0, 3.5, 3.0, 2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0]
TRUE_ALPHA = 2.0
# Ages 18 to 60 by hand: 18 to 19 is 2 × 3.0, then 5 years each of 4.0, 3.5, 3.0, 2.5, 2.5, 2.0, 2.0, 1.5,
# then age 60 is 1 × 1.5. That is 6 + 20 + 17.5 + 15 + 12.5 + 12.5 + 10 + 10 + 7.5 + 1.5 = 112.5, times 365 days.
TRUE_LIFETIME_NEW = 41062.5


def test_recovers_true_values():
    df = simulate_diaries(3000, TRUE_BAND_MEANS, TRUE_ALPHA, seed=1)
    with build_model(df):
        post = pm.sample(random_seed=2, progressbar=False).posterior
    # Limits from a trial over six seeds, where the largest errors were 14%, 4.5% and 26%
    assert float(post["alpha"].mean()) == pytest.approx(TRUE_ALPHA, rel=0.25)
    assert float(post["lifetime_new"].mean()) == pytest.approx(TRUE_LIFETIME_NEW, rel=0.10)
    assert post["band_mean"].mean(("chain", "draw")).values == pytest.approx(TRUE_BAND_MEANS, rel=0.40)


def test_p_find_by_hand():
    # 2 eye contacts, sure recognition, pool of 4: 1 − (3/4)² = 7/16
    idata = az.from_dict({"posterior": {"eye_contacts": np.array([[2.0]]), "p_recog": np.array([[1.0]])}})
    assert p_find(idata, pool_size=4).item() == pytest.approx(7 / 16)
