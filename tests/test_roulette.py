import numpy as np
import pytest

from soulmates.roulette import spin

# Made up inputs: 300 new people per year from age 15, a world pool of 1e9 and a European pool of 4e7 at every age
CURVE = np.maximum(np.arange(101) - 14, 0) * 300.0
POOLS = {"world": {age: 1e9 for age in range(18, 61)}, "europe": {age: 4e7 for age in range(18, 61)}}


def inputs_with_q(q):
    return {"curve": CURVE, "q": np.full(4000, q), "pools": POOLS}


def test_nobody_finds_anyone_with_q_zero():
    lives = spin(inputs_with_q(0.0), age=30, pool="world", recog_bounds=(0.1, 0.6), n_lives=1000, seed=1)
    assert not lives["found_partner"].any()
    assert lives["age_met"].isna().all()


def test_everyone_finds_someone_at_15_with_q_one():
    # the first meeting already matches, and the first new person is met at age 15
    lives = spin(inputs_with_q(1.0), age=30, pool="world", recog_bounds=(0.1, 0.6), n_lives=1000, seed=1)
    assert lives["found_partner"].all()
    assert (lives["age_met"] == 15).all()


def test_share_found_matches_the_formula():
    # At age 30 a life has met 16 × 300 = 4800 people. This q makes 1 − (1 − q) ^ 4800 exactly 0.5.
    # With 1000 lives, chance alone moves the share by about 0.016.
    q = 1 - 0.5 ** (1 / 4800)
    lives = spin(inputs_with_q(q), age=30, pool="world", recog_bounds=(0.1, 0.6), n_lives=1000, seed=1)
    assert lives["found_partner"].mean() == pytest.approx(0.5, abs=0.06)
    assert lives.loc[lives["found_partner"], "age_met"].between(15, 30).all()


def test_rejects_age_and_pool_outside_the_inputs():
    with pytest.raises(ValueError, match="age"):
        spin(inputs_with_q(0.0), age=17, pool="world", recog_bounds=(0.1, 0.6), n_lives=1, seed=1)
    with pytest.raises(ValueError, match="pool"):
        spin(inputs_with_q(0.0), age=30, pool="mars", recog_bounds=(0.1, 0.6), n_lives=1, seed=1)
