import numpy as np
import pytest

from soulmates.match import partner_chance
from soulmates.roulette import spin

# Made up inputs: 300 new people per year from age 15, a world pool of 1e9 and a European pool of 4e7 at
# every age, and one country that is twice as talkative as the typical one
CURVE = np.maximum(np.arange(101) - 14, 0) * 300.0
POOLS = {"world": {age: 1e9 for age in range(18, 61)}, "europe": {age: 4e7 for age in range(18, 61)}}
ARGS = dict(age=30, pool="world", recog_bounds=(0.3, 0.9), n_lives=1000, seed=1)


def inputs_with(q, k):
    return {"curve": CURVE, "q": np.full(4000, q), "k": np.full(4000, k), "countries": {"XX": 2.0}, "pools": POOLS}


def test_nobody_finds_anyone_with_q_zero():
    lives = spin(inputs_with(0.0, 3.0), **ARGS)
    assert not lives["found_partner"].any()
    assert lives["age_met"].isna().all()


def test_everyone_who_meets_someone_finds_a_partner_with_q_one():
    lives = spin(inputs_with(1.0, 3.0), **ARGS)
    met = lives["people_met"] >= 1
    assert lives.loc[met, "found_partner"].all()
    assert not lives.loc[~met, "found_partner"].any()
    assert lives.loc[met, "age_met"].between(15, 30).all()


def test_share_found_matches_the_formula():
    # At age 30 the average life has met 16 × 300 = 4800 people. With q = 1 in 4800 and k = 2, the formula
    # gives the share below. With 1000 lives, chance alone moves the share by about 0.016.
    q, k = 1 / 4800, 2.0
    lives = spin(inputs_with(q, k), **ARGS)
    assert lives["found_partner"].mean() == pytest.approx(partner_chance(4800, q, k), abs=0.06)
    assert lives.loc[lives["found_partner"], "age_met"].between(15, 30).all()


def test_a_country_factor_of_two_doubles_the_people_met():
    # k = 1e9 makes every life meet almost exactly the average: 4800 people, or 9600 in country XX
    typical = spin(inputs_with(0.0, 1e9), **ARGS)
    talkative = spin(inputs_with(0.0, 1e9), country="XX", **ARGS)
    assert typical["people_met"].median() == pytest.approx(4800, rel=0.01)
    assert talkative["people_met"].median() == pytest.approx(9600, rel=0.01)


def test_rejects_age_pool_and_country_outside_the_inputs():
    with pytest.raises(ValueError, match="age"):
        spin(inputs_with(0.0, 3.0), **{**ARGS, "age": 17})
    with pytest.raises(ValueError, match="pool"):
        spin(inputs_with(0.0, 3.0), **{**ARGS, "pool": "mars"})
    with pytest.raises(ValueError, match="country"):
        spin(inputs_with(0.0, 3.0), country="ZZ", **ARGS)
