"""Write data/demo_inputs.json for Soul Mate Roulette: the lifetime curve, the q draws and the pool sizes.

It fits the POLYMOD model and the free HCMST model with the same seeds as notebook 03, so it needs the
HCMST file in data/raw. The JSON holds model results and no survey rows, and it is committed to git.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pymc as pm

from soulmates.hcmst import load_hcmst
from soulmates.match import build_match_model, lifetime_curve
from soulmates.model import build_model
from soulmates.polymod import load_polymod
from soulmates.wpp import POLYMOD_COUNTRIES, pool_size

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def main():
    with build_model(load_polymod(DATA / "raw")):
        contacts = pm.sample(random_seed=2, progressbar=False)
    curve = lifetime_curve(contacts.posterior, start_age=15)
    hcmst = load_hcmst(DATA / "raw" / "HCMST 2017 to 2022 small public version 2.2.dta")
    with build_match_model(hcmst, curve):
        q = pm.sample(random_seed=2, progressbar=False).posterior["q"].values.ravel()
    wpp = pd.read_csv(DATA / "wpp_pool.csv")
    ages = range(18, 61)
    pools = {
        "world": {age: pool_size(wpp, age=age) for age in ages},
        "europe": {age: pool_size(wpp, age=age, locations=POLYMOD_COUNTRIES) for age in ages},
    }
    path = DATA / "demo_inputs.json"
    path.write_text(json.dumps({"curve": curve.tolist(), "q": q.tolist(), "pools": pools}))
    print(f"wrote {path}: {len(q)} q draws, q 1 in {1 / np.median(q):,.0f}, N(30) = {curve[30]:,.0f}, "
          f"world pool at 30 = {pools['world'][30]:,.0f}")


if __name__ == "__main__":
    main()
