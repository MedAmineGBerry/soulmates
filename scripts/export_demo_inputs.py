"""Write data/demo_inputs.json for Soul Mate Roulette: the lifetime curve, the country factors, the q and k
draws and the pool sizes.

It fits the contact model and the partner model with the same seeds as the notebooks, so it needs the
POLYMOD, France and HCMST files in data/raw. The JSON holds model results and no survey rows, and it is
committed to git.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pymc as pm

from soulmates.france import load_diaries
from soulmates.hcmst import load_hcmst
from soulmates.match import build_match_model, lifetime_curve
from soulmates.model import build_model
from soulmates.wpp import POLYMOD_COUNTRIES, pool_size

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def main():
    with build_model(load_diaries(DATA / "raw")):
        contacts = pm.sample(random_seed=2, progressbar=False).posterior
    curve = lifetime_curve(contacts, start_age=15)
    factor = contacts["country_factor"].mean(("chain", "draw"))
    countries = {str(c): float(factor.sel(country=c)) for c in factor["country"].values}
    hcmst = load_hcmst(DATA / "raw" / "HCMST 2017 to 2022 small public version 2.2.dta")
    with build_match_model(hcmst, curve):
        partners = pm.sample(random_seed=2, progressbar=False, target_accept=0.95).posterior
    q, k = partners["q"].values.ravel(), partners["k"].values.ravel()
    wpp = pd.read_csv(DATA / "wpp_pool.csv")
    ages = range(18, 61)
    pools = {
        "world": {age: pool_size(wpp, age=age) for age in ages},
        "europe": {age: pool_size(wpp, age=age, locations=POLYMOD_COUNTRIES) for age in ages},
    }
    path = DATA / "demo_inputs.json"
    path.write_text(json.dumps({"curve": curve.tolist(), "q": q.tolist(), "k": k.tolist(), "countries": countries, "pools": pools}))
    print(f"wrote {path}: {len(q)} draws, q 1 in {1 / np.median(q):,.0f}, k {np.median(k):.2f}, N(30) = {curve[30]:,.0f}, "
          f"N(60) = {curve[60]:,.0f}, countries {({c: round(f, 2) for c, f in countries.items()})}")


if __name__ == "__main__":
    main()
