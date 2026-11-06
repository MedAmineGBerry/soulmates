"""Load the 2017 wave of How Couples Meet and Stay Together: age, ever had a partner, survey weight."""
import warnings

import pandas as pd

COLUMNS = ["w1_ppage", "w1_partnership_status", "w1_weight_combo"]


def load_hcmst(path):
    with warnings.catch_warnings():
        # some string columns fall back to latin-1, none of the three columns used here is a string
        warnings.simplefilter("ignore", UnicodeWarning)
        d = pd.read_stata(path, convert_categoricals=False, columns=COLUMNS)
    d = d[d["w1_ppage"].between(18, 60)]
    return pd.DataFrame({
        "age": d["w1_ppage"].astype(int).to_numpy(),
        "ever_partnered": (d["w1_partnership_status"] != 4).to_numpy(),  # 4 is "never had a partner"
        "weight": d["w1_weight_combo"].astype(float).to_numpy(),
    })
