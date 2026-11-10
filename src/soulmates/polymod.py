"""Load the POLYMOD contact diaries into one row per participant."""
import pandas as pd

PREFIX = "2008_Mossong_POLYMOD_"


def contact_age(cont):
    # the exact age where it is given, else the middle of the estimated range
    return cont["cnt_age_exact"].fillna((cont["cnt_age_est_min"] + cont["cnt_age_est_max"]) / 2)


def load_polymod(raw_dir, window=5, max_drop=0.05):
    def read(name):
        return pd.read_csv(f"{raw_dir}/{PREFIX}{name}.csv", low_memory=False)

    part = (
        read("participant_common")
        .merge(read("sday")[["part_id", "dayofweek"]], on="part_id", how="left")
        .merge(read("hh_common")[["hh_id", "country"]], on="hh_id", how="left")
    )
    cont = read("contact_common").merge(part[["part_id", "part_age"]], on="part_id")
    no_age, no_day = part["part_age"].isna(), part["dayofweek"].isna()
    rules = [("no part_age", no_age), ("no dayofweek", no_day), ("no frequency_multi", cont["frequency_multi"].isna())]
    for rule, mask in rules:
        print(f"{rule}: {mask.sum()} of {len(mask)} rows ({mask.mean():.1%})")
        if mask.mean() > max_drop:
            raise ValueError(f"{rule} removes {mask.mean():.1%} of rows, more than max_drop = {max_drop:.1%}")
    part = part[~(no_age | no_day)]
    # New means "first time" (code 5) and within `window` years of the participant's age, because the
    # soul mate pool only holds people of that age. A missing frequency or age compares as False.
    in_window = (contact_age(cont) - cont["part_age"]).abs() <= window
    new = cont[(cont["frequency_multi"] == 5) & in_window].groupby("part_id").size()
    return pd.DataFrame({
        "age": part["part_age"].astype(int).to_numpy(),
        "new_contacts": part["part_id"].map(new).fillna(0).astype(int).to_numpy(),
        "weekend": part["dayofweek"].isin([0, 6]).to_numpy(),  # 0 is Sunday, 6 is Saturday
        "country": part["country"].to_numpy(),
    })
