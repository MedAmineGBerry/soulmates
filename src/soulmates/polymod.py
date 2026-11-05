"""Load the POLYMOD contact diaries into one row per participant."""
import pandas as pd

PREFIX = "2008_Mossong_POLYMOD_"


def load_polymod(raw_dir, max_drop=0.05):
    def read(name):
        return pd.read_csv(f"{raw_dir}/{PREFIX}{name}.csv", low_memory=False)

    part = read("participant_common").merge(read("sday")[["part_id", "dayofweek"]], on="part_id", how="left")
    cont = read("contact_common")
    no_age, no_day = part["part_age"].isna(), part["dayofweek"].isna()
    rules = [("no part_age", no_age), ("no dayofweek", no_day), ("no frequency_multi", cont["frequency_multi"].isna())]
    for rule, mask in rules:
        print(f"{rule}: {mask.sum()} of {len(mask)} rows ({mask.mean():.1%})")
        if mask.mean() > max_drop:
            raise ValueError(f"{rule} removes {mask.mean():.1%} of rows, more than max_drop = {max_drop:.1%}")
    part = part[~(no_age | no_day)]
    # A missing frequency_multi is never equal to 5, so those contacts do not count as new
    new = cont[cont["frequency_multi"] == 5].groupby("part_id").size()
    return pd.DataFrame({
        "age": part["part_age"].astype(int).to_numpy(),
        "new_contacts": part["part_id"].map(new).fillna(0).astype(int).to_numpy(),
        "weekend": part["dayofweek"].isin([0, 6]).to_numpy(),  # 0 is Sunday, 6 is Saturday
    })
