"""Load the France 2012 contact diaries (Béraud et al. 2015): one row per participant and diary day."""
import pandas as pd

from soulmates.polymod import contact_age, load_polymod

PREFIX = "2015_Beraud_France_"
COLUMNS = ["age", "new_contacts", "weekend", "country"]


def load_france_days(raw_dir, window=5):
    def read(name):
        return pd.read_csv(f"{raw_dir}/{PREFIX}{name}.csv", low_memory=False)

    part = read("participant_common")[["part_id", "part_age"]]
    days = read("sday").merge(part, on="part_id").dropna(subset=["part_age", "dayofweek"])
    # contact_extra says on which diary day (wave and studyDay) a contact was met
    cont = read("contact_common").merge(read("contact_extra"), on="cont_id").merge(part, on="part_id")
    in_window = (contact_age(cont) - cont["part_age"]).abs() <= window
    new = cont[(cont["frequency_multi"] == 5) & in_window].groupby(["part_id", "wave", "studyDay"]).size()
    days = days.merge(new.rename("new_contacts").reset_index(), on=["part_id", "wave", "studyDay"], how="left")
    return pd.DataFrame({
        "part_id": days["part_id"].to_numpy(),
        "wave": days["wave"].to_numpy(),
        "day": days["studyDay"].to_numpy(),
        "age": days["part_age"].astype(int).to_numpy(),
        "new_contacts": days["new_contacts"].fillna(0).astype(int).to_numpy(),
        "weekend": days["dayofweek"].isin([0, 6]).to_numpy(),  # 0 is Sunday, 6 is Saturday, as in POLYMOD
        "country": "FR",
    })


def load_diaries(raw_dir, window=5):
    # the 8 POLYMOD countries plus France, in the columns the contact model needs
    return pd.concat(
        [load_polymod(raw_dir, window)[COLUMNS], load_france_days(f"{raw_dir}/france2012", window)[COLUMNS]],
        ignore_index=True,
    )
