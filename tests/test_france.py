import pandas as pd

from soulmates.france import load_france_days

# Made up rows. Participant 1 is 30 and kept a diary on two days: a Sunday (code 0) and a Monday (code 1).
PARTICIPANTS = "part_id,part_age\n1,30\n2,40\n"
SDAY = "part_id,wave,studyDay,dayofweek\n1,1,1,0\n1,1,2,1\n2,1,1,3\n"
# Day 1 of participant 1: two first time contacts, aged 29 and 50. Day 2: one first time contact aged 33.
CONTACTS = (
    "part_id,cont_id,cnt_age_exact,cnt_age_est_min,cnt_age_est_max,frequency_multi\n"
    "1,101,29,,,5\n1,102,50,,,5\n1,103,33,,,5\n2,104,41,,,1\n"
)
CONTACT_EXTRA = "cont_id,wave,studyDay\n101,1,1\n102,1,1\n103,1,2\n104,1,1\n"


def test_load_france_days(tmp_path):
    files = {"participant_common": PARTICIPANTS, "sday": SDAY, "contact_common": CONTACTS, "contact_extra": CONTACT_EXTRA}
    for name, text in files.items():
        (tmp_path / f"2015_Beraud_France_{name}.csv").write_text(text)
    expected = pd.DataFrame({
        "part_id": [1, 1, 2],
        "wave": [1, 1, 1],
        "day": [1, 2, 1],
        "age": [30, 30, 40],
        "new_contacts": [1, 1, 0],  # the contact aged 50 is outside 5 years
        "weekend": [True, False, False],
        "country": ["FR", "FR", "FR"],
    })
    pd.testing.assert_frame_equal(load_france_days(tmp_path), expected)
