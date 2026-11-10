import pandas as pd
import pytest

from soulmates.polymod import load_polymod

# Made up rows. Participant 3 has no age and participant 4 has no day of the week, so both are dropped.
PARTICIPANTS = "part_id,hh_id,part_age\n1,11,30\n2,12,25\n3,13,\n4,14,40\n5,15,50\n6,16,60\n7,17,70\n"
HOUSEHOLDS = "hh_id,country\n11,DE\n12,GB\n13,IT\n14,PL\n15,DE\n16,FI\n17,LU\n"
SDAY = "part_id,dayofweek\n1,0\n2,3\n3,2\n4,\n5,6\n6,1\n7,5\n"
# Participant 1 is 30. Their first time contacts (code 5): one aged 31, one aged 36, one with an estimated
# range of 28 to 34 (middle 31) and one with no age. They also have a daily contact and one with no frequency.
# Participant 2 has a contact met less often and one with no frequency. Participants 5 to 7 have none.
CONTACTS = (
    "part_id,cont_id,cnt_age_exact,cnt_age_est_min,cnt_age_est_max,frequency_multi\n"
    "1,1,31,,,5\n1,2,36,,,5\n1,3,,28,34,5\n1,4,,,,5\n1,5,30,,,1\n1,6,30,,,\n2,7,25,,,4\n2,8,25,,,\n"
)


@pytest.fixture
def raw_dir(tmp_path):
    files = {"participant_common": PARTICIPANTS, "sday": SDAY, "contact_common": CONTACTS, "hh_common": HOUSEHOLDS}
    for name, text in files.items():
        (tmp_path / f"2008_Mossong_POLYMOD_{name}.csv").write_text(text)
    return tmp_path


def test_load_polymod(raw_dir):
    df = load_polymod(raw_dir, max_drop=1.0)
    expected = pd.DataFrame({
        "age": [30, 25, 50, 60, 70],
        "new_contacts": [2, 0, 0, 0, 0],  # the contacts aged 31 and about 31; 36 is outside 5 years
        "weekend": [True, False, True, False, False],
        "country": ["DE", "GB", "DE", "FI", "LU"],
    })
    pd.testing.assert_frame_equal(df, expected)


def test_a_wider_window_counts_the_contact_aged_36(raw_dir):
    assert load_polymod(raw_dir, window=10, max_drop=1.0)["new_contacts"].tolist() == [3, 0, 0, 0, 0]


def test_load_polymod_raises_above_max_drop(raw_dir):
    with pytest.raises(ValueError, match="no part_age"):
        load_polymod(raw_dir)
