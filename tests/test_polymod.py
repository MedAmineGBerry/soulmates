import pandas as pd
import pytest

from soulmates.polymod import load_polymod

# Made up rows. Participant 3 has no age and participant 4 has no day of the week, so both are dropped.
PARTICIPANTS = "part_id,part_age\n1,30\n2,25\n3,\n4,40\n5,50\n6,60\n7,70\n"
SDAY = "part_id,dayofweek\n1,0\n2,3\n3,2\n4,\n5,6\n6,1\n7,5\n"
# Participant 1 has two first time contacts, one daily contact and one contact with no frequency.
# Participant 2 has a contact met less often and one with no frequency. Participants 5 to 7 have none.
CONTACTS = "part_id,cont_id,frequency_multi\n1,1,5\n1,2,5\n1,3,1\n1,4,\n2,5,4\n2,6,\n"


@pytest.fixture
def raw_dir(tmp_path):
    for name, text in [("participant_common", PARTICIPANTS), ("sday", SDAY), ("contact_common", CONTACTS)]:
        (tmp_path / f"2008_Mossong_POLYMOD_{name}.csv").write_text(text)
    return tmp_path


def test_load_polymod(raw_dir):
    df = load_polymod(raw_dir, max_drop=1.0)
    expected = pd.DataFrame({
        "age": [30, 25, 50, 60, 70],
        "new_contacts": [2, 0, 0, 0, 0],
        "weekend": [True, False, True, False, False],
    })
    pd.testing.assert_frame_equal(df, expected)


def test_load_polymod_raises_above_max_drop(raw_dir):
    with pytest.raises(ValueError, match="no part_age"):
        load_polymod(raw_dir)
