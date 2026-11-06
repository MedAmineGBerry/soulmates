import pandas as pd

from soulmates.hcmst import load_hcmst


def test_load_hcmst(tmp_path):
    # Made up rows: ages 17 and 61 fall outside 18 to 60, status 4 means "never had a partner"
    raw = pd.DataFrame({
        "w1_ppage": [17, 18, 30, 60, 61],
        "w1_partnership_status": [1, 4, 1, 3, 2],
        "w1_weight_combo": [1.0, 0.5, 1.5, 2.0, 1.0],
        "other_column": [0, 0, 0, 0, 0],
    })
    path = tmp_path / "hcmst.dta"
    raw.to_stata(path, write_index=False)
    expected = pd.DataFrame({
        "age": [18, 30, 60],
        "ever_partnered": [False, True, True],
        "weight": [0.5, 1.5, 2.0],
    })
    pd.testing.assert_frame_equal(load_hcmst(path), expected)
