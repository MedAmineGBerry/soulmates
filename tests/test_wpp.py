import pandas as pd
import pytest

from soulmates.wpp import pool_size

# Made up rows: 1 to 5 thousand people at ages 28 to 32 in 2023, the same for World and DEU
WPP = pd.DataFrame({
    "location": ["World"] * 5 + ["DEU"] * 5,
    "year": [2023] * 10,
    "age": [28, 29, 30, 31, 32] * 2,
    "pop_thousands": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
})


def test_pool_size_sums_the_window():
    # ages 29 to 31 hold 2 + 3 + 4 = 9 thousand people
    assert pool_size(WPP, age=30, window=1) == 9000


def test_pool_size_adds_locations():
    assert pool_size(WPP, age=30, window=1, locations=("World", "DEU")) == 18000


def test_pool_size_raises_on_missing_rows():
    with pytest.raises(ValueError, match="expected 3"):
        pool_size(WPP, age=30, year=2030, window=1)
