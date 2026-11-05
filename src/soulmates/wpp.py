"""Pool size M from the UN World Population Prospects: everyone alive whose age is close to yours."""
POLYMOD_COUNTRIES = ("BEL", "DEU", "FIN", "GBR", "ITA", "LUX", "NLD", "POL")


def pool_size(wpp, age=30, year=2023, window=5, locations=("World",)):
    rows = wpp[
        wpp["location"].isin(locations)
        & (wpp["year"] == year)
        & wpp["age"].between(age - window, age + window)
    ]
    expected = len(locations) * (2 * window + 1)
    if len(rows) != expected:
        # a typo in a location or a year outside 1950 to 2023 would otherwise give M = 0
        raise ValueError(f"found {len(rows)} rows for {locations} in {year}, expected {expected}")
    return rows["pop_thousands"].sum() * 1000
