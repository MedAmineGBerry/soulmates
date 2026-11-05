"""Download the WPP 2024 single age file, check its MD5 and keep the rows for the soul mate pool."""
import hashlib
import pathlib
import urllib.request

import pandas as pd

NAME = "WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv.gz"
URL = f"https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/{NAME}"
MD5 = "f5e699a447a166783588c8919f18cc8b"  # recorded on 2026-11-05, the UN publishes no checksum
COUNTRIES = ["BEL", "DEU", "FIN", "GBR", "ITA", "LUX", "NLD", "POL"]  # the 8 POLYMOD countries
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def main():
    raw = DATA / "raw" / NAME
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        urllib.request.urlretrieve(URL, raw)
    if hashlib.md5(raw.read_bytes()).hexdigest() != MD5:
        raise SystemExit(f"{NAME}: checksum does not match the one recorded on 2026-11-05. "
                         f"The UN may have replaced the file. Delete {raw} and run again, or check the new file.")
    cols = ["Location", "ISO3_code", "Time", "AgeGrpStart", "PopTotal"]
    wpp = pd.read_csv(raw, encoding="utf-8-sig", usecols=cols, low_memory=False)  # the file starts with a BOM
    keep = wpp[(wpp["Location"] == "World") | wpp["ISO3_code"].isin(COUNTRIES)]
    pool = pd.DataFrame({
        "location": keep["ISO3_code"].fillna("World"),
        "year": keep["Time"],
        "age": keep["AgeGrpStart"],
        "pop_thousands": keep["PopTotal"],
    })
    pool.to_csv(DATA / "wpp_pool.csv", index=False)
    print(f"wrote {len(pool)} rows to {DATA / 'wpp_pool.csv'}")
    for label, locs in [("World", ["World"]), ("8 POLYMOD countries", COUNTRIES)]:
        rows = pool[pool["location"].isin(locs) & (pool["year"] == 2023) & pool["age"].between(25, 35)]
        print(f"{label}, aged 25 to 35 in 2023: {rows['pop_thousands'].sum() * 1000:,.0f} people")


if __name__ == "__main__":
    main()
