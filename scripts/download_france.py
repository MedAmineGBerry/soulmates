"""Download the four France 2012 files from Zenodo into data/raw/france2012 and check their MD5 checksums."""
import hashlib
import pathlib
import urllib.request

URL = "https://zenodo.org/api/records/3886590/files/{}/content"  # Béraud et al. 2015, version 3, CC BY-NC-ND 4.0
FILES = {
    "2015_Beraud_France_participant_common.csv": "8564fb60b58670c3e302e3aae58537f6",
    "2015_Beraud_France_contact_common.csv": "bf8ea2b9cb95b772868b1ceb5931e8d5",
    "2015_Beraud_France_contact_extra.csv": "1446fb7027517e731af91d9cba954af9",
    "2015_Beraud_France_sday.csv": "1fb1fb75df818a479e5e9e2f0a88635e",
}
RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "france2012"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, md5 in FILES.items():
        path = RAW / name
        if not path.exists():
            urllib.request.urlretrieve(URL.format(name), path)
        if hashlib.md5(path.read_bytes()).hexdigest() != md5:
            raise SystemExit(f"{name}: checksum does not match Zenodo. Delete {path} and run again.")
        print(f"{name}: ok")


if __name__ == "__main__":
    main()
