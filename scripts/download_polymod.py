"""Download the four POLYMOD files from Zenodo into data/raw and check their MD5 checksums."""
import hashlib
import pathlib
import urllib.request

URL = "https://zenodo.org/api/records/3874557/files/{}/content"  # POLYMOD version 2, licence CC BY-NC 4.0
FILES = {
    "2008_Mossong_POLYMOD_participant_common.csv": "ac784c281a71e15c7e64eaf2d6365709",
    "2008_Mossong_POLYMOD_contact_common.csv": "52baf32033bf780786fae6604043ec00",
    "2008_Mossong_POLYMOD_sday.csv": "148c5fc86ca10d385e3d047014e14a09",
    "2008_Mossong_POLYMOD_hh_common.csv": "d7fb1359ad84dba8cce4c444063940aa",
}
RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"


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
