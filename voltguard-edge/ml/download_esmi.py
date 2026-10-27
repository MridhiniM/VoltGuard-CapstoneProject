"""Download the ESMI minute-wise voltage data (Prayas Energy Group, CC0) and keep only Bengaluru sites.

Source: Harvard Dataverse doi:10.7910/DVN/CLLZZM. Files are gated by a guestbook (name, email,
institution, position, intended use); supply them via data/raw/esmi/gb.json (not committed) or env:
    python download_esmi.py            # writes data/raw/esmi/bengaluru_<file>.csv (rows unchanged, other cities dropped)
"""
import csv, io, json, os, sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "raw" / "esmi"
API = "https://dataverse.harvard.edu/api/access/datafile/{}?format=original"
LOCATIONS_ID = 3360848
FILES = {3360846: "2014", 3360855: "2015", 3360857: "2016a", 3360856: "2016b", 3360888: "2017a",
         3362275: "2017b", 3362278: "2017c", 3362468: "2018a", 3362279: "2018b", 3362276: "2018c"}


def guestbook():
    p = OUT / "gb.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    need = ["ESMI_NAME", "ESMI_EMAIL", "ESMI_INSTITUTION", "ESMI_POSITION", "ESMI_PURPOSE"]
    if not all(os.environ.get(k) for k in need):
        sys.exit("Provide data/raw/esmi/gb.json or env " + ", ".join(need))
    e = os.environ
    return {"guestbookResponse": {"name": e["ESMI_NAME"], "email": e["ESMI_EMAIL"], "institution": e["ESMI_INSTITUTION"],
                                  "position": e["ESMI_POSITION"], "answers": [{"id": 182, "value": e["ESMI_PURPOSE"]}]}}


def signed_url(file_id, gb):
    r = requests.post(API.format(file_id), json=gb, timeout=60)
    r.raise_for_status()
    return r.json()["data"]["signedUrl"]


def bengaluru_sites(gb):
    p = OUT / "locations.csv"
    if not p.exists():
        p.write_bytes(requests.get(signed_url(LOCATIONS_ID, gb), timeout=120).content)
    rows = list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8-sig"))))
    return {r["Location name"]: r for r in rows if r["District"].strip() == "Bengaluru Urban"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gb = guestbook()
    sites = bengaluru_sites(gb)
    print(f"{len(sites)} Bengaluru sites")
    for fid, tag in FILES.items():
        dst = OUT / f"bengaluru_{tag}.csv"
        if dst.exists():
            print("have", dst.name); continue
        tmp = dst.with_suffix(".part")
        kept = total = 0
        with requests.get(signed_url(fid, gb), stream=True, timeout=600) as r, open(tmp, "w", encoding="utf-8", newline="") as f:
            r.raise_for_status()
            lines = (l.decode("utf-8", "replace") for l in r.iter_lines())
            header = next(lines)
            f.write(header + "\n")
            for line in lines:
                total += 1
                name = line.split(",", 1)[0].strip('"')
                if name in sites:
                    f.write(line + "\n"); kept += 1
        tmp.rename(dst)
        print(f"{dst.name}: kept {kept} of {total} site-hours", flush=True)


if __name__ == "__main__":
    main()
