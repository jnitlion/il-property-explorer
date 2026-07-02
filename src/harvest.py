"""Download all pure-ionic-liquid datasets from NIST ILThermo.

Politely throttled (~3 requests/s incl. latency), resumable: already-cached
sets are skipped, so the script can be re-run after interruption or to pick
up new data. Raw JSON is cached one file per dataset in data/raw/.

Usage:  python src/harvest.py [--limit N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

BASE = "https://ilthermo.boulder.nist.gov/ILT2"
PROPS = {"viscosity": "PusA", "conductivity": "Ylwl", "density": "JkYu"}
DELAY = 0.30  # seconds between requests — be polite to NIST

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SEARCH = ROOT / "data" / "search"

session = requests.Session()
session.headers["User-Agent"] = "il-property-explorer (academic project; jesternitliong@gmail.com)"


def cache_name(setid: str) -> str:
    """Case-safe cache filename: setids are case-sensitive but Windows
    filenames are not, so append a bitmask of the uppercase positions."""
    mask = sum(1 << i for i, c in enumerate(setid) if c.isupper())
    return f"{setid}_{mask:02d}.json"


def get_json(url: str, params: dict | None = None, tries: int = 3):
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def load_search_results() -> dict[str, list]:
    """Fetch (or load cached) pure-compound search results per property."""
    results = {}
    for prop, code in PROPS.items():
        cache = SEARCH / f"{prop}.json"
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            print(f"searching {prop} ({code}) ...", flush=True)
            data = get_json(
                f"{BASE}/ilsearch",
                {"cmp": "", "ncmp": 1, "year": "", "auth": "", "keyw": "", "prp": code},
            )
            cache.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(DELAY)
        results[prop] = data["res"]
        print(f"{prop}: {len(data['res'])} datasets", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N downloads (0 = all)")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    SEARCH.mkdir(parents=True, exist_ok=True)

    results = load_search_results()
    setids = sorted({row[0] for rows in results.values() for row in rows})
    todo = [s for s in setids if not (RAW / cache_name(s)).exists()]
    print(f"{len(setids)} unique sets, {len(todo)} to download", flush=True)

    failures: list[str] = []
    t0 = time.time()
    done = 0
    for setid in todo:
        try:
            data = get_json(f"{BASE}/ilset", {"set": setid})
            (RAW / cache_name(setid)).write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:
            failures.append(setid)
            print(f"  FAIL {setid}: {e}", flush=True)
        done += 1
        if done % 100 == 0:
            rate = done / (time.time() - t0)
            eta_min = (len(todo) - done) / rate / 60 if rate else 0
            print(f"  {done}/{len(todo)}  ({rate:.1f}/s, ~{eta_min:.0f} min left)", flush=True)
        if args.limit and done >= args.limit:
            print("limit reached, stopping", flush=True)
            break
        time.sleep(DELAY)

    print(f"finished: {done - len(failures)} downloaded, {len(failures)} failures", flush=True)
    if failures:
        (ROOT / "data" / "failures.txt").write_text("\n".join(failures), encoding="utf-8")
        print("failed setids written to data/failures.txt (re-run to retry)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
