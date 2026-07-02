"""Parse raw ILThermo JSON into tidy tables.

Outputs:
  data/clean/points.parquet     one row per measured point
  data/clean/compounds.parquet  one row per unique compound

Rules:
  - keep only Liquid and Metastable liquid (supercooled) phases
  - skip frequency-dependent (AC) conductivity sets
  - convert molar volume / specific volume to density
  - kinematic viscosity kept as its own property (converted at fit stage)
  - best-effort water-content extraction from sample purity metadata
"""

import json
import re
from pathlib import Path

import pandas as pd

from harvest import PROPS, cache_name

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SEARCH = ROOT / "data" / "search"
CLEAN = ROOT / "data" / "clean"

TAG_RE = re.compile(r"<[^>]+>")

# header text -> (canonical property, transform on (value, mw))
PROP_HEADERS = {
    "Viscosity, Pa&#8226;s": ("viscosity", None),
    "Kinematic viscosity, m<SUP>2</SUP>/s": ("kinematic_viscosity", None),
    "Electrical conductivity, S/m": ("conductivity", None),
    "Specific density, kg/m<SUP>3</SUP>": ("density", None),
    "Molar volume, m<SUP>3</SUP>/mol": ("density", lambda v, mw: (mw / 1000.0) / v),
    "Specific volume, m<SUP>3</SUP>/kg": ("density", lambda v, mw: 1.0 / v),
}
KEEP_PHASES = {"Liquid", "Metastable liquid"}

WATER_PATTERNS = [
    (re.compile(r"([\d.]+(?:[eE][+-]?\d+)?)\s*water\s*mass\s*%", re.I), 1e4),   # mass % -> ppm
    (re.compile(r"water[^0-9%]*([\d.]+(?:[eE][+-]?\d+)?)\s*mass\s*%", re.I), 1e4),
    (re.compile(r"([\d.]+(?:[eE][+-]?\d+)?)\s*ppm[^a-zA-Z]{0,10}water", re.I), 1.0),
    (re.compile(r"water[^0-9%]*([\d.]+(?:[eE][+-]?\d+)?)\s*ppm", re.I), 1.0),
]


def extract_water_ppm(sample_rows) -> float | None:
    if not sample_rows:
        return None
    text = " ".join(" ".join(map(str, row)) for row in sample_rows)
    for pat, factor in WATER_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1)) * factor
            except ValueError:
                continue
    return None


def fnum(cell) -> tuple[float | None, float | None]:
    """A data cell is [value] or [value, uncertainty] as strings."""
    if not cell or cell[0] in ("", None):
        return None, None
    try:
        v = float(cell[0])
    except ValueError:
        return None, None
    u = None
    if len(cell) > 1 and cell[1] not in ("", None):
        try:
            u = float(cell[1])
        except ValueError:
            pass
    return v, u


def main() -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)

    # setid -> (short ref) from search results
    refs = {}
    for prop in PROPS:
        for row in json.loads((SEARCH / f"{prop}.json").read_text(encoding="utf-8"))["res"]:
            refs[row[0]] = row[1]

    points = []
    compounds = {}
    skipped_ac = skipped_phase = skipped_header = 0

    for setid, ref in refs.items():
        d = json.loads((RAW / cache_name(setid)).read_text(encoding="utf-8"))
        heads = [h[0] for h in d["dhead"]]
        phases = [h[1] if len(h) > 1 else None for h in d["dhead"]]

        if any("Frequency" in h for h in heads):
            skipped_ac += 1
            continue

        comp = d["components"][0]
        cid = comp["idout"]
        mw = float(comp["mw"])
        if cid not in compounds:
            compounds[cid] = {
                "compound_id": cid,
                "name": comp["name"],
                "formula": TAG_RE.sub("", comp.get("formula", "")),
                "mw": mw,
            }

        t_i = heads.index("Temperature, K") if "Temperature, K" in heads else None
        p_i = heads.index("Pressure, kPa") if "Pressure, kPa" in heads else None
        prop_i = prop_name = transform = None
        for i, h in enumerate(heads):
            if h in PROP_HEADERS:
                prop_i = i
                prop_name, transform = PROP_HEADERS[h]
                break
        if t_i is None or prop_i is None:
            skipped_header += 1
            continue

        phase = phases[prop_i]
        if phase not in KEEP_PHASES:
            skipped_phase += 1
            continue

        water_ppm = extract_water_ppm(comp.get("sample"))
        expmeth = d.get("expmeth") or None

        for row in d["data"]:
            t, _ = fnum(row[t_i])
            v, u = fnum(row[prop_i])
            if t is None or v is None or v <= 0:
                continue
            p = None
            if p_i is not None:
                p, _ = fnum(row[p_i])
            if transform:
                v = transform(v, mw)
                u = None  # uncertainty not propagated through conversion
            points.append(
                (setid, prop_name, cid, t, p, v, u,
                 phase == "Metastable liquid", water_ppm, ref, expmeth)
            )

    df = pd.DataFrame(
        points,
        columns=["setid", "prop", "compound_id", "T_K", "P_kPa", "value",
                 "uncertainty", "supercooled", "water_ppm", "ref", "expmeth"],
    )
    cdf = pd.DataFrame(compounds.values())
    per_prop = df.groupby(["prop", "compound_id"]).size().reset_index(name="n")
    for prop in df["prop"].unique():
        ids = set(per_prop.loc[per_prop["prop"] == prop, "compound_id"])
        cdf[f"n_{prop}"] = cdf["compound_id"].map(
            df[df["prop"] == prop].groupby("compound_id").size()).fillna(0).astype(int)

    df.to_parquet(CLEAN / "points.parquet", index=False)
    cdf.to_parquet(CLEAN / "compounds.parquet", index=False)

    print(f"points: {len(df)}  compounds: {len(cdf)}")
    print(f"skipped sets — AC conductivity: {skipped_ac}, non-liquid phase: {skipped_phase}, "
          f"unrecognized headers: {skipped_header}")
    print("\npoints per property:")
    print(df["prop"].value_counts().to_string())
    print("\nwater content known for "
          f"{df['water_ppm'].notna().mean():.0%} of points")
    print("\ntemperature span per compound (viscosity, K):")
    span = df[df["prop"] == "viscosity"].groupby("compound_id")["T_K"].agg(lambda s: s.max() - s.min())
    print(span.describe().round(1).to_string())


if __name__ == "__main__":
    main()
