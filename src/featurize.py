"""Resolve ionic-liquid names to per-ion SMILES and build RDKit features.

Resolution strategy (in order), applied to the cation and anion separately
after splitting the compound name ("<cation> <anion...>"):

  1. curated dictionary (common anions; exact normalized-name match)
  2. rule-based SMILES generation for parametric cation families
     (1-alkyl-3-methylimidazolium, N-alkylpyridinium, dialkylpyrrolidinium,
      dialkylpiperidinium, tetraalkylammonium/phosphonium)
  3. PubChem PUG REST name lookup (cached), accepted only if the returned
     structure carries the right charge sign
  4. fallback: whole-compound PubChem lookup (cached) split into fragments

Output: data/clean/features.parquet
"""

import json
import re
import time
import urllib.parse
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
ION_CACHE = CLEAN / "ion_smiles_cache.json"
COMPOUND_CACHE = CLEAN / "smiles_cache.json"

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON"
DELAY = 0.25

session = requests.Session()
session.headers["User-Agent"] = "il-property-explorer (academic project; jesternitliong@gmail.com)"

# ---------------------------------------------------------------- dictionaries

ALKYL = {
    "methyl": 1, "ethyl": 2, "propyl": 3, "butyl": 4, "pentyl": 5, "amyl": 5,
    "hexyl": 6, "heptyl": 7, "octyl": 8, "nonyl": 9, "decyl": 10,
    "undecyl": 11, "dodecyl": 12, "tridecyl": 13, "tetradecyl": 14,
    "pentadecyl": 15, "hexadecyl": 16, "octadecyl": 18,
}

TF2N = "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F"
ANION_SMILES = {
    "bis(trifluoromethylsulfonyl)imide": TF2N,
    "bis(trifluoromethanesulfonyl)imide": TF2N,
    "bis((trifluoromethyl)sulfonyl)imide": TF2N,
    "bis(trifluoromethane)sulfonimide": TF2N,
    "1,1,1-trifluoro-n-((trifluoromethyl)sulfonyl)methanesulfonamide": TF2N,
    "bis(perfluoromethylsulfonyl)imide": TF2N,
    "bis(fluorosulfonyl)imide": "[N-](S(=O)(=O)F)S(=O)(=O)F",
    "bis(pentafluoroethylsulfonyl)imide": "[N-](S(=O)(=O)C(F)(F)C(F)(F)F)S(=O)(=O)C(F)(F)C(F)(F)F",
    "bis(2,4,4-trimethylpentyl)phosphinate": "CC(C)(C)CC(C)CP(=O)([O-])CC(C)CC(C)(C)C",
    "tricyanomethane": "[C-](C#N)(C#N)C#N",
    "tetrafluoroborate": "[B-](F)(F)(F)F",
    "hexafluorophosphate": "F[P-](F)(F)(F)(F)F",
    "tris(pentafluoroethyl)trifluorophosphate": "C(C(F)(F)[P-](C(C(F)(F)F)(F)F)(C(C(F)(F)F)(F)F)(F)(F)F)(F)(F)F",
    "dicyanamide": "[N-](C#N)C#N",
    "tricyanomethanide": "[C-](C#N)(C#N)C#N",
    "tetracyanoborate": "[B-](C#N)(C#N)(C#N)C#N",
    "thiocyanate": "[S-]C#N",
    "trifluoromethanesulfonate": "[O-]S(=O)(=O)C(F)(F)F",
    "triflate": "[O-]S(=O)(=O)C(F)(F)F",
    "methanesulfonate": "CS(=O)(=O)[O-]",
    "acetate": "CC(=O)[O-]",
    "trifluoroacetate": "[O-]C(=O)C(F)(F)F",
    "formate": "[O-]C=O",
    "lactate": "CC(O)C(=O)[O-]",
    "chloride": "[Cl-]",
    "bromide": "[Br-]",
    "iodide": "[I-]",
    "nitrate": "[O-][N+]([O-])=O",
    "perchlorate": "[O-]Cl(=O)(=O)=O",
    "hydrogen sulfate": "OS(=O)(=O)[O-]",
    "hydrogensulfate": "OS(=O)(=O)[O-]",
    "methyl sulfate": "COS(=O)(=O)[O-]",
    "methylsulfate": "COS(=O)(=O)[O-]",
    "ethyl sulfate": "CCOS(=O)(=O)[O-]",
    "ethylsulfate": "CCOS(=O)(=O)[O-]",
    "octyl sulfate": "CCCCCCCCOS(=O)(=O)[O-]",
    "dihydrogen phosphate": "OP(=O)(O)[O-]",
    "diethyl phosphate": "CCOP(=O)([O-])OCC",
    "dimethyl phosphate": "COP(=O)([O-])OC",
    "dibutyl phosphate": "CCCCOP(=O)([O-])OCCCC",
    "p-toluenesulfonate": "Cc1ccc(cc1)S(=O)(=O)[O-]",
    "tosylate": "Cc1ccc(cc1)S(=O)(=O)[O-]",
    "4-methylbenzenesulfonate": "Cc1ccc(cc1)S(=O)(=O)[O-]",
    "salicylate": "OC1=CC=CC=C1C(=O)[O-]",
    "benzoate": "[O-]C(=O)c1ccccc1",
    "propanoate": "CCC(=O)[O-]",
    "butanoate": "CCCC(=O)[O-]",
    "glycinate": "NCC(=O)[O-]",
    "alaninate": "CC(N)C(=O)[O-]",
    "taurate": "NCCS(=O)(=O)[O-]",
    "pentanoate": "CCCCC(=O)[O-]",
    "hexanoate": "CCCCCC(=O)[O-]",
    "2-hydroxypropanoate": "CC(O)C(=O)[O-]",
    "glycolate": "OCC(=O)[O-]",
    "levulinate": "CC(=O)CCC(=O)[O-]",
    "propionate": "CCC(=O)[O-]",
}
# spacing variants: "dimethyl phosphate" == "dimethylphosphate"
ANION_SMILES.update({k.replace(" ", ""): v for k, v in list(ANION_SMILES.items()) if " " in k})


def chain(n: int) -> str:
    return "C" * n


def normalize_ion_name(name: str) -> str:
    """Canonicalize spelling variants: lowercase, brackets->parens,
    amide->imide (Tf2N/FSI naming), strip '1H-' ring-hydrogen markers."""
    n = name.lower().strip()
    n = n.replace("[", "(").replace("]", ")")
    n = n.replace("trifluromethyl", "trifluoromethyl")     # ILThermo typo
    n = n.replace("sulfonyl)amide", "sulfonyl)imide")
    n = n.replace("sulfonylamide", "sulfonylimide")
    n = n.replace("-1h-", "-").replace("1h-", "")
    n = re.sub(r"^\((r|s)\)-", "", n)                      # strip stereo prefixes
    return n


# substituent name -> SMILES prefix whose LAST atom is the attachment point
GROUPS = {name: chain(n) for name, n in ALKYL.items()}
GROUPS.update({
    "2-hydroxyethyl": "OCC",
    "3-hydroxypropyl": "OCCC",
    "2-methoxyethyl": "COCC",
    "methoxymethyl": "COC",
    "2-ethoxyethyl": "CCOCC",
    "allyl": "C=CC",
    "benzyl": "c1ccccc1C",
    "phenyl": "c1ccccc1",
    "cyanomethyl": "N#CC",
})

# odd-one-out cations that don't follow a parametric pattern
CATION_SPECIALS = {
    "2-hydroxy-n-methylethanaminium": "OCC[NH2+]C",
    "2-hydroxyethylammonium": "OCC[NH3+]",
    "ethanolammonium": "OCC[NH3+]",
    "bis(2-hydroxyethyl)ammonium": "OCC[NH2+]CCO",
    "tris(2-hydroxyethyl)ammonium": "OCC[NH+](CCO)CCO",
    "cholinium": "C[N+](C)(C)CCO",
    "choline": "C[N+](C)(C)CCO",
    "guanidinium": "NC(=[NH2+])N",
    "ammonium": "[NH4+]",
    "hydroxylammonium": "O[NH3+]",
    "n-methyl-2-hydroxyethylammonium": "OCC[NH2+]C",
    "diethanolammonium": "OCC[NH2+]CCO",
    "triethanolammonium": "OCC[NH+](CCO)CCO",
}


def sub(token: str) -> str | None:
    """Substituent token (possibly parenthesized) -> attachment SMILES."""
    return GROUPS.get(token.strip("()"))


def _split_two(s: str):
    """Split a concatenated pair of substituent names, e.g. 'ethyloctyl'."""
    for i in range(2, len(s) - 1):
        a, b = s[:i], s[i:]
        if a in GROUPS and b in GROUPS:
            return GROUPS[a], GROUPS[b]
    return None


def _onium(x: str) -> str:
    return "N" if x == "ammonium" else "P"


CATION_RULES = [
    # 1-R-3-R'-imidazolium (and reversed 3-R-1-R' locant order)
    (re.compile(r"^1-(.+?)-3-(.+?)imidazolium$"),
     lambda m: (lambda a, b: f"{a}n1cc[n+]({b})c1" if a and b else None)(
         sub(m.group(1)), sub(m.group(2)))),
    (re.compile(r"^3-(.+?)-1-(.+?)imidazolium$"),
     lambda m: (lambda a, b: f"{a}n1cc[n+]({b})c1" if a and b else None)(
         sub(m.group(1)), sub(m.group(2)))),
    # n-methyl-1-R-pyridinium written methyl-first
    (re.compile(r"^2-methyl-1-(.+?)pyridinium$"),
     lambda m: (lambda a: f"{a}[n+]1c(C)cccc1" if a else None)(sub(m.group(1)))),
    (re.compile(r"^3-methyl-1-(.+?)pyridinium$"),
     lambda m: (lambda a: f"{a}[n+]1cc(C)ccc1" if a else None)(sub(m.group(1)))),
    (re.compile(r"^4-methyl-1-(.+?)pyridinium$"),
     lambda m: (lambda a: f"{a}[n+]1ccc(C)cc1" if a else None)(sub(m.group(1)))),
    # 1-R-2,3-dimethylimidazolium
    (re.compile(r"^1-(.+?)-2,3-dimethylimidazolium$"),
     lambda m: (lambda a: f"{a}n1cc[n+](C)c1C" if a else None)(sub(m.group(1)))),
    # 1-R-n-methylpyridinium (ring-methylated)
    (re.compile(r"^1-(.+?)-2-methylpyridinium$"),
     lambda m: (lambda a: f"{a}[n+]1c(C)cccc1" if a else None)(sub(m.group(1)))),
    (re.compile(r"^1-(.+?)-3-methylpyridinium$"),
     lambda m: (lambda a: f"{a}[n+]1cc(C)ccc1" if a else None)(sub(m.group(1)))),
    (re.compile(r"^1-(.+?)-4-methylpyridinium$"),
     lambda m: (lambda a: f"{a}[n+]1ccc(C)cc1" if a else None)(sub(m.group(1)))),
    # 1-R-pyridinium
    (re.compile(r"^1-(.+?)pyridinium$"),
     lambda m: (lambda a: f"{a}[n+]1ccccc1" if a else None)(sub(m.group(1)))),
    # 1-R-1-R'-pyrrolidinium / piperidinium
    (re.compile(r"^1-(.+?)-1-(.+?)pyrrolidinium$"),
     lambda m: (lambda a, b: f"{a}[N+]1({b})CCCC1" if a and b else None)(
         sub(m.group(1)), sub(m.group(2)))),
    (re.compile(r"^1-(.+?)-1-(.+?)piperidinium$"),
     lambda m: (lambda a, b: f"{a}[N+]1({b})CCCCC1" if a and b else None)(
         sub(m.group(1)), sub(m.group(2)))),
    # tetraalkyl-onium
    (re.compile(r"^tetra(.+?)(ammonium|phosphonium)$"),
     lambda m: (lambda c, x: f"{c}[{x}+]({c})({c}){c}" if c else None)(
         sub(m.group(1)), _onium(m.group(2)))),
    # trialkyl(alkyl)-onium, e.g. trihexyl(tetradecyl)phosphonium
    (re.compile(r"^tri(.+?)\((.+?)\)(ammonium|phosphonium)$"),
     lambda m: (lambda a, b, x: f"{a}[{x}+]({a})({a}){b}" if a and b else None)(
         sub(m.group(1)), sub(m.group(2)), _onium(m.group(3)))),
    # trialkylalkyl-onium without parentheses, e.g. triethyloctylphosphonium
    (re.compile(r"^tri([a-z]+)(ammonium|phosphonium)$"),
     lambda m: (lambda pair, x: f"{pair[0]}[{x}+]({pair[0]})({pair[0]}){pair[1]}"
                if pair else None)(_split_two(m.group(1)), _onium(m.group(2)))),
    # protonated amines: mono/di/tri-alkylammonium (e.g. ethylammonium — EAN!)
    (re.compile(r"^di(.+?)ammonium$"),
     lambda m: (lambda a: f"{a}[NH2+]{a}" if a else None)(sub(m.group(1)))),
    (re.compile(r"^tri(.+?)ammonium$"),
     lambda m: (lambda a: f"{a}[NH+]({a}){a}" if a else None)(sub(m.group(1)))),
    (re.compile(r"^(.+?)ammonium$"),
     lambda m: (lambda a: f"{a}[NH3+]" if a else None)(sub(m.group(1)))),
]

# ---------------------------------------------------------------- resolution


def valid_ion(smiles: str, want_sign: int):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    q = Chem.GetFormalCharge(m)
    if (q > 0) if want_sign > 0 else (q < 0):
        return Chem.MolToSmiles(m)
    return None


def pubchem_lookup(name: str, cache: dict) -> str | None:
    if name in cache:
        return cache[name]
    url = PUG.format(urllib.parse.quote(name, safe=""))
    smiles = None
    for attempt in range(3):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                break
            r.raise_for_status()
            p = r.json()["PropertyTable"]["Properties"][0]
            smiles = p.get("SMILES") or p.get("CanonicalSMILES") or p.get("ConnectivitySMILES")
            break
        except Exception:
            if attempt == 2:
                break
            time.sleep(2 ** (attempt + 1))
    cache[name] = smiles
    time.sleep(DELAY)
    return smiles


def resolve_cation(name: str, cache: dict) -> str | None:
    norm = normalize_ion_name(name)
    s = CATION_SPECIALS.get(norm)
    if s:
        return valid_ion(s, +1)
    for pat, fn in CATION_RULES:
        m = pat.match(norm)
        if m:
            s = fn(m)
            if s:
                ok = valid_ion(s, +1)
                if ok:
                    return ok
    s = pubchem_lookup(name, cache)
    if s and "." not in s:
        return valid_ion(s, +1)
    return None


def resolve_anion(name: str, cache: dict) -> str | None:
    s = ANION_SMILES.get(normalize_ion_name(name))
    if s:
        return valid_ion(s, -1)
    s = pubchem_lookup(name, cache)
    if s and "." not in s:
        return valid_ion(s, -1)
    return None


def split_compound_smiles(smiles: str):
    """Whole-compound SMILES -> (cation, anion) if a clean pair."""
    mols = []
    for f in set(smiles.split(".")):
        m = Chem.MolFromSmiles(f)
        if m is None:
            return None
        mols.append((Chem.MolToSmiles(m), Chem.GetFormalCharge(m)))
    cats = [f for f, q in mols if q > 0]
    ans = [f for f, q in mols if q < 0]
    if len(cats) == 1 and len(ans) == 1:
        return cats[0], ans[0]
    return None

# ---------------------------------------------------------------- descriptors


def _longest_carbon_chain(mol) -> int:
    carbons = {a.GetIdx() for a in mol.GetAtoms()
               if a.GetSymbol() == "C" and not a.GetIsAromatic()}
    best = 0

    def dfs(idx, seen):
        nonlocal best
        best = max(best, len(seen))
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            j = nb.GetIdx()
            if j in carbons and j not in seen:
                dfs(j, seen | {j})

    for c in carbons:
        dfs(c, {c})
    return best


ION_DESCRIPTORS = {
    "mw": Descriptors.MolWt,
    "heavy_atoms": Descriptors.HeavyAtomCount,
    "tpsa": Descriptors.TPSA,
    "labute_asa": Descriptors.LabuteASA,
    "logp": Crippen.MolLogP,
    "rot_bonds": Descriptors.NumRotatableBonds,
    "rings": rdMolDescriptors.CalcNumRings,
    "arom_rings": rdMolDescriptors.CalcNumAromaticRings,
    "frac_csp3": rdMolDescriptors.CalcFractionCSP3,
    "hbd": rdMolDescriptors.CalcNumHBD,
    "hba": rdMolDescriptors.CalcNumHBA,
    "n_N": lambda m: sum(a.GetSymbol() == "N" for a in m.GetAtoms()),
    "n_O": lambda m: sum(a.GetSymbol() == "O" for a in m.GetAtoms()),
    "n_F": lambda m: sum(a.GetSymbol() == "F" for a in m.GetAtoms()),
    "n_S": lambda m: sum(a.GetSymbol() == "S" for a in m.GetAtoms()),
    "n_P": lambda m: sum(a.GetSymbol() == "P" for a in m.GetAtoms()),
    "charge": Chem.GetFormalCharge,
    "longest_chain": _longest_carbon_chain,
}


def ion_features(smiles: str, prefix: str) -> dict | None:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return {f"{prefix}_{k}": float(fn(m)) for k, fn in ION_DESCRIPTORS.items()}

# ---------------------------------------------------------------- main


def main() -> None:
    cdf = pd.read_parquet(CLEAN / "compounds.parquet")
    ion_cache = json.loads(ION_CACHE.read_text(encoding="utf-8")) if ION_CACHE.exists() else {}
    comp_cache = json.loads(COMPOUND_CACHE.read_text(encoding="utf-8")) if COMPOUND_CACHE.exists() else {}

    rows, fail = [], Counter()
    for i, (_, c) in enumerate(cdf.iterrows(), 1):
        name = c["name"]
        parts = name.split(" ", 1)
        cat = an = None
        if len(parts) == 2:
            cat = resolve_cation(parts[0], ion_cache)
            an = resolve_anion(parts[1], ion_cache)
        if not (cat and an):
            whole = comp_cache.get(name)
            if whole:
                pair = split_compound_smiles(whole)
                if pair:
                    cat, an = pair
        if not (cat and an):
            fail["unresolved"] += 1
            continue
        fc, fa = ion_features(cat, "cat"), ion_features(an, "an")
        if fc is None or fa is None:
            fail["rdkit_parse"] += 1
            continue
        rows.append({"compound_id": c["compound_id"], "name": name,
                     "cation_smiles": cat, "anion_smiles": an, **fc, **fa})
        if i % 200 == 0:
            print(f"  {i}/{len(cdf)} processed ({len(rows)} resolved)", flush=True)
            ION_CACHE.write_text(json.dumps(ion_cache), encoding="utf-8")

    ION_CACHE.write_text(json.dumps(ion_cache), encoding="utf-8")
    fdf = pd.DataFrame(rows)
    fdf.to_parquet(CLEAN / "features.parquet", index=False)
    print(f"\nfeaturized: {len(fdf)}/{len(cdf)} compounds ({len(fdf)/len(cdf):.0%})")
    for k, v in fail.items():
        print(f"  dropped — {k}: {v}")
    print(f"unique cations: {fdf['cation_smiles'].nunique()}, "
          f"unique anions: {fdf['anion_smiles'].nunique()}")


if __name__ == "__main__":
    main()
