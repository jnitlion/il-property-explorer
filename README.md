# Ionic-Liquid Property Explorer

Machine-learning dashboard for predicting the physical properties of ionic
liquids including temperature-dependent viscosity η(T), electrical/ionic conductivity σ(T), and density ρ(T), as well
as a VFT-derived glass-transition temperature Tg for new/unique cation–anion combinations.
Trained on experimental data from the NIST
[ILThermo](https://ilthermo.boulder.nist.gov) database.

Companion project to [jnitlion.github.io](https://github.com/jnitlion) —
see the exploration notes in `jes-website/ilthermo-exploration.md` for the
data survey and full pipeline design.

## Pipeline

| Stage | Script | Status |
|---|---|---|
| 1. Harvest raw data from ILThermo (8,314 sets) | `src/harvest.py` | ✅ done |
| 2. Clean → 89k points, 1,828 compounds | `src/clean.py` | ✅ done |
| 3. Curve fits: 970 VFT + derived Tg (median 198 K) | `src/fit_curves.py` | ✅ done |
| 4. Ion SMILES (rules + dictionary + PubChem) + RDKit | `src/featurize.py` | ✅ done (74–81% point coverage) |
| 5. Quantile models, cation-grouped CV | `src/train.py` | ✅ done |
| 6. Streamlit dashboard | `app.py` | ✅ done (`run-app.bat` to launch) |

Grouped-CV MAE (entire cation families held out): viscosity 0.43 ln-units,
conductivity 0.62 ln-units, density 27 kg/m³, Tg 19 K. Sanity check on
[BMIM][Tf2N] @ 298 K: η 0.050 Pa·s, σ 0.40 S/m, ρ 1436 kg/m³, Tg 187 K —
all matching experiment.

## Setup

```bash
pip install -r requirements.txt
python src/harvest.py          # ~1 h, resumable; cached in data/raw/
```

## Data notes

- Only pure (single-component) ionic liquids are harvested (`ncmp=1`).
- ILThermo has no glass-transition property; Tg is derived from the VFT fit
  of η(T): η = η₀·exp[B/(T−T₀)], with Tg defined by η(Tg) = 10¹² Pa·s.
- Model validation uses grouped splits (entire cations/anions held out) to
  measure real generalization to unseen ions.
- Raw data © NIST ILThermo; cite the database and original experimental
  references in any publication.
