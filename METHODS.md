# Methods

Technical documentation for the Ionic-Liquid Property Explorer: what the
models are, how they were validated, and how the data pipeline was built.

Live app: https://il-property-explorer.streamlit.app

---

## 1. Problem statement

Predict the temperature-dependent physical properties of an ionic liquid
(viscosity, ionic conductivity, density) and its glass-transition temperature
directly from the molecular structure of its cation and anion, including for
**cation-anion combinations that have never been measured**.

Training data comes from the NIST [ILThermo](https://ilthermo.boulder.nist.gov)
database of experimental measurements.

---

## 2. Machine-learning methods

### Model

All targets use scikit-learn's `HistGradientBoostingRegressor`
(histogram-based gradient boosting, the same family of algorithm as LightGBM):

```python
HistGradientBoostingRegressor(
    loss="quantile", quantile=q, max_iter=500, learning_rate=0.06,
    max_leaf_nodes=63, l2_regularization=1e-3, random_state=0,
)
```

Gradient boosting suits this problem because the data is tabular and modest in
size, the feature interactions are nonlinear, and no feature scaling is needed.

### Quantile regression for uncertainty

Each target is fit **three times**, at quantiles **q = 0.16, 0.50, 0.84**
(12 models total for 4 targets). The median (q50) is the prediction; the
16th-84th percentile spread is a ~1σ interval (for a Gaussian, that interval is
exactly ±1σ). This produces the shaded uncertainty bands in the dashboard and
lets the tool be explicit about how confident it is, rather than projecting a
single misleadingly precise number.

### Targets and transforms

| Target | Modeled quantity | T as feature? |
|---|---|---|
| Viscosity | `ln(eta / Pa*s)` | yes |
| Conductivity | `ln(sigma / S*m^-1)` | yes |
| Density | `rho` in kg/m3 | yes |
| Tg | `Tg` in K (derived, see §3) | no (scalar per compound) |

Viscosity and conductivity are log-transformed because they span orders of
magnitude and are approximately log-normal; training on raw values would let
the most viscous samples dominate the loss. Temperature is an input feature for
the three T-dependent properties, so a single model reproduces the whole curve
instead of requiring one model per temperature.

### Features: 36 per-ion descriptors

RDKit descriptors are computed **separately for the cation and the anion**
(18 each, prefixed `cat_` / `an_`), plus `T_K` where applicable:

- `mw`, `heavy_atoms`, `tpsa`, `labute_asa`, `logp`
- `rot_bonds`, `rings`, `arom_rings`, `frac_csp3`
- `hbd`, `hba`
- atom counts: `n_N`, `n_O`, `n_F`, `n_S`, `n_P`
- `charge` (formal charge), `longest_chain` (longest sp3 carbon path, an alkyl-tail proxy)

Separating cation from anion descriptors is what allows the model to generalize
to **new combinations** of already-known ions.

### Validation: grouped by cation

Validation is 5-fold `GroupKFold` **grouped by cation SMILES**, so entire cation
families are held out of training.

This is the single most important methodological choice in the project. A random
train/test split would be badly optimistic: each compound appears at dozens of
temperatures, so random row splitting leaks the same compound into both train and
test, and the resulting score measures interpolation while appearing to measure
generalization. Grouped CV answers the question the tool actually poses: *how
well does this predict an ion it has never seen?*

### Performance (grouped-CV MAE)

| Target | Samples | Cation groups | MAE |
|---|---|---|---|
| Viscosity | 15,804 | 171 | 0.427 ln-units (~x1.5) |
| Conductivity | 5,736 | 130 | 0.624 ln-units (~x1.9) |
| Density | 24,972 | 193 | 26.8 kg/m3 (~2%) |
| Tg | 418 | 138 | 19.0 K |

Density is excellent. Viscosity is solid: ~1.5x on a property that spans several
decades. Conductivity is the weakest, having the least data and the strongest
sensitivity to ion pairing.

**Sanity check**, [BMIM][Tf2N] at 298 K: predicted eta = 0.050 Pa*s,
sigma = 0.40 S/m, rho = 1436 kg/m3, Tg = 187 K. All match experiment.

Metrics are regenerated into `models/meta.json` on every training run.

---

## 3. Physics-informed step: deriving Tg

ILThermo contains **no glass-transition property**. Tg is derived from the
experimental viscosity curves instead.

For each compound, fit the Vogel-Fulcher-Tammann (VFT) form

```
ln eta = ln eta_0 + B / (T - T_0)
```

then invert the standard glass-transition criterion `eta(Tg) = 1e12 Pa*s`:

```
Tg = T_0 + B / (ln(1e12) - ln eta_0)
```

Fitting rules:

- VFT requires >= 5 points spanning >= 30 K.
- Arrhenius fallback (`ln eta = ln eta_0 + B/T`) for >= 3 points spanning >= 10 K.
- Robust fitting: `scipy.optimize.curve_fit` plus one outlier-rejection pass
  discarding residuals beyond 3 x 1.4826 x MAD.
- Conductivity is fit with the same VFT machinery (opposite sign); density is
  fit linearly in T.

Yield: **970 VFT** viscosity fits (+105 Arrhenius), 471 VFT conductivity fits
(+36 Arrhenius), 1,207 linear density fits (+347 constant).

**Validation of the derivation:** the resulting Tg values have a median of
**198 K** with **91% falling inside the physically expected 150-250 K window**,
which is the main evidence that the derived values are physical rather than
fitting artifacts.

---

## 4. Pipeline

| Stage | Script | Output |
|---|---|---|
| 1. Harvest | `src/harvest.py` | 8,314 raw dataset JSONs |
| 2. Clean | `src/clean.py` | 89,012 points, 1,828 compounds |
| 3. Curve fits | `src/fit_curves.py` | per-compound VFT/linear fits + derived Tg |
| 4. Featurize | `src/featurize.py` | 714 compounds, 211 cations x 116 anions |
| 5. Train | `src/train.py` | 12 quantile models + `meta.json` |
| 6. Dashboard | `app.py` | Streamlit app |

### Stage 1: Harvest

NIST ILThermo REST API:

- `ilprpls` lists property groups and their codes.
- `ilsearch?cmp=&ncmp=1&prp=<code>` searches datasets; `ncmp=1` restricts to
  pure compounds. Codes: **PusA** viscosity, **Ylwl** conductivity,
  **JkYu** density.
- `ilset?set=<setid>` returns a full dataset with metadata.

8,314 unique datasets, throttled to ~3 requests/second, identified by a
descriptive user agent, fully resumable (one cached JSON per setid, already-
downloaded sets are skipped).

> **Gotcha:** ILThermo setids are case-sensitive but Windows filenames are not,
> which silently collided three datasets. Cache filenames now encode the
> uppercase pattern as a bitmask suffix.

### Stage 2: Clean

Raw JSON to a tidy table (`data/clean/points.parquet`):

- Unit conversions: molar volume and specific volume converted to density.
- Phase filter keeps `Liquid` **and** `Metastable liquid`; supercooled points are
  deliberately retained because they are the most informative near Tg.
- Frequency-dependent (AC) conductivity datasets are skipped.
- HTML markup stripped from formulas.
- Per-point experimental uncertainties preserved.
- **Water content** extracted by regex from sample purity metadata (recoverable
  for ~75% of points). Water is the classic confounder in IL viscosity data.

### Stage 3: Curve fits

See §3. Filters applied: ambient pressure only (80-120 kPa, or pressure not
reported) and samples with <= 2 mass % water. Kinematic viscosity is converted to
dynamic viscosity using each compound's own fitted density.

### Stage 4: Featurize (the hardest stage)

ILThermo provides **no SMILES**, only compound names. The first approach, looking
up whole compound names on PubChem, resolved only **17%** of compounds.

The fix was to resolve at the **ion level**. Each name is split into cation and
anion, and each ion is resolved in this order:

1. **Curated anion dictionary** (~60 anions) with aggressive name normalization:
   bracket variants, `amide`/`imide` spelling differences, stereo prefixes,
   spacing variants, and at least one ILThermo typo (`trifluromethyl`).
2. **Rule-based SMILES generation** for parametric cation families:
   1-alkyl-3-alkylimidazolium, alkylpyridinium (including ring-methylated),
   dialkylpyrrolidinium, dialkylpiperidinium, tetraalkylammonium and
   tetraalkylphosphonium, trialkyl(alkyl)-onium, and protonated amines.
3. **PubChem PUG REST** name lookup, accepted only if the returned structure
   carries the correct charge sign.
4. **Fallback**: whole-compound lookup, split into fragments by formal charge.

Ion-level resolution collapses 1,828 compound names into ~990 unique ion names,
and every ion that resolves benefits *every* compound containing it.

Result: **714 compounds (39% by count) covering 74-81% of the actual data
points**, spanning **211 unique cations and 116 unique anions**.

> **Gotcha:** PubChem renamed its `CanonicalSMILES` response field to
> `ConnectivitySMILES`, which silently returned nulls for every lookup until
> caught. The parser now accepts either.

### Stage 5: Train

See §2. Filters mirror stage 3 (ambient pressure, <= 2 mass % water). Writes 12
`.joblib` models and `models/meta.json` with feature columns and metrics.

### Stage 6: Dashboard

Streamlit app (`app.py`):

- Select cation and anion from the database, or paste custom SMILES.
- Charge-sign validation (cation must be positive, anion negative).
- Predicts eta(T), sigma(T), rho(T) over 263-393 K with ~1sigma bands, plus Tg.
- Overlays the experimental ILThermo points when the exact pair exists in the
  training data.
- Displays a **warning banner for novel cation-anion combinations**, making the
  extrapolation explicit.

---

## 5. Limitations

- **2-D descriptors only.** No conformers, no polarizability, no explicit
  ion-pair geometry.
- **Tg is derived, not measured.** It inherits VFT extrapolation error on top of
  model error.
- **Extrapolation is real.** Predictions for ions chemically unlike anything in
  ILThermo are extrapolations; the uncertainty bands are indicative, not
  guarantees.
- **Coverage gap.** ~20-25% of measured points are unusable because their ion
  names never resolved to structures.
- **Conductivity is the weakest model**, limited by data volume and by physics
  (ion pairing) that 2-D descriptors capture poorly.

---

## 6. Reproducing

```bash
pip install -r requirements.txt
python src/harvest.py       # ~1 h, resumable
python src/clean.py
python src/fit_curves.py
python src/featurize.py     # network calls to PubChem, cached
python src/train.py
streamlit run app.py
```

Raw ILThermo JSON is not committed (regenerable via `harvest.py`); cleaned
tables and trained models are.

Data (c) NIST ILThermo. Cite the database and the original experimental
references in any publication.
