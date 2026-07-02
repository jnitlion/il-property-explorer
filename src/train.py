"""Train property-prediction models on featurized ILThermo data.

Models (HistGradientBoosting):
  viscosity     features + T -> ln(eta / Pa*s)
  conductivity  features + T -> ln(sigma / S*m-1)
  density       features + T -> rho / kg*m-3
  tg            features     -> Tg / K            (from VFT-derived values)

Each target gets three quantile models (q16 / q50 / q84) so the dashboard can
show a ~1-sigma predictive band. Validation is GROUPED by cation SMILES —
entire cation families are held out — which measures real generalization to
unseen ions (random splits would leak).

Outputs: models/<target>_q{16,50,84}.joblib, models/meta.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
MODELS = ROOT / "models"

QUANTILES = {"q16": 0.16, "q50": 0.50, "q84": 0.84}
MAX_WATER_PPM = 20000


def make_model(q: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=500, learning_rate=0.06,
        max_leaf_nodes=63, l2_regularization=1e-3, random_state=0,
    )


def grouped_cv_mae(X, y, groups, n_splits=5) -> float:
    maes = []
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        m = make_model(0.5).fit(X.iloc[tr], y.iloc[tr])
        maes.append(float(np.mean(np.abs(m.predict(X.iloc[te]) - y.iloc[te]))))
    return float(np.mean(maes))


def train_target(name, X, y, groups, meta):
    mae = grouped_cv_mae(X, y, groups)
    meta["targets"][name] = {
        "n_samples": int(len(y)),
        "n_cation_groups": int(groups.nunique()),
        "grouped_cv_mae": round(mae, 4),
    }
    for tag, q in QUANTILES.items():
        m = make_model(q).fit(X, y)
        joblib.dump(m, MODELS / f"{name}_{tag}.joblib")
    print(f"{name:14s} n={len(y):6d}  grouped-CV MAE = {mae:.3f}")


def main() -> None:
    MODELS.mkdir(exist_ok=True)

    feats = pd.read_parquet(CLEAN / "features.parquet")
    points = pd.read_parquet(CLEAN / "points.parquet")
    fits = pd.read_parquet(CLEAN / "fits.parquet")

    feat_cols = [c for c in feats.columns if c.startswith(("cat_", "an_"))]
    meta = {"feature_columns": feat_cols, "targets": {}}

    ambient = points["P_kPa"].isna() | points["P_kPa"].between(80, 120)
    dry = points["water_ppm"].isna() | (points["water_ppm"] <= MAX_WATER_PPM)
    pts = points[ambient & dry].merge(feats, on="compound_id", how="inner")

    for prop, log in (("viscosity", True), ("conductivity", True), ("density", False)):
        d = pts[pts["prop"] == prop].copy()
        y = np.log(d["value"]) if log else d["value"]
        X = d[feat_cols].copy()
        X["T_K"] = d["T_K"]
        train_target(prop, X.reset_index(drop=True), y.reset_index(drop=True),
                     d["cation_smiles"].reset_index(drop=True), meta)

    tg = (fits[(fits["prop"] == "viscosity") & fits["Tg_K"].between(100, 300)]
          .merge(feats, on="compound_id", how="inner"))
    train_target("tg", tg[feat_cols].reset_index(drop=True),
                 tg["Tg_K"].reset_index(drop=True),
                 tg["cation_smiles"].reset_index(drop=True), meta)

    meta["notes"] = {
        "viscosity": "target is ln(eta/Pa*s); T_K is an input feature",
        "conductivity": "target is ln(sigma/S*m-1); T_K is an input feature",
        "density": "target is rho in kg/m3; T_K is an input feature",
        "tg": "target is Tg in K derived from VFT fits (eta=1e12 Pa*s criterion)",
    }
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nmodels + meta written to", MODELS)


if __name__ == "__main__":
    main()
