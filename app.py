"""Ionic-Liquid Property Explorer — Streamlit dashboard.

Pick (or paste SMILES for) a cation and an anion; the models predict
viscosity, ionic conductivity, and density as functions of temperature,
plus a derived glass-transition temperature — with uncertainty bands.

Run:  streamlit run app.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from featurize import ion_features  # noqa: E402

CLEAN = ROOT / "data" / "clean"
MODELS = ROOT / "models"

st.set_page_config(page_title="Ionic-Liquid Property Explorer", page_icon="🧪", layout="wide")


@st.cache_resource
def load_models():
    meta = json.loads((MODELS / "meta.json").read_text(encoding="utf-8"))
    models = {}
    for target in meta["targets"]:
        models[target] = {q: joblib.load(MODELS / f"{target}_{q}.joblib")
                          for q in ("q16", "q50", "q84")}
    return meta, models


@st.cache_data
def load_ions():
    feats = pd.read_parquet(CLEAN / "features.parquet")
    # display label for each ion SMILES = most common name fragment
    cat = (feats.assign(label=feats["name"].str.split().str[0])
           .groupby("cation_smiles")["label"]
           .agg(lambda s: s.mode().iloc[0]).reset_index())
    an = (feats.assign(label=feats["name"].str.split(n=1).str[1])
          .groupby("anion_smiles")["label"]
          .agg(lambda s: s.mode().iloc[0]).reset_index())
    return feats, cat.sort_values("label"), an.sort_values("label")


@st.cache_data
def load_experimental():
    pts = pd.read_parquet(CLEAN / "points.parquet")
    ambient = pts["P_kPa"].isna() | pts["P_kPa"].between(80, 120)
    return pts[ambient]


def predict_curve(models, X_base: dict, feat_cols, T):
    rows = pd.DataFrame([{**X_base, "T_K": t} for t in T])[feat_cols + ["T_K"]]
    qs = np.sort(np.column_stack(
        [models[q].predict(rows) for q in ("q16", "q50", "q84")]), axis=1)
    return qs[:, 0], qs[:, 1], qs[:, 2]


def band_plot(T, lo, mid, hi, exp_T=None, exp_v=None, *, ylabel, logy, title):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.fill_between(T, lo, hi, alpha=0.25, color="#1e5c58", lw=0)
    ax.plot(T, mid, color="#1e5c58", lw=2, label="prediction")
    if exp_T is not None and len(exp_T):
        ax.scatter(exp_T, exp_v, s=14, color="#b45309", zorder=3,
                   label="ILThermo data")
        ax.legend(fontsize=8)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


meta, models = load_models()
feats, cations, anions = load_ions()
feat_cols = meta["feature_columns"]

st.title("🧪 Ionic-Liquid Property Explorer")
st.caption(
    "Machine-learning predictions trained on the NIST "
    "[ILThermo](https://ilthermo.boulder.nist.gov) database "
    f"({meta['targets']['viscosity']['n_samples']:,} viscosity measurements, "
    f"{len(feats):,} ionic liquids). Bands are ~1σ prediction intervals from "
    "quantile models; validation holds out entire cation families."
)

left, right = st.columns(2)
with left:
    st.subheader("Cation")
    cat_mode = st.radio("cation input", ["Pick from database", "Custom SMILES"],
                        horizontal=True, label_visibility="collapsed")
    if cat_mode == "Pick from database":
        row = st.selectbox("Cation", cations["label"] + "  ·  " + cations["cation_smiles"],
                           index=None, placeholder="e.g. 1-butyl-3-methylimidazolium")
        cat_smiles = row.rsplit("·", 1)[1].strip() if row else None
    else:
        cat_smiles = st.text_input("Cation SMILES (must carry a + charge)",
                                   placeholder="CCCCn1cc[n+](C)c1") or None
with right:
    st.subheader("Anion")
    an_mode = st.radio("anion input", ["Pick from database", "Custom SMILES"],
                       horizontal=True, label_visibility="collapsed")
    if an_mode == "Pick from database":
        row = st.selectbox("Anion", anions["label"] + "  ·  " + anions["anion_smiles"],
                           index=None, placeholder="e.g. bis(trifluoromethylsulfonyl)imide")
        an_smiles = row.rsplit("·", 1)[1].strip() if row else None
    else:
        an_smiles = st.text_input("Anion SMILES (must carry a − charge)",
                                  placeholder="[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F") or None

if not (cat_smiles and an_smiles):
    st.info("Choose a cation and an anion to see predictions.")
    st.stop()

fc = ion_features(cat_smiles, "cat")
fa = ion_features(an_smiles, "an")
if fc is None or fa is None:
    st.error("Could not parse one of the SMILES strings. Check the syntax "
             "(cation needs a positive, anion a negative formal charge).")
    st.stop()
if fc["cat_charge"] <= 0 or fa["an_charge"] >= 0:
    st.error("Charge check failed: the cation must be positive and the anion negative.")
    st.stop()

X_base = {**fc, **fa}

# is this exact pair in the training data?
known = feats[(feats["cation_smiles"] == cat_smiles) & (feats["anion_smiles"] == an_smiles)]
exp = load_experimental()
exp_pair = exp[exp["compound_id"].isin(known["compound_id"])] if len(known) else exp.iloc[0:0]

if len(known):
    st.success(f"This ionic liquid is in ILThermo as **{known.iloc[0]['name']}** — "
               "experimental points are overlaid; the model has seen this compound.")
else:
    st.warning("New cation–anion combination (not in the training data) — "
               "these are genuine extrapolative predictions; mind the bands.")

# Tg
tg_lo, tg_mid, tg_hi = (float(models["tg"][q].predict(
    pd.DataFrame([X_base])[feat_cols])[0]) for q in ("q16", "q50", "q84"))
tg_lo, tg_mid, tg_hi = sorted((tg_lo, tg_mid, tg_hi))
st.metric("Predicted glass-transition temperature (VFT, η = 10¹² Pa·s criterion)",
          f"{tg_mid:.0f} K  ({tg_mid - 273.15:.0f} °C)",
          delta=f"±{(tg_hi - tg_lo) / 2:.0f} K", delta_color="off")

T = np.linspace(263, 393, 66)
c1, c2, c3 = st.columns(3)
specs = [
    ("viscosity", "Viscosity (Pa·s)", True, np.exp, c1),
    ("conductivity", "Electrical conductivity (S/m)", True, np.exp, c2),
    ("density", "Density (kg/m³)", False, lambda x: x, c3),
]
for target, ylabel, logy, inv, col in specs:
    lo, mid, hi = predict_curve(models[target], X_base, feat_cols, T)
    e = exp_pair[exp_pair["prop"] == target]
    with col:
        st.pyplot(band_plot(T, inv(lo), inv(mid), inv(hi),
                            e["T_K"], e["value"],
                            ylabel=ylabel, logy=logy, title=ylabel.split(" (")[0]))

with st.expander("Model details & honest caveats"):
    st.markdown(f"""
- **Training data:** pure ionic liquids from NIST ILThermo (ambient pressure,
  samples ≤2 mass % water). Grouped cross-validation MAE (entire cation
  families held out): viscosity **{meta['targets']['viscosity']['grouped_cv_mae']:.2f}**
  ln-units, conductivity **{meta['targets']['conductivity']['grouped_cv_mae']:.2f}**
  ln-units, density **{meta['targets']['density']['grouped_cv_mae']:.0f}** kg/m³,
  Tg **{meta['targets']['tg']['grouped_cv_mae']:.0f}** K.
- **Tg** is not measured directly: it is derived from VFT fits of experimental
  viscosity curves (η(Tg) = 10¹² Pa·s), so it inherits VFT-extrapolation error.
- Predictions for ions **chemically unlike anything in ILThermo** are
  extrapolations; the uncertainty bands are indicative, not guarantees.
- Features are 2-D RDKit descriptors per ion — no conformers, no
  polarizability, no ion-pair geometry. Data © NIST; cite ILThermo and the
  original experimental papers.
""")
