"""Fit per-compound property curves from cleaned ILThermo points.

  viscosity     ln eta = ln eta0 + B/(T - T0)        (VFT; Arrhenius fallback)
  conductivity  ln sig = ln sig0 - B/(T - T0)        (VFT; Arrhenius fallback)
  density       rho = a + b*T                        (linear)

Tg is derived from the viscosity VFT fit as the temperature where
eta = 1e12 Pa*s (the standard glass-transition viscosity criterion).

Only near-ambient-pressure points are used (80-120 kPa, or pressure not
reported). Points from samples with >2 mass % water are excluded. Kinematic
viscosity is converted to dynamic viscosity via the compound's density fit.

Output: data/clean/fits.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"

ETA_GLASS = 1e12          # Pa*s at Tg
MAX_WATER_PPM = 20000     # exclude samples wetter than 2 mass %
MIN_PTS_VFT, MIN_SPAN_VFT = 5, 30.0
MIN_PTS_ARR, MIN_SPAN_ARR = 3, 10.0


def vft(T, ln_p0, B, T0):
    return ln_p0 + B / (T - T0)


def arrhenius(T, ln_p0, B):
    return ln_p0 + B / T


def robust_fit(func, T, y, p0, bounds):
    """curve_fit + one outlier-rejection pass (>3*MAD residuals)."""
    popt, _ = curve_fit(func, T, y, p0=p0, bounds=bounds, maxfev=20000)
    resid = y - func(T, *popt)
    mad = np.median(np.abs(resid - np.median(resid))) or 1e-9
    keep = np.abs(resid - np.median(resid)) < 3 * 1.4826 * mad
    if keep.sum() >= max(3, int(0.5 * len(T))) and not keep.all():
        popt, _ = curve_fit(func, T[keep], y[keep], p0=popt, bounds=bounds, maxfev=20000)
        resid = y[keep] - func(T[keep], *popt)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return popt, rmse


def fit_log_property(T, y, sign):
    """Fit ln(property) vs T. sign=+1: eta-like (grows on cooling);
    sign=-1: sigma-like (falls on cooling). Returns dict or None."""
    span = T.max() - T.min()
    n = len(T)
    if n >= MIN_PTS_VFT and span >= MIN_SPAN_VFT:
        try:
            p0 = [y.min() if sign > 0 else y.max(), sign * 800.0, max(T.min() - 80.0, 30.0)]
            lo = [-40, 100 if sign > 0 else -6000, 20]
            hi = [40, 6000 if sign > 0 else -100, T.min() - 5.0]
            popt, rmse = robust_fit(vft, T, y, p0, (lo, hi))
            return {"fit_type": "vft", "p1": popt[0], "p2": popt[1], "p3": popt[2], "rmse_log": rmse}
        except Exception:
            pass
    if n >= MIN_PTS_ARR and span >= MIN_SPAN_ARR:
        try:
            popt, rmse = robust_fit(arrhenius, T, y, [0.0, sign * 3000.0],
                                    ([-60, -30000], [60, 30000]))
            return {"fit_type": "arrhenius", "p1": popt[0], "p2": popt[1], "p3": np.nan, "rmse_log": rmse}
        except Exception:
            pass
    return None


def tg_from_vft(ln_eta0, B, T0):
    denom = np.log(ETA_GLASS) - ln_eta0
    if denom <= 0:
        return np.nan
    return T0 + B / denom


def main() -> None:
    df = pd.read_parquet(CLEAN / "points.parquet")

    ambient = df["P_kPa"].isna() | df["P_kPa"].between(80, 120)
    dry = df["water_ppm"].isna() | (df["water_ppm"] <= MAX_WATER_PPM)
    df = df[ambient & dry].copy()
    print(f"points after ambient-pressure + dryness filters: {len(df)}")

    fits = []

    # --- density first (needed for kinematic-viscosity conversion) ---
    rho_params = {}
    for cid, g in df[df["prop"] == "density"].groupby("compound_id"):
        T, v = g["T_K"].to_numpy(), g["value"].to_numpy()
        if len(T) >= 3 and T.max() - T.min() >= 10:
            b, a = np.polyfit(T, v, 1)
            pred = a + b * T
            rmse = float(np.sqrt(np.mean((v - pred) ** 2)))
            fit = {"fit_type": "linear", "p1": a, "p2": b, "p3": np.nan, "rmse_log": rmse}
        else:
            fit = {"fit_type": "const", "p1": float(v.mean()), "p2": 0.0, "p3": np.nan,
                   "rmse_log": float(v.std())}
        rho_params[cid] = (fit["p1"], fit["p2"])
        fits.append({"compound_id": cid, "prop": "density", **fit,
                     "n_points": len(T), "n_refs": g["ref"].nunique(),
                     "T_min": float(T.min()), "T_max": float(T.max()), "Tg_K": np.nan})

    # --- convert kinematic viscosity where a density fit exists ---
    kin = df[df["prop"] == "kinematic_viscosity"].copy()
    conv = kin["compound_id"].map(lambda c: rho_params.get(c))
    ok = conv.notna()
    if ok.any():
        rho = np.array([a + b * t for (a, b), t in zip(conv[ok], kin.loc[ok, "T_K"])])
        kin.loc[ok, "value"] = kin.loc[ok, "value"] * rho
        kin.loc[ok, "prop"] = "viscosity"
        df = pd.concat([df[df["prop"] != "kinematic_viscosity"], kin[ok]], ignore_index=True)
        print(f"kinematic viscosity converted for {int(ok.sum())} points "
              f"({int((~ok).sum())} dropped, no density fit)")

    # --- viscosity and conductivity VFT fits ---
    for prop, sign in (("viscosity", +1), ("conductivity", -1)):
        for cid, g in df[df["prop"] == prop].groupby("compound_id"):
            T, v = g["T_K"].to_numpy(float), g["value"].to_numpy(float)
            fit = fit_log_property(T, np.log(v), sign)
            if fit is None:
                continue
            tg = np.nan
            if prop == "viscosity" and fit["fit_type"] == "vft":
                tg = tg_from_vft(fit["p1"], fit["p2"], fit["p3"])
            fits.append({"compound_id": cid, "prop": prop, **fit,
                         "n_points": len(T), "n_refs": g["ref"].nunique(),
                         "T_min": float(T.min()), "T_max": float(T.max()), "Tg_K": tg})

    out = pd.DataFrame(fits)
    out.to_parquet(CLEAN / "fits.parquet", index=False)

    print("\nfits per property / type:")
    print(out.groupby(["prop", "fit_type"]).size().to_string())
    print("\nmedian rmse (log-space for vft/arrhenius, kg/m3 for density):")
    print(out.groupby("prop")["rmse_log"].median().round(4).to_string())
    tg = out.loc[out["Tg_K"].notna(), "Tg_K"]
    print(f"\nderived Tg: n={len(tg)}, median={tg.median():.0f} K, "
          f"IQR=[{tg.quantile(0.25):.0f}, {tg.quantile(0.75):.0f}] K")
    sane = tg.between(150, 250).mean()
    print(f"fraction of Tg in the physically typical 150-250 K window: {sane:.0%}")


if __name__ == "__main__":
    main()
