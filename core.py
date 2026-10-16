import numpy as np
import pandas as pd

CLASSES = ["Rendah (tahap 0-1)", "Sedang (tahap 2)", "Tinggi (tahap 3-5)"]
FEATS_F = ["h", "mo_sin", "mo_cos", "ref_r", "last_r", "m3_r", "m12", "clim", "hc", "logpop"]
FEATS_C = ["gpcd", "dev", "yoy", "mo_sin", "mo_cos", "clim", "hc", "logpop"]

RENAME = {
    "WATER_SYSTEM_ID": "sid", "SUPPLIER_NAME": "name", "COUNTY": "county",
    "HYDRO_REGION": "hydro", "CLIMATE_ZONE": "clim",
    "REPORT_PERIOD_START_DATE": "date", "POP_REPORT_PERIOD": "pop",
    "DWR_STANDARD_LEVEL": "level", "R-GPCD": "gpcd",
    "POTABLE_SUPPLY_GAL": "supply", "POTABLE_DEMAND_RES_GAL": "res_demand",
}
RAW = list(RENAME)


def seas(m):
    m = np.asarray(m, float)
    return np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)


def lag(d, k, col="gpcd"):
    """Exact calendar lag k months for the same supplier."""
    x = d[["sid", "t", col]].copy()
    x["t"] += k
    return d[["sid", "t"]].merge(x, on=["sid", "t"], how="left")[col].to_numpy()


def load_clean(path):
    d = pd.read_csv(path, usecols=RAW, parse_dates=["REPORT_PERIOD_START_DATE"]).rename(columns=RENAME)
    d = d[d.gpcd.between(10, 1000)].copy()

    lv = d.level.fillna("")
    ok = lv.str.match(r"^[0-5] \(") & ~lv.str.contains(",")
    d["stage"] = pd.to_numeric(lv.str[0].where(ok), errors="coerce")
    d["y3"] = pd.cut(d.stage, [-1, 1, 2, 5], labels=False)

    d["t"] = d.date.dt.year * 12 + d.date.dt.month
    d["hc"] = d.hydro.map({h: i for i, h in enumerate(sorted(d.hydro.dropna().unique()))})
    d["clim"] = pd.to_numeric(d.clim, errors="coerce")
    d["logpop"] = np.log(d["pop"].where(d["pop"] > 0))
    d = d.sort_values(["sid", "t"]).reset_index(drop=True)
    d["ref12"] = lag(d, 12)

    # Supply-side proxy: potable supply normalized to population and calendar days.
    days = d.date.dt.days_in_month
    d["supply_pcd"] = d.supply / (d["pop"] * days)
    d.loc[~d.supply_pcd.between(20, 1000), "supply_pcd"] = np.nan

    # Residential demand / potable supply. This is a pressure proxy, not a regulatory water-stress metric.
    d["res_pressure"] = d.gpcd / d.supply_pcd
    d.loc[~d.res_pressure.between(0, 2.0), "res_pressure"] = np.nan
    return d


def forecast_frame(d):
    """Direct multi-horizon demand frame for h=1..12."""
    L = np.column_stack([lag(d, k) for k in range(12)])
    b = d[["sid", "t", "clim", "hc", "logpop"]].copy()
    b["last"], b["m3"], b["m12"] = L[:, 0], np.nanmean(L[:, :3], axis=1), np.nanmean(L, axis=1)
    tg = d[["sid", "t", "gpcd"]].rename(columns={"t": "tt", "gpcd": "y"})
    out = []
    for h in range(1, 13):
        f = b.copy()
        f["h"], f["tt"], f["ref"] = h, f.t + h, L[:, 12 - h]
        out.append(f.merge(tg, on=["sid", "tt"], how="left"))
    F = pd.concat(out, ignore_index=True)
    F["mo_sin"], F["mo_cos"] = seas((F.tt - 1) % 12 + 1)
    for c in ("ref", "last", "m3"):
        F[c + "_r"] = F[c] / F.m12
    return F


def clf_matrix(g, supm, ref, mo, clim, hc, logpop):
    g = np.asarray(g, float)
    n = len(g)
    B = lambda x: np.broadcast_to(np.asarray(x, float), (n,))
    s, c = seas(mo)
    return pd.DataFrame({
        "gpcd": g, "dev": g / B(supm), "yoy": g / B(ref),
        "mo_sin": s, "mo_cos": c, "clim": B(clim), "hc": B(hc), "logpop": B(logpop),
    })[FEATS_C]


def supply_forecast_for_supplier(hist, future_t):
    """
    Seasonal-naive supply forecast (SPCD): use exact same month last year when available;
    otherwise supplier month-of-year median; finally supplier overall median.
    ML was benchmarked but seasonal-naive was better on 2025+ holdout.
    """
    hist = hist.dropna(subset=["supply_pcd"]).copy()
    if hist.empty:
        return np.repeat(np.nan, len(future_t))
    exact = hist.set_index("t")["supply_pcd"]
    month_med = hist.assign(mo=(hist.t - 1) % 12 + 1).groupby("mo").supply_pcd.median()
    fallback = float(hist.supply_pcd.median())
    vals = []
    for tt in np.asarray(future_t, int):
        v = exact.get(tt - 12, np.nan)
        if pd.isna(v):
            v = month_med.get((tt - 1) % 12 + 1, np.nan)
        if pd.isna(v):
            v = fallback
        vals.append(float(v))
    return np.asarray(vals)


def empirical_pressure_band(x, low_thr, high_thr):
    x = np.asarray(x, float)
    out = np.full(x.shape, "n/a", dtype=object)
    ok = np.isfinite(x)
    out[ok & (x < low_thr)] = "Low"
    out[ok & (x >= low_thr) & (x < high_thr)] = "Watch"
    out[ok & (x >= high_thr)] = "High"
    return out


def interval_from_residuals(pred, h, interval_residuals):
    """Apply horizon-specific relative residual calibration from a pre-test calibration year."""
    pred = np.asarray(pred, float)
    h = np.asarray(h, int)
    lo, hi = np.empty_like(pred), np.empty_like(pred)
    for i, (p, hh) in enumerate(zip(pred, h)):
        q = interval_residuals.get(str(int(hh)), interval_residuals.get("all", [-0.2, 0.2]))
        lo[i], hi[i] = p * (1 + q[0]), p * (1 + q[1])
        if lo[i] > hi[i]:
            lo[i], hi[i] = hi[i], lo[i]
    return np.maximum(lo, 0), np.maximum(hi, 0)


# ---- Data-center scenario ----
def wet_bulb(t_c, rh):
    return (t_c * np.arctan(0.151977 * (rh + 8.313659) ** 0.5) + np.arctan(t_c + rh)
            - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def wue_onsite(tw_c, mode="Fixed approach", lam=1.0):
    f = tw_c * 9 / 5 + 32
    w = (-0.0001896 * f ** 2 + 0.03095 * f + 0.4442) if mode == "Fixed approach" else (0.0005112 * f ** 2 - 0.04982 * f + 2.387)
    return lam * w


def dc_gal_per_day(mw, util, wue):
    return mw * 1000 * util * 24 * wue / 3.78541
