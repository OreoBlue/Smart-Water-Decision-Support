"""Usage: python train.py path/to/monthly_combined_dataset.csv"""
import sys, os, json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.metrics import accuracy_score, f1_score, recall_score
from core import *

os.makedirs("models", exist_ok=True); os.makedirs("data", exist_ok=True)
pd.read_csv(sys.argv[1], usecols=RAW).to_csv("data/processed.csv.gz", index=False)
d = load_clean("data/processed.csv.gz")
CUT = 2025 * 12 + 1          # test targets: Jan-2025 onward
CAL = 2024 * 12 + 1          # calibration targets: Jan-Dec 2024
mape = lambda y, p: float(np.mean(np.abs(y - p) / y) * 100)
mk = lambda: HGR(max_iter=220, learning_rate=0.08, random_state=0)

# 1) Residential demand forecasting, direct h=1..12
F = forecast_frame(d); lab = F.dropna(subset=["y"])
tr, te = lab[lab.tt < CUT], lab[lab.tt >= CUT]
reg = mk().fit(tr[FEATS_F], tr.y / tr.m12)
b = te.dropna(subset=["ref"])
p = reg.predict(b[FEATS_F]) * b.m12
met = {"demand_forecast": {
    "target": "Residential GPCD (demand), not physical water availability",
    "n_test": len(b), "mape_model_%": mape(b.y, p), "mape_seasonal_naive_%": mape(b.y, b.ref)
}}
for h in (1, 3, 6, 12):
    x = b[b.h == h]; pp = reg.predict(x[FEATS_F]) * x.m12
    met["demand_forecast"][f"h{h}"] = {"model_%": mape(x.y, pp), "naive_%": mape(x.y, x.ref)}

# 2) Prediction interval calibration without touching the 2025+ test set
pre = lab[lab.tt < CAL]
cal = lab[(lab.tt >= CAL) & (lab.tt < CUT)].dropna(subset=["ref"])
cal_model = mk().fit(pre[FEATS_F], pre.y / pre.m12)
cal_pred = cal_model.predict(cal[FEATS_F]) * cal.m12
rel_resid = (cal.y.to_numpy() - cal_pred) / np.maximum(cal_pred, 1e-6)
interval_residuals = {"all": [float(np.quantile(rel_resid, .10)), float(np.quantile(rel_resid, .90))]}
for h in range(1, 13):
    rr = rel_resid[cal.h.to_numpy() == h]
    interval_residuals[str(h)] = [float(np.quantile(rr, .10)), float(np.quantile(rr, .90))]
lo, hi = interval_from_residuals(p, b.h, interval_residuals)
met["demand_forecast"]["interval_80_test_coverage"] = float(((b.y.to_numpy() >= lo) & (b.y.to_numpy() <= hi)).mean())

# 3) Administrative DWR shortage classifier (secondary signal; not a physical availability label)
cm = lambda df, sm: clf_matrix(df.gpcd, df.sid.map(sm), df.ref12, (df.t - 1) % 12 + 1, df.clim, df.hc, df.logpop)
mc = lambda: HGC(max_iter=160, learning_rate=0.08, class_weight="balanced", random_state=0)
L = d.dropna(subset=["y3"])
sm_tr = d[d.t < CUT].groupby("sid").gpcd.mean()
a, c = L[L.t < CUT], L[L.t >= CUT]
clf_eval = mc().fit(cm(a, sm_tr), a.y3)
pc = clf_eval.predict(cm(c, sm_tr))
met["administrative_risk_classification"] = {
    "label": "DWR declared shortage stage grouped to 3 classes",
    "n_test": len(c), "acc": accuracy_score(c.y3, pc),
    "macro_f1": f1_score(c.y3, pc, average="macro"),
    "high_risk_recall": recall_score(c.y3, pc, labels=[2], average="macro", zero_division=0),
    "majority_acc": float((c.y3 == a.y3.mode()[0]).mean()),
    "warning": "Administrative label; use as secondary signal, not proof of physical stock availability."
}

# 4) Label-persistence audit: explains why future-stage accuracy alone is misleading
persist = {}
base = L[["sid", "t", "y3"]].copy()
for h in (1, 3, 6, 12):
    target = base.rename(columns={"t": "tt", "y3": "future"})
    z = base.copy(); z["tt"] = z.t + h; z = z.merge(target, on=["sid", "tt"]).dropna()
    z = z[z.tt >= CUT]
    persist[f"h{h}"] = {"n": int(len(z)), "carry_forward_acc": float((z.y3 == z.future).mean())}
met["label_persistence_audit"] = persist

# 5) Supply-side benchmark and empirical demand/supply pressure thresholds
s = d.dropna(subset=["supply_pcd"]).copy()
# Seasonal-naive supply test: same month last year
s["ref_supply"] = lag(s, 12, "supply_pcd")
s_test = s[(s.t >= CUT) & s.ref_supply.notna()]
met["supply_forecast"] = {
    "method_selected": "seasonal_naive_12m",
    "n_test": int(len(s_test)),
    "mape_%": mape(s_test.supply_pcd, s_test.ref_supply),
    "note": "A HistGradientBoosting supply model was benchmarked separately and was worse on the same holdout; the simpler seasonal method is retained."
}
pressure_train = d[(d.t < CUT) & d.res_pressure.notna()].res_pressure
pressure_thresholds = {
    "low_watch_q60": float(pressure_train.quantile(.60)),
    "watch_high_q85": float(pressure_train.quantile(.85)),
    "definition": "Empirical residential-demand / potable-supply pressure bands; not regulatory thresholds."
}
met["pressure_proxy"] = pressure_thresholds

# 6) Final deployment models
sm = d.groupby("sid").gpcd.mean()
joblib.dump({
    "reg": mk().fit(lab[FEATS_F], lab.y / lab.m12),
    "clf": mc().fit(cm(L, sm), L.y3),
    "sup_mean": sm.to_dict(),
    "interval_residuals": interval_residuals,
    "pressure_thresholds": pressure_thresholds,
}, "models/models.joblib", compress=3)
json.dump(met, open("models/metrics.json", "w"), indent=2)
print(json.dumps(met, indent=2))
