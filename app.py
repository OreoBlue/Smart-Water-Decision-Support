import json, joblib, numpy as np, pandas as pd, streamlit as st, altair as alt
from core import *

st.set_page_config(page_title="Smart Water Decision Support", page_icon="💧", layout="wide")

# ---------- Design tokens (ui-craft · dense-dashboard) ----------
ACC, OK, WARN, BAD = "oklch(55% 0.13 240)", "oklch(60% 0.12 150)", "oklch(72% 0.13 75)", "oklch(58% 0.15 25)"
BAND_COL = {"Low": OK, "Watch": WARN, "High": BAD, "n/a": "gray"}
BAND_ID = {"Low": "Rendah", "Watch": "Waspada", "High": "Tinggi", "n/a": "n/a"}
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown {{ font-family:'IBM Plex Sans',system-ui,sans-serif; font-size:14px; }}
.block-container {{ padding-top:1.25rem; max-width:1280px; }}
#MainMenu, footer {{ visibility:hidden; }}
.crumb {{ font-size:12px; opacity:.65; margin-bottom:4px; }}
.crumb b {{ font-weight:500; opacity:1; }}
h1.t {{ font-size:20px; font-weight:500; margin:0 0 2px 0; padding:0; }}
.sub {{ font-size:12px; opacity:.65; margin-bottom:12px; }}
.sec {{ font-size:14px; font-weight:500; margin:16px 0 8px 0; }}
.mono, .v, td.n {{ font-family:'IBM Plex Mono',monospace; font-variant-numeric:tabular-nums slashed-zero; }}
.grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }}
.m {{ border:1px solid rgba(128,128,128,.28); border-radius:8px; padding:12px 16px; }}
.m .l {{ font-size:12px; opacity:.65; }}
.m .v {{ font-size:22px; font-weight:500; margin:4px 0 0 0; }}
.m .s {{ font-size:12px; opacity:.6; }}
.exec {{ border:1px solid rgba(128,128,128,.28); border-left:2px solid {ACC}; border-radius:8px;
        padding:12px 16px; background:color-mix(in oklch, {ACC} 6%, transparent); margin:12px 0; }}
.exec .h {{ font-size:12px; opacity:.65; text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }}
.chip {{ display:inline-block; border:1px solid rgba(128,128,128,.4); border-radius:4px; padding:0 8px; font-size:12px; margin-right:4px; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }}
.up {{ color:{BAD}; }} .dn {{ color:{OK}; }}
section[data-testid="stSidebar"] .sec {{ font-size:12px; opacity:.7; text-transform:uppercase; letter-spacing:.04em; }}
</style>""", unsafe_allow_html=True)

fmt = lambda x, n=1: f"{x:,.{n}f}".replace(",", "§").replace(".", ",").replace("§", ".")
card = lambda l, v, s="": f'<div class="m"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{s}</div></div>'

@st.cache_data
def load():
    d = load_clean("data/processed.csv.gz")
    return d, forecast_frame(d)

d, F = load(); M = joblib.load("models/models.joblib"); met = json.load(open("models/metrics.json"))
sup = d.drop_duplicates("sid").set_index("sid")

# ---------- Sidebar ----------
sb = st.sidebar
sb.markdown('<div class="sec">Unit layanan</div>', unsafe_allow_html=True)
sid = sb.selectbox("Pemasok air", sup.index, format_func=lambda s: f"{sup['name'][s]} — {sup['county'][s]}", label_visibility="collapsed")
sb.markdown('<div class="sec">Skenario data center</div>', unsafe_allow_html=True)
on = sb.toggle("Aktifkan skenario", True)
mw = sb.slider("Beban IT (MW)", 1, 100, 10, disabled=not on)
util = sb.slider("Utilisasi rata-rata", 0.3, 1.0, 0.65, disabled=not on)
temp = sb.slider("Suhu udara (°C)", 15, 40, 30, disabled=not on)
rh = sb.slider("Kelembapan relatif (%)", 30, 95, 75, disabled=not on)
mode = sb.selectbox("Model pendinginan", ["Fixed approach", "Fixed cold water"], disabled=not on)
lam = sb.slider("Pengali efisiensi pendinginan (λ)", 0.5, 1.5, 1.0, disabled=not on,
                help="Nilai < 1 = lebih hemat air. Parameter skenario, bukan hasil kalibrasi data lokal.")

# ---------- Computation (unchanged logic) ----------
h = d[d.sid == sid].copy(); t0 = int(h.t.max())
f = F[(F.sid == sid) & (F.t == t0)].sort_values("h").reset_index(drop=True)
f["pred"] = M["reg"].predict(f[FEATS_F]) * f.m12
f["lo"], f["hi"] = interval_from_residuals(f.pred, f.h, M["interval_residuals"])
pop, clim, hc = float(np.exp(h.logpop.iloc[-1])), h.clim.iloc[-1], h.hc.iloc[-1]
f["supply_pcd"] = supply_forecast_for_supplier(h, f.tt)

wbt = wet_bulb(temp, rh); wue = wue_onsite(wbt, mode, lam); gal_day = dc_gal_per_day(mw, util, wue)
extra_gpcd = gal_day / pop if on else 0.0
f["scenario_demand"] = f.pred + extra_gpcd
f["pressure_base"] = f.pred / f.supply_pcd
f["pressure_scenario"] = f.scenario_demand / f.supply_pcd
pt = M["pressure_thresholds"]
f["band_base"] = empirical_pressure_band(f.pressure_base, pt["low_watch_q60"], pt["watch_high_q85"])
f["band_scen"] = empirical_pressure_band(f.pressure_scenario, pt["low_watch_q60"], pt["watch_high_q85"])

mo = (f.tt - 1) % 12 + 1; classes = M["clf"].classes_.astype(int)
probs = lambda g: M["clf"].predict_proba(clf_matrix(g, M["sup_mean"][sid], f.ref, mo, clim, hc, np.log(pop)))
p0, p1 = probs(f.pred), probs(f.scenario_demand)
lab = lambda p: [CLASSES[classes[i]] for i in p.argmax(1)]
ih = list(classes).index(2)
dates = pd.to_datetime(dict(year=(f.tt - 1) // 12, month=(f.tt - 1) % 12 + 1, day=1))
f["date"] = dates.values
ok_p = np.isfinite(f.pressure_base).any()

# ---------- Header ----------
st.markdown(f'<div class="crumb">Smart Water &nbsp;/&nbsp; Benchmark California &nbsp;/&nbsp; <b>{sup["county"][sid].title()}</b></div>'
            f'<h1 class="t">{sup["name"][sid]}</h1>'
            f'<div class="sub">Observasi terakhir <span class="mono">{h.date.max():%Y-%m}</span> · horizon 12 bulan · '
            f'populasi <span class="mono">{fmt(pop,0)}</span></div>', unsafe_allow_html=True)

dm, sm, cv = met["demand_forecast"], met["supply_forecast"], met["demand_forecast"]["interval_80_test_coverage"]
st.markdown('<div class="grid">' +
    card("MAPE demand (uji)", f"{fmt(dm['mape_model_%'],2)}%", f"baseline naïve {fmt(dm['mape_seasonal_naive_%'],2)}%") +
    card("MAPE supply (musiman)", f"{fmt(sm['mape_%'],2)}%", "seasonal-naive 12 bulan") +
    card("Cakupan interval 80%", f"{fmt(cv*100,1)}%", "target 80%") +
    card("Air data center", f"{fmt(gal_day*30/1e6,2)} jt gal/bln" if on else "—", f"+{fmt(extra_gpcd,1)} GPCD" if on else "skenario nonaktif") +
    '</div>', unsafe_allow_html=True)

# ---------- Executive summary ----------
def first_high(col):
    m = dates[f[col] == "High"]
    return m.iloc[0].strftime("%Y-%m") if len(m) else None
if ok_p:
    pk_b, pk_s = np.nanmax(f.pressure_base), np.nanmax(f.pressure_scenario)
    mb, ms = first_high("band_base"), first_high("band_scen")
    pk_m = dates[int(np.nanargmax(f.pressure_base))].strftime("%B %Y")
    txt = (f"Tekanan puncak baseline <b class='mono'>{fmt(pk_b,2)}</b> pada <b>{pk_m}</b>. "
           f"Status <b>Tinggi</b> baseline: <span class='mono'>{mb or 'tidak muncul'}</span>.")
    if on:
        dl = (pk_s / pk_b - 1) * 100
        txt += (f" Dengan skenario: puncak <b class='mono'>{fmt(pk_s,2)}</b> "
                f"(<span class='mono {'up' if dl>0 else 'dn'}'>{dl:+.1f}%</span>), "
                f"Tinggi mulai <span class='mono'>{ms or 'tidak muncul'}</span>.")
else:
    txt = "Data sisi pasokan tidak tersedia untuk pemasok ini, sehingga indikator tekanan tidak dapat dihitung andal."
st.markdown(f'<div class="exec"><div class="h">Ringkasan</div>{txt}</div>', unsafe_allow_html=True)

# ---------- Charts ----------
st.markdown('<div class="sec">Prakiraan kebutuhan air residensial (GPCD)</div>', unsafe_allow_html=True)
hist = h.tail(36)[["date", "gpcd"]].rename(columns={"gpcd": "v"}).assign(s="Histori")
fc = f[["date", "pred"]].rename(columns={"pred": "v"}).assign(s="Prakiraan")
parts = [hist, fc] + ([f[["date", "scenario_demand"]].rename(columns={"scenario_demand": "v"}).assign(s="Skenario")] if on else [])
L = pd.concat(parts)
dom = ["Histori", "Prakiraan", "Skenario"][: 3 if on else 2]
rng = ["gray", "#3b82c4", "#d9534f"][: len(dom)]
x = alt.X("date:T", title=None, axis=alt.Axis(format="%Y-%m"))
band = alt.Chart(f).mark_area(opacity=.15, color="#3b82c4").encode(x=x, y=alt.Y("lo:Q", title=None, scale=alt.Scale(zero=False)), y2="hi:Q")
lines = alt.Chart(L).mark_line(strokeWidth=2).encode(
    x=x, y=alt.Y("v:Q", scale=alt.Scale(zero=False), title=None),
    color=alt.Color("s:N", scale=alt.Scale(domain=dom, range=rng), legend=alt.Legend(orient="top", title=None)),
    strokeDash=alt.StrokeDash("s:N", scale=alt.Scale(domain=dom, range=[[1, 0], [1, 0], [5, 4]][: len(dom)]), legend=None))
st.altair_chart((band + lines).properties(height=240), width="stretch")
st.caption("Pita biru = interval prediksi 80% (kalibrasi residual per horizon).")

if ok_p:
    st.markdown('<div class="sec">Indikator tekanan (permintaan ÷ pasokan)</div>', unsafe_allow_html=True)
    P = f[["date", "pressure_base"]].rename(columns={"pressure_base": "v"}).assign(s="Baseline")
    if on: P = pd.concat([P, f[["date", "pressure_scenario"]].rename(columns={"pressure_scenario": "v"}).assign(s="Skenario")])
    pd_dom = ["Baseline", "Skenario"][: 2 if on else 1]
    pl = alt.Chart(P).mark_line(strokeWidth=2).encode(
        x=x, y=alt.Y("v:Q", scale=alt.Scale(zero=False), title=None),
        color=alt.Color("s:N", scale=alt.Scale(domain=pd_dom, range=["#3b82c4", "#d9534f"][: len(pd_dom)]), legend=alt.Legend(orient="top", title=None)))
    rules = alt.Chart(pd.DataFrame({"y": [pt["low_watch_q60"], pt["watch_high_q85"]], "c": ["Batas Waspada", "Batas Tinggi"]})
                      ).mark_rule(strokeDash=[4, 4], opacity=.6).encode(y="y:Q", color=alt.Color("c:N", scale=alt.Scale(range=["#e0a030", "#d9534f"]), legend=None))
    st.altair_chart((pl + rules).properties(height=180), width="stretch")
    st.caption(f"Garis putus-putus = kuantil empiris Q60 (<span class='mono'>{pt['low_watch_q60']:.2f}</span>) dan Q85 "
               f"(<span class='mono'>{pt['watch_high_q85']:.2f}</span>); bukan ambang regulasi.".replace("<span class='mono'>", "").replace("</span>", ""))

# ---------- Table ----------
st.markdown('<div class="sec">Rincian 12 bulan</div>', unsafe_allow_html=True)
rows = {"Bulan": dates.dt.strftime("%Y-%m"), "Demand": f.pred.round(1), "Bawah 80%": f.lo.round(1), "Atas 80%": f.hi.round(1),
        "Pasokan (PCD)": f.supply_pcd.round(1), "Tekanan": f.pressure_base.round(2), "Status": f.band_base.map(BAND_ID),
        "Sinyal DWR": lab(p0), "P(DWR tinggi)": p0[:, ih].round(2)}
if on:
    rows.update({"Demand skenario": f.scenario_demand.round(1), "Tekanan skenario": f.pressure_scenario.round(2),
                 "Status skenario": f.band_scen.map(BAND_ID), "Sinyal DWR skenario*": lab(p1)})
T = pd.DataFrame(rows)
inv = {v: k for k, v in BAND_ID.items()}
color = lambda v: f"color:{BAND_COL.get(inv.get(v,''), 'inherit')};font-weight:500" if v in inv else ""
sty = T.style.format(precision=2, na_rep="–")
sty = (sty.map if hasattr(sty, "map") else sty.applymap)(color, subset=[c for c in T.columns if c.startswith("Status")])
st.dataframe(sty, hide_index=True, width="stretch")
if on: st.caption("*Sinyal DWR bersifat administratif dan hanya indikatif terhadap skenario.")
st.download_button("⬇ Unduh CSV", T.to_csv(index=False).encode(), file_name=f"forecast_{sid}.csv", mime="text/csv")

# ---------- Model card ----------
with st.expander("Model card & keterbatasan"):
    st.markdown("""
- **Prakiraan demand** memprediksi pemakaian air residensial (R-GPCD), bukan stok waduk.
- **Prakiraan pasokan** memakai seasonal-naive 12 bulan karena mengungguli model boosting pada holdout 2025+.
- **Indikator tekanan** = GPCD prakiraan ÷ pasokan per kapita-hari; mengabaikan permintaan non-residensial dan bukan indeks stres air regulatif.
- **Sinyal DWR** memprediksi tahap shortage yang dideklarasikan; label administratif dan sangat persisten (baseline carry-forward >98%).
- California adalah dataset benchmark; klaim dibingkai sebagai metodologi yang dapat ditransfer, bukan model global.
""")
    st.json(met, expanded=False)
