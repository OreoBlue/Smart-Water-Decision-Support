"""Smart Water Decision Support — UI extensions for the ICONFEST 2026 demo.

Data-science calculations, saved models, metric definitions and existing dashboard
sections remain unchanged. Added: county-reference map selection, contextual help,
fixed dark UI, three scenario presets and subtle presentation styling.
"""

import html
import json
from pathlib import Path

import altair as alt
import folium
from folium.plugins import MarkerCluster
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core import *

st.set_page_config(page_title="Smart Water Decision Support", page_icon="💧", layout="wide")

# ---------- Design tokens ----------
ACC, OK, WARN, BAD = "#3699ff", "#31c48d", "#f3b547", "#fb6574"
BAND_COL = {"Low": OK, "Watch": WARN, "High": BAD, "n/a": "gray"}
BAND_ID = {"Low": "Rendah", "Watch": "Waspada", "High": "Tinggi", "n/a": "n/a"}

HELP = {
    "unit": "Pilih unit layanan air dari daftar ini atau klik salah satu titik di peta. Peta memakai referensi centroid county, bukan koordinat fasilitas supplier.",
    "map": "Titik mewakili unit layanan menggunakan pusat county yang tercatat pada dataset. Beberapa supplier dapat berada pada titik yang sama. Perbesar dan klik cluster untuk memisahkan penanda, lalu klik satu titik untuk memilih supplier.",
    "on": "Aktifkan untuk menambahkan estimasi kebutuhan air data center pada baseline residensial.",
    "mw": "Beban IT (MW) adalah kapasitas daya perangkat komputasi. Semakin besar daya aktif, semakin besar kebutuhan energi pendinginan dalam skenario.",
    "util": "Utilisasi rata-rata menyatakan bagian kapasitas IT yang aktif: 0,30 berarti 30%, sedangkan 1,00 berarti 100%.",
    "temp": "Suhu udara digunakan bersama kelembapan untuk menghitung wet-bulb temperature yang memengaruhi asumsi kebutuhan air pendinginan.",
    "rh": "Kelembapan relatif udara (%) membantu menghitung wet-bulb temperature sebagai input model penggunaan air pendinginan.",
    "mode": "Dua pendekatan empiris yang berbeda untuk menghubungkan kondisi wet-bulb dengan WUE. Keduanya bukan kalibrasi fasilitas data center tertentu.",
    "lam": "Pengali asumsi efisiensi WUE: < 1 lebih hemat air; = 1 acuan; > 1 lebih boros. Parameter skenario, bukan hasil kalibrasi lokal.",
    "preset": "Tiga contoh asumsi operasi data center untuk perbandingan what-if; bukan kelas baku industri dan bukan kategori risiko air. Sesudah memilih preset, setiap slider dapat diatur sendiri.",
    "demand": "MAPE pada periode uji 2025+: rata-rata kesalahan persentase absolut prakiraan demand residensial. MAPE lebih kecil berarti error rata-rata lebih rendah.",
    "supply": "MAPE evaluasi metode seasonal-naive untuk benchmark potable supply per kapita per hari. Metode musiman dipilih karena mengungguli supply boosting yang diuji.",
    "coverage": "Dari seluruh observasi uji, persentase nilai aktual yang termasuk dalam interval prakiraan nominal 80%. Cakupan empiris dapat berbeda dari angka nominal.",
    "water": "Estimasi air langsung di lokasi data center dari MW × utilisasi × 24 jam × WUE. Tampilan bulanan memakai asumsi 30 hari. Jejak air listrik tidak dihitung.",
    "forecast_chart": "Garis abu-abu adalah histori 36 bulan; biru adalah prakiraan 12 bulan; merah putus-putus adalah skenario saat aktif. Pita biru menunjukkan interval prediksi 80%.",
    "pressure_chart": "Pressure proxy = prakiraan residential GPCD ÷ benchmark potable supply PCD. Kategori Rendah/Waspada/Tinggi menggunakan Q60/Q85 historis; bukan indeks stres air atau ambang regulasi.",
    "table": "Tabel lengkap per bulan: prakiraan demand, interval 80%, benchmark pasokan, pressure, kategori empiris, sinyal administratif DWR, dan perbandingan skenario.",
}

PRESETS = {
    "Low": {"mw": 1, "util": 0.33, "temp": 15, "rh": 30, "mode": "Fixed approach", "lam": 0.60,
            "description": "Asumsi ringan: IT 1 MW, utilisasi 33%, suhu 15°C, RH 30%, λ 0,60."},
    "Medium": {"mw": 10, "util": 0.65, "temp": 30, "rh": 75, "mode": "Fixed approach", "lam": 1.00,
               "description": "Asumsi menengah: IT 10 MW, utilisasi 65%, suhu 30°C, RH 75%, λ 1,00."},
    "High": {"mw": 40, "util": 0.90, "temp": 35, "rh": 80, "mode": "Fixed cold water", "lam": 1.25,
             "description": "Asumsi berat: IT 40 MW, utilisasi 90%, suhu 35°C, RH 80%, λ 1,25."},
}
WIDGET_KEYS = {"mw": "dc_mw", "util": "dc_util", "temp": "dc_temp", "rh": "dc_rh",
               "mode": "dc_mode", "lam": "dc_lam"}


def fmt(x, n=1):
    return f"{x:,.{n}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def esc(value):
    return html.escape(str(value), quote=True)


def help_badge(description):
    """Native title is the accessible fallback; CSS also exposes on hover/focus."""
    safe = esc(description)
    return (f'<span class="sw-help" role="note" tabindex="0" title="{safe}" '
            f'aria-label="Bantuan: {safe}" data-tip="{safe}">?</span>')


def section(text, description, icon=""):
    st.markdown(f'<div class="sec"><span class="sec-icon">{icon}</span>{esc(text)}'
                f'{help_badge(description)}</div>', unsafe_allow_html=True)


def card(label, value, sub="", hint="", symbol=""):
    # Same four original metrics, same calculation and output, plus help and icons.
    return (f'<div class="m"><div class="metric-top">'
            f'<span class="metric-icon">{symbol}</span>'
            f'<span class="l">{esc(label)}{help_badge(hint)}</span></div>'
            f'<div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div></div>')


def apply_preset(name):
    p = PRESETS[name]
    for short, widget_key in WIDGET_KEYS.items():
        st.session_state[widget_key] = p[short]
    st.session_state["selected_preset"] = name
    st.session_state["scenario_enabled"] = True


def reset_scenario():
    apply_preset("Medium")


if "selected_preset" not in st.session_state:
    st.session_state["selected_preset"] = "Medium"
# Only set initial values if absent, not on every re-run (preserves manually adjusted sliders).
for short, widget_key in WIDGET_KEYS.items():
    st.session_state.setdefault(widget_key, PRESETS["Medium"][short])
st.session_state.setdefault("scenario_enabled", True)


# Fixed dark palette: no theme switch or light-mode dependency.
P = {"bg": "#090e19", "sidebar": "#161f31", "panel": "#101a2b", "panel2": "#11233b",
     "text": "#f0f5ff", "muted": "#a4b0c3", "border": "#263854", "grid": "#33465d"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root {{ --sw-bg:{P['bg']};--sw-panel:{P['panel']};--sw-panel2:{P['panel2']};--sw-text:{P['text']};
  --sw-muted:{P['muted']};--sw-border:{P['border']};--sw-grid:{P['grid']};--sw-accent:{ACC}; }}
html, body, .stApp, [class*="css"], .stMarkdown {{ font-family:'IBM Plex Sans',system-ui,sans-serif; }}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background:var(--sw-bg); color:var(--sw-text); }}
.block-container {{ max-width:1440px; padding-top:0.8rem; padding-bottom:3rem; }}
#MainMenu, footer {{ visibility:hidden; }}
header[data-testid="stHeader"] {{ background:transparent; }}
section[data-testid="stSidebar"] {{ background:{P['sidebar']}; border-right:1px solid var(--sw-border); }}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label {{ color:var(--sw-text); }}
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"] > div {{ background:var(--sw-panel);color:var(--sw-text);border-color:var(--sw-border); }}
section[data-testid="stSidebar"] [data-baseweb="select"] span {{ color:var(--sw-text); }}
.stApp [data-testid="stMarkdownContainer"] p {{ color:inherit; }}
.crumb {{ color:var(--sw-muted);font-size:12px;margin:3px 0 5px; }}
.crumb b {{ color:var(--sw-text);font-weight:600; }}
h1.t {{ color:var(--sw-text);font-size:25px;font-weight:700;line-height:1.2;margin:0 0 5px;padding:0; }}
.sub {{ color:var(--sw-muted);font-size:12px;margin-bottom:10px; }}
.sec {{ color:var(--sw-text);font-size:15px;font-weight:650;margin:16px 0 8px;display:flex;align-items:center;gap:7px; }}
.sec-icon {{ color:var(--sw-accent); }}
.mono,.v,td.n {{ font-family:'IBM Plex Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums; }}
.grid {{ display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px; }}
.m {{ position:relative;border:1px solid var(--sw-border);border-radius:12px;padding:14px 15px;
      background:var(--sw-panel);min-width:0; }}
.metric-top {{ display:flex;align-items:center;gap:8px; }}
.metric-icon {{ color:var(--sw-accent);font-size:19px;min-width:23px; }}
.m .l {{ color:var(--sw-muted);font-size:12px; }}
.m .v {{ color:var(--sw-text);font-size:23px;font-weight:600;margin:7px 0 2px;overflow-wrap:anywhere; }}
.m .s {{ color:var(--sw-muted);font-size:12px; }}
.exec {{ border:1px solid var(--sw-border);border-left:3px solid var(--sw-accent);border-radius:12px;
        padding:13px 16px;background:var(--sw-panel2);margin:12px 0 10px;color:var(--sw-text); }}
.exec .h {{ color:var(--sw-muted);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px; }}
.up {{ color:{BAD}; }} .dn {{ color:{OK}; }}
.sw-help {{ display:inline-flex;vertical-align:middle;justify-content:center;align-items:center;position:relative;
   flex-shrink:0;cursor:help;width:17px;height:17px;margin-left:5px;border:1px solid {ACC}77;
   color:{ACC};font-size:11px;font-weight:700;line-height:1;border-radius:50%;background:{ACC}16; }}
.sw-help:hover::after,.sw-help:focus::after {{ content:attr(data-tip);display:block;position:absolute;z-index:999999;
    bottom:calc(100% + 9px);left:50%;transform:translateX(-50%);width:max-content;max-width:300px;
    min-width:195px;white-space:normal;line-height:1.5;text-align:left;padding:9px 11px;
    border-radius:9px;background:#edf4ff;color:#132238;
    border:1px solid {P['border']};box-shadow:0 12px 35px #0004;font-size:12px;font-weight:400; }}
.sidebar-brand {{ padding:5px 0 10px; }}
.sidebar-brand .brand-name {{ font-size:20px;line-height:1.2;font-weight:700;color:var(--sw-text); }}
.sidebar-brand .brand-line {{ font-size:11px;color:var(--sw-muted);margin-top:6px; }}
.sidebar-brand .brand-icon {{ font-size:22px;color:{ACC};margin-right:6px; }}
section[data-testid="stSidebar"] .sec {{ color:var(--sw-muted);font-size:11px;letter-spacing:.07em;
  text-transform:uppercase;font-weight:700;margin:17px 0 7px; }}
.map-caption {{ font-size:12px;color:var(--sw-muted);line-height:1.45;margin:-2px 0 8px; }}
.map-head {{ display:flex;align-items:center;gap:8px;justify-content:space-between; }}
.action-note {{ color:var(--sw-muted);font-size:11px;text-align:right; }}
[data-testid="stVegaLiteChart"], [data-testid="stDataFrame"] {{ border:1px solid var(--sw-border); border-radius:12px; overflow:hidden; background:var(--sw-panel); }}
@media(max-width:980px) {{ .grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .m .v {{ font-size:21px; }} }}
@media(max-width:560px) {{ .grid {{ grid-template-columns:1fr; }} h1.t {{ font-size:21px; }} }}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    d = load_clean("data/processed.csv.gz")
    return d, forecast_frame(d)


@st.cache_data
def load_locations():
    filename = Path("data/supplier_locations.csv")
    if not filename.is_file():
        return pd.DataFrame(columns=["sid", "name", "county", "county_primary", "lat", "lon"])
    frame = pd.read_csv(filename, dtype={"sid": str})
    for key in ("lat", "lon"):
        frame[key] = pd.to_numeric(frame[key], errors="coerce")
    return frame.dropna(subset=["lat", "lon"]).drop_duplicates("sid")


d, F = load()
M = joblib.load("models/models.joblib")
with open("models/metrics.json", encoding="utf-8") as fd:
    met = json.load(fd)
sup = d.drop_duplicates("sid").set_index("sid")

# ---------- Sidebar: original selection and original six controls stay in place ----------
sb = st.sidebar
sb.markdown('<div class="sidebar-brand"><span class="brand-icon">💧</span>'
            '<span class="brand-name">Smart Water<br>Decision Support</span>'
            '<div class="brand-line">Forecasting · Pressure Proxy · Scenario Analysis</div></div>',
            unsafe_allow_html=True)

# Assigning a widget key after instantiation on the same run is forbidden in Streamlit.
# Queue map clicks and set the value before the sidebar selectbox is instantiated.
next_sid = st.session_state.pop("map_pending_sid", None)
if next_sid in sup.index:
    st.session_state["selected_sid"] = next_sid
if st.session_state.get("selected_sid") not in sup.index:
    st.session_state["selected_sid"] = sup.index[0]

sb.markdown('<div class="sec">UNIT LAYANAN ' + help_badge(HELP["unit"]) + '</div>', unsafe_allow_html=True)
sid = sb.selectbox("Pemasok air", sup.index,
                   format_func=lambda s: f"{sup['name'][s]} — {sup['county'][s]}",
                   label_visibility="collapsed", key="selected_sid", help=HELP["unit"])
sb.markdown('<div class="sec">Skenario data center</div>', unsafe_allow_html=True)
on = sb.toggle("Aktifkan skenario", key="scenario_enabled", help=HELP["on"])

sb.markdown('<div class="sec">Preset skenario ' + help_badge(HELP["preset"]) + '</div>', unsafe_allow_html=True)
pre_cols = sb.columns(3, gap="small")
for col, name in zip(pre_cols, PRESETS):
    with col:
        sb_key = "sw_preset_" + name.lower()
        st.button(name, use_container_width=True,
                  type="primary" if st.session_state.get("selected_preset") == name else "secondary",
                  key=sb_key, on_click=apply_preset, args=(name,),
                  help=PRESETS[name]["description"] + " Klik untuk menerapkan pada seluruh slider.")
sb.caption("Preset terakhir: " + st.session_state["selected_preset"] + ". "
           + PRESETS[st.session_state["selected_preset"]]["description"]
           + " Nilainya dapat disesuaikan manual.")

mw = sb.slider("Beban IT (MW)", 1, 100, key="dc_mw", disabled=not on, help=HELP["mw"])
util = sb.slider("Utilisasi rata-rata", 0.3, 1.0, key="dc_util", disabled=not on, help=HELP["util"])
temp = sb.slider("Suhu udara (°C)", 15, 40, key="dc_temp", disabled=not on, help=HELP["temp"])
rh = sb.slider("Kelembapan relatif (%)", 30, 95, key="dc_rh", disabled=not on, help=HELP["rh"])
mode = sb.selectbox("Model pendinginan", ["Fixed approach", "Fixed cold water"],
                    key="dc_mode", disabled=not on, help=HELP["mode"])
lam = sb.slider("Pengali efisiensi pendinginan (λ)", 0.5, 1.5,
                key="dc_lam", disabled=not on, help=HELP["lam"])

# ---------- Original computation logic (preserved) ----------
h = d[d.sid == sid].copy()
t0 = int(h.t.max())
f = F[(F.sid == sid) & (F.t == t0)].sort_values("h").reset_index(drop=True)
f["pred"] = M["reg"].predict(f[FEATS_F]) * f.m12
f["lo"], f["hi"] = interval_from_residuals(f.pred, f.h, M["interval_residuals"])
pop, clim, hc = float(np.exp(h.logpop.iloc[-1])), h.clim.iloc[-1], h.hc.iloc[-1]
f["supply_pcd"] = supply_forecast_for_supplier(h, f.tt)

wbt = wet_bulb(temp, rh)
wue = wue_onsite(wbt, mode, lam)
gal_day = dc_gal_per_day(mw, util, wue)
extra_gpcd = gal_day / pop if on else 0.0
f["scenario_demand"] = f.pred + extra_gpcd
f["pressure_base"] = f.pred / f.supply_pcd
f["pressure_scenario"] = f.scenario_demand / f.supply_pcd
pt = M["pressure_thresholds"]
f["band_base"] = empirical_pressure_band(f.pressure_base, pt["low_watch_q60"], pt["watch_high_q85"])
f["band_scen"] = empirical_pressure_band(f.pressure_scenario, pt["low_watch_q60"], pt["watch_high_q85"])

mo = (f.tt - 1) % 12 + 1
classes = M["clf"].classes_.astype(int)
probs = lambda g: M["clf"].predict_proba(clf_matrix(g, M["sup_mean"][sid], f.ref, mo, clim, hc, np.log(pop)))
p0, p1 = probs(f.pred), probs(f.scenario_demand)
lab = lambda p: [CLASSES[classes[i]] for i in p.argmax(1)]
ih = list(classes).index(2)
dates = pd.to_datetime(dict(year=(f.tt - 1) // 12, month=(f.tt - 1) % 12 + 1, day=1))
f["date"] = dates.values
ok_p = np.isfinite(f.pressure_base).any()

# ---------- Reset scenario; the dashboard stays in dark mode ----------
head_space, reset_space = st.columns([9, 1.7], vertical_alignment="center")
with reset_space:
    st.button("↺ Reset skenario", on_click=reset_scenario, key="reset_dc_scenario",
              help="Kembalikan seluruh parameter skenario ke preset Medium. Tidak mengubah model atau data.")

# ---------- Original header ----------
st.markdown(f'<div class="crumb">Smart Water &nbsp;/&nbsp; Benchmark California &nbsp;/&nbsp; <b>{esc(str(sup["county"][sid]).title())}</b></div>'
            f'<h1 class="t">{esc(sup["name"][sid])}</h1>'
            f'<div class="sub">Observasi terakhir <span class="mono">{h.date.max():%Y-%m}</span> · horizon 12 bulan · '
            f'populasi <span class="mono">{fmt(pop,0)}</span></div>', unsafe_allow_html=True)

# ---------- New California map: between observation subtitle and FOUR ORIGINAL KPI cards ----------
section("Peta unit layanan California", HELP["map"], "◉")
locations = load_locations()
locations = locations[locations["sid"].isin(sup.index)]
if not locations.empty:
    # The CARTO tiles require a key (Streamlit Secrets); use an OSM fallback if not configured.
    # Never write the key to source code or to a GitHub commit.
    try:
        carto_api_key = st.secrets.get("CARTO_API_KEY", "")
    except Exception:  # Streamlit may raise its own exception when no secrets file exists.
        carto_api_key = ""

    mapobj = folium.Map(
        location=[36.6, -119.4],
        zoom_start=6,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    if carto_api_key:
        folium.TileLayer(
            tiles=(
                "https://basemaps.cartocdn.com/dark_all/"
                f"{{z}}/{{x}}/{{y}}.png?key={carto_api_key}"
            ),
            attr="© OpenStreetMap contributors © CARTO",
            name="CARTO Dark Matter",
            max_zoom=20,
        ).add_to(mapobj)
    else:
        folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap fallback").add_to(mapobj)
        st.warning("CARTO_API_KEY belum terpasang di Streamlit Secrets. Peta tetap ditampilkan menggunakan OpenStreetMap sebagai cadangan.")

    marks = MarkerCluster(
        name="Supplier (referensi county)",
        options={"spiderfyOnMaxZoom": True, "showCoverageOnHover": False,
                 "zoomToBoundsOnClick": True, "maxClusterRadius": 45},
    )
    marks.add_to(mapobj)
    for rec in locations.itertuples(index=False):
        is_selected = rec.sid == sid
        # IDs are the tooltip text; st_folium returns this on click for sidebar synchronization.
        folium.CircleMarker(
            location=[float(rec.lat), float(rec.lon)],
            radius=8 if is_selected else 5,
            weight=2 if is_selected else 1,
            color="#a5e6ff" if is_selected else ACC,
            fill=True,
            fill_color="#55c9ff" if is_selected else "#4289fb",
            fill_opacity=0.97 if is_selected else 0.76,
            tooltip=esc(rec.sid),
            popup=folium.Popup(
                f'<b>{esc(rec.name)}</b><br>County: {esc(rec.county)}'
                f'<br>ID: {esc(rec.sid)}<br><small>Titik referensi county (bukan alamat supplier)</small>',
                max_width=260,
            ),
        ).add_to(marks)
    map_result = st_folium(
        mapobj, height=315, use_container_width=True,
        returned_objects=["last_object_clicked_tooltip"], key=f"sw_map_{sid}",
    )
    chosen = (map_result or {}).get("last_object_clicked_tooltip")
    if chosen in sup.index and chosen != sid:
        st.session_state["map_pending_sid"] = chosen
        st.rerun()
    st.markdown(
        f'<div class="map-caption">Menampilkan <b>{len(locations)}</b> unit layanan. '
        'Klik cluster untuk memperbesar/memisahkan titik, lalu klik titik untuk memilih unit layanan. '
        '<b>Lokasi merupakan centroid county sebagai referensi visual, bukan koordinat fasilitas.</b> '
        'Sumber: US Census TIGERweb, ACS26 (2026).</div>', unsafe_allow_html=True,
    )
else:
    st.info("File lokasi supplier tidak ditemukan. Tambahkan data/supplier_locations.csv untuk mengaktifkan peta.")

# ---------- Original four KPI cards, with added individual help ----------
dm, sm, cv = met["demand_forecast"], met["supply_forecast"], met["demand_forecast"]["interval_80_test_coverage"]
st.markdown('<div class="grid">' +
    card("MAPE demand (uji)", f"{fmt(dm['mape_model_%'],2)}%",
         f"baseline naïve {fmt(dm['mape_seasonal_naive_%'],2)}%", HELP["demand"], "▥") +
    card("MAPE supply (musiman)", f"{fmt(sm['mape_%'],2)}%",
         "seasonal-naive 12 bulan", HELP["supply"], "◈") +
    card("Cakupan interval 80%", f"{fmt(cv*100,1)}%", "target 80%", HELP["coverage"], "◎") +
    card("Air data center", f"{fmt(gal_day*30/1e6,2)} jt gal/bln" if on else "—",
         f"+{fmt(extra_gpcd,1)} GPCD" if on else "skenario nonaktif", HELP["water"], "◉") +
    '</div>', unsafe_allow_html=True)

# ---------- Original executive summary ----------
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

# ---------- Original demand chart (same series and forecast interval), plus contextual help and fixed dark styling ----------
section("Prakiraan kebutuhan air residensial (GPCD)", HELP["forecast_chart"], "⌁")
hist = h.tail(36)[["date", "gpcd"]].rename(columns={"gpcd": "v"}).assign(s="Histori")
fc = f[["date", "pred"]].rename(columns={"pred": "v"}).assign(s="Prakiraan")
parts = [hist, fc] + ([f[["date", "scenario_demand"]].rename(columns={"scenario_demand": "v"}).assign(s="Skenario")] if on else [])
L = pd.concat(parts)
dom = ["Histori", "Prakiraan", "Skenario"][: 3 if on else 2]
rng = ["#919aa9", ACC, BAD][: len(dom)]
x = alt.X("date:T", title=None, axis=alt.Axis(format="%Y-%m", labelOverlap=True))
band = alt.Chart(f).mark_area(opacity=.15, color=ACC).encode(
    x=x, y=alt.Y("lo:Q", title=None, scale=alt.Scale(zero=False)), y2="hi:Q")
lines = alt.Chart(L).mark_line(strokeWidth=2).encode(
    x=x, y=alt.Y("v:Q", scale=alt.Scale(zero=False), title=None),
    color=alt.Color("s:N", scale=alt.Scale(domain=dom, range=rng), legend=alt.Legend(orient="top", title=None)),
    strokeDash=alt.StrokeDash("s:N", scale=alt.Scale(domain=dom, range=[[1, 0], [1, 0], [5, 4]][:len(dom)]), legend=None))
chart_opts = {"view": {"stroke": "transparent"},
              "background": P["panel"],
              "axis": {"labelColor": P["muted"], "titleColor": P["text"], "gridColor": P["grid"],
                       "domainColor": P["border"], "tickColor": P["border"]},
              "legend": {"labelColor": P["text"], "titleColor": P["text"]}}
st.altair_chart((band + lines).properties(height=240).configure(**chart_opts), width="stretch")
st.caption("Pita biru = interval prediksi 80% (kalibrasi residual per horizon).")

if ok_p:
    section("Indikator tekanan (permintaan ÷ pasokan)", HELP["pressure_chart"], "⌁")
    Pdata = f[["date", "pressure_base"]].rename(columns={"pressure_base": "v"}).assign(s="Baseline")
    if on:
        Pdata = pd.concat([Pdata, f[["date", "pressure_scenario"]].rename(columns={"pressure_scenario": "v"}).assign(s="Skenario")])
    pd_dom = ["Baseline", "Skenario"][: 2 if on else 1]
    pl = alt.Chart(Pdata).mark_line(strokeWidth=2).encode(
        x=x, y=alt.Y("v:Q", scale=alt.Scale(zero=False), title=None),
        color=alt.Color("s:N", scale=alt.Scale(domain=pd_dom, range=[ACC, BAD][:len(pd_dom)]),
                        legend=alt.Legend(orient="top", title=None)))
    rules = alt.Chart(pd.DataFrame({"y": [pt["low_watch_q60"], pt["watch_high_q85"]],
                                     "c": ["Batas Waspada", "Batas Tinggi"]})
                      ).mark_rule(strokeDash=[4, 4], opacity=.75).encode(
        y="y:Q", color=alt.Color("c:N", scale=alt.Scale(domain=["Batas Waspada", "Batas Tinggi"],
                                                         range=[WARN, BAD]), legend=None))
    st.altair_chart((pl + rules).properties(height=180).configure(**chart_opts), width="stretch")
    st.caption(f"Garis putus-putus = kuantil empiris Q60 ({pt['low_watch_q60']:.2f}) dan "
               f"Q85 ({pt['watch_high_q85']:.2f}); bukan ambang regulasi.")

# ---------- Original monthly table + help; original download button remains one ----------
section("Rincian 12 bulan", HELP["table"], "▤")
rows = {"Bulan": dates.dt.strftime("%Y-%m"), "Demand": f.pred.round(1),
        "Bawah 80%": f.lo.round(1), "Atas 80%": f.hi.round(1),
        "Pasokan (PCD)": f.supply_pcd.round(1), "Tekanan": f.pressure_base.round(2),
        "Status": f.band_base.map(BAND_ID), "Sinyal DWR": lab(p0), "P(DWR tinggi)": p0[:, ih].round(2)}
if on:
    rows.update({"Demand skenario": f.scenario_demand.round(1), "Tekanan skenario": f.pressure_scenario.round(2),
                 "Status skenario": f.band_scen.map(BAND_ID), "Sinyal DWR skenario*": lab(p1)})
T = pd.DataFrame(rows)
inv = {v: k for k, v in BAND_ID.items()}
color = lambda v: f"color:{BAND_COL.get(inv.get(v,''), 'inherit')};font-weight:500" if v in inv else ""
sty = T.style.format(precision=2, na_rep="–")
sty = (sty.map if hasattr(sty, "map") else sty.applymap)(color,
             subset=[c for c in T.columns if c.startswith("Status")])
st.dataframe(sty, hide_index=True, width="stretch")
if on:
    st.caption("*Sinyal DWR bersifat administratif dan hanya indikatif terhadap skenario.")
st.download_button("⬇ Unduh CSV", T.to_csv(index=False).encode(),
                   file_name=f"forecast_{sid}.csv", mime="text/csv")

# ---------- Original model card and limitations (preserved) ----------
with st.expander("Model card & keterbatasan"):
    st.markdown("""
- **Prakiraan demand** memprediksi pemakaian air residensial (R-GPCD), bukan stok waduk.
- **Prakiraan pasokan** memakai seasonal-naive 12 bulan karena mengungguli model boosting pada holdout 2025+.
- **Indikator tekanan** = GPCD prakiraan ÷ pasokan per kapita-hari; mengabaikan permintaan non-residensial dan bukan indeks stres air regulatif.
- **Sinyal DWR** memprediksi tahap shortage yang dideklarasikan; label administratif dan sangat persisten (baseline carry-forward >98%).
- California adalah dataset benchmark; klaim dibingkai sebagai metodologi yang dapat ditransfer, bukan model global.
""")
    st.json(met, expanded=False)
