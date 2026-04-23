import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import re

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title="QA System — Smart Dashboard", layout="wide", page_icon="🏭")

st.markdown("""
<style>
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden !important; }
.block-container { padding-top: 1rem !important; }
.main { background-color: #f4f6f9; }
h1 { color: #1e3d59; }
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border-left: 5px solid #1e3d59;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    padding: 10px;
}
.info-box  { background:#e8f4fd; border-left:4px solid #2196F3; padding:10px 15px; border-radius:4px; margin:8px 0; font-size:.9em; }
.warn-box  { background:#fff8e1; border-left:4px solid #FFC107; padding:8px 12px; border-radius:4px; margin:4px 0; font-size:.85em; }
.ok-box    { background:#e8f5e9; border-left:4px solid #4caf50; padding:8px 12px; border-radius:4px; margin:4px 0; }
.viol-box  { background:#fdecea; border-left:4px solid #f44336; padding:8px 12px; border-radius:4px; margin:4px 0; font-size:.85em; }
.mapping-card { background:#fff; border:1px solid #dde3ed; border-radius:8px; padding:14px; margin:6px 0; }
.tag-attr { background:#e3f2fd; color:#1565c0; padding:2px 8px; border-radius:10px; font-size:.8em; font-weight:600; }
.tag-var  { background:#f3e5f5; color:#6a1b9a; padding:2px 8px; border-radius:10px; font-size:.8em; font-weight:600; }
.tag-date { background:#e8f5e9; color:#2e7d32; padding:2px 8px; border-radius:10px; font-size:.8em; font-weight:600; }
.tag-cat  { background:#fff3e0; color:#e65100; padding:2px 8px; border-radius:10px; font-size:.8em; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ===================================================================
# KONSTANTA SPC
# ===================================================================
SPC_FACTORS = {
    2:  (1.880, 0.000, 3.267, 2.659, 0.000, 3.267, 1.128),
    3:  (1.023, 0.000, 2.574, 1.954, 0.000, 2.568, 1.693),
    4:  (0.729, 0.000, 2.282, 1.628, 0.000, 2.266, 2.059),
    5:  (0.577, 0.000, 2.114, 1.427, 0.000, 2.089, 2.326),
    6:  (0.483, 0.000, 2.004, 1.287, 0.030, 1.970, 2.534),
    7:  (0.419, 0.076, 1.924, 1.182, 0.118, 1.882, 2.704),
    8:  (0.373, 0.136, 1.864, 1.099, 0.185, 1.815, 2.847),
    9:  (0.337, 0.184, 1.816, 1.032, 0.239, 1.761, 2.970),
    10: (0.308, 0.223, 1.777, 0.975, 0.284, 1.716, 3.078),
}
def get_spc(n): return SPC_FACTORS[max(2, min(10, int(n)))]

# ===================================================================
# SMART COLUMN DETECTOR
# ===================================================================
# Keyword sets untuk tiap peran kolom
ROLE_KEYWORDS = {
    "date":          ["tanggal","date","tgl","waktu","time","periode","period","datetime","hari","day","bulan","month","tahun","year","timestamp"],
    "sample_size":   ["qty_check","qty check","quantity check","n_check","jumlah cek","jumlah check","sample size","sample_size","inspeksi","checked","total check","n_inspeksi","jumlah_inspeksi","n_sample","lot size","batch size"],
    "defect_count":  ["qty_ng","qty ng","jumlah ng","ng","defect","defects","reject","rejected","jumlah reject","cacat","banyak cacat","n_defect","count_defect","defect_count","nonconforming","nc"],
    "defect_type":   ["jenis_defect","jenis defect","jenis_cacat","jenis cacat","tipe defect","type defect","defect type","jenis reject","reject type","mode kegagalan","failure mode","cacat_type"],
    "measurement":   ["diameter","panjang","lebar","tinggi","berat","suhu","tekanan","tebal","kedalaman","ukuran","dimensi","length","width","height","weight","temperature","pressure","thickness","depth","size","dimension","mm","cm","gram","kg","celcius","fahrenheit","ohm","volt","ampere","rpm","kpa","mpa","psi"],
    "line":          ["line","lini","mesin","machine","operator","shift","area","station","workstation","pos","position","proses","process"],
    "product":       ["produk","product","tipe_produk","tipe produk","type","model","sku","part","part_no","part_number","item","artikel"],
    "size_variant":  ["ukuran","size","variant","varian","spec","spesifikasi","grade"],
}

def normalize(s):
    """Lowercase, strip, ganti spasi+underscore jadi spasi."""
    return re.sub(r'[\s_]+', ' ', str(s).lower().strip())

def detect_column_role(col_name, series: pd.Series):
    """
    Deteksi peran sebuah kolom berdasarkan nama + karakteristik data.
    Return: role string
    """
    cn = normalize(col_name)

    # Cek date dulu (prioritas tinggi)
    if series.dtype == 'datetime64[ns]' or pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    for kw in ROLE_KEYWORDS["date"]:
        if kw in cn:
            try:
                pd.to_datetime(series.dropna().head(5))
                return "date"
            except: pass

    # Cek nama kolom vs keyword
    for role, keywords in ROLE_KEYWORDS.items():
        for kw in keywords:
            if kw == cn or cn.startswith(kw) or kw in cn:
                # Validasi tambahan berdasarkan dtype
                if role in ("sample_size", "defect_count") and not pd.api.types.is_numeric_dtype(series):
                    continue
                if role == "measurement" and not pd.api.types.is_numeric_dtype(series):
                    continue
                return role

    # Fallback: inferensi dari data
    if pd.api.types.is_numeric_dtype(series):
        nuniq = series.nunique()
        vmax  = series.max()
        vmean = series.mean()
        # Heuristik: kolom count/integer dengan nilai tidak terlalu besar
        if series.dtype in [np.int64, np.int32] or (series.dropna() == series.dropna().astype(int)).all():
            if vmax < 100000 and vmean < 10000:
                return "numeric_count"
        return "measurement"  # angka kontinu → kemungkinan variabel

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        nuniq = series.nunique()
        nrows = len(series.dropna())
        if nuniq < 20 and nrows > 0 and nuniq / nrows < 0.3:
            return "categorical"

    return "unknown"


def auto_detect_all_columns(df):
    """
    Jalankan deteksi untuk semua kolom, kembalikan dict {col_name: role}.
    Lalu resolve konflik (ambil yang paling cocok untuk tiap peran).
    """
    detected = {}
    for col in df.columns:
        detected[col] = detect_column_role(col, df[col])

    return detected


def resolve_mapping(df, detected):
    """
    Dari hasil deteksi, pilih SATU kolom terbaik per peran penting.
    Return mapping dict: {role: col_name or None}
    """
    mapping = {
        "date": None,
        "sample_size": None,
        "defect_count": None,
        "defect_type": None,
        "measurement": [],    # bisa banyak
        "line": None,
        "product": None,
        "size_variant": None,
        "categorical": [],
    }

    # Prioritaskan match nama spesifik
    for col, role in detected.items():
        if role == "date" and mapping["date"] is None:
            mapping["date"] = col
        elif role == "sample_size" and mapping["sample_size"] is None:
            mapping["sample_size"] = col
        elif role == "defect_count" and mapping["defect_count"] is None:
            mapping["defect_count"] = col
        elif role == "defect_type" and mapping["defect_type"] is None:
            mapping["defect_type"] = col
        elif role == "measurement":
            mapping["measurement"].append(col)
        elif role == "line" and mapping["line"] is None:
            mapping["line"] = col
        elif role == "product" and mapping["product"] is None:
            mapping["product"] = col
        elif role == "size_variant" and mapping["size_variant"] is None:
            mapping["size_variant"] = col
        elif role == "categorical":
            mapping["categorical"].append(col)
        elif role == "numeric_count":
            # Jika sample_size masih kosong, pakai ini
            if mapping["sample_size"] is None:
                mapping["sample_size"] = col
            elif mapping["defect_count"] is None:
                mapping["defect_count"] = col

    return mapping


def classify_dataset_type(mapping):
    """
    Tentukan tipe dataset utama berdasarkan kolom yang berhasil dipetakan.
    Return: "attribute", "variable", "mixed", "unknown"
    """
    has_attr = mapping["sample_size"] and mapping["defect_count"]
    has_var  = len(mapping["measurement"]) > 0

    if has_attr and has_var:   return "mixed"
    if has_attr:               return "attribute"
    if has_var:                return "variable"
    return "unknown"

# ===================================================================
# SPC HELPERS (sama seperti sebelumnya)
# ===================================================================
def detect_violations(values, ucl, lcl, cl):
    values = list(values)
    n = len(values)
    sigma = (ucl - cl) / 3 if (ucl - cl) != 0 else 1e-9
    violations = {1: [], 2: [], 3: [], 4: []}
    for i, v in enumerate(values):
        if v > ucl or v < lcl:
            violations[1].append(i)
    for i in range(7, n):
        w = values[i-7:i+1]
        if all(x > cl for x in w) or all(x < cl for x in w):
            violations[2].append(i)
    for i in range(5, n):
        w = values[i-5:i+1]
        d = [w[j+1]-w[j] for j in range(5)]
        if all(x > 0 for x in d) or all(x < 0 for x in d):
            violations[3].append(i)
    for i in range(2, n):
        w = values[i-2:i+1]
        if sum(1 for x in w if abs(x - cl) > 2*sigma) >= 2:
            violations[4].append(i)
    return violations

def show_violation_summary(violations):
    names = {1:"Rule 1 (Beyond 3σ)", 2:"Rule 2 (8 titik satu sisi CL)", 3:"Rule 3 (6 titik trend)", 4:"Rule 4 (2/3 di Zona A)"}
    has_v = any(v for v in violations.values())
    if not has_v:
        st.markdown('<div class="ok-box">✅ <b>In-Control</b> — Tidak ada pelanggaran terdeteksi.</div>', unsafe_allow_html=True)
    else:
        for r, idxs in violations.items():
            if idxs:
                st.markdown(f'<div class="viol-box">⚠️ <b>{names[r]}</b> — {len(idxs)} titik: {idxs[:8]}{"..." if len(idxs)>8 else ""}</div>', unsafe_allow_html=True)

def add_ctrl_lines(fig, dates, ucl, lcl, cl, row=None):
    """
    Tambah garis UCL/LCL/CL.
    row=None -> plain go.Figure() -> pakai add_hline biasa.
    row=int  -> make_subplots -> add_hline tidak support row/col,
                pakai add_shape + add_annotation manual.
    """
    sigma = (ucl - cl) / 3 if ucl != cl else 1e-9

    if row is None:
        # Plain Figure
        for mult, c in [(2, "rgba(255,165,0,0.3)"), (1, "rgba(0,128,0,0.2)")]:
            fig.add_hline(y=cl + mult*sigma, line_dash="dot", line_color=c, line_width=1)
            if cl - mult*sigma > 0:
                fig.add_hline(y=cl - mult*sigma, line_dash="dot", line_color=c, line_width=1)
        fig.add_hline(y=ucl, line_dash="dash", line_color="red", line_width=1.5,
                      annotation_text=f"UCL={ucl:.4f}", annotation_position="top right")
        fig.add_hline(y=lcl, line_dash="dash", line_color="red", line_width=1.5,
                      annotation_text=f"LCL={lcl:.4f}", annotation_position="bottom right")
        fig.add_hline(y=cl, line_color="green", line_width=2,
                      annotation_text=f"CL={cl:.4f}", annotation_position="top left")
    else:
        # Subplot: add_hline tidak support row/col -> pakai add_shape
        yref = "y" if row == 1 else f"y{row}"
        lines = [
            (ucl,              "red",                   "dash",  f"UCL={ucl:.4f}"),
            (lcl,              "red",                   "dash",  f"LCL={lcl:.4f}"),
            (cl,               "green",                 "solid", f"CL={cl:.4f}"),
            (cl + sigma,       "rgba(0,128,0,0.5)",     "dot",   ""),
            (cl - sigma,       "rgba(0,128,0,0.5)",     "dot",   ""),
            (cl + 2*sigma,     "rgba(255,165,0,0.6)",   "dot",   ""),
            (cl - 2*sigma,     "rgba(255,165,0,0.6)",   "dot",   ""),
        ]
        for y_val, color, dash, label in lines:
            if y_val < 0:
                continue
            fig.add_shape(
                type="line", xref="paper", x0=0, x1=1,
                yref=yref, y0=y_val, y1=y_val,
                line=dict(color=color, width=2 if dash=="solid" else 1,
                          dash="dash" if dash=="dash" else ("dot" if dash=="dot" else "solid"))
            )
            if label:
                fig.add_annotation(
                    xref="paper", x=1.01, yref=yref, y=y_val,
                    text=label, showarrow=False,
                    font=dict(size=9, color=color),
                    xanchor="left"
                )


def plot_violations(fig, dates, values, violations, name, row=None):
    """Plot data + highlight violations. row=None → plain Figure; row=int → subplot."""
    colors_map = {1: 'red', 2: 'orange', 3: 'purple', 4: 'magenta'}
    names_map  = {1: 'Rule1:3σ', 2: 'Rule2:8-run', 3: 'Rule3:Trend', 4: 'Rule4:ZoneA'}
    trace_kwargs = {} if row is None else {"row": row, "col": 1}

    fig.add_trace(go.Scatter(
        x=dates, y=values, mode='lines+markers', name=name,
        line=dict(color='#1e3d59', width=2), marker=dict(color='#1e3d59', size=6)
    ), **trace_kwargs)

    for rule, idxs in violations.items():
        if idxs:
            vd = [dates[i] for i in idxs if i < len(dates)]
            vv = [values[i] for i in idxs if i < len(values)]
            if vd:
                fig.add_trace(go.Scatter(
                    x=vd, y=vv, mode='markers', name=names_map[rule],
                    marker=dict(color=colors_map[rule], size=12, symbol='x', line=dict(width=2))
                ), **trace_kwargs)

# ===================================================================
# CHART RENDERERS
# ===================================================================
def render_p_chart(df, date_col, qty_col, ng_col):
    st.markdown('<div class="info-box">📌 <b>p-Chart</b> — Proporsi defect, sampel boleh bervariasi.</div>', unsafe_allow_html=True)
    daily = df.groupby(date_col)[[qty_col, ng_col]].sum().reset_index()
    daily = daily[daily[qty_col] > 0]
    if len(daily) < 3: st.warning("Minimal 3 periode."); return
    daily['p'] = daily[ng_col] / daily[qty_col]
    p_bar = daily[ng_col].sum() / daily[qty_col].sum()
    n_i   = daily[qty_col].values
    ucl_v = p_bar + 3*np.sqrt(p_bar*(1-p_bar)/n_i)
    lcl_v = np.maximum(0, p_bar - 3*np.sqrt(p_bar*(1-p_bar)/n_i))
    dates  = daily[date_col].tolist()
    p_vals = daily['p'].tolist()
    viol   = detect_violations(p_vals, np.mean(ucl_v), np.mean(lcl_v), p_bar)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates+dates[::-1], y=list(ucl_v)+list(lcl_v[::-1]),
        fill='toself', fillcolor='rgba(255,0,0,0.06)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=dates, y=ucl_v, mode='lines', line=dict(dash='dash',color='red',width=1.5), name='UCL'))
    fig.add_trace(go.Scatter(x=dates, y=lcl_v, mode='lines', line=dict(dash='dash',color='red',width=1.5), name='LCL'))
    fig.add_hline(y=p_bar, line_color='green', line_width=2, annotation_text=f"p̄={p_bar:.4f}")
    colors = ['red' if i in viol[1] else '#1e3d59' for i in range(len(p_vals))]
    fig.add_trace(go.Scatter(x=dates, y=p_vals, mode='lines+markers', name='p',
        line=dict(color='#1e3d59', width=2), marker=dict(color=colors, size=8)))
    for r,idxs in viol.items():
        if idxs:
            fig.add_trace(go.Scatter(x=[dates[i] for i in idxs if i<len(dates)],
                y=[p_vals[i] for i in idxs if i<len(p_vals)], mode='markers',
                name=f'Rule {r}', marker=dict(color={1:'red',2:'orange',3:'purple',4:'magenta'}[r], size=12, symbol='x', line=dict(width=2))))
    fig.update_layout(title="p-Chart (Proportion Defective)", height=420, yaxis_title="Proporsi Defect")
    st.plotly_chart(fig, use_container_width=True)
    show_violation_summary(viol)
    with st.expander("📊 Statistik"):
        st.write(f"p̄={p_bar:.4f} | Total inspeksi={daily[qty_col].sum():,} | Total NG={daily[ng_col].sum():,}")

def render_np_chart(df, date_col, qty_col, ng_col):
    st.markdown('<div class="info-box">📌 <b>np-Chart</b> — Jumlah defect, sampel sebaiknya tetap.</div>', unsafe_allow_html=True)
    daily = df.groupby(date_col)[[qty_col, ng_col]].sum().reset_index()
    daily = daily[daily[qty_col] > 0]
    if len(daily) < 3: st.warning("Minimal 3 periode."); return
    n_bar = daily[qty_col].mean()
    p_bar = daily[ng_col].sum() / daily[qty_col].sum()
    np_bar = n_bar * p_bar
    ucl_np = np_bar + 3*np.sqrt(np_bar*(1-p_bar))
    lcl_np = max(0, np_bar - 3*np.sqrt(np_bar*(1-p_bar)))
    cv = daily[qty_col].std() / daily[qty_col].mean()
    if cv > 0.10:
        st.markdown(f'<div class="warn-box">⚠️ CV ukuran sampel = {cv*100:.1f}%. Sampel tidak konstan — pertimbangkan p-Chart.</div>', unsafe_allow_html=True)
    dates  = daily[date_col].tolist()
    np_vals = daily[ng_col].tolist()
    viol   = detect_violations(np_vals, ucl_np, lcl_np, np_bar)
    fig = go.Figure()
    add_ctrl_lines(fig, dates, ucl_np, lcl_np, np_bar)
    plot_violations(fig, dates, np_vals, viol, "np")
    fig.update_layout(title="np-Chart (Number of Defective)", height=400, yaxis_title="Jumlah Defect")
    st.plotly_chart(fig, use_container_width=True)
    show_violation_summary(viol)

def render_c_chart(df, date_col, ng_col):
    st.markdown('<div class="info-box">📌 <b>c-Chart</b> — Jumlah defect per unit/area tetap.</div>', unsafe_allow_html=True)
    daily = df.groupby(date_col)[ng_col].sum().reset_index()
    if len(daily) < 3: st.warning("Minimal 3 periode."); return
    c_vals = daily[ng_col].tolist()
    c_bar  = np.mean(c_vals)
    ucl_c  = c_bar + 3*np.sqrt(c_bar)
    lcl_c  = max(0, c_bar - 3*np.sqrt(c_bar))
    dates  = daily[date_col].tolist()
    viol   = detect_violations(c_vals, ucl_c, lcl_c, c_bar)
    fig = go.Figure()
    add_ctrl_lines(fig, dates, ucl_c, lcl_c, c_bar)
    plot_violations(fig, dates, c_vals, viol, "c")
    fig.update_layout(title="c-Chart (Count of Defects)", height=400, yaxis_title="Jumlah Defect")
    st.plotly_chart(fig, use_container_width=True)
    show_violation_summary(viol)

def render_u_chart(df, date_col, qty_col, ng_col):
    st.markdown('<div class="info-box">📌 <b>u-Chart</b> — Defect per unit, area inspeksi boleh bervariasi.</div>', unsafe_allow_html=True)
    daily = df.groupby(date_col)[[qty_col, ng_col]].sum().reset_index()
    daily = daily[daily[qty_col] > 0]
    if len(daily) < 3: st.warning("Minimal 3 periode."); return
    daily['u'] = daily[ng_col] / daily[qty_col]
    u_bar  = daily[ng_col].sum() / daily[qty_col].sum()
    n_i    = daily[qty_col].values
    ucl_v  = u_bar + 3*np.sqrt(u_bar/n_i)
    lcl_v  = np.maximum(0, u_bar - 3*np.sqrt(u_bar/n_i))
    dates  = daily[date_col].tolist()
    u_vals = daily['u'].tolist()
    viol   = detect_violations(u_vals, float(np.mean(ucl_v)), float(np.mean(lcl_v)), u_bar)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates+dates[::-1], y=list(ucl_v)+list(lcl_v[::-1]),
        fill='toself', fillcolor='rgba(255,0,0,0.06)', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=dates, y=ucl_v, mode='lines', line=dict(dash='dash',color='red',width=1.5), name='UCL'))
    fig.add_trace(go.Scatter(x=dates, y=lcl_v, mode='lines', line=dict(dash='dash',color='red',width=1.5), name='LCL'))
    fig.add_hline(y=u_bar, line_color='green', line_width=2, annotation_text=f"ū={u_bar:.4f}")
    colors = ['red' if i in viol[1] else '#1e3d59' for i in range(len(u_vals))]
    fig.add_trace(go.Scatter(x=dates, y=u_vals, mode='lines+markers', name='u',
        line=dict(color='#1e3d59', width=2), marker=dict(color=colors, size=8)))
    fig.update_layout(title="u-Chart (Defects per Unit)", height=400, yaxis_title="Defect / Unit")
    st.plotly_chart(fig, use_container_width=True)
    show_violation_summary(viol)

def render_imr_chart(df, meas_col, date_col=None):
    st.markdown('<div class="info-box">📌 <b>I-MR Chart</b> — Data individual (n=1), satu pengukuran per titik.</div>', unsafe_allow_html=True)
    vals  = df[meas_col].dropna().values
    if len(vals) < 5: st.warning("Minimal 5 data."); return
    dates = df[date_col].values[:len(vals)] if date_col and date_col in df.columns else list(range(1, len(vals)+1))
    mr = np.abs(np.diff(vals)); mr_dates = dates[1:] if hasattr(dates,'__getitem__') else list(range(2, len(vals)+1))
    x_bar = np.mean(vals); mr_bar = np.mean(mr)
    E2=2.660; D4_mr=3.267; D3_mr=0.0; d2=1.128
    ucl_i=x_bar+E2*mr_bar; lcl_i=x_bar-E2*mr_bar
    ucl_mr=D4_mr*mr_bar;    lcl_mr=D3_mr*mr_bar
    viol_i  = detect_violations(list(vals), ucl_i, lcl_i, x_bar)
    viol_mr = detect_violations(list(mr),   ucl_mr, lcl_mr, mr_bar)
    fig = make_subplots(rows=2, cols=1, subplot_titles=["I Chart (Individual)", "MR Chart (Moving Range)"], vertical_spacing=0.12)
    add_ctrl_lines(fig, dates, ucl_i, lcl_i, x_bar, row=1)
    add_ctrl_lines(fig, mr_dates, ucl_mr, lcl_mr, mr_bar, row=2)
    plot_violations(fig, list(dates), list(vals), viol_i, "Individual", row=1)
    plot_violations(fig, list(mr_dates), list(mr), viol_mr, "MR", row=2)
    fig.update_layout(height=600, title_text=f"I-MR Chart — {meas_col}")
    st.plotly_chart(fig, use_container_width=True)
    c1,c2 = st.columns(2)
    with c1: show_violation_summary(viol_i)
    with c2: show_violation_summary(viol_mr)
    with st.expander("📊 Statistik"):
        st.write(f"X̄={x_bar:.4f} | MR̄={mr_bar:.4f} | σ̂={mr_bar/d2:.4f} | UCL_I={ucl_i:.4f} | LCL_I={lcl_i:.4f}")

def render_xbar_r_chart(df, meas_col, subg_col=None, n=5):
    st.markdown('<div class="info-box">📌 <b>X̄-R Chart</b> — Subgroup kecil (n=2–10), data variabel.</div>', unsafe_allow_html=True)
    if subg_col and subg_col in df.columns:
        grouped = df.groupby(subg_col)[meas_col]
        xbar = grouped.mean(); r_vals = grouped.apply(lambda x: x.max()-x.min())
        n_use = int(df.groupby(subg_col).size().mean()); dates = xbar.index.tolist()
    else:
        vals = df[meas_col].dropna().values; n_use = int(n)
        num_g = len(vals)//n_use
        if num_g < 3: st.warning("Tidak cukup data subgroup."); return
        g = vals[:num_g*n_use].reshape(num_g, n_use)
        xbar = pd.Series(g.mean(axis=1)); r_vals = pd.Series(g.max(axis=1)-g.min(axis=1))
        dates = list(range(1, num_g+1))
    A2,D3,D4,A3,B3,B4,d2 = get_spc(n_use)
    r_bar=r_vals.mean(); xbar_bar=xbar.mean()
    ucl_x=xbar_bar+A2*r_bar; lcl_x=xbar_bar-A2*r_bar
    ucl_r=D4*r_bar;           lcl_r=D3*r_bar
    viol_x = detect_violations(list(xbar), ucl_x, lcl_x, xbar_bar)
    viol_r = detect_violations(list(r_vals), ucl_r, lcl_r, r_bar)
    fig = make_subplots(rows=2, cols=1, subplot_titles=["X̄ Chart", "R Chart"], vertical_spacing=0.12)
    add_ctrl_lines(fig, dates, ucl_x, lcl_x, xbar_bar, row=1)
    add_ctrl_lines(fig, dates, ucl_r, lcl_r, r_bar, row=2)
    plot_violations(fig, dates, list(xbar), viol_x, "X̄", row=1)
    plot_violations(fig, dates, list(r_vals), viol_r, "R", row=2)
    fig.update_layout(height=600, title_text=f"X̄-R Chart — {meas_col}")
    st.plotly_chart(fig, use_container_width=True)
    c1,c2 = st.columns(2)
    with c1: show_violation_summary(viol_x)
    with c2: show_violation_summary(viol_r)

def render_xbar_s_chart(df, meas_col, subg_col=None, n=10):
    st.markdown('<div class="info-box">📌 <b>X̄-S Chart</b> — Subgroup besar (n≥8), presisi lebih tinggi.</div>', unsafe_allow_html=True)
    if subg_col and subg_col in df.columns:
        grouped = df.groupby(subg_col)[meas_col]
        xbar = grouped.mean(); s_vals = grouped.std(ddof=1)
        n_use = int(df.groupby(subg_col).size().mean()); dates = xbar.index.tolist()
    else:
        vals = df[meas_col].dropna().values; n_use = int(n)
        num_g = len(vals)//n_use
        if num_g < 3: st.warning("Tidak cukup data subgroup."); return
        g = vals[:num_g*n_use].reshape(num_g, n_use)
        xbar = pd.Series(g.mean(axis=1)); s_vals = pd.Series(g.std(axis=1, ddof=1))
        dates = list(range(1, num_g+1))
    A2,D3,D4,A3,B3,B4,d2 = get_spc(n_use)
    s_bar=s_vals.mean(); xbar_bar=xbar.mean()
    ucl_x=xbar_bar+A3*s_bar; lcl_x=xbar_bar-A3*s_bar
    ucl_s=B4*s_bar;           lcl_s=B3*s_bar
    viol_x = detect_violations(list(xbar), ucl_x, lcl_x, xbar_bar)
    viol_s = detect_violations(list(s_vals), ucl_s, lcl_s, s_bar)
    fig = make_subplots(rows=2, cols=1, subplot_titles=["X̄ Chart", "S Chart"], vertical_spacing=0.12)
    add_ctrl_lines(fig, dates, ucl_x, lcl_x, xbar_bar, row=1)
    add_ctrl_lines(fig, dates, ucl_s, lcl_s, s_bar, row=2)
    plot_violations(fig, dates, list(xbar), viol_x, "X̄", row=1)
    plot_violations(fig, dates, list(s_vals), viol_s, "S", row=2)
    fig.update_layout(height=600, title_text=f"X̄-S Chart — {meas_col}")
    st.plotly_chart(fig, use_container_width=True)

# ===================================================================
# COLUMN MAPPING UI — interaktif, user bisa koreksi
# ===================================================================
def show_mapping_ui(df, detected, mapping):
    """
    Tampilkan hasil deteksi + beri opsi user untuk koreksi manual.
    Return mapping yang sudah dikoreksi.
    """
    st.markdown("### 🔬 Hasil Deteksi Kolom Otomatis")
    st.markdown('<div class="info-box">Sistem mendeteksi peran setiap kolom secara otomatis. Koreksi jika ada yang salah.</div>', unsafe_allow_html=True)

    role_labels = {
        "date": ("🗓️ Tanggal/Waktu", "tag-date"),
        "sample_size": ("🔢 Ukuran Sampel (n)", "tag-attr"),
        "defect_count": ("❌ Jumlah NG/Defect", "tag-attr"),
        "defect_type": ("🏷️ Jenis Defect", "tag-cat"),
        "measurement": ("📏 Pengukuran Variabel", "tag-var"),
        "line": ("🏭 Line/Mesin/Shift", "tag-cat"),
        "product": ("📦 Produk/Tipe", "tag-cat"),
        "size_variant": ("📐 Ukuran/Varian", "tag-cat"),
        "categorical": ("🔤 Kategori Lain", "tag-cat"),
        "numeric_count": ("🔢 Count Numerik", "tag-attr"),
        "unknown": ("❓ Tidak Dikenali", "tag-cat"),
    }

    col_options = ["— (tidak dipakai)"] + list(df.columns)
    num_options  = ["— (tidak dipakai)"] + [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    date_options = ["— (tidak dipakai)"] + list(df.columns)

    # Tabel ringkasan
    rows_html = ""
    for col, role in detected.items():
        label, tag = role_labels.get(role, ("❓", "tag-cat"))
        sample_vals = str(df[col].dropna().head(3).tolist())[:60]
        rows_html += f"<tr><td><b>{col}</b></td><td><span class='{tag}'>{label}</span></td><td style='color:#888;font-size:.8em'>{df[col].dtype}</td><td style='color:#888;font-size:.8em'>{sample_vals}</td></tr>"

    st.markdown(f"""
    <table style='width:100%;border-collapse:collapse;font-size:.88em'>
    <tr style='background:#f0f4f8'><th align='left' style='padding:6px'>Kolom</th><th align='left' style='padding:6px'>Peran Terdeteksi</th><th style='padding:6px'>Dtype</th><th style='padding:6px'>Contoh Nilai</th></tr>
    {rows_html}
    </table>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ✏️ Koreksi Manual (opsional)")
    st.caption("Ubah jika sistem salah mendeteksi kolom.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        def_date = mapping["date"] if mapping["date"] in df.columns else col_options[0]
        mapping["date"] = st.selectbox("🗓️ Kolom Tanggal", col_options, index=col_options.index(def_date) if def_date in col_options else 0, key="map_date")
        if mapping["date"] == "— (tidak dipakai)": mapping["date"] = None

    with c2:
        def_n = mapping["sample_size"] if mapping["sample_size"] in df.columns else num_options[0]
        mapping["sample_size"] = st.selectbox("🔢 Kolom Sample Size (n)", num_options, index=num_options.index(def_n) if def_n in num_options else 0, key="map_n")
        if mapping["sample_size"] == "— (tidak dipakai)": mapping["sample_size"] = None

    with c3:
        def_ng = mapping["defect_count"] if mapping["defect_count"] in df.columns else num_options[0]
        mapping["defect_count"] = st.selectbox("❌ Kolom Jumlah NG/Defect", num_options, index=num_options.index(def_ng) if def_ng in num_options else 0, key="map_ng")
        if mapping["defect_count"] == "— (tidak dipakai)": mapping["defect_count"] = None

    with c4:
        def_dt = mapping["defect_type"] if mapping["defect_type"] in df.columns else col_options[0]
        mapping["defect_type"] = st.selectbox("🏷️ Kolom Jenis Defect", col_options, index=col_options.index(def_dt) if def_dt in col_options else 0, key="map_dt")
        if mapping["defect_type"] == "— (tidak dipakai)": mapping["defect_type"] = None

    # Measurement cols
    avail_meas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    default_meas = [m for m in mapping["measurement"] if m in avail_meas]
    mapping["measurement"] = st.multiselect("📏 Kolom Pengukuran Variabel (bisa pilih banyak)", avail_meas, default=default_meas, key="map_meas")

    # Stratification cols
    cat_candidates = [c for c in df.columns if df[c].nunique() < 50 and c not in [mapping["date"], mapping["sample_size"], mapping["defect_count"]]]
    default_strat = [c for c in [mapping["line"], mapping["product"], mapping["size_variant"]] if c and c in cat_candidates]
    strat_cols = st.multiselect("🏭 Kolom Stratifikasi (Line, Shift, Produk, dll)", cat_candidates, default=default_strat, key="map_strat")
    mapping["strat_cols"] = strat_cols

    return mapping

# ===================================================================
# MAIN DASHBOARD
# ===================================================================
st.sidebar.title("Quality Assurance Dashboard")
st.sidebar.write("Upload File Data Produksi:")
st.sidebar.markdown('<div class="info-box" style="font-size:.82em">✅ <b>Format bebas!</b> Sistem otomatis mendeteksi kolom. Tidak perlu format khusus.<br><br>Mendukung: <b>xlsx, xls, csv, tsv</b></div>', unsafe_allow_html=True)

uploaded_files = st.sidebar.file_uploader(
    "Drop file di sini (bisa lebih dari satu)",
    type=["xlsx", "xls", "csv", "tsv"],
    accept_multiple_files=True
)
st.sidebar.markdown("---")
st.sidebar.info("Sistem cerdas: deteksi kolom otomatis, tidak bergantung nama kolom tertentu.")

# ============================================================
if uploaded_files:
    try:
        # BACA SEMUA FILE
        all_df = []
        for file in uploaded_files:
            ext = file.name.lower().split(".")[-1]
            if ext in ("xlsx", "xlsm"):
                tmp = pd.read_excel(file)
            elif ext == "xls":
                tmp = pd.read_excel(file, engine="xlrd")
            elif ext == "csv":
                tmp = pd.read_csv(file)
            elif ext == "tsv":
                tmp = pd.read_csv(file, sep="\t")
            else:
                tmp = pd.read_csv(file)
            tmp["_source_file"] = file.name
            all_df.append(tmp)

        df_raw = pd.concat(all_df, ignore_index=True)

        # AUTO PARSE DATE
        for col in df_raw.columns:
            if df_raw[col].dtype == object:
                try:
                    parsed = pd.to_datetime(df_raw[col], infer_datetime_format=True, errors='raise')
                    df_raw[col] = parsed
                except: pass

        # DETEKSI KOLOM
        detected = auto_detect_all_columns(df_raw)
        mapping  = resolve_mapping(df_raw, detected)
        ds_type  = classify_dataset_type(mapping)

        # HEADER
        st.title("🏭 Production Quality Dashboard")
        badge_color = {"attribute":"#1565c0","variable":"#6a1b9a","mixed":"#2e7d32","unknown":"#616161"}.get(ds_type,"#616161")
        badge_label = {"attribute":"📊 Data Atribut","variable":"📏 Data Variabel","mixed":"🔀 Mixed (Atribut + Variabel)","unknown":"❓ Tipe Tidak Terdeteksi"}.get(ds_type,"❓")
        st.markdown(f'<span style="background:{badge_color};color:white;padding:4px 14px;border-radius:12px;font-size:.9em;font-weight:600">{badge_label}</span>&nbsp;&nbsp;<span style="color:#888;font-size:.85em">{len(df_raw):,} baris • {len(df_raw.columns)} kolom • {len(uploaded_files)} file</span>', unsafe_allow_html=True)
        st.divider()

        # COLUMN MAPPING UI
        with st.expander("🔬 Konfigurasi Kolom (Auto-Detect + Koreksi Manual)", expanded=(ds_type=="unknown")):
            mapping = show_mapping_ui(df_raw, detected, mapping)

        # FILTER GLOBAL
        df = df_raw.copy()
        filter_cols = mapping.get("strat_cols", [])
        if filter_cols:
            with st.expander("🔎 Global Filter"):
                fcols = st.columns(min(len(filter_cols), 4))
                for i, fc in enumerate(filter_cols):
                    with fcols[i % 4]:
                        opts = ["All"] + sorted(df[fc].dropna().astype(str).unique().tolist())
                        sel  = st.selectbox(f"Filter {fc}:", opts, key=f"filt_{fc}")
                        if sel != "All":
                            df = df[df[fc].astype(str) == sel]

        # Tambah kolom Bulan jika ada tanggal
        if mapping["date"]:
            try:
                df[mapping["date"]] = pd.to_datetime(df[mapping["date"]])
                df["_Bulan"] = df[mapping["date"]].dt.strftime("%b %Y")
                # Filter bulan
                bulan_opts = ["All"] + sorted(df["_Bulan"].unique(), key=lambda x: pd.to_datetime(x, format="%b %Y"))
                sel_bulan  = st.sidebar.selectbox("📅 Filter Bulan:", bulan_opts)
                if sel_bulan != "All":
                    df = df[df["_Bulan"] == sel_bulan]
            except: pass

        st.divider()

        # ─── METRIK RINGKASAN ───
        has_attr = mapping["sample_size"] and mapping["defect_count"]
        if has_attr:
            total_n   = df[mapping["sample_size"]].sum()
            total_ng  = df[mapping["defect_count"]].sum()
            defect_r  = (total_ng / total_n * 100) if total_n > 0 else 0
            yield_v   = 1 - (total_ng / total_n) if total_n > 0 else 0
            sigma_lvl = (norm.ppf(yield_v) + 1.5) if 0 < yield_v < 1 else (6.0 if yield_v >= 1 else 0.0)

            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Total Produksi", f"{total_n:,.0f}")
            k2.metric("Total NG", f"{total_ng:,.0f}", delta_color="inverse")
            k3.metric("Defect Rate", f"{defect_r:.2f}%", delta_color="inverse")
            k4.metric("Sigma Level", f"{sigma_lvl:.2f}", delta="Target: 4.0")
            st.divider()

        # ─── TABS ───
        tab_list = ["📊 Pareto & Distribusi", "🔍 Stratifikasi", "📈 Control Chart", "🔗 Korelasi", "📥 Raw Data"]
        tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

        # ── TAB 1: PARETO ──
        with tab1:
            st.subheader("Pareto & Distribusi Defect")
            if mapping["defect_type"] and mapping["defect_count"]:
                pareto = df.groupby(mapping["defect_type"])[mapping["defect_count"]].sum().sort_values(ascending=False).reset_index()
                c1, c2 = st.columns([2,1])
                with c1:
                    # Pareto dengan garis kumulatif
                    pareto["cumulative_pct"] = pareto[mapping["defect_count"]].cumsum() / pareto[mapping["defect_count"]].sum() * 100
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(x=pareto[mapping["defect_type"]], y=pareto[mapping["defect_count"]],
                        name="Frekuensi", marker_color="#1e3d59", text=pareto[mapping["defect_count"]], textposition='outside'))
                    fig_bar.add_trace(go.Scatter(x=pareto[mapping["defect_type"]], y=pareto["cumulative_pct"],
                        mode='lines+markers', name="Kumulatif %", yaxis="y2",
                        line=dict(color="red", width=2), marker=dict(size=8)))
                    fig_bar.add_hline(y=80, line_dash="dot", line_color="orange",
                                      annotation_text="80% (Pareto)", yref="y2")
                    fig_bar.update_layout(
                        title="Pareto Chart — Frekuensi Defect",
                        yaxis=dict(title="Jumlah NG"),
                        yaxis2=dict(title="Kumulatif %", overlaying="y", side="right", range=[0,105]),
                        height=420
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                with c2:
                    fig_pie = px.pie(pareto, values=mapping["defect_count"], names=mapping["defect_type"],
                                     hole=0.4, title="Distribusi Defect")
                    st.plotly_chart(fig_pie, use_container_width=True)
            elif mapping["defect_count"]:
                st.info("Kolom Jenis Defect tidak terdeteksi. Menampilkan distribusi total NG per periode.")
                if mapping["date"]:
                    daily = df.groupby(mapping["date"])[mapping["defect_count"]].sum().reset_index()
                    fig_ts = px.bar(daily, x=mapping["date"], y=mapping["defect_count"], title="Total NG per Periode")
                    st.plotly_chart(fig_ts, use_container_width=True)
            elif mapping["measurement"]:
                sel_m = st.selectbox("Pilih kolom untuk histogram:", mapping["measurement"])
                fig_h = px.histogram(df, x=sel_m, nbins=30, marginal="box", title=f"Distribusi {sel_m}")
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.warning("Tidak cukup kolom terdeteksi untuk Pareto. Silakan koreksi mapping di bagian atas.")

        # ── TAB 2: STRATIFIKASI ──
        with tab2:
            st.subheader("Stratifikasi & Drill-Down")
            strat_avail = mapping.get("strat_cols", [])
            if mapping["defect_type"]: strat_avail = list(set(strat_avail + [mapping["defect_type"]]))
            if "_Bulan" in df.columns: strat_avail = list(set(strat_avail + ["_Bulan"]))

            if strat_avail and mapping["defect_count"]:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### Sunburst Drill-Down")
                    sel_path = st.multiselect("Layer (urutan hierarki):", strat_avail,
                                              default=strat_avail[:min(2, len(strat_avail))], key="sun_path")
                    if sel_path:
                        fig_sun = px.sunburst(df, path=sel_path, values=mapping["defect_count"],
                                              color=mapping["defect_count"], color_continuous_scale='Reds')
                        st.plotly_chart(fig_sun, use_container_width=True)
                with c2:
                    st.markdown("##### Heatmap Defect")
                    if len(strat_avail) >= 2:
                        row_col = st.selectbox("Baris:", strat_avail, index=0, key="hm_row")
                        col_col = st.selectbox("Kolom:", strat_avail, index=min(1, len(strat_avail)-1), key="hm_col")
                        if row_col != col_col:
                            hm = df.groupby([row_col, col_col])[mapping["defect_count"]].sum().reset_index()
                            hm = hm.pivot(index=row_col, columns=col_col, values=mapping["defect_count"]).fillna(0)
                            fig_hm = px.imshow(hm, text_auto=True, color_continuous_scale="Reds")
                            st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Tidak cukup kolom kategori atau defect untuk stratifikasi.")

        # ── TAB 3: CONTROL CHART ──
        with tab3:
            st.subheader("Statistical Process Control (SPC)")
            ds_type_now = classify_dataset_type(mapping)

            cc_mode = st.radio("Mode:", ["🤖 Auto-Recommend", "🎯 Manual"], horizontal=True)

            st.divider()

            CHART_GUIDE = """
| Data | Kondisi | Chart |
|---|---|---|
| Atribut (NG/OK) | Sampel bervariasi | **p-Chart** |
| Atribut (NG/OK) | Sampel tetap | **np-Chart** |
| Atribut (jumlah cacat) | Area tetap | **c-Chart** |
| Atribut (jumlah cacat) | Area variasi | **u-Chart** |
| Variabel (ukuran, berat) | n=1 per titik | **I-MR Chart** |
| Variabel | Subgroup n=2–8 | **X̄-R Chart** |
| Variabel | Subgroup n≥8 | **X̄-S Chart** |
"""
            with st.expander("📖 Panduan Pemilihan Chart"):
                st.markdown(CHART_GUIDE)

            if cc_mode == "🤖 Auto-Recommend":
                if has_attr and mapping["date"]:
                    sample_cv = df.groupby(mapping["date"])[mapping["sample_size"]].sum().std() / \
                                df.groupby(mapping["date"])[mapping["sample_size"]].sum().mean()
                    st.markdown("#### p-Chart (Rekomendasi Utama)")
                    render_p_chart(df, mapping["date"], mapping["sample_size"], mapping["defect_count"])
                    st.markdown("---")
                    if sample_cv < 0.10:
                        st.markdown("#### np-Chart")
                        render_np_chart(df, mapping["date"], mapping["sample_size"], mapping["defect_count"])
                        st.markdown("---")
                    st.markdown("#### c-Chart")
                    render_c_chart(df, mapping["date"], mapping["defect_count"])
                    st.markdown("---")
                    st.markdown("#### u-Chart")
                    render_u_chart(df, mapping["date"], mapping["sample_size"], mapping["defect_count"])

                if mapping["measurement"] and mapping["date"]:
                    st.markdown("---")
                    for m_col in mapping["measurement"][:3]:
                        st.markdown(f"#### I-MR Chart — {m_col}")
                        render_imr_chart(df, m_col, mapping["date"])
                        st.markdown("---")

                if not has_attr and not mapping["measurement"]:
                    st.warning("⚠️ Tidak cukup kolom terdeteksi untuk membuat control chart. Silakan koreksi mapping kolom di atas.")

            else:  # Manual
                CHART_OPTIONS = {
                    "p-Chart (Proporsi Defect)": "p",
                    "np-Chart (Jumlah Defect, sampel tetap)": "np",
                    "c-Chart (Count Defect, unit tetap)": "c",
                    "u-Chart (Defect/Unit, unit variabel)": "u",
                    "I-MR Chart (Individual, n=1)": "imr",
                    "X̄-R Chart (Mean & Range, subgroup kecil)": "xbar_r",
                    "X̄-S Chart (Mean & StdDev, subgroup besar)": "xbar_s",
                }
                sel = st.selectbox("Pilih jenis chart:", list(CHART_OPTIONS.keys()))
                ct  = CHART_OPTIONS[sel]

                num_cols  = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                all_cols  = list(df.columns)
                num_opts  = ["— (tidak dipakai)"] + num_cols
                all_opts  = ["— (tidak dipakai)"] + all_cols

                def pick(label, opts, default):
                    idx = opts.index(default) if default in opts else 0
                    return st.selectbox(label, opts, index=idx)

                d_col = mapping["date"]
                n_col = mapping["sample_size"]
                ng_col = mapping["defect_count"]
                m_cols = mapping["measurement"]

                if ct in ("p","np","c","u"):
                    c1,c2,c3 = st.columns(3)
                    with c1: d_col  = pick("🗓️ Kolom Tanggal:", all_opts, d_col or all_opts[0])
                    with c2: n_col  = pick("🔢 Sample Size (n):", num_opts, n_col or num_opts[0])
                    with c3: ng_col = pick("❌ Jumlah NG:", num_opts, ng_col or num_opts[0])
                    d_col  = None if d_col  == "— (tidak dipakai)" else d_col
                    n_col  = None if n_col  == "— (tidak dipakai)" else n_col
                    ng_col = None if ng_col == "— (tidak dipakai)" else ng_col

                if ct in ("imr","xbar_r","xbar_s"):
                    m_sel = st.selectbox("📏 Kolom Pengukuran:", num_cols, index=0 if not m_cols else num_cols.index(m_cols[0]) if m_cols[0] in num_cols else 0)
                    d_col_sel = st.selectbox("🗓️ Kolom Tanggal/Urutan:", all_opts, index=all_opts.index(d_col) if d_col in all_opts else 0)
                    d_col = None if d_col_sel == "— (tidak dipakai)" else d_col_sel

                if st.button("▶ Render Chart", type="primary"):
                    if ct == "p" and d_col and n_col and ng_col:
                        render_p_chart(df, d_col, n_col, ng_col)
                    elif ct == "np" and d_col and n_col and ng_col:
                        render_np_chart(df, d_col, n_col, ng_col)
                    elif ct == "c" and d_col and ng_col:
                        render_c_chart(df, d_col, ng_col)
                    elif ct == "u" and d_col and n_col and ng_col:
                        render_u_chart(df, d_col, n_col, ng_col)
                    elif ct == "imr":
                        render_imr_chart(df, m_sel, d_col)
                    elif ct == "xbar_r":
                        n_sg = st.session_state.get("sg_size", 5)
                        render_xbar_r_chart(df, m_sel, None, n_sg)
                    elif ct == "xbar_s":
                        n_sg = st.session_state.get("sg_size_s", 10)
                        render_xbar_s_chart(df, m_sel, None, n_sg)
                    else:
                        st.warning("Pilih semua kolom yang diperlukan terlebih dahulu.")

        # ── TAB 4: KORELASI ──
        with tab4:
            st.subheader("Analisis Korelasi & Scatter")
            num_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "_source_file"]
            if len(num_cols_all) >= 2:
                c1, c2 = st.columns(2)
                with c1:
                    x_col = st.selectbox("Sumbu X:", num_cols_all, index=0)
                with c2:
                    y_col = st.selectbox("Sumbu Y:", num_cols_all, index=min(1, len(num_cols_all)-1))

                color_col = None
                if mapping.get("strat_cols"):
                    color_col = st.selectbox("Warna (stratifikasi):", ["— (tidak ada)"] + mapping.get("strat_cols",[]))
                    if color_col == "— (tidak ada)": color_col = None

                if x_col != y_col:
                    corr = df[x_col].corr(df[y_col])
                    fig_sc = px.scatter(df, x=x_col, y=y_col, color=color_col,
                                        trendline="ols", trendline_scope="overall",
                                        trendline_color_override="red",
                                        title=f"Scatter: {x_col} vs {y_col}")
                    st.plotly_chart(fig_sc, use_container_width=True)
                    st.markdown(f"**Koefisien Korelasi (r):** `{corr:.3f}` — {'Korelasi kuat' if abs(corr)>0.7 else 'Korelasi sedang' if abs(corr)>0.3 else 'Korelasi lemah'}")

                # Correlation matrix
                if len(num_cols_all) > 2:
                    with st.expander("📊 Correlation Matrix"):
                        corr_mat = df[num_cols_all].corr()
                        fig_cm = px.imshow(corr_mat, text_auto=".2f", color_continuous_scale="RdBu_r",
                                           zmin=-1, zmax=1, title="Correlation Matrix")
                        st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Minimal 2 kolom numerik diperlukan untuk analisis korelasi.")

        # ── TAB 5: RAW DATA ──
        with tab5:
            st.subheader("Raw Data")
            st.caption(f"{len(df):,} baris × {len(df.columns)} kolom")
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download sebagai CSV", csv, "filtered_data.csv", "text/csv")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
        st.exception(e)

else:
    # LANDING PAGE
    st.title("🏭 Production Quality Dashboard")
    st.markdown("""
    <div style="background:white;border-radius:12px;padding:32px;box-shadow:0 4px 12px rgba(0,0,0,0.08);max-width:700px">
    <h3 style="color:#1e3d59;margin-top:0">📂 Upload File Produksi Anda</h3>
    <p style="color:#555">Dashboard ini <b>tidak memerlukan format kolom tertentu</b>. Sistem otomatis mendeteksi peran setiap kolom.</p>
    <hr>
    <b>Tipe data yang didukung:</b>
    <ul>
    <li>📊 <b>Data Atribut</b> — kolom qty check + qty NG → p, np, c, u Chart</li>
    <li>📏 <b>Data Variabel</b> — kolom pengukuran (diameter, berat, suhu) → I-MR, X̄-R, X̄-S Chart</li>
    <li>🔀 <b>Mixed</b> — keduanya dalam satu file</li>
    </ul>
    <b>Format file:</b> .xlsx, .xls, .csv, .tsv<br><br>
    <b>Contoh nama kolom yang dikenali otomatis:</b><br>
    <code>Tanggal, Date, qty check, qty ng, Jenis_Defect, diameter, berat, Line, Shift, ...</code><br>
    dan ratusan variasi lainnya — termasuk nama dalam Bahasa Indonesia maupun Inggris.
    </div>
    """, unsafe_allow_html=True)
