import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# -------------------------------------------------------------------
# 1. KONFIGURASI TAMPILAN
# -------------------------------------------------------------------
st.set_page_config(page_title="QA System Visualization", layout="wide", page_icon="🏭")

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
    .block-container { padding-top: 1rem !important; }
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    h1 { color: #1e3d59; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border-left: 5px solid #1e3d59;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 10px;
    }
    .chart-info-box {
        background-color: #e8f4fd;
        border-left: 4px solid #2196F3;
        padding: 10px 15px;
        border-radius: 4px;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    .violation-box {
        background-color: #fdecea;
        border-left: 4px solid #f44336;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 4px 0;
        font-size: 0.85em;
    }
    .ok-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# KONSTANTA FAKTOR CONTROL CHART (ISO / AIAG SPC Standard)
# -------------------------------------------------------------------
# Faktor untuk X̄-R dan X̄-S chart berdasarkan ukuran subgroup n
SPC_FACTORS = {
    #  n:   A2,    D3,    D4,    A3,    B3,    B4,    d2
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

def get_spc_factors(n):
    """Ambil faktor SPC, clamp ke n=2 atau n=10 jika di luar range."""
    n = max(2, min(10, int(n)))
    return SPC_FACTORS[n]

# -------------------------------------------------------------------
# FUNGSI HELPER: DETEKSI VIOLATION (Western Electric Rules)
# -------------------------------------------------------------------
def detect_violations(values, ucl, lcl, cl):
    """
    Mendeteksi pelanggaran Western Electric Rules:
    Rule 1: 1 titik di luar 3σ (UCL/LCL)
    Rule 2: 8 titik berturut-turut di satu sisi CL
    Rule 3: 6 titik berturut-turut naik atau turun
    Rule 4: 2 dari 3 titik di zona A (>2σ)
    """
    n = len(values)
    sigma = (ucl - cl) / 3 if (ucl - cl) != 0 else 1
    violations = {1: [], 2: [], 3: [], 4: []}

    for i, v in enumerate(values):
        # Rule 1
        if v > ucl or v < lcl:
            violations[1].append(i)

    # Rule 2: 8 titik berturut di satu sisi
    for i in range(7, n):
        window = values[i-7:i+1]
        if all(w > cl for w in window) or all(w < cl for w in window):
            violations[2].append(i)

    # Rule 3: 6 titik berturut naik atau turun
    for i in range(5, n):
        window = values[i-5:i+1]
        diffs = [window[j+1] - window[j] for j in range(5)]
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            violations[3].append(i)

    # Rule 4: 2 dari 3 titik di zona A (> 2σ dari CL)
    for i in range(2, n):
        window = values[i-2:i+1]
        count_zone_a = sum(1 for w in window if abs(w - cl) > 2 * sigma)
        if count_zone_a >= 2:
            violations[4].append(i)

    return violations

def add_control_lines(fig, dates, ucl, lcl, cl, ucl2=None, lcl2=None, row=1):
    """Tambahkan garis UCL, LCL, CL, dan zona 1σ/2σ ke figure."""
    sigma = (ucl - cl) / 3 if (ucl - cl) != 0 else 0

    # Zone fills
    fig.add_trace(go.Scatter(x=list(dates) + list(dates)[::-1],
        y=[ucl]*len(dates) + [cl + 2*sigma]*len(dates),
        fill='toself', fillcolor='rgba(255,0,0,0.05)', line=dict(width=0),
        showlegend=False, hoverinfo='skip'), row=row, col=1)
    fig.add_trace(go.Scatter(x=list(dates) + list(dates)[::-1],
        y=[lcl]*len(dates) + [max(0, cl - 2*sigma)]*len(dates),
        fill='toself', fillcolor='rgba(255,0,0,0.05)', line=dict(width=0),
        showlegend=False, hoverinfo='skip'), row=row, col=1)

    # Lines
    fig.add_hline(y=ucl, line_dash="dash", line_color="red", line_width=1.5,
                  annotation_text=f"UCL={ucl:.4f}", annotation_position="top right", row=row, col=1)
    fig.add_hline(y=lcl, line_dash="dash", line_color="red", line_width=1.5,
                  annotation_text=f"LCL={lcl:.4f}", annotation_position="bottom right", row=row, col=1)
    fig.add_hline(y=cl, line_color="green", line_width=2,
                  annotation_text=f"CL={cl:.4f}", annotation_position="top left", row=row, col=1)
    # 1σ dan 2σ lines (tipis)
    for mult, color in [(1, "rgba(0,128,0,0.3)"), (2, "rgba(255,165,0,0.4)")]:
        fig.add_hline(y=cl + mult*sigma, line_dash="dot", line_color=color, line_width=1, row=row, col=1)
        if cl - mult*sigma >= 0:
            fig.add_hline(y=cl - mult*sigma, line_dash="dot", line_color=color, line_width=1, row=row, col=1)

def plot_with_violations(fig, dates, values, violations, series_name, row=1):
    """Plot data dengan highlight titik-titik yang melanggar."""
    # Data normal
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode='lines+markers',
        name=series_name,
        line=dict(color='#1e3d59', width=2),
        marker=dict(color='#1e3d59', size=6)
    ), row=row, col=1)

    # Highlight violations
    all_viol_idx = set()
    viol_colors = {1: 'red', 2: 'orange', 3: 'purple', 4: 'magenta'}
    viol_names = {1: 'Rule 1: Beyond 3σ', 2: 'Rule 2: 8 sisi CL', 3: 'Rule 3: Trend', 4: 'Rule 4: Zona A'}

    for rule, idxs in violations.items():
        if idxs:
            viol_dates = [dates[i] for i in idxs if i < len(dates)]
            viol_vals  = [values[i] for i in idxs if i < len(values)]
            if viol_dates:
                fig.add_trace(go.Scatter(
                    x=viol_dates, y=viol_vals,
                    mode='markers',
                    name=viol_names[rule],
                    marker=dict(color=viol_colors[rule], size=12, symbol='x', line=dict(width=2))
                ), row=row, col=1)
            all_viol_idx.update(idxs)

    return all_viol_idx

def show_violation_summary(violations):
    """Tampilkan ringkasan pelanggaran dalam format yang rapi."""
    viol_names = {
        1: "Rule 1 (Beyond 3σ UCL/LCL)",
        2: "Rule 2 (8 titik berturut satu sisi CL)",
        3: "Rule 3 (6 titik trend naik/turun)",
        4: "Rule 4 (2 dari 3 di Zona A)"
    }
    has_viol = any(v for v in violations.values())
    if not has_viol:
        st.markdown('<div class="ok-box">✅ <strong>Proses In-Control:</strong> Tidak ada pelanggaran Western Electric Rules terdeteksi.</div>', unsafe_allow_html=True)
    else:
        for rule, idxs in violations.items():
            if idxs:
                st.markdown(f'<div class="violation-box">⚠️ <strong>{viol_names[rule]}</strong> — Terdeteksi pada {len(idxs)} titik: indeks {idxs[:10]}{"..." if len(idxs)>10 else ""}</div>', unsafe_allow_html=True)

# ===================================================================
# FUNGSI RENDER SETIAP JENIS CONTROL CHART
# ===================================================================

def render_xbar_r_chart(df, measurement_col, subgroup_col=None, subgroup_size=5):
    """X̄-R Chart: Rata-rata dan Range untuk data variabel."""
    st.markdown('<div class="chart-info-box">📌 <strong>X̄-R Chart</strong> — Digunakan untuk data variabel (pengukuran kontinu) dengan ukuran subgroup kecil (n = 2–10). Memantau rata-rata (posisi) dan range (dispersi) proses.</div>', unsafe_allow_html=True)

    try:
        if subgroup_col and subgroup_col in df.columns:
            grouped = df.groupby(subgroup_col)[measurement_col]
            xbar = grouped.mean()
            r_vals = grouped.apply(lambda x: x.max() - x.min())
            n = int(df.groupby(subgroup_col).size().mean())
            dates = xbar.index.tolist()
        else:
            values = df[measurement_col].dropna().values
            n = int(subgroup_size)
            num_groups = len(values) // n
            if num_groups < 3:
                st.warning("⚠️ Tidak cukup data untuk membentuk subgroup. Minimal 3 subgroup diperlukan.")
                return
            values = values[:num_groups * n]
            groups = values.reshape(num_groups, n)
            xbar = pd.Series(groups.mean(axis=1))
            r_vals = pd.Series(groups.max(axis=1) - groups.min(axis=1))
            dates = list(range(1, num_groups + 1))
            n = n

        A2, D3, D4, A3, B3, B4, d2 = get_spc_factors(n)
        r_bar = r_vals.mean()
        xbar_bar = xbar.mean()

        ucl_x = xbar_bar + A2 * r_bar
        lcl_x = xbar_bar - A2 * r_bar
        ucl_r = D4 * r_bar
        lcl_r = D3 * r_bar

        fig = make_subplots(rows=2, cols=1, subplot_titles=["X̄ Chart (Mean)", "R Chart (Range)"],
                            vertical_spacing=0.12)

        viol_x = detect_violations(list(xbar), ucl_x, lcl_x, xbar_bar)
        viol_r = detect_violations(list(r_vals), ucl_r, lcl_r, r_bar)

        add_control_lines(fig, dates, ucl_x, lcl_x, xbar_bar, row=1)
        add_control_lines(fig, dates, ucl_r, lcl_r, r_bar, row=2)
        plot_with_violations(fig, dates, list(xbar), viol_x, "X̄", row=1)
        plot_with_violations(fig, dates, list(r_vals), viol_r, "Range", row=2)

        fig.update_layout(height=600, title_text="X̄-R Control Chart", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Violation X̄ Chart:**")
            show_violation_summary(viol_x)
        with col2:
            st.markdown("**Violation R Chart:**")
            show_violation_summary(viol_r)

        with st.expander("📊 Statistik X̄-R Chart"):
            st.write(f"- **n (ukuran subgroup):** {n} | **Faktor A2:** {A2} | **D3:** {D3} | **D4:** {D4}")
            st.write(f"- **X̄̄ (Grand Mean):** {xbar_bar:.4f}")
            st.write(f"- **R̄ (Mean Range):** {r_bar:.4f}")
            st.write(f"- **UCL X̄:** {ucl_x:.4f} | **LCL X̄:** {lcl_x:.4f}")
            st.write(f"- **UCL R:** {ucl_r:.4f} | **LCL R:** {lcl_r:.4f}")

    except Exception as e:
        st.error(f"Error X̄-R Chart: {e}")


def render_xbar_s_chart(df, measurement_col, subgroup_col=None, subgroup_size=10):
    """X̄-S Chart: Rata-rata dan Standar Deviasi (n besar)."""
    st.markdown('<div class="chart-info-box">📌 <strong>X̄-S Chart</strong> — Digunakan untuk data variabel dengan ukuran subgroup besar (n ≥ 8). Lebih presisi dari X̄-R karena menggunakan standar deviasi penuh.</div>', unsafe_allow_html=True)

    try:
        if subgroup_col and subgroup_col in df.columns:
            grouped = df.groupby(subgroup_col)[measurement_col]
            xbar = grouped.mean()
            s_vals = grouped.std(ddof=1)
            n = int(df.groupby(subgroup_col).size().mean())
            dates = xbar.index.tolist()
        else:
            values = df[measurement_col].dropna().values
            n = int(subgroup_size)
            num_groups = len(values) // n
            if num_groups < 3:
                st.warning("⚠️ Tidak cukup data untuk membentuk subgroup.")
                return
            values = values[:num_groups * n]
            groups = values.reshape(num_groups, n)
            xbar = pd.Series(groups.mean(axis=1))
            s_vals = pd.Series(groups.std(axis=1, ddof=1))
            dates = list(range(1, num_groups + 1))

        A2, D3, D4, A3, B3, B4, d2 = get_spc_factors(n)
        s_bar = s_vals.mean()
        xbar_bar = xbar.mean()

        ucl_x = xbar_bar + A3 * s_bar
        lcl_x = xbar_bar - A3 * s_bar
        ucl_s = B4 * s_bar
        lcl_s = B3 * s_bar

        fig = make_subplots(rows=2, cols=1, subplot_titles=["X̄ Chart (Mean)", "S Chart (Std Dev)"],
                            vertical_spacing=0.12)

        viol_x = detect_violations(list(xbar), ucl_x, lcl_x, xbar_bar)
        viol_s = detect_violations(list(s_vals), ucl_s, lcl_s, s_bar)

        add_control_lines(fig, dates, ucl_x, lcl_x, xbar_bar, row=1)
        add_control_lines(fig, dates, ucl_s, lcl_s, s_bar, row=2)
        plot_with_violations(fig, dates, list(xbar), viol_x, "X̄", row=1)
        plot_with_violations(fig, dates, list(s_vals), viol_s, "S", row=2)

        fig.update_layout(height=600, title_text="X̄-S Control Chart", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Violation X̄ Chart:**")
            show_violation_summary(viol_x)
        with col2:
            st.markdown("**Violation S Chart:**")
            show_violation_summary(viol_s)

        with st.expander("📊 Statistik X̄-S Chart"):
            st.write(f"- **n (ukuran subgroup):** {n} | **A3:** {A3} | **B3:** {B3} | **B4:** {B4}")
            st.write(f"- **X̄̄:** {xbar_bar:.4f} | **S̄:** {s_bar:.4f}")
            st.write(f"- **UCL X̄:** {ucl_x:.4f} | **LCL X̄:** {lcl_x:.4f}")
            st.write(f"- **UCL S:** {ucl_s:.4f} | **LCL S:** {lcl_s:.4f}")

    except Exception as e:
        st.error(f"Error X̄-S Chart: {e}")


def render_imr_chart(df, measurement_col, date_col=None):
    """I-MR Chart: Individual dan Moving Range (data satu per satu)."""
    st.markdown('<div class="chart-info-box">📌 <strong>I-MR Chart</strong> — Digunakan jika data diambil satu per satu (n=1), misalnya pemantauan harian tunggal, batch proses, atau data kimia. Paling cocok untuk data lambat atau mahal.</div>', unsafe_allow_html=True)

    try:
        values = df[measurement_col].dropna().values
        if len(values) < 5:
            st.warning("⚠️ Minimal 5 data individual diperlukan.")
            return

        dates = df[date_col].values if date_col and date_col in df.columns else list(range(1, len(values) + 1))
        dates = dates[:len(values)]

        # Moving Range (|Xi - Xi-1|)
        mr = np.abs(np.diff(values))
        mr_dates = dates[1:]

        d2 = 1.128  # faktor untuk n=2 (moving range berukuran 2)
        E2 = 2.660  # = 3/d2 untuk n=2
        D4_mr = 3.267  # untuk n=2
        D3_mr = 0.0

        x_bar = np.mean(values)
        mr_bar = np.mean(mr)

        ucl_i = x_bar + E2 * mr_bar
        lcl_i = x_bar - E2 * mr_bar
        ucl_mr = D4_mr * mr_bar
        lcl_mr = D3_mr * mr_bar

        fig = make_subplots(rows=2, cols=1, subplot_titles=["I Chart (Individual)", "MR Chart (Moving Range)"],
                            vertical_spacing=0.12)

        viol_i  = detect_violations(list(values), ucl_i, lcl_i, x_bar)
        viol_mr = detect_violations(list(mr), ucl_mr, lcl_mr, mr_bar)

        add_control_lines(fig, dates, ucl_i, lcl_i, x_bar, row=1)
        add_control_lines(fig, mr_dates, ucl_mr, lcl_mr, mr_bar, row=2)
        plot_with_violations(fig, dates, list(values), viol_i, "Individual", row=1)
        plot_with_violations(fig, mr_dates, list(mr), viol_mr, "Moving Range", row=2)

        fig.update_layout(height=600, title_text="I-MR Control Chart", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Violation I Chart:**")
            show_violation_summary(viol_i)
        with col2:
            st.markdown("**Violation MR Chart:**")
            show_violation_summary(viol_mr)

        with st.expander("📊 Statistik I-MR Chart"):
            st.write(f"- **X̄ (Mean):** {x_bar:.4f} | **MR̄:** {mr_bar:.4f}")
            st.write(f"- **UCL I:** {ucl_i:.4f} | **LCL I:** {lcl_i:.4f}")
            st.write(f"- **UCL MR:** {ucl_mr:.4f} | **LCL MR:** {lcl_mr:.4f}")
            st.write(f"- **Estimasi σ (= MR̄/d2):** {mr_bar/d2:.4f}")

    except Exception as e:
        st.error(f"Error I-MR Chart: {e}")


def render_p_chart(df, date_col, qty_col, ng_col):
    """p-Chart: Proporsi defect (sampel tidak harus tetap)."""
    st.markdown('<div class="chart-info-box">📌 <strong>p-Chart</strong> — Mengukur proporsi/persentase produk cacat. Ukuran sampel boleh bervariasi setiap periode. Paling umum untuk data atribut dengan sampel besar.</div>', unsafe_allow_html=True)

    try:
        daily = df.groupby(date_col)[[qty_col, ng_col]].sum().reset_index()
        daily = daily[daily[qty_col] > 0]
        if len(daily) < 3:
            st.warning("⚠️ Minimal 3 periode data diperlukan untuk p-Chart.")
            return

        daily['p'] = daily[ng_col] / daily[qty_col]
        p_bar = daily[ng_col].sum() / daily[qty_col].sum()
        n_i = daily[qty_col].values

        ucl_vals = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / n_i)
        lcl_vals = np.maximum(0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / n_i))

        dates = daily[date_col].tolist()
        p_vals = daily['p'].tolist()

        # Deteksi violasi menggunakan UCL rata-rata
        ucl_avg = np.mean(ucl_vals)
        lcl_avg = np.mean(lcl_vals)
        violations = detect_violations(p_vals, ucl_avg, lcl_avg, p_bar)

        fig = go.Figure()
        # Zone fill
        fig.add_trace(go.Scatter(x=dates + dates[::-1],
            y=list(ucl_vals) + list(lcl_vals[::-1]),
            fill='toself', fillcolor='rgba(255,0,0,0.07)',
            line=dict(width=0), showlegend=False, hoverinfo='skip'))

        # UCL/LCL garis variabel
        fig.add_trace(go.Scatter(x=dates, y=ucl_vals, mode='lines',
            line=dict(dash='dash', color='red', width=1.5), name='UCL'))
        fig.add_trace(go.Scatter(x=dates, y=lcl_vals, mode='lines',
            line=dict(dash='dash', color='red', width=1.5), name='LCL'))
        fig.add_hline(y=p_bar, line_color='green', line_width=2,
                      annotation_text=f"p̄={p_bar:.4f}")

        # Data
        colors = ['red' if v in violations[1] else '#1e3d59' for v in range(len(p_vals))]
        fig.add_trace(go.Scatter(x=dates, y=p_vals, mode='lines+markers',
            name='Proporsi Defect',
            line=dict(color='#1e3d59', width=2),
            marker=dict(color=colors, size=8)))

        # Highlight violations
        for rule, idxs in violations.items():
            if idxs:
                viol_dates = [dates[i] for i in idxs if i < len(dates)]
                viol_vals  = [p_vals[i] for i in idxs if i < len(p_vals)]
                if viol_dates:
                    fig.add_trace(go.Scatter(x=viol_dates, y=viol_vals, mode='markers',
                        name=f'Rule {rule} Violation',
                        marker=dict(color='red', size=12, symbol='x', line=dict(width=2))))

        fig.update_layout(title="p-Chart (Proportion Defective)", yaxis_title="Proporsi Defect (p)",
                          xaxis_title="Tanggal / Periode", height=450)
        st.plotly_chart(fig, use_container_width=True)
        show_violation_summary(violations)

        with st.expander("📊 Statistik p-Chart"):
            st.write(f"- **p̄ (Mean Proportion):** {p_bar:.4f} ({p_bar*100:.2f}%)")
            st.write(f"- **Total Inspeksi:** {daily[qty_col].sum():,} | **Total Defect:** {daily[ng_col].sum():,}")
            st.write(f"- **UCL (rata-rata):** {ucl_avg:.4f} | **LCL (rata-rata):** {lcl_avg:.4f}")
            st.dataframe(daily.rename(columns={qty_col: 'n', ng_col: 'np', 'p': 'p_i'}).assign(
                UCL=ucl_vals.round(4), LCL=lcl_vals.round(4)), use_container_width=True)

    except Exception as e:
        st.error(f"Error p-Chart: {e}")


def render_np_chart(df, date_col, qty_col, ng_col):
    """np-Chart: Jumlah unit defect (sampel TETAP)."""
    st.markdown('<div class="chart-info-box">📌 <strong>np-Chart</strong> — Mengukur jumlah fisik produk cacat per lot. <em>Ukuran sampel harus sama/tetap</em> setiap periode. Lebih intuitif dari p-chart karena menggunakan angka bulat.</div>', unsafe_allow_html=True)

    try:
        daily = df.groupby(date_col)[[qty_col, ng_col]].sum().reset_index()
        daily = daily[daily[qty_col] > 0]
        if len(daily) < 3:
            st.warning("⚠️ Minimal 3 periode data diperlukan.")
            return

        n_vals = daily[qty_col].values
        n_mean = np.mean(n_vals)
        n_std  = np.std(n_vals)
        if n_std / n_mean > 0.10:
            st.warning(f"⚠️ **Peringatan:** Ukuran sampel bervariasi ({n_std/n_mean*100:.1f}% CV). np-Chart mengasumsikan sampel tetap. Pertimbangkan menggunakan **p-Chart** sebagai gantinya.")

        n_bar = n_mean
        p_bar = daily[ng_col].sum() / daily[qty_col].sum()
        np_bar = n_bar * p_bar

        ucl_np = np_bar + 3 * np.sqrt(np_bar * (1 - p_bar))
        lcl_np = max(0, np_bar - 3 * np.sqrt(np_bar * (1 - p_bar)))

        dates = daily[date_col].tolist()
        np_vals = daily[ng_col].tolist()
        violations = detect_violations(np_vals, ucl_np, lcl_np, np_bar)

        fig = go.Figure()
        add_control_lines(fig, dates, ucl_np, lcl_np, np_bar)
        plot_with_violations(fig, dates, np_vals, violations, "Jumlah Defect (np)")
        fig.update_layout(title="np-Chart (Number of Defective)", yaxis_title="Jumlah Defect",
                          xaxis_title="Tanggal / Periode", height=400)
        st.plotly_chart(fig, use_container_width=True)
        show_violation_summary(violations)

        with st.expander("📊 Statistik np-Chart"):
            st.write(f"- **n̄ (Rata-rata ukuran sampel):** {n_bar:.1f}")
            st.write(f"- **p̄:** {p_bar:.4f} | **n̄p̄:** {np_bar:.4f}")
            st.write(f"- **UCL:** {ucl_np:.4f} | **LCL:** {lcl_np:.4f}")

    except Exception as e:
        st.error(f"Error np-Chart: {e}")


def render_c_chart(df, date_col, defect_count_col):
    """c-Chart: Jumlah defect per unit (ukuran unit tetap)."""
    st.markdown('<div class="chart-info-box">📌 <strong>c-Chart</strong> — Mengukur jumlah <em>cacat (defect)</em> dalam satu unit atau area inspeksi yang ukurannya <em>tetap/konstan</em>. Contoh: jumlah gelembung per panel, jumlah kesalahan per halaman.</div>', unsafe_allow_html=True)

    try:
        daily = df.groupby(date_col)[defect_count_col].sum().reset_index()
        if len(daily) < 3:
            st.warning("⚠️ Minimal 3 periode data diperlukan.")
            return

        dates = daily[date_col].tolist()
        c_vals = daily[defect_count_col].tolist()
        c_bar  = np.mean(c_vals)

        ucl_c = c_bar + 3 * np.sqrt(c_bar)
        lcl_c = max(0, c_bar - 3 * np.sqrt(c_bar))
        violations = detect_violations(c_vals, ucl_c, lcl_c, c_bar)

        fig = go.Figure()
        add_control_lines(fig, dates, ucl_c, lcl_c, c_bar)
        plot_with_violations(fig, dates, c_vals, violations, "Jumlah Defect (c)")
        fig.update_layout(title="c-Chart (Count of Defects per Unit)", yaxis_title="Jumlah Defect",
                          xaxis_title="Tanggal / Periode", height=400)
        st.plotly_chart(fig, use_container_width=True)
        show_violation_summary(violations)

        with st.expander("📊 Statistik c-Chart"):
            st.write(f"- **c̄ (Mean Defect Count):** {c_bar:.4f}")
            st.write(f"- **UCL:** {ucl_c:.4f} | **LCL:** {lcl_c:.4f}")
            st.write(f"- **Total Defect:** {sum(c_vals)} | **Jumlah Periode:** {len(c_vals)}")

    except Exception as e:
        st.error(f"Error c-Chart: {e}")


def render_u_chart(df, date_col, qty_col, defect_count_col):
    """u-Chart: Rata-rata defect per unit (ukuran unit bisa berbeda)."""
    st.markdown('<div class="chart-info-box">📌 <strong>u-Chart</strong> — Mengukur rata-rata jumlah cacat per unit. Berbeda dari c-chart, ukuran area inspeksi <em>boleh bervariasi</em>. Contoh: jumlah cacat per m² kain, per produk dengan panjang berbeda.</div>', unsafe_allow_html=True)

    try:
        daily = df.groupby(date_col)[[qty_col, defect_count_col]].sum().reset_index()
        daily = daily[daily[qty_col] > 0]
        if len(daily) < 3:
            st.warning("⚠️ Minimal 3 periode data diperlukan.")
            return

        daily['u'] = daily[defect_count_col] / daily[qty_col]
        u_bar = daily[defect_count_col].sum() / daily[qty_col].sum()
        n_i   = daily[qty_col].values

        ucl_vals = u_bar + 3 * np.sqrt(u_bar / n_i)
        lcl_vals = np.maximum(0, u_bar - 3 * np.sqrt(u_bar / n_i))

        dates  = daily[date_col].tolist()
        u_vals = daily['u'].tolist()

        ucl_avg = np.mean(ucl_vals)
        lcl_avg = np.mean(lcl_vals)
        violations = detect_violations(u_vals, ucl_avg, lcl_avg, u_bar)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates + dates[::-1],
            y=list(ucl_vals) + list(lcl_vals[::-1]),
            fill='toself', fillcolor='rgba(255,0,0,0.07)',
            line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=dates, y=ucl_vals, mode='lines',
            line=dict(dash='dash', color='red', width=1.5), name='UCL'))
        fig.add_trace(go.Scatter(x=dates, y=lcl_vals, mode='lines',
            line=dict(dash='dash', color='red', width=1.5), name='LCL'))
        fig.add_hline(y=u_bar, line_color='green', line_width=2,
                      annotation_text=f"ū={u_bar:.4f}")

        colors = ['red' if i in violations[1] else '#1e3d59' for i in range(len(u_vals))]
        fig.add_trace(go.Scatter(x=dates, y=u_vals, mode='lines+markers',
            name='Defect per Unit (u)',
            line=dict(color='#1e3d59', width=2),
            marker=dict(color=colors, size=8)))

        for rule, idxs in violations.items():
            if idxs:
                vd = [dates[i] for i in idxs if i < len(dates)]
                vv = [u_vals[i] for i in idxs if i < len(u_vals)]
                if vd:
                    fig.add_trace(go.Scatter(x=vd, y=vv, mode='markers',
                        name=f'Rule {rule}',
                        marker=dict(color='red', size=12, symbol='x', line=dict(width=2))))

        fig.update_layout(title="u-Chart (Defects per Unit)", yaxis_title="Defect per Unit (u)",
                          xaxis_title="Tanggal / Periode", height=400)
        st.plotly_chart(fig, use_container_width=True)
        show_violation_summary(violations)

        with st.expander("📊 Statistik u-Chart"):
            st.write(f"- **ū (Mean Defects per Unit):** {u_bar:.4f}")
            st.write(f"- **Total Defect:** {daily[defect_count_col].sum():,} | **Total Unit:** {daily[qty_col].sum():,}")
            st.write(f"- **UCL (rata-rata):** {ucl_avg:.4f} | **LCL (rata-rata):** {lcl_avg:.4f}")

    except Exception as e:
        st.error(f"Error u-Chart: {e}")


# ===================================================================
# PANDUAN PEMILIHAN CHART
# ===================================================================
def show_chart_guide():
    st.markdown("""
    ### 🗺️ Panduan Pemilihan Control Chart

    | Jenis Data | Kondisi | Chart yang Tepat |
    |---|---|---|
    | **Variabel** (pengukuran: mm, gram, °C) | Subgroup kecil, n = 2–8 | **X̄-R Chart** |
    | **Variabel** | Subgroup besar, n ≥ 8 | **X̄-S Chart** |
    | **Variabel** | Data satu per satu (n=1) | **I-MR Chart** |
    | **Atribut** (defective/NG) | Sampel bervariasi, hitung proporsi | **p-Chart** |
    | **Atribut** (defective/NG) | Sampel tetap, hitung jumlah | **np-Chart** |
    | **Atribut** (defect counts) | Area inspeksi tetap | **c-Chart** |
    | **Atribut** (defect counts) | Area inspeksi bervariasi | **u-Chart** |

    > **Tips:** Data yang sudah ada di file Excel (qty check + qty NG) paling cocok untuk **p-Chart** atau **np-Chart**.
    > Untuk **X̄-R / X̄-S / I-MR**, upload file dengan kolom data pengukuran (dimensi, berat, dll).
    """)


# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
st.sidebar.title("Quality Assurance Dashboard")
st.sidebar.write("Upload Data Produksi:")
uploaded_files = st.sidebar.file_uploader("Drop satu atau beberapa file Excel di sini", type=["xlsx", "xls"], accept_multiple_files=True)
st.sidebar.markdown("---")
st.sidebar.info("Dashboard ini otomatis menghitung Sigma Level dan Seven Tools Quality secara real-time.")

# -------------------------------------------------------------------
# LOGIKA UTAMA
# -------------------------------------------------------------------
if uploaded_files:
    try:
        all_df = []
        for file in uploaded_files:
            temp_df = pd.read_excel(file)
            all_df.append(temp_df)

        df_raw = pd.concat(all_df, ignore_index=True)

        if 'Shift' in df_raw.columns:
            df_raw['Shift'] = df_raw['Shift'].astype(str)

        req_cols = ['Tanggal', 'quantity check', 'qty ng', 'Jenis_Defect']
        if not all(col in df_raw.columns for col in req_cols):
            st.error("❌ Format Excel Salah! Pastikan ada kolom: Tanggal, quantity check, qty ng, Jenis_Defect")
        else:
            df_raw['Tanggal'] = pd.to_datetime(df_raw['Tanggal'])
            df_raw['Bulan'] = df_raw['Tanggal'].dt.strftime('%B %Y')

            st.title("Production Quality Dashboard")

            with st.expander("🔎 Global Filter (Klik untuk Memfilter Data)"):
                col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                with col_f1:
                    lines = ["All"] + sorted(list(df_raw['Line'].unique())) if 'Line' in df_raw.columns else []
                    sel_line = st.selectbox("Filter Line:", lines)
                with col_f2:
                    sizes = ["All"] + sorted(list(df_raw['Ukuran'].unique())) if 'Ukuran' in df_raw.columns else []
                    sel_size = st.selectbox("Filter Ukuran:", sizes)
                with col_f3:
                    types = ["All"] + sorted(list(df_raw['Tipe_Produk'].unique())) if 'Tipe_Produk' in df_raw.columns else []
                    sel_type = st.selectbox("Filter Tipe Produk:", types)
                with col_f4:
                    month_list = ["All"] + sorted(list(df_raw['Bulan'].unique()), key=lambda x: pd.to_datetime(x))
                    sel_month = st.selectbox("Filter Bulan:", month_list)

                df = df_raw.copy()
                if sel_line != "All": df = df[df['Line'] == sel_line]
                if sel_size != "All": df = df[df['Ukuran'] == sel_size]
                if sel_type != "All": df = df[df['Tipe_Produk'] == sel_type]
                if sel_month != "All": df = df[df['Bulan'] == sel_month]

            st.divider()

            # METRIK & SIGMA LEVEL
            total_cek = df['quantity check'].sum()
            total_ng  = df['qty ng'].sum()
            defect_rate = (total_ng / total_cek * 100) if total_cek > 0 else 0

            yield_val = 1 - (total_ng / total_cek) if total_cek > 0 else 0
            if yield_val >= 0.9999999:
                sigma_level = 6.0
            elif yield_val <= 0.0000001:
                sigma_level = 0.0
            else:
                sigma_level = norm.ppf(yield_val) + 1.5

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Produksi", f"{total_cek:,.0f}")
            k2.metric("Total NG", f"{total_ng:,.0f}", delta_color="inverse")
            k3.metric("Defect Rate", f"{defect_rate:.2f}%", delta_color="inverse")
            k4.metric("Sigma Level", f"{sigma_level:.2f}", delta="Target: 4.0")

            # TABS
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Pareto", "🔍 Stratifikasi", "📈 Control Chart",
                "🔗 Scatter Plot", "📋 Histogram", "📥 Raw Data"
            ])

            # TAB 1: PARETO
            with tab1:
                st.subheader("Pareto Analysis")
                pareto_df = df.groupby('Jenis_Defect')['qty ng'].sum().sort_values(ascending=False).reset_index()
                col_p1, col_p2 = st.columns([2, 1])
                with col_p1:
                    fig_bar = px.bar(pareto_df, x='Jenis_Defect', y='qty ng', color='Jenis_Defect',
                                     text='qty ng', title="Defect by Type (Frequency)")
                    st.plotly_chart(fig_bar, use_container_width=True)
                with col_p2:
                    fig_pie = px.pie(pareto_df, values='qty ng', names='Jenis_Defect',
                                     hole=0.4, title="Defect Distribution")
                    st.plotly_chart(fig_pie, use_container_width=True)

            # TAB 2: STRATIFIKASI
            with tab2:
                st.subheader("Advanced Stratification")
                col_strat1, col_strat2 = st.columns(2)
                with col_strat1:
                    st.markdown("##### Drill-Down Analysis")
                    path_options = ['Bulan', 'Line', 'Shift', 'Tipe_Produk', 'Jenis_Defect']
                    selected_path = st.multiselect("Urutan Layer:", path_options, default=['Line', 'Jenis_Defect'])
                    if selected_path:
                        fig_sun = px.sunburst(df, path=selected_path, values='qty ng',
                                              color='qty ng', color_continuous_scale='Reds')
                        st.plotly_chart(fig_sun, use_container_width=True)
                with col_strat2:
                    st.markdown("##### Defect Heatmap")
                    if 'Line' in df.columns and 'Jenis_Defect' in df.columns:
                        heatmap_data = df.groupby(['Jenis_Defect', 'Line'])['qty ng'].sum().reset_index()
                        heatmap_data = heatmap_data.pivot(index='Jenis_Defect', columns='Line', values='qty ng').fillna(0)
                        fig_heat = px.imshow(heatmap_data, text_auto=True, color_continuous_scale="Reds")
                        st.plotly_chart(fig_heat, use_container_width=True)

            # ============================================================
            # TAB 3: CONTROL CHART — LENGKAP
            # ============================================================
            with tab3:
                st.subheader("Statistical Process Control (SPC) — Control Charts")

                cc_mode = st.radio(
                    "Mode Pemilihan Chart:",
                    ["🤖 Auto-Recommend (berdasarkan data)", "🎯 Manual (pilih sendiri)", "📖 Panduan Pemilihan"],
                    horizontal=True
                )

                st.divider()

                if cc_mode == "📖 Panduan Pemilihan":
                    show_chart_guide()

                elif cc_mode == "🤖 Auto-Recommend (berdasarkan data)":
                    st.markdown("#### Rekomendasi Otomatis Berdasarkan Struktur Data")

                    # Analisis struktur data
                    has_attr = 'quantity check' in df.columns and 'qty ng' in df.columns
                    sample_cv = df.groupby('Tanggal')['quantity check'].sum().std() / df.groupby('Tanggal')['quantity check'].sum().mean() if has_attr else 1.0
                    sample_sizes = df.groupby('Tanggal')['quantity check'].sum()
                    is_fixed_sample = sample_cv < 0.10

                    num_var_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    excl = ['quantity check', 'qty ng']
                    measurement_cols = [c for c in num_var_cols if c not in excl and df[c].nunique() > 10]

                    rec_col1, rec_col2 = st.columns(2)
                    with rec_col1:
                        st.markdown("**📋 Analisis Data:**")
                        st.write(f"- Kolom atribut (qty check + qty NG): {'✅ Ada' if has_attr else '❌ Tidak ada'}")
                        st.write(f"- Variasi ukuran sampel harian (CV): {sample_cv*100:.1f}%")
                        st.write(f"- Sampel relatif tetap: {'✅ Ya' if is_fixed_sample else '⚠️ Tidak (gunakan p-Chart)'}")
                        st.write(f"- Kolom pengukuran variabel: {measurement_cols if measurement_cols else 'Tidak ditemukan'}")
                    with rec_col2:
                        st.markdown("**💡 Rekomendasi Chart:**")
                        if has_attr:
                            if is_fixed_sample:
                                st.success("✅ **p-Chart & np-Chart** — Sampel relatif tetap, keduanya applicable.")
                            else:
                                st.success("✅ **p-Chart** — Sampel bervariasi, proporsi lebih tepat.")
                            st.info("✅ **c-Chart / u-Chart** — Berdasarkan total defect per periode.")
                        if measurement_cols:
                            st.success(f"✅ **I-MR Chart** — Tersedia kolom pengukuran: {measurement_cols}")

                    st.divider()

                    # RENDER CHART YANG DIREKOMENDASIKAN
                    if has_attr:
                        st.markdown("### p-Chart (Rekomendasi Utama)")
                        render_p_chart(df, 'Tanggal', 'quantity check', 'qty ng')

                        st.markdown("---")
                        if is_fixed_sample:
                            st.markdown("### np-Chart")
                            render_np_chart(df, 'Tanggal', 'quantity check', 'qty ng')
                            st.markdown("---")

                        st.markdown("### c-Chart (Total Defect per Hari)")
                        render_c_chart(df, 'Tanggal', 'qty ng')

                        st.markdown("---")
                        st.markdown("### u-Chart (Defect per Unit per Hari)")
                        render_u_chart(df, 'Tanggal', 'quantity check', 'qty ng')

                    if measurement_cols:
                        st.markdown("---")
                        sel_meas = st.selectbox("Pilih kolom pengukuran untuk I-MR Chart:", measurement_cols)
                        st.markdown(f"### I-MR Chart — {sel_meas}")
                        render_imr_chart(df, sel_meas, 'Tanggal')

                else:  # Manual selection
                    st.markdown("#### Pilih Jenis Control Chart")

                    chart_options = {
                        "p-Chart (Proporsi Defect — sampel variabel)": "p",
                        "np-Chart (Jumlah Defect — sampel tetap)": "np",
                        "c-Chart (Jumlah Defect per Unit tetap)": "c",
                        "u-Chart (Defect per Unit — area variabel)": "u",
                        "X̄-R Chart (Mean & Range — subgroup kecil)": "xbar_r",
                        "X̄-S Chart (Mean & Std Dev — subgroup besar)": "xbar_s",
                        "I-MR Chart (Individual & Moving Range)": "imr",
                    }

                    selected_chart = st.selectbox("Pilih jenis chart:", list(chart_options.keys()))
                    chart_type = chart_options[selected_chart]

                    st.divider()

                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    all_cols = df.columns.tolist()

                    # Atribut charts
                    if chart_type == "p":
                        qty_col = st.selectbox("Kolom ukuran sampel (n):", num_cols,
                                               index=num_cols.index('quantity check') if 'quantity check' in num_cols else 0)
                        ng_col  = st.selectbox("Kolom jumlah defect/NG:", num_cols,
                                               index=num_cols.index('qty ng') if 'qty ng' in num_cols else 1)
                        if st.button("▶ Render p-Chart"):
                            render_p_chart(df, 'Tanggal', qty_col, ng_col)

                    elif chart_type == "np":
                        qty_col = st.selectbox("Kolom ukuran sampel (n):", num_cols,
                                               index=num_cols.index('quantity check') if 'quantity check' in num_cols else 0)
                        ng_col  = st.selectbox("Kolom jumlah defect/NG:", num_cols,
                                               index=num_cols.index('qty ng') if 'qty ng' in num_cols else 1)
                        if st.button("▶ Render np-Chart"):
                            render_np_chart(df, 'Tanggal', qty_col, ng_col)

                    elif chart_type == "c":
                        def_col = st.selectbox("Kolom jumlah defect:", num_cols,
                                               index=num_cols.index('qty ng') if 'qty ng' in num_cols else 0)
                        if st.button("▶ Render c-Chart"):
                            render_c_chart(df, 'Tanggal', def_col)

                    elif chart_type == "u":
                        qty_col = st.selectbox("Kolom ukuran unit inspeksi:", num_cols,
                                               index=num_cols.index('quantity check') if 'quantity check' in num_cols else 0)
                        def_col = st.selectbox("Kolom total defect:", num_cols,
                                               index=num_cols.index('qty ng') if 'qty ng' in num_cols else 1)
                        if st.button("▶ Render u-Chart"):
                            render_u_chart(df, 'Tanggal', qty_col, def_col)

                    # Variable charts
                    elif chart_type == "xbar_r":
                        meas_col = st.selectbox("Kolom data pengukuran:", num_cols)
                        subg_col_option = ["(Tidak ada — bentuk subgroup otomatis)"] + all_cols
                        subg_col = st.selectbox("Kolom subgroup (opsional):", subg_col_option)
                        subg_size = st.slider("Ukuran subgroup (jika otomatis):", 2, 10, 5)
                        sc = None if subg_col == "(Tidak ada — bentuk subgroup otomatis)" else subg_col
                        if st.button("▶ Render X̄-R Chart"):
                            render_xbar_r_chart(df, meas_col, sc, subg_size)

                    elif chart_type == "xbar_s":
                        meas_col = st.selectbox("Kolom data pengukuran:", num_cols)
                        subg_col_option = ["(Tidak ada — bentuk subgroup otomatis)"] + all_cols
                        subg_col = st.selectbox("Kolom subgroup (opsional):", subg_col_option)
                        subg_size = st.slider("Ukuran subgroup (jika otomatis):", 8, 25, 10)
                        sc = None if subg_col == "(Tidak ada — bentuk subgroup otomatis)" else subg_col
                        if st.button("▶ Render X̄-S Chart"):
                            render_xbar_s_chart(df, meas_col, sc, subg_size)

                    elif chart_type == "imr":
                        meas_col = st.selectbox("Kolom data pengukuran:", num_cols)
                        date_col_opt = ["(Gunakan index)"] + all_cols
                        date_col_sel = st.selectbox("Kolom tanggal/urutan:", date_col_opt)
                        dc = None if date_col_sel == "(Gunakan index)" else date_col_sel
                        if st.button("▶ Render I-MR Chart"):
                            render_imr_chart(df, meas_col, dc)

            # TAB 4: SCATTER
            with tab4:
                st.subheader("Correlation Analysis")
                correlation = df['quantity check'].corr(df['qty ng'])
                fig_scat = px.scatter(df, x='quantity check', y='qty ng', color='Shift',
                                      trendline="ols", trendline_scope="overall",
                                      trendline_color_override="red")
                st.plotly_chart(fig_scat, use_container_width=True)
                st.write(f"**Koefisien Korelasi (r):** {correlation:.2f}")

            # TAB 5: HISTOGRAM
            with tab5:
                st.subheader("Distribution of Defects")
                df['Defect_Rate_Row'] = (df['qty ng'] / df['quantity check']) * 100
                fig_hist = px.histogram(df, x="Defect_Rate_Row", nbins=20, marginal="box")
                fig_hist.add_vrect(x0=0, x1=2.0, fillcolor="green", opacity=0.1, annotation_text="Target Zone")
                st.plotly_chart(fig_hist, use_container_width=True)

            # TAB 6: RAW DATA
            with tab6:
                st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.exception(e)

else:
    st.info("Silakan upload satu atau beberapa file Excel produksi di sidebar untuk memulai.")
