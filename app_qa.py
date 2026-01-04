import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. SIDEBAR (MULTI-UPLOAD AKTIF)
# -------------------------------------------------------------------
st.sidebar.title("Quality Assurance Dashboard")
st.sidebar.write("Upload Data Produksi:")
# Sekarang bisa upload banyak file sekaligus
uploaded_files = st.sidebar.file_uploader("Drop satu atau beberapa file Excel", type=["xlsx", "xls"], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.info("Dashboard ini otomatis menggabungkan data dari semua file yang di-upload.")

# -------------------------------------------------------------------
# 3. LOGIKA UTAMA
# -------------------------------------------------------------------
if uploaded_files:
    try:
        # Menggabungkan semua file yang di-upload
        list_df = []
        for f in uploaded_files:
            list_df.append(pd.read_excel(f))
        
        df = pd.concat(list_df, ignore_index=True)
        
        # Pre-processing
        if 'Shift' in df.columns:
            df['Shift'] = df['Shift'].astype(str)
        
        req_cols = ['Tanggal', 'quantity check', 'qty ng', 'Jenis_Defect']
        if not all(col in df.columns for col in req_cols):
            st.error("❌ Format Excel Salah! Pastikan ada kolom: Tanggal, quantity check, qty ng, Jenis_Defect")
        else:
            df['Tanggal'] = pd.to_datetime(df['Tanggal'])
            # Tambahkan kolom Bulan untuk analisis per bulan
            df['Bulan'] = df['Tanggal'].dt.strftime('%B %Y')

            # --- HEADER ---
            st.title("Production Quality Dashboard")
            st.caption(f"Menampilkan data gabungan dari {len(uploaded_files)} file produksi.")
            
            # --- GLOBAL FILTER ---
            with st.expander("🔎 Global Filter (Klik untuk Memfilter Data)"):
                col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                
                with col_f1:
                    lines = ["All"] + sorted(list(df['Line'].unique())) if 'Line' in df.columns else []
                    sel_line = st.selectbox("Filter Line:", lines)
                with col_f2:
                    sizes = ["All"] + sorted(list(df['Ukuran'].unique())) if 'Ukuran' in df.columns else []
                    sel_size = st.selectbox("Filter Ukuran:", sizes)
                with col_f3:
                    types = ["All"] + sorted(list(df['Tipe_Produk'].unique())) if 'Tipe_Produk' in df.columns else []
                    sel_type = st.selectbox("Filter Tipe Produk:", types)
                with col_f4:
                    months = ["All"] + sorted(list(df['Bulan'].unique()))
                    sel_month = st.selectbox("Filter Bulan:", months)
                
                # Eksekusi Filter
                if sel_line != "All": df = df[df['Line'] == sel_line]
                if sel_size != "All": df = df[df['Ukuran'] == sel_size]
                if sel_type != "All": df = df[df['Tipe_Produk'] == sel_type]
                if sel_month != "All": df = df[df['Bulan'] == sel_month]

            st.divider()

            # --- HITUNG METRIK & SIGMA LEVEL ---
            total_cek = df['quantity check'].sum()
            total_ng = df['qty ng'].sum()
            defect_rate = (total_ng / total_cek * 100) if total_cek > 0 else 0
            yield_val = 1 - (total_ng / total_cek) if total_cek > 0 else 0
            
            if yield_val >= 0.9999999: sigma_level = 6.0
            elif yield_val <= 0.0000001: sigma_level = 0.0
            else: sigma_level = norm.ppf(yield_val) + 1.5

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Produksi", f"{total_cek:,.0f}")
            k2.metric("Total NG", f"{total_ng:,.0f}", delta_color="inverse")
            k3.metric("Defect Rate", f"{defect_rate:.2f}%", delta_color="inverse")
            k4.metric("Sigma Level", f"{sigma_level:.2f}", delta="Target: 4.0")

            # --- TABS VISUALISASI ---
            # Menambahkan tab "Trend Bulanan"
            tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📈 Trend Bulanan", "📊 Pareto", "🔍 Stratifikasi", 
                "📈 Control Chart", "🔗 Scatter Plot", "📋 Histogram", "📥 Raw Data"
            ])

            # TAB 0: TREND BULANAN (Fitur Baru)
            with tab0:
                st.subheader("Monthly Performance Trend")
                monthly_data = df.groupby('Bulan').agg({
                    'quantity check': 'sum',
                    'qty ng': 'sum'
                }).reset_index()
                monthly_data['Rate (%)'] = (monthly_data['qty ng'] / monthly_data['quantity check']) * 100
                
                fig_trend = px.line(monthly_data, x='Bulan', y='Rate (%)', markers=True, text=monthly_data['Rate (%)'].apply(lambda x: f'{x:.2f}%'))
                fig_trend.add_hline(y=2.0, line_dash="dot", line_color="red", annotation_text="Target 2%")
                st.plotly_chart(fig_trend, width="stretch")
                st.write("Ringkasan Bulanan:", monthly_data)

            # TAB 1: PARETO
            with tab1:
                st.subheader("Pareto Analysis")
                pareto_df = df.groupby('Jenis_Defect')['qty ng'].sum().sort_values(ascending=False).reset_index()
                col_p1, col_p2 = st.columns([2, 1])
                with col_p1:
                    fig_bar = px.bar(pareto_df, x='Jenis_Defect', y='qty ng', color='Jenis_Defect', text='qty ng')
                    st.plotly_chart(fig_bar, width="stretch")
                with col_p2:
                    fig_pie = px.pie(pareto_df, values='qty ng', names='Jenis_Defect', hole=0.4)
                    st.plotly_chart(fig_pie, width="stretch")

            # TAB 2: STRATIFICATION (Update: Tambah Bulan di Path)
            with tab2:
                st.subheader("Advanced Stratification")
                path_options = ['Bulan', 'Line', 'Shift', 'Tipe_Produk', 'Jenis_Defect']
                selected_path = st.multiselect("Urutan Layer:", path_options, default=['Bulan', 'Line', 'Jenis_Defect'])
                if selected_path:
                    fig_sun = px.sunburst(df, path=selected_path, values='qty ng', color='qty ng', color_continuous_scale='Reds')
                    st.plotly_chart(fig_sun, width="stretch")

            # TAB 3: CONTROL CHART (SPC)
            with tab3:
                st.subheader("P-Chart (Daily Control)")
                daily = df.groupby('Tanggal')[['quantity check', 'qty ng']].sum().reset_index()
                daily['Rate'] = (daily['qty ng'] / daily['quantity check']) * 100
                mean_r = daily['Rate'].mean()
                std_r = daily['Rate'].std()
                ucl, lcl = mean_r + (3*std_r), max(0, mean_r - (3*std_r))
                fig_ctrl = go.Figure()
                fig_ctrl.add_trace(go.Scatter(x=daily['Tanggal'], y=daily['Rate'], mode='lines+markers', name='Rate'))
                fig_ctrl.add_hline(y=ucl, line_dash="dot", line_color="red", annotation_text="UCL")
                fig_ctrl.add_hline(y=mean_r, line_color="green", annotation_text="Avg")
                st.plotly_chart(fig_ctrl, width="stretch")

            # TAB 4: SCATTER PLOT
            with tab4:
                st.subheader("Correlation Analysis")
                fig_scat = px.scatter(df, x='quantity check', y='qty ng', color='Bulan', trendline="ols")
                st.plotly_chart(fig_scat, width="stretch")

            # TAB 5: HISTOGRAM
            with tab5:
                st.subheader("Distribution of Defects")
                df['Defect_Rate_Row'] = (df['qty ng'] / df['quantity check']) * 100
                fig_hist = px.histogram(df, x="Defect_Rate_Row", nbins=20, marginal="box")
                st.plotly_chart(fig_hist, width="stretch")

            # TAB 6: RAW DATA
            with tab6:
                st.dataframe(df, width="stretch")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("Silakan upload satu atau beberapa file Excel produksi untuk memulai analisis akumulasi.")
