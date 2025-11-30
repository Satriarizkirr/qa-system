import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------------
# 1. KONFIGURASI TAMPILAN
# -------------------------------------------------------------------
st.set_page_config(page_title="QA System Visualization", layout="wide", page_icon="🏭")

st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    h1 { color: #1e3d59; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border-left: 5px solid #1e3d59;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. SIDEBAR
# -------------------------------------------------------------------
st.sidebar.title("Quality Assurance Data Visualization")
st.sidebar.write("Upload Data Produksi:")
uploaded_file = st.sidebar.file_uploader("Drop file Excel di sini", type=["xlsx", "xls"])

st.sidebar.markdown("---")


# -------------------------------------------------------------------
# 3. LOGIKA UTAMA
# -------------------------------------------------------------------
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        
        # Pastikan kolom Shift jadi string biar dianggap kategori, bukan angka
        if 'Shift' in df.columns:
            df['Shift'] = df['Shift'].astype(str)

        # Validasi Kolom
        req_cols = ['Tanggal', 'quantity check', 'qty ng', 'Jenis_Defect']
        if not all(col in df.columns for col in req_cols):
            st.error("❌ Format Excel Salah! Pastikan nama kolom benar.")
        else:
            df['Tanggal'] = pd.to_datetime(df['Tanggal'])

            # --- HEADER ---
            st.title("Production Quality Dashboard")
            

            # --- GLOBAL FILTER ---
            with st.expander("🔎 Global Filter (Klik untuk Membuka)"):
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    lines = ["All"] + list(df['Line'].unique()) if 'Line' in df.columns else []
                    sel_line = st.selectbox("Filter Line:", lines)
                with col_f2:
                    sizes = ["All"] + list(df['Ukuran'].unique()) if 'Ukuran' in df.columns else []
                    sel_size = st.selectbox("Filter Ukuran:", sizes)
                
                # Apply Filter
                if sel_line != "All": df = df[df['Line'] == sel_line]
                if sel_size != "All": df = df[df['Ukuran'] == sel_size]

            st.divider()

            # --- KPI ---
            total_cek = df['quantity check'].sum()
            total_ng = df['qty ng'].sum()
            defect_rate = (total_ng / total_cek * 100) if total_cek > 0 else 0
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Produksi", f"{total_cek:,.0f}")
            k2.metric("Total NG", f"{total_ng:,.0f}", delta_color="inverse")
            k3.metric("Defect Rate", f"{defect_rate:.2f}%", delta_color="inverse")
            k4.metric("Sigma Level", "3.8", delta="Target: 4.0")

            # --- TABS ---
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Pareto", 
                "🔍 Stratifikasi", 
                "📈 Control Chart", 
                "🔗 Scatter Plot", 
                "📋 Histogram", 
                "📥 Data"
            ])

            # TAB 1: PARETO
            with tab1:
                st.subheader("1. Pareto Chart")
                col_p1, col_p2 = st.columns([2, 1])
                pareto_df = df.groupby('Jenis_Defect')['qty ng'].sum().sort_values(ascending=False).reset_index()
                with col_p1:
                    fig_bar = px.bar(pareto_df, x='Jenis_Defect', y='qty ng', color='Jenis_Defect', text='qty ng')
                    fig_bar.update_layout(showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
                with col_p2:
                    fig_pie = px.pie(pareto_df, values='qty ng', names='Jenis_Defect', color='Jenis_Defect', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)

            # TAB 2: STRATIFICATION
            with tab2:
                st.subheader("2. Advanced Stratification Analysis")
                st.caption("Memecah masalah menjadi lapisan yang lebih detail.")
                
                col_strat1, col_strat2 = st.columns(2)
                
                # A. SUNBURST
                with col_strat1:
                    st.markdown("#####  Drill-Down Analysis")
                    path_options = ['Line', 'Shift', 'Jenis_Defect', 'Ukuran']
                    selected_path = st.multiselect("Pilih Urutan Layer:", path_options, default=['Line', 'Jenis_Defect'])
                    
                    if selected_path:
                        fig_sun = px.sunburst(df, path=selected_path, values='qty ng', 
                                              color='qty ng', color_continuous_scale='RdBu_r')
                        st.plotly_chart(fig_sun, use_container_width=True)
                    else:
                        st.warning("Pilih minimal 1 layer.")

                # B. HEATMAP
                with col_strat2:
                    st.markdown("##### Defect Heatmap")
                    if 'Line' in df.columns and 'Jenis_Defect' in df.columns:
                        heatmap_data = df.groupby(['Jenis_Defect', 'Line'])['qty ng'].sum().reset_index()
                        heatmap_data = heatmap_data.pivot(index='Jenis_Defect', columns='Line', values='qty ng').fillna(0)
                        
                        fig_heat = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale="Reds")
                        st.plotly_chart(fig_heat, use_container_width=True)

                st.markdown("---")
                
                # C. STACKED BAR
                st.markdown("##### 📊 Perbandingan Antar Shift")
                if 'Shift' in df.columns:
                    shift_chart = df.groupby(['Shift', 'Jenis_Defect'])['qty ng'].sum().reset_index()
                    fig_stack = px.bar(shift_chart, x='Shift', y='qty ng', color='Jenis_Defect', barmode='stack')
                    st.plotly_chart(fig_stack, use_container_width=True)

            # -------------------------------------------------------
            # TAB 3: CONTROL CHART (FITUR BARU: Auto-Highlight Merah)
            # -------------------------------------------------------
            with tab3:
                st.subheader("3. Control Chart (SPC)")
                daily = df.groupby('Tanggal')[['quantity check', 'qty ng']].sum().reset_index()
                daily['Rate'] = (daily['qty ng'] / daily['quantity check']) * 100
                mean_rate = daily['Rate'].mean()
                std_dev = daily['Rate'].std()
                ucl, lcl = mean_rate + (2*std_dev), max(0, mean_rate - (2*std_dev))
                
                fig_ctrl = go.Figure()
                # Garis utama (Warna default biru plotly, tidak diubah)
                fig_ctrl.add_trace(go.Scatter(x=daily['Tanggal'], y=daily['Rate'], mode='lines+markers', name='Defect Rate'))
                
                # --- FITUR TAMBAHAN: Highlight Outliers ---
                outliers = daily[(daily['Rate'] > ucl) | (daily['Rate'] < lcl)]
                if not outliers.empty:
                    fig_ctrl.add_trace(go.Scatter(
                        x=outliers['Tanggal'], y=outliers['Rate'],
                        mode='markers', name='Outlier (Bahaya)',
                        marker=dict(color='red', size=12, symbol='circle-open-dot') # Merah khusus outlier
                    ))

                # Garis Batas
                fig_ctrl.add_hline(y=ucl, line_dash="dot", line_color="red", annotation_text="UCL")
                fig_ctrl.add_hline(y=lcl, line_dash="dot", line_color="red", annotation_text="LCL")
                fig_ctrl.add_hline(y=mean_rate, line_color="green", annotation_text="Avg")
                
                st.plotly_chart(fig_ctrl, use_container_width=True)
                
                # Peringatan Teks Otomatis
                if not outliers.empty:
                    st.error(f"⚠️ **PERINGATAN:** Ditemukan {len(outliers)} hari Out-of-Control (Titik Merah). Harap investigasi!")
                else:
                    st.success("✅ Proses Stabil (Semua dalam batas kontrol).")

            # -------------------------------------------------------
            # TAB 4: SCATTER PLOT (FITUR BARU: Statistik R-Value)
            # -------------------------------------------------------
            with tab4:
                st.subheader("4. Scatter Diagram")
                
                # Hitung Statistik
                correlation = df['quantity check'].corr(df['qty ng'])
                r_squared = correlation ** 2
                
                col_sc1, col_sc2 = st.columns([3, 1])
                
                with col_sc1:
                    try:
                        fig_scat = px.scatter(df, x='quantity check', y='qty ng', color='Shift', 
                                              trendline="ols", trendline_scope="overall", trendline_color_override="red")
                        st.plotly_chart(fig_scat, use_container_width=True)
                    except:
                        st.plotly_chart(px.scatter(df, x='quantity check', y='qty ng', color='Shift'), use_container_width=True)
                
                with col_sc2:
                    st.markdown("#### Statistik")
                    st.metric("Korelasi (r)", f"{correlation:.2f}")
                    st.metric("R-Squared", f"{r_squared:.2f}")
                    if correlation > 0.7:
                        st.warning("Korelasi Kuat! Produksi tinggi = Reject tinggi.")
                    elif correlation < 0.3:
                        st.success("Korelasi Lemah. Aman.")

            # -------------------------------------------------------
            # TAB 5: HISTOGRAM (FITUR BARU: Safe Zone)
            # -------------------------------------------------------
            with tab5:
                st.subheader("5. Histogram")
                df['Row_Rate'] = (df['qty ng'] / df['quantity check']) * 100
                
                fig_hist = px.histogram(df, x="Row_Rate", nbins=20, marginal="box", title="Distribusi Defect Rate")
                
                # Tambah Shading Area Hijau (Zona Aman < 2%) - Tanpa ubah warna bar
                fig_hist.add_vrect(x0=0, x1=2.0, fillcolor="green", opacity=0.1, 
                                   annotation_text="Zona Aman", annotation_position="top right")
                
                st.plotly_chart(fig_hist, use_container_width=True)
                st.caption("Area hijau transparan menunjukkan target defect rate yang ideal (<2%).")

            # TAB 6: DATA
            with tab6:
                st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")