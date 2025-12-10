import streamlit as st, pandas as pd, plotly.express as px, plotly.graph_objects as go, re, os
from datetime import datetime

st.set_page_config(page_title="Monitor Neumonía Chile", layout="wide")

boxcox_regiones = {
    "Total País": False, "Región de Arica y Parinacota": False, "Región de Tarapacá": True, "Región de Antofagasta": True,
    "Región de Atacama": True, "Región de Coquimbo": True, "Región de Valparaíso": False, "Región Metropolitana": False,
    "Región de O'Higgins": True, "Región del Maule": True, "Región de Ñuble": False, "Región del Biobío": True,
    "Región de La Araucanía": True, "Región de Los Ríos": True, "Región de Los Lagos": False, "Región de Aysén": False, "Región de Magallanes": True
}

@st.cache_data
def cargar_datos():
    try:
        df_h = pd.read_csv("base_neumonia_dashboard_READY.csv", encoding="utf-8")
        df_h["fecha"] = pd.to_datetime(df_h["fecha"])
    except FileNotFoundError: st.error("Error crítico: no encuentro 'base_neumonia_dashboard_READY.csv'."); st.stop()
    try:
        df_p = pd.read_csv("predicciones_dashboard.csv", encoding="utf-8")
        df_p["fecha"] = pd.to_datetime(df_p["fecha"])
    except FileNotFoundError: st.warning("No encontré 'predicciones_dashboard.csv'."); df_p = pd.DataFrame()
    try: df_m = pd.read_csv("modelos_neumonia.csv", encoding="utf-8")
    except FileNotFoundError: st.warning("No encontré 'modelos_neumonia.csv'."); df_m = pd.DataFrame()
    try:
        df_i = pd.read_csv("serie_imputada_dashboard.csv", encoding="utf-8")
        df_i["fecha"] = pd.to_datetime(df_i["fecha"])
    except FileNotFoundError: st.warning("No encontré 'serie_imputada_dashboard.csv'."); df_i = pd.DataFrame()
    return df_h, df_p, df_m, df_i

@st.cache_data
def cargar_codigo_dashboard(nombre="dashboard.py"):
    return open(nombre, "r", encoding="utf-8").read() if os.path.exists(nombre) else None

df_hist, df_pred, df_modelos, df_imput = cargar_datos()

with st.container():
    _, col_c, _ = st.columns([1, 4, 1])
    with col_c:
        st.subheader("Filtros de visualización")
        c1, c2 = st.columns([1, 2])
        with c1:
            lst = sorted(df_hist["Region"].unique().tolist())
            if "Total País" in lst: lst.remove("Total País")
            lst.insert(0, "Total País")
            region_sel = st.selectbox("Seleccione región:", lst)
        with c2:
            f_min, f_max = df_hist["fecha"].min(), df_hist["fecha"].max()
            f_max_tot = max(f_max, df_pred["fecha"].max()) if not df_pred.empty else f_max
            def_start = max(f_min, pd.Timestamp("2017-01-01"))
            rango = st.slider("Rango temporal:", min_value=f_min.to_pydatetime(), max_value=f_max_tot.to_pydatetime(), value=(def_start.to_pydatetime(), f_max_tot.to_pydatetime()), format="YYYY-MM-DD")

st.markdown("---"); fi, ff = pd.to_datetime(rango[0]), pd.to_datetime(rango[1])

if region_sel == "Total País":
    df_hr = df_hist[df_hist["Region"] == "Total País"].copy() if "Total País" in df_hist["Region"].unique() else df_hist.groupby(["fecha", "Año", "Semana"], as_index=False)["Casos"].sum().assign(Region="Total País")
else: df_hr = df_hist[df_hist["Region"] == region_sel].copy()
df_h_pl = df_hr[(df_hr["fecha"] >= fi) & (df_hr["fecha"] <= ff)].copy()

if not df_pred.empty:
    df_pr = df_pred[df_pred["Region"] == region_sel].copy()
    df_p_pl = df_pr[(df_pr["fecha"] >= fi) & (df_pr["fecha"] <= ff)].copy()
else: df_p_pl = pd.DataFrame()

if not df_imput.empty:
    df_ir = df_imput[df_imput["Region"] == region_sel].copy()
    df_ir["Imputado"] = df_ir["Imputado"].astype(str).str.upper().isin(["TRUE", "1"])
    df_i_pl = df_ir[(df_ir["fecha"] >= fi) & (df_ir["fecha"] <= ff)].copy()
else: df_i_pl = pd.DataFrame()

mod_clean = None
if not df_modelos.empty:
    row = df_modelos.loc[df_modelos["Region"] == region_sel]
    if not row.empty:
        mt = str(row["Modelo"].iloc[0]).replace("Regression with ", "").replace(" regression with ", "").replace(" errors", "").strip()
        m = re.search(r"ARIMA.*", mt); mod_clean = m.group(0) if m else mt

st.title(f"Vigilancia epidemiológica: {region_sel} | Modelo: {mod_clean}" if mod_clean else f"Vigilancia epidemiológica: {region_sel}")
st.markdown("Monitor de atenciones de urgencia por neumonía. Red pública de salud.")
bc = boxcox_regiones.get(region_sel, None)
if bc is True: st.caption("Transformación Box-Cox aplicada al modelo ARIMA de esta región.")
elif bc is False: st.caption("Modelo ARIMA ajustado en escala original (sin transformación Box-Cox).")

if not df_h_pl.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Promedio histórico", f"{int(df_h_pl['Casos'].mean()):,}".replace(",", "."))
    c2.metric("Total acumulado", f"{int(df_h_pl['Casos'].sum()):,}".replace(",", "."))
    c3.metric("Peak histórico", f"{int(df_h_pl['Casos'].max()):,}".replace(",", "."))
    c4.metric("Peak proyectado 2025", f"{int(df_p_pl['Casos'].max()):,}".replace(",", ".") if not df_p_pl.empty else "N/A", delta_color="inverse")
st.divider()

with st.container():
    _, c2, _ = st.columns([1, 4, 1])
    with c2:
        st.subheader("Evolución temporal y pronóstico")
        _, tm, _ = st.columns([1, 3, 6])
        show_orig = tm.toggle("Visualizar datos originales en pandemia", value=False)
        fig = go.Figure()
        p_ini, p_fin = pd.to_datetime("2020-03-15"), pd.to_datetime("2021-12-31")
        
        if not df_i_pl.empty:
            fig.add_trace(go.Scatter(x=df_i_pl["fecha"], y=df_i_pl["Casos"], mode="lines", name="Datos observados", line=dict(color="#2c3e50", width=2.5)))
            d_imp = df_i_pl[(df_i_pl["Imputado"]) & (df_i_pl["fecha"] >= p_ini) & (df_i_pl["fecha"] <= p_fin)]
            if not d_imp.empty: fig.add_trace(go.Scatter(x=d_imp["fecha"], y=d_imp["Casos"], mode="lines", name="Datos imputados", line=dict(color="#e74c3c", width=2.5)))
        else:
            fig.add_trace(go.Scatter(x=df_h_pl["fecha"], y=df_h_pl["Casos"], mode="lines", name="Datos observados", line=dict(color="#2c3e50", width=2.5)))

        if show_orig and not df_h_pl.empty:
            dp = df_h_pl[(df_h_pl["fecha"] >= p_ini) & (df_h_pl["fecha"] <= p_fin)]
            if not dp.empty: fig.add_trace(go.Scatter(x=dp["fecha"], y=dp["Casos"], mode="lines", name="Datos originales pandemia", line=dict(color="rgba(80,80,80,0.7)", width=2, dash="dot")))

        if not df_p_pl.empty:
            fig.add_trace(go.Scatter(x=pd.concat([df_p_pl["fecha"], df_p_pl["fecha"][::-1]]), y=pd.concat([df_p_pl["Upper"], df_p_pl["Lower"][::-1]]), fill="toself", fillcolor="rgba(39, 174, 96, 0.2)", line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=True, name="Intervalo 90%"))
            fig.add_trace(go.Scatter(x=df_p_pl["fecha"], y=df_p_pl["Casos"], mode="lines", name="Pronóstico 2025", line=dict(color="#27ae60", width=3, dash="solid")))

        fig.add_vrect(x0="2020-03-15", x1="2021-12-31", fillcolor="gray", opacity=0.1, layer="below", line_width=0, annotation_text="Pandemia COVID-19", annotation_position="top left")
        fig.update_layout(xaxis_title="Fecha", yaxis_title="Casos semanales", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified", height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

with st.container():
    _, c2, _ = st.columns([1, 4, 1])
    with c2:
        st.subheader("Comparativa estacional")
        fs = px.line(df_h_pl, x="Semana", y="Casos", color="Año", title="Curvas epidemiológicas superpuestas", color_discrete_sequence=px.colors.qualitative.Dark24)
        fs.update_layout(xaxis_title="Semana (1-52)", yaxis_title="Casos", hovermode="x unified")
        st.plotly_chart(fs, use_container_width=True)

with st.expander("Ver datos detallados"):
    t1, t2 = st.tabs(["Proyección", "Historia"])
    with t1:
        if not df_p_pl.empty: st.dataframe(df_p_pl[["fecha", "Region", "Casos", "Lower", "Upper"]].style.format({"Casos": "{:.0f}", "Lower": "{:.0f}", "Upper": "{:.0f}"}))
        else: st.info("Sin proyección disponible para este filtro.")
    with t2: st.dataframe(df_h_pl.sort_values("fecha", ascending=False).head(100))

st.divider(); st.subheader("Descarga de recursos"); ca, cb = st.columns([2, 1])
with ca:
    st.markdown("**Bases de datos disponibles**")
    if os.path.exists("base_neumonia_dashboard_READY.csv"): st.download_button("Serie con datos observados (CSV)", open("base_neumonia_dashboard_READY.csv", "rb"), "base_neumonia.csv", "text/csv")
    if os.path.exists("serie_imputada_dashboard.csv"): st.download_button("Serie con datos imputados (CSV)", open("serie_imputada_dashboard.csv", "rb"), "serie_imputada.csv", "text/csv")
    if os.path.exists("predicciones_dashboard.csv"): st.download_button("Predicciones 2025 (CSV)", open("predicciones_dashboard.csv", "rb"), "predicciones.csv", "text/csv")
    if os.path.exists("modelos_neumonia.csv"): st.download_button("Tabla de modelos (CSV)", open("modelos_neumonia.csv", "rb"), "modelos.csv", "text/csv")
with cb:
    st.markdown("**Código fuente del dashboard**")
    code = cargar_codigo_dashboard("dashboard.py")
    if code: st.download_button("Código implementación Dashboard (PY)", code, "dashboard_neumonia.py", "text/x-python")
    else: st.error("No encontré el archivo 'dashboard.py' en el directorio actual.")




