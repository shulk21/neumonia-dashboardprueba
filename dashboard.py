import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import os
from datetime import datetime

# -------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# -------------------------------------------------------------
st.set_page_config(
    page_title="Monitor Neumonía Chile",
    layout="wide"
)

# -------------------------------------------------------------
# 1. CARGA DE DATOS
# -------------------------------------------------------------
@st.cache_data
def cargar_datos():
    # 1) Historia original
    try:
        df_hist = pd.read_csv("base_neumonia_dashboard_READY.csv", encoding="utf-8")
        df_hist["fecha"] = pd.to_datetime(df_hist["fecha"])
    except FileNotFoundError:
        st.error("Error crítico: no encuentro 'base_neumonia_dashboard_READY.csv'.")
        st.stop()

    # 2) Predicciones
    try:
        df_pred = pd.read_csv("predicciones_dashboard.csv", encoding="utf-8")
        df_pred["fecha"] = pd.to_datetime(df_pred["fecha"])
    except FileNotFoundError:
        st.warning("No encontré 'predicciones_dashboard.csv'. El dashboard mostrará solo historia.")
        df_pred = pd.DataFrame()

    # 3) Modelos ARIMA por región
    try:
        df_modelos = pd.read_csv("modelos_neumonia.csv", encoding="utf-8")
    except FileNotFoundError:
        st.warning("No encontré 'modelos_neumonia.csv'. No se mostrará el modelo ARIMA en el título.")
        df_modelos = pd.DataFrame()

    # 4) Serie histórica con imputación de pandemia
    try:
        df_imput = pd.read_csv("serie_imputada_dashboard.csv", encoding="utf-8")
        df_imput["fecha"] = pd.to_datetime(df_imput["fecha"])
    except FileNotFoundError:
        st.warning("No encontré 'serie_imputada_dashboard.csv'. No se mostrará la serie imputada.")
        df_imput = pd.DataFrame()

    return df_hist, df_pred, df_modelos, df_imput


@st.cache_data
def cargar_codigo_dashboard(nombre_archivo="dashboard.py"):
    """Lee el propio archivo del dashboard para ofrecerlo como descarga."""
    if os.path.exists(nombre_archivo):
        with open(nombre_archivo, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return None


df_hist, df_pred, df_modelos, df_imput = cargar_datos()

# -------------------------------------------------------------
# 2. FILTROS (CENTRADOS + RANGO DE FECHAS)
# -------------------------------------------------------------
with st.container():
    col_margin_izq, col_centro, col_margin_der = st.columns([1, 4, 1])
    with col_centro:
        st.subheader("Filtros de visualización")

        col_filtro_1, col_filtro_2 = st.columns([1, 2])

        # ---------------- Región ----------------
        with col_filtro_1:
            lista_regiones = sorted(df_hist["Region"].unique().tolist())
            if "Total País" not in lista_regiones:
                lista_regiones.insert(0, "Total País")
            else:
                lista_regiones.remove("Total País")
                lista_regiones.insert(0, "Total País")

            region_sel = st.selectbox("Seleccione región:", lista_regiones)

        # ---------------- Rango de fechas ----------------
        with col_filtro_2:
            fecha_min_hist = df_hist["fecha"].min()
            fecha_max_hist = df_hist["fecha"].max()

            if not df_pred.empty:
                fecha_max_pred = df_pred["fecha"].max()
                fecha_max_total = max(fecha_max_hist, fecha_max_pred)
            else:
                fecha_max_total = fecha_max_hist

            # Inicio por defecto: máximo entre primera fecha y 2017-01-01
            default_start = max(fecha_min_hist, pd.Timestamp("2017-01-01"))

            rango_fechas = st.slider(
                "Rango temporal:",
                min_value=fecha_min_hist.to_pydatetime(),
                max_value=fecha_max_total.to_pydatetime(),
                value=(
                    default_start.to_pydatetime(),
                    fecha_max_total.to_pydatetime()
                ),
                format="YYYY-MM-DD"
            )

st.markdown("---")

fecha_ini, fecha_fin = rango_fechas
fecha_ini = pd.to_datetime(fecha_ini)
fecha_fin = pd.to_datetime(fecha_fin)

# -------------------------------------------------------------
# 3. FILTRADO DE DATOS POR REGIÓN + FECHAS
# -------------------------------------------------------------
# Historia original por región
if region_sel == "Total País":
    # si la base no trae explícitamente "Total País", agregamos sumando regiones
    if "Total País" in df_hist["Region"].unique():
        df_hist_region = df_hist[df_hist["Region"] == "Total País"].copy()
    else:
        df_hist_region = (
            df_hist
            .groupby(["fecha", "Año", "Semana"], as_index=False)["Casos"]
            .sum()
        )
        df_hist_region["Region"] = "Total País"
else:
    df_hist_region = df_hist[df_hist["Region"] == region_sel].copy()

df_hist_plot = df_hist_region[
    (df_hist_region["fecha"] >= fecha_ini) &
    (df_hist_region["fecha"] <= fecha_fin)
].copy()

# Predicciones
if not df_pred.empty:
    if region_sel == "Total País":
        df_pred_region = df_pred[df_pred["Region"] == "Total País"].copy()
    else:
        df_pred_region = df_pred[df_pred["Region"] == region_sel].copy()

    df_pred_plot = df_pred_region[
        (df_pred_region["fecha"] >= fecha_ini) &
        (df_pred_region["fecha"] <= fecha_fin)
    ].copy()
else:
    df_pred_plot = pd.DataFrame()

# Serie imputada
if not df_imput.empty:
    if region_sel == "Total País":
        df_imp_region = df_imput[df_imput["Region"] == "Total País"].copy()
    else:
        df_imp_region = df_imput[df_imput["Region"] == region_sel].copy()

    df_imp_region["Imputado"] = (
        df_imp_region["Imputado"]
        .astype(str)
        .str.upper()
        .isin(["TRUE", "1"])
    )

    df_imp_plot = df_imp_region[
        (df_imp_region["fecha"] >= fecha_ini) &
        (df_imp_region["fecha"] <= fecha_fin)
    ].copy()
else:
    df_imp_plot = pd.DataFrame()

# -------------------------------------------------------------
# 4. TÍTULO Y MODELO ARIMA
# -------------------------------------------------------------
modelo_limpio = None
if not df_modelos.empty:
    fila_mod = df_modelos.loc[df_modelos["Region"] == region_sel]
    if not fila_mod.empty:
        modelo_txt = str(fila_mod["Modelo"].iloc[0])
        modelo_limpio = (
            modelo_txt
            .replace("Regression with ", "")
            .replace(" regression with ", "")
            .replace(" errors", "")
            .strip()
        )
        m = re.search(r"ARIMA.*", modelo_limpio)
        if m:
            modelo_limpio = m.group(0)

if modelo_limpio:
    st.title(f"Vigilancia epidemiológica: {region_sel} | Modelo: {modelo_limpio}")
else:
    st.title(f"Vigilancia epidemiológica: {region_sel}")

st.markdown("Monitor de atenciones de urgencia por neumonía. Red pública de salud.")

# -------------------------------------------------------------
# 5. KPIs
# -------------------------------------------------------------
if not df_hist_plot.empty:
    promedio_hist = int(df_hist_plot["Casos"].mean())
    total_hist = int(df_hist_plot["Casos"].sum())
    peak_hist = int(df_hist_plot["Casos"].max())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Promedio histórico", f"{promedio_hist:,}".replace(",", "."))
    col2.metric("Total acumulado", f"{total_hist:,}".replace(",", "."))
    col3.metric("Peak histórico", f"{peak_hist:,}".replace(",", "."))

    if not df_pred_plot.empty:
        peak_proyectado = int(df_pred_plot["Casos"].max())
        col4.metric("Peak proyectado 2025", f"{peak_proyectado:,}".replace(",", "."), delta_color="inverse")
    else:
        col4.metric("Peak proyectado 2025", "N/A")

st.divider()

# -------------------------------------------------------------
# 6. GRÁFICO PRINCIPAL (CENTRADO + TOGGLE PANDEMIA)
# -------------------------------------------------------------
with st.container():
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        st.subheader("Evolución temporal y pronóstico")

        # Toggle para mostrar datos originales en pandemia
        tgl_left, tgl_mid, tgl_right = st.columns([1, 3, 6])
        with tgl_mid:
            mostrar_original_pandemia = st.toggle(
                "Visualizar datos originales en pandemia",
                value=False
            )

        fig = go.Figure()

        pand_ini = pd.to_datetime("2020-03-15")
        pand_fin = pd.to_datetime("2021-12-31")

        # 1) Línea base (serie imputada si existe, si no la original)
        if not df_imp_plot.empty:
            # serie corregida completa
            fig.add_trace(go.Scatter(
                x=df_imp_plot["fecha"],
                y=df_imp_plot["Casos"],
                mode="lines",
                name="Datos observados",
                line=dict(color="#2c3e50", width=2.5)
            ))

            # tramo imputado en pandemia
            df_imp_only = df_imp_plot[
                (df_imp_plot["Imputado"]) &
                (df_imp_plot["fecha"] >= pand_ini) &
                (df_imp_plot["fecha"] <= pand_fin)
            ]
            if not df_imp_only.empty:
                fig.add_trace(go.Scatter(
                    x=df_imp_only["fecha"],
                    y=df_imp_only["Casos"],
                    mode="lines",
                    name="Datos imputados",
                    line=dict(color="#e74c3c", width=2.5)
                ))
        else:
            # si no hay serie imputada, usamos la original
            fig.add_trace(go.Scatter(
                x=df_hist_plot["fecha"],
                y=df_hist_plot["Casos"],
                mode="lines",
                name="Datos observados",
                line=dict(color="#2c3e50", width=2.5)
            ))

        # 2) Datos originales en pandemia (solo si toggle activado)
        if mostrar_original_pandemia and not df_hist_plot.empty:
            df_pand_orig = df_hist_plot[
                (df_hist_plot["fecha"] >= pand_ini) &
                (df_hist_plot["fecha"] <= pand_fin)
            ]
            if not df_pand_orig.empty:
                fig.add_trace(go.Scatter(
                    x=df_pand_orig["fecha"],
                    y=df_pand_orig["Casos"],
                    mode="lines",
                    name="Datos originales pandemia",
                    line=dict(color="rgba(80,80,80,0.7)", width=2, dash="dot")
                ))

        # 3) Pronóstico
        if not df_pred_plot.empty:
            # banda de confianza 90 %
            fig.add_trace(go.Scatter(
                x=pd.concat([df_pred_plot["fecha"], df_pred_plot["fecha"][::-1]]),
                y=pd.concat([df_pred_plot["Upper"], df_pred_plot["Lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(39, 174, 96, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="Intervalo 90%"
            ))
            # media pronosticada
            fig.add_trace(go.Scatter(
                x=df_pred_plot["fecha"],
                y=df_pred_plot["Casos"],
                mode="lines",
                name="Pronóstico 2025",
                line=dict(color="#27ae60", width=3, dash="solid")
            ))

        # 4) Sombreado pandemia
        fig.add_vrect(
            x0="2020-03-15", x1="2021-12-31",
            fillcolor="gray", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Pandemia COVID-19",
            annotation_position="top left"
        )

        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Casos semanales",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            height=500,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 7. GRÁFICO ESTACIONAL (CENTRADO)
# -------------------------------------------------------------
with st.container():
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        st.subheader("Comparativa estacional")

        fig_season = px.line(
            df_hist_plot,
            x="Semana",
            y="Casos",
            color="Año",
            title="Curvas epidemiológicas superpuestas",
            color_discrete_sequence=px.colors.qualitative.Dark24
        )

        fig_season.update_layout(
            xaxis_title="Semana (1-52)",
            yaxis_title="Casos",
            hovermode="x unified"
        )

        st.plotly_chart(fig_season, use_container_width=True)

# -------------------------------------------------------------
# 8. TABLAS
# -------------------------------------------------------------
with st.expander("Ver datos detallados"):
    tab1, tab2 = st.tabs(["Proyección", "Historia"])

    with tab1:
        if not df_pred_plot.empty:
            st.dataframe(
                df_pred_plot[["fecha", "Region", "Casos", "Lower", "Upper"]]
                .style.format({"Casos": "{:.0f}", "Lower": "{:.0f}", "Upper": "{:.0f}"})
            )
        else:
            st.info("Sin proyección disponible para este filtro.")

    with tab2:
        st.dataframe(
            df_hist_plot.sort_values("fecha", ascending=False).head(100)
        )

# -------------------------------------------------------------
# 9. DESCARGAS (4 BASES + CÓDIGO)
# -------------------------------------------------------------
st.divider()
st.subheader("Descarga de recursos")

col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown("**Bases de datos disponibles**")

    if os.path.exists("base_neumonia_dashboard_READY.csv"):
        with open("base_neumonia_dashboard_READY.csv", "rb") as f:
            st.download_button(
                "Serie con datos observados (CSV)",
                f,
                "base_neumonia.csv",
                "text/csv"
            )

    if os.path.exists("serie_imputada_dashboard.csv"):
        with open("serie_imputada_dashboard.csv", "rb") as f:
            st.download_button(
                "Serie con datos imputados (CSV)",
                f,
                "serie_imputada.csv",
                "text/csv"
            )

    if os.path.exists("predicciones_dashboard.csv"):
        with open("predicciones_dashboard.csv", "rb") as f:
            st.download_button(
                "Predicciones 2025 (CSV)",
                f,
                "predicciones.csv",
                "text/csv"
            )

    if os.path.exists("modelos_neumonia.csv"):
        with open("modelos_neumonia.csv", "rb") as f:
            st.download_button(
                "Tabla de modelos (CSV)",
                f,
                "modelos.csv",
                "text/csv"
            )

with col_b:
    st.markdown("**Código fuente del dashboard**")
    codigo = cargar_codigo_dashboard("dashboard.py")
    if codigo:
        st.download_button(
            label="Código implementación Dashboard (PY)",
            data=codigo,
            file_name="dashboard_neumonia.py",
            mime="text/x-python"
        )
    else:
        st.error("No encontré el archivo 'dashboard.py' en el directorio actual.")



