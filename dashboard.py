import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Monitor Neumonía Chile",
    page_icon="🏥",
    layout="wide"
)

# --- 1. CARGA DE DATOS ---
@st.cache_data
def cargar_datos():
    # A. Cargar Historia (La base limpia)
    try:
        df_hist = pd.read_csv("base_neumonia_dashboard_READY.csv", encoding="utf-8")
        df_hist['fecha'] = pd.to_datetime(df_hist['fecha'])
    except FileNotFoundError:
        st.error("⚠️ Error Crítico: No encuentro el archivo 'base_neumonia_dashboard_READY.csv'.")
        st.stop()
    
    # B. Cargar Predicciones (El archivo nuevo generado en R)
    try:
        df_pred = pd.read_csv("predicciones_dashboard.csv", encoding="utf-8")
        df_pred['fecha'] = pd.to_datetime(df_pred['fecha'])
    except FileNotFoundError:
        st.warning("⚠️ Advertencia: No encontré 'predicciones_dashboard.csv'. El dashboard mostrará solo historia.")
        df_pred = pd.DataFrame()  # DataFrame vacío para evitar errores
        
    return df_hist, df_pred

# Ejecutar carga
df_hist, df_pred = cargar_datos()

# --- 2. BARRA LATERAL (FILTROS) ---
st.sidebar.header("🔍 Filtros de Visualización")

# A. Selector de Región
lista_regiones = sorted(df_hist['Region'].unique().tolist())
if "Total País" not in lista_regiones:
    lista_regiones.insert(0, "Total País")
else:
    lista_regiones.remove("Total País")
    lista_regiones.insert(0, "Total País")

region_sel = st.sidebar.selectbox("Seleccione Macro-Zona:", lista_regiones)

# B. Selector de Años (Zoom Histórico)
min_anio = int(df_hist['Año'].min())
max_anio = int(df_hist['Año'].max())

# Si el mínimo es mayor que 2021, usamos ese como inicio para evitar errores
default_inicio = 2017 if min_anio <= 2017 else min_anio

anios_sel = st.sidebar.slider(
    "Rango Histórico:",
    min_anio,
    max_anio,
    (default_inicio, max_anio)
)

# --- 3. LÓGICA DE FILTRADO ---

# A. Filtrar Historia
df_hist_zoom = df_hist[(df_hist['Año'] >= anios_sel[0]) & (df_hist['Año'] <= anios_sel[1])]

if region_sel == "Total País":
    # Si es Total País, agrupamos sumando las regiones (si el csv no traía la fila Total)
    if "Total País" in df_hist['Region'].unique():
        df_hist_plot = df_hist_zoom[df_hist_zoom['Region'] == "Total País"]
    else:
        df_hist_plot = df_hist_zoom.groupby(['fecha', 'Año', 'Semana'])['Casos'].sum().reset_index()
    
    # Filtrar Predicción (Total País)
    if not df_pred.empty:
        df_pred_plot = df_pred[df_pred['Region'] == "Total País"]
    else:
        df_pred_plot = pd.DataFrame()
else:
    # Región específica
    df_hist_plot = df_hist_zoom[df_hist_zoom['Region'] == region_sel]
    
    if not df_pred.empty:
        df_pred_plot = df_pred[df_pred['Region'] == region_sel]
    else:
        df_pred_plot = pd.DataFrame()

# --- 4. KPIs (INDICADORES CLAVE) ---
st.title(f"🦠 Vigilancia Epidemiológica: {region_sel}")
st.markdown("Monitor de atenciones de urgencia por Neumonía (CIE-10 J12-J18). Red Pública de Salud.")

if not df_hist_plot.empty:
    ultimo_dato_real = df_hist_plot.iloc[-1]  # Última fila histórica
    casos_actuales = int(ultimo_dato_real['Casos'])
    fecha_actual = ultimo_dato_real['fecha'].strftime("%d-%m-%Y")
    promedio_hist = int(df_hist_plot['Casos'].mean())
    
    col1, col2, col3, col4 = st.columns(4)
    
    # 🔴 AQUÍ ESTABA EL ERROR: antes decía fecha_dato (que no existe)
    col1.metric("Fecha Último Dato", fecha_actual)
    col2.metric("Casos Última Semana", f"{casos_actuales:,}".replace(",", "."))
    col3.metric("Promedio Periodo", f"{promedio_hist:,}".replace(",", "."))
    
    if not df_pred_plot.empty:
        peak_proyectado = int(df_pred_plot['Casos'].max())
        col4.metric("Peak Proyectado 2025", f"{peak_proyectado:,}".replace(",", "."), delta_color="inverse")
    else:
        col4.metric("Peak Proyectado", "N/A")

st.divider()

# --- 5. GRÁFICO PRINCIPAL (EVOLUCIÓN + PRONÓSTICO) ---
st.subheader("📈 Evolución Temporal y Pronóstico (Modelo SARIMAX)")

fig = go.Figure()

# CAPA 1: Historia (Línea Azul)
fig.add_trace(go.Scatter(
    x=df_hist_plot['fecha'], 
    y=df_hist_plot['Casos'],
    mode='lines',
    name='Datos Observados',
    line=dict(color='#2c3e50', width=2.5)
))

# CAPA 2: Predicción (Línea Verde + Sombra)
if not df_pred_plot.empty:
    # Sombra de Confianza (95%)
    fig.add_trace(go.Scatter(
        x=pd.concat([df_pred_plot['fecha'], df_pred_plot['fecha'][::-1]]),
        y=pd.concat([df_pred_plot['Upper'], df_pred_plot['Lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(39, 174, 96, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Intervalo Confianza 95%'
    ))
    
    # Línea Central de Pronóstico
    fig.add_trace(go.Scatter(
        x=df_pred_plot['fecha'],
        y=df_pred_plot['Casos'],
        mode='lines',
        name='Pronóstico (Esperado)',
        line=dict(color='#27ae60', width=3, dash='solid')
    ))

# CAPA 3: Zona de Pandemia (Sombreado Gris)
fig.add_vrect(
    x0="2020-03-15", x1="2021-12-31",
    fillcolor="gray", opacity=0.1,
    layer="below", line_width=0,
    annotation_text="COVID-19 (Intervención)", annotation_position="top left"
)

fig.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Nº Atenciones Semanales",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=500,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# --- 6. GRÁFICO ESTACIONAL (COMPARATIVA INVIERNOS) ---
st.subheader("❄️ Comparativa Estacional (Ciclos Anuales)")
st.markdown("Este gráfico permite comparar la intensidad del invierno actual con años anteriores.")

fig_season = px.line(
    df_hist_plot, 
    x='Semana', 
    y='Casos', 
    color='Año',
    title="Curvas Epidemiológicas Superpuestas",
    color_discrete_sequence=px.colors.qualitative.Dark24
)

fig_season.update_layout(
    xaxis_title="Semana Epidemiológica (1-52)",
    yaxis_title="Casos",
    hovermode="x unified"
)

st.plotly_chart(fig_season, use_container_width=True)

# --- 7. TABLA DE DATOS (EXPANDIBLE) ---
with st.expander("📥 Ver Datos Detallados (Historia + Proyección)"):
    tab1, tab2 = st.tabs(["Proyección 2025", "Historia Reciente"])
    
    with tab1:
        if not df_pred_plot.empty:
            st.dataframe(
                df_pred_plot[['fecha', 'Region', 'Casos', 'Lower', 'Upper']]
                .style.format({"Casos": "{:.0f}", "Lower": "{:.0f}", "Upper": "{:.0f}"})
            )
        else:
            st.info("No hay proyección disponible para esta selección.")
            
    with tab2:
        st.dataframe(
            df_hist_plot.sort_values('fecha', ascending=False).head(100)
        )
