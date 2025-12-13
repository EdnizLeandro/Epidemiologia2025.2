# ============================================================
# DASHBOARD EPIDEMIOLÓGICO COVID-PE
# VISUALIZAÇÃO A PARTIR DE CACHE (ALTA PERFORMANCE)
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ------------------------------------------------------------
# CONFIGURAÇÃO DO APP
# ------------------------------------------------------------

st.set_page_config(
    page_title="COVID-PE | MODELOS EPIDEMIOLÓGICOS",
    layout="wide"
)

st.title("📊 COVID-19 EM PERNAMBUCO — MODELAGEM EPIDEMIOLÓGICA")
st.markdown(
    """
    **MODELOS:** SIR, SEIR, SEIRD, SEIRV  
    **ARQUITETURA:** SIMULAÇÃO OFFLINE + DASHBOARD ONLINE  
    **DESEMPENHO:** CACHE PARQUET PRÉ-COMPUTADO
    """
)

# ------------------------------------------------------------
# CARREGAR CACHE
# ------------------------------------------------------------

CACHE_FILE = Path(__file__).parent / "cache.parquet"

@st.cache_data
def carregar_cache():
    if not CACHE_FILE.exists():
        st.error("Arquivo cache.parquet não encontrado. Execute gerar_cache.py primeiro.")
        st.stop()

    df = pd.read_parquet(CACHE_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = carregar_cache()

# ------------------------------------------------------------
# SIDEBAR — CONTROLES
# ------------------------------------------------------------

st.sidebar.header("🎛️ CONTROLES")

municipios = ["TODOS"] + sorted(df["municipio"].unique())
modelos = sorted(df["modelo"].unique())

sel_muni = st.sidebar.selectbox("MUNICÍPIO", municipios)
sel_modelo = st.sidebar.selectbox("MODELO EPIDEMIOLÓGICO", modelos)

data_min = df["date"].min().date()
data_max = df["date"].max().date()

ini, fim = st.sidebar.date_input(
    "PERÍODO",
    [data_min, data_max],
    min_value=data_min,
    max_value=data_max
)

# ------------------------------------------------------------
# LINHA DO TEMPO COMPLETA (REGRA DE OURO)
# ------------------------------------------------------------

datas_completas = pd.date_range(
    df["date"].min(),
    df["date"].max(),
    freq="D"
)

# ------------------------------------------------------------
# FILTRAGEM + AGREGAÇÃO CORRETA
# ------------------------------------------------------------

if sel_muni == "TODOS":
    dff = (
        df[df["modelo"] == sel_modelo]
        .groupby("date")[["S", "E", "I", "R", "D"]]
        .sum()
        .reindex(datas_completas, fill_value=0)
        .reset_index()
        .rename(columns={"index": "date"})
    )
else:
    dff = (
        df[
            (df["municipio"] == sel_muni) &
            (df["modelo"] == sel_modelo)
        ]
        .set_index("date")
        .reindex(datas_completas)
        .reset_index()
        .rename(columns={"index": "date"})
    )

# Datas BR
dff["DATA"] = dff["date"].dt.strftime("%d/%m/%Y")

# Aplicar período
mask = (
    (dff["date"] >= pd.to_datetime(ini)) &
    (dff["date"] <= pd.to_datetime(fim))
)
dff = dff[mask]

if dff.empty:
    st.warning("NENHUM DADO PARA OS FILTROS SELECIONADOS.")
    st.stop()

# ------------------------------------------------------------
# VISÃO GERAL
# ------------------------------------------------------------

st.subheader("📌 VISÃO GERAL")

c1, c2, c3, c4 = st.columns(4)

c1.metric("MUNICÍPIO", sel_muni)
c2.metric("MODELO", sel_modelo)
c3.metric("DATA INICIAL", dff["DATA"].iloc[0])
c4.metric("DATA FINAL", dff["DATA"].iloc[-1])

# ------------------------------------------------------------
# GRÁFICO PRINCIPAL
# ------------------------------------------------------------

st.subheader("📉 EVOLUÇÃO DOS COMPARTIMENTOS")

cols_plot = [c for c in ["S", "E", "I", "R", "D"] if c in dff.columns]

fig = px.line(
    dff,
    x="DATA",
    y=cols_plot,
    labels={
        "value": "POPULAÇÃO",
        "DATA": "DATA",
        "variable": "COMPARTIMENTO"
    },
    title=f"MODELO {sel_modelo} — {sel_muni}"
)

fig.update_layout(
    template="plotly_white",
    title_font=dict(size=22),
    xaxis=dict(type="category"),  # garante todas as datas
    legend=dict(
        orientation="h",
        y=-0.25,
        x=0.5,
        xanchor="center"
    )
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# TABELA NUMÉRICA
# ------------------------------------------------------------

with st.expander("📄 VER DADOS NUMÉRICOS"):
    st.dataframe(
        dff[["DATA"] + cols_plot].reset_index(drop=True),
        use_container_width=True
    )

# ------------------------------------------------------------
# DOWNLOAD
# ------------------------------------------------------------

st.subheader("📥 EXPORTAÇÃO")

csv = dff[["DATA"] + cols_plot].to_csv(index=False).encode("utf-8")

st.download_button(
    label="BAIXAR RESULTADOS (CSV)",
    data=csv,
    file_name=f"{sel_muni}_{sel_modelo}_simulacao.csv",
    mime="text/csv"
)

st.caption("Dashboard epidemiológico otimizado com cache pré-processado.")
