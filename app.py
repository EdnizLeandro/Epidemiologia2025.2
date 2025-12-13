# ============================================================
#  DASHBOARD EPIDEMIOLÓGICO COVID-PE
#  VISUALIZAÇÃO PROFISSIONAL A PARTIR DE CACHE
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
    **Modelos disponíveis:** SIR, SEIR, SEIRD e SEIRV  
    **Fonte:** Base epidemiológica tratada + simulações offline  
    **Performance:** Cache pré-computado (parquet)
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
    df["DATA"] = df["date"].dt.strftime("%d/%m/%Y")
    return df

df = carregar_cache()

# ------------------------------------------------------------
# SIDEBAR — CONTROLES
# ------------------------------------------------------------

st.sidebar.header("🎛️ CONTROLES")

municipios = sorted(df["municipio"].unique())
modelos = sorted(df["modelo"].unique())

sel_muni = st.sidebar.selectbox("MUNICÍPIO", municipios)
sel_modelo = st.sidebar.selectbox("MODELO EPIDEMIOLÓGICO", modelos)

datas = sorted(df["date"].unique())
ini, fim = st.sidebar.date_input(
    "PERÍODO",
    [datas[0], datas[-1]],
    min_value=datas[0],
    max_value=datas[-1]
)

# ------------------------------------------------------------
# FILTRAGEM
# ------------------------------------------------------------

mask = (
    (df["municipio"] == sel_muni) &
    (df["modelo"] == sel_modelo) &
    (df["date"] >= pd.to_datetime(ini)) &
    (df["date"] <= pd.to_datetime(fim))
)

dff = df[mask].copy()

if dff.empty:
    st.warning("NENHUM DADO PARA OS FILTROS SELECIONADOS.")
    st.stop()

# ------------------------------------------------------------
# VISÃO GERAL
# ------------------------------------------------------------

st.subheader("📌 VISÃO GERAL DA SIMULAÇÃO")

col1, col2, col3, col4 = st.columns(4)

col1.metric("MUNICÍPIO", sel_muni)
col2.metric("MODELO", sel_modelo)
col3.metric("DATA INICIAL", dff["DATA"].iloc[0])
col4.metric("DATA FINAL", dff["DATA"].iloc[-1])

# ------------------------------------------------------------
# GRÁFICO PRINCIPAL
# ------------------------------------------------------------

st.subheader("📉 EVOLUÇÃO DOS COMPARTIMENTOS")

cols_plot = [c for c in ["S","E","I","R","D"] if c in dff.columns]

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
    legend=dict(
        orientation="h",
        y=-0.25,
        x=0.5,
        xanchor="center"
    )
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# TABELA FINAL
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
    "BAIXAR RESULTADOS (CSV)",
    csv,
    file_name=f"{sel_muni}_{sel_modelo}_simulacao.csv",
    mime="text/csv"
)

st.caption("Dashboard otimizado com cache epidemiológico pré-processado.")
