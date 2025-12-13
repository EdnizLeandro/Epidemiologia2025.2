# ============================================================
# COVID-PE — DADOS REAIS + MODELOS EPIDEMIOLÓGICOS (ROBUSTO)
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ------------------------------------------------------------
# CONFIGURAÇÃO DE DEBUG (MOSTRA ERROS NA TELA)
# ------------------------------------------------------------
st.set_option("client.showErrorDetails", True)

# ------------------------------------------------------------
# CAMINHOS
# ------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_REAL = BASE_DIR / "covid_pe_seir_ready.parquet"
DATA_MODEL = BASE_DIR / "cache.parquet"

st.set_page_config(
    page_title="COVID-PE | Dados Reais e Modelos",
    layout="wide"
)

st.title("📊 COVID-19 EM PERNAMBUCO — DADOS REAIS E MODELOS EPIDEMIOLÓGICOS")

# ------------------------------------------------------------
# LOADERS DEFENSIVOS
# ------------------------------------------------------------
@st.cache_data
def load_real():
    try:
        if not DATA_REAL.exists():
            raise FileNotFoundError("covid_pe_seir_ready.parquet não encontrado.")

        df = pd.read_parquet(DATA_REAL)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["municipio"] = df["municipio"].astype(str).str.upper().str.strip()
        df = df.dropna(subset=["date"])
        return df

    except Exception as e:
        st.error("❌ ERRO AO CARREGAR DADOS REAIS")
        st.exception(e)
        st.stop()


@st.cache_data
def load_model():
    try:
        if not DATA_MODEL.exists():
            raise FileNotFoundError("cache.parquet não encontrado.")

        df = pd.read_parquet(DATA_MODEL)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["municipio"] = df["municipio"].astype(str).str.upper().str.strip()
        df["modelo"] = df["modelo"].astype(str).str.upper().str.strip()
        df = df.dropna(subset=["date"])
        return df

    except Exception as e:
        st.error("❌ ERRO AO CARREGAR DADOS DO MODELO")
        st.exception(e)
        st.stop()


df_real = load_real()
df_model = load_model()

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.header("🎛️ CONTROLES")

municipios = sorted(set(df_real["municipio"]).intersection(df_model["municipio"]))
municipios = ["TODOS"] + municipios

modelos = sorted(df_model["modelo"].unique())

sel_muni = st.sidebar.selectbox("MUNICÍPIO", municipios)
sel_modelo = st.sidebar.selectbox("MODELO EPIDEMIOLÓGICO", modelos)

min_date = min(df_real["date"].min(), df_model["date"].min()).date()
max_date = max(df_real["date"].max(), df_model["date"].max()).date()

ini, fim = st.sidebar.date_input(
    "PERÍODO",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

ini = pd.to_datetime(ini)
fim = pd.to_datetime(fim)

# ------------------------------------------------------------
# FILTRAGEM SEGURA
# ------------------------------------------------------------
real = df_real[(df_real["date"] >= ini) & (df_real["date"] <= fim)].copy()
model = df_model[(df_model["date"] >= ini) & (df_model["date"] <= fim)].copy()

if sel_muni != "TODOS":
    real = real[real["municipio"] == sel_muni]
    model = model[model["municipio"] == sel_muni]

model = model[model["modelo"] == sel_modelo]

# ------------------------------------------------------------
# CHECAGENS CRÍTICAS
# ------------------------------------------------------------
if real.empty:
    st.warning("⚠️ Dados REAIS vazios após filtros.")
    st.stop()

if model.empty:
    st.warning("⚠️ Dados do MODELO vazios após filtros.")
    st.stop()

# ------------------------------------------------------------
# ABAS
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 DADOS OBSERVADOS",
    "🧮 MODELO EPIDEMIOLÓGICO",
    "⚖️ REAL × MODELO"
])

# ============================================================
# ABA 1 — DADOS OBSERVADOS
# ============================================================
with tab1:
    st.subheader("📈 CASOS DIÁRIOS E MÉDIA MÓVEL")

    if {"new_cases", "ma7"}.issubset(real.columns):
        fig = px.line(real, x="date", y=["new_cases", "ma7"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Colunas new_cases / ma7 não disponíveis.")

    st.subheader("📈 CASOS ACUMULADOS")
    if "cum_cases" in real.columns:
        fig = px.line(real, x="date", y="cum_cases")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Coluna cum_cases não disponível.")

    st.subheader("📉 INFECTANTES ESTIMADOS (I_EST)")
    if "I_est" in real.columns:
        fig = px.line(real, x="date", y="I_est")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Coluna I_est não disponível.")

# ============================================================
# ABA 2 — MODELOS EPIDEMIOLÓGICOS
# ============================================================
with tab2:
    st.subheader(f"📉 MODELO {sel_modelo}")

    compartimentos = [c for c in ["S", "E", "I", "R", "D", "V"] if c in model.columns]

    if not compartimentos:
        st.warning("Modelo não possui compartimentos para plotar.")
    else:
        fig = px.line(model, x="date", y=compartimentos)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 PROPORÇÃO DA POPULAÇÃO (%)")
        total = model[compartimentos].sum(axis=1)

        model_pct = model.copy()
        for c in compartimentos:
            model_pct[c] = 100 * model[c] / total

        fig = px.area(model_pct, x="date", y=compartimentos)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ABA 3 — COMPARAÇÃO REAL × MODELO
# ============================================================
with tab3:
    if "I_est" in real.columns and "I" in model.columns:
        comp = pd.merge(
            real[["date", "I_est"]],
            model[["date", "I"]],
            on="date",
            how="inner"
        )

        if comp.empty:
            st.warning("Sem interseção temporal para comparação.")
        else:
            st.subheader("⚖️ INFECTANTES — REAL × MODELO")
            fig = px.line(comp, x="date", y=["I_est", "I"])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Comparação indisponível para este modelo.")
