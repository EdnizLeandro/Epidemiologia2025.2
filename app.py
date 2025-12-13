# ============================================================
# DASHBOARD COVID-PE — DADOS REAIS + MODELOS EPIDEMIOLÓGICOS
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ------------------------------------------------------------
# CONFIGURAÇÃO
# ------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_REAL = BASE_DIR / "covid_pe_seir_ready.parquet"
DATA_MODEL = BASE_DIR / "cache.parquet"

st.set_page_config(
    page_title="COVID-PE | Dados Reais e Modelos Epidemiológicos",
    layout="wide"
)

st.title("📊 COVID-19 EM PERNAMBUCO — DADOS REAIS E MODELAGEM")

# ------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------
@st.cache_data
def load_real():
    if not DATA_REAL.exists():
        st.error("Arquivo covid_pe_seir_ready.parquet não encontrado.")
        st.stop()
    df = pd.read_parquet(DATA_REAL)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_model():
    if not DATA_MODEL.exists():
        st.error("Arquivo cache.parquet não encontrado.")
        st.stop()
    df = pd.read_parquet(DATA_MODEL)
    df["date"] = pd.to_datetime(df["date"])
    return df

df_real = load_real()
df_model = load_model()

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.header("🎛️ CONTROLES")

municipios = ["Todos"] + sorted(df_real["municipio"].unique())
modelos = sorted(df_model["modelo"].unique())

sel_muni = st.sidebar.selectbox("MUNICÍPIO", municipios)
sel_modelo = st.sidebar.selectbox("MODELO", modelos)

min_date = min(df_real["date"].min(), df_model["date"].min()).date()
max_date = max(df_real["date"].max(), df_model["date"].max()).date()

ini, fim = st.sidebar.date_input("PERÍODO", [min_date, max_date])
ini, fim = pd.to_datetime(ini), pd.to_datetime(fim)

# ------------------------------------------------------------
# FILTRAGEM
# ------------------------------------------------------------
real = df_real[(df_real["date"] >= ini) & (df_real["date"] <= fim)]
model = df_model[(df_model["date"] >= ini) & (df_model["date"] <= fim)]

if sel_muni != "Todos":
    real = real[real["municipio"] == sel_muni]
    model = model[model["municipio"] == sel_muni]

model = model[model["modelo"] == sel_modelo]

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
    fig = px.line(real, x="date", y=["new_cases", "ma7"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 CASOS ACUMULADOS")
    fig = px.line(real, x="date", y="cum_cases")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 INFECTANTES ESTIMADOS (I_EST)")
    fig = px.line(real, x="date", y="I_est")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ABA 2 — MODELOS
# ============================================================
with tab2:
    st.subheader(f"📉 MODELO {sel_modelo}")
    fig = px.line(
        model,
        x="date",
        y=["S", "E", "I", "R"] if "E" in model else ["S", "I", "R"]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 PROPORÇÃO DA POPULAÇÃO")
    cols = [c for c in ["S","E","I","R","D","V"] if c in model.columns]
    total = model[cols].sum(axis=1)

    model_pct = model.copy()
    for c in cols:
        model_pct[c] = 100 * model[c] / total

    fig = px.area(model_pct, x="date", y=cols)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ABA 3 — COMPARAÇÃO
# ============================================================
with tab3:
    if not real.empty and not model.empty:
        comp = pd.merge(
            real[["date","I_est"]],
            model[["date","I"]],
            on="date",
            how="inner"
        )

        st.subheader("⚖️ INFECTANTES: REAL × MODELO")
        fig = px.line(comp, x="date", y=["I_est","I"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dados insuficientes para comparação.")

