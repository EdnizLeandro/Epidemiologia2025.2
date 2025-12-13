# ============================================================
# DASHBOARD COVID-PE — DADOS EPIDEMIOLÓGICOS + MODELO SEIR
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

# ------------------------------
# CONFIGURAÇÕES GERAIS
# ------------------------------
BASE_DIR = Path(__file__).parent
DATA_PARQUET = BASE_DIR / "covid_pe_seir_ready.parquet"

st.set_page_config(
    layout="wide",
    page_title="COVID-PE Dashboard + Modelo SEIR"
)

st.title("📊 Dashboard COVID-PE — Dados Epidemiológicos + Modelo SEIR")

# Cores padrão
COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#17becf"
COLOR_TREND = "#ff7f0e"
COLOR_GRAY = "#7f7f7f"
COLOR_RED = "#d62728"


def apply_plot_styling(fig):
    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=22, color=COLOR_PRIMARY),
        font=dict(size=14),
        legend=dict(
            title="",
            orientation="h",
            y=-0.25,
            x=0.5,
            xanchor="center"
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig


# ------------------------------
# CARREGAR DADOS
# ------------------------------
@st.cache_data
def load_data_parquet():
    if not DATA_PARQUET.exists():
        st.error("Arquivo covid_pe_seir_ready.parquet não encontrado.")
        st.stop()

    df = pd.read_parquet(DATA_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    return df


df = load_data_parquet()

# ------------------------------
# SIDEBAR — FILTROS
# ------------------------------
st.sidebar.header("🔎 Filtros e Parâmetros")

munis = sorted(df["municipio"].dropna().unique())
sel_muni = st.sidebar.selectbox("Município", ["Todos"] + munis)

min_date, max_date = df["date"].min(), df["date"].max()
start_date, end_date = st.sidebar.date_input("Período", [min_date, max_date])

beta = st.sidebar.slider("β — Taxa de transmissão", 0.0, 2.0, 0.6, 0.01)
sigma = st.sidebar.slider("σ — Taxa de incubação", 0.0, 1.0, 1/5, 0.01)
gamma = st.sidebar.slider("γ — Taxa de recuperação", 0.0, 1.0, 1/7, 0.01)

init_days = st.sidebar.number_input("Dias p/ estimar I0", 1, 60, 7)
run_seir = st.sidebar.button("▶ Rodar simulação SEIR")

# ------------------------------
# FILTRAR DADOS
# ------------------------------
mask = (df["date"] >= start_date) & (df["date"] <= end_date)
dff = df[mask].copy()

if sel_muni != "Todos":
    dff = dff[dff["municipio"] == sel_muni]

if dff.empty:
    st.stop()

# ------------------------------
# RESUMO
# ------------------------------
st.header(f"📌 Resumo — {sel_muni if sel_muni != 'Todos' else 'Estado inteiro'}")

col1, col2, col3 = st.columns(3)
col1.metric("Casos novos", int(dff["new_cases"].sum()))
col2.metric("Pico diário", int(dff["new_cases"].max()))
col3.metric("População", int(dff["population"].median()))

# ============================================================
# 📊 GRÁFICOS — DADOS OBSERVADOS
# ============================================================

# 1️⃣ Casos diários + média móvel
st.subheader("📈 Casos Diários + Média Móvel (7 dias)")
fig = px.line(dff, x="date", y=["new_cases", "ma7"])
st.plotly_chart(apply_plot_styling(fig), use_container_width=True)

# 2️⃣ Casos acumulados
st.subheader("📈 Casos Acumulados")
fig = px.line(dff, x="date", y="cum_cases", color_discrete_sequence=[COLOR_TREND])
st.plotly_chart(apply_plot_styling(fig), use_container_width=True)

# 3️⃣ Infectantes estimados
st.subheader("📉 Infectantes Estimados (I_est)")
fig = px.line(dff, x="date", y="I_est", color_discrete_sequence=[COLOR_SECONDARY])
st.plotly_chart(apply_plot_styling(fig), use_container_width=True)

# 4️⃣ Taxa de crescimento
st.subheader("📊 Taxa de Crescimento Diário (%)")
dff["growth_rate"] = dff["new_cases"].pct_change() * 100
fig = px.line(dff, x="date", y="growth_rate", color_discrete_sequence=[COLOR_RED])
st.plotly_chart(apply_plot_styling(fig), use_container_width=True)

# 5️⃣ Proporção da população infectada
st.subheader("📊 Proporção da População Infectada (%)")
dff["infected_pct"] = 100 * dff["cum_cases"] / dff["population"]
fig = px.line(dff, x="date", y="infected_pct", color_discrete_sequence=[COLOR_GRAY])
st.plotly_chart(apply_plot_styling(fig), use_container_width=True)

# ============================================================
# 🧮 MODELO SEIR
# ============================================================

def run_seir_simulation(N, E0, I0, R0, beta, sigma, gamma, days):
    S0 = N - E0 - I0 - R0
    S, E, I, R = [S0], [E0], [I0], [R0]

    for _ in range(days):
        S.append(S[-1] - beta * S[-1] * I[-1] / N)
        E.append(E[-1] + beta * S[-1] * I[-1] / N - sigma * E[-1])
        I.append(I[-1] + sigma * E[-1] - gamma * I[-1])
        R.append(R[-1] + gamma * I[-1])

    return pd.DataFrame({
        "date": pd.date_range(dff["date"].max(), periods=len(S)),
        "S": S, "E": E, "I": I, "R": R
    })


if run_seir:
    I0 = max(1, int(dff.tail(init_days)["new_cases"].sum()))
    E0 = max(I0 * 2, int(dff.tail(int(1/sigma))["new_cases"].sum()))
    R0 = int(dff["R_est"].iloc[-1]) if "R_est" in dff else 0

    sim = run_seir_simulation(
        int(dff["population"].median()), E0, I0, R0,
        beta, sigma, gamma, 180
    )

    st.subheader("📉 Simulação SEIR — Valores Absolutos")
    fig = px.line(sim, x="date", y=["S", "E", "I", "R"])
    st.plotly_chart(apply_plot_styling(fig), use_container_width=True)

    st.subheader("📊 Simulação SEIR — Proporção da População")
    sim_pct = sim.copy()
    N = sim_pct[["S","E","I","R"]].sum(axis=1)
    for c in ["S","E","I","R"]:
        sim_pct[c] = 100 * sim_pct[c] / N

    fig = px.area(sim_pct, x="date", y=["S","E","I","R"])
    st.plotly_chart(apply_plot_styling(fig), use_container_width=True)
