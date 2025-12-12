# ============================================================
#  STREAMLIT — DASHBOARD COVID-PE + MODELO SEIR COMPLETO
#  Usa APENAS o arquivo covid_pe_seir_ready.parquet
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
        st.error("Arquivo covid_pe_seir_ready.parquet não encontrado na pasta do app.")
        st.stop()

    df = pd.read_parquet(DATA_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    return df


df = load_data_parquet()


# ------------------------------
# SIDEBAR — FILTROS
# ------------------------------

st.sidebar.header("🔎 Filtros e Parâmetros")

# Município
munis = sorted(df['municipio'].dropna().unique())
sel_muni = st.sidebar.selectbox("Selecione o município", ["Todos"] + munis)

# Período
min_date = df["date"].min()
max_date = df["date"].max()
date_range = st.sidebar.date_input("Período:", [min_date, max_date])

start_date, end_date = date_range

# Parâmetros SEIR
beta = st.sidebar.slider("β — Taxa de transmissão", 0.0, 2.0, 0.6, 0.01)
sigma = st.sidebar.slider("σ — Taxa de incubação", 0.0, 1.0, 1/5, 0.01)
gamma = st.sidebar.slider("γ — Taxa de recuperação", 0.0, 1.0, 1/7, 0.01)

init_days = st.sidebar.number_input("Dias p/ estimar I0", 1, 60, 7)

run_seir = st.sidebar.button("▶ Rodar simulação SEIR")


# ------------------------------
# FILTRAR A BASE
# ------------------------------
mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
dff = df[mask].copy()

if sel_muni != "Todos":
    dff = dff[dff["municipio"] == sel_muni]

if dff.empty:
    st.error("Sem dados para o período ou município selecionado.")
    st.stop()


# ------------------------------
# RESUMO RÁPIDO
# ------------------------------
st.header(f"📌 Resumo — {sel_muni if sel_muni != 'Todos' else 'Estado inteiro'}")

col1, col2, col3 = st.columns(3)

col1.metric("Período (dias)", (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)
col1.metric("Casos novos", int(dff["new_cases"].sum()))

col2.metric("Acumulado máx", int(dff["cum_cases"].max()))
col2.metric("Pico diário", int(dff["new_cases"].max()))

pop_est = int(dff["population"].median())
col3.metric("População (N)", pop_est)


# ------------------------------
# GRÁFICOS — Evolução dos Casos
# ------------------------------

st.subheader("📈 Casos Diários + Média Móvel")

fig = px.line(
    dff,
    x="date",
    y=["new_cases", "ma7"],
    color_discrete_map={"new_cases": COLOR_PRIMARY, "ma7": COLOR_TREND},
    labels={"value": "Casos", "date": "Data"}
)
st.plotly_chart(apply_plot_styling(fig), use_container_width=True)


st.subheader("📉 Infectantes (I_est)")

fig2 = px.line(
    dff,
    x="date",
    y="I_est",
    color_discrete_sequence=[COLOR_SECONDARY],
)
st.plotly_chart(apply_plot_styling(fig2), use_container_width=True)


# ------------------------------
# FUNÇÃO SEIR
# ------------------------------
def run_seir_simulation(N, E0, I0, R0, beta, sigma, gamma, days):
    S0 = N - E0 - I0 - R0

    S, E, I, R = [S0], [E0], [I0], [R0]

    for _ in range(days):
        S_new = S[-1] - (beta * S[-1] * I[-1] / N)
        E_new = E[-1] + (beta * S[-1] * I[-1] / N - sigma * E[-1])
        I_new = I[-1] + (sigma * E[-1] - gamma * I[-1])
        R_new = R[-1] + gamma * I[-1]

        S.append(max(S_new, 0))
        E.append(max(E_new, 0))
        I.append(max(I_new, 0))
        R.append(max(R_new, 0))

    timeline = pd.date_range(dff["date"].max(), periods=len(S), freq="D")

    return pd.DataFrame({"date": timeline, "S": S, "E": E, "I": I, "R": R})


# ------------------------------
# SIMULAÇÃO SEIR
# ------------------------------
if run_seir:

    last = dff.sort_values("date").tail(init_days)
    I0 = int(last["new_cases"].sum())
    if I0 <= 0:
        I0 = max(1, int(dff["new_cases"].median()))

    D_inc = max(1, int(round(1/sigma)))
    E0 = int(dff.tail(D_inc)["new_cases"].sum())
    if E0 <= 0:
        E0 = max(1, I0 * 2)

    R0 = int(dff["R_est"].iloc[-1]) if "R_est" in dff.columns else 0

    st.write(f"➡️ Condições iniciais: S0={pop_est - E0 - I0 - R0}, E0={E0}, I0={I0}, R0={R0}")

    days_sim = st.slider("Dias para simular", 30, 365, 180)

    sim_df = run_seir_simulation(pop_est, E0, I0, R0, beta, sigma, gamma, days_sim)

    st.subheader("📉 Simulação SEIR — Projeção")

    fig_seir = px.line(
        sim_df,
        x="date",
        y=["S", "E", "I", "R"],
        labels={"value": "População", "date": "Data", "variable": "Compartimento"},
        color_discrete_map={
            "S": COLOR_PRIMARY,
            "E": COLOR_TREND,
            "I": COLOR_SECONDARY,
            "R": COLOR_GRAY,
        }
    )
    st.plotly_chart(apply_plot_styling(fig_seir), use_container_width=True)

