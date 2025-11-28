import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

# ---------------------- CONFIGURAÇÕES --------------------------
BASE_DIR = Path(__file__).parent
DATA_PARQUET = BASE_DIR / "covid_pe_seir_ready.parquet"
DATA_CSV = BASE_DIR / "covid_pe_seir_ready.csv"

st.set_page_config(
    layout="wide",
    page_title="COVID-PE — Dashboard + SEIR",
)

# Paleta de cores profissional
COLOR_NEW = "#1f77b4"      # Azul
COLOR_MA7 = "#ff7f0e"      # Laranja
COLOR_I_EST = "#d62728"    # Vermelho
COLOR_S = "#2ca02c"        # Verde
COLOR_E = "#9467bd"        # Roxo
COLOR_I = "#d62728"        # Vermelho
COLOR_R = "#8c564b"        # Marrom

st.title("📊 Dashboard COVID-PE - Dados e Simulação SEIR")


# ---------------------- CARREGAR DADOS -------------------------
@st.cache_data
def load_data():
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
    else:
        st.error("❌ Arquivo covid_pe_seir_ready.* não encontrado na mesma pasta do script.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()


# ---------------------- SIDEBAR -------------------------------
st.sidebar.header("⚙️ Filtros e Parâmetros")

municipios = sorted(df["municipio"].unique())
sel_muni = st.sidebar.selectbox("Município", ["Todos"] + municipios)

# Intervalo de datas
min_dt, max_dt = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Período de análise", [min_dt, max_dt])

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]

# Parâmetros SEIR
st.sidebar.subheader("🧪 Parâmetros da Simulação SEIR")

beta = st.sidebar.slider("Taxa de transmissão (β)", 0.0, 2.0, 0.6, 0.01)
sigma = st.sidebar.slider("Taxa de incubação (σ = 1/latência)", 0.01, 1.0, 1/5, 0.01)
gamma = st.sidebar.slider("Taxa de recuperação (γ = 1/período infeccioso)", 0.01, 1.0, 1/7, 0.01)

init_days = st.sidebar.number_input(
    "Dias utilizados para estimar I₀", 1, 60, 7
)

st.sidebar.markdown("---")
run_seir = st.sidebar.button("▶️ Rodar Simulação SEIR")


# ---------------------- FILTRO DE DADOS ------------------------
mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
dff = df[mask].copy()

if sel_muni != "Todos":
    dff = dff[dff["municipio"] == sel_muni]

if dff.empty:
    st.error("⚠️ Não há dados disponíveis para o período e município selecionados.")
    st.stop()


# ---------------------- RESUMO ---------------------------------
st.header(f"📌 Resumo — {sel_muni if sel_muni!='Todos' else 'Todos os municípios'}")

col1, col2, col3 = st.columns(3)
col1.metric("Período analisado (dias)", (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)
col1.metric("Total de novos casos", int(dff["new_cases"].sum()))
col2.metric("Casos acumulados (máx.)", int(dff["cum_cases"].max()))
col2.metric("Pico diário de casos", int(dff["new_cases"].max()))
col3.metric("População estimada", int(dff["population"].median()))


# ---------------------- GRÁFICOS -------------------------------
st.subheader("📈 Casos diários e média móvel (7 dias)")

fig = px.line(
    dff,
    x="date",
    y=["new_cases", "ma7"],
    labels={"value": "Casos", "date": "Data"},
    title="Casos diários e média móvel",
    color_discrete_map={
        "new_cases": COLOR_NEW,
        "ma7": COLOR_MA7
    }
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔥 Estimativa de infectantes (I_est)")

fig2 = px.line(
    dff,
    x="date",
    y="I_est",
    title="Estimativa de indivíduos infectantes (Iₜ)",
    color_discrete_sequence=[COLOR_I_EST]
)
st.plotly_chart(fig2, use_container_width=True)


# Top 20 municípios
if sel_muni == "Todos":
    st.subheader(" Top 20 municípios — novos casos no período selecionado")

    top = (
        dff.groupby("municipio")["new_cases"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )

    fig3 = px.bar(
        top,
        x="municipio",
        y="new_cases",
        title="Top 20 municípios por novos casos",
        color="new_cases",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig3, use_container_width=True)


# ---------------------- MODELO SEIR ----------------------------
def run_seir_simulation(N, E0, I0, R0, beta, sigma, gamma, days):
    S0 = N - E0 - I0 - R0
    S, E, I, R = [S0], [E0], [I0], [R0]

    for _ in range(days):
        S_t = S[-1] - beta * S[-1] * I[-1] / N
        E_t = E[-1] + beta * S[-1] * I[-1] / N - sigma * E[-1]
        I_t = I[-1] + sigma * E[-1] - gamma * I[-1]
        R_t = R[-1] + gamma * I[-1]

        S.append(max(S_t, 0))
        E.append(max(E_t, 0))
        I.append(max(I_t, 0))
        R.append(max(R_t, 0))

    start_date = dff["date"].max()
    dates = pd.date_range(start_date, periods=len(S), freq="D")

    return pd.DataFrame({"date": dates, "S": S, "E": E, "I": I, "R": R})


# ---------------------- EXECUTAR SEIR --------------------------
if run_seir:

    N = int(dff["population"].median()) if sel_muni == "Todos" else int(dff["population"].iloc[0])

    last = dff.sort_values("date").tail(init_days)
    I0 = int(last["new_cases"].sum())

    D_INC = max(1, int(round(1 / sigma)))
    E0 = int(dff.sort_values("date").tail(D_INC)["new_cases"].sum())

    R0 = int(dff["R_est"].iloc[-1]) if "R_est" in dff.columns else 0

    st.info(f"**Estimativas iniciais:** I₀={I0}, E₀={E0}, R₀={R0}, População={N}")

    days_sim = st.slider("Dias para simular", 30, 365, 120)

    sim_df = run_seir_simulation(N, E0, I0, R0, beta, sigma, gamma, days_sim)

    st.subheader("🧮 Simulação do Modelo SEIR")

    fig_seir = px.line(
        sim_df,
        x="date",
        y=["S", "E", "I", "R"],
        title="Evolução dos grupos do modelo SEIR",
        color_discrete_map={
            "S": COLOR_S,
            "E": COLOR_E,
            "I": COLOR_I,
            "R": COLOR_R,
        }
    )

    st.plotly_chart(fig_seir, use_container_width=True)

