# ================================================================
# Dashboard COVID-PE + Simulação SEIR (Versão Otimizada)
# ================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ================================================================
# CONFIG
# ================================================================
st.set_page_config(layout="wide", page_title="COVID-PE Dashboard + SEIR")

BASE_DIR = Path(__file__).parent
DATA_PARQUET = BASE_DIR / "covid_pe_seir_ready.parquet"
DATA_CSV = BASE_DIR / "covid_pe_seir_ready.csv"

st.title("Dashboard COVID-PE — Dados e Simulação SEIR")


# ================================================================
# LOAD DATA
# ================================================================
@st.cache_data
def load_data():
    """Load parquet or CSV from script directory."""
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
    else:
        st.error("Nenhum arquivo covid_pe_seir_ready.* encontrado na pasta do script.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])
    return df


df = load_data()


# ================================================================
# SIDEBAR — INPUTS
# ================================================================
st.sidebar.header("Filtros e parâmetros")

# Município
municipios = sorted(df["municipio"].unique())
sel_muni = st.sidebar.selectbox("Município", ["Todos"] + municipios)

# Intervalo de datas
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Período", [min_date, max_date])

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]

# Parâmetros SEIR
st.sidebar.subheader("Parâmetros SEIR (simulação)")
beta = st.sidebar.slider("Taxa de transmissão (β)", 0.0, 2.0, 0.6, 0.01)
sigma = st.sidebar.slider("Taxa incubação (σ = 1/latência)", 0.01, 1.0, 0.2, 0.01)
gamma = st.sidebar.slider("Taxa recuperação (γ = 1/duração infecciosa)", 0.01, 1.0, 0.14, 0.01)
init_days = st.sidebar.number_input("Dias para estimar I0", 1, 60, 7)

st.sidebar.markdown("---")
run_seir = st.sidebar.button("Rodar simulação SEIR")


# ================================================================
# FILTER DATA
# ================================================================
mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
dff = df[mask].copy()

if sel_muni != "Todos":
    dff = dff[dff["municipio"] == sel_muni]

# Se filtrou e ficou vazio
if dff.empty:
    st.error("Nenhum dado disponível para o período e município selecionados.")
    st.stop()


# ================================================================
# HEADER SUMMARY
# ================================================================
muni_label = "Todos os municípios" if sel_muni == "Todos" else sel_muni
st.header(f"Resumo — {muni_label}")

col1, col2, col3 = st.columns(3)
col1.metric("Período (dias)", (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)
col1.metric("Total casos novos", int(dff["new_cases"].sum()))
col2.metric("Casos acumulados (máx)", int(dff["cum_cases"].max()))
col2.metric("Pico diário", int(dff["new_cases"].max()))
col3.metric("População (mediana)", int(dff["population"].median()))


# ================================================================
# TIMESERIES: CASOS
# ================================================================
st.subheader("Casos diários e média móvel (7 dias)")
fig_casos = px.line(
    dff,
    x="date",
    y=["new_cases", "ma7"],
    labels={"value": "casos", "date": "data"},
    title="Casos diários e média móvel (7d)"
)
st.plotly_chart(fig_casos, use_container_width=True)

st.subheader("Estimativa de infectantes — I_est")
fig_iest = px.line(
    dff, x="date", y="I_est", title="I_est — estimativa pelo somatório recente"
)
st.plotly_chart(fig_iest, use_container_width=True)


# ================================================================
# TOP MUNICIPIOS
# ================================================================
if sel_muni == "Todos":
    st.subheader("Top 20 municípios por novos casos no período")
    top = (
        dff.groupby("municipio")["new_cases"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )
    fig_top = px.bar(top, x="municipio", y="new_cases", title="Top 20 municípios")
    st.plotly_chart(fig_top, use_container_width=True)


# ================================================================
# SEIR MODEL
# ================================================================
def run_seir_simulation(N, E0, I0, R0, beta, sigma, gamma, days):
    """Classic SEIR deterministic model."""
    S0 = max(N - E0 - I0 - R0, 0)
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

    start = dff["date"].max()
    dates = pd.date_range(start, periods=len(S), freq="D")

    return pd.DataFrame({"date": dates, "S": S, "E": E, "I": I, "R": R})


# ================================================================
# EXECUTA SIMULAÇÃO
# ================================================================
if run_seir:
    # Define população
    N = int(dff["population"].median()) if sel_muni == "Todos" else int(dff["population"].iloc[0])

    # Estimativas iniciais
    last = dff.sort_values("date").tail(init_days)
    I0 = int(last["new_cases"].sum())
    D_INC = max(1, int(round(1 / sigma)))
    E0 = int(dff.sort_values("date").tail(D_INC)["new_cases"].sum())
    R0 = int(dff["R_est"].iloc[-1]) if "R_est" in dff.columns else 0

    st.info(f"Estimativas iniciais — I0={I0}, E0={E0}, R0={R0}, N={N}")

    days_sim = st.slider("Dias para simular", 30, 365, 120)

    sim = run_seir_simulation(N, E0, I0, R0, beta, sigma, gamma, days_sim)

    st.subheader("Simulação SEIR")
    fig_seir = px.line(sim, x="date", y=["S", "E", "I", "R"], title="Modelo SEIR")
    st.plotly_chart(fig_seir, use_container_width=True)


# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.write("Arquivos covid_pe_seir_ready (CSV/Parquet) devem estar na mesma pasta do script.")
