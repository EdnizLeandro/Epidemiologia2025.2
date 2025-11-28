import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

# -------- CONFIG --------
BASE_DIR = Path(__file__).parent
DATA_PARQUET = BASE_DIR / "covid_pe_seir_ready.parquet"
DATA_CSV = BASE_DIR / "covid_pe_seir_ready.csv"

st.set_page_config(layout="wide", page_title="COVID-PE Dashboard + SEIR")

st.title("Dashboard COVID-PE — Dados e SEIR (interativo)")

# load dataset
@st.cache_data
def load_data():
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
    else:
        st.error("Arquivos covid_pe_seir_ready não encontrados na pasta do script.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# -------------------- SIDEBAR --------------------
st.sidebar.header("Filtros e parâmetros")
munis = sorted(df['municipio'].unique())
sel_muni = st.sidebar.selectbox("Município", ["Todos"] + munis, index=0)

# Date range
min_date = df['date'].min()
max_date = df['date'].max()

date_range = st.sidebar.date_input("Período", [min_date, max_date])
if isinstance(date_range, (list, tuple)):
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0]
else:
    start_date = end_date = date_range

# SEIR parameters
st.sidebar.subheader("Parâmetros SEIR (simulação)")
beta = st.sidebar.slider("Taxa de transmissão (β)", 0.0, 2.0, 0.6, 0.01)
sigma = st.sidebar.slider("Taxa incubação (σ = 1/latência)", 0.0, 1.0, 1/5, 0.01)
gamma = st.sidebar.slider("Taxa recuperação (γ = 1/infectious)", 0.0, 1.0, 1/7, 0.01)
init_days = st.sidebar.number_input("Dias p/ estimar I0 (soma de novos casos)", 1, 60, 7)

st.sidebar.markdown("---")
run_seir = st.sidebar.button("Rodar SEIR")


# -------------------- FILTER --------------------
mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
dff = df[mask].copy()

if sel_muni != "Todos":
    dff = dff[dff['municipio'] == sel_muni]

if dff.empty:
    st.error("Não há dados para o período ou município selecionado.")
    st.stop()


# -------------------- HEADER SUMMARY --------------------
st.header(f"Resumo — {'Todos os municípios' if sel_muni=='Todos' else sel_muni}")

col1, col2, col3 = st.columns(3)
col1.metric("Período (dias)", (pd.to_datetime(end_date)-pd.to_datetime(start_date)).days + 1)
col1.metric("Total casos (novos)", int(dff['new_cases'].sum()))
col2.metric("Casos acumulados (máx)", int(dff['cum_cases'].max()))
col2.metric("Pico diário (máx new_cases)", int(dff['new_cases'].max()))
col3.metric("População (ex. média)", int(dff['population'].median()))


# -------------------- PLOTS --------------------
st.subheader("Casos diários e média móvel (7d)")
fig = px.line(dff, x='date', y=['new_cases','ma7'], labels={'value':'casos','date':'data'},
              title='Casos diários e média móvel')
st.plotly_chart(fig, use_container_width=True)

st.subheader("Índice estimado: I_est (soma últimos dias)")
fig2 = px.line(dff, x='date', y='I_est', title='Estimativa de infectantes I_t')
st.plotly_chart(fig2, use_container_width=True)

if sel_muni == "Todos":
    st.subheader("Top 20 municípios por casos no período")
    top = dff.groupby('municipio')['new_cases'].sum().sort_values(ascending=False).head(20).reset_index()
    fig3 = px.bar(top, x='municipio', y='new_cases')
    st.plotly_chart(fig3, use_container_width=True)


# -------------------- SEIR SIMULATION --------------------
def run_seir_simulation(N, E0, I0, R0, beta, sigma, gamma, days):
    S0 = N - E0 - I0 - R0
    S, E, I, R = [S0], [E0], [I0], [R0]
    dt = 1.0

    for t in range(days):
        S_t = S[-1] - (beta * S[-1] * I[-1] / N) * dt
        E_t = E[-1] + (beta * S[-1] * I[-1] / N - sigma * E[-1]) * dt
        I_t = I[-1] + (sigma * E[-1] - gamma * I[-1]) * dt
        R_t = R[-1] + (gamma * I[-1]) * dt

        S.append(max(S_t,0))
        E.append(max(E_t,0))
        I.append(max(I_t,0))
        R.append(max(R_t,0))

    # começa da última data observada (mais natural)
    start = dff['date'].max()
    days_index = pd.date_range(start, periods=len(S), freq='D')

    return pd.DataFrame({"date": days_index, "S": S, "E": E, "I": I, "R": R})


if run_seir:
    if sel_muni == "Todos":
        st.warning("Selecione um município para simulação ou use população média.")

    # Escolhe população
    N = int(dff['population'].median()) if sel_muni == "Todos" else int(dff['population'].iloc[0])

    # Estimar I0
    last = dff.sort_values("date").tail(init_days)
    I0 = int(last['new_cases'].sum())

    # Estimar E0 usando latência = 1/sigma
    D_INC = max(1, int(round(1/sigma)))
    E0 = int(dff.sort_values("date").tail(D_INC)['new_cases'].sum())

    # Estimar R0 se existir
    R0 = int(dff['R_est'].iloc[-1]) if 'R_est' in dff.columns else 0

    st.info(f"Estimativas iniciais: I0={I0}, E0={E0}, R0={R0}, N={N}")

    days_sim = st.slider("Dias para simular", 30, 365, 120)

    sim_df = run_seir_simulation(N, E0, I0, R0, beta, sigma, gamma, days_sim)

    st.subheader("Simulação SEIR")
    fig_seir = px.line(sim_df, x='date', y=['S','E','I','R'], title="SEIR Simulation")
    st.plotly_chart(fig_seir, use_container_width=True)

st.markdown("---")
st.write("Arquivos covid_pe_seir_ready (CSV/Parquet) devem estar na mesma pasta deste script.")
