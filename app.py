# ============================================================
# DASHBOARD COVID-PE + SIMULADOR SEIR
# Revisado, corrigido e amplamente comentado
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# CONFIGURAÇÕES GERAIS DO APP
# ------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PARQUET = BASE_DIR / "covid_pe_seir_ready.parquet"
DATA_CSV = BASE_DIR / "covid_pe_seir_ready.csv"

st.set_page_config(
    layout="wide",
    page_title="COVID-PE Dashboard + Modelo SEIR"
)

st.title("📊 Dashboard COVID-PE - Dados Epidemiológicos + SEIR Interativo")


# ------------------------------------------------------------
# FUNÇÃO PARA CARREGAR DADOS (COM CACHE)
# ------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Carrega o dataframe a partir de Parquet ou CSV.
    Converte 'date' para datetime.
    Aplica validações mínimas.
    """

    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)

    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)

    else:
        st.error(" Arquivos covid_pe_seir_ready não encontrados na pasta do script.")
        st.stop()

    # Garantir coluna 'date'
    if "date" not in df.columns:
        st.error(" O dataset não contém a coluna 'date'.")
        st.stop()

    # Converter datas
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Remover linhas com datas inválidas
    df = df.dropna(subset=["date"])

    return df


df = load_data()


# ------------------------------------------------------------
# SIDEBAR — FILTROS E PARÂMETROS
# ------------------------------------------------------------
st.sidebar.header("🔎 Filtros e Parâmetros")

# Filtro de município
munis = sorted(df['municipio'].dropna().unique())
sel_muni = st.sidebar.selectbox("Selecione o município", ["Todos"] + munis, index=0)

# Filtro de período
min_date, max_date = df['date'].min(), df['date'].max()

date_range = st.sidebar.date_input("Período", [min_date, max_date])

# Garantir estrutura da data
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

# Parâmetros do SEIR
st.sidebar.subheader("⚙ Parâmetros do modelo SEIR")
beta = st.sidebar.slider("Taxa de transmissão (β)", 0.0, 2.0, 0.6, 0.01)
sigma = st.sidebar.slider("Taxa de incubação (σ = 1/latência)", 0.0, 1.0, 1/5, 0.01)
gamma = st.sidebar.slider("Taxa de recuperação (γ = 1/infectious)", 0.0, 1.0, 1/7, 0.01)
init_days = st.sidebar.number_input("Dias p/ estimar I0", 1, 60, 7)

st.sidebar.markdown("---")
run_seir = st.sidebar.button("▶ Rodar simulação SEIR")


# ------------------------------------------------------------
# APLICAR FILTROS NOS DADOS
# ------------------------------------------------------------
mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
dff = df[mask].copy()

if sel_muni != "Todos":
    dff = dff[dff['municipio'] == sel_muni]

if dff.empty:
    st.error(" Não há dados para o período ou município selecionado.")
    st.stop()


# ------------------------------------------------------------
# RESUMO GERAL
# ------------------------------------------------------------
st.header(f" Resumo - {sel_muni if sel_muni != 'Todos' else 'Todos os municípios'}")

col1, col2, col3 = st.columns(3)

col1.metric("Período (dias)", (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)
col1.metric("Total de casos novos", int(dff['new_cases'].sum()))

col2.metric("Casos acumulados (máximo)", int(dff['cum_cases'].max()))
col2.metric("Pico diário (new_cases)", int(dff['new_cases'].max()))

pop_est = dff['population'].median() if sel_muni == "Todos" else dff['population'].iloc[0]
col3.metric("População estimada", int(pop_est))


# ------------------------------------------------------------
# GRÁFICOS PRINCIPAIS
# ------------------------------------------------------------
st.subheader("📈 Casos diários e média móvel (7 dias)")
fig = px.line(
    dff,
    x='date',
    y=['new_cases', 'ma7'],
    labels={'value': 'Casos', 'date': 'Data'},
    title='Evolução dos casos diários'
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📉 Estimativa de infectantes (I_est)")
fig2 = px.line(dff, x='date', y='I_est', title='Estimativa de infectantes I(t)')
st.plotly_chart(fig2, use_container_width=True)

if sel_muni == "Todos":
    st.subheader(" Top 20 municípios por número de casos")
    top20 = (
        dff.groupby('municipio')['new_cases']
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )
    fig3 = px.bar(top20, x='municipio', y='new_cases')
    st.plotly_chart(fig3, use_container_width=True)


# ------------------------------------------------------------
# FUNÇÃO DE SIMULAÇÃO SEIR
# ------------------------------------------------------------
def run_seir_simulation(N, E0, I0, R0, beta, sigma, gamma, days):
    """Simula o modelo SEIR clássico usando discretização simples."""

    S0 = N - E0 - I0 - R0
    S, E, I, R = [S0], [E0], [I0], [R0]

    for t in range(days):
        S_t = S[-1] - (beta * S[-1] * I[-1] / N)
        E_t = E[-1] + (beta * S[-1] * I[-1] / N - sigma * E[-1])
        I_t = I[-1] + (sigma * E[-1] - gamma * I[-1])
        R_t = R[-1] + (gamma * I[-1])

        S.append(max(S_t, 0))
        E.append(max(E_t, 0))
        I.append(max(I_t, 0))
        R.append(max(R_t, 0))

    # Datas começam um dia após o último registro real
    start = dff['date'].max()
    timeline = pd.date_range(start, periods=len(S), freq='D')

    return pd.DataFrame({"date": timeline, "S": S, "E": E, "I": I, "R": R})


# ------------------------------------------------------------
# EXECUTAR SIMULAÇÃO SEIR
# ------------------------------------------------------------
if run_seir:

    if sel_muni == "Todos":
        st.warning("⚠ Recomenda-se selecionar um município para simulação.")

    # População
    N = int(pop_est)

    # Estimativas iniciais
    last = dff.sort_values("date").tail(init_days)
    I0 = int(last['new_cases'].sum())

    D_INC = max(1, int(round(1 / sigma)))
    E0 = int(dff.sort_values("date").tail(D_INC)['new_cases'].sum())

    R0 = int(dff['R_est'].iloc[-1]) if 'R_est' in dff.columns else 0

    st.info(f"🔹 Parâmetros estimados: I0={I0}, E0={E0}, R0={R0}, N={N}")

    days_sim = st.slider("Dias para simular", 30, 365, 120)

    sim_df = run_seir_simulation(N, E0, I0, R0, beta, sigma, gamma, days_sim)

    st.subheader("📉 Simulação SEIR")
    fig_seir = px.line(sim_df, x='date', y=['S', 'E', 'I', 'R'], title="Modelo SEIR")
    st.plotly_chart(fig_seir, use_container_width=True)
