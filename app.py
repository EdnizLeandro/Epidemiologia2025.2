
# DASHBOARD COVID-PE + SIMULADOR SEIR 

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

# CONFIGURAÇÕES GERAIS DO APP
BASE_DIR = Path(__file__).parent
DATA_PARQUET = BASE_DIR / "covid_pe_seir_ready.parquet"
DATA_CSV = BASE_DIR / "covid_pe_seir_ready.csv"

st.set_page_config(
    layout="wide",
    page_title="COVID-PE Dashboard + Modelo SEIR"
)

st.title("📊 Dashboard COVID-PE - Dados Epidemiológicos + SEIR Interativo")

# TEMA CORPORATIVO PLOTLY
COLOR_PRIMARY = "#1f77b4"    # azul profissional
COLOR_SECONDARY = "#17becf"  # teal
COLOR_TREND = "#ff7f0e"      # laranja
COLOR_GRAY = "#7f7f7f"


def apply_plot_styling(fig):
    """Aplica estilo visual profissional a qualquer gráfico Plotly."""
    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=22, color=COLOR_PRIMARY),
        font=dict(size=14, color="#333"),
        legend=dict(
            title="",
            orientation="h",
            y=-0.25,
            x=0.5,
            xanchor="center"
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=12))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=12))
    return fig

# FUNÇÃO PARA CARREGAR DADOS
@st.cache_data
def load_data():
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
    else:
        st.error(" Arquivos covid_pe_seir_ready não encontrados.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])


df = load_data()


# SIDEBAR - FILTROS
st.sidebar.header("🔎 Filtros e Parâmetros")

munis = sorted(df['municipio'].dropna().unique())
sel_muni = st.sidebar.selectbox("Selecione o município", ["Todos"] + munis)

min_date, max_date = df['date'].min(), df['date'].max()
date_range = st.sidebar.date_input("Período", [min_date, max_date])

start_date, end_date = date_range

# Parâmetros SEIR
st.sidebar.subheader("⚙ Parâmetros do modelo SEIR")
beta = st.sidebar.slider("Taxa de transmissão (β)", 0.0, 2.0, 0.6, 0.01)
sigma = st.sidebar.slider("Taxa de incubação (σ)", 0.0, 1.0, 1/5, 0.01)
gamma = st.sidebar.slider("Taxa de recuperação (γ)", 0.0, 1.0, 1/7, 0.01)
init_days = st.sidebar.number_input("Dias p/ estimar I0", 1, 60, 7)

run_seir = st.sidebar.button("▶ Rodar simulação SEIR")

# APLICAR FILTROS NA BASE
mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
dff = df[mask].copy()

if sel_muni != "Todos":
    dff = dff[dff['municipio'] == sel_muni]

if dff.empty:
    st.error(" Não há dados para o período ou município selecionado.")
    st.stop()

# RESUMO
st.header(f"📌 Resumo - {sel_muni if sel_muni != 'Todos' else 'Todos os municípios'}")

col1, col2, col3 = st.columns(3)

col1.metric("Período (dias)", (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)
col1.metric("Casos novos no período", int(dff['new_cases'].sum()))

col2.metric("Casos acumulados máximos", int(dff['cum_cases'].max()))
col2.metric("Pico diário de casos", int(dff['new_cases'].max()))

pop_est = dff['population'].median() if sel_muni == "Todos" else dff['population'].iloc[0]
col3.metric("População estimada", int(pop_est))

# GRÁFICO 1 - Casos Diários + Média Móvel
st.subheader("📈 Evolução dos Casos Diários (com Média Móvel)")

fig = px.line(
    dff,
    x='date',
    y=['new_cases', 'ma7'],
    labels={
        "date": "Data",
        "value": "Número de Casos",
        "variable": "Variável"
    },
    title="Casos Diários e Tendência (Média Móvel de 7 dias)",
    color_discrete_map={
        "new_cases": COLOR_PRIMARY,
        "ma7": COLOR_TREND
    }
)
fig = apply_plot_styling(fig)
st.plotly_chart(fig, use_container_width=True)

# GRÁFICO 2 - Estimativa de Infectantes
st.subheader("📉 Estimativa de Infectantes (I_est)")

fig2 = px.line(
    dff,
    x="date",
    y="I_est",
    labels={"date": "Data", "I_est": "Estimativa de Infectantes"},
    title="Estimativa de Infectantes ao Longo do Tempo",
    color_discrete_sequence=[COLOR_SECONDARY]
)
fig2 = apply_plot_styling(fig2)
st.plotly_chart(fig2, use_container_width=True)

# GRÁFICO 3  Top 20 Municípios
if sel_muni == "Todos":
    st.subheader(" Top 20 Municípios com Mais Casos Acumulados")

    top20 = (
        dff.groupby("municipio")["new_cases"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )

    fig3 = px.bar(
        top20,
        x="municipio",
        y="new_cases",
        labels={"municipio": "Município", "new_cases": "Total de Casos Acumulados"},
        title="Top 20 Municípios do Estado por Total de Casos",
        color="new_cases",
        color_continuous_scale="Blues"
    )
    fig3 = apply_plot_styling(fig3)
    st.plotly_chart(fig3, use_container_width=True)

# SIMULAÇÃO SEIR
def run_seir_simulation(N, E0, I0, R0, beta, sigma, gamma, days):
    S0 = N - E0 - I0 - R0
    S, E, I, R = [S0], [E0], [I0], [R0]

    for _ in range(days):
        S_t = S[-1] - (beta * S[-1] * I[-1] / N)
        E_t = E[-1] + (beta * S[-1] * I[-1] / N - sigma * E[-1])
        I_t = I[-1] + (sigma * E[-1] - gamma * I[-1])
        R_t = R[-1] + (gamma * I[-1])

        S.append(max(S_t, 0))
        E.append(max(E_t, 0))
        I.append(max(I_t, 0))
        R.append(max(R_t, 0))

    start = dff['date'].max()
    timeline = pd.date_range(start, periods=len(S), freq='D')

    return pd.DataFrame({"date": timeline, "S": S, "E": E, "I": I, "R": R})


if run_seir:
    last = dff.sort_values("date").tail(init_days)
    I0 = int(last["new_cases"].sum())

    D_INC = max(1, int(round(1/sigma)))
    E0 = int(dff.sort_values("date").tail(D_INC)["new_cases"].sum())

    R0 = int(dff['R_est'].iloc[-1]) if 'R_est' in dff.columns else 0

    days_sim = st.slider("Dias para simular", 30, 365, 120)

    sim_df = run_seir_simulation(pop_est, E0, I0, R0, beta, sigma, gamma, days_sim)

    st.subheader("📉 Simulação SEIR — Projeções")
    
    fig_seir = px.line(
        sim_df,
        x="date",
        y=["S", "E", "I", "R"],
        title="Modelo SEIR — Evolução Projetada",
        labels={"value": "População", "date": "Data", "variable": "Compartimento"},
        color_discrete_map={
            "S": COLOR_PRIMARY,
            "E": COLOR_TREND,
            "I": COLOR_SECONDARY,
            "R": COLOR_GRAY,
        }
    )
    fig_seir = apply_plot_styling(fig_seir)
    st.plotly_chart(fig_seir, use_container_width=True)
