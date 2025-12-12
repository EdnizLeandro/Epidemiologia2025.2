# app_seir_modelos.py
"""
SEIR Suite — Versão LEVE (sem MCMC)
Modelos: SIR, SEIR, SEIRD, SEIRV
Ajuste: Least-Squares (rápido)
Visualização: Plotly
Dados: covid_pe_seir_ready.parquet (mesma pasta)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.integrate import odeint
from scipy.optimize import least_squares
import plotly.express as px
import plotly.graph_objects as go
import base64

# ============================================================
# CONFIGURAÇÃO DO APP
# ============================================================
st.set_page_config(layout="wide",
                   page_title="COVID-PE — Modelos Epidemiológicos")

st.title("🔬 Modelos Epidemiológicos — COVID-PE (Versão LEVE — sem MCMC)")


# ============================================================
# CAMINHOS PARA OS ARQUIVOS
# ============================================================
DEFAULT_PARQUET = Path(__file__).parent / "covid_pe_seir_ready.parquet"
DEFAULT_CSV = Path(__file__).parent / "covid_pe_seir_ready.csv"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def apply_style(fig):
    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=20, color="#1f77b4"),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=80, b=60)
    )
    return fig


def download_link(df, filename="export.csv"):
    data = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(data).decode()
    return f'<a download="{filename}" href="data:file/csv;base64,{b64}">📥 Baixar {filename}</a>'


# ============================================================
# ODEs DE CADA MODELO
# ============================================================
def sir_ode(y, t, beta, gamma, N):
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def seir_ode(y, t, beta, sigma, gamma, N):
    S, E, I, R = y
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]


def seird_ode(y, t, beta, sigma, gamma, mu, N):
    S, E, I, R, D = y
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I - mu * I
    dR = gamma * I
    dD = mu * I
    return [dS, dE, dI, dR, dD]


def seirv_ode(y, t, beta, sigma, gamma, v_rate, N):
    S, E, I, R = y
    dS = -beta * S * I / N - v_rate * S
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I + v_rate * S
    return [dS, dE, dI, dR]


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================
@st.cache_data
def load_data(uploaded=None):
    if uploaded is not None:
        if uploaded.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded)
        else:
            df = pd.read_csv(uploaded, parse_dates=["date"])
    else:
        if DEFAULT_PARQUET.exists():
            df = pd.read_parquet(DEFAULT_PARQUET)
        elif DEFAULT_CSV.exists():
            df = pd.read_csv(DEFAULT_CSV, parse_dates=["date"])
        else:
            return None

    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    
    if "i_est" not in df.columns:
        df = df.sort_values("date")
        df["i_est"] = df["new_cases"].rolling(7, min_periods=1).sum()

    if "r_est" not in df.columns:
        df["r_est"] = 0

    return df


uploaded = st.sidebar.file_uploader("Upload (opcional) CSV/Parquet pré-processado", type=["csv","parquet"])
df = load_data(uploaded)

if df is None:
    st.error("Arquivo pre-processado não encontrado. Coloque covid_pe_seir_ready.parquet na mesma pasta ou faça upload.")
    st.stop()


# ============================================================
# FILTROS
# ============================================================
munis = sorted(df["municipio"].dropna().unique()) if "municipio" in df.columns else []
sel_muni = st.sidebar.selectbox("Município", ["Todos"] + munis)

date_min = df["date"].min().date()
date_max = df["date"].max().date()

start_date, end_date = st.sidebar.date_input(
    "Período",
    [date_min, date_max],
    min_value=date_min,
    max_value=date_max
)

mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
dff = df[mask].copy()

if sel_muni != "Todos":
    dff = dff[dff["municipio"] == sel_muni]

if dff.empty:
    st.error("Sem dados para o filtro selecionado.")
    st.stop()


# ============================================================
# SELEÇÃO DO MODELO
# ============================================================
model_choice = st.sidebar.selectbox("Modelo:", ["SIR", "SEIR", "SEIRD", "SEIRV"])


# ============================================================
# PARÂMETROS INICIAIS
# ============================================================
beta0 = st.sidebar.slider("β inicial", 0.0, 2.0, 0.6)
sigma0 = st.sidebar.slider("σ inicial", 0.01, 1.0, 1/5)
gamma0 = st.sidebar.slider("γ inicial", 0.01, 1.0, 1/7)
mu0 = st.sidebar.slider("μ inicial (SEIRD)", 0.0, 0.2, 0.01)
v0 = st.sidebar.slider("Taxa de vacinação v (SEIRV)", 0.0, 0.1, 0.0)

init_days = st.sidebar.number_input("Dias para I0", 1, 60, 7)

run_sim = st.sidebar.button("▶ Rodar ajuste + simulação")


# ============================================================
# ESTIMAR I0, E0, R0, S0
# ============================================================
last = dff.sort_values("date").tail(init_days)
I0 = int(last["new_cases"].sum())
if I0 <= 0:
    I0 = max(1, int(dff["new_cases"].median()))

D_inc = max(1, int(round(1/sigma0)))
E0 = int(dff.tail(D_inc)["new_cases"].sum())
if E0 <= 0:
    E0 = I0 * 2

R0 = int(dff["r_est"].iloc[-1])
pop = int(dff["population"].median())
S0 = max(pop - E0 - I0 - R0, 0)

st.write(f"**Condições iniciais estimadas:** S0={S0}, E0={E0}, I0={I0}, R0={R0}")


# ============================================================
# GRÁFICO DOS DADOS
# ============================================================
st.subheader("📈 Séries observadas")
fig = px.line(
    dff.sort_values("date"),
    x="date",
    y=["new_cases", "i_est", "r_est"],
    title="Casos observados"
)
st.plotly_chart(apply_style(fig), use_container_width=True)


# ============================================================
# AJUSTE POR LEAST-SQUARES
# ============================================================
def fit_model():
    t = np.arange(len(y_obs))

    if model_choice == "SIR":
        def resid(p):
            b, g = p
            sol = odeint(lambda y, tt: sir_ode(y, tt, b, g, pop), [S0,I0,R0], t)
            return sol[:,1] - y_obs
        x0 = [beta0, gamma0]
        bounds = ([0,0.001], [5,1])

    elif model_choice == "SEIR":
        def resid(p):
            b, s, g = p
            sol = odeint(lambda y, tt: seir_ode(y, tt, b, s, g, pop), [S0,E0,I0,R0], t)
            return sol[:,2] - y_obs
        x0 = [beta0, sigma0, gamma0]
        bounds = ([0,0.001,0.001], [5,1,1])

    elif model_choice == "SEIRD":
        def resid(p):
            b, s, g, m = p
            sol = odeint(lambda y, tt: seird_ode(y, tt, b, s, g, m, pop), [S0,E0,I0,R0,0], t)
            return sol[:,2] - y_obs
        x0 = [beta0, sigma0, gamma0, mu0]
        bounds = ([0,0.001,0.001,0.0], [5,1,1,0.5])

    else:  # SEIRV
        def resid(p):
            b, s, g, v = p
            sol = odeint(lambda y, tt: seirv_ode(y, tt, b, s, g, v, pop), [S0,E0,I0,R0], t)
            return sol[:,2] - y_obs
        x0 = [beta0, sigma0, gamma0, v0]
        bounds = ([0,0.001,0.001,0.0], [5,1,1,0.2])

    res = least_squares(resid, x0, bounds=bounds, max_nfev=30000)
    return res.x


y_obs = dff.sort_values("date")["i_est"].values


# ============================================================
# EXECUTAR AJUSTE E SIMULAÇÃO
# ============================================================
if run_sim:

    params = fit_model()
    st.success(f"Parâmetros ajustados: {params}")

    # SIMULAÇÃO FUTURA
    days = st.slider("Dias de projeção", 30, 720, 180)
    t = np.arange(days)

    if model_choice == "SIR":
        sol = odeint(lambda y,tt: sir_ode(y,tt,*params,pop), [S0,I0,R0], t)
        cols = ["S","I","R"]

    elif model_choice == "SEIR":
        sol = odeint(lambda y,tt: seir_ode(y,tt,*params,pop), [S0,E0,I0,R0], t)
        cols = ["S","E","I","R"]

    elif model_choice == "SEIRD":
        sol = odeint(lambda y,tt: seird_ode(y,tt,*params,pop), [S0,E0,I0,R0,0], t)
        cols = ["S","E","I","R","D"]

    else:
        sol = odeint(lambda y,tt: seirv_ode(y,tt,*params,pop), [S0,E0,I0,R0], t)
        cols = ["S","E","I","R"]

    proj = pd.DataFrame(sol, columns=cols)
    proj["date"] = pd.date_range(dff["date"].max(), periods=days, freq="D")

    # GRÁFICO
    st.subheader("📉 Projeção futura")
    fig_proj = px.line(proj, x="date", y=cols, title="Projeção (modelo ajustado)")
    st.plotly_chart(apply_style(fig_proj), use_container_width=True)

    # DOWNLOAD
    st.markdown(download_link(proj, f"projecao_{model_choice}.csv"), unsafe_allow_html=True)


else:
    st.info("Clique em ▶ Rodar ajuste + simulação para iniciar.")


