# ============================================================
#  SEIR SUITE — VERSÃO LEVE + CORRIGIDA
#  MODELOS: SIR / SEIR / SEIRD / SEIRV
#  AJUSTE: LEAST-SQUARES
#  DATAS EM FORMATO BRASILEIRO
#  TÍTULOS E EIXOS EM MAIÚSCULO
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.integrate import odeint
from scipy.optimize import least_squares
import plotly.express as px
import plotly.graph_objects as go
import base64

# ------------------------------------------------------------
# CONFIGURAÇÃO DO APP
# ------------------------------------------------------------

st.set_page_config(layout="wide", page_title="COVID-PE — MODELOS EPIDEMIOLÓGICOS (BR)")

st.title("🔬 MODELAGEM EPIDEMIOLÓGICA COVID19-PE — SIR / SEIR / SEIRD / SEIRV (BR)")


# ------------------------------------------------------------
# CAMINHOS DOS ARQUIVOS
# ------------------------------------------------------------

DEFAULT_PARQUET = Path(__file__).parent / "covid_pe_seir_ready.parquet"
DEFAULT_CSV = Path(__file__).parent / "covid_pe_seir_ready.csv"


# ------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------------------

def estilo(fig):
    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=22, color="#1f77b4"),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=80, b=60),
        xaxis_title="DATA",
        yaxis_title="VALOR"
    )
    return fig


def baixar(df, nome="export.csv"):
    data = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(data).decode()
    return f'<a download="{nome}" href="data:file/csv;base64,{b64}">📥 BAIXAR {nome}</a>'


# ------------------------------------------------------------
# ODEs DOS MODELOS
# ------------------------------------------------------------

def sir_ode(y, t, beta, gamma, N):
    S, I, R = y
    return [
        -beta*S*I/N,
        beta*S*I/N - gamma*I,
        gamma*I
    ]

def seir_ode(y, t, beta, sigma, gamma, N):
    S, E, I, R = y
    return [
        -beta*S*I/N,
        beta*S*I/N - sigma*E,
        sigma*E - gamma*I,
        gamma*I
    ]

def seird_ode(y, t, beta, sigma, gamma, mu, N):
    S, E, I, R, D = y
    return [
        -beta*S*I/N,
        beta*S*I/N - sigma*E,
        sigma*E - gamma*I - mu*I,
        gamma*I,
        mu*I
    ]

def seirv_ode(y, t, beta, sigma, gamma, v, N):
    S, E, I, R = y
    return [
        -beta*S*I/N - v*S,
        beta*S*I/N - sigma*E,
        sigma*E - gamma*I,
        gamma*I + v*S
    ]


# ------------------------------------------------------------
# CARREGAMENTO DO ARQUIVO
# ------------------------------------------------------------

@st.cache_data
def carregar(uploaded=None):
    if uploaded is not None:
        if uploaded.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded)
        else:
            df = pd.read_csv(uploaded)
    else:
        if DEFAULT_PARQUET.exists():
            df = pd.read_parquet(DEFAULT_PARQUET)
        else:
            df = pd.read_csv(DEFAULT_CSV)

    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Datas BR
    df["data_br"] = df["date"].dt.strftime("%d/%m/%Y")

    # Estimar I_est se faltar
    if "i_est" not in df.columns:
        df = df.sort_values("date")
        df["i_est"] = df["new_cases"].rolling(7, min_periods=1).sum()

    if "r_est" not in df.columns:
        df["r_est"] = 0

    return df


uploaded = st.sidebar.file_uploader("Upload opcional do DATASET (CSV ou PARQUET)", type=["csv","parquet"])
df = carregar(uploaded)

if df is None:
    st.error("Nenhum arquivo encontrado.")
    st.stop()


# ------------------------------------------------------------
# FILTROS
# ------------------------------------------------------------

munis = sorted(df["municipio"].dropna().unique()) if "municipio" in df.columns else []
sel_muni = st.sidebar.selectbox("MUNICÍPIO", ["TODOS"] + munis)

dmin = df["date"].min().date()
dmax = df["date"].max().date()

ini, fim = st.sidebar.date_input("PERÍODO", [dmin, dmax])

mask = (df["date"] >= pd.to_datetime(ini)) & (df["date"] <= pd.to_datetime(fim))
dff = df[mask].copy()

if sel_muni != "TODOS":
    dff = dff[dff["municipio"] == sel_muni]

if dff.empty:
    st.error("SEM DADOS PARA O FILTRO SELECIONADO.")
    st.stop()


# ------------------------------------------------------------
# PARÂMETROS
# ------------------------------------------------------------

st.sidebar.header("PARÂMETROS DO MODELO")

modelo = st.sidebar.selectbox("MODELO", ["SIR","SEIR","SEIRD","SEIRV"])

beta0 = st.sidebar.slider("BETA (β)", 0.0, 2.0, 0.6)
sigma0 = st.sidebar.slider("SIGMA (σ)", 0.01, 1.0, 0.2)
gamma0 = st.sidebar.slider("GAMMA (γ)", 0.01, 1.0, 0.14)
mu0 = st.sidebar.slider("MU (μ) — MORTALIDADE", 0.0, 0.2, 0.01)
v0 = st.sidebar.slider("TAXA DE VACINAÇÃO (v)", 0.0, 0.1, 0.0)

init_days = st.sidebar.number_input("DIAS PARA I0", 1, 60, 7)

botao = st.sidebar.button("▶ RODAR AJUSTE + SIMULAÇÃO")


# ------------------------------------------------------------
# ESTIMAR I0, E0, S0, R0
# ------------------------------------------------------------

last = dff.sort_values("date").tail(init_days)

I0 = int(max(last["new_cases"].sum(), 1))

E0 = int(max(dff.tail(int(1/sigma0))["new_cases"].sum(), I0*2))

R0 = int(dff["r_est"].iloc[-1])

N = int(dff["population"].median())

S0 = max(N - E0 - I0 - R0, 0)

st.write(f"**COND. INICIAIS:** S0={S0}, E0={E0}, I0={I0}, R0={R0}")


# ------------------------------------------------------------
# GRÁFICO DOS CASOS OBSERVADOS
# ------------------------------------------------------------

st.subheader("📈 CASOS OBSERVADOS (BR)")

fig_obs = px.line(
    dff.sort_values("date"),
    x="data_br",
    y=["new_cases","i_est","r_est"],
    labels={"value":"VALOR","variable":"VARIÁVEL","data_br":"DATA"},
    title="CASOS OBSERVADOS"
)
st.plotly_chart(estilo(fig_obs), use_container_width=True)


# ------------------------------------------------------------
# AJUSTE — LEAST SQUARES
# ------------------------------------------------------------

y_obs = dff.sort_values("date")["i_est"].values
t_obs = np.arange(len(y_obs))

def ajustar():
    if modelo == "SIR":
        def resid(p):
            b, g = p
            sol = odeint(lambda y,t: sir_ode(y,t,b,g,N), [S0,I0,R0], t_obs)
            return sol[:,1] - y_obs
        x0 = [beta0, gamma0]
        bounds = ([0,0.001],[5,1])

    elif modelo == "SEIR":
        def resid(p):
            b,s,g = p
            sol = odeint(lambda y,t: seir_ode(y,t,b,s,g,N), [S0,E0,I0,R0], t_obs)
            return sol[:,2] - y_obs
        x0 = [beta0,sigma0,gamma0]
        bounds = ([0,0.001,0.001],[5,1,1])

    elif modelo == "SEIRD":
        def resid(p):
            b,s,g,m = p
            sol = odeint(lambda y,t: seird_ode(y,t,b,s,g,m,N), [S0,E0,I0,R0,0], t_obs)
            return sol[:,2] - y_obs
        x0 = [beta0,sigma0,gamma0,mu0]
        bounds = ([0,0.001,0.001,0],[5,1,1,0.5])

    else: # SEIRV
        def resid(p):
            b,s,g,v = p
            sol = odeint(lambda y,t: seirv_ode(y,t,b,s,g,v,N), [S0,E0,I0,R0], t_obs)
            return sol[:,2] - y_obs
        x0 = [beta0,sigma0,gamma0,v0]
        bounds = ([0,0.001,0.001,0],[5,1,1,0.2])

    return least_squares(resid, x0, bounds=bounds, max_nfev=50000).x


# ------------------------------------------------------------
# EXECUÇÃO
# ------------------------------------------------------------

if botao:

    params = ajustar()
    st.success(f"PARÂMETROS AJUSTADOS: {params}")

    # ---------------- PROJEÇÃO -----------------
    dias = st.slider("DIAS PARA PROJEÇÃO", 30, 720, 180)
    t = np.arange(dias)

    if modelo == "SIR":
        sol = odeint(lambda y,t: sir_ode(y,t,*params,N), [S0,I0,R0], t)
        cols = ["S","I","R"]

    elif modelo == "SEIR":
        sol = odeint(lambda y,t: seir_ode(y,t,*params,N), [S0,E0,I0,R0], t)
        cols = ["S","E","I","R"]

    elif modelo == "SEIRD":
        sol = odeint(lambda y,t: seird_ode(y,t,*params,N), [S0,E0,I0,R0,0], t)
        cols = ["S","E","I","R","D"]

    else:
        sol = odeint(lambda y,t: seirv_ode(y,t,*params,N), [S0,E0,I0,R0], t)
        cols = ["S","E","I","R"]

    proj = pd.DataFrame(sol, columns=cols)
    proj["DATA"] = pd.date_range(dff["date"].max(), periods=dias, freq="D").strftime("%d/%m/%Y")

    st.subheader("📉 PROJEÇÃO FUTURA (BR)")

    fig_proj = px.line(
        proj,
        x="DATA",
        y=cols,
        title="PROJEÇÃO FUTURA",
        labels={"DATA":"DATA"}
    )

    st.plotly_chart(estilo(fig_proj), use_container_width=True)

    # download
    st.markdown(baixar(proj, f"projecao_{modelo}.csv"), unsafe_allow_html=True)

