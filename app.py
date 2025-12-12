# app_seir_all_versions.py
"""
Streamlit app — Multi-model epidemiological simulator & fitter
Supports: SIR, SEIR, SEIRD, SEIRV
Provides: deterministic simulation, least-squares fit, optional MCMC (emcee)
Data source (default): covid_pe_seir_ready.parquet (same folder as this script)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.integrate import odeint
from scipy.optimize import least_squares
import plotly.express as px
import plotly.graph_objects as go
import time, base64

# Optional MCMC libs
try:
    import emcee
    import corner
    HAS_MCMC = True
except Exception:
    HAS_MCMC = False

# -------------------------
# App config
# -------------------------
st.set_page_config(layout="wide", page_title="SEIR Suite — COVID-PE", initial_sidebar_state="expanded")
st.title("🔬 SEIR Suite — Modelos epidemiológicos (SIR / SEIR / SEIRD / SEIRV) — COVID-PE")

# default data file (relative to script)
DEFAULT_PARQUET = Path(__file__).parent / "covid_pe_seir_ready.parquet"
DEFAULT_CSV = Path(__file__).parent / "covid_pe_seir_ready.csv"

# colors
C_PRIMARY = "#1f77b4"
C_SECOND = "#17becf"
C_TREND = "#ff7f0e"
C_FATAL = "#d62728"
C_RECOV = "#2ca02c"

# -------------------------
# Helpers
# -------------------------
def apply_plotly_style(fig):
    fig.update_layout(template="plotly_white",
                      title_font=dict(size=20, color=C_PRIMARY),
                      legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
                      margin=dict(l=40, r=40, t=80, b=60))
    return fig

def download_link(df, name="export.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}">Download {name}</a>'
    return href

# -------------------------
# ODEs for models
# -------------------------
def sir_ode(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def seir_ode(y, t, beta, sigma, gamma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def seird_ode(y, t, beta, sigma, gamma, mu, N):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I - mu * I
    dRdt = gamma * I
    dDdt = mu * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def seirv_ode(y, t, beta, sigma, gamma, v_rate, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N - v_rate * S
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I + v_rate * S
    return [dSdt, dEdt, dIdt, dRdt]

# -------------------------
# Data loader (relative paths + upload)
# -------------------------
@st.cache_data
def load_data(path_parquet=None, path_csv=None, uploaded_file=None):
    # priority: uploaded -> parquet -> csv -> default parquet/csv
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, parse_dates=["date"])
        except Exception as e:
            st.error(f"Erro lendo arquivo enviado: {e}")
            raise
    else:
        # use given paths or defaults
        path_parquet = Path(path_parquet) if path_parquet else DEFAULT_PARQUET
        path_csv = Path(path_csv) if path_csv else DEFAULT_CSV

        if path_parquet.exists():
            df = pd.read_parquet(path_parquet)
        elif path_csv.exists():
            df = pd.read_csv(path_csv, parse_dates=["date"])
        else:
            return None

    # normalize columns to lowercase
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df

st.sidebar.header("Dados")
uploaded = st.sidebar.file_uploader("Upload (opcional) — CSV/Parquet pré-processado", type=["csv","parquet"])

df = load_data(uploaded_file=uploaded)

if df is None:
    st.error("Arquivo de dados pré-processado não encontrado. Coloque 'covid_pe_seir_ready.parquet' (ou .csv) na mesma pasta do app ou faça upload.")
    st.stop()

# ensure required columns exist
required = ["date", "new_cases", "cum_cases", "population"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Colunas obrigatórias ausentes no arquivo: {missing}. Ajuste seu arquivo.")
    st.stop()

# optional columns
if "i_est" not in df.columns:
    df = df.sort_values("date")
    df["i_est"] = df["new_cases"].rolling(7, min_periods=1).sum()
if "r_est" not in df.columns:
    df["r_est"] = 0

# -------------------------
# Sidebar: filters & model choices
# -------------------------
st.sidebar.header("Filtros & Modelos")
munis = sorted(df['municipio'].dropna().unique()) if 'municipio' in df.columns else []
sel_muni = st.sidebar.selectbox("Município", ["Todos"] + munis)
min_date = df['date'].min().date()
max_date = df['date'].max().date()
date_range = st.sidebar.date_input("Período", [min_date, max_date], min_value=min_date, max_value=max_date)
start_date, end_date = date_range

model_choice = st.sidebar.selectbox("Modelo", ["SIR", "SEIR", "SEIRD", "SEIRV"])
fit_method = st.sidebar.selectbox("Método de estimação", ["Least-squares (rápido)", "MCMC (opcional)"])
if fit_method == "MCMC (opcional)" and not HAS_MCMC:
    st.sidebar.warning("emcee/corner não instalados — MCMC não estará disponível.")

# priors / initial slider hints
st.sidebar.header("Parâmetros iniciais / priors")
beta_init = st.sidebar.slider("β inicial", 0.0, 2.0, 0.6, 0.01)
sigma_init = st.sidebar.slider("σ inicial (SEIR)", 0.02, 1.0, 1/5, 0.01)
gamma_init = st.sidebar.slider("γ inicial", 0.01, 1.0, 1/7, 0.01)
mu_init = st.sidebar.slider("μ inicial (SEIRD)", 0.0, 0.2, 0.01, 0.001)
v_init = st.sidebar.slider("v (vacinação diária frac.)", 0.0, 0.1, 0.0, 0.001)
init_days = st.sidebar.number_input("Dias p/ I0 (últimos dias)", min_value=1, max_value=60, value=7)

# MCMC options
if HAS_MCMC:
    n_walkers = st.sidebar.slider("Walkers (MCMC)", 16, 256, 64)
    n_steps = st.sidebar.slider("Steps por walker (MCMC)", 100, 2000, 500, step=50)
    burn_in = st.sidebar.number_input("Burn-in", min_value=0, value=int(n_steps*0.2), max_value=n_steps//2)

run_button = st.sidebar.button("▶ Rodar ajuste & simulação")

# -------------------------
# Apply filters
# -------------------------
mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
dff = df[mask].copy()
if sel_muni != "Todos" and 'municipio' in dff.columns:
    dff = dff[dff['municipio'] == sel_muni]

if dff.empty:
    st.error("Sem dados para o período/município selecionado.")
    st.stop()

# data summary
st.subheader("Resumo de dados")
c1, c2, c3 = st.columns(3)
c1.metric("Período (dias)", (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)
c1.metric("Casos no período", int(dff['new_cases'].sum()))
c2.metric("Acumulado (máx)", int(dff['cum_cases'].max()))
c2.metric("Pico diário", int(dff['new_cases'].max()))
pop_est = int(dff['population'].median())
c3.metric("População (N) usada", pop_est)

st.write("Amostra (final da série):")
st.dataframe(dff.sort_values("date").tail(8))

# plot observed
st.subheader("Séries observadas")
fig_obs = px.line(dff.sort_values('date'), x='date', y=['new_cases','i_est','r_est'], labels={'value':'Contagem','date':'Data','variable':'Série'})
st.plotly_chart(apply_plotly_style(fig_obs), use_container_width=True)

# -------------------------
# Initial conditions estimation
# -------------------------
last_win = dff.sort_values('date').tail(init_days)
I0 = int(last_win['new_cases'].sum()) if last_win['new_cases'].sum() > 0 else max(1, int(dff['new_cases'].median()))
D_inc_guess = max(1, int(round(1/sigma_init))) if sigma_init>0 else 5
E0 = int(dff.sort_values('date').tail(D_inc_guess)['new_cases'].sum())
if E0 <= 0:
    E0 = max(1, I0*2)
R0_init = int(dff['r_est'].iloc[-1]) if 'r_est' in dff.columns else 0
S0 = pop_est - E0 - I0 - R0_init
if S0 < 0:
    S0 = max(0, pop_est - (E0 + I0 + R0_init))

st.write(f"Estimativas iniciais (autom.): S0={S0}, E0={E0}, I0={I0}, R0={R0_init}")

# observed I series
y_obs = dff.sort_values('date')['i_est'].values
t_obs_len = len(y_obs)

# residual functions for least squares
def resid_sir(p):
    beta, gamma = p
    t = np.arange(t_obs_len)
    y0 = [S0, I0, R0_init]
    sol = odeint(lambda y, tt: sir_ode(y, tt, beta, gamma, pop_est), y0, t)
    I_model = sol[:,1]
    return I_model - y_obs

def resid_seir(p):
    beta, sigma, gamma = p
    t = np.arange(t_obs_len)
    y0 = [S0, E0, I0, R0_init]
    sol = odeint(lambda y, tt: seir_ode(y, tt, beta, sigma, gamma, pop_est), y0, t)
    I_model = sol[:,2]
    return I_model - y_obs

def resid_seird(p):
    beta, sigma, gamma, mu = p
    t = np.arange(t_obs_len)
    y0 = [S0, E0, I0, R0_init, 0]
    sol = odeint(lambda y, tt: seird_ode(y, tt, beta, sigma, gamma, mu, pop_est), y0, t)
    I_model = sol[:,2]
    return I_model - y_obs

def resid_seirv(p):
    beta, sigma, gamma, v = p
    t = np.arange(t_obs_len)
    y0 = [S0, E0, I0, R0_init]
    sol = odeint(lambda y, tt: seirv_ode(y, tt, beta, sigma, gamma, v, pop_est), y0, t)
    I_model = sol[:,2]
    return I_model - y_obs

# -------------------------
# Run fit & sim when requested
# -------------------------
fit_results = {}
mcmc_output = None
if run_button:
    st.info("Executando ajuste (least-squares)...")
    if model_choice == "SIR":
        x0 = [beta_init, gamma_init]
        bounds = ([0.0, 0.001], [5.0, 1.0])
        res = least_squares(resid_sir, x0, bounds=bounds, xtol=1e-8, ftol=1e-8, max_nfev=20000)
        beta_fit, gamma_fit = res.x
        fit_results.update({'beta':float(beta_fit),'gamma':float(gamma_fit)})
        R0_val = beta_fit/gamma_fit if gamma_fit>0 else np.nan
        st.success(f"SIR fit: β={beta_fit:.4f}, γ={gamma_fit:.4f}, R₀={R0_val:.3f}")
        # simulate longer horizon
        days_sim = max(180, t_obs_len+120)
        t = np.arange(days_sim)
        y0 = [S0, I0, R0_init]
        sol = odeint(lambda y, tt: sir_ode(y, tt, beta_fit, gamma_fit, pop_est), y0, t)
        sim_df = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_sim, freq='D'),
                               'S':sol[:,0],'I':sol[:,1],'R':sol[:,2]})
    elif model_choice == "SEIR":
        x0 = [beta_init, sigma_init, gamma_init]
        bounds = ([0.0, 0.01, 0.01],[5.0,1.0,1.0])
        res = least_squares(resid_seir, x0, bounds=bounds, xtol=1e-8, ftol=1e-8, max_nfev=30000)
        beta_fit, sigma_fit, gamma_fit = res.x
        fit_results.update({'beta':float(beta_fit),'sigma':float(sigma_fit),'gamma':float(gamma_fit)})
        R0_val = beta_fit/gamma_fit if gamma_fit>0 else np.nan
        st.success(f"SEIR fit: β={beta_fit:.4f}, σ={sigma_fit:.4f}, γ={gamma_fit:.4f}, R₀={R0_val:.3f}")
        days_sim = max(180, t_obs_len+120)
        t = np.arange(days_sim)
        y0 = [S0, E0, I0, R0_init]
        sol = odeint(lambda y, tt: seir_ode(y, tt, beta_fit, sigma_fit, gamma_fit, pop_est), y0, t)
        sim_df = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_sim, freq='D'),
                               'S':sol[:,0],'E':sol[:,1],'I':sol[:,2],'R':sol[:,3]})
    elif model_choice == "SEIRD":
        x0 = [beta_init, sigma_init, gamma_init, mu_init]
        bounds = ([0.0, 0.01, 0.01, 0.0],[5.0,1.0,1.0,0.5])
        res = least_squares(resid_seird, x0, bounds=bounds, xtol=1e-8, ftol=1e-8, max_nfev=40000)
        beta_fit, sigma_fit, gamma_fit, mu_fit = res.x
        fit_results.update({'beta':float(beta_fit),'sigma':float(sigma_fit),'gamma':float(gamma_fit),'mu':float(mu_fit)})
        R0_val = beta_fit/gamma_fit if gamma_fit>0 else np.nan
        st.success(f"SEIRD fit: β={beta_fit:.4f}, σ={sigma_fit:.4f}, γ={gamma_fit:.4f}, μ={mu_fit:.4f}, R₀={R0_val:.3f}")
        days_sim = max(180, t_obs_len+120)
        t = np.arange(days_sim)
        y0 = [S0, E0, I0, R0_init, 0]
        sol = odeint(lambda y, tt: seird_ode(y, tt, beta_fit, sigma_fit, gamma_fit, mu_fit, pop_est), y0, t)
        sim_df = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_sim, freq='D'),
                               'S':sol[:,0],'E':sol[:,1],'I':sol[:,2],'R':sol[:,3],'D':sol[:,4]})
    else:  # SEIRV
        x0 = [beta_init, sigma_init, gamma_init, v_init]
        bounds = ([0.0, 0.01, 0.01, 0.0],[5.0,1.0,1.0,0.2])
        res = least_squares(resid_seirv, x0, bounds=bounds, xtol=1e-8, ftol=1e-8, max_nfev=40000)
        beta_fit, sigma_fit, gamma_fit, v_fit = res.x
        fit_results.update({'beta':float(beta_fit),'sigma':float(sigma_fit),'gamma':float(gamma_fit),'v_rate':float(v_fit)})
        R0_val = beta_fit/gamma_fit if gamma_fit>0 else np.nan
        st.success(f"SEIRV fit: β={beta_fit:.4f}, σ={sigma_fit:.4f}, γ={gamma_fit:.4f}, v={v_fit:.4f}, R₀={R0_val:.3f}")
        days_sim = max(180, t_obs_len+120)
        t = np.arange(days_sim)
        y0 = [S0, E0, I0, R0_init]
        sol = odeint(lambda y, tt: seirv_ode(y, tt, beta_fit, sigma_fit, gamma_fit, v_fit, pop_est), y0, t)
        sim_df = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_sim, freq='D'),
                               'S':sol[:,0],'E':sol[:,1],'I':sol[:,2],'R':sol[:,3]})

    # plot fit vs observed (I)
    st.subheader("Ajuste — Observado vs Modelo (I)")
    fig_fit = go.Figure()
    fig_fit.add_trace(go.Scatter(x=dff.sort_values('date')['date'], y=y_obs, mode='markers+lines', name='I_obs', marker=dict(color='black')))
    fig_fit.add_trace(go.Scatter(x=sim_df['date'][:t_obs_len], y=sim_df['I'][:t_obs_len], mode='lines', name='I_model', line=dict(color=C_FATAL)))
    fig_fit.update_layout(title="Observado (I_est) vs Modelo (I)", xaxis_title="Data", yaxis_title="Infectados")
    st.plotly_chart(apply_plotly_style(fig_fit), use_container_width=True)

    # residuals
    resid = sim_df['I'][:t_obs_len].values - y_obs
    st.subheader("Resíduos do ajuste")
    fig_res = px.line(x=dff.sort_values('date')['date'], y=resid, labels={'x':'date','y':'residual'}, title='Resíduos (I_model - I_obs)')
    st.plotly_chart(apply_plotly_style(fig_res), use_container_width=True)

    # -------------------------
    # Optional MCMC (only SIR implemented here for brevity)
    # -------------------------
    if fit_method.startswith("MCMC") and HAS_MCMC:
        st.subheader("MCMC (emcee) — posterior (SIR only, extendable)")
        # implement SIR posterior sampling (extendable)
        if model_choice == "SIR":
            def log_prior(theta):
                b, g, logs = theta
                if not (0.0 < b < 5.0 and 0.001 < g < 1.0 and -10 < logs < 5):
                    return -np.inf
                return 0.0
            def log_like(theta, y):
                b, g, logs = theta
                s = np.exp(logs)
                t = np.arange(len(y))
                y0 = [S0, I0, R0_init]
                sol = odeint(lambda y, tt: sir_ode(y, tt, b, g, pop_est), y0, t)
                Im = sol[:,1]
                return -0.5 * np.sum(((y - Im)/s)**2 + np.log(2*np.pi*s**2))
            def log_prob(theta, y):
                lp = log_prior(theta)
                if not np.isfinite(lp):
                    return -np.inf
                return lp + log_like(theta, y)
            # p0 around least-squares fit (if exists) else initials
            p0 = np.array([fit_results.get('beta',beta_init), fit_results.get('gamma',gamma_init), np.log(np.std(y_obs - sim_df['I'][:t_obs_len].values)+1e-2)])
            ndim = len(p0)
            nwalkers = max(16, 64)
            p0_walkers = p0 + 1e-3 * np.random.randn(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(y_obs,))
            with st.spinner("Executando MCMC — isso pode levar alguns minutos..."):
                t0 = time.time()
                sampler.run_mcmc(p0_walkers, 500, progress=True)
                t1 = time.time()
            st.success(f"MCMC finalizado em {(t1-t0):.1f}s")
            samples = sampler.get_chain(discard=50, flat=True)
            med = np.median(samples, axis=0)
            st.write("Medianas (posterior):", med)
            try:
                fig_corner = corner.corner(samples, labels=["β","γ","log_s"], show_titles=True)
                st.pyplot(fig_corner)
            except Exception:
                st.write("Instale 'corner' para ver corner plots.")
            mcmc_output = {"sampler":sampler, "samples":samples}

    # -------------------------
    # Projection / forecast
    # -------------------------
    st.subheader("Projeção futura (modelo ajustado)")
    days_forecast = st.slider("Dias para projetar", 30, 720, 180)
    # use fitted params if present
    if model_choice == "SIR":
        b_use = fit_results.get('beta', beta_init); g_use = fit_results.get('gamma', gamma_init)
        y0 = [S0, I0, R0_init]
        solf = odeint(lambda y, tt: sir_ode(y, tt, b_use, g_use, pop_est), y0, np.arange(days_forecast))
        df_proj = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_forecast, freq='D'),
                                'S':solf[:,0],'I':solf[:,1],'R':solf[:,2]})
    elif model_choice == "SEIR":
        b_use = fit_results.get('beta', beta_init); s_use = fit_results.get('sigma', sigma_init); g_use = fit_results.get('gamma', gamma_init)
        y0 = [S0, E0, I0, R0_init]
        solf = odeint(lambda y, tt: seir_ode(y, tt, b_use, s_use, g_use, pop_est), y0, np.arange(days_forecast))
        df_proj = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_forecast, freq='D'),
                                'S':solf[:,0],'E':solf[:,1],'I':solf[:,2],'R':solf[:,3]})
    elif model_choice == "SEIRD":
        b_use = fit_results.get('beta', beta_init); s_use = fit_results.get('sigma', sigma_init); g_use = fit_results.get('gamma', gamma_init); mu_use = fit_results.get('mu', mu_init)
        y0 = [S0, E0, I0, R0_init, 0]
        solf = odeint(lambda y, tt: seird_ode(y, tt, b_use, s_use, g_use, mu_use, pop_est), y0, np.arange(days_forecast))
        df_proj = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_forecast, freq='D'),
                                'S':solf[:,0],'E':solf[:,1],'I':solf[:,2],'R':solf[:,3],'D':solf[:,4]})
    else:
        b_use = fit_results.get('beta', beta_init); s_use = fit_results.get('sigma', sigma_init); g_use = fit_results.get('gamma', gamma_init); v_use = fit_results.get('v_rate', v_init)
        y0 = [S0, E0, I0, R0_init]
        solf = odeint(lambda y, tt: seirv_ode(y, tt, b_use, s_use, g_use, v_use, pop_est), y0, np.arange(days_forecast))
        df_proj = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_forecast, freq='D'),
                                'S':solf[:,0],'E':solf[:,1],'I':solf[:,2],'R':solf[:,3]})

    ycols = [c for c in df_proj.columns if c!='date']
    fig_proj = px.line(df_proj, x='date', y=ycols, labels={'value':'População','variable':'Compartimento'}, title="Projeção futura")
    st.plotly_chart(apply_plotly_style(fig_proj), use_container_width=True)

    st.markdown(download_link(df_proj, f"projection_{model_choice.lower()}.csv"), unsafe_allow_html=True)
    st.subheader("Parâmetros estimados")
    st.json(fit_results)
else:
    st.info("Clique em 'Rodar ajuste & simulação' (barra lateral) para estimar parâmetros e gerar projeções.")

# footer
st.markdown("---")
st.markdown("**Observações:** Use este app com arquivo pré-processado (agregado por dia) contendo pelo menos `date`, `new_cases`, `cum_cases`, `population`. `i_est` e `r_est` são opcionais (i_est é estimado automaticamente). MCMC é opcional e requer `emcee`/`corner`.")
