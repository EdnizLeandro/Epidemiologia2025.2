# app_seir_all_versions.py
"""
Streamlit app — Multi-model epidemiological simulator & fitter
Supports: SIR, SEIR, SEIRD, SEIRV
Provides: deterministic simulation, least-squares fit, optional MCMC (emcee)
Data source: /mnt/data/covid_pe_seir_ready.parquet (preprocessed, agregada)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.integrate import odeint
from scipy.optimize import least_squares
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io, base64, time, json

# Try optional packages for MCMC and corner plots
try:
    import emcee
    import corner
    HAS_MCMC = True
except Exception:
    HAS_MCMC = False

# -------------------------
# Config
# -------------------------
st.set_page_config(layout="wide", page_title="SEIR Suite — COVID-PE", initial_sidebar_state="expanded")
st.title("🔬 SEIR Suite — Modelos epidemiológicos (SIR / SEIR / SEIRD / SEIRV) — COVID-PE")

BASE_PARQUET_PATH = Path(__file__).parent / "covid_pe_seir_ready.parquet"


# Styling constants
COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#17becf"
COLOR_TREND = "#ff7f0e"
COLOR_FATAL = "#d62728"
COLOR_RECOV = "#2ca02c"
FIGURE_HEIGHT = 450

# -------------------------
# Utilities
# -------------------------
def apply_plotly_style(fig):
    fig.update_layout(template="plotly_white", title_font=dict(size=20, color=COLOR_PRIMARY),
                      legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
                      margin=dict(l=40, r=40, t=80, b=60))
    return fig

def safe_int(x, fallback=0):
    try:
        return int(np.round(float(x)))
    except Exception:
        return int(fallback)

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

def download_link_bytes(data_bytes, filename, mime="application/octet-stream"):
    b64 = base64.b64encode(data_bytes).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# -------------------------
# Model ODEs and simulators
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
    # mu: mortality rate from I -> D
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I - mu * I
    dRdt = gamma * I
    dDdt = mu * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def seirv_ode(y, t, beta, sigma, gamma, v_rate, N):
    # v_rate: vaccination rate moving S -> R (simple model)
    S, E, I, R = y
    dSdt = -beta * S * I / N - v_rate * S
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I + v_rate * S
    return [dSdt, dEdt, dIdt, dRdt]

def integrate_model(ode_func, params, y0, N, days):
    t = np.arange(days)
    sol = odeint(lambda y, tt: ode_func(y, tt, *params, N), y0, t)
    return t, sol

# -------------------------
# Load data (preprocessed parquet)
# -------------------------
@st.cache_data
def load_ready_parquet(path=BASE_PARQUET_PATH):
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

df = load_ready_parquet()
if df is None:
    st.error(f"Arquivo não encontrado: {BASE_PARQUET_PATH}. Faça upload do arquivo pre-processado ou coloque o parquet nessa pasta.")
    st.stop()

# Check minimal columns
min_required = ['date', 'new_cases', 'cum_cases', 'population']
missing = [c for c in min_required if c not in df.columns]
if missing:
    st.error(f"Colunas necessárias ausentes no arquivo: {missing}. O arquivo precisa conter pelo menos: {min_required}")
    st.stop()

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Configuração — Dados & Modelo")

uploaded = st.sidebar.file_uploader("Opcional: carregar outro arquivo pronto (csv/parquet)", type=['csv','parquet'])
if uploaded is not None:
    try:
        if uploaded.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded)
        else:
            df = pd.read_csv(uploaded, parse_dates=['date'])
        df.columns = [c.lower() for c in df.columns]
    except Exception as e:
        st.sidebar.error(f"Erro ao ler arquivo: {e}")

# Normalize and defaults
if 'i_est' not in df.columns:
    df = df.sort_values('date')
    df['i_est'] = df['new_cases'].rolling(7, min_periods=1).sum()
if 'r_est' not in df.columns:
    df['r_est'] = 0

# Filters
munis = sorted(df['municipio'].dropna().unique()) if 'municipio' in df.columns else []
sel_muni = st.sidebar.selectbox("Município", ["Todos"] + munis)
min_date = df['date'].min().date()
max_date = df['date'].max().date()
date_range = st.sidebar.date_input("Período", [min_date, max_date], min_value=min_date, max_value=max_date)

start_date, end_date = date_range
mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
dff = df[mask].copy()
if sel_muni != "Todos" and 'municipio' in dff.columns:
    dff = dff[dff['municipio'] == sel_muni]

if dff.empty:
    st.error("Não há dados para o período/município selecionado.")
    st.stop()

# Model selection
st.sidebar.header("Modelos & Ajuste")
model_type = st.sidebar.selectbox("Modelo epidemiológico", ["SIR", "SEIR", "SEIRD", "SEIRV"])
fit_method = st.sidebar.selectbox("Método de estimação", ["Least-squares (rápido)", "MCMC (detalhado, opcional)"])
if fit_method == "MCMC (detalhado, opcional)":
    if not HAS_MCMC:
        st.sidebar.warning("Pacote 'emcee' não encontrado — instale 'emcee' e 'corner' para usar MCMC.")
mcmc_walkers = st.sidebar.slider("Walkers (MCMC)", min_value=16, max_value=256, value=64)
mcmc_steps = st.sidebar.slider("Steps por walker (MCMC)", min_value=100, max_value=2000, value=500, step=50)
mcmc_burn = st.sidebar.number_input("Burn-in (MCMC)", min_value=0, max_value=mcmc_steps//2, value=int(mcmc_steps*0.2))

# Slider priors / controls
st.sidebar.header("Parâmetros iniciais / priors")
beta_init = st.sidebar.slider("β inicial", 0.0, 2.0, 0.6, 0.01)
sigma_init = st.sidebar.slider("σ inicial (SEIR)", 0.02, 1.0, 1/5, 0.01)
gamma_init = st.sidebar.slider("γ inicial", 0.01, 1.0, 1/7, 0.01)
mu_init = st.sidebar.slider("μ inicial (taxa mortalidade, SEIRD)", 0.0, 0.2, 0.01, 0.001)
vax_rate_init = st.sidebar.slider("v (vacinação diaria, fração S->R)", 0.0, 0.1, 0.0, 0.001)
init_days = st.sidebar.number_input("Dias p/ I0 (últimos dias)", min_value=1, max_value=60, value=7)

# Run buttons
if st.sidebar.button("Rodar ajuste & simulação"):
    run_flag = True
else:
    run_flag = False

# -------------------------
# Data summary & preview
# -------------------------
st.subheader("Resumo dos Dados")
c1, c2, c3 = st.columns(3)
c1.metric("Período (dias)", (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)
c1.metric("Casos no período", int(dff['new_cases'].sum()))
c2.metric("Acumulado (máx)", int(dff['cum_cases'].max()))
c2.metric("Pico diário", int(dff['new_cases'].max()))
pop_est = int(dff['population'].median())
c3.metric("População (N) usada", pop_est)

st.write("Exemplo dos dados (fim da série):")
st.dataframe(dff.sort_values('date').tail(8))

# Plot observed series
st.subheader("Séries observadas")
fig_obs = px.line(dff.sort_values('date'), x='date', y=['new_cases','i_est','r_est'], labels={'value':'Contagem','date':'Data','variable':'Série'},
                  title="new_cases, I_est, R_est")
st.plotly_chart(apply_plotly_style(fig_obs), use_container_width=True)

# -------------------------
# Estimate initial conditions
# -------------------------
# I0: sum of last init_days new_cases (fallback median)
last_window = dff.sort_values('date').tail(init_days)
I0 = int(last_window['new_cases'].sum()) if last_window['new_cases'].sum() > 0 else max(1, int(dff['new_cases'].median()))
# E0 guess: use incubation period guess (~1/sigma_init) days of new_cases
D_inc_guess = max(1, int(round(1/sigma_init)))
E0 = int(dff.sort_values('date').tail(D_inc_guess)['new_cases'].sum())
if E0 <= 0:
    E0 = max(1, I0 * 2)
R0_init = int(dff['r_est'].iloc[-1]) if 'r_est' in dff.columns else 0
S0 = pop_est - E0 - I0 - R0_init
if S0 < 0:
    S0 = max(0, pop_est - (E0 + I0 + R0_init))

st.write(f"Estimativas iniciais (autom.): S0={S0}, E0={E0}, I0={I0}, R0={R0_init}")

# -------------------------
# Model residual functions for least-squares
# -------------------------
y_obs = dff.sort_values('date')['i_est'].values
t_obs_len = len(y_obs)

def residuals_sir(params):
    beta, gamma = params
    days = t_obs_len
    _, sol = integrate_model(sir_ode, (beta, gamma), [S0,I0,R0_init], pop_est, days) if False else None
    # we call odeint directly to avoid lambda confusion
    t = np.arange(days)
    y0 = [S0, I0, R0_init]
    sol = odeint(lambda y,t0: sir_ode(y,t0,beta,gamma,pop_est), y0, t)
    I_model = sol[:,1]
    return I_model - y_obs

def residuals_seir(params):
    beta, sigma, gamma = params
    days = t_obs_len
    t = np.arange(days)
    y0 = [S0, E0, I0, R0_init]
    sol = odeint(lambda y,t0: seir_ode(y,t0,beta,sigma,gamma,pop_est), y0, t)
    I_model = sol[:,2]
    return I_model - y_obs

def residuals_seird(params):
    beta, sigma, gamma, mu = params
    days = t_obs_len
    t = np.arange(days)
    y0 = [S0, E0, I0, R0_init, 0]
    sol = odeint(lambda y,t0: seird_ode(y,t0,beta,sigma,gamma,mu,pop_est), y0, t)
    I_model = sol[:,2]
    return I_model - y_obs

def residuals_seirv(params):
    beta, sigma, gamma, v_rate = params
    days = t_obs_len
    t = np.arange(days)
    y0 = [S0, E0, I0, R0_init]
    sol = odeint(lambda y,t0: seirv_ode(y,t0,beta,sigma,gamma,v_rate,pop_est), y0, t)
    I_model = sol[:,2]
    return I_model - y_obs

# -------------------------
# Fit least-squares (deterministic)
# -------------------------
fit_results = {}
if run_flag:
    st.info("Executando ajuste (least-squares)...")
    # Choose residual function and bounds based on model_type
    if model_type == "SIR":
        x0 = [beta_init, gamma_init]
        bounds = ([0.0, 0.001], [5.0, 1.0])
        res = least_squares(residuals_sir, x0, bounds=bounds, xtol=1e-8, ftol=1e-8, max_nfev=20000)
        beta_fit, gamma_fit = res.x
        fit_results['beta'] = float(beta_fit); fit_results['gamma'] = float(gamma_fit)
        R0_val = beta_fit / gamma_fit if gamma_fit>0 else np.nan
        st.success(f"Least-squares SIR: β={beta_fit:.4f}, γ={gamma_fit:.4f}, R₀={R0_val:.3f}")
        # simulate full horizon for plotting
        days_sim = max(120, t_obs_len+120)
        t_sim = np.arange(days_sim)
        y0 = [S0,I0,R0_init]
        sol = odeint(lambda y,t0: sir_ode(y,t0,beta_fit,gamma_fit,pop_est), y0, t_sim)
        sim_df = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_sim, freq='D'),
                               'S': sol[:,0], 'I': sol[:,1], 'R': sol[:,2]})
    elif model_type == "SEIR":
        x0 = [beta_init, sigma_init, gamma_init]
        bounds = ([0.0, 0.01, 0.01], [5.0, 1.0, 1.0])
        res = least_squares(residuals_seir, x0, bounds=bounds, xtol=1e-8, ftol=1e-8, max_nfev=30000)
        beta_fit, sigma_fit, gamma_fit = res.x
        fit_results.update({'beta':float(beta_fit),'sigma':float(sigma_fit),'gamma':float(gamma_fit)})
        R0_val = beta_fit / gamma_fit if gamma_fit>0 else np.nan
        st.success(f"Least-squares SEIR: β={beta_fit:.4f}, σ={sigma_fit:.4f}, γ={gamma_fit:.4f}, R₀={R0_val:.3f}")
        days_sim = max(120, t_obs_len+120)
        t_sim = np.arange(days_sim)
        y0 = [S0,E0,I0,R0_init]
        sol = odeint(lambda y,t0: seir_ode(y,t0,beta_fit,sigma_fit,gamma_fit,pop_est), y0, t_sim)
        sim_df = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_sim, freq='D'),
                               'S': sol[:,0], 'E': sol[:,1], 'I': sol[:,2], 'R': sol[:,3]})
    elif model_type == "SEIRD":
        x0 = [beta_init, sigma_init, gamma_init, mu_init]
        bounds = ([0.0, 0.01, 0.01, 0.0], [5.0, 1.0, 1.0, 0.5])
        res = least_squares(residuals_seird, x0, bounds=bounds, xtol=1e-8, ftol=1e-8, max_nfev=40000)
        beta_fit, sigma_fit, gamma_fit, mu_fit = res.x
        fit_results.update({'beta':float(beta_fit),'sigma':float(sigma_fit),'gamma':float(gamma_fit),'mu':float(mu_fit)})
        R0_val = beta_fit / gamma_fit if gamma_fit>0 else np.nan
        st.success(f"Least-squares SEIRD: β={beta_fit:.4f}, σ={sigma_fit:.4f}, γ={gamma_fit:.4f}, μ={mu_fit:.4f}, R₀={R0_val:.3f}")
        days_sim = max(120, t_obs_len+120)
        t_sim = np.arange(days_sim)
        y0 = [S0,E0,I0,R0_init,0]
        sol = odeint(lambda y,t0: seird_ode(y,t0,beta_fit,sigma_fit,gamma_fit,mu_fit,pop_est), y0, t_sim)
        sim_df = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_sim, freq='D'),
                               'S': sol[:,0], 'E': sol[:,1], 'I': sol[:,2], 'R': sol[:,3], 'D': sol[:,4]})
    elif model_type == "SEIRV":
        x0 = [beta_init, sigma_init, gamma_init, vax_rate_init]
        bounds = ([0.0, 0.01, 0.01, 0.0], [5.0, 1.0, 1.0, 0.2])
        res = least_squares(residuals_seirv, x0, bounds=bounds, xtol=1e-8, ftol=1e-8, max_nfev=40000)
        beta_fit, sigma_fit, gamma_fit, v_fit = res.x
        fit_results.update({'beta':float(beta_fit),'sigma':float(sigma_fit),'gamma':float(gamma_fit),'v_rate':float(v_fit)})
        R0_val = beta_fit / gamma_fit if gamma_fit>0 else np.nan
        st.success(f"Least-squares SEIRV: β={beta_fit:.4f}, σ={sigma_fit:.4f}, γ={gamma_fit:.4f}, v={v_fit:.4f}, R₀={R0_val:.3f}")
        days_sim = max(120, t_obs_len+120)
        t_sim = np.arange(days_sim)
        y0 = [S0,E0,I0,R0_init]
        sol = odeint(lambda y,t0: seirv_ode(y,t0,beta_fit,sigma_fit,gamma_fit,v_fit,pop_est), y0, t_sim)
        sim_df = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_sim, freq='D'),
                               'S': sol[:,0], 'E': sol[:,1], 'I': sol[:,2], 'R': sol[:,3]})
    else:
        st.error("Modelo desconhecido.")
        sim_df = None

    # Plot fitted model vs observed
    st.subheader("Ajuste — Observado vs Modelo (I)")
    fig_fit = go.Figure()
    fig_fit.add_trace(go.Scatter(x=dff.sort_values('date')['date'], y=y_obs, mode='markers+lines', name='I_obs', marker=dict(color='black')))
    fig_fit.add_trace(go.Scatter(x=sim_df['date'][:t_obs_len], y=sim_df['I'][:t_obs_len], mode='lines', name='I_model_fit', line=dict(color=COLOR_FATAL)))
    fig_fit.update_layout(title="Observado (I_est) vs Modelo (I)", xaxis_title="Data", yaxis_title="Infectados")
    st.plotly_chart(apply_plotly_style(fig_fit), use_container_width=True)

    # Residuals plot
    resid = sim_df['I'][:t_obs_len].values - y_obs
    st.subheader("Resíduos do Ajuste")
    fig_res = px.line(x=dff.sort_values('date')['date'], y=resid, labels={'x':'date','y':'residual'}, title='Resíduos (I_model - I_obs)')
    st.plotly_chart(apply_plotly_style(fig_res), use_container_width=True)

    # -------------------------
    # Optional MCMC (emcee)
    # -------------------------
    mcmc_out = None
    if fit_method.startswith("MCMC") and HAS_MCMC:
        st.subheader("MCMC (emcee) — posterior sampling")
        # define log-prior and log-likelihood functions based on model
        def log_prior_sir(theta):
            b, g, logs = theta
            if not (0.0 < b < 5.0 and 0.001 < g < 1.0 and -10 < logs < 5):
                return -np.inf
            return 0.0

        def log_like_sir(theta, y):
            b, g, logs = theta
            s = np.exp(logs)
            days = len(y)
            y0 = [S0,I0,R0_init]
            sol = odeint(lambda y,t0: sir_ode(y,t0,b,g,pop_est), y0, np.arange(days))
            Im = sol[:,1]
            # Gaussian likelihood
            return -0.5 * np.sum(((y - Im)/s)**2 + np.log(2*np.pi*s**2))

        def log_prob_sir(theta, y):
            lp = log_prior_sir(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_like_sir(theta, y)

        # similar wrappers for other models...
        if model_type == "SIR":
            ndim = 3
            p0 = np.array([fit_results.get('beta',beta_init), fit_results.get('gamma',gamma_init), np.log(np.std(y_obs - sim_df['I'][:t_obs_len].values)+1e-2)])
            nwalkers = max(16, mcmc_walkers)
            p0_walkers = p0 + 1e-3 * np.random.randn(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_sir, args=(y_obs,))
            with st.spinner("Executando MCMC — isso pode levar tempo..."):
                t0 = time.time()
                sampler.run_mcmc(p0_walkers, mcmc_steps, progress=True)
                t1 = time.time()
            st.success(f"MCMC finalizado em {(t1-t0):.1f}s")
            samples = sampler.get_chain(discard=int(mcmc_burn), flat=True)
            # summarize
            med = np.median(samples, axis=0)
            st.write("Medianas (posterior):", med)
            # corner plot if available
            try:
                fig_corner = corner.corner(samples, labels=["β","γ","log_s"], show_titles=True)
                st.pyplot(fig_corner)
            except Exception:
                st.write("corner plot não disponível.")
            mcmc_out = {"sampler":sampler, "samples":samples}
        else:
            st.info("MCMC: implementado apenas para SIR no modo automático aqui. Podemos estender para demais modelos se desejar.")
            # (For brevity, MCMC implemented for SIR in this version. Can extend to SEIR/SEIRD/SEIRV on request.)

    elif fit_method.startswith("MCMC") and not HAS_MCMC:
        st.warning("MCMC solicitado, mas pacote 'emcee' não está instalado. Instale via 'pip install emcee corner' para ativar.")

    # -------------------------
    # Simulation result plot (future projection)
    # -------------------------
    st.subheader("Projeção futura (modelo ajustado)")
    # allow user to pick forecast horizon
    days_forecast = st.slider("Dias de previsão", min_value=30, max_value=720, value=180)
    # Re-simulate using fitted parameters (or initial guesses if fit failed)
    if model_type == "SIR":
        b_use = fit_results.get('beta', beta_init)
        g_use = fit_results.get('gamma', gamma_init)
        t_sim = np.arange(days_forecast)
        y0 = [S0,I0,R0_init]
        solf = odeint(lambda y,t0: sir_ode(y,t0,b_use,g_use,pop_est), y0, np.arange(days_forecast))
        df_proj = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_forecast, freq='D'),
                                'S':solf[:,0],'I':solf[:,1],'R':solf[:,2]})
    elif model_type == "SEIR":
        b_use = fit_results.get('beta', beta_init)
        s_use = fit_results.get('sigma', sigma_init)
        g_use = fit_results.get('gamma', gamma_init)
        y0 = [S0,E0,I0,R0_init]
        solf = odeint(lambda y,t0: seir_ode(y,t0,b_use,s_use,g_use,pop_est), y0, np.arange(days_forecast))
        df_proj = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_forecast, freq='D'),
                                'S':solf[:,0],'E':solf[:,1],'I':solf[:,2],'R':solf[:,3]})
    elif model_type == "SEIRD":
        b_use = fit_results.get('beta', beta_init)
        s_use = fit_results.get('sigma', sigma_init)
        g_use = fit_results.get('gamma', gamma_init)
        mu_use = fit_results.get('mu', mu_init)
        y0 = [S0,E0,I0,R0_init,0]
        solf = odeint(lambda y,t0: seird_ode(y,t0,b_use,s_use,g_use,mu_use,pop_est), y0, np.arange(days_forecast))
        df_proj = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_forecast, freq='D'),
                                'S':solf[:,0],'E':solf[:,1],'I':solf[:,2],'R':solf[:,3],'D':solf[:,4]})
    else:  # SEIRV
        b_use = fit_results.get('beta', beta_init)
        s_use = fit_results.get('sigma', sigma_init)
        g_use = fit_results.get('gamma', gamma_init)
        v_use = fit_results.get('v_rate', vax_rate_init)
        y0 = [S0,E0,I0,R0_init]
        solf = odeint(lambda y,t0: seirv_ode(y,t0,b_use,s_use,g_use,v_use,pop_est), y0, np.arange(days_forecast))
        df_proj = pd.DataFrame({'date': pd.date_range(dff['date'].max(), periods=days_forecast, freq='D'),
                                'S':solf[:,0],'E':solf[:,1],'I':solf[:,2],'R':solf[:,3]})

    # Plot projection
    ycols = [c for c in df_proj.columns if c!='date']
    fig_proj = px.line(df_proj, x='date', y=ycols, labels={'value':'População','variable':'Compartimento'}, title="Projeção com parâmetros ajustados")
    st.plotly_chart(apply_plotly_style(fig_proj), use_container_width=True)

    # Download simulation CSV
    csv_bytes = df_proj.to_csv(index=False).encode('utf-8')
    st.markdown(download_link_bytes(csv_bytes, f"projection_{model_type.lower()}.csv", "text/csv"), unsafe_allow_html=True)

    # Summarize params
    st.subheader("Parâmetros usados / estimados")
    st.json(fit_results)

else:
    st.info("Ajuste/simulação não executada. Clique em 'Rodar ajuste & simulação' na barra lateral para iniciar.")

# -------------------------
# Footer / tips
# -------------------------
st.markdown("---")
st.markdown("""
**Notas técnicas e recomendações**
- O app espera arquivo pré-processado (agregado por dia): `date`, `new_cases`, `cum_cases`, `population`.
- `I_est` e `R_est` são usados como observações; se não houver, `I_est` é estimado como soma móvel de 7 dias.
- Least-squares é rápido e dá estimativas pontuais. MCMC (emcee) fornece incertezas, mas é custoso.
- Para estender MCMC às versões SEIR/SEIRD/SEIRV basta adaptar a função `log_posterior` análoga ao caso SIR (posso fazer isso).
- Se quiser, adiciono: SIRD com óbitos separados, modelagem por faixa etária, β(t) variável por janelas, vacinação por compartimentos detalhados (primeira/segunda dose), ou ajuste por contas de subnotificação.
""")
