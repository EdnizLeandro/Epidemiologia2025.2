# ============================================================
# COVID-19 EM PERNAMBUCO
# DADOS OBSERVADOS + MODELOS EPIDEMIOLÓGICOS
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import traceback

# ------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# ------------------------------------------------------------
st.set_option("client.showErrorDetails", True)

BASE_DIR = Path(__file__).parent
DATA_REAL = BASE_DIR / "covid_pe_seir_ready.parquet"
DATA_MODEL = BASE_DIR / "cache.parquet"

# ------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------------------
def read_parquet_safe(path: Path):
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


def format_plot_br(fig):
    fig.update_xaxes(
        tickformat="%d/%m/%Y",
        title_font=dict(size=16),
        tickfont=dict(size=12)
    )
    fig.update_yaxes(
        title_font=dict(size=16),
        tickfont=dict(size=12)
    )
    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=22),
        font=dict(size=14),
        legend_title_text=""
    )
    return fig


# ------------------------------------------------------------
# APP PRINCIPAL
# ------------------------------------------------------------
def main():

    st.set_page_config(
        page_title="COVID-PE | MODELOS EPIDEMIOLÓGICOS",
        layout="wide"
    )

    st.title("📊 COVID-19 EM PERNAMBUCO — DADOS E MODELOS EPIDEMIOLÓGICOS")

    # --------------------------------------------------------
    # VERIFICAÇÃO DE ARQUIVOS
    # --------------------------------------------------------
    if not DATA_REAL.exists():
        st.error("ARQUIVO covid_pe_seir_ready.parquet NÃO ENCONTRADO.")
        st.stop()

    if not DATA_MODEL.exists():
        st.error("ARQUIVO cache.parquet NÃO ENCONTRADO.")
        st.stop()

    # --------------------------------------------------------
    # CARREGAMENTO
    # --------------------------------------------------------
    df_real = read_parquet_safe(DATA_REAL)
    df_model = read_parquet_safe(DATA_MODEL)

    # --------------------------------------------------------
    # NORMALIZAÇÃO
    # --------------------------------------------------------
    df_real["date"] = pd.to_datetime(df_real["date"], errors="coerce")
    df_model["date"] = pd.to_datetime(df_model["date"], errors="coerce")

    df_real = df_real.dropna(subset=["date"])
    df_model = df_model.dropna(subset=["date"])

    df_real["municipio"] = df_real["municipio"].str.upper().str.strip()
    df_model["municipio"] = df_model["municipio"].str.upper().str.strip()
    df_model["modelo"] = df_model["modelo"].str.upper().str.strip()

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------
    st.sidebar.header("🎛️ FILTROS")

    municipios = ["TODOS"] + sorted(df_real["municipio"].unique())
    modelos = sorted(df_model["modelo"].unique())

    sel_muni = st.sidebar.selectbox("MUNICÍPIO", municipios)
    sel_modelo = st.sidebar.selectbox("MODELO EPIDEMIOLÓGICO", modelos)

    min_date = df_real["date"].min().date()
    max_date = df_real["date"].max().date()

    ini, fim = st.sidebar.date_input(
        "PERÍODO DE ANÁLISE",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    ini = pd.to_datetime(ini)
    fim = pd.to_datetime(fim)

    # --------------------------------------------------------
    # DADOS OBSERVADOS
    # --------------------------------------------------------
    real = df_real[(df_real["date"] >= ini) & (df_real["date"] <= fim)]

    if sel_muni == "TODOS":
        real = (
            real
            .groupby("date", as_index=False)
            .agg({
                "new_cases": "sum",
                "cum_cases": "sum",
                "I_est": "sum"
            })
        )
    else:
        real = real[real["municipio"] == sel_muni]

    # --------------------------------------------------------
    # DADOS DOS MODELOS
    # --------------------------------------------------------
    model = df_model[
        (df_model["date"] >= ini) &
        (df_model["date"] <= fim) &
        (df_model["modelo"] == sel_modelo)
    ]

    compartimentos = [c for c in ["S", "E", "I", "R", "D", "V"] if c in model.columns]

    if sel_muni == "TODOS":
        model = model.groupby("date", as_index=False)[compartimentos].sum()
    else:
        model = model[model["municipio"] == sel_muni]

    # --------------------------------------------------------
    # ABAS
    # --------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "📊 DADOS OBSERVADOS",
        "🧮 MODELOS EPIDEMIOLÓGICOS",
        "⚖️ OBSERVADO × MODELO"
    ])

    # ================= TAB 1 =================
    with tab1:
        fig1 = px.line(
            real,
            x="date",
            y=["new_cases", "I_est"],
            labels={
                "date": "DATA",
                "value": "NÚMERO DE CASOS",
                "variable": "SÉRIE"
            },
            title=(
                "CASOS OBSERVADOS — ESTADO DE PERNAMBUCO"
                if sel_muni == "TODOS"
                else f"CASOS OBSERVADOS — {sel_muni}"
            )
        )
        st.plotly_chart(format_plot_br(fig1), width="stretch")

        fig2 = px.line(
            real,
            x="date",
            y="cum_cases",
            labels={
                "date": "DATA",
                "cum_cases": "CASOS ACUMULADOS"
            },
            title=(
                "CASOS ACUMULADOS — ESTADO DE PERNAMBUCO"
                if sel_muni == "TODOS"
                else f"CASOS ACUMULADOS — {sel_muni}"
            )
        )
        st.plotly_chart(format_plot_br(fig2), width="stretch")

    # ================= TAB 2 =================
    with tab2:
        if compartimentos:
            fig3 = px.line(
                model,
                x="date",
                y=compartimentos,
                labels={
                    "date": "DATA",
                    "value": "POPULAÇÃO",
                    "variable": "COMPARTIMENTO"
                },
                title=f"MODELO {sel_modelo} — EVOLUÇÃO TEMPORAL"
            )
            st.plotly_chart(format_plot_br(fig3), width="stretch")

            model_pct = model.copy()
            total = model_pct[compartimentos].sum(axis=1)
            for c in compartimentos:
                model_pct[c] = 100 * model_pct[c] / total

            fig4 = px.area(
                model_pct,
                x="date",
                y=compartimentos,
                labels={
                    "date": "DATA",
                    "value": "PERCENTUAL DA POPULAÇÃO (%)",
                    "variable": "COMPARTIMENTO"
                },
                title=f"MODELO {sel_modelo} — PROPORÇÃO DA POPULAÇÃO"
            )
            st.plotly_chart(format_plot_br(fig4), width="stretch")
        else:
            st.warning("MODELO SEM COMPARTIMENTOS DISPONÍVEIS.")

    # ================= TAB 3 =================
    with tab3:
        if "I_est" in real.columns and "I" in model.columns:
            comp = pd.merge(
                real[["date", "I_est"]],
                model[["date", "I"]],
                on="date",
                how="inner"
            )

            fig5 = px.line(
                comp,
                x="date",
                y=["I_est", "I"],
                labels={
                    "date": "DATA",
                    "value": "INFECTANTES",
                    "variable": "SÉRIE"
                },
                title="INFECTANTES — DADOS OBSERVADOS VS MODELO"
            )
            st.plotly_chart(format_plot_br(fig5), width="stretch")
        else:
            st.info("COMPARAÇÃO NÃO DISPONÍVEL PARA ESTE MODELO.")


# ------------------------------------------------------------
# EXECUÇÃO SEGURA
# ------------------------------------------------------------
try:
    main()
except Exception:
    st.error("❌ ERRO CRÍTICO NO APLICATIVO")
    st.code(traceback.format_exc())
