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

COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#17becf"
COLOR_TREND = "#ff7f0e"
COLOR_GRAY = "#7f7f7f"


def read_parquet_safe(path: Path):
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


# ------------------------------------------------------------
# APP PRINCIPAL
# ------------------------------------------------------------
def main():

    st.set_page_config(
        page_title="COVID-PE | Modelos Epidemiológicos",
        layout="wide"
    )

    st.title("📊 COVID-19 EM PERNAMBUCO — DADOS E MODELOS")

    # --------------------------------------------------------
    # VERIFICAÇÃO DE ARQUIVOS
    # --------------------------------------------------------
    if not DATA_REAL.exists():
        st.error("Arquivo covid_pe_seir_ready.parquet não encontrado.")
        st.stop()

    if not DATA_MODEL.exists():
        st.error("Arquivo cache.parquet não encontrado.")
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
    sel_modelo = st.sidebar.selectbox("MODELO", modelos)

    min_date = df_real["date"].min().date()
    max_date = df_real["date"].max().date()

    ini, fim = st.sidebar.date_input(
        "PERÍODO",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    ini = pd.to_datetime(ini)
    fim = pd.to_datetime(fim)

    # --------------------------------------------------------
    # DADOS OBSERVADOS (AGREGAÇÃO CORRETA)
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
    # DADOS DOS MODELOS (AGREGAÇÃO CORRETA)
    # --------------------------------------------------------
    model = df_model[
        (df_model["date"] >= ini) &
        (df_model["date"] <= fim) &
        (df_model["modelo"] == sel_modelo)
    ]

    compartimentos = [c for c in ["S", "E", "I", "R", "D", "V"] if c in model.columns]

    if sel_muni == "TODOS":
        model = (
            model
            .groupby("date", as_index=False)[compartimentos]
            .sum()
        )
    else:
        model = model[model["municipio"] == sel_muni]

    # --------------------------------------------------------
    # ABAS
    # --------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "📊 DADOS OBSERVADOS",
        "🧮 MODELOS",
        "⚖️ OBSERVADO × MODELO"
    ])

    # ================= TAB 1 =================
    with tab1:
        st.subheader("CASOS OBSERVADOS")

        fig = px.line(
            real,
            x="date",
            y=["new_cases", "I_est"],
            labels={
                "date": "DATA",
                "value": "NÚMERO DE CASOS",
                "variable": "SÉRIE"
            },
            title="CASOS OBSERVADOS — PERNAMBUCO" if sel_muni == "TODOS"
                  else f"CASOS OBSERVADOS — {sel_muni}"
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("CASOS ACUMULADOS")
        fig2 = px.line(
            real,
            x="date",
            y="cum_cases",
            labels={"date": "DATA", "cum_cases": "CASOS ACUMULADOS"}
        )
        st.plotly_chart(fig2, width="stretch")

    # ================= TAB 2 =================
    with tab2:
        st.subheader(f"MODELO {sel_modelo}")

        if compartimentos:
            fig = px.line(
                model,
                x="date",
                y=compartimentos,
                labels={
                    "date": "DATA",
                    "value": "POPULAÇÃO",
                    "variable": "COMPARTIMENTO"
                }
            )
            st.plotly_chart(fig, width="stretch")

            model_pct = model.copy()
            total = model_pct[compartimentos].sum(axis=1)
            for c in compartimentos:
                model_pct[c] = 100 * model_pct[c] / total

            st.subheader("PROPORÇÃO DA POPULAÇÃO (%)")
            fig_area = px.area(
                model_pct,
                x="date",
                y=compartimentos,
                labels={
                    "date": "DATA",
                    "value": "PERCENTUAL (%)",
                    "variable": "COMPARTIMENTO"
                }
            )
            st.plotly_chart(fig_area, width="stretch")
        else:
            st.warning("Modelo não possui compartimentos disponíveis.")

    # ================= TAB 3 =================
    with tab3:
        if "I_est" in real.columns and "I" in model.columns:
            comp = pd.merge(
                real[["date", "I_est"]],
                model[["date", "I"]],
                on="date",
                how="inner"
            )

            fig = px.line(
                comp,
                x="date",
                y=["I_est", "I"],
                labels={
                    "date": "DATA",
                    "value": "INFECTANTES",
                    "variable": "SÉRIE"
                },
                title="INFECTANTES — OBSERVADO × MODELO"
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Comparação não disponível para este modelo.")


# ------------------------------------------------------------
# EXECUÇÃO SEGURA
# ------------------------------------------------------------
try:
    main()
except Exception:
    st.error("❌ ERRO CRÍTICO NO APP")
    st.code(traceback.format_exc())
