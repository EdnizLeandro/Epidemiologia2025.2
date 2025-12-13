# ============================================================
# COVID-PE — APP ROBUSTO (NUNCA FECHA SEM MOSTRAR ERRO)
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import traceback

# ------------------------------------------------------------
# CONFIGURAÇÃO CRÍTICA
# ------------------------------------------------------------
st.set_option("client.showErrorDetails", True)

BASE_DIR = Path(__file__).parent
DATA_REAL = BASE_DIR / "covid_pe_seir_ready.parquet"
DATA_MODEL = BASE_DIR / "cache.parquet"


def main():
    # --------------------------------------------------------
    # CONFIG DO APP
    # --------------------------------------------------------
    st.set_page_config(
        page_title="COVID-PE | Dados Reais e Modelos",
        layout="wide"
    )

    st.title("📊 COVID-19 EM PERNAMBUCO — DADOS REAIS E MODELOS")

    # --------------------------------------------------------
    # LOADERS (SEM CACHE DE PROPÓSITO PARA DEBUG)
    # --------------------------------------------------------
    if not DATA_REAL.exists():
        st.error(f"Arquivo NÃO encontrado: {DATA_REAL}")
        st.stop()

    if not DATA_MODEL.exists():
        st.error(f"Arquivo NÃO encontrado: {DATA_MODEL}")
        st.stop()

    df_real = pd.read_parquet(DATA_REAL)
    df_model = pd.read_parquet(DATA_MODEL)

    # Normalização pesada (EVITA 90% DOS CRASHES)
    df_real["date"] = pd.to_datetime(df_real["date"], errors="coerce")
    df_model["date"] = pd.to_datetime(df_model["date"], errors="coerce")

    df_real = df_real.dropna(subset=["date"])
    df_model = df_model.dropna(subset=["date"])

    df_real["municipio"] = df_real["municipio"].astype(str).str.upper().str.strip()
    df_model["municipio"] = df_model["municipio"].astype(str).str.upper().str.strip()
    df_model["modelo"] = df_model["modelo"].astype(str).str.upper().str.strip()

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------
    st.sidebar.header("🎛️ CONTROLES")

    municipios = sorted(set(df_real["municipio"]).intersection(df_model["municipio"]))
    municipios = ["TODOS"] + municipios
    modelos = sorted(df_model["modelo"].unique())

    sel_muni = st.sidebar.selectbox("MUNICÍPIO", municipios)
    sel_modelo = st.sidebar.selectbox("MODELO", modelos)

    min_date = min(df_real["date"].min(), df_model["date"].min()).date()
    max_date = max(df_real["date"].max(), df_model["date"].max()).date()

    ini, fim = st.sidebar.date_input(
        "PERÍODO",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    ini = pd.to_datetime(ini)
    fim = pd.to_datetime(fim)

    # --------------------------------------------------------
    # FILTRAGEM
    # --------------------------------------------------------
    real = df_real[(df_real["date"] >= ini) & (df_real["date"] <= fim)]
    model = df_model[(df_model["date"] >= ini) & (df_model["date"] <= fim)]

    if sel_muni != "TODOS":
        real = real[real["municipio"] == sel_muni]
        model = model[model["municipio"] == sel_muni]

    model = model[model["modelo"] == sel_modelo]

    # --------------------------------------------------------
    # VALIDADORES (IMPEDem FECHAMENTO)
    # --------------------------------------------------------
    if real.empty:
        st.warning("⚠️ Dados REAIS vazios após filtros.")
        st.stop()

    if model.empty:
        st.warning("⚠️ Dados do MODELO vazios após filtros.")
        st.stop()

    # --------------------------------------------------------
    # ABAS
    # --------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "📊 DADOS OBSERVADOS",
        "🧮 MODELOS",
        "⚖️ COMPARAÇÃO"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        if {"new_cases", "ma7"}.issubset(real.columns):
            st.subheader("CASOS DIÁRIOS E MÉDIA MÓVEL")
            st.plotly_chart(px.line(real, x="date", y=["new_cases", "ma7"]), True)
        else:
            st.info("Colunas new_cases / ma7 ausentes.")

    # ---------------- TAB 2 ----------------
    with tab2:
        comps = [c for c in ["S", "E", "I", "R", "D", "V"] if c in model.columns]

        if comps:
            st.subheader(f"MODELO {sel_modelo}")
            st.plotly_chart(px.line(model, x="date", y=comps), True)
        else:
            st.warning("Modelo sem compartimentos.")

    # ---------------- TAB 3 ----------------
    with tab3:
        if "I_est" in real.columns and "I" in model.columns:
            comp = pd.merge(
                real[["date", "I_est"]],
                model[["date", "I"]],
                on="date",
                how="inner"
            )

            if not comp.empty:
                st.subheader("INFECTANTES — REAL × MODELO")
                st.plotly_chart(px.line(comp, x="date", y=["I_est", "I"]), True)
            else:
                st.warning("Sem datas em comum.")
        else:
            st.info("Comparação indisponível.")


# ============================================================
# BLOCO GLOBAL — NUNCA DEIXA O APP FECHAR
# ============================================================
try:
    main()

except Exception as e:
    st.error("❌ ERRO CRÍTICO — O APP NÃO FOI FINALIZADO")
    st.code(traceback.format_exc())
