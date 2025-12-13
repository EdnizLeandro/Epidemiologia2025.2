# ============================================================
# COVID-PE — DADOS REAIS + MODELOS EPIDEMIOLÓGICOS
# VERSÃO DEFINITIVA (ROBUSTA / CLOUD SAFE)
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


def read_parquet_safe(path: Path):
    """Leitura defensiva de parquet (evita crash silencioso)."""
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


def main():
    # --------------------------------------------------------
    # APP CONFIG
    # --------------------------------------------------------
    st.set_page_config(
        page_title="COVID-PE | Dados Reais e Modelos",
        layout="wide"
    )

    st.title("📊 COVID-19 EM PERNAMBUCO — DADOS REAIS E MODELOS EPIDEMIOLÓGICOS")

    # --------------------------------------------------------
    # VERIFICAÇÃO DE ARQUIVOS
    # --------------------------------------------------------
    if not DATA_REAL.exists():
        st.error(f"Arquivo NÃO encontrado: {DATA_REAL}")
        st.stop()

    if not DATA_MODEL.exists():
        st.error(f"Arquivo NÃO encontrado: {DATA_MODEL}")
        st.stop()

    # --------------------------------------------------------
    # LOAD DADOS (SEM CACHE PARA DEBUG)
    # --------------------------------------------------------
    df_real = read_parquet_safe(DATA_REAL)
    df_model = read_parquet_safe(DATA_MODEL)

    # --------------------------------------------------------
    # NORMALIZAÇÃO (EVITA 90% DOS ERROS)
    # --------------------------------------------------------
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

    municipios_validos = sorted(
        set(df_real["municipio"]).intersection(df_model["municipio"])
    )
    municipios = ["TODOS"] + municipios_validos
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
    # FILTRAGEM SEGURA
    # --------------------------------------------------------
    real = df_real[(df_real["date"] >= ini) & (df_real["date"] <= fim)]
    model = df_model[(df_model["date"] >= ini) & (df_model["date"] <= fim)]

    if sel_muni != "TODOS":
        real = real[real["municipio"] == sel_muni]
        model = model[model["municipio"] == sel_muni]

    model = model[model["modelo"] == sel_modelo]

    # --------------------------------------------------------
    # CHECAGENS
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
        "🧮 MODELOS EPIDEMIOLÓGICOS",
        "⚖️ REAL × MODELO"
    ])

    # ================= TAB 1 =================
    with tab1:
        st.subheader("CASOS DIÁRIOS E MÉDIA MÓVEL")
        if {"new_cases", "ma7"}.issubset(real.columns):
            st.plotly_chart(
                px.line(real, x="date", y=["new_cases", "ma7"]),
                use_container_width=True
            )
        else:
            st.info("Colunas new_cases / ma7 não disponíveis.")

        st.subheader("CASOS ACUMULADOS")
        if "cum_cases" in real.columns:
            st.plotly_chart(
                px.line(real, x="date", y="cum_cases"),
                use_container_width=True
            )

        st.subheader("INFECTANTES ESTIMADOS (I_EST)")
        if "I_est" in real.columns:
            st.plotly_chart(
                px.line(real, x="date", y="I_est"),
                use_container_width=True
            )

    # ================= TAB 2 =================
    with tab2:
        st.subheader(f"MODELO {sel_modelo}")

        compartimentos = [c for c in ["S", "E", "I", "R", "D", "V"] if c in model.columns]

        if not compartimentos:
            st.warning("Modelo não possui compartimentos.")
        else:
            st.plotly_chart(
                px.line(model, x="date", y=compartimentos),
                use_container_width=True
            )

            total = model[compartimentos].sum(axis=1)
            model_pct = model.copy()
            for c in compartimentos:
                model_pct[c] = 100 * model[c] / total

            st.subheader("PROPORÇÃO DA POPULAÇÃO (%)")
            st.plotly_chart(
                px.area(model_pct, x="date", y=compartimentos),
                use_container_width=True
            )

    # ================= TAB 3 =================
    with tab3:
        if "I_est" in real.columns and "I" in model.columns:
            comp = pd.merge(
                real[["date", "I_est"]],
                model[["date", "I"]],
                on="date",
                how="inner"
            )

            if comp.empty:
                st.warning("Sem interseção temporal para comparação.")
            else:
                st.subheader("INFECTANTES — REAL × MODELO")
                st.plotly_chart(
                    px.line(comp, x="date", y=["I_est", "I"]),
                    use_container_width=True
                )
        else:
            st.info("Comparação indisponível para este modelo.")


# ------------------------------------------------------------
# BLOCO GLOBAL — IMPEDE FECHAMENTO SILENCIOSO
# ------------------------------------------------------------
try:
    main()

except Exception:
    st.error("❌ ERRO CRÍTICO — TRACEBACK COMPLETO ABAIXO")
    st.code(traceback.format_exc())
