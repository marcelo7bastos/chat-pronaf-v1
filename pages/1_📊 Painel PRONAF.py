
###############################################################################
# pages/1_📊 Painel PRONAF.py                                                 #
# --------------------------------------------------------------------------- #
# Painel interativo (dashboard) com estatísticas agregadas da base PRONAF.    #
# Utiliza a mesma função `load_pronaf()` já cacheada em `core/data_loader.py` #
# e exibe KPIs, filtros e gráficos simples.                                   #
###############################################################################
 
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Painel PRONAF", page_icon="📊", layout="wide")
st.title("Painel PRONAF")

import os
import pandas as pd
import plotly.express as px
#from backend.utils import render_footer

from core.data_loader import load_pronaf          # lê o Parquet cacheado


# ─────────────────────────────────────────────────────────────────────────────
# Carrega dados
# ─────────────────────────────────────────────────────────────────────────────
df = load_pronaf()   # DataFrame com ~2,5 mi linhas (cache já cuida do custo)

# ─────────────────────────────────────────────────────────────────────────────
# Barra lateral: filtros
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filtros")

    # UF
    ufs = sorted(df["CD_ESTADO"].unique())
    uf_escolhida = st.selectbox("Estado (UF)", ["TODOS"] + ufs)

    # Ano
    anos = sorted(df["ANO"].unique())
    ano_min, ano_max = st.select_slider(
        "Intervalo de anos",
        options=anos,
        value=(anos[0], anos[-1]),
    )

    # Sexo
    sexos = sorted(df["SEXO_BIOLOGICO"].unique())
    sexo_escolhido = st.multiselect("Sexo biológico", sexos, default=sexos)

# ─────────────────────────────────────────────────────────────────────────────
# Aplica filtros
# ─────────────────────────────────────────────────────────────────────────────
df_filt = df.copy()

if uf_escolhida != "TODOS":
    df_filt = df_filt[df_filt["CD_ESTADO"] == uf_escolhida]

df_filt = df_filt[
    (df_filt["ANO"] >= ano_min) &
    (df_filt["ANO"] <= ano_max) &
    (df_filt["SEXO_BIOLOGICO"].isin(sexo_escolhido))
]

# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    total_credito = df_filt["VL_PARC_CREDITO"].sum()
    st.metric("💰 Crédito concedido (R$)", f"{total_credito:,.0f}".replace(",", "."))

with col2:
    total_oper = df_filt["CD_CPF_CNPJ"].count()
    st.metric("📄 Nº de operações", f"{total_oper:,}".replace(",", "."))

with col3:
    total_benef = df_filt["CD_CPF_CNPJ"].nunique()
    st.metric("👥 Beneficiários únicos", f"{total_benef:,}".replace(",", "."))

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 1 – Evolução do crédito por ano
# ─────────────────────────────────────────────────────────────────────────────
df_ano = (
    df_filt.groupby("ANO", as_index=False)["VL_PARC_CREDITO"].sum()
    .rename(columns={"VL_PARC_CREDITO": "Crédito"})
)

fig1 = px.bar(
    df_ano,
    x="ANO",
    y="Crédito",
    title="Evolução anual do crédito (R$)",
    labels={"Crédito": "Valor (R$)"},
)
st.plotly_chart(fig1, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 2 – Distribuição por sexo
# ─────────────────────────────────────────────────────────────────────────────
df_sexo = (
    df_filt.groupby("SEXO_BIOLOGICO", as_index=False)["VL_PARC_CREDITO"].sum()
    .rename(columns={"VL_PARC_CREDITO": "Crédito"})
)

fig2 = px.pie(
    df_sexo,
    names="SEXO_BIOLOGICO",
    values="Crédito",
    title="Distribuição do crédito por sexo",
)
st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabela detalhada (opcional)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📋 Ver tabela detalhada"):
    st.dataframe(
        df_filt[["ANO", "CD_ESTADO", "SEXO_BIOLOGICO", "VL_PARC_CREDITO"]],
        use_container_width=True,
    )



# Exibe o rodapé chamando a função
#render_footer()