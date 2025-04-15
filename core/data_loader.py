import pandas as pd
import streamlit as st

@st.cache_data
def load_pronaf(path: str = "data/pronaf.parquet") -> pd.DataFrame:
    """Lê o Parquet do PRONAF com cache de Streamlit."""
    return pd.read_parquet(path)