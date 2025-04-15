# Funções auxiliares para o pipeline para consultas ao PRONAF

from core.data_loader import load_pronaf
import pandas as pd

# -------- Função de negócio --------
def consulta_pronaf_por_estado(cd_estado: str) -> str:
    """Resumo do PRONAF por UF em Markdown."""
    df: pd.DataFrame = load_pronaf()
    df_estado = df[df["CD_ESTADO"] == cd_estado.upper()]

    if df_estado.empty:
        return (
            f"Nenhum dado encontrado para o estado '{cd_estado}'. "
            "Verifique o código UF."
        )

    resumo = (
        df_estado
        .groupby(["ANO", "SEXO_BIOLOGICO"])
        .agg(
            Soma_VL_PARC_CREDITO=("VL_PARC_CREDITO", "sum"),
            Quantidade_Operacoes=("CD_CPF_CNPJ", "count"),
            Quantidade_Beneficiarios=("CD_CPF_CNPJ", "nunique"),
        )
        .reset_index()
    )

    tabela = resumo.to_markdown(index=False, floatfmt=".2f")
    return f"Resumo dos dados do PRONAF para o estado {cd_estado.upper()}:\n\n{tabela}"


# -------- Especificação p/ OpenAI function‑calling --------
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "consulta_pronaf_por_estado",
            "description": "Consulta dados agregados do PRONAF por estado (UF).",
            "parameters": {
                "type": "object",
                "properties": {
                    "cd_estado": {
                        "type": "string",
                        "description": "Código do estado (UF), como 'SP', 'BA', 'RS'…"
                    }
                },
                "required": ["cd_estado"]
            },
        },
    }
]
