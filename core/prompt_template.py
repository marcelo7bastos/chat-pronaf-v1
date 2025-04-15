"""
Contém o template base para a mensagem de sistema + função de montagem.
"""

SYSTEM_MSG = (
    "Você é um assistente especializado em crédito rural, com ênfase nas "
    "operações destinadas à agricultura familiar (PRONAF)."
)

RAW_TEMPLATE = (
    "Os dados a seguir foram recuperados via RAG.\n\n"
    "Contexto Recuperado:\n"
    "-----------------------------------------------------------\n"
    "{context}\n"
    "-----------------------------------------------------------\n\n"
    "Pergunta: {question}\n\n"
    "Se o contexto não contiver informações suficientes ou a pergunta "
    "requerer dados específicos de um estado (ex.: 'SP', 'RS'), "
    "invoque a função 'consulta_pronaf_por_estado'.\n"
    "Os dados da função são extraídos do Banco Central do Brasil "
    "(https://www.bcb.gov.br/estabilidadefinanceira/creditorural).\n\n"
    "Com base no contexto e nos dados oficiais, responda em linguagem simples, "
    "sendo informativo e proativo. Caso ainda não haja dados suficientes, "
    "explique isso e peça ao usuário para detalhar melhor a pergunta."
)

def build_prompt(context: str, question: str) -> str:
    """Insere contexto e pergunta no template."""
    return RAW_TEMPLATE.format(context=context, question=question)
