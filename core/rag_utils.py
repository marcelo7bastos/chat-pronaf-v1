"""
Funções auxiliares para o pipeline RAG.
"""

from collections.abc import Sequence
from typing import Any

def format_docs(
    docs: Sequence[Any],
    *,
    sep: str = "\n\n",
    header_key: str = "page"
) -> str:
    """
    Concatena os conteúdos dos documentos retornados pelo retriever.

    Parameters
    ----------
    docs :
        Sequência de objetos que possuam o atributo ``page_content``.
        Se contiverem ``metadata[header_key]`` o valor é usado como cabeçalho.
    sep :
        Separador entre documentos (default: duas quebras de linha).
    header_key :
        Nome da chave nos metadados cujo valor será exibido no cabeçalho
        (padrão: ``"page"``).

    Returns
    -------
    str
        Texto único pronto para ser enviado ao prompt.
    """
    chunks: list[str] = []

    for doc in docs:
        content = getattr(doc, "page_content", "").strip()
        if not content:
            continue                                # ignora vazios

        header = ""
        meta = getattr(doc, "metadata", None)
        if meta and meta.get(header_key):
            header = f"{header_key.capitalize()}: {meta[header_key]}\n"

        chunks.append(f"{header}{content}")

    return sep.join(chunks)
