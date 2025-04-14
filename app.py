# ─── compatibilidade SQLite / Chroma ────────────────────────────────
import sys, importlib, platform

try:
    if platform.system() != "Windows":          # Linux, macOS, Cloud…
        import pysqlite3                        # wheel ≥ 0.5.3
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
        importlib.invalidate_caches()
except ModuleNotFoundError:
    # Estamos no Windows (ou o wheel não foi instalado). 
    # Continuamos com o sqlite3 da stdlib, que já é ≥ 3.41 no Python 3.12.
    pass
# ────────────────────────────────────────────────────────────────────


import streamlit as st

# 🎯 CONFIGURAÇÕES INICIAIS DA PÁGINA - PRIMEIRO comando Streamlit
st.set_page_config(
    page_title="Chatbot PRONAF",
    page_icon="🤖",
    layout="wide"
)

import pandas as pd
import openai
import json

# 🧠 CONFIGURAÇÃO DA CHAVE DE API
# Use st.secrets para segurança no Community Cloud (ou carregue de variável de ambiente local)
#from dotenv import load_dotenv
import os

# Importações para o vector store e recuperação
# from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  # se necessário carregar novos documentos
#from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma



# Importações para RAG e prompt
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


#load_dotenv()  # Carrega variáveis do .env

openai.api_key = st.secrets["openai_api_key"]


######### Código de carregamento do vector store, RAG e funções do RAG #########
# 🗂️ CARREGAMENTO DO VETOR STORE
# Configurar os embeddings com o modelo escolhido
# embedding_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") # melhor modelo, mas mais pesado
#embedding_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # modelo mais leve e rápido, mas menos preciso

# Modelo encapsulado em função para facilitar o cache
@st.cache_resource(show_spinner="Carregando embeddings…")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_engine = load_embeddings()


# Carregar o vector store persistido
try:
    vector_db = Chroma(
        persist_directory=r"data\data-rag\persist_directory", 
        embedding_function=embedding_engine
    )
except Exception as e:
    raise RuntimeError(f"Erro ao carregar o vector_db: {e}")

# Verificar se o vector_db foi carregado corretamente
if vector_db is None:
    raise RuntimeError("O vector_db não foi carregado corretamente. Verifique o caminho e os dados.")

# Definir o número de documentos a serem recuperados
n_documentos = 3


#### Função para formatar os documentos recuperados para o RAG
def format_docs(documentos):
    """
    Concatena os conteúdos dos documentos em um único texto com separador de duas quebras de linha.
    Adicionalmente, inclui um cabeçalho com metadados relevantes (por exemplo, número da página), se disponíveis.

    Args:
        documentos (list): Lista de objetos que possuem os atributos 'page_content' e, opcionalmente, 'metadata'.

    Returns:
        str: Texto concatenado dos conteúdos dos documentos que possuem conteúdo válido.
    """
    formatted_texts = []  # Lista para armazenar os textos formatados

    for doc in documentos:
        # Verifica se o objeto possui o atributo 'page_content'
        if hasattr(doc, "page_content"):
            content = doc.page_content.strip()  # Remove espaços extras do início e fim
            # Se o conteúdo não estiver vazio
            if content:
                header = ""
                # Se existirem metadados e, em particular, o número da página estiver disponível, insere o cabeçalho
                if hasattr(doc, "metadata") and doc.metadata.get("page"):
                    header = f"Página: {doc.metadata.get('page')}\n"
                formatted_texts.append(f"{header}{content}")

    # Junta os textos formatados usando duas quebras de linha como separador
    return "\n\n".join(formatted_texts)
##############





# 🧠 Título e Introdução
st.title("📊 Chatbot PRONAF")
st.write("Interaja com os dados do Programa Nacional de Fortalecimento da Agricultura Familiar (PRONAF).")
st.markdown("Aqui você pode consultar informações sobre o PRONAF e as linhas de crédito para o Agricultor Famliar" \
            "e, também, sobre crédito agrícola por **estado**, **sexo** e **ano**.")

# ℹ️ Instruções
st.markdown("##### ℹ️ Alguns exemplos:")
st.markdown("""
- 🧮 *"Qual o valor total de crédito para o estado SP?"*
- 🗓️ *"Quantas operações foram realizadas no RS em 2024?"*
- 👩‍🌾 *"Quantas agricultoras mulheres houve em MG em 2025?"*
- 📊 *"Qual o total de crédito disponibilizado para homens e mulheres em Minas Gerais no ano de 2024?"*
- 💰 *"Valor total de crédito concedido a agricultoras no Paraná em 2024?"*
""")
st.markdown("---")



# 📦 CARREGAMENTO DO DATASET COM CACHE
@st.cache_data
def carregar_dados():
    #df = pd.read_csv("data/pronaf.csv")
    df = pd.read_parquet("data/pronaf.parquet")
    return df

# 🔧 FUNÇÃO LOCAL PARA CONSULTAR O DATASET
def consulta_pronaf_por_estado(cd_estado: str) -> str:
    """
    Filtra os dados do PRONAF por estado e retorna um resumo em Markdown.
    """
    df = carregar_dados()
    df_estado = df[df["CD_ESTADO"] == cd_estado.upper()]
    if df_estado.empty:
        return f"Nenhum dado encontrado para o estado '{cd_estado}'. Verifique o código UF."
    
    resumo = df_estado.groupby(["ANO", "SEXO_BIOLOGICO"]).agg({
        "VL_PARC_CREDITO": "sum",
        "CD_CPF_CNPJ": ["count", "nunique"]
    }).reset_index()
    
    resumo.columns = ["ANO", "SEXO_BIOLOGICO", "Soma_VL_PARC_CREDITO", "Quantidade_Operacoes", "Quantidade_Beneficiarios"]
    
    tabela = resumo.to_markdown(index=False, floatfmt=".2f")
    
    return f"Resumo dos dados do PRONAF para o estado {cd_estado}:\n\n{tabela}"

# 📜 DESCRIÇÃO DA FUNÇÃO PARA FUNCTION CALLING (TOOLS)
ferramentas = [
    {
        "type": "function",
        "function": {
            "name": "consulta_pronaf_por_estado",
            "description": "Consulta dados agregados do PRONAF por estado (UF)",
            "parameters": {
                "type": "object",
                "properties": {
                    "cd_estado": {
                        "type": "string",
                        "description": "Código do estado (UF), como 'SP', 'BA', 'RS', etc."
                    }
                },
                "required": ["cd_estado"]
            }
        }
    }
]

# 💬 CONTROLE DE HISTÓRICO DE MENSAGENS
if "mensagens" not in st.session_state:
    st.session_state.mensagens = [
        {"role": "system", "content": "Você é um assistente que responde com base nos dados do PRONAF brasileiro."}
    ]

# 📥 INTERFACE DE ENTRADA DO USUÁRIO
pergunta = st.chat_input("Digite sua pergunta sobre os dados do PRONAF...")

# 🔁 ADICIONA A PERGUNTA À CONVERSA E PROCESSA
if pergunta:
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    
    # ------------------------------------------------------------------
    # Etapa 2: Recuperar o contexto via RAG
    # ------------------------------------------------------------------
    # Aqui usamos a função que recupera os documentos e os formata
    formatted_context = (vector_db.as_retriever(k=n_documentos) | format_docs).invoke(pergunta)

        # Exibir o contexto recuperado para verificação (para debug)
    # st.markdown("### Contexto Recuperado via RAG:")
    # st.text_area("Contexto", formatted_context, height=300)

    # ------------------------------------------------------------------
    # Etapa 3: Construir o Prompt Combinado
    # ------------------------------------------------------------------
    prompt_template = ( "Você é um assistente especializado em crédito rural, com ênfase nas operações destinadas à agricultura familiar, "
                        "ou seja, nos dados do PRONAF.\n\n"
                        "Os dados a seguir foram recuperados via RAG\n\n" 
                        "Contexto Recuperado (obtido via RAG):\n"
                        "-----------------------------------------------------------\n" 
                        "{context}\n" 
                        "-----------------------------------------------------------\n\n" 
                        "Pergunta: {question}\n\n" 
                        "Se o contexto não contiver informações suficientes para responder à pergunta, ou se a consulta exigir"
                        "dados específicos de um estado (por exemplo, o código de um estado como 'SP', 'RS', etc.), " 
                        "por favor, invoque a função 'consulta_pronaf_por_estado' para obter um resumo dos dados do PRONAF para o estado em questão. "
                        "Os dados obtidos a partir da função 'consulta_pronaf_por_estado' foram extraídos da base de dados oficial disponibilizada pelo Banco Central do Brasil"
                        "disponíveis em https://www.bcb.gov.br/estabilidadefinanceira/creditorural."
                        "Com base nos dados oficiais acima e no contexto fornecido, responda utilizando linguagem simples, sendo informativo e proativo."
                        "Caso mesmo após a consulta não haja dados suficientes para responder à pergunta, "
                        "informe que não há dados suficientes para elaborar uma resposta"
                         "e solicite, cordialmente, que o usuário aprimore a pergunta." )


    combined_prompt = prompt_template.format(context=formatted_context, question=pergunta)
    
    # Para debug, você pode exibir o prompt combinado
    # st.markdown("### Prompt Combinado para a LLM:")
    # st.text_area("Prompt", combined_prompt, height=300)
    
    # Imprimi na tela >> Adiciona o prompt combinado à conversa (como uma mensagem do usuário)
    #st.session_state.mensagens.append({"role": "user", "content": combined_prompt})


    # ------------------------------------------------------------------
    # Etapa 4: Enviar o prompt combinado para a LLM
    # ------------------------------------------------------------------
    # 🔍 PRIMEIRA CHAMADA PARA VERIFICAR SE A LLM VAI USAR UMA TOOL
    resposta = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.mensagens,
        tools=ferramentas
    )

    mensagem_modelo = resposta.choices[0].message
    tool_calls = mensagem_modelo.tool_calls

    if tool_calls:
        # ✅ A LLM DECIDIU USAR UMA FUNÇÃO
        st.session_state.mensagens.append(mensagem_modelo.model_dump())  # Adiciona o pedido de tool_call à conversa

        # 🚀 EXECUTA A FUNÇÃO LOCALMENTE E ADICIONA RESPOSTAS
        for call in tool_calls:
            argumentos = json.loads(call.function.arguments)
            resultado_funcao = consulta_pronaf_por_estado(**argumentos)


            st.session_state.mensagens.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": resultado_funcao
            })


        # 🔁 SEGUNDA CHAMADA PARA OBTER A RESPOSTA FINAL
        resposta_final = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.mensagens
        )
        mensagem_final = resposta_final.choices[0].message
        st.session_state.mensagens.append({"role": "assistant", "content": mensagem_final.content})
    else:
        # ✅ A LLM RESPONDEU DIRETAMENTE (SEM USAR FUNÇÃO)
        st.session_state.mensagens.append({"role": "assistant", "content": mensagem_modelo.content})

# 💬 EXIBE O HISTÓRICO DO CHAT
for mensagem in st.session_state.mensagens:
    if mensagem["role"] in ["user", "assistant"]:
        with st.chat_message(mensagem["role"]):
            st.markdown(mensagem["content"])


