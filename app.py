import streamlit as st
import pandas as pd
import openai
import json

# 🧠 CONFIGURAÇÃO DA CHAVE DE API
# Use st.secrets para segurança no Community Cloud (ou carregue de variável de ambiente local)
#from dotenv import load_dotenv
import os

#load_dotenv()  # Carrega variáveis do .env

openai.api_key = st.secrets["openai_api_key"]


# 🎯 CONFIGURAÇÕES INICIAIS DA PÁGINA
st.set_page_config(
    page_title="Chatbot PRONAF",
    page_icon="🤖",
    layout="wide"
)

# 🧠 Título e Introdução
st.title("📊 Chatbot PRONAF")
st.write("Interaja com os dados do Programa Nacional de Fortalecimento da Agricultura Familiar (PRONAF).")
st.markdown("Aqui você pode consultar informações sobre crédito agrícola por **estado**, **sexo** e **ano**.")

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


