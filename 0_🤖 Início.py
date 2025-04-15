###############################################################################
# 0_🤖 Início.py  –  Página “home” do Chat‑PRONAF                             #
# ----------------------------------------------------------------------------#
# Esta página orquestra toda a interação:                                     #
#   • patch de compatibilidade SQLite (necessário para ChromaDB)              #
#   • carga dos recursos (embeddings + vetor store)                           #
#   • UI de boas‑vindas e caixa de chat                                       #
#   • pipeline RAG + OpenAI Function‑Calling                                  #
###############################################################################

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Ajuste de compatibilidade SQLite  (precisa vir ANTES de importar Chroma)
# ─────────────────────────────────────────────────────────────────────────────
from utils.sqlite_patch import patch_sqlite
patch_sqlite()                      # garante sqlite3 ≥ 3.35 em Linux/Cloud

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Imports padrão
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import openai, json                 # OpenAI SDK + utilitário p/ argumentos

# Recursos do projeto (módulos que você criou)
from core.rag_engine     import get_vector_db, N_DOCS
from core.rag_utils      import format_docs
from core.prompt_template import SYSTEM_MSG, build_prompt
from core.tools          import TOOLS_SPEC, consulta_pronaf_por_estado

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Configuração da página Streamlit  (DEVE ser o 1º comando Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chatbot PRONAF",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# 3‑1.  Sidebar                                                             │                               │
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # --- Cabeçalho da barra lateral ----------------------------------------
    #st.image("https://raw.githubusercontent.com/marcelo7bastos/chat-pronaf/main/.github/logo_pronaf.png", use_container_width=True) # se tiver um logo
    st.markdown("### 🤖 Chat‑PRONAF")
    st.caption("Assistente para consulta de dados do Programa Nacional de "
               "Fortalecimento da Agricultura Familiar.")
    st.caption("O chat-pronaf foi treinado com dados do [Manual de Crédito Rural](https://www3.bcb.gov.br/mcr) e, também," \
                "com parte dos dados das [Tabelas e Microdados do Crédito Rural e do Proagro](https://www.bcb.gov.br/estabilidadefinanceira/tabelas-credito-rural-proagro).")
   

    st.markdown("---")

    # --- Controles de configuração -----------------------------------------
    # 1) Quantidade de documentos que o retriever retorna
    k_docs = st.slider(
        "📄 Nº de documentos do RAG",
        min_value=1, max_value=10, value=N_DOCS, step=1,
        help="Quantos trechos de contexto serão buscados na base vetorial."
    )

    # 2) Temperatura da LLM (só como exemplo)
    temperature = st.slider(
        "🔥 Temperatura da resposta",
        min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="0 = resposta mais objetiva, 1 = mais criativa."
    )

    st.markdown("---")

    # --- Créditos / links ---------------------------------------------------
with st.sidebar:
    # … sliders / outras seções …

    # --- Rodapé personalizado --------------------------------------------
    st.markdown(
        """
        <style>
            .custom-footer { font-size: 0.85rem; line-height: 1.4; }
            .custom-footer .icon {
                width: 14px;
                height: 14px;
                margin-right: 4px;
                vertical-align: text-bottom;
            }
        </style>

        <div class="custom-footer">
            Feito por <strong>Marcelo Cabreira Bastos</strong> |
            Contato:
            <a href="mailto:marcelo.cabreira@mda.gov.br">
                marcelo.cabreira@mda.gov.br
            </a> |
            <a href="https://www.linkedin.com/in/marcelo-cabreira-bastos/" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png"
                     alt="LinkedIn" class="icon">
                LinkedIn
            </a> |
            <a href="https://api.whatsapp.com/send?phone=5561981983931" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/124/124034.png"
                     alt="WhatsApp" class="icon">
                WhatsApp
            </a>
        </div>
        """,
        unsafe_allow_html=True,   # permite HTML bruto
    )

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Chave da OpenAI (lida de .streamlit/secrets.toml)
# ─────────────────────────────────────────────────────────────────────────────
openai.api_key = st.secrets["openai_api_key"]

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Carrega (ou reconecta) ao banco vetorial do RAG
#     A função interna já tem @st.cache_resource, então só é carregado 1x
# ─────────────────────────────────────────────────────────────────────────────
vector_db = get_vector_db()

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Cabeçalho e instruções de uso
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 Chatbot PRONAF")
st.markdown(
    "Interaja com os dados do **Programa Nacional de Fortalecimento da "
    "Agricultura Familiar (PRONAF)**.\n"
    "Tire dúvidas sobre dados, regras e funcionamento do PRONAF."
    "Exemplos de perguntas:\n"
    "- *O que é o PRONAF?*\n"
    "- *Qual o valor total de crédito para o estado SP?*\n"
    "- *Quantas operações foram realizadas no RS em 2024?*\n"
    "- *Quantas agricultoras mulheres houve em MG em 2025?*"
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Estado da conversa
# ─────────────────────────────────────────────────────────────────────────────
if "msgs" not in st.session_state:
    # A primeira mensagem define o “papel” do assistente
    st.session_state.msgs = [{"role": "system", "content": SYSTEM_MSG}]

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Caixa de entrada do chat
# ─────────────────────────────────────────────────────────────────────────────
pergunta = st.chat_input("Digite sua pergunta…")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Pipeline de processamento quando o usuário envia algo
# ─────────────────────────────────────────────────────────────────────────────
if pergunta:
    # 9.1  adiciona a pergunta ao histórico
    st.session_state.msgs.append({"role": "user", "content": pergunta})

    # 9.2  Recupera contexto via RAG (k documentos) e formata
    contexto = (
        vector_db.as_retriever(k=N_DOCS) | format_docs
    ).invoke(pergunta)

    # 9.3  Constrói o prompt combinando pergunta + contexto
    prompt = build_prompt(contexto, pergunta)
    #st.session_state.msgs.append({"role": "user", "content": prompt})

    # 9.4  Primeira chamada à LLM para ver se ela solicita uma função
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.msgs,
        tools=TOOLS_SPEC,           # descreve a função disponível
    )

    msg = resp.choices[0].message

    # 9.5  Se a LLM pedir uma function‑call…
    if msg.tool_calls:
        st.session_state.msgs.append(msg.model_dump())   # log da chamada

        # executa cada chamada solicitada (pode haver mais de uma)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            resultado = consulta_pronaf_por_estado(**args)

            st.session_state.msgs.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": resultado,
            })

        # 9.6  Segunda chamada: agora a LLM responde com base no resultado
        final = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.msgs,
        )
        st.session_state.msgs.append(
            {"role": "assistant", "content": final.choices[0].message.content}
        )
    else:
        # A LLM respondeu direto, sem precisar da função
        st.session_state.msgs.append(
            {"role": "assistant", "content": msg.content}
        )

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Renderiza o histórico do chat (usuário e assistente)
# ─────────────────────────────────────────────────────────────────────────────
for m in st.session_state.msgs:
    if m["role"] in ("user", "assistant"):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])