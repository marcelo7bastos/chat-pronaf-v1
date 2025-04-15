# 🤖 Chatbot PRONAF

Este projeto implementa um chatbot com **RAG (Retrieval-Augmented Generation)** para responder perguntas sobre o PRONAF — Programa Nacional de Fortalecimento da Agricultura Familiar — utilizando dados oficiais do Banco Central e documentos complementares.

O sistema é desenvolvido em **Python com Streamlit**, e utiliza **modelos da OpenAI**, **embeddings HuggingFace**, e um **vetor store com ChromaDB**.

---

## 📂 Estrutura do Projeto

# 📁 Estrutura de Diretórios — CHAT-PRONAF

```plaintext
CHAT-PRONAF/
│
├── .streamlit/                     # ⚙️ Configurações específicas do Streamlit
│   ├── config.toml                 # Tema, layout e menu lateral
│   └── secrets.toml                # Chaves de API e segredos (não versionado)
│
├── config/                         # 📘 Scripts e anotações de estudo e testes locais
│   ├── anotacoes.txt               # Notas manuais ou ideias de desenvolvimento
│   ├── data_preper.py              # Script para preparação de dados (parquet etc.)
│   ├── primeiro_rag.py             # Primeiro experimento com RAG
│   └── tira_teima.py               # Rascunhos para testes de comportamento
│
├── core/                           # 🔍 Lógica central da aplicação
│   ├── data_loader.py              # Carregamento e pré-processamento do dataset
│   ├── prompt_template.py          # Template base do prompt usado pela LLM
│   ├── rag_engine.py               # Inicialização do RAG: embeddings e ChromaDB
│   └── tools.py                    # Funções chamadas por `tool_calling` (ex: agregações por estado)
│
├── data/                           # 📦 Conjunto de dados usado pela aplicação
│   ├── data-rag/                   # Armazenamento dos vetores (Chroma persistente)
│   └── pronaf.parquet              # Base de dados do PRONAF, convertida para Parquet
│
├── pages/                          # 📄 Múltiplas páginas para navegação no app Streamlit
│   └── 1_📘 Em Construção.py      # Página placeholder para expansão futura
│
├── utils/                          # 🔧 Utilitários e patches de compatibilidade
│   ├── sqlite_patch.py             # Solução para uso de Chroma com sqlite no Cloud
│   └── utils.py                    # Funções auxiliares genéricas (ex: parse, helpers)
│
├── 0_🏠 Início.py                  # 🏁 Página inicial principal do aplicativo
├── requirements.txt               # 📋 Lista de dependências do Python
├── .gitignore                     # 🧼 Arquivos/pastas ignorados pelo Git
├── LICENSE                        # 📜 Licença de uso do projeto
└── README.md                      # 📝 Documentação principal do projeto


---

## 🚀 Como executar localmente

1. Clone o repositório:


git clone https://github.com/seu-usuario/chat-pronaf.git
cd chat-pronaf

2. Crie um ambiente virtual:
(sugestão)
python -m venv venv
source venv/bin/activate  # ou `venv\Scripts\activate` no Windows

3. Instale as dependências:
pip install -r requirements.txt

4. Configure a chave da OpenAI no arquivo .streamlit/secrets.toml:
[general]
openai_api_key = "sua-chave-da-openai"

5. Execute o app:
streamlit run app.py

---

## 🛠 Principais Tecnologias

A seguir, as tecnologias essenciais utilizadas no projeto:

- **Streamlit**  
  Framework para a criação de aplicações web interativas e intuitivas.

- **LangChain**  
  Biblioteca que simplifica a integração e orquestração de modelos de linguagem (LLMs).

- **ChromaDB**  
  Banco de dados otimizado para gerenciar embeddings e realizar buscas semânticas.

- **OpenAI GPT**  
  Modelo avançado de linguagem natural capaz de compreender e gerar textos com alta qualidade.

- **HuggingFace Transformers**  
  Coleção robusta de modelos e ferramentas para processamento de linguagem natural e aprendizado profundo.

- **Pandas**  
  Biblioteca Python essencial para análise e manipulação eficiente de dados.

- **Parquet**  
  Formato de arquivo colunar projetado para armazenamento e consulta rápida de grandes volumes de dados.

---

## 🧠 Funcionalidades
- 🔍 Chat com dados estruturados do PRONAF (RAG + CSV/parquet)

- 📄 Busca contextual em documentos oficiais via embeddings

- ⚙️ Tool calling com execução de funções customizadas

- 🧭 Interface multipágina com navegação clara

---

## 🧾 Fontes dos Dados
Banco Central do Brasil: www.bcb.gov.br/estabilidadefinanceira/creditorural

---

## 👨‍💻 Autor
Marcelo Bastos
Servidor no Ministério do Desenvolvimento Agrário e Agricultura Familiar
Desenvolvido como parte de estudos em IA Generativa e Ciência de Dados para Administração Pública.

---

## 📄 Licença
Este projeto está licenciado sob a licença MIT — veja o arquivo LICENSE para mais detalhes.