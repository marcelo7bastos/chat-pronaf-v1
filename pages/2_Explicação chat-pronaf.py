import streamlit as st
import graphviz

# Cabeçalho da página
st.title("Como Funciona o Chat-PRONAF?")
st.markdown(
    """
    O **Chat-PRONAF** integra duas técnicas avançadas para fornecer respostas robustas e contextualizadas:
    - **RAG (Retrieval Augmented Generation)**
    - **Function Calling**
    """
)

st.markdown("---")

# Seção sobre o RAG
st.header("1. RAG (Retrieval Augmented Generation)")
st.markdown(
    """
    **RAG** é uma técnica que combina a capacidade de _recuperação de informações_ com a geração de texto por meio de modelos de linguagem.
    
    **Como funciona no Chat-PRONAF:**
    
    1. **Recuperação de Contexto:**  
       Ao receber uma pergunta do usuário, o sistema consulta um banco de dados vetorial (representado pela variável `vector_db`) para buscar documentos que contenham dados e informações relevantes sobre o PRONAF.
    
    2. **Formatação dos Dados:**  
       Os documentos recuperados são formatados e adicionados ao prompt da LLM, enriquecendo o contexto com informações importantes.
       
    3. **Geração de Resposta:**  
       Com o contexto integrado, a LLM (modelo de linguagem) gera uma resposta fundamentada nas informações extraídas, aumentando a relevância e precisão da resposta.
    """
)

st.markdown("---")

# Seção sobre Function Calling
st.header("2. Function Calling")
st.markdown(
    """
    A técnica de **Function Calling** permite que o Chat-PRONAF execute funções específicas quando a própria LLM identifica que é necessário obter dados complementares ou realizar alguma operação.
    
    **Como funciona no Chat-PRONAF:**
    
    1. **Análise da Pergunta:**  
       Após montar o prompt com o contexto, a LLM é chamada para gerar uma resposta.
    
    2. **Decisão da LLM:**  
       Se a LLM entender que a resposta depende de dados processados por uma função (por exemplo, consultar a base de dados ou calcular algum valor), ela insere em sua resposta uma indicação para a chamada de uma função.
       
    3. **Execução da Função:**  
       O sistema detecta essa indicação, executa a função correspondente (no exemplo, `consulta_pronaf_por_estado`) com os argumentos fornecidos, e adiciona o resultado ao histórico da conversa.
       
    4. **Resposta Final:**  
       Em uma nova chamada à LLM, o resultado da função é incorporado ao contexto para que a resposta final seja gerada de forma mais completa e precisa.
    """
)

st.markdown("---")

# Fluxo de processamento (fluxograma com Graphviz)
st.header("Fluxo de Processamento")
flow_code = """
graph TD
   A[Usuário envia pergunta]
   B[Recuperação via RAG (consulta ao vector_db)]
   C[Formatação dos documentos e atualização do prompt]
   D[Primeira chamada à LLM]
   E{LLM solicita<br>function calling?}
   F[Executa função solicitada<br> (ex: consulta_pronaf_por_estado)]
   G[Adiciona resultado da função<br> ao histórico]
   H[Segunda chamada à LLM<br> com o resultado incluído]
   I[Resposta final para o usuário]
   
   A --> B
   B --> C
   C --> D
   D --> E
   E -- Sim --> F
   F --> G
   G --> H
   H --> I
   E -- Não --> I
"""
st.graphviz_chart(flow_code)

st.markdown("---")

# Resumo final
st.markdown(
    """
    **Resumo:**  
    O Chat-PRONAF combina a robustez da recuperação de informações com a flexibilidade da execução de funções.
    Essa integração permite responder de forma contextualizada e dinâmica, utilizando:
    
    - **RAG:** Para buscar e formatar dados relevantes do PRONAF.
    - **Function Calling:** Para realizar operações específicas sempre que a LLM identificar essa necessidade.
    
    Essa arquitetura híbrida garante respostas precisas e adaptadas aos dados e regras do PRONAF.
    """
)
