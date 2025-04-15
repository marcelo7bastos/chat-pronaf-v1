from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
PERSIST_DIR = "data/data-rag/persist_directory"
N_DOCS      = 3   # k do retriever

def get_embeddings():
    # pode colocar @st.cache_resource aqui se quiser
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def get_vector_db():
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings()
    )
