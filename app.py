import streamlit as st
from src.config_loader import get_config
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load configuration from config.yaml
config = get_config()
vector_storedb = config["vector_store_db"]
embedding_model = config["ollama_embedding"]["embedding_model"]
llm = config["ollama_model"]["ollama_llm"]

def load_vectorstore_retriever(vector_storedb, embedding_model):
    """
    Load a vector store retriever based on user-specified configuration.

    Parameters
    ----------
    vector_storedb : str
        The type of vector database to use ('faiss' or 'chroma').

    embedding_model : str
        The Ollama embedding model to use for vector representation.

    Returns
    -------
    retriever : langchain.schema.retriever.BaseRetriever
        A retriever object compatible with LangChain pipelines.
    """
    embedding = OllamaEmbeddings(model=embedding_model)

    if vector_storedb == 'faiss':
        vector_db_faiss = FAISS.load_local(
            folder_path="C:/Users/BW/Desktop/Basic gen ai chatbot project/vector_store_dbs/faiss_vecdb",
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )
        faiss_retriever = vector_db_faiss.as_retriever()
        return faiss_retriever

    elif vector_storedb == "chroma":
        vector_db_chroma = Chroma(
            persist_directory="C:/Users/BW/Desktop/Basic gen ai chatbot project/vector_store_dbs/chroma_vecdb",
            embedding_function=embedding
        )
        chroma_retriever = vector_db_chroma.as_retriever()
        return chroma_retriever

# Initialize retriever using selected vector store
retriever = load_vectorstore_retriever(vector_storedb=vector_storedb, embedding_model=embedding_model)

# Initialize LLM using Ollama
llm = Ollama(model=llm)

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a rational thinker. Use this context:\n{context}\n\nQuestion: {input}"
)

# Create document chain to process retrieved documents and generate final response
doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# Create full retrieval chain to retrieve and answer questions
retrieval_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=doc_chain
)

# Streamlit UI 

# Title of the web app
st.title("Gen AI App")

# User input field for question
question = st.text_input("Enter your question")

# Submit button to trigger response generation
if st.button("Submit"):
    with st.spinner("Generating response..."):
        # Run the retrieval chain with user's input
        response = retrieval_chain.invoke({"input": question})

    # Display the LLM's response
    st.write("Response:")
    st.write(response["answer"])