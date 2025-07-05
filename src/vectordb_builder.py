from langchain_community.vectorstores import FAISS, Chroma
from src.logger import log_component_start, log_component_end, logger_for_vectordb_builder

def create_vector_store_db(data_splits, embedder, vector_storedb):
    """
    Creates a vector store (FAISS or Chroma) from the provided data splits using the specified embedding model.
    Parameters:
    ----------
    data_splits : list
        A list of langchain.schema.Document objects representing the split data to be embedded.
    embedder : BaseEmbedding
        An embedding model instance compatible with LangChain (e.g., OpenAIEmbeddings, HuggingFaceEmbeddings).
    vector_storedb : str
        Specifies the vector store type to use. Supported values are 'faiss' and 'chroma'.
    Returns:
    -------
    Retriever
        A retriever object generated from the created vector store, which can be used for similarity search or downstream LLM chains.
    Raises:
    ------
    Exception
        Any exceptions raised during vector store creation will be logged and re-raised.
    Notes:
    -----
    - The vector stores are persisted locally to specified directories.
    - Logging is performed at each step to track component execution and failures.
    """

    try: 
        if vector_storedb == 'faiss':

            # Start logging for FAISS vector DB creation
            log_component_start(logger_for_vectordb_builder, 'Vectordb Builder Component')
            logger_for_vectordb_builder.info('Vector store db selected is: FAISS')

            # Create FAISS vector store and persist to local disk
            vector_db_faiss = FAISS.from_documents(documents=data_splits, embedding=embedder)
            vector_db_faiss.save_local("C:/Users/BW/Desktop/Basic gen ai chatbot project/vector_store_dbs/faiss_vecdb") 

            # Convert FAISS vector store to retriever
            faiss_retriever = vector_db_faiss.as_retriever()

            logger_for_vectordb_builder.info('Embedding of data splits completed and vector store FAISS is created')
            log_component_end(logger_for_vectordb_builder, 'Vectordb Builder Component')

            return faiss_retriever
        
        elif vector_storedb == 'chroma':

            # Start logging for Chroma vector DB creation
            log_component_start(logger_for_vectordb_builder, 'Vectordb Builder Component')
            logger_for_vectordb_builder.info('Vector store db selected is: CHROMA')

            # Create Chroma vector store and persist to local disk
            vector_db_chroma = Chroma.from_documents(
                documents=data_splits,
                embedding=embedder,
                persist_directory="C:/Users/BW/Desktop/Basic gen ai chatbot project/vector_store_dbs/chroma_vecdb"
            ) 

            # Convert Chroma vector store to retriever
            chroma_retriever = vector_db_chroma.as_retriever()
            
            logger_for_vectordb_builder.info('Embedding of data splits completed and vector store CHROMA is created')
            log_component_end(logger_for_vectordb_builder, 'Vectordb Builder Component')
            
            return chroma_retriever

    except Exception as e:
        # Log any exceptions raised during the vector DB creation
        logger_for_vectordb_builder.debug(f'Error encountered in vectordb builder component. error: {e}')
        log_component_end(logger_for_vectordb_builder, 'Vectordb Builder Component')
