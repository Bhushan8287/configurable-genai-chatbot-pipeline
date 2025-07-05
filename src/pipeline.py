from src.data_loader import data_loader
from src.data_splitter import data_splitter
from src.embedder import embedder
from src.vectordb_builder import create_vector_store_db
from src.prompt_builder import build_prompt_chain
from src.run_retriever_chain import invoke_chain
from src.config_loader import get_config
from src.logger import log_component_start, log_component_end, logger_for_pipeline_code


def run_pipeline():
    """
    Orchestrates the entire GenAI pipeline, which includes:
    loading data, splitting it into chunks, embedding it, storing embeddings in a vector database,
    building a retrieval-based prompt chain, and invoking the chain to get an LLM response.
    
    Workflow Steps
    --------------
    1. Load raw data using the `data_loader` component.
    2. Split the loaded data into chunks using `data_splitter`.
    3. Initialize and configure the embedding model using `embedder`.
    4. Create a vector database (FAISS or Chroma) using `create_vector_store_db`.
    5. Construct a retrieval chain by combining the retriever and prompt via `build_prompt_chain`.
    6. Invoke the retrieval chain with a configured query using `invoke_chain`.
    
    Returns
    -------
    None

    Notes
    -----
    - Configuration values such as vector store type and query are fetched using `get_config()`.
    - Logging is used at every stage to ensure traceability and debugging support.
    - If an error occurs during execution, it is logged and then re-raised to halt execution.
    """
    # Load configuration file
    config = get_config()
    vector_store = config["vector_store_db"]

    try:
        # Start pipeline logging
        log_component_start(logger_for_pipeline_code, 'Pipeline Component')
        logger_for_pipeline_code.info('Pipeline execution has started')

        # Step 1: Load raw input data
        loaded_data = data_loader()

        # Step 2: Split data into smaller chunks for processing
        data_splits = data_splitter(loaded_data=loaded_data)

        # Step 3: Create embedding model and return updated data_splits
        ollama_embedding, data_splits = embedder(data_splits=data_splits)

        # Step 4: Build the vector store and obtain a retriever
        retriever = create_vector_store_db(
            data_splits=data_splits,
            embedder=ollama_embedding,
            vector_storedb=vector_store
        )

        # Step 5: Build the LLM-powered prompt chain using the retriever
        retriever_chain = build_prompt_chain(retriever=retriever)

        # Step 6: Execute the retrieval chain with a configured query
        invoke_chain(retrieval_chain=retriever_chain)

        # End pipeline logging
        logger_for_pipeline_code.info('Pipeline execution has finished')
        log_component_end(logger_for_pipeline_code, 'Pipeline Component')
    
    except Exception as pl_e:
        # Log if any step in the pipeline fails
        logger_for_pipeline_code.debug(f'Error encountered during execution of the pipeline. error {pl_e}')
        raise
