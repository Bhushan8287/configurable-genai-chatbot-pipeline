from langchain_community.embeddings import OllamaEmbeddings
from src.logger import log_component_start, log_component_end, logger_for_embedder
from src.config_loader import get_config

def embedder(data_splits):
    """
    Initializes and returns an Ollama embedding model along with the provided data splits.
    Parameters:
    ----------
    data_splits : list
        A list of langchain.schema.Document objects that represent the content to be embedded.
    Returns:
    -------
    tuple
        A tuple containing:
        - ollama_embedding (OllamaEmbeddings): The initialized embedding model.
        - data_splits (list): The original list of documents passed into the function.
    Notes:
    -----
    - The embedding model name is loaded from a configuration file via `get_config()`.
    - Component start and end are logged for observability.
    - This function does not perform embedding itself — it prepares the embedder for downstream use.
    - All errors are caught, logged, and the component is gracefully ended on failure.
    Raises:
    -------
    Exception
        Exceptions are caught and logged, but not re-raised in this implementation.
    """
    try: 
        # Start logging for the Embedder component
        log_component_start(logger_for_embedder, 'Embedder Component')

        # Load configuration to fetch the embedding model name
        config = get_config()
        embedding_model = config['ollama_embedding']['embedding_model']

        # Log the embedding model being used
        logger_for_embedder.info(f'Embedding model in use: {embedding_model}')

        # Initialize the Ollama embedding model using the model name from config
        ollama_embedding = OllamaEmbeddings(model=embedding_model)

        # Confirm embedder object is ready to be passed forward
        logger_for_embedder.info('ollama embedder object passed to next component')

        # End logging for the Embedder component
        log_component_end(logger_for_embedder, 'Embedder Component')

        # Return both the embedder and original data splits for downstream use
        return ollama_embedding, data_splits
    
    except Exception as e:
        # Log any exceptions encountered during initialization
        logger_for_embedder.debug(f'Error encountered in embedder component. error: {e}')
        log_component_end(logger_for_embedder, 'Embedder Component')
