import logging

# Logging Configuration Utilities

def get_logging_config():
    """
    Set up the global logging configuration for the pipeline.

    Returns
    -------
    logging : module
        The configured logging module with file and console handlers.
    """
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            # Log to a file
            logging.FileHandler(r'C:\Users\BW\Desktop\Basic gen ai chatbot project\logs\pipeline_logs.log'),
            # Also log to the console
            logging.StreamHandler()
        ]
    )
    return logging

# Initialize and return the configured logging module
logging_config = get_logging_config()

def log_component_start(logger, component_name: str):
    """
    Log a standardized message indicating the start of a pipeline component.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance for the specific pipeline component.
    component_name : str
        The name of the pipeline component.
    """
    logger.info(f"{'='*10} Starting {component_name} {'='*10}")

def log_component_end(logger, component_name: str):
    """
    Log a standardized message indicating the end of a pipeline component.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance for the specific pipeline component.
    component_name : str
        The name of the pipeline component.
    """
    logger.info(f"{'='*10} Finished {component_name} {'='*10}\n")

# =============================================================================
# Component-Specific Loggers
# =============================================================================

# Logger for the data loader component
logger_for_loading_data = logging_config.getLogger('Data_loader_component')
logger_for_loading_data.setLevel(get_logging_config().DEBUG)

# Logger for configuration file loading
logger_for_config_file = logging_config.getLogger('Loading_config_file_component')
logger_for_config_file.setLevel(get_logging_config().DEBUG)

# Logger for text splitter component
logger_for_data_splitter = logging_config.getLogger('Data_splitter_component')
logger_for_data_splitter.setLevel(get_logging_config().DEBUG)

# Logger for embedding component
logger_for_embedder = logging_config.getLogger('Embedder_component')
logger_for_embedder.setLevel(get_logging_config().DEBUG)

# Logger for vector database builder
logger_for_vectordb_builder = logging_config.getLogger('Vectordb_builder_component')
logger_for_vectordb_builder.setLevel(get_logging_config().DEBUG)

# Logger for prompt construction
logger_for_prompt_builder = logging_config.getLogger('Prompt_builder_component')
logger_for_prompt_builder.setLevel(get_logging_config().DEBUG)

# Logger for retrieval chain execution
logger_for_retrieval_chain = logging_config.getLogger('Retrieval_chain_component')
logger_for_retrieval_chain.setLevel(get_logging_config().DEBUG)

# Logger for the overall pipeline controller
logger_for_pipeline_code = logging_config.getLogger('Pipeline_component')
logger_for_pipeline_code.setLevel(get_logging_config().DEBUG)

# Suppress Noisy Logs from Dependencies
# Suppress debug/info logs from networking and external libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)
