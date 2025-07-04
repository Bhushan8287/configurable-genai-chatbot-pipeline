import os
import logging
import warnings

# Logging config function
def get_logging_config():
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(r'C:\Users\BW\Desktop\Basic gen ai chatbot project\logs\pipeline_logs.log'),
            logging.StreamHandler()
        ]
    )
    return logging

logging_config = get_logging_config()

def log_component_start(logger, component_name: str):
    """
    Logs a standardized start message for a pipeline component.
    """
    logger.info(f"{'='*10} Starting {component_name} {'='*10}")

def log_component_end(logger, component_name: str):
    """
    Logs a standardized end message for a pipeline component, with spacing.
    """
    logger.info(f"{'='*10} Finished {component_name} {'='*10}\n")

logger_for_loading_data = logging_config.getLogger('Data_loader_component')
logger_for_loading_data.setLevel(get_logging_config().DEBUG)

logger_for_config_file = logging_config.getLogger('Loading_config_file_component')
logger_for_config_file.setLevel(get_logging_config().DEBUG)

logger_for_data_splitter= logging_config.getLogger('Data_splitter_component')
logger_for_data_splitter.setLevel(get_logging_config().DEBUG)

logger_for_embedder = logging_config.getLogger('Embedder_component')
logger_for_embedder.setLevel(get_logging_config().DEBUG)

logger_for_vectordb_builder = logging_config.getLogger('Vectordb_builder_component')
logger_for_vectordb_builder.setLevel(get_logging_config().DEBUG)

logger_for_prompt_builder = logging_config.getLogger('Prompt_builder_component')
logger_for_prompt_builder.setLevel(get_logging_config().DEBUG)

logger_for_retrieval_chain = logging_config.getLogger('Retrieval_chain_component')
logger_for_retrieval_chain.setLevel(get_logging_config().DEBUG)

logger_for_pipeline_code = logging_config.getLogger('Pipeline_component')
logger_for_pipeline_code.setLevel(get_logging_config().DEBUG)

# Suppress noisy HTTP logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)
