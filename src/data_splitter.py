from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter, RecursiveJsonSplitter
from src.logger import log_component_start, log_component_end, logger_for_data_splitter
from src.config_loader import get_config

def data_splitter(loaded_data):
    """
    Splits input documents into smaller chunks using RecursiveCharacterTextSplitter.

    Parameters
    ----------
    loaded_data : list
        A list of documents to be split. Each document should be a LangChain Document object or compatible with the splitter.

    Returns
    -------
    list
        A list of smaller document chunks obtained after splitting the original input.
        
    Notes
    -----
    - Configuration for chunk size and chunk overlap is loaded from the application config.
    - Uses LangChain's `RecursiveCharacterTextSplitter` for text segmentation.
    - Logs the start and end of the component, and logs errors if any occur.
    """
    try:
        # Log start of the data splitter component
        log_component_start(logger_for_data_splitter, 'Data Splitter Component')

        # Load chunking configuration from config file
        config = get_config()
        chunk_size = config['revursive_text_splitter']['chunk_size']
        chunk_overlap = config['revursive_text_splitter']['chunk_overlap']

        # Initialize text splitter with specified chunk size and overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap)
        )

        # Log data splitting
        logger_for_data_splitter.info(f'Data splitting started. chunk_size: {chunk_size} and chunk_overlap: {chunk_overlap}')
        
        # Perform the split operation
        data_splits = splitter.split_documents(loaded_data)

        # Log successful split
        logger_for_data_splitter.info('Data split completed')

        # Log end of the data splitter component
        log_component_end(logger_for_data_splitter, 'Data Splitter Component')

        return data_splits

    except Exception as e:
        # Log error if any exception is encountered
        logger_for_data_splitter.debug(f'Error encounterd in data splitter component. error: {e}')
        log_component_end(logger_for_data_splitter, 'Data Splitter Component')
