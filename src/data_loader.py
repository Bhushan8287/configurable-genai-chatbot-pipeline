from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from src.logger import log_component_start, log_component_end, logger_for_loading_data
from src.config_loader import get_config

def data_loader():
    """
    Loads data from a file (either .txt or .pdf) and returns it as a list of LangChain Document objects.

    Returns
    -------
    list
        A list of Document objects representing the loaded data.
        
    Notes
    -----
    - The file path is retrieved from the configuration file under `data_file -> data_file_path`.
    - Supports loading of `.txt` and `.pdf` files.
    - For `.txt`, it reads the entire content as a single Document.
    - For `.pdf`, it uses PyPDFLoader to load and split the document.
    - Logs the start, success, and any exceptions encountered during the data loading process.
    """
    try:
        # Log start of the data loader component
        log_component_start(logger_for_loading_data, 'Data Loader Component')

        # Load configuration and fetch the path to the data file
        config = get_config()
        data_file_path = config['data_file']['data_file_path']
        logger_for_loading_data.info('Data is being loaded')

        # Load text file and wrap content in a Document object
        if data_file_path.endswith(".txt"):
            with open(data_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            doc = Document(page_content=content, metadata={"source": "text_data.txt"}) # change source as its hardcoded
            loaded_data = [doc]

        # Use PyPDFLoader for PDF files
        elif data_file_path.endswith(".pdf"):
            pdf_loader = PyPDFLoader(data_file_path)
            loaded_data = pdf_loader.load()

        # Log successful loading
        logger_for_loading_data.info('Data loaded succesfully')

        # Log end of the data loader component
        log_component_end(logger_for_loading_data, 'Data Loader Component')

        return loaded_data

    except Exception as e:
        # Log any error encountered during data loading
        logger_for_loading_data.debug(f'Error encountered in data loader component. error: {e}')

        # Ensure the component end is always logged
        log_component_end(logger_for_loading_data, 'Data Loader Component')
