from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from src.logger import log_component_start, log_component_end, logger_for_retrieval_chain
from src.config_loader import get_config

def invoke_chain(retrieval_chain):
    """
    Executes a retrieval chain with a predefined query and logs the result.

    Parameters
    ----------
    retrieval_chain : RetrievalChain
        A LangChain retrieval chain that combines document retrieval and LLM-based response generation.

    Returns
    -------
    None
    
    Notes
    -----
    - The input query is loaded from a configuration file.
    - The response from the LLM is logged but not returned or saved.
    - Logging is used extensively to trace execution and debug errors.
    - This function is typically the final step in a LangChain pipeline.
    """
    try:
        # Start logging for the retrieval chain execution component
        log_component_start(logger_for_retrieval_chain, 'Run Retriever Chain Component')

        # Load configuration to get the user query
        config = get_config()
        query = config["query"]

        # Invoke the retrieval chain with the provided query
        run_chain = retrieval_chain.invoke({"input": query})
        logger_for_retrieval_chain.info('Retrieval chain executed')

        # Extract and log the LLM's response
        llm_response = run_chain['answer']
        logger_for_retrieval_chain.info("LLM's response recieved")

        logger_for_retrieval_chain.info(f"LLM's response: {llm_response}")

        # End logging for the component
        log_component_end(logger_for_retrieval_chain, 'Run Retriever Chain Component')

    except Exception as e:
        # Log any error that occurs during retrieval chain execution
        logger_for_retrieval_chain.debug(f'Error encountered in run retriever chain component. error: {e}')
        log_component_end(logger_for_retrieval_chain, 'Run Retriever Chain Component')
