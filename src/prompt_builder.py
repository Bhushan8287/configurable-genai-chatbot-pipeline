from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from src.logger import log_component_start, log_component_end, logger_for_prompt_builder
from src.config_loader import get_config

def build_prompt_chain(retriever):
    """
    Builds a LangChain retrieval chain using an Ollama LLM, a document combination chain,
    and a prompt template loaded from configuration.
    
    Parameters
    ----------
    retriever : BaseRetriever
        A retriever instance used to fetch relevant documents based on input queries.

    Returns
    -------
    RetrievalChain
        A fully assembled retrieval chain that combines document retrieval and prompt-based response generation.
        
    Notes
    -----
    - The system instruction for the prompt template is loaded from the configuration file.
    - Ollama LLM is initialized using the model specified in the config.
    - Logging is used to trace component execution and any errors encountered.
    - The `ChatPromptTemplate` defines how retrieved documents are used within the LLM chain.
    """
    try:
        # Start logging for the Prompt Builder component
        log_component_start(logger_for_prompt_builder, 'Prompt Builder Component')

        # Load configuration values
        config = get_config()
        ollama_llm = config["ollama_model"]["ollama_llm"]
        system_instruction = config["chatprompttemplate_system_instruction"]

        # Initialize the Ollama LLM using the model from config
        llm = Ollama(model=ollama_llm)
        logger_for_prompt_builder.info(f'Selected LLM model: {ollama_llm}')

        # Create a chat prompt template using the system instruction from config
        prompt = ChatPromptTemplate.from_template(system_instruction)

        # Create a document combination chain (stuff method)
        doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

        # Build the retrieval chain by combining retriever with the document chain
        retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain)

        logger_for_prompt_builder.info('Retrievel chain built')
        log_component_end(logger_for_prompt_builder, 'Prompt Builder Component')

        # Return the assembled retrieval chain
        return retrieval_chain
    
    except Exception as e:
        # Log any error that occurs during prompt chain construction
        logger_for_prompt_builder.debug(f'Error encountered in prompt builder component. error: {e}')
        log_component_end(logger_for_prompt_builder, 'Prompt Builder Component')
