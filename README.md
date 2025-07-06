# Configurable GenAI Chatbot Pipeline

A modular, configurable GenAI chatbot pipeline demonstrating foundational GenAI concepts. This project performs end-to-end retrieval-augmented generation (RAG) from data loading to generating responses via LLMs, with support for multiple vector stores and embedding models, using `.txt` and `.pdf` documents as knowledge sources.

---

## 🔍 Project Overview

This project is designed to showcase core GenAI and RAG concepts in a clean, maintainable pipeline-based architecture. Each pipeline component is modularized for specific responsibilities, and all steps are logged for transparency. Configuration flexibility is achieved using a central `config.yaml` file, allowing users to switch between vector stores, LLMs, embeddings, and data formats without changing core logic.

---

## 💡 Motivation

The goal was to move beyond notebooks and implement GenAI functionality in a scalable, extensible codebase using modern engineering practices—modularity, logging, configuration, and environment management.

---

## ⚙️ Pipeline Architecture

```text
main.py
│
├── config_loader.py         → Loads config.yaml
├── data_loader.py           → Loads data (.txt/.pdf)
├── data_splitter.py         → Splits data using RCTSplitter
├── embedder.py              → Creates embedding model object
├── vectordb_builder.py      → Builds FAISS/Chroma vector DB and returns retriever
├── prompt_builder.py        → Builds prompt and retrieval chain
├── run_retriever_chain.py   → Executes the LLM with user query
└── app.py                   → Streamlit app for querying (separate from pipeline)
```

---

## 🚀 Features

* 📄 **Supports `.txt` and `.pdf`** data formats
* 🧠 **Local Ollama LLMs and embedding models** supported
* 🗃️ **FAISS and Chroma** vector database options
* ⚙️ **Flexible configuration** via `config.yaml`
* 📦 **Fully modular architecture** with component-level logging
* 🌐 **Simple Streamlit UI** for user interaction
* 📝 **Data credit** to [Stanford Encyclopedia of Philosophy (SEP)](https://plato.stanford.edu/entries/critical-thinking)

---

## 🧰 Tech Stack

* [LangChain](https://github.com/langchain-ai/langchain)
* [Ollama](https://ollama.com/)
* [FAISS CPU](https://github.com/facebookresearch/faiss)
* [ChromaDB](https://www.trychroma.com/)
* [Streamlit](https://streamlit.io/)
* [PyPDF](https://pypi.org/project/pypdf/)

---

## 🛠️ Getting Started

```bash
# Clone repo
git clone <your-repo-url>
cd your-project

# Create virtual env and install dependencies
python -m venv venv
venv\Scripts\activate   # or source venv/bin/activate on Linux
pip install -r requirements.txt

# Edit these paths manually in:
# src/config_loader.py → path to config.yaml
# src/logger.py        → path for logs
# src/vectordb_builder.py → output folders for FAISS/Chroma
```

---

## 🗂️ Project Structure

```
project/
│
├── config/
│   └── config.yaml
├── data/
│   ├── text_data.txt
│   └── pdf_data.pdf
├── logs/
│   └── pipeline_logs.log
├── src/
│   ├── config_loader.py
│   ├── data_loader.py
│   ├── data_splitter.py
│   ├── embedder.py
│   ├── logger.py
│   ├── prompt_builder.py
│   ├── run_retriever_chain.py
│   └── vectordb_builder.py
├── vector_store_dbs/
│   ├── faiss_vecdb/
│   └── chroma_vecdb/
├── app.py
├── app_working.png
├── main.py
└── requirements.txt
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
data_file:
  data_file_path: "data/text_data.txt"

revursive_text_splitter:
  chunk_size: 300
  chunk_overlap: 60

ollama_embedding:
  embedding_model: "mxbai-embed-large:335m"

ollama_model:
  ollama_llm: "gemma3:1b"

vector_store_db: faiss

chatprompttemplate_system_instruction: >
  You are a professional critical thinker. Use this context: {context} Question: {input}

query: "What abilities are needed for thinking critically?"
```

---

## 📘 Data Source

The data used in `.txt` and `.pdf` files is sourced from the **Stanford Encyclopedia of Philosophy** article on *Critical Thinking*.
🔗 [Source: SEP - Critical Thinking](https://plato.stanford.edu/entries/critical-thinking/#RequKnow)

All credit goes to SEP.

---

## 💬 Sample Questions

* What abilities are needed for thinking critically?
* Explain is the process of critical thinking?
* Explain the role of dispositions in critical thinking.

---

## ⚠️ Limitations

* No multi-turn chat history
* Only local Ollama-compatible models supported
* Limited data formats (no HTML/JSON yet)
* Not optimized for production environments

---

## 📈 Future Improvements

* Add chat memory/history
* Add support for JSON, HTML, and Wikipedia data loaders
* Support remote LLMs (e.g., OpenAI, Anthropic)

---

## 🏷️ License / Acknowledgments

* 📚 Content from SEP: [https://plato.stanford.edu/entries/critical-thinking](https://plato.stanford.edu/entries/critical-thinking)

---
