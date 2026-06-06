# Ollama RAG Pipeline

A simple Retrieval-Augmented Generation (RAG) implementation using LangChain and Ollama for local, privacy-preserving document analysis and question answering.

## Overview

This project demonstrates a complete RAG pipeline that allows you to:

- 📄 Load and process PDF documents
- 🔍 Create semantic embeddings using local models
- 💾 Store embeddings in a vector database
- 🤖 Query documents using a local LLM

The entire system runs locally using [Ollama](https://ollama.ai), ensuring your data never leaves your machine.

## Architecture

```
PDF Documents
    ↓
Document Loader (PyPDFLoader)
    ↓
Text Splitter (RecursiveCharacterTextSplitter)
    ↓
Embeddings (OllamaEmbeddings + nomic-embed-text)
    ↓
Vector Store (Chroma)
    ↓
RAG Chain ← Retriever ← Vector Store
    ↓
LLM (ChatOllama + qwen3)
    ↓
Answer
```

## Prerequisites

### System Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

### Required Ollama Models

Pull the following models before running the pipeline:

```bash
# Embedding model
ollama pull nomic-embed-text

# LLM model
ollama pull qwen3:8b
```

Verify Ollama is running:
```bash
ollama serve
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ollama_rag
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the project root:

```env
DATA_PATH="./data"
PDF_FILENAME="Your Document Name.pdf"
CHROMA_PATH="Chroma_db"
```

**Configuration Options:**
- `DATA_PATH`: Directory containing PDF files
- `PDF_FILENAME`: Name of the PDF to load
- `CHROMA_PATH`: Directory where the vector database is persisted

## Usage

### Quick Start

Run the pipeline with default settings:

```bash
python rag_local.py
```

This will:
1. Load the PDF document
2. Split it into chunks
3. Generate embeddings
4. Index documents into Chroma
5. Answer predefined questions

### Using the Pipeline Programmatically

```python
from rag_local import (
    load_documents,
    split_documents,
    get_embeddings_function,
    index_documents,
    create_rag_chain,
    query_rag
)

# Step 1: Load documents
docs = load_documents()

# Step 2: Split into chunks
chunks = split_documents(docs)

# Step 3: Get embeddings
embeddings = get_embeddings_function(model_name="nomic-embed-text")

# Step 4: Index documents
vector_store = index_documents(chunks, embeddings)

# Step 5: Create RAG chain
rag_chain = create_rag_chain(vector_store, llm_model_name="qwen3:8b")

# Step 6: Query
answer = rag_chain.invoke({"question": "What is the document about?"})
```

### Querying an Existing Vector Store

If you've already indexed documents and want to query without re-indexing:

```python
from rag_local import get_vector_store, create_rag_chain, query_rag
from rag_local import get_embeddings_function

embeddings = get_embeddings_function()
vector_store = get_vector_store(embeddings)
rag_chain = create_rag_chain(vector_store)
query_rag(rag_chain, "Your question here")
```

## Project Structure

```
ollama_rag/
├── rag_local.py           # Main RAG pipeline implementation
├── .env                   # Configuration (create from .env.example)
├── .gitignore            # Git ignore rules
├── data/                 # Directory for PDF documents
│   └── your_document.pdf
├── Chroma_db/            # Persisted vector database
└── venv/                 # Python virtual environment
```

## API Reference

### Core Functions

#### `load_documents()`
Loads PDF documents from `DATA_PATH/PDF_FILENAME`.

**Returns:** List of LangChain Document objects

#### `split_documents(documents, chunk_size=1000, chunk_overlap=200)`
Splits documents into smaller chunks for processing.

**Parameters:**
- `documents`: List of Document objects
- `chunk_size`: Size of each chunk (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

**Returns:** List of split Document objects

#### `get_embeddings_function(model_name="nomic-embed-text")`
Creates an embeddings function using Ollama.

**Parameters:**
- `model_name`: Ollama model name (default: "nomic-embed-text")

**Returns:** OllamaEmbeddings instance

#### `index_documents(chunks, embedding_function, persist_directory="Chroma_db")`
Indexes documents into a Chroma vector store.

**Parameters:**
- `chunks`: List of split documents
- `embedding_function`: Embeddings instance
- `persist_directory`: Path to persist the vector database

**Returns:** Chroma vector store instance

#### `create_rag_chain(vector_store, llm_model_name="qwen3:8b", context_window=8192)`
Creates a RAG chain combining retriever, prompt, and LLM.

**Parameters:**
- `vector_store`: Chroma vector store instance
- `llm_model_name`: Ollama LLM model name
- `context_window`: LLM context window size

**Returns:** RAG chain (LangChain LCEL chain)

#### `query_rag(chain, question)`
Queries the RAG chain with a question.

**Parameters:**
- `chain`: RAG chain instance
- `question`: Question to ask

## Configuration Parameters

### Document Processing

- **chunk_size**: 1000 characters per chunk
- **chunk_overlap**: 200 character overlap between chunks

### Retrieval

- **search_type**: Similarity search
- **k**: Retrieve 3 most similar documents

### LLM Settings

- **temperature**: 0 (deterministic output)
- **context_window**: 8192 tokens (for qwen3:8b)

## Performance Notes

- **First Run**: Initial indexing may take several minutes depending on PDF size
- **Subsequent Runs**: Reuses cached embeddings from Chroma
- **Memory**: Embeddings and LLM run locally; memory depends on model size
- **Speed**: Limited by Ollama performance; consider using smaller models on resource-constrained systems

## Troubleshooting

### "Connection refused" error
Ensure Ollama is running:
```bash
ollama serve
```

### Model not found error
Pull the required models:
```bash
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

### Out of memory errors
Use a smaller LLM model:
```python
rag_chain = create_rag_chain(vector_store, llm_model_name="phi:latest")
```

### Slow responses
- Reduce context_window size
- Use a smaller embedding model: `nomic-embed-text-small`
- Reduce the number of retrieved documents (k parameter)

## Dependencies

- **langchain**: LLM orchestration framework
- **langchain-community**: Community integrations
- **langchain-ollama**: Ollama integration
- **pypdf**: PDF loading
- **chromadb**: Vector database
- **python-dotenv**: Environment variable management

For a complete list, see `requirements.txt` or install via:
```bash
pip install langchain langchain-community langchain-ollama pypdf chromadb python-dotenv
```

## Future Improvements

- [ ] Add support for multiple document formats (DOCX, TXT, etc.)
- [ ] Implement custom prompt templates
- [ ] Add result caching to reduce redundant queries
- [ ] Create a web interface
- [ ] Support for multi-document cross-referencing
- [ ] Implement document versioning
- [ ] Add conversation history/memory
- [ ] Performance optimization and benchmarking

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Resources

- [LangChain Documentation](https://python.langchain.com)
- [Ollama](https://ollama.ai)
- [Chroma Vector Database](https://www.trychroma.com)
- [RAG Best Practices](https://python.langchain.com/docs/use_cases/question_answering/)
