# Ollama RAG Pipeline

A simple Retrieval-Augmented Generation (RAG) implementation using LangChain and Ollama for local, privacy-preserving document analysis and question answering.

## Overview

This project demonstrates a complete RAG pipeline that allows you to:

- 📄 Load and process PDF documents
- 🔍 Create semantic embeddings using local models
- 💾 Store embeddings in a vector database
- 🤖 Query documents using a local LLM
- 🛠️ Run a small tool-calling agent for date/time demos

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

### Agent Demo

The repository also includes `agent_local.py`, a small LangChain tool-calling agent that uses a local Ollama model (`qwen3:0.6b` by default). The agent can:

- 🕐 Return the current date and time (`get_current_datetime`)
- 🌐 Search the web (`web_search`)
- 🔍 Semantically search the local knowledge base (`semantic_search`) — retrieved passages from the indexed PDF are grounded into answers with page citations

Run it from the project virtual environment:

```bash
source venv/bin/activate
python agent_local.py
```

The agent prints a `[tool]` line whenever a tool is invoked, so you can follow the agent's flow:

```text
[tool] get_current_datetime invoked with format=%Y-%m-%d %H:%M:%S
[tool] semantic_search invoked with query='architecture patterns for AI agents'
```

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

## Demo

Below is a real output from running the agent with `qwen3:4b` (`get_agent_llm(model_name="qwen3:4b")`). It shows the agent deciding when to call a tool and how it grounds answers in the retrieved document.

| # | User query | Tool invoked | Result |
|---|------------|--------------|--------|
| 1 | "Search for local cafes in London and rank the top 3 based on user reviews." | `web_search` | ✅ Ranked the top 3 sources by user reviews |
| 2 | "What is the current date?" | `get_current_datetime` | ✅ Current date returned |
| 3 | "What time is it right now? Use HH:MM format." | `get_current_datetime` | ✅ Correct time returned (`17:25`) |
| 4 | "What are the common architecture patterns…" | `semantic_search` | ✅ Grounded answer quoting Page 9 of the document |
| 5 | "Tell me a joke." | *(none)* | ✅ Answered directly — no tool needed |

```text
Invoking agent...
Input: Search for local cafes in London and rank the top 3 based on user reviews.
[tool] web_search invoked with query='top 3 local cafes in London ranked by user reviews'

Agent Response:
Based on the web search results, here are the top platforms where you can find
London cafes ranked by user reviews (though the exact top 3 names aren't
explicitly listed in the results):

1. **Yusker** – Lists the *24 best cafés in London* ranked by **real community
   signals** (not paid reviews). This platform emphasizes authentic local feedback.
2. **Tripadvisor** – Features the *10 best cafes in London* (note: the "2026" date
   in the result appears to be a typo) based on **traveller reviews**.
3. **Not Another Sunday** – Highlights *15 highest-ranked independent cafes* in
   London using metrics like **NRI (local reputation)**, **buzz**, and
   **community credibility**.

While the search results don't provide the specific top 3 cafe names, these
platforms are the most relevant sources for user-reviewed rankings in London. For
the most current and detailed lists, I recommend checking Yusker or Tripadvisor
directly (as they prioritize real-time user feedback).

*Note: The "2026" date in Tripadvisor's result likely indicates a placeholder or
error, as current data would be for 2024/2025.*

────────────────────────────────────────────────────────────────────────────────

Invoking agent...
Input: What is the current date?
[tool] get_current_datetime invoked with format=%Y-%m-%d

Agent Response:
The current date is **2026-09-28**.

────────────────────────────────────────────────────────────────────────────────

Invoking agent...
Input: What time is it right now? Use HH:MM format.
[tool] get_current_datetime invoked with format=%H:%M

Agent Response:
The current time is **17:25**.

────────────────────────────────────────────────────────────────────────────────

Invoking agent...
Input: What are the common architecture patterns for building effective AI agents
described in the document?
[tool] semantic_search invoked with query='common architecture patterns for building effective AI agents'
Retrieved 3 documents for 'common architecture patterns for building effective AI agents'.

Agent Response:
The document describes **foundational design principles** as key architecture
patterns for building effective AI agents, with a focus on **"Start simple, scale
intelligently"** as a core pattern. This approach emphasizes beginning with
single-purpose agents that excel at one task before gradually evolving into more
complex systems to match business requirements.

As noted in Chapter 3 (Page 9), the document states:
> *"We suggest teams begin with single-purpose agents that do one thing well,
> then gradually develop them into more sophisticated systems as your
> requirements..."*

This pattern helps avoid over-engineering while aligning technical complexity
with business value. The document highlights that architectural choices must be
tailored to specific use cases (e.g., customer support agents vs. multi-domain
research systems), but the foundational pattern of incremental scaling is
explicitly emphasized as a best practice.

**Source**: Page 9 of *Building Effective AI Agents: Architecture Patterns and
Implementation Frameworks* (Chapter 3).

────────────────────────────────────────────────────────────────────────────────

Invoking agent...
Input: Tell me a joke.

Agent Response:
Here's a quick joke for you:

**Why don't scientists trust atoms?**
*Because they make up everything!*

😄
```

> **Note:** This showcase was run with `qwen3:4b` for dependable tool use. The agent's default model is the much faster `qwen3:0.6b`, which is lighter and quicker but can be less reliable at following tool instructions. Choose per run by passing `model_name` to `get_agent_llm()`, e.g. `get_agent_llm(model_name="qwen3:4b")` for reliability or `get_agent_llm(model_name="qwen3:0.6b")` for speed.

## Project Structure

```
ollama_rag/
├── rag_local.py           # Main RAG pipeline implementation
├── agent_local.py         # LangChain agent example with a custom date/time tool
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
