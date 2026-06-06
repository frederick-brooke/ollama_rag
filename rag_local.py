import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
PDF_FILENAME = os.getenv("PDF_FILENAME")
CHROMA_PATH = os.getenv("CHROMA_PATH")

def load_documents():
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# documents = load_documents()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
        )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks.")
    return split_docs

# loaded_docs = load_documents()
# chunks = split_documents(loaded_docs)

def get_embeddings_function(model_name="nomic-embed-text"):
    #Ensure ollama server is running and the model is available
    embeddings = OllamaEmbeddings(model=model_name)
    print(f"Using Ollama model: {model_name} for embeddings.")
    return embeddings

# embedding_function = get_embedding_function()

def get_vector_store(embedding_function, persist_directory=CHROMA_PATH):
    vector_store = Chroma.from_documents(
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
    print(f"Vector store created and persisted at: {persist_directory}")
    return vector_store

# embedding_function = get_embeddings_function()
# vector_store = get_vector_store(embedding_function)

def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    vector_store.persist()
    print(f"Indexed {len(chunks)} chunks into vector store at: {persist_directory}")
    return vector_store

# vector_store = index_documents(chunks, embedding_function)

def create_rag_chain(vector_store, llm_model_name="qwen3:8b", context_window=8192):
    llm = ChatOllama(
        model=llm_model_name,
        temperature=0,
        num_ctx=context_window
    )
    print(f"Initialized ChatOllama with model: {llm_model_name} and context window: {context_window}")

    retriever = vector_store.as_retriever(
        search_type="similarity", # Use similarity search for retrieval
        search_kwargs={"k": 3} # Number of similar documents to retrieve
    )
    print("Configured retriever to use similarity search with k=3.")

    template = "Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}"

    prompt = ChatPromptTemplate.from_template(template)
    print("Created chat prompt template for RAG chain.")

    rag_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("Constructed RAG chain with retriever, prompt, and LLM.")
    return rag_chain

# vector_store = get_vector_store(embedding_function) # assuming DB is already indexed
# rag_chain = create_rag_chain(vector_store)

def query_rag(chain, question):
    print(f"Querying RAG chain with question: {question}")
    answer = chain.invoke({"question": question})
    print(f"RAG chain returned answer: {answer}")

if __name__ == "__main__":
    # 1. Load docs
    docs = load_documents()

    # 2. Split docs
    chunks = split_documents(docs)

    # 3. Get embeddings function
    embedding_function = get_embeddings_function()

    # 4. Index Documents (Only needs to be done once per document set)
    # Check if DB exists, if not, index. For simplicity, we might re-index here.
    # A more robust approach would check if indexing is needed.

    print("Attempting to index documents...")
    vector_store = index_documents(chunks, embedding_function)

    # To load existing DB instead:
    # vector_store = get_vector_store(embedding_function)

    # 5. Create RAG Chain
    rag_chain = create_rag_chain(vector_store, llm_model_name="qwen3:8b")

    # 6. Query RAG Chain
    query_question = "What are the main topics covered in the document?"
    query_rag(rag_chain, query_question)

    query_question2 = "Summarize the introduction section from the document."
    query_rag(rag_chain, query_question2)