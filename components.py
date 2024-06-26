from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
)
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.memory import ConversationSummaryBufferMemory,ConversationBufferMemory
import os
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

LOCAL_VECTOR_STORE_DIR = Path("./data").resolve().parent.joinpath("data", "vector_stores")

def active_chain_tracing(api_key):
    from uuid import uuid4
    from langsmith import Client
    # Langsmith for chain tracing
    unique_id = uuid4().hex[0:8]
    unique_id = "5b844b9e"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    client = Client()
    
    return client

def pdf_loader(path):
    loader = PyPDFLoader(path)
    documents = loader.load_and_split()
    
    return documents


def summaries_chain(llm):
    template = """Provide a detailed summary of the uploaded document, highlighting the key points and main arguments presented in the text. Including some detail information about the provided document.
    Document: {documents}

    Summary in the following language: {language}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = ({
            "documents": itemgetter("documents"),
            "language": itemgetter("language"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def create_memory():
    """Creates a ConversationSummaryBufferMemory for gpt-3.5-turbo
    Creates a ConversationBufferMemory for the other models."""

    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history',
        output_key="answer",
        input_key="question",
    )  
    return memory

def langchain_document_loader(TMP_DIR):
    """
    Load files from TMP_DIR (temporary directory) as documents. Files can be in txt, pdf, CSV or docx format.
    https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
    """

    documents = []

    txt_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader
    )
    documents.extend(pdf_loader.load())

    csv_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.csv", loader_cls=CSVLoader, show_progress=True,
        loader_kwargs={"encoding":"utf8"}
    )
    documents.extend(csv_loader.load())

    doc_loader = DirectoryLoader(
        TMP_DIR.as_posix(),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )
    documents.extend(doc_loader.load())
    return documents


def split_documents(documents):
    # Create a RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " ", ""],    
        chunk_size = 1600,
        chunk_overlap= 200
    )

    # Text splitting
    chunks = text_splitter.split_documents(documents=documents)
    print(f"number of chunks: {len(chunks)}")
    return chunks

def create_vectorstore(documents, google_api_key, vectorstore_name ):
    
    embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
    )
    """Create a Chroma vector database."""
    # persist_directory = (LOCAL_VECTOR_STORE_DIR.as_posix() + "/" + vectorstore_name)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        # persist_directory=persist_directory
    )
    return vector_store


def vectorstore_backed_retriever(vectorstore,search_type="similarity",k=4,score_threshold=None):
    """create a vectorsore-backed retriever
    Parameters: 
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 4) 
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs={}
    if k is not None:
        search_kwargs['k'] = k
    if score_threshold is not None:
        search_kwargs['score_threshold'] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    return retriever