import os
import warnings
import logging
import shutil
from pathlib import Path
import dotenv

from langchain_ai21 import ChatAI21
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    JSONLoader,
    TextLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

# Configure Logging
logger = logging.getLogger(__name__)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import shutil
import os

def delete_vector_db(persist_directory):
    """Force delete the existing vector database and recreate it with correct permissions."""
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory, ignore_errors=True)  # Deletes the entire folder
            os.makedirs(persist_directory, mode=0o777, exist_ok=True)  # Ensures correct permissions
            print(f"Deleted and recreated vector database at {persist_directory}")
        except Exception as e:
            print(f"Error deleting vector database: {e}")


def get_loader_for_file(file_path: Path):
    """Return appropriate loader based on file extension."""
    extension = file_path.suffix.lower()
    
    loaders = {
        '.csv': lambda f: CSVLoader(str(f)),
        '.pdf': lambda f: PyPDFLoader(str(f)),
        '.json': lambda f: JSONLoader(str(f), jq_schema='.', text_content=False),
        '.txt': lambda f: TextLoader(str(f)),
        '.xlsx': lambda f: UnstructuredExcelLoader(str(f)),
        '.xls': lambda f: UnstructuredExcelLoader(str(f))
    }
    
    loader_func = loaders.get(extension)
    if not loader_func:
        raise ValueError(f"Unsupported file type: {extension}")
    
    return loader_func(file_path)

def setup_qa_chain(force_reload=False):
    """
    Setup the QA chain with the latest dataset using AI21 Labs.
    """
    try:
        logger.info("Starting QA chain setup")
        
        # Check API key
        AI21_API_KEY = os.getenv("AI21_API_KEY")
        if not AI21_API_KEY:
            logger.error("AI21_API_KEY not found in environment variables")
            raise ValueError("AI21_API_KEY not found in environment variables")
            
        # Check datasets directory
        datasets_dir = Path("datasets")
        if not datasets_dir.exists():
            logger.error("Datasets directory not found")
            raise Exception("No datasets directory found")
        
        # Search for supported files recursively
        supported_extensions = ('.csv', '.pdf', '.json', '.txt', '.xlsx', '.xls')
        all_files = []
        for ext in supported_extensions:
            all_files.extend(datasets_dir.rglob(f"*{ext}"))
        
        if not all_files:
            logger.error("No supported files found in datasets directory or its subdirectories")
            raise Exception("No supported files found")
        
        # Get latest dataset across all subdirectories
        latest_dataset = max(all_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Selected dataset: {latest_dataset}")
        
        # Database directory
        persist_directory = "chroma_db"
        delete_vector_db(persist_directory)  # Ensure a fresh DB
        
        # Initialize AI21 Labs LLM
        llm = ChatAI21(
            api_key=os.getenv("AI21_API_KEY"),
            model="jamba-large-1.6-2025-03",  # ✅ Explicitly add `model`
            temperature=0.7,
            max_tokens=512
        )

        # Load & split dataset using appropriate loader
        try:
            loader = get_loader_for_file(latest_dataset)
            data = loader.load()
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text = text_splitter.split_documents(data)

        # Create embeddings
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Create new vector database
        vectordb = Chroma.from_documents(
            documents=text,
            embedding=embedding, 
            persist_directory=persist_directory
        )

        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 100})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        
        logger.info("QA chain setup completed successfully")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Error in setup_qa_chain: {str(e)}", exc_info=True)
        raise
