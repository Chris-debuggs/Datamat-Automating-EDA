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
from langchain_community.document_loaders import CSVLoader
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
        
        # Get latest dataset
        csv_files = list(datasets_dir.glob("*.csv"))
        if not csv_files:
            logger.error("No CSV files found in datasets directory")
            raise Exception("No CSV files found in datasets directory")
        
        latest_dataset = max(csv_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Selected dataset: {latest_dataset}")
        
        # Database directory
        persist_directory = "chroma_db"
        delete_vector_db(persist_directory)  # Ensure a fresh DB
        
        # Initialize AI21 Labs LLM
        llm = ChatAI21(
            api_key=os.getenv("AI21_API_KEY"),
            model="jamba-instruct",  # ✅ Explicitly add `model`
            temperature=0.7,
            max_tokens=512
        )


        # Load & split dataset
        loader = CSVLoader(str(latest_dataset))
        data = loader.load()

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
