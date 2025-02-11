import os
import warnings
import logging
import shutil
from pathlib import Path
from logger import logging

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.callbacks import StdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Make qa_chain global so we don't recreate it for every request
qa_chain = None

logger = logging.getLogger(__name__)

def setup_qa_chain(force_reload=False):
    """
    Setup the QA chain with the latest dataset using Groq
    force_reload: If True, forces recreation of the chain even if it exists
    """
    try:
        logger.info("Starting QA chain setup")
        logger.debug(f"Force reload: {force_reload}")
        
        # Check API key
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY not found in environment variables")
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
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
        
        # Check database directory
        persist_directory = "chroma_db"
        
        # Force close any existing connections
        import sqlite3
        try:
            conn = sqlite3.connect(f"{persist_directory}/chroma.db")
            conn.close()
        except Exception as e:
            logger.warning(f"Database cleanup attempt: {e}")

        # Check directory permissions before proceeding
        if os.path.exists(persist_directory):
            try:
                # Test write permissions
                test_file = os.path.join(persist_directory, "test_write")
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                logger.info("Write permissions verified for chroma_db")
            except PermissionError:
                logger.error("No write permissions for chroma_db directory")
                # Try to fix permissions
                try:
                    os.chmod(persist_directory, 0o777)
                    logger.info("Fixed permissions for chroma_db directory")
                except Exception as e:
                    logger.error(f"Failed to fix permissions: {e}")
                    raise
        else:
            # Create directory with proper permissions
            os.makedirs(persist_directory, mode=0o777, exist_ok=True)
            logger.info(f"Created {persist_directory} with write permissions")
        
        # Initialize Groq LLM
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="gemma2-9b-it",  # Using Gemma2 model
            temperature=0.7,
            max_tokens=512
        )

        loader = CSVLoader(str(latest_dataset))
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text = text_splitter.split_documents(data)

        # Create new embeddings and vector database
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}  # BGE models need normalization
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