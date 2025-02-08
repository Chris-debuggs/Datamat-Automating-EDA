import os
import warnings
import logging
import shutil
from pathlib import Path
from langchain.globals import set_debug, set_verbose

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", "LangChainDeprecationWarning")
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

set_debug(False)
set_verbose(False)

from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.callbacks import StdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Make qa_chain global so we don't recreate it for every request
qa_chain = None

def setup_qa_chain(force_reload=False):
    """
    Setup the QA chain with the latest dataset
    force_reload: If True, forces recreation of the chain even if it exists
    """
    global qa_chain
    
    # Force reload or no existing chain
    if force_reload or qa_chain is None:
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        KEY = "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"
        
        # Fixed HuggingFaceEndpoint initialization
        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            temperature=0.7,
            huggingfacehub_api_token=KEY,
            model_kwargs={"max_length": 128}
        )

        # Get the latest dataset from the datasets directory
        datasets_dir = Path("datasets")
        if not datasets_dir.exists():
            raise Exception("No datasets directory found")
        
        # Get the most recent CSV file
        csv_files = list(datasets_dir.glob("*.csv"))
        if not csv_files:
            raise Exception("No CSV files found in datasets directory")
        
        latest_dataset = max(csv_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading dataset: {latest_dataset}")
        
        loader = CSVLoader(str(latest_dataset))
        data = loader.load()

        text_spilitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text = text_spilitter.split_documents(data)

        persist_directory = "db"
        
        # Delete existing vector database if it exists
        if os.path.exists(persist_directory):
            try:
                shutil.rmtree(persist_directory)
                print(f"Deleted existing vector database at {persist_directory}")
            except Exception as e:
                print(f"Error deleting vector database: {e}")
        
        # Create new embeddings and vector database
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

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
        
        print("QA chain reloaded with new dataset")
    
    return qa_chain