import os
import warnings
import logging
import shutil
from pathlib import Path
from langchain.globals import set_debug, set_verbose
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.callbacks import StdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Make qa_chain global so we don't recreate it for every request
qa_chain = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", "LangChainDeprecationWarning")
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

set_debug(False)
set_verbose(False)


def delete_vector_db(persist_directory):
    """
    Force delete the existing vector database if it exists
    """
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory, ignore_errors=True)  # Ignores permission errors
            os.makedirs(persist_directory, exist_ok=True)  # Recreate the directory
            print(f"Deleted and recreated vector database at {persist_directory}")
        except Exception as e:
            print(f"Error deleting vector database: {e}")


def setup_qa_chain(force_reload=False):
    """
    Setup the QA chain with the latest dataset
    force_reload: If True, forces recreation of the chain even if it exists
    """
    global qa_chain

    # Force reload or no existing chain
    if force_reload or qa_chain is None:
        model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Or any other suitable model

        try:
            # Load the model locally.  This might take a considerable amount of time and disk space.
            tokenizer = AutoTokenizer.from_pretrained(model_id,device_map ="cpu")
            model = AutoModelForCausalLM.from_pretrained(model_id,device_map ="cpu")
            
            # Create a pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=3500,
                max_new_tokens=500,
                temperature=0.7
            )

            # Use the pipeline with Langchain
            llm = HuggingFacePipeline(pipeline=pipe)

        except Exception as e:
            print(f"Error loading model: {e}")
            return None # Or raise the exception, depending on error handling preference.


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
        delete_vector_db(persist_directory)

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

