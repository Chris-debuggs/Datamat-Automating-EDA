import os
import warnings
import logging
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

def setup_qa_chain():
    global qa_chain
    if qa_chain is not None:
        return qa_chain
        
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    KEY = "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"
    
    # Fixed HuggingFaceEndpoint initialization
    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        temperature=0.7,
        huggingfacehub_api_token=KEY,
        model_kwargs={"max_length": 128}
    )

    loader = CSVLoader("/home/voidreaper/Projects/Mini-Project/datamat/experiments/corporate_stress_dataset_formatted.csv")
    data = loader.load()

    text_spilitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = text_spilitter.split_documents(data)

    persist_directory = "db"
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Fixed Chroma initialization
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
    
    return qa_chain