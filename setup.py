from setuptools import find_packages,setup

setup(
    name="datamat",
    version="0.1.0",
    author= "chris nevin, ayana, aravindh",
    author_email= "chrisselfinit@gmail.com",
    install_requirements = ["langchain_huggingface","bitsandbytes","accelerate","langchain","streamlit","python-dotenv", "PyPDF2", "torch", "lanngchain_community", "transformers"],
    packages= find_packages()
)