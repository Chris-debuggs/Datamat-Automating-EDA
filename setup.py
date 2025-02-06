from setuptools import find_packages, setup

setup(
    name="datamat",
    version="0.1.0",
    author="chris nevin, ayana, aravindh",
    author_email="chrisselfinit@gmail.com",
    install_requires=[
        "langchain",
        "streamlit",
        "python-dotenv",
        "PyPDF2",
        "langchain-community",
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "langchain-huggingface",
        "pandas",
        "numpy",
        "fastapi",
        "pydantic",
        "uvicorn"
    ],
    packages=find_packages()
)