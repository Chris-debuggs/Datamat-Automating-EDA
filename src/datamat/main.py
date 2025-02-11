import os
# Set this before importing any HuggingFace modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from logging_config import setup_logging
import logging

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

logger.info("Starting application...")
app = FastAPI() 