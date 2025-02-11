from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, HttpUrl
from groq_utils import setup_qa_chain
import os
from pathlib import Path
from datetime import datetime
import aiohttp
import aiofiles
from typing import Optional, List
import shutil
import logging
from logger import logging
import kaggle

# Configure logging for this module
logger = logging.getLogger(__name__)

app = FastAPI(title="DATAmat Groq API")

# Configure download directory
DOWNLOAD_DIR = Path("datasets")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Initialize QA chain
qa_chain = setup_qa_chain()

class Query(BaseModel):
    question: str

class DatasetDownload(BaseModel):
    url: HttpUrl
    filename: Optional[str] = None

class KaggleDatasetDownload(BaseModel):
    dataset_name: str  # Format: "username/dataset-name"
    filename: Optional[str] = None

@app.post("/groq/ask")
async def ask_question(query: Query):
    """Endpoint to ask questions using Groq's LLM"""
    try:
        result = qa_chain.invoke({"query": query.question})
        return {"answer": result["result"]}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question with Groq: {str(e)}"
        )

@app.post("/groq/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and process a new dataset file"""
    try:
        logger.info(f"Starting upload for file: {file.filename}")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = DOWNLOAD_DIR / safe_filename
        
        logger.debug(f"Generated safe filename: {safe_filename}")
        logger.debug(f"Target file path: {file_path}")
        
        # Log directory permissions
        logger.debug(f"Download directory permissions: {oct(DOWNLOAD_DIR.stat().st_mode)[-3:]}")
        
        # Save uploaded file
        try:
            logger.info("Attempting to save file...")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"File saved successfully at {file_path}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}", exc_info=True)
            raise
        finally:
            file.file.close()
        
        # Update the QA chain
        logger.info("Updating QA chain with new dataset")
        global qa_chain
        qa_chain = setup_qa_chain(force_reload=True)
        
        response = {
            "message": "Dataset uploaded and processed successfully with Groq",
            "filename": safe_filename,
            "path": str(file_path),
            "size_bytes": os.path.getsize(file_path)
        }
        logger.info(f"Upload successful: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error in upload_dataset: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading dataset for Groq processing: {str(e)}"
        )

@app.post("/groq/download-dataset")
async def download_dataset(dataset: DatasetDownload):
    """Download a dataset from a URL"""
    try:
        # Generate filename if not provided
        if not dataset.filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = Path(str(dataset.url)).suffix or '.csv'
            dataset.filename = f"dataset_{timestamp}{file_extension}"
        
        # Ensure filename is safe
        safe_filename = Path(dataset.filename).name
        file_path = DOWNLOAD_DIR / safe_filename
        
        # Download file
        async with aiohttp.ClientSession() as session:
            async with session.get(str(dataset.url)) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download file from URL. Status: {response.status}"
                    )
                
                content = await response.read()
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(content)
        
        # Update the QA chain with the new dataset
        global qa_chain
        qa_chain = setup_qa_chain(force_reload=True)
        
        return {
            "message": "Dataset downloaded and processed successfully with Groq",
            "filename": safe_filename,
            "path": str(file_path),
            "size_bytes": os.path.getsize(file_path)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading dataset for Groq processing: {str(e)}"
        )

@app.get("/groq/list-datasets")
async def list_datasets():
    """List all available datasets"""
    try:
        files = []
        for file_path in DOWNLOAD_DIR.glob('*'):
            files.append({
                "filename": file_path.name,
                "size_bytes": os.path.getsize(file_path),
                "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
            })
        return {"datasets": files}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing datasets: {str(e)}"
        )

@app.get("/groq/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Verify Groq API key is set
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in environment variables")
        return {"status": "healthy", "backend": "groq"}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.post("/groq/download-kaggle-dataset")
async def download_kaggle_dataset(dataset: KaggleDatasetDownload):
    """Download a dataset from Kaggle using the dataset name"""
    try:
        logger.info(f"Starting Kaggle dataset download: {dataset.dataset_name}")
        
        # Check if Kaggle API credentials exist
        kaggle_dir = Path.home() / '.kaggle'
        if not (kaggle_dir / 'kaggle.json').exists():
            logger.error("Kaggle API credentials not found")
            raise HTTPException(
                status_code=400,
                detail="Kaggle API credentials not found. Please configure your Kaggle API token."
            )

        # Create download directory if it doesn't exist
        download_path = DOWNLOAD_DIR / "kaggle"
        download_path.mkdir(exist_ok=True)
        
        try:
            # Download the dataset
            logger.info(f"Downloading dataset to {download_path}")
            kaggle.api.dataset_download_files(
                dataset.dataset_name,
                path=str(download_path),
                unzip=True
            )
            
            # Get the downloaded files
            downloaded_files = list(download_path.glob('*'))
            if not downloaded_files:
                raise Exception("No files were downloaded")

            logger.info(f"Successfully downloaded: {[f.name for f in downloaded_files]}")
            
            # Update the QA chain with new dataset
            global qa_chain
            qa_chain = setup_qa_chain(force_reload=True)
            
            return {
                "message": "Kaggle dataset downloaded successfully",
                "files": [str(f.name) for f in downloaded_files],
                "download_path": str(download_path)
            }
            
        except Exception as e:
            logger.error(f"Error downloading Kaggle dataset: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error downloading Kaggle dataset: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Error in download_kaggle_dataset: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing Kaggle dataset download: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Using port 8001 to avoid conflict with original API 