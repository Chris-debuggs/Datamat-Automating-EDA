from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, HttpUrl
from utils import setup_qa_chain
import os
from pathlib import Path
from datetime import datetime
import aiohttp
import aiofiles
from typing import Optional, List
import shutil
from logger import logging
import kaggle

# Configure logging for this module
logger = logging.getLogger(__name__)

app = FastAPI(title="DATAmat API")

# Configure download directory
DOWNLOAD_DIR = Path("datasets")
DOWNLOAD_DIR.mkdir(exist_ok=True)

class KaggleDatasetDownload(BaseModel):
    dataset_name: str  # Format: "username/dataset-name"
    filename: Optional[str] = None

# Initialize QA chain
qa_chain = setup_qa_chain()

class Query(BaseModel):
    question: str

class DatasetDownload(BaseModel):
    url: HttpUrl
    filename: Optional[str] = None

@app.post("/ask")
async def ask_question(query: Query):
    try:
        result = qa_chain.invoke({"query": query.question})
        return {"answer": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = DOWNLOAD_DIR / safe_filename
        
        # Save uploaded file
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        finally:
            file.file.close()
        
        # Update the QA chain with the new dataset
        global qa_chain
        qa_chain = setup_qa_chain(force_reload=True)
        
        return {
            "message": "Dataset uploaded and processed successfully",
            "filename": safe_filename,
            "path": str(file_path),
            "size_bytes": os.path.getsize(file_path)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading dataset: {str(e)}"
        )

@app.post("/download-dataset")
async def download_dataset(dataset: DatasetDownload):
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
            "message": "Dataset downloaded and processed successfully",
            "filename": safe_filename,
            "path": str(file_path),
            "size_bytes": os.path.getsize(file_path)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading dataset: {str(e)}"
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

@app.get("/list-datasets")
async def list_datasets():
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 